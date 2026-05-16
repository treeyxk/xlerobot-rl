"""M1 v0 Semantic Executive for color-block tabletop tasks.

This is the Month 1 / W2 module from the v3.0 project plan:
- rule parser for red/blue/green cube pick instructions
- HSV/SAM2 semantic grounding through the perception module
- target/distractor assignment
- frozen `SemanticExecutiveState` output

It intentionally does not implement tracking, navigation, grasp execution, or
safety approval. Those belong to M1.3/M2/M4/M5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from xlerobot_rl.interfaces import SemanticExecutiveState, TargetState
from xlerobot_rl.interfaces.semantic import FailureReason
from xlerobot_rl.perception import GroundedObject, GroundingPipeline


DEFAULT_TASK_GRAPH = [
    "find_target",
    "navigate_to_skill_success_region",
    "grasp_target",
    "verify_success",
]
DEFAULT_MIN_TARGET_CONFIDENCE = 0.25

COLOR_ALIASES = {
    "red": ("red", "hong", "红"),
    "blue": ("blue", "lan", "蓝"),
    "green": ("green", "lv", "绿"),
}


@dataclass
class ColorBlockM1Result:
    """Full M1 v0 result plus raw detections for debug visualization."""

    task: dict[str, Any]
    state: SemanticExecutiveState
    grounded_objects: list[GroundedObject]


def parse_color_block_instruction(instruction: str) -> dict[str, Any]:
    """Parse the M1 v0 supported color-block pick commands."""
    text = instruction.lower()
    target_color = None
    for color, aliases in COLOR_ALIASES.items():
        if any(alias in text for alias in aliases):
            target_color = color
            break

    if target_color is None:
        raise ValueError(f"Cannot parse target color from instruction: {instruction}")

    return {
        "task_type": "pick",
        "target": {"color": target_color, "category": "cube"},
        "constraints": {"must_grasp_target_only": True},
    }


def target_state_from_grounded_object(obj: GroundedObject) -> TargetState:
    """Convert perception output into the frozen M1 schema."""
    return TargetState(
        object_id=obj.object_id,
        name=obj.name,
        bbox=obj.bbox,
        mask=obj.mask,
        pos_camera=obj.pos_camera,
        pos_base=obj.pos_world,
        confidence=obj.confidence,
        attributes=obj.attributes,
        is_target=obj.is_target,
        detection_method=obj.detection_method,
    )


class ColorBlockSemanticExecutive:
    """Semantic Executive v0 for target-visible red/blue/green cube tasks."""

    def __init__(
        self,
        grounding_pipeline: GroundingPipeline | None = None,
        use_sam2: bool = False,
        min_target_confidence: float = DEFAULT_MIN_TARGET_CONFIDENCE,
    ):
        self.grounding_pipeline = grounding_pipeline or GroundingPipeline(use_sam2=use_sam2)
        self.min_target_confidence = min_target_confidence

    def run(
        self,
        instruction: str,
        rgb: np.ndarray,
        depth_meters: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
    ) -> ColorBlockM1Result:
        try:
            task = parse_color_block_instruction(instruction)
        except ValueError:
            state = self.build_failure_state(
                instruction=instruction,
                failure_reason="ambiguous_target",
            )
            return ColorBlockM1Result(task={}, state=state, grounded_objects=[])

        objects = self.grounding_pipeline.detect_colored_cubes(
            rgb=rgb,
            depth_meters=depth_meters,
            K=K,
            T_world_camera=T_world_camera,
            target_color=task["target"]["color"],
        )
        state = self.build_state(instruction=instruction, task=task, objects=objects)
        return ColorBlockM1Result(task=task, state=state, grounded_objects=objects)

    def build_state(
        self,
        instruction: str,
        task: dict[str, Any],
        objects: list[GroundedObject],
    ) -> SemanticExecutiveState:
        target_color = task["target"]["color"]
        target_states = [target_state_from_grounded_object(obj) for obj in objects]
        failure_reason = self.evaluate_grounding_result(
            task=task,
            objects=target_states,
        )
        if failure_reason != "none":
            return self.build_failure_state(
                instruction=instruction,
                failure_reason=failure_reason,
                scene_objects=target_states,
            )

        target = next(obj for obj in target_states if obj.attributes.get("color") == target_color)

        return SemanticExecutiveState(
            instruction=instruction,
            task_graph=DEFAULT_TASK_GRAPH.copy(),
            current_subgoal="find_target",
            target=target,
            scene_objects=target_states,
            candidate_skills=["top_grasp"],
            selected_skill="top_grasp",
            navigation_mode="READY_TO_GRASP",
            execution_status="ready",
            failure_reason="none",
        )

    def evaluate_grounding_result(
        self,
        task: dict[str, Any],
        objects: list[TargetState],
    ) -> FailureReason:
        """Minimal M1 v0 grounding gate.

        This only checks structural conditions needed before handoff proposals:
        target exists, exactly one target candidate exists, and confidence is high
        enough. It deliberately does not do VLM verification or tracking.
        """
        target_color = task["target"]["color"]
        target_candidates = [
            obj for obj in objects if obj.attributes.get("color") == target_color
        ]
        if not target_candidates:
            return "target_not_found"
        if len(target_candidates) > 1:
            return "ambiguous_target"
        if target_candidates[0].confidence < self.min_target_confidence:
            return "low_confidence"
        return "none"

    def build_failure_state(
        self,
        instruction: str,
        failure_reason: FailureReason,
        scene_objects: list[TargetState] | None = None,
    ) -> SemanticExecutiveState:
        return SemanticExecutiveState(
            instruction=instruction,
            task_graph=DEFAULT_TASK_GRAPH.copy(),
            current_subgoal="find_target",
            target=None,
            scene_objects=scene_objects or [],
            candidate_skills=["top_grasp"],
            selected_skill=None,
            navigation_mode="SAFE_STOP",
            execution_status="failure",
            failure_reason=failure_reason,
        )
