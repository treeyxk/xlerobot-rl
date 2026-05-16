"""M1 Semantic Executive schemas.

These dataclasses are the frozen M1 v0 contract for color-block demos. They are
kept small on purpose: later tracker/search/navigation extensions should add
fields through an ADR rather than changing these names casually.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import time

import numpy as np


NavigationMode = Literal[
    "SEARCH",
    "SEMANTIC_NAV",
    "SKILL_AWARE_APPROACH",
    "READY_TO_GRASP",
    "GRASP",
    "DONE",
    "SAFE_STOP",
]

ExecutionStatus = Literal[
    "idle",
    "tracking",
    "ready",
    "executing",
    "success",
    "failure",
]

FailureReason = Literal[
    "none",
    "target_not_found",
    "low_confidence",
    "ambiguous_target",
    "wrong_object",
    "target_lost",
    "unsafe",
    "timeout",
]


@dataclass
class TargetState:
    """Semantic state for one grounded object instance."""

    object_id: int
    name: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    pos_camera: np.ndarray
    pos_base: np.ndarray
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    is_target: bool = False
    detection_method: str = "unknown"
    last_seen_timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        assert self.mask.ndim == 2, f"mask must be 2D, got {self.mask.shape}"
        assert self.pos_camera.shape == (3,), f"pos_camera must be (3,), got {self.pos_camera.shape}"
        assert self.pos_base.shape == (3,), f"pos_base must be (3,), got {self.pos_base.shape}"
        assert len(self.bbox) == 4, f"bbox must have 4 ints, got {self.bbox}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence out of range: {self.confidence}"

    def to_json_dict(self, include_mask: bool = False) -> dict[str, Any]:
        data = {
            "object_id": self.object_id,
            "name": self.name,
            "attributes": self.attributes,
            "bbox": list(self.bbox),
            "pos_camera": self.pos_camera.round(4).tolist(),
            "pos_base": self.pos_base.round(4).tolist(),
            "confidence": round(float(self.confidence), 4),
            "is_target": self.is_target,
            "detection_method": self.detection_method,
            "last_seen_timestamp": float(self.last_seen_timestamp),
        }
        if include_mask:
            data["mask"] = self.mask.astype(bool).tolist()
        return data


@dataclass
class SemanticExecutiveState:
    """Frozen M1 v0 output consumed by M5/M2/M3/M4."""

    instruction: str
    task_graph: list[str]
    current_subgoal: str
    target: TargetState | None
    scene_objects: list[TargetState]
    candidate_skills: list[str]
    selected_skill: str | None
    navigation_mode: NavigationMode
    execution_status: ExecutionStatus = "ready"
    failure_reason: FailureReason = "none"
    search_goal: np.ndarray | None = None
    local_nav_goal: np.ndarray | None = None
    success_scores: dict[str, float] = field(default_factory=dict)
    uncertainty_scores: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.target is not None:
            assert self.target.is_target, "target must have is_target=True"
            target_ids = [obj.object_id for obj in self.scene_objects if obj.is_target]
            assert self.target.object_id in target_ids, "target must be included in scene_objects"
        assert self.selected_skill is None or self.selected_skill in self.candidate_skills

    @property
    def distractors(self) -> list[TargetState]:
        if self.target is None:
            return self.scene_objects
        return [obj for obj in self.scene_objects if obj.object_id != self.target.object_id]

    def to_json_dict(self, include_masks: bool = False) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "task_graph": self.task_graph,
            "current_subgoal": self.current_subgoal,
            "target": self.target.to_json_dict(include_mask=include_masks) if self.target else None,
            "scene_objects": [
                obj.to_json_dict(include_mask=include_masks) for obj in self.scene_objects
            ],
            "distractors": [
                obj.to_json_dict(include_mask=include_masks) for obj in self.distractors
            ],
            "candidate_skills": self.candidate_skills,
            "selected_skill": self.selected_skill,
            "navigation_mode": self.navigation_mode,
            "execution_status": self.execution_status,
            "failure_reason": self.failure_reason,
            "search_goal": self.search_goal.round(4).tolist() if self.search_goal is not None else None,
            "local_nav_goal": (
                self.local_nav_goal.round(4).tolist() if self.local_nav_goal is not None else None
            ),
            "success_scores": self.success_scores,
            "uncertainty_scores": self.uncertainty_scores,
            "timestamp": float(self.timestamp),
        }
