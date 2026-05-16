"""M1 v0 color-block Semantic Executive demo.

Scope from the v3.0 project plan:
- rule parser for color-block pick instructions
- HSV/SAM2 grounding
- RGB-D 3D back-projection
- structured JSON output with target/distractors, selected skill, and nav mode

This script does not implement tracking, navigation, or policy execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

import xlerobot_rl.sim.envs  # noqa: F401
from xlerobot_rl.modules.semantic_executive import ColorBlockSemanticExecutive
from xlerobot_rl.perception import GroundedObject, GroundingPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_DIR = REPO_ROOT / "data/debug"


def expected_color_from_instruction(instruction: str) -> str | None:
    text = instruction.lower()
    if "红" in text or "red" in text or "hong" in text:
        return "red"
    if "蓝" in text or "blue" in text or "lan" in text:
        return "blue"
    if "绿" in text or "green" in text or "lv" in text:
        return "green"
    return None


def extract_head_camera(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = obs["sensor_data"]["head_camera"]["rgb"][0].cpu().numpy()
    depth_mm = obs["sensor_data"]["head_camera"]["depth"][0].cpu().numpy().squeeze()
    depth_meters = depth_mm.astype(np.float32) / 1000.0

    sensor_param = obs["sensor_param"]["head_camera"]
    K = sensor_param["intrinsic_cv"][0].cpu().numpy()
    extrinsic_cv = sensor_param["extrinsic_cv"][0].cpu().numpy()
    if extrinsic_cv.shape == (3, 4):
        T_cam_world = np.eye(4, dtype=np.float32)
        T_cam_world[:3, :] = extrinsic_cv
    else:
        T_cam_world = extrinsic_cv

    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    return rgb, depth_meters, K, T_world_cam


def save_overlay(rgb: np.ndarray, objects: list[GroundedObject], out_path: Path) -> None:
    overlay = rgb.copy()
    colors = {
        "red": np.array([255, 0, 0], dtype=np.uint8),
        "blue": np.array([0, 80, 255], dtype=np.uint8),
        "green": np.array([0, 220, 0], dtype=np.uint8),
    }
    for obj in objects:
        color_name = obj.attributes.get("color", "red")
        color = colors.get(color_name, np.array([255, 255, 255], dtype=np.uint8))
        mask_layer = np.zeros_like(rgb)
        mask_layer[obj.mask] = color
        overlay = cv2.addWeighted(overlay, 0.75, mask_layer, 0.25, 0)

        x1, y1, x2, y2 = obj.bbox
        line_color = tuple(int(v) for v in color.tolist())
        label = "target" if obj.is_target else "distractor"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), line_color, 2)
        cv2.putText(
            overlay,
            f"{obj.name}:{label}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            line_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def check_m1_v0(semantic_json: dict, json_path: Path, overlay_path: Path) -> list[str]:
    failures = []
    expected_color = expected_color_from_instruction(semantic_json["instruction"])
    target = semantic_json["target"]

    if semantic_json["execution_status"] != "ready":
        failures.append(f"execution_status={semantic_json['execution_status']!r}, expected 'ready'")
    if semantic_json["failure_reason"] != "none":
        failures.append(f"failure_reason={semantic_json['failure_reason']!r}, expected 'none'")
    if semantic_json["navigation_mode"] != "READY_TO_GRASP":
        failures.append(f"navigation_mode={semantic_json['navigation_mode']!r}, expected 'READY_TO_GRASP'")
    if semantic_json["selected_skill"] != "top_grasp":
        failures.append(f"selected_skill={semantic_json['selected_skill']!r}, expected 'top_grasp'")
    if target is None:
        failures.append("target is None")
    elif expected_color is not None and target["attributes"].get("color") != expected_color:
        failures.append(
            f"target color={target['attributes'].get('color')!r}, expected {expected_color!r}"
        )
    if not semantic_json["scene_objects"]:
        failures.append("scene_objects is empty")
    if not json_path.exists():
        failures.append(f"missing JSON output: {json_path}")
    if not overlay_path.exists():
        failures.append(f"missing overlay output: {overlay_path}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="抓红色色块")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--check", action="store_true", help="Fail if M1 v0 ready-state checks fail.")
    args = parser.parse_args()

    env = gym.make(
        "StaticArmGrasp-v0",
        num_envs=1,
        obs_mode="rgbd",
        sim_backend=args.sim_backend,
        include_distractors=True,
    ).unwrapped
    obs, _ = env.reset(seed=args.seed)

    rgb, depth_meters, K, T_world_cam = extract_head_camera(obs)
    pipeline = GroundingPipeline(use_sam2=args.use_sam2)
    executive = ColorBlockSemanticExecutive(grounding_pipeline=pipeline)
    result = executive.run(
        instruction=args.instruction,
        rgb=rgb,
        depth_meters=depth_meters,
        K=K,
        T_world_camera=T_world_cam,
    )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEBUG_DIR / "m1_v0_semantic_state.json"
    overlay_path = DEBUG_DIR / "m1_v0_grounding_overlay.png"
    semantic_json = result.state.to_json_dict()
    json_path.write_text(json.dumps(semantic_json, indent=2, ensure_ascii=False) + "\n")
    save_overlay(rgb, result.grounded_objects, overlay_path)

    print(json.dumps(semantic_json, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved overlay: {overlay_path}")

    if args.check:
        failures = check_m1_v0(semantic_json, json_path, overlay_path)
        if failures:
            print("\nM1 v0 check failed:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nM1 v0 check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
