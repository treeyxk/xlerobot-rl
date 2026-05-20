"""Solve head-camera hand-eye calibration from AprilTag samples.

Input samples must contain:
  - right_follower_qpos in LeRobot joint names, degrees
  - T_camera_tag from AprilTag PnP

The solver uses the right TCP frame:
  Fixed_Jaw_2 frame + [0, -0.107, 0] meters

It estimates both:
  - T_base_camera
  - T_tcp_tag

by minimizing:
  T_base_tcp_i @ T_tcp_tag ~= T_base_camera @ T_camera_tag_i
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


URDF_PATH = Path("xlerobot_rl/sim/assets/urdf/xlerobot.urdf")
TCP_OFFSET_IN_FIXED_JAW = np.array([0.0, -0.107, 0.0], dtype=np.float64)

LEROBOT_TO_URDF_RIGHT = {
    "shoulder_pan": "Rotation_R",
    "shoulder_lift": "Pitch_R",
    "elbow_flex": "Elbow_R",
    "wrist_flex": "Wrist_Pitch_R",
    "wrist_roll": "Wrist_Roll_R",
    "gripper": "Jaw_R",
}

# LeRobot calibrated positive direction vs. URDF positive joint axis.
# Found from the AprilTag hand-eye dataset sign search and consistent with the right-arm
# shoulder lift / wrist roll axis conventions in the URDF.
LEROBOT_TO_URDF_SIGN = {
    "shoulder_pan": 1.0,
    "shoulder_lift": -1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": -1.0,
    "gripper": 1.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/real/calibration/handeye_apriltag_merged"),
    )
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=URDF_PATH)
    parser.add_argument(
        "--tcp-offset",
        type=float,
        nargs=3,
        default=TCP_OFFSET_IN_FIXED_JAW.tolist(),
        metavar=("X", "Y", "Z"),
        help="TCP offset in Fixed_Jaw_2 frame, meters.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="Weight for translation residuals in meters.",
    )
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=0.05,
        help="Weight for rotation residuals in radians. Lower values favor translation fit.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    return parser


def transform_inverse(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv


def transform_from_rotvec_t(rotvec: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def transform_to_rotvec_t(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3].copy()


def pose_to_matrix(pose: sapien.Pose) -> np.ndarray:
    return pose.to_transformation_matrix().astype(np.float64)


class SapienFk:
    def __init__(self, urdf_path: Path, tcp_offset: np.ndarray):
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 100.0)
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(urdf_path))
        self.robot.set_pose(sapien.Pose([0, 0, 0]))

        self.active_joints = [j.name for j in self.robot.get_active_joints()]
        self.qpos_index = {name: idx for idx, name in enumerate(self.active_joints)}
        self.fixed_jaw = {l.name: l for l in self.robot.get_links()}["Fixed_Jaw_2"]
        self.tcp_offset = tcp_offset.astype(np.float64)

    def qpos_from_lerobot(self, qpos_deg: dict[str, float]) -> np.ndarray:
        qpos = np.zeros(len(self.active_joints), dtype=np.float64)
        for lerobot_name, urdf_name in LEROBOT_TO_URDF_RIGHT.items():
            if lerobot_name not in qpos_deg:
                continue
            if urdf_name not in self.qpos_index:
                raise KeyError(f"URDF active joint not found: {urdf_name}")
            sign = LEROBOT_TO_URDF_SIGN[lerobot_name]
            qpos[self.qpos_index[urdf_name]] = np.deg2rad(sign * float(qpos_deg[lerobot_name]))
        return qpos

    def base_tcp(self, qpos_deg: dict[str, float]) -> np.ndarray:
        self.robot.set_qpos(self.qpos_from_lerobot(qpos_deg))
        self.scene.update_render()
        T_base_fixed = pose_to_matrix(self.fixed_jaw.get_entity_pose())
        T_fixed_tcp = np.eye(4, dtype=np.float64)
        T_fixed_tcp[:3, 3] = self.tcp_offset
        return T_base_fixed @ T_fixed_tcp


def load_samples(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def initial_guess(base_tcp: list[np.ndarray], camera_tag: list[np.ndarray]) -> np.ndarray:
    # Start with a plausible camera pose from pairwise translations:
    # T_base_camera roughly maps camera tag positions into the arm workspace.
    tcp_positions = np.array([T[:3, 3] for T in base_tcp])
    tag_positions_cam = np.array([T[:3, 3] for T in camera_tag])
    x_t = tcp_positions.mean(axis=0) - tag_positions_cam.mean(axis=0)
    y_t = np.zeros(3, dtype=np.float64)
    x_r = np.zeros(3, dtype=np.float64)
    y_r = np.zeros(3, dtype=np.float64)
    return np.r_[x_r, x_t, y_r, y_t]


def residuals(
    params: np.ndarray,
    base_tcp: list[np.ndarray],
    camera_tag: list[np.ndarray],
    position_weight: float,
    rotation_weight: float,
) -> np.ndarray:
    T_base_camera = transform_from_rotvec_t(params[0:3], params[3:6])
    T_tcp_tag = transform_from_rotvec_t(params[6:9], params[9:12])

    res = []
    for T_base_tcp_i, T_camera_tag_i in zip(base_tcp, camera_tag):
        lhs = T_base_tcp_i @ T_tcp_tag
        rhs = T_base_camera @ T_camera_tag_i
        err = transform_inverse(lhs) @ rhs
        res.extend((position_weight * err[:3, 3]).tolist())
        rot_err = Rotation.from_matrix(err[:3, :3]).as_rotvec()
        res.extend((rotation_weight * rot_err).tolist())
    return np.asarray(res, dtype=np.float64)


def solve(
    base_tcp: list[np.ndarray],
    camera_tag: list[np.ndarray],
    position_weight: float,
    rotation_weight: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    guesses = [initial_guess(base_tcp, camera_tag)]
    # Add a few robust orientation seeds because hand-eye can have local minima.
    for yaw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        g = guesses[0].copy()
        g[0:3] = Rotation.from_euler("z", yaw).as_rotvec()
        guesses.append(g)

    best = None
    for guess in guesses:
        result = least_squares(
            residuals,
            guess,
            args=(base_tcp, camera_tag, position_weight, rotation_weight),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=5000,
        )
        if best is None or result.cost < best.cost:
            best = result

    assert best is not None
    params = best.x
    return (
        transform_from_rotvec_t(params[0:3], params[3:6]),
        transform_from_rotvec_t(params[6:9], params[9:12]),
        best,
    )


def compute_per_sample_errors(
    T_base_camera: np.ndarray,
    T_tcp_tag: np.ndarray,
    base_tcp: list[np.ndarray],
    camera_tag: list[np.ndarray],
) -> list[dict[str, float]]:
    errors = []
    for T_base_tcp_i, T_camera_tag_i in zip(base_tcp, camera_tag):
        lhs = T_base_tcp_i @ T_tcp_tag
        rhs = T_base_camera @ T_camera_tag_i
        err = transform_inverse(lhs) @ rhs
        trans = float(np.linalg.norm(err[:3, 3]))
        rot = float(np.linalg.norm(Rotation.from_matrix(err[:3, :3]).as_rotvec()))
        errors.append({"translation_m": trans, "rotation_rad": rot, "rotation_deg": float(np.rad2deg(rot))})
    return errors


def summarize_errors(errors: list[dict[str, float]]) -> dict[str, float]:
    t = np.array([e["translation_m"] for e in errors])
    r = np.array([e["rotation_deg"] for e in errors])
    return {
        "translation_mean_m": float(t.mean()),
        "translation_median_m": float(np.median(t)),
        "translation_p95_m": float(np.percentile(t, 95)),
        "translation_max_m": float(t.max()),
        "rotation_mean_deg": float(r.mean()),
        "rotation_median_deg": float(np.median(r)),
        "rotation_p95_deg": float(np.percentile(r, 95)),
        "rotation_max_deg": float(r.max()),
    }


def matrix_list(T: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in T]


def main() -> int:
    args = build_parser().parse_args()
    samples_path = args.samples or (args.dataset_dir / "samples.jsonl")
    output_path = args.output or (args.dataset_dir / "solve_result.yaml")
    rows = load_samples(samples_path, args.max_samples)
    if len(rows) < 6:
        print(f"ERROR: need at least 6 samples, got {len(rows)}")
        return 1

    fk = SapienFk(args.urdf, np.asarray(args.tcp_offset, dtype=np.float64))
    base_tcp = [fk.base_tcp(r["right_follower_qpos"]) for r in rows]
    camera_tag = [np.asarray(r["T_camera_tag"], dtype=np.float64) for r in rows]

    T_base_camera, T_tcp_tag, result = solve(
        base_tcp,
        camera_tag,
        position_weight=args.position_weight,
        rotation_weight=args.rotation_weight,
    )
    errors = compute_per_sample_errors(T_base_camera, T_tcp_tag, base_tcp, camera_tag)
    summary = summarize_errors(errors)

    output = {
        "schema_version": "handeye_solve_v0",
        "dataset_dir": str(args.dataset_dir),
        "samples": len(rows),
        "urdf": str(args.urdf),
        "ee_frame": {
            "name": "right_tcp",
            "base_link_for_offset": "Fixed_Jaw_2",
            "tcp_offset_in_fixed_jaw_m": [float(x) for x in args.tcp_offset],
        },
        "joint_mapping": LEROBOT_TO_URDF_RIGHT,
        "joint_signs": LEROBOT_TO_URDF_SIGN,
        "solver": {
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "position_weight": args.position_weight,
            "rotation_weight": args.rotation_weight,
        },
        "T_base_camera": matrix_list(T_base_camera),
        "T_camera_base": matrix_list(transform_inverse(T_base_camera)),
        "T_tcp_tag": matrix_list(T_tcp_tag),
        "T_tag_tcp": matrix_list(transform_inverse(T_tcp_tag)),
        "error_summary": summary,
        "sample_errors": errors,
    }
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))

    print(f"Solved {len(rows)} samples")
    print(f"  success: {result.success}")
    print(f"  output:  {output_path}")
    print("  T_base_camera translation:", np.round(T_base_camera[:3, 3], 4).tolist())
    print("  T_tcp_tag translation:    ", np.round(T_tcp_tag[:3, 3], 4).tolist())
    print(
        "  translation error mean/median/p95/max:",
        f"{summary['translation_mean_m']*100:.1f} /",
        f"{summary['translation_median_m']*100:.1f} /",
        f"{summary['translation_p95_m']*100:.1f} /",
        f"{summary['translation_max_m']*100:.1f} cm",
    )
    print(
        "  rotation error mean/median/p95/max:",
        f"{summary['rotation_mean_deg']:.1f} /",
        f"{summary['rotation_median_deg']:.1f} /",
        f"{summary['rotation_p95_deg']:.1f} /",
        f"{summary['rotation_max_deg']:.1f} deg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
