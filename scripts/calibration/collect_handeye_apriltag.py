"""Collect AprilTag hand-eye calibration samples.

Each accepted sample stores:
  - raw head camera RGB image
  - annotated preview image
  - AprilTag corners and T_camera_tag
  - right follower joint positions
  - timestamp and camera/calibration metadata

Controls:
  1 / !   : shoulder_pan + / -
  2 / @   : shoulder_lift + / -
  3 / #   : elbow_flex + / -
  4 / $   : wrist_flex + / -
  5 / %   : wrist_roll + / -
  6 / ^   : gripper + / -
  r       : sync jog targets from current joint positions
  SPACE / c : save current sample when the target tag is visible
  q / ESC   : quit
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


APRILTAG_DICTS = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

DEFAULT_INTRINSICS = Path("configs/calibration/head_camera_intrinsics_1280x720.yaml")
JOG_KEYMAP = {
    ord("1"): ("shoulder_pan", +1.0),
    ord("!"): ("shoulder_pan", -1.0),
    ord("2"): ("shoulder_lift", +1.0),
    ord("@"): ("shoulder_lift", -1.0),
    ord("3"): ("elbow_flex", +1.0),
    ord("#"): ("elbow_flex", -1.0),
    ord("4"): ("wrist_flex", +1.0),
    ord("$"): ("wrist_flex", -1.0),
    ord("5"): ("wrist_roll", +1.0),
    ord("%"): ("wrist_roll", -1.0),
    ord("6"): ("gripper", +1.0),
    ord("^"): ("gripper", -1.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", default="/dev/xlerobot_right_follower")
    parser.add_argument("--follower-id", default="right_follower")
    parser.add_argument("--camera-index", default="/dev/xlerobot_head_camera")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--tag-family", choices=sorted(APRILTAG_DICTS), default="tag36h11")
    parser.add_argument("--tag-id", type=int, default=10)
    parser.add_argument(
        "--tag-size-m",
        type=float,
        required=True,
        help="Physical AprilTag black-square side length in meters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real/calibration/handeye_apriltag"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument("--jog-step", type=float, default=2.0, help="Joint jog step in degrees.")
    parser.add_argument(
        "--gripper-jog-step",
        type=float,
        default=5.0,
        help="Gripper jog step in LeRobot calibrated position units.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="LeRobot safety clip for each commanded jog step.",
    )
    parser.add_argument(
        "--jog-settle-s",
        type=float,
        default=0.15,
        help="Delay after each jog before reading back present joint positions.",
    )
    parser.add_argument(
        "--p-coefficient",
        type=int,
        default=None,
        help="Optional motor P_Coefficient override after connect. Try 24 or 32 if jogs are too weak.",
    )
    parser.add_argument(
        "--fast-jog",
        action="store_true",
        help="Use faster jog defaults: 5 deg step, 30 max-relative-target, 0.05s settle.",
    )
    parser.add_argument(
        "--connect-calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow LeRobot to run calibration on connect. Disabled by default.",
    )
    return parser


def git_commit() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open() as f:
        cfg = yaml.safe_load(f)

    k_data = cfg["intrinsic_matrix"]["data"]
    camera_matrix = np.asarray(k_data, dtype=np.float64)
    dist_coeffs = np.asarray(cfg["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs, cfg


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    source: int | str = int(args.camera_index) if str(args.camera_index).isdigit() else args.camera_index
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def make_detector(tag_family: str) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[tag_family])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


def connect_follower(args: argparse.Namespace) -> SO101Follower:
    cfg = SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        cameras={},
        max_relative_target=args.max_relative_target,
    )
    robot = SO101Follower(cfg)
    robot.connect(calibrate=args.connect_calibrate)
    if args.p_coefficient is not None:
        for motor in robot.bus.motors:
            robot.bus.write("P_Coefficient", motor, args.p_coefficient)
    return robot


def detect_target(
    detector: cv2.aruco.ArucoDetector,
    frame_bgr: np.ndarray,
    tag_id: int,
) -> tuple[np.ndarray | None, list[int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, []

    ids_flat = [int(i) for i in ids.reshape(-1)]
    for idx, detected_id in enumerate(ids_flat):
        if detected_id == tag_id:
            return corners[idx].reshape(4, 2).astype(np.float64), ids_flat
    return None, ids_flat


def tag_object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    # Order matches OpenCV ArUco/AprilTag detected corners:
    # top-left, top-right, bottom-right, bottom-left in the tag image.
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_tag_pose(
    corners: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_points = tag_object_points(tag_size_m)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed for detected AprilTag")

    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return rvec.reshape(3), tvec.reshape(3), transform


def read_joint_positions(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    qpos = {key.removesuffix(".pos"): float(value) for key, value in obs.items() if key.endswith(".pos")}
    if not qpos:
        raise RuntimeError("no joint positions returned by right follower")
    return dict(sorted(qpos.items()))


def jog_joint(
    robot: SO101Follower,
    jog_targets: dict[str, float],
    joint: str,
    direction: float,
    args: argparse.Namespace,
) -> None:
    if joint not in jog_targets:
        raise RuntimeError(f"joint {joint!r} not found in follower observation: {sorted(jog_targets)}")

    present_before = read_joint_positions(robot)
    step = args.gripper_jog_step if joint == "gripper" else args.jog_step
    target = jog_targets[joint] + direction * step
    jog_targets[joint] = target

    # Send the full current target vector. This matches normal position-control usage better than
    # partial single-joint writes and keeps the non-jogged joints held at their last targets.
    action = {f"{name}.pos": value for name, value in jog_targets.items()}
    sent = robot.send_action(action)
    for name in jog_targets:
        sent_key = f"{name}.pos"
        if sent_key in sent:
            jog_targets[name] = float(sent[sent_key])

    if args.jog_settle_s > 0:
        time.sleep(args.jog_settle_s)
    present_after = read_joint_positions(robot)

    before = present_before.get(joint, float("nan"))
    after = present_after.get(joint, float("nan"))
    sent_target = jog_targets[joint]
    moved = after - before
    print(
        f"Jog {joint:13s} {direction * step:+.2f} | "
        f"present {before:.2f} -> {after:.2f} (delta {moved:+.2f}) | "
        f"sent target {sent_target:.2f}"
    )
    if abs(moved) < 0.05 and abs(sent_target - before) > 0.5:
        print(
            f"WARN: {joint} did not visibly move after the jog. "
            "Try the opposite direction, press r to resync, or reduce --jog-step."
        )


def sync_jog_targets(robot: SO101Follower, jog_targets: dict[str, float]) -> None:
    jog_targets.clear()
    jog_targets.update(read_joint_positions(robot))


def handle_jog_key(
    key: int,
    robot: SO101Follower,
    jog_targets: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if key not in JOG_KEYMAP:
        return False
    joint, direction = JOG_KEYMAP[key]
    jog_joint(robot, jog_targets, joint, direction, args)
    return True


def draw_preview(
    frame_bgr: np.ndarray,
    corners: np.ndarray | None,
    detected_ids: list[int],
    tag_id: int,
    sample_count: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    visible = corners is not None
    color = (0, 255, 0) if visible else (0, 0, 255)
    status = f"target id={tag_id} visible={visible} detected={detected_ids} saved={sample_count}"
    cv2.putText(vis, status, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        "Jog: 1/! pan 2/@ lift 3/# elbow 4/$ wrist_flex 5/% wrist_roll 6/^ grip | r sync | SPACE save | q quit",
        (16, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if corners is not None:
        pts = corners.reshape(4, 2).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for idx, pt in enumerate(pts):
            cv2.circle(vis, tuple(pt), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(idx),
                tuple(pt + np.array([6, -6])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def write_session_metadata(args: argparse.Namespace, intrinsics_cfg: dict[str, Any]) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "session.yaml"
    content = {
        "schema_version": "handeye_apriltag_v0",
        "created_at": now_utc(),
        "git_commit": git_commit(),
        "robot": {
            "follower_id": args.follower_id,
            "follower_port": args.follower_port,
            "joint_position_units": "LeRobot calibrated units; use_degrees=True config",
            "jog_step": args.jog_step,
            "gripper_jog_step": args.gripper_jog_step,
            "max_relative_target": args.max_relative_target,
            "jog_settle_s": args.jog_settle_s,
            "p_coefficient": args.p_coefficient,
            "fast_jog": args.fast_jog,
        },
        "camera": {
            "index_or_path": args.camera_index,
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "intrinsics_file": str(args.intrinsics),
            "intrinsic_matrix": intrinsics_cfg["intrinsic_matrix"]["data"],
            "distortion_coefficients": intrinsics_cfg["distortion_coefficients"],
            "distortion_model": intrinsics_cfg.get("distortion_model"),
        },
        "tag": {
            "family": args.tag_family,
            "id": args.tag_id,
            "size_m": args.tag_size_m,
            "object_point_order": "top_left, top_right, bottom_right, bottom_left",
        },
        "files": {
            "samples_jsonl": "samples.jsonl",
            "raw_images": "raw/",
            "annotated_images": "annotated/",
        },
        "notes": [
            "Move the arm to a stable pose before saving each sample.",
            "Do not move or touch the tag relative to the wrist after collection starts.",
            "These samples are for solving T_base_head_camera and T_ee_tag offline.",
        ],
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
    return path


def append_sample(path: Path, sample: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_sample(
    args: argparse.Namespace,
    index: int,
    frame_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    corners: np.ndarray,
    qpos: dict[str, float],
    detected_ids: list[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any]:
    rvec, tvec, transform = estimate_tag_pose(
        corners,
        args.tag_size_m,
        camera_matrix,
        dist_coeffs,
    )

    raw_dir = args.output_dir / "raw"
    annotated_dir = args.output_dir / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = f"sample_{index:04d}_{timestamp}"
    raw_path = raw_dir / f"{stem}_raw.png"
    annotated_path = annotated_dir / f"{stem}_annotated.png"

    if not cv2.imwrite(str(raw_path), frame_bgr):
        raise RuntimeError(f"failed to save raw image: {raw_path}")
    if not cv2.imwrite(str(annotated_path), annotated_bgr):
        raise RuntimeError(f"failed to save annotated image: {annotated_path}")

    return {
        "schema_version": "handeye_apriltag_sample_v0",
        "sample_index": index,
        "timestamp_utc": now_utc(),
        "raw_image": str(raw_path.relative_to(args.output_dir)),
        "annotated_image": str(annotated_path.relative_to(args.output_dir)),
        "detected_ids": detected_ids,
        "tag_id": args.tag_id,
        "tag_family": args.tag_family,
        "tag_size_m": args.tag_size_m,
        "corners_px": corners.tolist(),
        "rvec_camera_tag": rvec.tolist(),
        "tvec_camera_tag_m": tvec.tolist(),
        "T_camera_tag": transform.tolist(),
        "right_follower_qpos": qpos,
    }


def print_startup(args: argparse.Namespace, session_path: Path) -> None:
    print("Hand-eye AprilTag collection")
    print(f"  output_dir:    {args.output_dir}")
    print(f"  session:       {session_path}")
    print(f"  follower:      {args.follower_id} @ {args.follower_port}")
    print(f"  camera:        {args.camera_index} {args.camera_width}x{args.camera_height}@{args.camera_fps:g}")
    print(f"  tag:           {args.tag_family} id={args.tag_id} size={args.tag_size_m:g}m")
    print("\nControls:")
    print("  1 / !  shoulder_pan   + / -")
    print("  2 / @  shoulder_lift  + / -")
    print("  3 / #  elbow_flex     + / -")
    print("  4 / $  wrist_flex     + / -")
    print("  5 / %  wrist_roll     + / -")
    print("  6 / ^  gripper        + / -")
    print("  r      sync jog targets from current joint positions")
    print("  SPACE/c save sample, q/ESC quit")
    print("Before each save: stop moving the arm, make sure the tag is fully visible, then press SPACE.")


def main() -> int:
    args = build_parser().parse_args()
    if args.fast_jog:
        args.jog_step = 5.0
        args.gripper_jog_step = 8.0
        args.max_relative_target = 30.0
        args.jog_settle_s = 0.05
    if not math.isfinite(args.tag_size_m) or args.tag_size_m <= 0:
        print("ERROR: --tag-size-m must be a positive number in meters")
        return 1

    camera_matrix, dist_coeffs, intrinsics_cfg = load_intrinsics(args.intrinsics)
    session_path = write_session_metadata(args, intrinsics_cfg)
    samples_path = args.output_dir / "samples.jsonl"
    detector = make_detector(args.tag_family)

    cap = open_camera(args)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera {args.camera_index}")
        return 1

    robot: SO101Follower | None = None
    try:
        robot = connect_follower(args)
        print_startup(args, session_path)

        for _ in range(max(args.warmup_frames, 0)):
            cap.read()

        index = args.start_index
        window = "collect_handeye_apriltag"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        jog_targets = read_joint_positions(robot)
        print("Initial jog targets:")
        for joint, value in jog_targets.items():
            print(f"  {joint:13s}: {value:.2f}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: failed to read frame")
                continue

            corners, detected_ids = detect_target(detector, frame, args.tag_id)
            annotated = draw_preview(frame, corners, detected_ids, args.tag_id, index - args.start_index)
            display = annotated
            if args.preview_scale != 1.0:
                display = cv2.resize(
                    annotated,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                sync_jog_targets(robot, jog_targets)
                print("Synced jog targets from current follower positions.")
                continue
            if handle_jog_key(key, robot, jog_targets, args):
                continue
            if key not in (ord(" "), ord("c")):
                continue

            if corners is None:
                print(f"SKIP: target id={args.tag_id} not visible; detected={detected_ids}")
                continue

            # Give the bus read a clear timestamp after the operator has stopped moving.
            qpos = read_joint_positions(robot)
            sample = save_sample(
                args=args,
                index=index,
                frame_bgr=frame,
                annotated_bgr=annotated,
                corners=corners,
                qpos=qpos,
                detected_ids=detected_ids,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
            append_sample(samples_path, sample)
            t = sample["tvec_camera_tag_m"]
            print(
                f"Saved sample {index:04d}: tag id={args.tag_id}, "
                f"t_camera_tag=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m"
            )
            index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot is not None and robot.is_connected:
            robot.disconnect()

    print(f"Done. Samples metadata: {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
