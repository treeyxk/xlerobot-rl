"""BC demo recording helper for right-arm leader-follower collection.

Default behavior is dry-run: write a dataset_info.yaml template and print the
LeRobot commands needed to calibrate, teleoperate, and record. Use --run-record
only after calibration and a short teleop smoke test pass.
"""
from __future__ import annotations

import argparse
import json
import select
import shutil
import sys
import termios
import subprocess
import time
import tty
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from xlerobot_rl.real.camera_geometry import DEFAULT_EXTRINSICS, DEFAULT_INTRINSICS, RealCameraGeometry
from xlerobot_rl.real.red_cube_detector import detect_red_cube_rgbd, draw_red_cube_debug


DEFAULT_LEADER_ID = "left_leader"
DEFAULT_FOLLOWER_ID = "right_follower"
DEFAULT_DATASET_NAME = "m4_target_grasp_v0_smoke"
DEFAULT_TASK = "Pick up the red cube with the right arm"
DEFAULT_LEADER_PORT = "/dev/xlerobot_left_leader"
DEFAULT_FOLLOWER_PORT = "/dev/xlerobot_right_follower"
DEFAULT_CAMERA_INDEX = "/dev/xlerobot_head_camera"
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30
DEFAULT_VCODEC = "h264"
DEFAULT_CUBE_SIZE_M = 0.03
COLOR_TO_ID = {"red": 0, "blue": 1, "green": 2}
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-:=" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _format_command(parts: list[str]) -> str:
    lines = [parts[0]]
    for part in parts[1:]:
        lines.append(f"  {_quote(part)}")
    return " \\\n".join(lines)


def _camera_config(args: argparse.Namespace) -> str:
    camera_index_or_path: int | str = args.camera_index
    if isinstance(camera_index_or_path, str) and camera_index_or_path.isdigit():
        camera_index_or_path = int(camera_index_or_path)

    # Current LeRobot/draccus accepts dict values as JSON strings for CLI args.
    return json.dumps(
        {
            "front": {
                "type": "opencv",
                "index_or_path": camera_index_or_path,
                "width": args.camera_width,
                "height": args.camera_height,
                "fps": args.camera_fps,
            }
        },
        separators=(",", ":"),
    )


def build_record_parts(args: argparse.Namespace) -> list[str]:
    dataset_repo_id = args.dataset_repo_id or f"local/{args.dataset_name}"
    parts = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.id={args.follower_id}",
        f"--robot.max_relative_target={args.max_relative_target}",
        f"--robot.cameras={_camera_config(args)}",
        "--teleop.type=so101_leader",
        f"--teleop.port={args.leader_port}",
        f"--teleop.id={args.leader_id}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={args.raw_dataset_root}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.single_task={args.instruction}",
        "--dataset.push_to_hub=false",
        "--dataset.video=true",
        f"--dataset.vcodec={args.vcodec}",
        f"--display_data={str(args.display_data).lower()}",
    ]
    if getattr(args, "resume_recording", False):
        parts.append("--resume=true")
    return parts


def build_teleop_parts(args: argparse.Namespace, teleop_time_s: float | None = None) -> list[str]:
    return [
        "lerobot-teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        f"--robot.id={args.follower_id}",
        f"--robot.max_relative_target={args.max_relative_target}",
        "--teleop.type=so101_leader",
        f"--teleop.port={args.leader_port}",
        f"--teleop.id={args.leader_id}",
        f"--fps={args.fps}",
        f"--teleop_time_s={args.teleop_time_s if teleop_time_s is None else teleop_time_s}",
        f"--display_data={str(args.display_data).lower()}",
    ]


def build_teleop_command(args: argparse.Namespace) -> str:
    return _format_command(build_teleop_parts(args))


def build_calibration_commands(args: argparse.Namespace) -> dict[str, str]:
    return {
        "Calibrate leader": _format_command(
            [
                "lerobot-calibrate",
                "--teleop.type=so101_leader",
                f"--teleop.port={_quote(args.leader_port)}",
                f"--teleop.id={_quote(args.leader_id)}",
            ]
        ),
        "Calibrate follower": _format_command(
            [
                "lerobot-calibrate",
                "--robot.type=so101_follower",
                f"--robot.port={_quote(args.follower_port)}",
                f"--robot.id={_quote(args.follower_id)}",
            ]
        ),
    }


def git_commit() -> str:
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


def _target_snapshot_yaml(target_snapshot: dict | None) -> str:
    if target_snapshot is None:
        return """target_snapshot:
  status: "not_captured"
"""
    if not target_snapshot.get("success"):
        reason = str(target_snapshot.get("reason", "unknown")).replace('"', '\\"')
        return f"""target_snapshot:
  status: "failed"
  reason: "{reason}"
"""

    target_pos = target_snapshot["object"]["pos_base_m"]
    target_camera = target_snapshot["object"]["pos_camera_m"]
    bbox = target_snapshot["object"]["bbox"]
    centroid = target_snapshot["object"]["attributes"]["centroid_px"]
    debug_path = target_snapshot.get("debug_image", "")
    result_path = target_snapshot.get("result_json", "")
    return f"""target_snapshot:
  status: "captured"
  source: "real_rgbd_hsv"
  target_pos_base_initial_m: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]
  target_pos_camera_initial_m: [{target_camera[0]:.6f}, {target_camera[1]:.6f}, {target_camera[2]:.6f}]
  target_visible_initial: true
  bbox_initial_px: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]
  centroid_initial_px: [{centroid[0]:.3f}, {centroid[1]:.3f}]
  depth_median_m: {target_snapshot["depth_median_m"]:.6f}
  cube_size_m: {target_snapshot["object"]["attributes"]["cube_size_m"]:.6f}
  debug_image: "{debug_path}"
  result_json: "{result_path}"
"""


def write_dataset_info(args: argparse.Namespace, target_snapshot: dict | None = None) -> Path:
    args.bc_dataset_root.mkdir(parents=True, exist_ok=True)
    target_color_id = COLOR_TO_ID[args.target_color]
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    target_snapshot_block = _target_snapshot_yaml(target_snapshot)
    content = f"""schema_version: "bc_m4_v0"
source: "real_lerobot"
env_id: "TargetConditionedArmGrasp-v0"
robot: "XLeRobot 0.4.0"
arm: "right"
intended_control_hz: {args.fps}
action_space: "right_arm_joint_delta_6d"
joint_order:
  - shoulder_pan
  - shoulder_lift
  - elbow_flex
  - wrist_flex
  - wrist_roll
gripper_convention: "0=closed, 1=open"
skill_id_map:
  top_grasp: 0
color_id_map:
  red: 0
  blue: 1
  green: 2
dataset_name: "{args.dataset_name}"
raw_lerobot_root: "{args.raw_dataset_root}"
target_color: "{args.target_color}"
target_color_id: {target_color_id}
instruction: "{args.instruction}"
num_episodes_planned: {args.num_episodes}
episode_time_s: {args.episode_time_s}
reset_time_s: {args.reset_time_s}
leader_id: "{args.leader_id}"
follower_id: "{args.follower_id}"
leader_port: "{args.leader_port}"
follower_port: "{args.follower_port}"
camera:
  type: "opencv"
  index_or_path: "{args.camera_index}"
  width: {args.camera_width}
  height: {args.camera_height}
  fps: {args.camera_fps}
target_grounding:
  enabled: {str(args.target_snapshot).lower()}
  detector: "hsv_rgbd_red_cube_v0"
  intrinsics: "{args.intrinsics}"
  extrinsics: "{args.extrinsics}"
  cube_size_m: {args.cube_size_m}
  min_area_px: {args.target_min_area}
{target_snapshot_block.rstrip()}
video_codec: "{args.vcodec}"
created_at: "{created_at}"
git_commit: "{git_commit()}"
notes: "Smoke/BC recording metadata. Pre-record target snapshot seeds target-conditioned conversion."
"""
    path = args.bc_dataset_root / "dataset_info.yaml"
    path.write_text(content)
    return path


def capture_target_snapshot(args: argparse.Namespace) -> dict:
    if args.target_color != "red":
        return {
            "success": False,
            "reason": "pre-record RGB-D snapshot currently supports red target only",
        }

    try:
        import pyrealsense2 as rs
    except Exception as exc:
        return {"success": False, "reason": f"pyrealsense2 import failed: {exc}"}

    try:
        geometry = RealCameraGeometry.from_config(args.intrinsics, args.extrinsics)
        pipeline = rs.pipeline()
        config = rs.config()
        if args.realsense_serial:
            config.enable_device(args.realsense_serial)
        config.enable_stream(
            rs.stream.color,
            args.camera_width,
            args.camera_height,
            rs.format.bgr8,
            args.camera_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            args.camera_width,
            args.camera_height,
            rs.format.z16,
            args.camera_fps,
        )
    except Exception as exc:
        return {"success": False, "reason": f"target snapshot setup failed: {exc}"}

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        for _ in range(max(args.target_warmup_frames, 0)):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return {"success": False, "reason": "failed to capture aligned RGB-D frames"}

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * depth_scale

        obj, result = detect_red_cube_rgbd(
            frame_bgr=color_bgr,
            depth_m=depth_m,
            geometry=geometry,
            cube_size_m=args.cube_size_m,
            min_area=args.target_min_area,
        )
        debug_bgr = draw_red_cube_debug(color_bgr, obj, mask=obj.mask if obj is not None else None)
        result["captured_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        result["depth_scale"] = depth_scale
        result["camera"] = {
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "serial": args.realsense_serial,
        }
        if obj is not None:
            result["object"] = {
                "name": obj.name,
                "bbox": list(obj.bbox),
                "confidence": float(obj.confidence),
                "pos_camera_m": obj.pos_camera.tolist(),
                "pos_base_m": obj.pos_world.tolist(),
                "attributes": obj.attributes,
                "detection_method": obj.detection_method,
            }

        args.bc_dataset_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir = args.bc_dataset_root / "target_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = "pre_record_target"
        rgb_path = snapshot_dir / f"{stem}_rgb.png"
        debug_path = snapshot_dir / f"{stem}_debug.png"
        mask_path = snapshot_dir / f"{stem}_mask.png"
        depth_path = snapshot_dir / f"{stem}_depth_mm.png"
        result_path = snapshot_dir / f"{stem}_result.json"
        cv2.imwrite(str(rgb_path), color_bgr)
        cv2.imwrite(str(debug_path), debug_bgr)
        cv2.imwrite(str(depth_path), np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
        if obj is not None:
            cv2.imwrite(str(mask_path), obj.mask.astype(np.uint8) * 255)
        result["rgb_image"] = str(rgb_path)
        result["debug_image"] = str(debug_path)
        result["depth_image"] = str(depth_path)
        if obj is not None:
            result["mask_image"] = str(mask_path)
        result["result_json"] = str(result_path)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as exc:
        return {"success": False, "reason": f"target snapshot capture failed: {exc}"}
    finally:
        if started:
            pipeline.stop()
            time.sleep(max(args.post_snapshot_release_s, 0.0))


def capture_target_snapshot_with_retries(args: argparse.Namespace) -> dict:
    attempts = max(args.target_snapshot_retries, 1)
    result: dict = {"success": False, "reason": "not attempted"}
    for attempt in range(attempts):
        result = capture_target_snapshot(args)
        if result.get("success"):
            return result
        reason = str(result.get("reason", "unknown"))
        if attempt < attempts - 1:
            print(
                f"WARNING: target snapshot failed ({attempt + 1}/{attempts}): {reason}"
            )
            print(f"Retrying in {args.target_snapshot_retry_delay_s:.1f}s...")
            time.sleep(args.target_snapshot_retry_delay_s)
    return result


def check_console_scripts() -> list[str]:
    return [
        name
        for name in ("lerobot-calibrate", "lerobot-teleoperate", "lerobot-record")
        if shutil.which(name) is None
    ]


def has_lerobot_episode_data(dataset_root: Path) -> bool:
    return any(dataset_root.glob("data/chunk-*/file-*.parquet")) or any(
        dataset_root.glob("videos/*/chunk-*/file-*.mp4")
    )


def check_record_output_path(args: argparse.Namespace) -> bool:
    if not args.raw_dataset_root.exists():
        return True

    print("\nERROR: LeRobot dataset root already exists:")
    print(f"  {args.raw_dataset_root}")
    if has_lerobot_episode_data(args.raw_dataset_root):
        print("It appears to contain episode data. Use a new --dataset-name to avoid mixing runs.")
    else:
        print("It does not appear to contain episode parquet/video files; it may be a failed partial run.")
        print("Remove it manually if you want to reuse the same dataset name.")
    print("\nRecommended: rerun with a fresh name, for example:")
    print(f"  --dataset-name {args.dataset_name}_run2")
    return False


def print_recording_reminders(args: argparse.Namespace) -> None:
    print("\nPre-record checklist:")
    print(f"  - Target cube is {args.target_color} and is visible from camera {args.camera_index}.")
    print("  - Distractor blocks are placed, but not blocking the target.")
    print("  - Right follower arm starts from a safe neutral pose.")
    print("  - Leader arm can move freely and does not hit the table or camera.")
    print("  - Emergency stop / power switch is reachable.")
    if args.target_snapshot:
        print("  - A pre-record target RGB-D snapshot will be captured before recording.")
    print(f"  - You are about to record {args.num_episodes} episode(s), {args.episode_time_s}s each.")


def wait_for_space_to_record() -> bool:
    prompt = "\nPress SPACE to start recording, or q to cancel: "
    if not sys.stdin.isatty():
        answer = input("\nType 'start' to begin recording, or anything else to cancel: ")
        return answer.strip().lower() == "start"

    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                print("start")
                return True
            if ch.lower() == "q":
                print("cancel")
                return False
            if ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_space_or_quit(prompt: str) -> str:
    if not sys.stdin.isatty():
        answer = input(f"\n{prompt} Type 'start' or 'q': ")
        return "space" if answer.strip().lower() == "start" else "q"

    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                print("space")
                return "space"
            if ch.lower() == "q":
                print("quit")
                return "q"
            if ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def poll_space_or_quit(timeout_s: float = 0.0) -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    ch = sys.stdin.read(1)
    if ch == " ":
        return "space"
    if ch.lower() == "q":
        return "q"
    if ch == "\x03":
        raise KeyboardInterrupt
    return None


def ask_keep_episode() -> str:
    prompt = "Save this episode? [y] save / [n] discard / [q] quit: "
    if not sys.stdin.isatty():
        answer = input(prompt)
        answer = answer.strip().lower()
        if answer.startswith("y"):
            return "save"
        if answer.startswith("q"):
            return "quit"
        return "discard"

    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1).lower()
            if ch == "y":
                print("save")
                return "save"
            if ch == "n":
                print("discard")
                return "discard"
            if ch == "q":
                print("quit")
                return "quit"
            if ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def send_lerobot_finish_key() -> bool:
    """Ask LeRobot's internal keyboard listener to end the current episode.

    LeRobot 0.4.4 uses Right Arrow as "exit current episode and save".
    We keep Space as the operator-facing key and synthesize Right Arrow here.
    """

    try:
        from pynput.keyboard import Controller, Key
    except Exception as exc:
        print(f"WARNING: pynput is unavailable; press Right Arrow manually to end episode. ({exc})")
        return False

    keyboard = Controller()
    keyboard.press(Key.right)
    keyboard.release(Key.right)
    return True


def dataset_has_saved_episode(dataset_root: Path) -> bool:
    return any(dataset_root.glob("data/chunk-*/file-*.parquet"))


def connect_follower_for_ready_return(args: argparse.Namespace):
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            cameras={},
            max_relative_target=args.return_max_relative_target,
        )
    )
    follower.connect()
    return follower


def connect_leader_for_ready_return(args: argparse.Namespace, lock: bool = True):
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id=args.leader_id,
        )
    )
    leader.connect()
    if lock:
        # Leader arms are normally passive for teleop. Enable torque only during
        # automatic ready return / locked waiting, then disconnect to release.
        leader.bus.enable_torque()
    return leader


def read_follower_pose(follower) -> dict[str, float]:
    obs = follower.get_observation()
    return {name: float(obs[f"{name}.pos"]) for name in MOTOR_ORDER if f"{name}.pos" in obs}


def read_leader_pose(leader) -> dict[str, float]:
    action = leader.get_action()
    return {name: float(action[f"{name}.pos"]) for name in MOTOR_ORDER if f"{name}.pos" in action}


def load_or_capture_dual_ready_pose(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    print("\nReady-pose setup")
    print("Starting in-process leader-follower teleop for ready-pose setup.")
    print("Move the LEFT leader arm until the RIGHT follower reaches the desired start pose.")
    print("Open both grippers and keep the arms clear of the head camera target view.")
    print("Press SPACE to lock both arms and capture ready pose, or q to cancel.")
    follower = connect_follower_for_ready_return(args)
    leader = connect_leader_for_ready_return(args, lock=False)
    try:
        period = 1.0 / max(args.fps, 1e-6)
        deadline = time.perf_counter() + max(args.ready_setup_time_s, 0.0)
        fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        old_settings = termios.tcgetattr(fd) if fd is not None else None
        if fd is not None:
            tty.setcbreak(fd)
        try:
            while time.perf_counter() < deadline:
                loop_start = time.perf_counter()
                leader_action = leader.get_action()
                follower.send_action(leader_action)
                key = poll_space_or_quit(0.0)
                if key == "q":
                    raise KeyboardInterrupt("ready-pose capture cancelled")
                if key == "space":
                    print("space")
                    break
                time.sleep(max(period - (time.perf_counter() - loop_start), 0.0))
            else:
                raise TimeoutError("ready-pose setup timed out")
        finally:
            if fd is not None and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Lock first, then read. This avoids recording a pose after the passive
        # leader sags from gravity.
        follower_pose = read_follower_pose(follower)
        follower.send_action({f"{name}.pos": value for name, value in follower_pose.items()})
        leader_pose = read_leader_pose(leader)
        leader.bus.sync_write("Goal_Position", leader_pose)
        leader.bus.enable_torque()
        time.sleep(max(args.ready_setup_release_s, 0.0))
        follower_pose = read_follower_pose(follower)
        leader_pose = read_leader_pose(leader)
    finally:
        follower.disconnect()
        leader.disconnect()

    for label, pose in [("follower", follower_pose), ("leader", leader_pose)]:
        missing = [name for name in MOTOR_ORDER if name not in pose]
        if missing:
            raise RuntimeError(f"{label} observation missing ready-pose joints: {missing}")

    print("Captured follower ready pose:")
    print("  " + ", ".join(f"{name}={follower_pose[name]:+.2f}" for name in MOTOR_ORDER))
    print("Captured leader ready pose:")
    print("  " + ", ".join(f"{name}={leader_pose[name]:+.2f}" for name in MOTOR_ORDER))
    return {"follower": follower_pose, "leader": leader_pose}


def interpolate_pose(start_pose: dict[str, float], ready_pose: dict[str, float], step_size: float) -> int:
    max_delta = max(abs(ready_pose[name] - start_pose[name]) for name in MOTOR_ORDER)
    return max(1, int(np.ceil(max_delta / max(step_size, 1e-6))))


def return_follower_to_ready_pose(args: argparse.Namespace, ready_pose: dict[str, float]) -> None:
    print("Returning follower to ready pose...")
    follower = connect_follower_for_ready_return(args)
    try:
        start_pose = read_follower_pose(follower)
        steps = interpolate_pose(start_pose, ready_pose, args.return_step_deg)
        period = 1.0 / max(args.return_fps, 1e-6)
        for step in range(1, steps + 1):
            alpha = step / steps
            action = {
                f"{name}.pos": start_pose[name] + (ready_pose[name] - start_pose[name]) * alpha
                for name in MOTOR_ORDER
            }
            follower.send_action(action)
            time.sleep(period)
        final_pose = read_follower_pose(follower)
        max_err = max(abs(ready_pose[name] - final_pose[name]) for name in MOTOR_ORDER)
        print(f"Ready return done. max joint error: {max_err:.2f}")
    finally:
        follower.disconnect()


def return_leader_to_ready_pose(args: argparse.Namespace, ready_pose: dict[str, float]) -> None:
    print("Returning leader to ready pose...")
    leader = connect_leader_for_ready_return(args)
    try:
        start_pose = read_leader_pose(leader)
        steps = interpolate_pose(start_pose, ready_pose, args.return_step_deg)
        period = 1.0 / max(args.return_fps, 1e-6)
        for step in range(1, steps + 1):
            alpha = step / steps
            goal_pos = {
                name: start_pose[name] + (ready_pose[name] - start_pose[name]) * alpha
                for name in MOTOR_ORDER
            }
            leader.bus.sync_write("Goal_Position", goal_pos)
            time.sleep(period)
        final_pose = read_leader_pose(leader)
        max_err = max(abs(ready_pose[name] - final_pose[name]) for name in MOTOR_ORDER)
        print(f"Leader ready return done. max joint error: {max_err:.2f}")
    finally:
        leader.disconnect()


def return_dual_to_ready_pose(args: argparse.Namespace, ready_pose: dict[str, dict[str, float]]) -> None:
    print("\nReturning both arms to ready pose...")
    # Return follower first so the active robot arm gets out of the task area,
    # then move the passive leader back to the matching pose.
    return_follower_to_ready_pose(args, ready_pose["follower"])
    return_leader_to_ready_pose(args, ready_pose["leader"])


def lock_dual_at_ready_pose(args: argparse.Namespace, ready_pose: dict[str, dict[str, float]]) -> tuple[object, object]:
    print("\nReturning both arms to ready pose and locking them...")
    follower = connect_follower_for_ready_return(args)
    leader = connect_leader_for_ready_return(args, lock=True)
    try:
        follower_start = read_follower_pose(follower)
        leader_start = read_leader_pose(leader)
        steps = max(
            interpolate_pose(follower_start, ready_pose["follower"], args.return_step_deg),
            interpolate_pose(leader_start, ready_pose["leader"], args.return_step_deg),
        )
        period = 1.0 / max(args.return_fps, 1e-6)
        for step in range(1, steps + 1):
            alpha = step / steps
            follower_action = {
                f"{name}.pos": follower_start[name]
                + (ready_pose["follower"][name] - follower_start[name]) * alpha
                for name in MOTOR_ORDER
            }
            leader_goal = {
                name: leader_start[name] + (ready_pose["leader"][name] - leader_start[name]) * alpha
                for name in MOTOR_ORDER
            }
            follower.send_action(follower_action)
            leader.bus.sync_write("Goal_Position", leader_goal)
            time.sleep(period)
        follower_final = read_follower_pose(follower)
        leader_final = read_leader_pose(leader)
        follower_err = max(abs(ready_pose["follower"][name] - follower_final[name]) for name in MOTOR_ORDER)
        leader_err = max(abs(ready_pose["leader"][name] - leader_final[name]) for name in MOTOR_ORDER)
        print(f"Ready lock done. follower max err={follower_err:.2f}, leader max err={leader_err:.2f}")
        return follower, leader
    except Exception:
        follower.disconnect()
        leader.disconnect()
        raise


def release_dual_ready_lock(follower, leader) -> None:
    for device in (follower, leader):
        try:
            device.disconnect()
        except Exception as exc:
            print(f"WARNING: failed to disconnect ready lock device: {exc}")


def run_single_record_attempt(args: argparse.Namespace) -> int:
    record_parts = build_record_parts(args)
    process = subprocess.Popen(record_parts)
    print("\nRecording is running.")
    print("Press SPACE to end this episode. If SPACE does not stop it, press Right Arrow once.")
    key = wait_for_space_or_quit("Press SPACE to end episode, or q to request stop: ")
    if key == "space":
        send_lerobot_finish_key()
    else:
        process.terminate()
        return 130

    try:
        return process.wait(timeout=max(20.0, args.reset_time_s + 20.0))
    except subprocess.TimeoutExpired:
        print("WARNING: lerobot-record did not exit after finish key; terminating it.")
        process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def copy_args_for_dataset(args: argparse.Namespace, dataset_name: str) -> argparse.Namespace:
    ep_args = copy(args)
    ep_args.dataset_name = dataset_name
    ep_args.dataset_repo_id = f"local/{dataset_name}"
    ep_args.raw_dataset_root = Path("data/real/lerobot") / dataset_name
    ep_args.bc_dataset_root = Path("data/bc") / dataset_name
    ep_args.num_episodes = 1
    ep_args.resume_recording = False
    return ep_args


def run_continuous_recording(args: argparse.Namespace) -> int:
    if not args.run_record:
        print("\nContinuous mode is a recording mode. Add --run-record to actually collect data.")
        return 0

    session_root = Path("data/bc") / args.dataset_name
    session_root.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    discarded: list[str] = []
    trial_idx = 0
    ready_pose = load_or_capture_dual_ready_pose(args) if args.auto_return_ready else None

    print("\nContinuous recording mode")
    print("  SPACE starts one episode.")
    print("  SPACE ends the current episode.")
    print("  After it ends: y saves, n discards, q quits.")
    print("  Each saved episode is stored as a separate LeRobot dataset root.")
    if ready_pose is not None:
        print("  Before each trial, both arms return to ready pose and stay locked until SPACE.")

    while args.max_trials is None or trial_idx < args.max_trials:
        ready_lock = None
        if ready_pose is not None:
            ready_lock = lock_dual_at_ready_pose(args, ready_pose)
            print("Both arms are locked at ready pose.")

        key = wait_for_space_or_quit(f"\nTrial {trial_idx:03d} ready. Press SPACE to start recording, or q to quit: ")
        if key == "q":
            if ready_lock is not None:
                release_dual_ready_lock(*ready_lock)
            break

        dataset_name = f"{args.dataset_name}_ep{trial_idx:03d}"
        ep_args = copy_args_for_dataset(args, dataset_name)

        if ep_args.raw_dataset_root.exists() or ep_args.bc_dataset_root.exists():
            print(f"ERROR: trial dataset already exists: {dataset_name}")
            print("Use a fresh --dataset-name or remove the existing trial directory manually.")
            return 1

        target_snapshot = None
        info_path = write_dataset_info(ep_args)
        if ep_args.target_snapshot:
            print("\nCapturing pre-record target RGB-D snapshot...")
            target_snapshot = capture_target_snapshot_with_retries(ep_args)
            info_path = write_dataset_info(ep_args, target_snapshot=target_snapshot)
            if target_snapshot.get("success"):
                pos = target_snapshot["object"]["pos_base_m"]
                print(
                    "Target snapshot OK: "
                    f"pos_base_m=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                )
            else:
                print(f"WARNING: target snapshot failed: {target_snapshot.get('reason', 'unknown')}")
                if ep_args.require_target_snapshot:
                    print("Skipping this trial because --require-target-snapshot was set.")
                    shutil.rmtree(ep_args.bc_dataset_root, ignore_errors=True)
                    if ready_lock is not None:
                        release_dual_ready_lock(*ready_lock)
                    trial_idx += 1
                    continue
        print(f"Metadata: {info_path}")

        if ready_lock is not None:
            print("Releasing ready lock and starting recording...")
            release_dual_ready_lock(*ready_lock)
            ready_lock = None
            time.sleep(0.2)

        returncode = run_single_record_attempt(ep_args)
        if returncode != 0:
            print(f"WARNING: lerobot-record exited with code {returncode}")
            choice = "discard"
        elif not dataset_has_saved_episode(ep_args.raw_dataset_root):
            print("WARNING: no saved episode parquet found after recording.")
            choice = "discard"
        else:
            choice = ask_keep_episode()

        if choice == "save":
            kept.append(dataset_name)
            print(f"Saved trial: {dataset_name}")
        else:
            discarded.append(dataset_name)
            shutil.rmtree(ep_args.raw_dataset_root, ignore_errors=True)
            shutil.rmtree(ep_args.bc_dataset_root, ignore_errors=True)
            print(f"Discarded trial: {dataset_name}")
            if choice == "quit":
                break

        session_summary = {
            "session_dataset_name": args.dataset_name,
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "kept_datasets": kept,
            "discarded_datasets": discarded,
            "ready_pose": ready_pose,
        }
        (session_root / "continuous_session.json").write_text(
            json.dumps(session_summary, indent=2, ensure_ascii=False)
        )
        trial_idx += 1
        time.sleep(0.2)

    print("\nContinuous recording finished.")
    print(f"Kept {len(kept)} dataset(s): {kept}")
    print(f"Session summary: {session_root / 'continuous_session.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset/session name. Defaults to an automatic timestamped name in continuous mode.",
    )
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--raw-dataset-root", type=Path, default=None)
    parser.add_argument("--bc-dataset-root", type=Path, default=None)
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default="red")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--episode-time-s", type=float, default=15)
    parser.add_argument("--reset-time-s", type=float, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--teleop-time-s", type=float, default=10)
    parser.add_argument("--max-relative-target", type=float, default=15)
    parser.add_argument("--camera-index", default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument(
        "--target-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture a pre-record RGB-D target snapshot and write it into dataset metadata.",
    )
    parser.add_argument(
        "--require-target-snapshot",
        action="store_true",
        help="Abort recording if the pre-record target snapshot fails.",
    )
    parser.add_argument(
        "--realsense-serial",
        default=None,
        help="Optional RealSense serial for the pre-record target snapshot.",
    )
    parser.add_argument("--cube-size-m", type=float, default=DEFAULT_CUBE_SIZE_M)
    parser.add_argument("--target-min-area", type=int, default=300)
    parser.add_argument("--target-warmup-frames", type=int, default=30)
    parser.add_argument(
        "--target-snapshot-retries",
        type=int,
        default=3,
        help="Retry count for pre-record target snapshot, useful when the camera is still busy.",
    )
    parser.add_argument(
        "--target-snapshot-retry-delay-s",
        type=float,
        default=1.5,
        help="Delay between target snapshot retries.",
    )
    parser.add_argument(
        "--post-snapshot-release-s",
        type=float,
        default=1.0,
        help="Wait after releasing RealSense before starting LeRobot/OpenCV recording.",
    )
    parser.add_argument(
        "--vcodec",
        default=DEFAULT_VCODEC,
        help="LeRobot video codec. h264 is easier to inspect locally than the default libsvtav1.",
    )
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-record",
        action="store_true",
        help="Actually execute lerobot-record. Default only prints commands.",
    )
    parser.add_argument(
        "--continuous-record",
        action="store_true",
        help="Collect repeated one-episode demos: SPACE start, SPACE end, then save/discard.",
    )
    parser.add_argument(
        "--auto-return-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before each continuous-record trial, move both leader and follower back to the "
            "ready poses captured at startup and keep them locked until SPACE."
        ),
    )
    parser.add_argument(
        "--return-step-deg",
        type=float,
        default=3.0,
        help="Maximum interpolation step per joint for auto return.",
    )
    parser.add_argument(
        "--return-fps",
        type=float,
        default=15.0,
        help="Command rate for auto return.",
    )
    parser.add_argument(
        "--return-max-relative-target",
        type=float,
        default=5.0,
        help="LeRobot safety clamp for each auto-return command.",
    )
    parser.add_argument(
        "--ready-setup-time-s",
        type=float,
        default=300.0,
        help="Max duration for initial leader-follower ready-pose setup teleop.",
    )
    parser.add_argument(
        "--ready-setup-release-s",
        type=float,
        default=1.0,
        help="Wait after stopping ready setup teleop before reading both ready poses.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum trials in --continuous-record mode. Default runs until q.",
    )
    parser.add_argument(
        "--ready-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause before recording and wait for SPACE. Enabled by default.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset_name is None:
        if args.continuous_record:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.dataset_name = f"m4_target_grasp_v0_bc_session_{ts}"
        else:
            args.dataset_name = DEFAULT_DATASET_NAME

    if args.raw_dataset_root is None:
        args.raw_dataset_root = Path("data/real/lerobot") / args.dataset_name
    if args.bc_dataset_root is None:
        args.bc_dataset_root = Path("data/bc") / args.dataset_name

    missing = check_console_scripts()
    if missing:
        print("Missing LeRobot console scripts:")
        for name in missing:
            print(f"  {name}")
        print("Activate the xlerobot-rl conda environment or reinstall LeRobot.")
        return 1

    if args.continuous_record:
        return run_continuous_recording(args)

    info_path = write_dataset_info(args)
    record_parts = build_record_parts(args)
    record_command = _format_command(record_parts)

    print(f"Wrote dataset metadata template: {info_path}")
    print("\nSafety order:")
    print("  1. Run calibration commands if not already calibrated.")
    print("  2. Run the 10s teleop smoke test.")
    print("  3. Only then run record.")

    for title, command in build_calibration_commands(args).items():
        print(f"\n# {title}")
        print(command)

    print("\n# Teleop smoke test")
    print(build_teleop_command(args))

    print("\n# Record dataset")
    print(record_command)

    if not args.run_record:
        print("\nDry run only. Add --run-record to execute lerobot-record.")
        return 0

    if not check_record_output_path(args):
        return 1

    print_recording_reminders(args)
    if args.ready_prompt and not wait_for_space_to_record():
        print("Recording cancelled before lerobot-record was started.")
        return 130

    target_snapshot = None
    if args.target_snapshot:
        print("\nCapturing pre-record target RGB-D snapshot...")
        target_snapshot = capture_target_snapshot_with_retries(args)
        info_path = write_dataset_info(args, target_snapshot=target_snapshot)
        if target_snapshot.get("success"):
            pos = target_snapshot["object"]["pos_base_m"]
            print(
                "Target snapshot OK: "
                f"pos_base_m=[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
            )
            print(f"Updated dataset metadata: {info_path}")
            print(f"Snapshot debug image: {target_snapshot.get('debug_image')}")
        else:
            print(f"WARNING: target snapshot failed: {target_snapshot.get('reason', 'unknown')}")
            print(f"Updated dataset metadata with failure status: {info_path}")
            if args.require_target_snapshot:
                print("Aborting because --require-target-snapshot was set.")
                return 2

    print("\nExecuting lerobot-record...")
    return subprocess.run(record_parts).returncode


if __name__ == "__main__":
    raise SystemExit(main())
