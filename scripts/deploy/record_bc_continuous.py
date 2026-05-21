"""Continuous in-process BC demo recorder for right-arm leader-follower data.

This avoids the torque gap caused by repeatedly launching lerobot-record as a
subprocess. Leader, follower, camera, and dataset stay in one Python process.
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep


DEFAULT_LEADER_PORT = "/dev/xlerobot_left_leader"
DEFAULT_FOLLOWER_PORT = "/dev/xlerobot_right_follower"
DEFAULT_CAMERA_INDEX = "/dev/xlerobot_head_camera"
DEFAULT_TASK = "Pick up the red cube with the right arm"
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def auto_session_name() -> str:
    return f"m4_target_grasp_v0_bc_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


def wait_key(prompt: str, allowed: set[str]) -> str:
    if not sys.stdin.isatty():
        while True:
            answer = input(prompt).strip().lower()
            if answer == "start" and " " in allowed:
                return " "
            if answer[:1] in allowed:
                return answer[:1]

    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in allowed:
                print("space" if ch == " " else ch)
                return ch
            if ch == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def retry_or_quit(prompt: str) -> bool:
    key = wait_key(f"{prompt} Press r to retry, or q to quit: ", {"r", "q"})
    return key == "r"


def comm_retry(
    label: str,
    fn,
    *args,
    attempts: int,
    sleep_s: float,
    warn: bool = True,
    **kwargs,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            last_exc = exc
            if warn:
                print(f"WARNING: {label} failed ({attempt}/{attempts}): {exc}")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def get_leader_action(args: argparse.Namespace, leader: SO101Leader) -> dict[str, float]:
    return comm_retry(
        "leader get_action",
        leader.get_action,
        attempts=args.comm_attempts,
        sleep_s=args.comm_retry_sleep_s,
    )


def get_follower_observation(args: argparse.Namespace, follower: SO101Follower) -> dict[str, object]:
    return comm_retry(
        "follower get_observation",
        follower.get_observation,
        attempts=args.comm_attempts,
        sleep_s=args.comm_retry_sleep_s,
    )


def send_follower_action(
    args: argparse.Namespace,
    follower: SO101Follower,
    action: dict[str, float],
) -> dict[str, float]:
    return comm_retry(
        "follower send_action",
        follower.send_action,
        action,
        attempts=args.comm_attempts,
        sleep_s=args.comm_retry_sleep_s,
    )


def sync_write_leader_goal(
    args: argparse.Namespace,
    leader: SO101Leader,
    goal: dict[str, float],
) -> None:
    comm_retry(
        "leader goal write",
        leader.bus.sync_write,
        "Goal_Position",
        goal,
        attempts=args.comm_attempts,
        sleep_s=args.comm_retry_sleep_s,
    )


def set_leader_torque(
    leader: SO101Leader,
    enabled: bool,
    *,
    num_retry: int,
    attempts: int,
) -> bool:
    action = "enable" if enabled else "disable"
    for attempt in range(1, attempts + 1):
        try:
            if enabled:
                leader.bus.enable_torque(num_retry=num_retry)
            else:
                leader.bus.disable_torque(num_retry=num_retry)
            return True
        except ConnectionError as exc:
            print(f"WARNING: leader torque {action} failed ({attempt}/{attempts}): {exc}")
            time.sleep(0.2)
    return False


def poll_key(timeout_s: float = 0.0) -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def pose_from_prefixed(data: dict[str, float]) -> dict[str, float]:
    return {name: float(data[f"{name}.pos"]) for name in MOTOR_ORDER if f"{name}.pos" in data}


def action_from_pose(pose: dict[str, float]) -> dict[str, float]:
    return {f"{name}.pos": pose[name] for name in MOTOR_ORDER}


def max_pose_error(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a[name] - b[name]) for name in MOTOR_ORDER)


def make_devices(args: argparse.Namespace) -> tuple[SO101Leader, SO101Follower]:
    camera_source: int | Path
    camera_source = int(args.camera_index) if str(args.camera_index).isdigit() else Path(args.camera_index)
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_source,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    }
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            max_relative_target=args.max_relative_target,
            cameras=cameras,
        )
    )
    return leader, follower


def create_dataset(args: argparse.Namespace, follower: SO101Follower) -> LeRobotDataset:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
    )
    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.dataset_name}",
        fps=args.fps,
        root=args.raw_dataset_root,
        robot_type=follower.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=args.vcodec,
    )
    return dataset


def write_session_info(args: argparse.Namespace) -> None:
    args.bc_dataset_root.mkdir(parents=True, exist_ok=True)
    info = {
        "schema_version": "bc_m4_v0",
        "source": "real_lerobot_inprocess",
        "dataset_name": args.dataset_name,
        "raw_lerobot_root": str(args.raw_dataset_root),
        "target_color": args.target_color,
        "target_color_id": {"red": 0, "blue": 1, "green": 2}[args.target_color],
        "instruction": args.instruction,
        "leader_port": args.leader_port,
        "follower_port": args.follower_port,
        "camera": {
            "type": "opencv",
            "index_or_path": str(args.camera_index),
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
        },
        "control_fps": args.fps,
        "created_at": now_utc(),
        "git_commit": git_commit(),
        "notes": "In-process continuous BC collection. Target snapshot/converter fields are added later.",
    }
    (args.bc_dataset_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )


def run_ready_setup(args: argparse.Namespace, leader: SO101Leader, follower: SO101Follower) -> dict[str, dict[str, float]]:
    print("\nReady setup: leader-follower teleop is active.")
    print("Move LEFT leader; RIGHT follower tracks it. Open grippers.")
    print("Press SPACE to lock and capture ready pose, or q to cancel.")
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old_settings = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)
    period = 1.0 / args.fps
    try:
        while True:
            loop_start = time.perf_counter()
            leader_action = get_leader_action(args, leader)
            send_follower_action(args, follower, leader_action)
            key = poll_key(0.0)
            if key == "q":
                raise KeyboardInterrupt("ready setup cancelled")
            if key == " ":
                print("space")
                break
            precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    follower_pose = pose_from_prefixed(get_follower_observation(args, follower))
    send_follower_action(args, follower, action_from_pose(follower_pose))
    leader_pose = pose_from_prefixed(get_leader_action(args, leader))
    if not set_leader_torque(
        leader,
        True,
        num_retry=args.torque_num_retry,
        attempts=args.torque_attempts,
    ):
        raise ConnectionError("failed to lock leader torque while capturing ready pose")
    sync_write_leader_goal(args, leader, leader_pose)
    time.sleep(args.ready_lock_s)
    follower_pose = pose_from_prefixed(get_follower_observation(args, follower))
    leader_pose = pose_from_prefixed(get_leader_action(args, leader))
    print("Captured follower ready:", ", ".join(f"{k}={v:+.2f}" for k, v in follower_pose.items()))
    print("Captured leader ready:", ", ".join(f"{k}={v:+.2f}" for k, v in leader_pose.items()))
    return {"follower": follower_pose, "leader": leader_pose}


def return_and_lock_ready(
    args: argparse.Namespace,
    leader: SO101Leader,
    follower: SO101Follower,
    ready_pose: dict[str, dict[str, float]],
) -> None:
    print("\nReturning both arms to ready pose and holding...")
    while not set_leader_torque(
        leader,
        True,
        num_retry=args.torque_num_retry,
        attempts=args.torque_attempts,
    ):
        if not retry_or_quit("Failed to lock leader torque. Check USB/power if needed."):
            raise KeyboardInterrupt("leader torque lock cancelled")
    follower_start = pose_from_prefixed(get_follower_observation(args, follower))
    leader_start = pose_from_prefixed(get_leader_action(args, leader))
    max_delta = max(
        max_pose_error(follower_start, ready_pose["follower"]),
        max_pose_error(leader_start, ready_pose["leader"]),
    )
    steps = max(1, int(np.ceil(max_delta / max(args.return_step_deg, 1e-6))))
    period = 1.0 / args.return_fps
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
        send_follower_action(args, follower, follower_action)
        sync_write_leader_goal(args, leader, leader_goal)
        precise_sleep(period)
    follower_err = max_pose_error(
        pose_from_prefixed(get_follower_observation(args, follower)),
        ready_pose["follower"],
    )
    leader_err = max_pose_error(pose_from_prefixed(get_leader_action(args, leader)), ready_pose["leader"])
    print(f"Ready hold active. follower err={follower_err:.2f}, leader err={leader_err:.2f}")


def record_one_episode(
    args: argparse.Namespace,
    leader: SO101Leader,
    follower: SO101Follower,
    dataset: LeRobotDataset,
) -> int:
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    print("\nRecording. Press SPACE to end this episode.")
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old_settings = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)
    period = 1.0 / args.fps
    start_t = time.perf_counter()
    frames = 0
    try:
        while True:
            loop_start = time.perf_counter()
            obs = get_follower_observation(args, follower)
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            leader_action = get_leader_action(args, leader)
            action_values = teleop_action_processor((leader_action, obs))
            robot_action = robot_action_processor((action_values, obs))
            send_follower_action(args, follower, robot_action)

            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": args.instruction})
            frames += 1

            key = poll_key(0.0)
            if key == " ":
                print("space")
                break
            if time.perf_counter() - start_t >= args.episode_time_s:
                print("Episode time limit reached.")
                break
            precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--leader-id", default="left_leader")
    parser.add_argument("--follower-id", default="right_follower")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--raw-dataset-root", type=Path, default=None)
    parser.add_argument("--bc-dataset-root", type=Path, default=None)
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default="red")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--episode-time-s", type=float, default=20.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-relative-target", type=float, default=15.0)
    parser.add_argument("--camera-index", default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--return-step-deg", type=float, default=3.0)
    parser.add_argument("--return-fps", type=float, default=15.0)
    parser.add_argument("--ready-lock-s", type=float, default=0.5)
    parser.add_argument("--torque-num-retry", type=int, default=5)
    parser.add_argument("--torque-attempts", type=int, default=3)
    parser.add_argument("--comm-attempts", type=int, default=8)
    parser.add_argument("--comm-retry-sleep-s", type=float, default=0.08)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_name is None:
        args.dataset_name = auto_session_name()
    if args.raw_dataset_root is None:
        args.raw_dataset_root = Path("data/real/lerobot") / args.dataset_name
    if args.bc_dataset_root is None:
        args.bc_dataset_root = Path("data/bc") / args.dataset_name
    if args.raw_dataset_root.exists():
        print(f"ERROR: dataset root already exists: {args.raw_dataset_root}")
        return 1

    leader, follower = make_devices(args)
    dataset = None
    kept = 0
    discarded = 0
    try:
        print("Connecting leader/follower/camera...")
        leader.connect()
        follower.connect()
        dataset = create_dataset(args, follower)
        write_session_info(args)
        ready_pose = run_ready_setup(args, leader, follower)

        while args.max_episodes is None or kept < args.max_episodes:
            return_and_lock_ready(args, leader, follower, ready_pose)
            key = wait_key("\nReady. Press SPACE to start episode, or q to quit: ", {" ", "q"})
            if key == "q":
                break
            while not set_leader_torque(
                leader,
                False,
                num_retry=args.torque_num_retry,
                attempts=args.torque_attempts,
            ):
                if not retry_or_quit("Failed to release leader torque. Check USB/power if needed."):
                    raise KeyboardInterrupt("leader torque release cancelled")
            frames = record_one_episode(args, leader, follower, dataset)
            choice = wait_key(f"Episode captured ({frames} frames). Save? [y/n/q]: ", {"y", "n", "q"})
            if choice == "y":
                dataset.save_episode()
                kept += 1
                print(f"Saved episode {kept - 1}.")
            else:
                dataset._wait_image_writer()
                dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                discarded += 1
                print("Discarded episode.")
                if choice == "q":
                    break
            summary = {
                "dataset_name": args.dataset_name,
                "updated_at": now_utc(),
                "kept_episodes": kept,
                "discarded_episodes": discarded,
                "ready_pose": ready_pose,
            }
            (args.bc_dataset_root / "continuous_session.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
    finally:
        if dataset is not None:
            dataset.finalize()
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()

    print(f"\nDone. Saved episodes: {kept}, discarded: {discarded}")
    print(f"Dataset: {args.raw_dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
