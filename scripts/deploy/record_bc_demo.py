"""BC demo recording helper for right-arm leader-follower collection.

Default behavior is dry-run: write a dataset_info.yaml template and print the
LeRobot commands needed to calibrate, teleoperate, and record. Use --run-record
only after calibration and a short teleop smoke test pass.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LEADER_ID = "left_leader"
DEFAULT_FOLLOWER_ID = "right_follower"
DEFAULT_DATASET_NAME = "m4_target_grasp_v0_smoke"
DEFAULT_TASK = "Pick up the red cube with the right arm"
COLOR_TO_ID = {"red": 0, "blue": 1, "green": 2}


def _quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-:" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _format_command(parts: list[str]) -> str:
    lines = [parts[0]]
    for part in parts[1:]:
        lines.append(f"  {part}")
    return " \\\n".join(lines)


def _camera_config(args: argparse.Namespace) -> str:
    return (
        "{front: {type: opencv, index_or_path: "
        f"{args.camera_index}, width: {args.camera_width}, "
        f"height: {args.camera_height}, fps: {args.camera_fps}}}}}"
    )


def build_record_parts(args: argparse.Namespace) -> list[str]:
    dataset_repo_id = args.dataset_repo_id or f"local/{args.dataset_name}"
    return [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={_quote(args.follower_port)}",
        f"--robot.id={_quote(args.follower_id)}",
        f"--robot.max_relative_target={args.max_relative_target}",
        f"--robot.cameras={_quote(_camera_config(args))}",
        "--teleop.type=so101_leader",
        f"--teleop.port={_quote(args.leader_port)}",
        f"--teleop.id={_quote(args.leader_id)}",
        f"--dataset.repo_id={_quote(dataset_repo_id)}",
        f"--dataset.root={_quote(str(args.raw_dataset_root))}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.single_task={_quote(args.instruction)}",
        "--dataset.push_to_hub=false",
        "--dataset.video=true",
        f"--display_data={str(args.display_data).lower()}",
    ]


def build_teleop_command(args: argparse.Namespace) -> str:
    return _format_command(
        [
            "lerobot-teleoperate",
            "--robot.type=so101_follower",
            f"--robot.port={_quote(args.follower_port)}",
            f"--robot.id={_quote(args.follower_id)}",
            f"--robot.max_relative_target={args.max_relative_target}",
            "--teleop.type=so101_leader",
            f"--teleop.port={_quote(args.leader_port)}",
            f"--teleop.id={_quote(args.leader_id)}",
            f"--fps={args.fps}",
            f"--teleop_time_s={args.teleop_time_s}",
            f"--display_data={str(args.display_data).lower()}",
        ]
    )


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


def write_dataset_info(args: argparse.Namespace) -> Path:
    args.bc_dataset_root.mkdir(parents=True, exist_ok=True)
    target_color_id = COLOR_TO_ID[args.target_color]
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
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
created_at: "{created_at}"
git_commit: "{git_commit()}"
notes: "Smoke/BC recording metadata. Target conditioning fields may require post-processing."
"""
    path = args.bc_dataset_root / "dataset_info.yaml"
    path.write_text(content)
    return path


def check_console_scripts() -> list[str]:
    return [
        name
        for name in ("lerobot-calibrate", "lerobot-teleoperate", "lerobot-record")
        if shutil.which(name) is None
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    parser.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
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
    parser.add_argument("--camera-index", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run-record",
        action="store_true",
        help="Actually execute lerobot-record. Default only prints commands.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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

    print("\nExecuting lerobot-record...")
    return subprocess.run(record_parts).returncode


if __name__ == "__main__":
    raise SystemExit(main())
