"""Right-arm master-slave bring-up helper.

This script is intentionally dry-run only. It lists likely serial ports and prints
the LeRobot commands needed for the right-arm follower + detached left-arm leader
workflow. It does not connect to motors or send actions.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LEADER_ID = "left_leader"
DEFAULT_FOLLOWER_ID = "right_follower"
DEFAULT_DATASET_REPO_ID = "local/xlerobot_right_arm_smoke"
DEFAULT_DATASET_ROOT = "data/real/lerobot/xlerobot_right_arm_smoke"


@dataclass(frozen=True)
class Ports:
    leader: str
    follower: str


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


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as serial_list_ports
    except ImportError:
        candidates = sorted(
            str(path)
            for pattern in ("ttyACM*", "ttyUSB*", "tty.usb*")
            for path in Path("/dev").glob(pattern)
        )
        return candidates

    ports = [port.device for port in serial_list_ports.comports()]
    return sorted(ports)


def print_ports(_: argparse.Namespace) -> int:
    ports = list_ports()
    if not ports:
        print("No serial ports found under pyserial.")
        print("Check USB connection, robot power, and Linux dialout permissions.")
        print("You can also run: lerobot-find-port")
        return 1

    print("Detected serial ports:")
    for port in ports:
        print(f"  {port}")
    print("\nUse lerobot-find-port if you need to identify which board is leader/follower.")
    return 0


def build_commands(args: argparse.Namespace) -> dict[str, str]:
    ports = Ports(leader=args.leader_port, follower=args.follower_port)
    leader_id = args.leader_id
    follower_id = args.follower_id

    calibrate_leader = _format_command(
        [
            "lerobot-calibrate",
            "--teleop.type=so101_leader",
            f"--teleop.port={_quote(ports.leader)}",
            f"--teleop.id={_quote(leader_id)}",
        ]
    )
    calibrate_follower = _format_command(
        [
            "lerobot-calibrate",
            "--robot.type=so101_follower",
            f"--robot.port={_quote(ports.follower)}",
            f"--robot.id={_quote(follower_id)}",
        ]
    )
    teleop = _format_command(
        [
            "lerobot-teleoperate",
            "--robot.type=so101_follower",
            f"--robot.port={_quote(ports.follower)}",
            f"--robot.id={_quote(follower_id)}",
            f"--robot.max_relative_target={args.max_relative_target}",
            "--teleop.type=so101_leader",
            f"--teleop.port={_quote(ports.leader)}",
            f"--teleop.id={_quote(leader_id)}",
            f"--fps={args.fps}",
            f"--teleop_time_s={args.teleop_time_s}",
            f"--display_data={str(args.display_data).lower()}",
        ]
    )

    camera = (
        "{front: {type: opencv, index_or_path: "
        f"{args.camera_index}, width: {args.camera_width}, "
        f"height: {args.camera_height}, fps: {args.camera_fps}}}}}"
    )
    record = _format_command(
        [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={_quote(ports.follower)}",
            f"--robot.id={_quote(follower_id)}",
            f"--robot.max_relative_target={args.max_relative_target}",
            f"--robot.cameras={_quote(camera)}",
            "--teleop.type=so101_leader",
            f"--teleop.port={_quote(ports.leader)}",
            f"--teleop.id={_quote(leader_id)}",
            f"--dataset.repo_id={_quote(args.dataset_repo_id)}",
            f"--dataset.root={_quote(args.dataset_root)}",
            f"--dataset.num_episodes={args.num_episodes}",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.reset_time_s={args.reset_time_s}",
            f"--dataset.single_task={_quote(args.single_task)}",
            "--dataset.push_to_hub=false",
            "--dataset.video=true",
            f"--display_data={str(args.display_data).lower()}",
        ]
    )

    return {
        "1. Calibrate leader": calibrate_leader,
        "2. Calibrate follower": calibrate_follower,
        "3. Smoke-test teleop": teleop,
        "4. Record smoke dataset": record,
    }


def print_commands(args: argparse.Namespace) -> int:
    missing = [
        name
        for name in ("lerobot-calibrate", "lerobot-teleoperate", "lerobot-record")
        if shutil.which(name) is None
    ]
    if missing:
        print("Missing LeRobot console scripts:")
        for name in missing:
            print(f"  {name}")
        print("Activate the xlerobot-rl conda environment or reinstall LeRobot.")
        return 1

    for title, command in build_commands(args).items():
        print(f"\n# {title}")
        print(command)
    print("\nRun the commands in order. The teleop command moves the follower arm.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ports = subparsers.add_parser("ports", help="List likely serial ports.")
    ports.set_defaults(func=print_ports)

    commands = subparsers.add_parser("commands", help="Print LeRobot bring-up commands.")
    commands.add_argument("--leader-port", required=True, help="Detached left leader arm port.")
    commands.add_argument("--follower-port", required=True, help="Right follower arm port.")
    commands.add_argument("--leader-id", default=DEFAULT_LEADER_ID)
    commands.add_argument("--follower-id", default=DEFAULT_FOLLOWER_ID)
    commands.add_argument("--fps", type=int, default=15)
    commands.add_argument("--teleop-time-s", type=float, default=10)
    commands.add_argument("--max-relative-target", type=float, default=15)
    commands.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)
    commands.add_argument("--camera-index", default="0")
    commands.add_argument("--camera-width", type=int, default=640)
    commands.add_argument("--camera-height", type=int, default=480)
    commands.add_argument("--camera-fps", type=int, default=30)
    commands.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    commands.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    commands.add_argument("--num-episodes", type=int, default=2)
    commands.add_argument("--episode-time-s", type=float, default=15)
    commands.add_argument("--reset-time-s", type=float, default=10)
    commands.add_argument(
        "--single-task",
        default="Pick up the red cube with the right arm",
    )
    commands.set_defaults(func=print_commands)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
