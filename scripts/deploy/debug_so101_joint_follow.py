"""Debug SO101 leader/follower joint tracking.

Default mode is read-only: it connects to the SO101 leader and follower, reads
their positions, and prints the selected joint delta. Use --send to also send
leader positions to the follower and print the action actually sent.
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from lerobot.utils.robot_utils import precise_sleep


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _pos(data: dict[str, float], motor: str) -> float:
    return float(data[f"{motor}.pos"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-id", default="left_leader")
    parser.add_argument("--follower-id", default="right_follower")
    parser.add_argument("--joint", default="shoulder_lift", choices=MOTORS)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--max-relative-target", type=float, default=15.0)
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send leader actions to follower. Without this, the script is read-only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id=args.leader_id,
        )
    )
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            max_relative_target=args.max_relative_target if args.send else None,
            cameras={},
        )
    )

    leader.connect()
    follower.connect()

    print(f"Debug joint: {args.joint}")
    print(f"Mode: {'SEND' if args.send else 'READ-ONLY'}")
    print("Columns: leader, follower_before, sent, follower_after, leader-follower_after")

    period = 1.0 / args.fps
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration_s:
            loop_start = time.perf_counter()
            action = leader.get_action()
            obs_before = follower.get_observation()
            sent = action
            obs_after = obs_before

            if args.send:
                sent = follower.send_action(action)
                obs_after = follower.get_observation()

            leader_v = _pos(action, args.joint)
            before_v = _pos(obs_before, args.joint)
            sent_v = _pos(sent, args.joint)
            after_v = _pos(obs_after, args.joint)
            err = leader_v - after_v
            print(
                f"{args.joint}: "
                f"leader={leader_v:+8.2f} "
                f"before={before_v:+8.2f} "
                f"sent={sent_v:+8.2f} "
                f"after={after_v:+8.2f} "
                f"err={err:+8.2f}"
            )

            dt = time.perf_counter() - loop_start
            precise_sleep(max(period - dt, 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        leader.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
