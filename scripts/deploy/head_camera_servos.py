"""Control the two head-camera servos and read their angles.

This utility talks directly to a Feetech servo bus through LeRobot. Use
``scan`` first if the head servo port or IDs are not confirmed.
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


DEFAULT_PORT = "/dev/xlerobot_right_follower"
DEFAULT_PAN_ID = 7
DEFAULT_TILT_ID = 8
DEFAULT_CENTER_RAW = 2048
STS3215_RESOLUTION = 4096
PAN = "head_pan"
TILT = "head_tilt"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_bus(args: argparse.Namespace) -> FeetechMotorsBus:
    motors = {
        PAN: Motor(id=args.pan_id, model=args.model, norm_mode=MotorNormMode.DEGREES),
        TILT: Motor(id=args.tilt_id, model=args.model, norm_mode=MotorNormMode.DEGREES),
    }
    return FeetechMotorsBus(args.port, motors)


def raw_to_deg(raw: float, center_raw: int) -> float:
    return (float(raw) - center_raw) * 360.0 / (STS3215_RESOLUTION - 1)


def deg_to_raw(deg: float, center_raw: int) -> int:
    return int(round(center_raw + deg * (STS3215_RESOLUTION - 1) / 360.0))


def read_raw(bus: FeetechMotorsBus, *, retries: int) -> dict[str, int]:
    values = bus.sync_read("Present_Position", [PAN, TILT], normalize=False, num_retry=retries)
    return {name: int(values[name]) for name in (PAN, TILT)}


def raw_as_deg(values: dict[str, int], args: argparse.Namespace) -> dict[str, float]:
    return {name: raw_to_deg(values[name], args.center_raw) for name in (PAN, TILT)}


def write_angles(
    bus: FeetechMotorsBus,
    pan: float,
    tilt: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    target_deg = {
        PAN: clamp(pan, args.pan_min, args.pan_max),
        TILT: clamp(tilt, args.tilt_min, args.tilt_max),
    }
    target_raw = {
        name: int(clamp(deg_to_raw(value, args.center_raw), args.raw_min, args.raw_max))
        for name, value in target_deg.items()
    }
    bus.sync_write("Goal_Position", target_raw, normalize=False, num_retry=args.retries)
    return target_deg


def print_angles(prefix: str, raw_values: dict[str, int], args: argparse.Namespace) -> None:
    deg = raw_as_deg(raw_values, args)
    print(
        f"{prefix} "
        f"pan={deg[PAN]:+.2f} deg raw={raw_values[PAN]}, "
        f"tilt={deg[TILT]:+.2f} deg raw={raw_values[TILT]}"
    )


def connect_bus(args: argparse.Namespace) -> FeetechMotorsBus:
    bus = make_bus(args)
    bus.connect()
    return bus


def cmd_scan(args: argparse.Namespace) -> int:
    port = args.port
    if port == DEFAULT_PORT and not Path(port).exists():
        print(f"WARNING: default port does not exist: {port}")
        print("Use --port /dev/ttyUSB? or create a udev symlink for the head servo bus.")
    print(f"Scanning Feetech servos on {port} ...")
    result = FeetechMotorsBus.scan_port(port)
    if not result:
        print("No motors found.")
        return 1
    print("\nFound motors:")
    for baudrate, ids in result.items():
        print(f"  baudrate={baudrate}: ids={ids}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    bus = connect_bus(args)
    try:
        print_angles("current:", read_raw(bus, retries=args.retries), args)
    finally:
        bus.disconnect(disable_torque=False)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    bus = connect_bus(args)
    try:
        if args.enable_torque:
            bus.enable_torque([PAN, TILT], num_retry=args.retries)
        current_raw = read_raw(bus, retries=args.retries)
        current_deg = raw_as_deg(current_raw, args)
        pan = current_deg[PAN] if args.pan is None else args.pan
        tilt = current_deg[TILT] if args.tilt is None else args.tilt
        target = write_angles(bus, pan, tilt, args)
        target_raw = {name: deg_to_raw(value, args.center_raw) for name, value in target.items()}
        print_angles("target: ", target_raw, args)
        if args.wait_s > 0:
            time.sleep(args.wait_s)
            print_angles("current:", read_raw(bus, retries=args.retries), args)
        if args.disable_torque:
            bus.disable_torque([PAN, TILT], num_retry=args.retries)
    finally:
        bus.disconnect(disable_torque=False)
    return 0


def poll_key(timeout_s: float = 0.0) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch


def cmd_jog(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("jog mode requires an interactive terminal")
    bus = connect_bus(args)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        bus.enable_torque([PAN, TILT], num_retry=args.retries)
        current_raw = read_raw(bus, retries=args.retries)
        target = raw_as_deg(current_raw, args)
        print_angles("current:", current_raw, args)
        print("\nJog controls:")
        print("  a/d: pan -/+")
        print("  w/s: tilt +/-")
        print("  r: read current")
        print("  h: hold current as target")
        print("  0: go to zero")
        print("  q: quit")
        print(f"step={args.step_deg:.2f} deg, limits pan=[{args.pan_min},{args.pan_max}], tilt=[{args.tilt_min},{args.tilt_max}]")
        while True:
            key = poll_key(0.1)
            if key is None:
                continue
            if key == "q":
                print("q")
                break
            if key == "a":
                target[PAN] -= args.step_deg
            elif key == "d":
                target[PAN] += args.step_deg
            elif key == "w":
                target[TILT] += args.step_deg
            elif key == "s":
                target[TILT] -= args.step_deg
            elif key == "r":
                print_angles("current:", read_raw(bus, retries=args.retries), args)
                continue
            elif key == "h":
                current_raw = read_raw(bus, retries=args.retries)
                target = raw_as_deg(current_raw, args)
                write_angles(bus, target[PAN], target[TILT], args)
                print_angles("hold:   ", current_raw, args)
                continue
            elif key == "0":
                target = {PAN: 0.0, TILT: 0.0}
            else:
                continue
            target = write_angles(bus, target[PAN], target[TILT], args)
            target_raw = {name: deg_to_raw(value, args.center_raw) for name, value in target.items()}
            print_angles("target: ", target_raw, args)
            time.sleep(args.settle_s)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if args.disable_torque:
            bus.disable_torque([PAN, TILT], num_retry=args.retries)
        bus.disconnect(disable_torque=False)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default=DEFAULT_PORT, help="Feetech servo serial port.")
    parser.add_argument("--pan-id", type=int, default=DEFAULT_PAN_ID)
    parser.add_argument("--tilt-id", type=int, default=DEFAULT_TILT_ID)
    parser.add_argument("--model", default="sts3215")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pan-min", type=float, default=-90.0)
    parser.add_argument("--pan-max", type=float, default=90.0)
    parser.add_argument("--tilt-min", type=float, default=-45.0)
    parser.add_argument("--tilt-max", type=float, default=45.0)
    parser.add_argument("--center-raw", type=int, default=DEFAULT_CENTER_RAW)
    parser.add_argument("--raw-min", type=int, default=0)
    parser.add_argument("--raw-max", type=int, default=4095)
    parser.add_argument("--disable-torque", action="store_true", help="Disable torque when command exits.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan a port for Feetech servo IDs.")
    scan.add_argument("--port", default=DEFAULT_PORT)
    scan.set_defaults(func=cmd_scan)

    status = subparsers.add_parser("status", help="Read current pan/tilt angles.")
    add_common_args(status)
    status.set_defaults(func=cmd_status)

    set_cmd = subparsers.add_parser("set", help="Set one or both head servo angles in degrees.")
    add_common_args(set_cmd)
    set_cmd.add_argument("--pan", type=float, default=None)
    set_cmd.add_argument("--tilt", type=float, default=None)
    set_cmd.add_argument("--wait-s", type=float, default=0.5)
    set_cmd.add_argument("--enable-torque", action=argparse.BooleanOptionalAction, default=True)
    set_cmd.set_defaults(func=cmd_set)

    jog = subparsers.add_parser("jog", help="Interactively jog and hold the head-camera servos.")
    add_common_args(jog)
    jog.add_argument("--step-deg", type=float, default=2.0)
    jog.add_argument("--settle-s", type=float, default=0.05)
    jog.set_defaults(func=cmd_jog)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
