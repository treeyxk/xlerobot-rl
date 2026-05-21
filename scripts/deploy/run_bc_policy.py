"""Run a minimal BC checkpoint on the right follower arm.

The default mode is dry-run: it reads camera/state, predicts actions, and logs
them without commanding motors. Use --execute only after dry-run looks sane.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE_PATH = REPO_ROOT / "scripts" / "train" / "train_bc_overfit.py"
spec = importlib.util.spec_from_file_location("xlerobot_train_bc_overfit", TRAIN_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load training module from {TRAIN_MODULE_PATH}")
train_bc_overfit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_bc_overfit
spec.loader.exec_module(train_bc_overfit)

Normalizer = train_bc_overfit.Normalizer
TinyBCPolicy = train_bc_overfit.TinyBCPolicy
load_feature_dims = train_bc_overfit.load_feature_dims


DEFAULT_FOLLOWER_PORT = "/dev/xlerobot_right_follower"
DEFAULT_CAMERA_INDEX = "/dev/xlerobot_head_camera"
DEFAULT_CHECKPOINT = Path("outputs/bc_overfit/red_62ep_final_v0/checkpoint_last.pt")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--reference-dataset-root",
        type=Path,
        default=Path("data/real/lerobot/m4_target_grasp_v0_bc_red_62ep_final_20260521"),
        help="Dataset root used only to recover state/action feature dimensions.",
    )
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default="right_follower")
    parser.add_argument("--camera-index", default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--action-scale", type=float, default=0.3)
    parser.add_argument("--max-delta-deg", type=float, default=2.0)
    parser.add_argument("--max-relative-target", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true", help="Actually command the follower arm.")
    parser.add_argument("--comm-attempts", type=int, default=10)
    parser.add_argument("--comm-retry-sleep-s", type=float, default=0.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="CSV log path. Defaults to outputs/bc_rollout/<timestamp>.csv.",
    )
    return parser


def comm_retry(
    label: str,
    fn,
    *args,
    attempts: int,
    sleep_s: float,
    **kwargs,
):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            last_exc = exc
            print(f"WARNING: {label} failed ({attempt}/{attempts}): {exc}")
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


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


def pose_from_observation(obs: dict[str, object]) -> dict[str, float]:
    pose = {}
    for name in MOTOR_ORDER:
        key = f"{name}.pos"
        if key not in obs:
            raise KeyError(f"missing motor position in observation: {key}")
        pose[name] = float(obs[key])
    return pose


def action_from_values(values: np.ndarray) -> dict[str, float]:
    return {f"{name}.pos": float(values[idx]) for idx, name in enumerate(MOTOR_ORDER)}


def find_image(obs: dict[str, object], image_key: str) -> np.ndarray | torch.Tensor:
    candidates = [
        image_key,
        image_key.removeprefix("observation.images."),
        "front",
    ]
    for key in candidates:
        value = obs.get(key)
        if hasattr(value, "shape") and len(value.shape) == 3:
            return value
    for key, value in obs.items():
        if hasattr(value, "shape") and len(value.shape) == 3:
            return value
    raise KeyError(f"no image-like observation found; keys={sorted(obs)}")


def image_to_tensor(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach()
    else:
        tensor = torch.from_numpy(np.asarray(image))
    if tensor.ndim != 3:
        raise ValueError(f"expected 3D image, got {tuple(tensor.shape)}")
    if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float()
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.unsqueeze(0).to(device)


def normalizer_from_checkpoint(value: dict[str, list[float]]) -> Normalizer:
    return Normalizer(mean=value["mean"], std=value["std"])


def make_follower(args: argparse.Namespace) -> SO101Follower:
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
    return SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            max_relative_target=args.max_relative_target,
            cameras=cameras,
        )
    )


def load_policy(args: argparse.Namespace, device: torch.device) -> tuple[TinyBCPolicy, Normalizer, Normalizer, str]:
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_key = ckpt.get("image_key", "observation.images.front")
    image_size = tuple(int(v) for v in ckpt.get("image_size", [120, 160]))
    state_norm = normalizer_from_checkpoint(ckpt["state_normalizer"])
    action_norm = normalizer_from_checkpoint(ckpt["action_normalizer"])
    state_dim, action_dim = load_feature_dims(args.reference_dataset_root, image_key)
    model = TinyBCPolicy(state_dim=state_dim, action_dim=action_dim, image_size=image_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, state_norm, action_norm, image_key


def clipped_command(current: np.ndarray, pred: np.ndarray, action_scale: float, max_delta_deg: float) -> np.ndarray:
    desired = current + action_scale * (pred - current)
    delta = np.clip(desired - current, -max_delta_deg, max_delta_deg)
    return current + delta


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    fieldnames = ["step", "time_s", "executed"]
    for prefix in ["state", "pred", "cmd"]:
        fieldnames.extend(f"{prefix}_{name}" for name in MOTOR_ORDER)
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def main() -> int:
    args = build_parser().parse_args()
    if args.log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "execute" if args.execute else "dryrun"
        args.log_path = Path("outputs/bc_rollout") / f"{mode}_{stamp}.csv"

    device = torch.device(args.device)
    model, state_norm, action_norm, image_key = load_policy(args, device)
    follower = make_follower(args)
    log_file = None

    print("BC policy runner")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"  duration: {args.duration_s:.1f}s, fps={args.fps:.1f}")
    print(f"  action_scale={args.action_scale:.2f}, max_delta_deg={args.max_delta_deg:.2f}")
    if args.execute:
        print("  Press q during rollout to stop early. Keep emergency stop reachable.")
    else:
        print("  Dry-run will not command motors. Use --execute to send actions.")

    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old_settings = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)
    try:
        follower.connect()
        log_file, writer = open_log(args.log_path)
        period = 1.0 / args.fps
        start_t = time.perf_counter()
        step = 0
        with torch.no_grad():
            while True:
                loop_start = time.perf_counter()
                elapsed = loop_start - start_t
                if elapsed >= args.duration_s:
                    break
                if poll_key(0.0) == "q":
                    print("q")
                    break

                obs = comm_retry(
                    "follower get_observation",
                    follower.get_observation,
                    attempts=args.comm_attempts,
                    sleep_s=args.comm_retry_sleep_s,
                )
                pose = pose_from_observation(obs)
                state = torch.tensor(
                    [[pose[name] for name in MOTOR_ORDER]],
                    dtype=torch.float32,
                    device=device,
                )
                image = image_to_tensor(find_image(obs, image_key), device)
                pred_n = model(image, state_norm.normalize(state))
                pred = action_norm.denormalize(pred_n).squeeze(0).detach().cpu().numpy()
                current = state.squeeze(0).detach().cpu().numpy()
                cmd = clipped_command(current, pred, args.action_scale, args.max_delta_deg)

                if args.execute:
                    comm_retry(
                        "follower send_action",
                        follower.send_action,
                        action_from_values(cmd),
                        attempts=args.comm_attempts,
                        sleep_s=args.comm_retry_sleep_s,
                    )

                row = {"step": step, "time_s": f"{elapsed:.4f}", "executed": int(args.execute)}
                for prefix, values in [("state", current), ("pred", pred), ("cmd", cmd)]:
                    row.update({f"{prefix}_{name}": f"{float(values[idx]):.6f}" for idx, name in enumerate(MOTOR_ORDER)})
                writer.writerow(row)
                if step % max(1, int(args.fps)) == 0:
                    max_delta = float(np.max(np.abs(cmd - current)))
                    print(
                        f"step={step:04d} t={elapsed:5.2f}s "
                        f"max_cmd_delta={max_delta:.2f}deg "
                        f"gripper pred/cmd={pred[-1]:+.2f}/{cmd[-1]:+.2f}"
                    )
                step += 1
                precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if log_file is not None:
            log_file.close()
        if follower.is_connected:
            follower.disconnect()

    print(f"\nWrote rollout log: {args.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
