"""Evaluate a minimal BC checkpoint on a held-out local LeRobot dataset."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE_PATH = REPO_ROOT / "scripts" / "train" / "train_bc_overfit.py"
spec = importlib.util.spec_from_file_location("xlerobot_train_bc_overfit", TRAIN_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load training module from {TRAIN_MODULE_PATH}")
train_bc_overfit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_bc_overfit
spec.loader.exec_module(train_bc_overfit)

LeRobotBCDataset = train_bc_overfit.LeRobotBCDataset
Normalizer = train_bc_overfit.Normalizer
TinyBCPolicy = train_bc_overfit.TinyBCPolicy
load_feature_dims = train_bc_overfit.load_feature_dims


MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class EvalTotals:
    total_abs: torch.Tensor
    total_sq: torch.Tensor
    total_count: int = 0

    @classmethod
    def create(cls, action_dim: int) -> "EvalTotals":
        return cls(total_abs=torch.zeros(action_dim), total_sq=torch.zeros(action_dim))

    def update(self, error: torch.Tensor) -> None:
        error = error.detach().cpu()
        self.total_abs += error.abs().sum(dim=0)
        self.total_sq += error.square().sum(dim=0)
        self.total_count += int(error.shape[0])

    @property
    def mae_per_joint(self) -> torch.Tensor:
        return self.total_abs / max(self.total_count, 1)

    @property
    def rmse_per_joint(self) -> torch.Tensor:
        return torch.sqrt(self.total_sq / max(self.total_count, 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/bc_overfit/red_62ep_final_v0/checkpoint_last.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/real/lerobot/m4_target_grasp_v0_bc_red_test_10ep_merged_20260521"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bc_eval/red_62ep_on_test_10ep_20260521"),
    )
    return parser


def normalizer_from_checkpoint(value: dict[str, list[float]]) -> Normalizer:
    return Normalizer(mean=value["mean"], std=value["std"])


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def write_episode_metrics(
    output_path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_index", "frames", "action_mae_deg", "action_rmse_deg"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    image_key = ckpt.get("image_key", "observation.images.front")
    image_size = tuple(int(v) for v in ckpt.get("image_size", [120, 160]))
    state_norm = normalizer_from_checkpoint(ckpt["state_normalizer"])
    action_norm = normalizer_from_checkpoint(ckpt["action_normalizer"])

    state_dim, action_dim = load_feature_dims(args.dataset_root, image_key)
    model = TinyBCPolicy(state_dim=state_dim, action_dim=action_dim, image_size=image_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = LeRobotBCDataset(args.dataset_root, image_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    totals = EvalTotals.create(action_dim)
    norm_loss_sum = 0.0
    frame_count = 0

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            state_n = state_norm.normalize(state)
            action_n = action_norm.normalize(action)
            pred_n = model(image, state_n)
            pred = action_norm.denormalize(pred_n)
            error = pred - action

            totals.update(error)
            norm_loss_sum += F.mse_loss(pred_n, action_n, reduction="sum").item()
            frame_count += int(action.shape[0])
            predictions.append(pred.detach().cpu().numpy())
            targets.append(action.detach().cpu().numpy())

    pred_all = np.concatenate(predictions, axis=0)
    target_all = np.concatenate(targets, axis=0)
    error_all = pred_all - target_all

    data = pd.read_parquet(args.dataset_root / "data" / "chunk-000" / "file-000.parquet")
    episode_rows = []
    for episode_index, group in data.groupby("episode_index", sort=True):
        indices = group["index"].to_numpy(dtype=np.int64)
        err = error_all[indices]
        episode_rows.append(
            {
                "episode_index": int(episode_index),
                "frames": int(len(group)),
                "action_mae_deg": float(np.abs(err).mean()),
                "action_rmse_deg": float(np.sqrt(np.square(err).mean())),
            }
        )

    mae_per_joint = totals.mae_per_joint.numpy()
    rmse_per_joint = totals.rmse_per_joint.numpy()
    summary = {
        "checkpoint": str(args.checkpoint),
        "train_dataset_root": ckpt.get("dataset_root"),
        "eval_dataset_root": str(args.dataset_root),
        "frames": frame_count,
        "episodes": int(data["episode_index"].nunique()),
        "loss_norm_mse": norm_loss_sum / max(frame_count * action_dim, 1),
        "action_mae_deg": float(np.abs(error_all).mean()),
        "action_rmse_deg": float(np.sqrt(np.square(error_all).mean())),
        "per_joint": {
            MOTOR_ORDER[idx] if idx < len(MOTOR_ORDER) else f"joint_{idx}": {
                "mae_deg": float(mae_per_joint[idx]),
                "rmse_deg": float(rmse_per_joint[idx]),
            }
            for idx in range(action_dim)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_episode_metrics(args.output_dir / "episode_metrics.csv", episode_rows)

    print(f"checkpoint: {args.checkpoint}")
    print(f"eval dataset: {args.dataset_root}")
    print(f"episodes={summary['episodes']} frames={summary['frames']}")
    print(
        f"loss={summary['loss_norm_mse']:.6f} "
        f"mae={summary['action_mae_deg']:.3f}deg "
        f"rmse={summary['action_rmse_deg']:.3f}deg"
    )
    print("\nper-joint:")
    for name, values in summary["per_joint"].items():
        print(f"  {name:13s} mae={values['mae_deg']:.3f}deg rmse={values['rmse_deg']:.3f}deg")
    print(f"\nWrote summary: {args.output_dir / 'summary.json'}")
    print(f"Wrote episode metrics: {args.output_dir / 'episode_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
