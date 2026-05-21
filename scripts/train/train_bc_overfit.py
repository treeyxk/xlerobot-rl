"""Minimal BC overfit trainer for local LeRobot grasp datasets.

The goal is not deployment quality. This script checks that images, state, and
actions line up by training a small image+state policy to memorize a dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_DATASET_ROOT = Path("data/real/lerobot/m4_target_grasp_v0_bc_red_25ep_merged_20260520")
DEFAULT_OUTPUT_DIR = Path("outputs/bc_overfit/red_25ep_v0")


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def from_tensor(cls, value: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        std = value.std(dim=0).clamp_min(eps)
        return cls(mean=value.mean(dim=0).tolist(), std=std.tolist())

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=value.dtype, device=value.device)
        std = torch.tensor(self.std, dtype=value.dtype, device=value.device)
        return (value - mean) / std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=value.dtype, device=value.device)
        std = torch.tensor(self.std, dtype=value.dtype, device=value.device)
        return value * std + mean


class LeRobotBCDataset(Dataset):
    def __init__(self, root: Path, image_key: str) -> None:
        self.root = root
        self.dataset = LeRobotDataset(f"local/{root.name}", root=root)
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        return {
            "image": item[self.image_key],
            "state": item["observation.state"].float(),
            "action": item["action"].float(),
        }


class TinyBCPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, image_size: tuple[int, int]) -> None:
        super().__init__()
        height, width = image_size
        self.image_size = image_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(96 * 4 * 4 + state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )
        self.register_buffer("_dummy_image_shape", torch.tensor([height, width]), persistent=False)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        visual = self.cnn(image)
        return self.mlp(torch.cat([visual, state], dim=-1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def load_feature_dims(root: Path, image_key: str) -> tuple[int, int]:
    info = json.loads((root / "meta" / "info.json").read_text())
    if image_key not in info["features"]:
        raise KeyError(f"{image_key} not found in dataset features: {sorted(info['features'])}")
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    return state_dim, action_dim


def load_normalizers(root: Path) -> tuple[Normalizer, Normalizer]:
    data = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    states = torch.from_numpy(np.stack(data["observation.state"].to_numpy()).astype("float32"))
    actions = torch.from_numpy(np.stack(data["action"].to_numpy()).astype("float32"))
    return Normalizer.from_tensor(states), Normalizer.from_tensor(actions)


def write_config(args: argparse.Namespace, state_norm: Normalizer, action_norm: Normalizer) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_root": str(args.dataset_root),
        "image_key": args.image_key,
        "image_size": [args.image_height, args.image_width],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_frames": args.max_frames,
        "state_normalizer": asdict(state_norm),
        "action_normalizer": asdict(action_norm),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    state_norm: Normalizer,
    action_norm: Normalizer,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_normalizer": asdict(state_norm),
            "action_normalizer": asdict(action_norm),
            "image_size": [args.image_height, args.image_width],
            "dataset_root": str(args.dataset_root),
            "image_key": args.image_key,
        },
        path,
    )


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_dim, action_dim = load_feature_dims(args.dataset_root, args.image_key)
    state_norm, action_norm = load_normalizers(args.dataset_root)
    write_config(args, state_norm, action_norm)

    dataset: Dataset = LeRobotBCDataset(args.dataset_root, args.image_key)
    if args.max_frames is not None:
        dataset = Subset(dataset, range(min(args.max_frames, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    device = torch.device(args.device)
    model = TinyBCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        image_size=(args.image_height, args.image_width),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "loss_norm_mse", "action_mae_deg", "action_rmse_deg"],
        )
        writer.writeheader()

        initial_loss: float | None = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_abs = 0.0
            total_sq = 0.0
            total_values = 0

            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)

                state_n = state_norm.normalize(state)
                action_n = action_norm.normalize(action)
                pred_n = model(image, state_n)
                loss = F.mse_loss(pred_n, action_n)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = action_norm.denormalize(pred_n)
                    error = pred - action
                    total_abs += error.abs().sum().item()
                    total_sq += error.square().sum().item()
                    total_values += error.numel()
                    total_loss += loss.item() * action.shape[0]

            epoch_loss = total_loss / len(dataset)
            if initial_loss is None:
                initial_loss = epoch_loss
            mae = total_abs / total_values
            rmse = (total_sq / total_values) ** 0.5
            writer.writerow(
                {
                    "epoch": epoch,
                    "loss_norm_mse": f"{epoch_loss:.8f}",
                    "action_mae_deg": f"{mae:.6f}",
                    "action_rmse_deg": f"{rmse:.6f}",
                }
            )
            f.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"loss={epoch_loss:.6f} "
                f"mae={mae:.3f}deg rmse={rmse:.3f}deg"
            )

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    epoch,
                    state_norm,
                    action_norm,
                    args,
                )

    save_checkpoint(
        args.output_dir / "checkpoint_last.pt",
        model,
        optimizer,
        args.epochs,
        state_norm,
        action_norm,
        args,
    )
    print(f"\nWrote metrics: {metrics_path}")
    print(f"Wrote checkpoint: {args.output_dir / 'checkpoint_last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
