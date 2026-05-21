"""Train a minimal terminal-window reward classifier.

This is a first-pass binary classifier for real red-cube episodes. It samples a
small window from the end of each episode and predicts success vs failure from
RGB plus joint state. Episode-level validation averages frame probabilities.
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
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST = Path("configs/reward/reward_dataset_v0.json")
DEFAULT_OUTPUT_DIR = Path("outputs/reward_classifier/red_cube_v0")


@dataclass
class Normalizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def from_tensor(cls, value: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        return cls(
            mean=value.mean(dim=0).tolist(),
            std=value.std(dim=0).clamp_min(eps).tolist(),
        )

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=value.dtype, device=value.device)
        std = torch.tensor(self.std, dtype=value.dtype, device=value.device)
        return (value - mean) / std


@dataclass(frozen=True)
class EpisodeRef:
    source_index: int
    category: str
    label: int
    episode_index: int
    start: int
    end: int


@dataclass(frozen=True)
class FrameSample:
    episode_ref_index: int
    frame_index: int


class TinyRewardClassifier(nn.Module):
    def __init__(self, state_dim: int, image_size: tuple[int, int]) -> None:
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(96 * 4 * 4 + state_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 1),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        visual = self.cnn(image)
        return self.head(torch.cat([visual, state], dim=-1)).squeeze(-1)


class RewardFrameDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        image_key: str,
        episode_refs: list[EpisodeRef],
        samples: list[FrameSample],
    ) -> None:
        self.roots = roots
        self.image_key = image_key
        self.episode_refs = episode_refs
        self.samples = samples
        self.datasets = [LeRobotDataset(f"local/{root.name}", root=root) for root in roots]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        ep = self.episode_refs[sample.episode_ref_index]
        item = self.datasets[ep.source_index][sample.frame_index]
        return {
            "image": item[self.image_key],
            "state": item["observation.state"].float(),
            "label": torch.tensor(float(ep.label), dtype=torch.float32),
            "episode_ref_index": torch.tensor(sample.episode_ref_index, dtype=torch.long),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--terminal-frames", type=int, default=12)
    parser.add_argument(
        "--window-end-offset-frames",
        type=int,
        default=0,
        help="Skip this many final frames before taking the terminal window.",
    )
    parser.add_argument("--val-episodes-per-category", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def feature_state_dim(root: Path) -> int:
    info = load_info(root)
    return int(info["features"]["observation.state"]["shape"][0])


def build_episode_refs(manifest: dict) -> tuple[list[Path], list[EpisodeRef]]:
    roots: list[Path] = []
    refs: list[EpisodeRef] = []
    for source_index, entry in enumerate(manifest["datasets"]):
        root = Path(entry["root"])
        roots.append(root)
        episodes = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        for episode_index, row in episodes.iterrows():
            refs.append(
                EpisodeRef(
                    source_index=source_index,
                    category=entry["category"],
                    label=int(entry["label"]),
                    episode_index=int(episode_index),
                    start=int(row["dataset_from_index"]),
                    end=int(row["dataset_to_index"]),
                )
            )
    return roots, refs


def split_episode_refs(
    episode_refs: list[EpisodeRef],
    val_episodes_per_category: int,
) -> tuple[list[int], list[int]]:
    by_category: dict[str, list[int]] = {}
    for idx, ref in enumerate(episode_refs):
        by_category.setdefault(ref.category, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for _, indices in sorted(by_category.items()):
        indices = sorted(indices, key=lambda idx: episode_refs[idx].episode_index)
        n_val = min(val_episodes_per_category, max(len(indices) - 1, 1))
        val_set = set(indices[-n_val:])
        for idx in indices:
            if idx in val_set:
                val_indices.append(idx)
            else:
                train_indices.append(idx)
    return train_indices, val_indices


def terminal_frame_indices(start: int, end: int, count: int, end_offset: int = 0) -> list[int]:
    window_end = max(start, end - max(end_offset, 0))
    first = max(start, window_end - count)
    return list(range(first, window_end))


def build_samples(
    episode_indices: list[int],
    refs: list[EpisodeRef],
    terminal_frames: int,
    window_end_offset_frames: int,
) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for ref_index in episode_indices:
        ref = refs[ref_index]
        for frame_index in terminal_frame_indices(
            ref.start,
            ref.end,
            terminal_frames,
            window_end_offset_frames,
        ):
            samples.append(FrameSample(episode_ref_index=ref_index, frame_index=frame_index))
    return samples


def load_state_normalizer(
    roots: list[Path],
    train_refs: list[EpisodeRef],
    train_indices: list[int],
    terminal_frames: int,
    window_end_offset_frames: int,
) -> Normalizer:
    states: list[np.ndarray] = []
    data_cache: dict[int, pd.DataFrame] = {}
    for ref_index in train_indices:
        ref = train_refs[ref_index]
        if ref.source_index not in data_cache:
            data_cache[ref.source_index] = pd.read_parquet(
                roots[ref.source_index] / "data" / "chunk-000" / "file-000.parquet"
            )
        data = data_cache[ref.source_index]
        window = data.iloc[
            terminal_frame_indices(
                ref.start,
                ref.end,
                terminal_frames,
                window_end_offset_frames,
            )
        ]
        states.append(np.stack(window["observation.state"].to_numpy()).astype("float32"))
    return Normalizer.from_tensor(torch.from_numpy(np.concatenate(states, axis=0)))


def save_config(args: argparse.Namespace, manifest: dict, state_norm: Normalizer, train_eps: int, val_eps: int) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "manifest": str(args.manifest),
        "manifest_name": manifest.get("name"),
        "image_key": manifest.get("image_key", "observation.images.front"),
        "image_size": [args.image_height, args.image_width],
        "terminal_frames": args.terminal_frames,
        "window_end_offset_frames": args.window_end_offset_frames,
        "val_episodes_per_category": args.val_episodes_per_category,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "state_normalizer": asdict(state_norm),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    state_norm: Normalizer,
    args: argparse.Namespace,
    manifest: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_normalizer": asdict(state_norm),
            "image_size": [args.image_height, args.image_width],
            "manifest": manifest,
            "manifest_path": str(args.manifest),
            "terminal_frames": args.terminal_frames,
            "window_end_offset_frames": args.window_end_offset_frames,
        },
        path,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    state_norm: Normalizer,
    episode_refs: list[EpisodeRef],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    frame_correct = 0
    episode_probs: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            ep_indices = batch["episode_ref_index"].cpu().numpy().tolist()
            logits = model(image, state_norm.normalize(state))
            loss = F.binary_cross_entropy_with_logits(logits, label, reduction="sum")
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            frame_correct += int((pred == label).sum().item())
            total_loss += float(loss.item())
            total_count += int(label.numel())
            for ep_idx, value in zip(ep_indices, prob.detach().cpu().numpy().tolist()):
                episode_probs.setdefault(int(ep_idx), []).append(float(value))

    rows: list[dict[str, object]] = []
    ep_correct = 0
    category_total: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    for ep_idx, probs in sorted(episode_probs.items()):
        ref = episode_refs[ep_idx]
        mean_prob = float(np.mean(probs))
        pred = int(mean_prob >= 0.5)
        correct = int(pred == ref.label)
        ep_correct += correct
        category_total[ref.category] = category_total.get(ref.category, 0) + 1
        category_correct[ref.category] = category_correct.get(ref.category, 0) + correct
        rows.append(
            {
                "category": ref.category,
                "source_episode_index": ref.episode_index,
                "label": ref.label,
                "prob_success": mean_prob,
                "pred": pred,
                "correct": bool(correct),
            }
        )

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "frame_accuracy": frame_correct / max(total_count, 1),
        "episode_accuracy": ep_correct / max(len(episode_probs), 1),
    }
    for category, total in category_total.items():
        metrics[f"episode_accuracy/{category}"] = category_correct[category] / max(total, 1)
    return metrics, rows


def write_episode_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "source_episode_index", "label", "prob_success", "pred", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    image_key = manifest.get("image_key", "observation.images.front")
    roots, episode_refs = build_episode_refs(manifest)
    state_dim = feature_state_dim(roots[0])

    train_ep_indices, val_ep_indices = split_episode_refs(
        episode_refs,
        val_episodes_per_category=args.val_episodes_per_category,
    )
    train_samples = build_samples(
        train_ep_indices,
        episode_refs,
        args.terminal_frames,
        args.window_end_offset_frames,
    )
    val_samples = build_samples(
        val_ep_indices,
        episode_refs,
        args.terminal_frames,
        args.window_end_offset_frames,
    )
    state_norm = load_state_normalizer(
        roots,
        episode_refs,
        train_ep_indices,
        args.terminal_frames,
        args.window_end_offset_frames,
    )
    save_config(args, manifest, state_norm, len(train_ep_indices), len(val_ep_indices))

    train_dataset = RewardFrameDataset(roots, image_key, episode_refs, train_samples)
    val_dataset = RewardFrameDataset(roots, image_key, episode_refs, val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    labels = torch.tensor([episode_refs[s.episode_ref_index].label for s in train_samples], dtype=torch.float32)
    pos = labels.sum().clamp_min(1.0)
    neg = (labels.numel() - labels.sum()).clamp_min(1.0)
    pos_weight = (neg / pos).item()

    device = torch.device(args.device)
    model = TinyRewardClassifier(
        state_dim=state_dim,
        image_size=(args.image_height, args.image_width),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    print(f"manifest: {args.manifest}")
    print(f"train episodes={len(train_ep_indices)}, val episodes={len(val_ep_indices)}")
    print(
        f"train samples={len(train_samples)}, val samples={len(val_samples)}, "
        f"terminal_frames={args.terminal_frames}, "
        f"window_end_offset_frames={args.window_end_offset_frames}, "
        f"pos_weight={pos_weight:.3f}"
    )

    metrics_path = args.output_dir / "metrics.csv"
    best_val_acc = -1.0
    best_epoch = 0
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_frame_acc", "val_episode_acc"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_count = 0
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                logits = model(image, state_norm.normalize(state))
                loss = criterion(logits, label)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * int(label.numel())
                train_count += int(label.numel())

            val_metrics, val_rows = evaluate(model, val_loader, state_norm, episode_refs, device)
            train_epoch_loss = train_loss / max(train_count, 1)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_epoch_loss:.8f}",
                    "val_loss": f"{val_metrics['loss']:.8f}",
                    "val_frame_acc": f"{val_metrics['frame_accuracy']:.6f}",
                    "val_episode_acc": f"{val_metrics['episode_accuracy']:.6f}",
                }
            )
            f.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_epoch_loss:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_frame_acc={val_metrics['frame_accuracy']*100:.1f}% "
                f"val_ep_acc={val_metrics['episode_accuracy']*100:.1f}%"
            )

            if val_metrics["episode_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["episode_accuracy"]
                best_epoch = epoch
                save_checkpoint(
                    args.output_dir / "checkpoint_best.pt",
                    model,
                    optimizer,
                    epoch,
                    state_norm,
                    args,
                    manifest,
                )
                write_episode_rows(args.output_dir / "val_episode_predictions_best.csv", val_rows)

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    epoch,
                    state_norm,
                    args,
                    manifest,
                )

    final_metrics, final_rows = evaluate(model, val_loader, state_norm, episode_refs, device)
    write_episode_rows(args.output_dir / "val_episode_predictions_last.csv", final_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "train_episodes": len(train_ep_indices),
                "val_episodes": len(val_ep_indices),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "best_epoch": best_epoch,
                "best_val_episode_accuracy": best_val_acc,
                "last_val_metrics": final_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    save_checkpoint(
        args.output_dir / "checkpoint_last.pt",
        model,
        optimizer,
        args.epochs,
        state_norm,
        args,
        manifest,
    )
    print(f"\nWrote metrics: {metrics_path}")
    print(f"Wrote best checkpoint: {args.output_dir / 'checkpoint_best.pt'}")
    print(f"Wrote last checkpoint: {args.output_dir / 'checkpoint_last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
