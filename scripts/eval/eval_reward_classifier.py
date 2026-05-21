"""Evaluate a reward classifier checkpoint on a manifest-defined holdout set."""
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE_PATH = REPO_ROOT / "scripts" / "train" / "train_reward_classifier.py"
spec = importlib.util.spec_from_file_location("xlerobot_train_reward_classifier", TRAIN_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load training module from {TRAIN_MODULE_PATH}")
train_reward_classifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_reward_classifier
spec.loader.exec_module(train_reward_classifier)

Normalizer = train_reward_classifier.Normalizer
TinyRewardClassifier = train_reward_classifier.TinyRewardClassifier


DEFAULT_CHECKPOINT = Path("outputs/reward_classifier/red_cube_v0/checkpoint_best.pt")
DEFAULT_MANIFEST = Path("configs/reward/reward_holdout_v0.json")
DEFAULT_OUTPUT_DIR = Path("outputs/reward_classifier/red_cube_v0_holdout_eval")


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


class RewardEvalDataset(Dataset):
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--terminal-frames", type=int, default=None)
    parser.add_argument(
        "--window-end-offset-frames",
        type=int,
        default=None,
        help="Skip this many final frames before taking the evaluation window.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def normalizer_from_checkpoint(value: dict[str, list[float]]) -> Normalizer:
    return Normalizer(mean=value["mean"], std=value["std"])


def load_state_dim(root: Path) -> int:
    info = json.loads((root / "meta" / "info.json").read_text())
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


def terminal_frame_indices(start: int, end: int, count: int, end_offset: int = 0) -> list[int]:
    window_end = max(start, end - max(end_offset, 0))
    first = max(start, window_end - count)
    return list(range(first, window_end))


def build_samples(
    episode_refs: list[EpisodeRef],
    terminal_frames: int,
    window_end_offset_frames: int,
) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for ref_index, ref in enumerate(episode_refs):
        for frame_index in terminal_frame_indices(
            ref.start,
            ref.end,
            terminal_frames,
            window_end_offset_frames,
        ):
            samples.append(FrameSample(episode_ref_index=ref_index, frame_index=frame_index))
    return samples


def write_episode_predictions(path: Path, rows: list[dict[str, object]]) -> None:
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
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    manifest = load_manifest(args.manifest)

    image_key = manifest.get("image_key", ckpt["manifest"].get("image_key", "observation.images.front"))
    image_size = tuple(int(v) for v in ckpt.get("image_size", [120, 160]))
    terminal_frames = int(args.terminal_frames or ckpt.get("terminal_frames", 12))
    window_end_offset_frames = int(
        args.window_end_offset_frames
        if args.window_end_offset_frames is not None
        else ckpt.get("window_end_offset_frames", 0)
    )
    state_norm = normalizer_from_checkpoint(ckpt["state_normalizer"])

    roots, episode_refs = build_episode_refs(manifest)
    samples = build_samples(episode_refs, terminal_frames, window_end_offset_frames)
    dataset = RewardEvalDataset(roots, image_key, episode_refs, samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    model = TinyRewardClassifier(
        state_dim=load_state_dim(roots[0]),
        image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frame_correct = 0
    frame_count = 0
    episode_probs: dict[int, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            ep_indices = batch["episode_ref_index"].cpu().numpy().tolist()

            logits = model(image, state_norm.normalize(state))
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            frame_correct += int((pred == label).sum().item())
            frame_count += int(label.numel())
            for ep_idx, value in zip(ep_indices, prob.detach().cpu().numpy().tolist()):
                episode_probs.setdefault(int(ep_idx), []).append(float(value))

    rows: list[dict[str, object]] = []
    category_total: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    episode_correct = 0
    for ep_idx, probs in sorted(episode_probs.items()):
        ref = episode_refs[ep_idx]
        mean_prob = float(np.mean(probs))
        pred = int(mean_prob >= 0.5)
        correct = int(pred == ref.label)
        episode_correct += correct
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

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "episodes": len(episode_refs),
        "frames": frame_count,
        "terminal_frames": terminal_frames,
        "window_end_offset_frames": window_end_offset_frames,
        "frame_accuracy": frame_correct / max(frame_count, 1),
        "episode_accuracy": episode_correct / max(len(episode_refs), 1),
        "per_category": {
            category: {
                "episodes": category_total[category],
                "correct": category_correct.get(category, 0),
                "episode_accuracy": category_correct.get(category, 0) / max(category_total[category], 1),
            }
            for category in sorted(category_total)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_episode_predictions(args.output_dir / "episode_predictions.csv", rows)

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest: {args.manifest}")
    print(
        f"episodes={summary['episodes']} frames={summary['frames']} "
        f"terminal_frames={terminal_frames} "
        f"window_end_offset_frames={window_end_offset_frames}"
    )
    print(
        f"frame_acc={summary['frame_accuracy']*100:.1f}% "
        f"episode_acc={summary['episode_accuracy']*100:.1f}%"
    )
    print("\nper-category:")
    for category, values in summary["per_category"].items():
        print(
            f"  {category:18s} "
            f"{values['correct']}/{values['episodes']} "
            f"({values['episode_accuracy']*100:.1f}%)"
        )
    print(f"\nWrote summary: {args.output_dir / 'summary.json'}")
    print(f"Wrote predictions: {args.output_dir / 'episode_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
