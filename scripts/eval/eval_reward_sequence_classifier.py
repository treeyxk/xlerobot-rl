"""Evaluate a reward sequence classifier checkpoint on a manifest."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULE_PATH = REPO_ROOT / "scripts" / "train" / "train_reward_sequence_classifier.py"
spec = importlib.util.spec_from_file_location("xlerobot_train_reward_sequence_classifier", TRAIN_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load training module from {TRAIN_MODULE_PATH}")
train_seq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_seq
spec.loader.exec_module(train_seq)

Normalizer = train_seq.Normalizer
RewardSequenceDataset = train_seq.RewardSequenceDataset
TinyRewardSequenceClassifier = train_seq.TinyRewardSequenceClassifier
build_episode_refs = train_seq.build_episode_refs
feature_state_dim = train_seq.feature_state_dim
load_manifest = train_seq.load_manifest


DEFAULT_CHECKPOINT = Path("outputs/reward_sequence_classifier/red_cube_v0/checkpoint_best.pt")
DEFAULT_MANIFEST = Path("configs/reward/reward_holdout_v0.json")
DEFAULT_OUTPUT_DIR = Path("outputs/reward_sequence_classifier/red_cube_v0_holdout_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-span-frames", type=int, default=None)
    parser.add_argument("--window-end-offset-frames", type=int, default=None)
    parser.add_argument("--sequence-scope", choices=("tail", "full"), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def normalizer_from_checkpoint(value: dict[str, list[float]]) -> Normalizer:
    return Normalizer(mean=value["mean"], std=value["std"])


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
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    manifest = load_manifest(args.manifest)
    image_key = manifest.get("image_key", ckpt["manifest"].get("image_key", "observation.images.front"))
    image_size = tuple(int(v) for v in ckpt.get("image_size", [120, 160]))
    sequence_length = int(args.sequence_length or ckpt.get("sequence_length", 16))
    sequence_span_frames = int(args.sequence_span_frames or ckpt.get("sequence_span_frames", 120))
    sequence_scope = args.sequence_scope or ckpt.get("sequence_scope", "tail")
    window_end_offset_frames = int(
        args.window_end_offset_frames
        if args.window_end_offset_frames is not None
        else ckpt.get("window_end_offset_frames", 0)
    )
    state_norm = normalizer_from_checkpoint(ckpt["state_normalizer"])

    roots, episode_refs = build_episode_refs(manifest)
    all_indices = list(range(len(episode_refs)))
    dataset = RewardSequenceDataset(
        roots,
        image_key,
        episode_refs,
        all_indices,
        sequence_length,
        sequence_span_frames,
        window_end_offset_frames,
        sequence_scope,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    model = TinyRewardSequenceClassifier(
        state_dim=feature_state_dim(roots[0]),
        image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    correct_count = 0
    category_total: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            ep_indices = batch["episode_ref_index"].cpu().numpy().tolist()
            logits = model(image, state_norm.normalize(state))
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).int()
            for ep_idx, prob_value, pred_value in zip(
                ep_indices,
                prob.detach().cpu().numpy().tolist(),
                pred.detach().cpu().numpy().tolist(),
            ):
                ref = episode_refs[int(ep_idx)]
                correct = int(pred_value) == ref.label
                correct_count += int(correct)
                category_total[ref.category] = category_total.get(ref.category, 0) + 1
                category_correct[ref.category] = category_correct.get(ref.category, 0) + int(correct)
                rows.append(
                    {
                        "category": ref.category,
                        "source_episode_index": ref.episode_index,
                        "label": ref.label,
                        "prob_success": float(prob_value),
                        "pred": int(pred_value),
                        "correct": bool(correct),
                    }
                )

    summary = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "episodes": len(episode_refs),
        "sequence_length": sequence_length,
        "sequence_span_frames": sequence_span_frames,
        "sequence_scope": sequence_scope,
        "window_end_offset_frames": window_end_offset_frames,
        "episode_accuracy": correct_count / max(len(episode_refs), 1),
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
    write_episode_rows(args.output_dir / "episode_predictions.csv", rows)

    print(f"checkpoint: {args.checkpoint}")
    print(f"manifest: {args.manifest}")
    print(
        f"episodes={summary['episodes']} "
        f"sequence_length={sequence_length} "
        f"sequence_span_frames={sequence_span_frames} "
        f"sequence_scope={sequence_scope} "
        f"window_end_offset_frames={window_end_offset_frames}"
    )
    print(f"episode_acc={summary['episode_accuracy']*100:.1f}%")
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
