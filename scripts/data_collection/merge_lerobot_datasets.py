"""Merge local LeRobot datasets by re-encoding saved episodes.

This script is intentionally conservative: it loads each source dataset through
LeRobotDataset, decodes only frames referenced by saved episodes, and writes a
new dataset. That fixes source videos that contain discarded trial frames.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-roots",
        type=Path,
        nargs="+",
        required=True,
        help="Source local LeRobot dataset roots in merge order.",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output raw LeRobot root. Defaults to data/real/lerobot/<dataset-name>.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Output report root. Defaults to data/bc/<dataset-name>.",
    )
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument(
        "--include-episodes",
        default=None,
        help=(
            "Optional comma-separated source episode indices to keep from each source, "
            "for example '0,1,3,4'. By default all episodes are kept."
        ),
    )
    return parser


def repo_id_from_root(root: Path) -> str:
    return f"local/{root.name}"


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def video_key_from_features(features: dict) -> str:
    keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one video feature, got {keys}")
    return keys[0]


def image_to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"expected 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def scalar_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def parse_episode_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.add(int(part))
    return indices


def create_output_dataset(
    first_source: LeRobotDataset,
    output_root: Path,
    dataset_name: str,
    vcodec: str,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=f"local/{dataset_name}",
        fps=first_source.fps,
        root=output_root,
        robot_type=first_source.meta.robot_type,
        features=first_source.features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=vcodec,
    )


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
    include_episodes = parse_episode_indices(args.include_episodes)

    output_root = args.output_root or Path("data/real/lerobot") / args.dataset_name
    report_root = args.report_root or Path("data/bc") / args.dataset_name
    if output_root.exists():
        print(f"ERROR: output root already exists: {output_root}")
        return 1
    report_root.mkdir(parents=True, exist_ok=True)

    source_roots = [root.resolve() for root in args.source_roots]
    for root in source_roots:
        if not (root / "meta" / "info.json").exists():
            print(f"ERROR: missing LeRobot dataset info: {root}")
            return 1

    first_source = LeRobotDataset(repo_id_from_root(source_roots[0]), root=source_roots[0])
    video_key = video_key_from_features(first_source.features)
    merged = create_output_dataset(first_source, output_root, args.dataset_name, args.vcodec)

    report: dict[str, object] = {
        "dataset_name": args.dataset_name,
        "created_at": now_utc(),
        "output_root": str(output_root),
        "sources": [],
        "video_key": video_key,
        "fps": first_source.fps,
    }

    merged_episode_count = 0
    merged_frame_count = 0
    try:
        for root in source_roots:
            source = LeRobotDataset(repo_id_from_root(root), root=root)
            if source.features != first_source.features:
                raise ValueError(f"features mismatch in {root}")
            if source.fps != first_source.fps:
                raise ValueError(f"fps mismatch in {root}: {source.fps} != {first_source.fps}")

            info = load_info(root)
            episodes = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            if include_episodes is not None:
                episodes = episodes.iloc[
                    [idx for idx in sorted(include_episodes) if 0 <= idx < len(episodes)]
                ]
            source_report = {
                "root": str(root),
                "source_episodes": int(info["total_episodes"]),
                "source_frames": int(info["total_frames"]),
                "included_source_episode_indices": (
                    [int(idx) for idx in episodes.index.tolist()] if include_episodes is not None else "all"
                ),
                "merged_episodes": int(len(episodes)),
                "merged_frames": int(episodes["length"].sum()),
            }
            print(
                f"Merging {root.name}: "
                f"{source_report['merged_episodes']} episodes, {source_report['merged_frames']} frames"
            )

            for _, episode in episodes.iterrows():
                start = int(episode["dataset_from_index"])
                end = int(episode["dataset_to_index"])
                for frame_index in range(start, end):
                    item = source[frame_index]
                    frame = {
                        "observation.state": item["observation.state"],
                        "action": item["action"],
                        video_key: image_to_hwc_uint8(item[video_key]),
                        "task": item["task"],
                    }
                    merged.add_frame(frame)
                merged.save_episode()
                merged_episode_count += 1
                merged_frame_count += end - start
                print(
                    f"  saved merged episode {merged_episode_count - 1} "
                    f"({end - start} frames)"
                )

            report["sources"].append(source_report)

        report["merged_episodes"] = merged_episode_count
        report["merged_frames"] = merged_frame_count
        (report_root / "merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        merged.finalize()

    print(f"\nMerged dataset: {output_root}")
    print(f"Report: {report_root / 'merge_report.json'}")
    print(f"Episodes: {merged_episode_count}, frames: {merged_frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
