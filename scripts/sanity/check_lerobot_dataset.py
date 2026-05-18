"""Check a raw LeRobot dataset for basic recording integrity."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


@dataclass
class CheckResult:
    level: str
    message: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to a raw LeRobot dataset, e.g. data/real/lerobot/<dataset_name>.",
    )
    parser.add_argument("--expect-episodes", type=int, default=None)
    parser.add_argument("--expect-fps", type=float, default=None)
    parser.add_argument("--expect-width", type=int, default=None)
    parser.add_argument("--expect-height", type=int, default=None)
    parser.add_argument("--expect-action-dim", type=int, default=6)
    parser.add_argument("--expect-state-dim", type=int, default=6)
    parser.add_argument("--decode-video", action=argparse.BooleanOptionalAction, default=True)
    return parser


def add(results: list[CheckResult], level: str, message: str) -> None:
    results.append(CheckResult(level=level, message=message))


def load_info(root: Path, results: list[CheckResult]) -> dict[str, Any] | None:
    path = root / "meta" / "info.json"
    if not path.exists():
        add(results, "FAIL", f"missing {path}")
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        add(results, "FAIL", f"failed to parse {path}: {exc}")
        return None


def read_parquet(path: Path, results: list[CheckResult]) -> pd.DataFrame | None:
    if not path.exists():
        add(results, "FAIL", f"missing {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        add(results, "FAIL", f"failed to read {path}: {exc}")
        return None


def check_video(
    video_path: Path,
    info: dict[str, Any],
    args: argparse.Namespace,
    results: list[CheckResult],
) -> None:
    if not video_path.exists():
        add(results, "FAIL", f"missing video {video_path}")
        return
    if video_path.stat().st_size == 0:
        add(results, "FAIL", f"empty video {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        add(results, "FAIL", f"OpenCV failed to open video {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\nVideo:")
    print(f"  path: {video_path}")
    print(f"  size: {video_path.stat().st_size} bytes")
    print(f"  frames/fps/resolution: {frame_count} / {fps:.2f} / {width}x{height}")

    if frame_count <= 0:
        add(results, "FAIL", "video frame count is zero")
    if args.expect_fps is not None and abs(fps - args.expect_fps) > 0.5:
        add(results, "WARN", f"video fps {fps:.2f} differs from expected {args.expect_fps:.2f}")
    if args.expect_width is not None and width != args.expect_width:
        add(results, "FAIL", f"video width {width} != expected {args.expect_width}")
    if args.expect_height is not None and height != args.expect_height:
        add(results, "FAIL", f"video height {height} != expected {args.expect_height}")

    expected_frames = int(info.get("total_frames", 0))
    if expected_frames and frame_count != expected_frames:
        add(results, "WARN", f"video frames {frame_count} != info total_frames {expected_frames}")

    if args.decode_video:
        for idx in [0, max(frame_count // 2, 0), max(frame_count - 1, 0)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                add(results, "FAIL", f"failed to decode video frame {idx}")
                continue
            if frame.shape[:2] != (height, width):
                add(results, "FAIL", f"decoded frame {idx} shape {frame.shape} mismatches metadata")
    cap.release()


def check_dataset(args: argparse.Namespace) -> int:
    root = args.dataset_root
    results: list[CheckResult] = []

    print(f"Dataset root: {root}")
    if not root.exists():
        print("Result: FAIL")
        print(f"  missing dataset root {root}")
        return 1

    info = load_info(root, results)
    if info is None:
        print_results(results)
        return 1

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = float(info.get("fps", 0))
    features = info.get("features", {})

    print("\nInfo:")
    print(f"  total_episodes: {total_episodes}")
    print(f"  total_frames: {total_frames}")
    print(f"  fps: {fps:g}")
    print(f"  feature keys: {sorted(features)}")

    if total_episodes <= 0:
        add(results, "FAIL", "info.json reports total_episodes <= 0; dataset was not finalized")
    if total_frames <= 0:
        add(results, "FAIL", "info.json reports total_frames <= 0")
    if args.expect_episodes is not None and total_episodes != args.expect_episodes:
        add(results, "FAIL", f"total_episodes {total_episodes} != expected {args.expect_episodes}")
    if args.expect_fps is not None and abs(fps - args.expect_fps) > 0.01:
        add(results, "WARN", f"info fps {fps:.2f} differs from expected {args.expect_fps:.2f}")

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data = read_parquet(data_path, results)
    if data is not None:
        print("\nData parquet:")
        print(f"  path: {data_path}")
        print(f"  shape: {data.shape}")
        print(f"  columns: {list(data.columns)}")
        check_columns(data, total_frames, args, results)

    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = read_parquet(episodes_path, results)
    if episodes is not None:
        print("\nEpisodes parquet:")
        print(f"  path: {episodes_path}")
        print(f"  shape: {episodes.shape}")
        if len(episodes) != total_episodes:
            add(results, "FAIL", f"episodes rows {len(episodes)} != info total_episodes {total_episodes}")
        if len(episodes) > 0 and "length" in episodes:
            lengths = [int(v) for v in episodes["length"].tolist()]
            print(f"  lengths: {lengths}")
            if sum(lengths) != total_frames:
                add(results, "FAIL", f"sum episode lengths {sum(lengths)} != info total_frames {total_frames}")

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        print(f"\nStats: {stats_path} ({stats_path.stat().st_size} bytes)")
    else:
        add(results, "FAIL", f"missing {stats_path}")

    tasks_path = root / "meta" / "tasks.parquet"
    tasks = read_parquet(tasks_path, results)
    if tasks is not None:
        print("\nTasks:")
        print(f"  shape: {tasks.shape}")
        print(f"  task labels: {tasks.index.tolist()}")
        if len(tasks) <= 0:
            add(results, "FAIL", "tasks.parquet has no tasks")

    video_key = next((key for key in features if key.startswith("observation.images.")), None)
    if video_key:
        video_path = root / "videos" / video_key / "chunk-000" / "file-000.mp4"
        check_video(video_path, info, args, results)
    else:
        add(results, "WARN", "no observation.images.* video feature in info.json")

    return print_results(results)


def check_columns(
    data: pd.DataFrame,
    total_frames: int,
    args: argparse.Namespace,
    results: list[CheckResult],
) -> None:
    required = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
    for column in required:
        if column not in data.columns:
            add(results, "FAIL", f"missing data column {column}")

    if total_frames and len(data) != total_frames:
        add(results, "FAIL", f"data rows {len(data)} != info total_frames {total_frames}")

    if "action" in data.columns and len(data) > 0:
        action_dim = len(data.iloc[0]["action"])
        print(f"  action_dim: {action_dim}")
        if action_dim != args.expect_action_dim:
            add(results, "FAIL", f"action dim {action_dim} != expected {args.expect_action_dim}")

    if "observation.state" in data.columns and len(data) > 0:
        state_dim = len(data.iloc[0]["observation.state"])
        print(f"  state_dim: {state_dim}")
        if state_dim != args.expect_state_dim:
            add(results, "FAIL", f"state dim {state_dim} != expected {args.expect_state_dim}")

    if "timestamp" in data.columns and len(data) > 1:
        timestamps = data["timestamp"].astype(float)
        print(f"  timestamp range: {timestamps.min():.3f} -> {timestamps.max():.3f}")
        if not timestamps.is_monotonic_increasing:
            add(results, "FAIL", "timestamps are not monotonic increasing")
        deltas = timestamps.diff().dropna()
        if (deltas <= 0).any():
            add(results, "FAIL", "timestamps contain non-positive frame deltas")
        if args.expect_fps is not None:
            expected_dt = 1.0 / args.expect_fps
            large_gaps = int((deltas > expected_dt * 2.0).sum())
            if large_gaps:
                add(results, "WARN", f"timestamps contain {large_gaps} gaps > {expected_dt * 2.0:.3f}s")

    if "frame_index" in data.columns and len(data) > 0:
        first = int(data["frame_index"].iloc[0])
        last = int(data["frame_index"].iloc[-1])
        print(f"  frame_index range: {first} -> {last}")
        if first != 0 or last != len(data) - 1:
            add(results, "FAIL", f"frame_index range {first}->{last} does not match row count {len(data)}")


def print_results(results: list[CheckResult]) -> int:
    fails = [r for r in results if r.level == "FAIL"]
    warns = [r for r in results if r.level == "WARN"]

    print("\nChecks:")
    if not results:
        print("  PASS: no issues found")
    else:
        for result in results:
            print(f"  {result.level}: {result.message}")

    if fails:
        print("\nResult: FAIL")
        return 1
    if warns:
        print("\nResult: WARN")
        return 2
    print("\nResult: PASS")
    return 0


def main() -> int:
    return check_dataset(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
