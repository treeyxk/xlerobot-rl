"""Measure OpenCV camera capture stability for a requested profile."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", default="/dev/video6")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--warmup-s", type=float, default=1)
    parser.add_argument("--save-frame", type=Path, default=Path("data/debug/camera_fps_first_frame.jpg"))
    return parser


def open_camera(index_or_path: str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    if index_or_path.isdigit():
        cap = cv2.VideoCapture(int(index_or_path), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index_or_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main() -> int:
    args = build_parser().parse_args()
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera {args.camera_index}")
        return 1

    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip()

    print("Requested profile:")
    print(f"  camera: {args.camera_index}")
    print(f"  width/height/fps: {args.width}x{args.height}@{args.fps:g}")
    print("OpenCV reported profile:")
    print(f"  width/height/fps: {reported_width}x{reported_height}@{reported_fps:.2f}")
    print(f"  fourcc: {fourcc or 'unknown'}")

    warmup_end = time.perf_counter() + args.warmup_s
    while time.perf_counter() < warmup_end:
        cap.read()

    timestamps: list[float] = []
    first_frame: np.ndarray | None = None
    failed_reads = 0
    end = time.perf_counter() + args.duration_s
    while time.perf_counter() < end:
        ok, frame = cap.read()
        now = time.perf_counter()
        if not ok:
            failed_reads += 1
            continue
        if first_frame is None:
            first_frame = frame
        timestamps.append(now)

    cap.release()

    if len(timestamps) < 2:
        print("ERROR: captured fewer than 2 frames")
        return 1

    intervals = np.diff(np.array(timestamps))
    measured_duration = timestamps[-1] - timestamps[0]
    measured_fps = (len(timestamps) - 1) / measured_duration
    target_period = 1.0 / args.fps
    slow_frames = int(np.sum(intervals > target_period * 1.5))

    print("\nCapture result:")
    print(f"  frames: {len(timestamps)}")
    print(f"  failed_reads: {failed_reads}")
    print(f"  measured_duration_s: {measured_duration:.3f}")
    print(f"  measured_fps: {measured_fps:.2f}")
    print(f"  frame_interval_ms mean/p95/max: {intervals.mean()*1000:.1f} / {np.percentile(intervals, 95)*1000:.1f} / {intervals.max()*1000:.1f}")
    print(f"  slow_frames_>{target_period*1.5*1000:.1f}ms: {slow_frames}")

    if first_frame is not None and args.save_frame:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), first_frame)
        print(f"  saved_first_frame: {args.save_frame}")

    if measured_fps < args.fps * 0.9 or slow_frames > max(3, len(timestamps) * 0.02):
        print("\nResult: WARN, capture may be unstable for this profile.")
        return 2

    print("\nResult: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
