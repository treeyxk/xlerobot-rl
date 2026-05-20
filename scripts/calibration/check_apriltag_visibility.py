"""Check whether the head camera can see an AprilTag across robot poses.

This is a visibility/debug tool only. It detects tags in the live RGB stream,
draws tag corners and IDs, and can save snapshots for later inspection.

Controls:
  SPACE / c : save annotated snapshot
  q / ESC   : quit
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


APRILTAG_DICTS = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", default="/dev/xlerobot_head_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--tag-family", choices=sorted(APRILTAG_DICTS), default="tag36h11")
    parser.add_argument("--target-id", type=int, default=None, help="Optional tag ID to require.")
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real/calibration/apriltag_visibility"),
    )
    parser.add_argument("--prefix", default="apriltag")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture one frame after warmup, print detections, save annotated image, and exit.",
    )
    return parser


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    source: int | str = int(args.camera_index) if str(args.camera_index).isdigit() else args.camera_index
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def make_detector(tag_family: str) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[tag_family])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(detector: cv2.aruco.ArucoDetector, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
    return corners, ids.reshape(-1)


def draw_detections(frame: np.ndarray, corners: list[np.ndarray], ids: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    if len(corners):
        cv2.aruco.drawDetectedMarkers(vis, corners, ids.reshape(-1, 1).astype(np.int32))

    for tag_corners, tag_id in zip(corners, ids):
        pts = tag_corners.reshape(4, 2)
        center = pts.mean(axis=0)
        cv2.circle(vis, tuple(np.round(center).astype(int)), 4, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"id={int(tag_id)}",
            tuple(np.round(center + np.array([8, -8])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def save_snapshot(image: np.ndarray, output_dir: Path, prefix: str, index: int, ids: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    id_text = "none" if ids.size == 0 else "-".join(str(int(i)) for i in ids)
    path = output_dir / f"{prefix}_{index:04d}_{timestamp}_ids-{id_text}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save image: {path}")
    return path


def print_detections(ids: np.ndarray, target_id: int | None) -> None:
    if ids.size == 0:
        print("Detected: none")
        return
    found = [int(i) for i in ids]
    if target_id is None:
        print(f"Detected IDs: {found}")
        return
    print(f"Detected IDs: {found} | target_id={target_id} | visible={target_id in found}")


def main() -> int:
    args = build_parser().parse_args()
    detector = make_detector(args.tag_family)
    cap = open_camera(args)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera {args.camera_index}")
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera:")
    print(f"  requested: {args.camera_index} {args.width}x{args.height}@{args.fps:g}")
    print(f"  actual:    {actual_width}x{actual_height}@{actual_fps:.2f}")
    print(f"AprilTag family: {args.tag_family}")
    print(f"Output dir: {args.output_dir}")

    for _ in range(max(args.warmup_frames, 0)):
        cap.read()

    index = 0
    window = "check_apriltag_visibility"
    if not args.no_preview:
        print("\nControls: SPACE/c save, q/ESC quit")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: failed to read frame")
                continue

            corners, ids = detect(detector, frame)
            vis = draw_detections(frame, corners, ids)
            status = "NO TAG" if ids.size == 0 else f"IDs: {[int(i) for i in ids]}"
            if args.target_id is not None:
                status += f"  target={args.target_id} visible={args.target_id in set(map(int, ids))}"
            cv2.putText(
                vis,
                status,
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if ids.size else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if args.no_preview:
                print_detections(ids, args.target_id)
                path = save_snapshot(vis, args.output_dir, args.prefix, index, ids)
                print(f"Saved: {path}")
                return 0 if ids.size else 2

            display = vis
            if args.preview_scale != 1.0:
                display = cv2.resize(
                    vis,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("c")):
                print_detections(ids, args.target_id)
                path = save_snapshot(vis, args.output_dir, args.prefix, index, ids)
                print(f"Saved: {path}")
                index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
