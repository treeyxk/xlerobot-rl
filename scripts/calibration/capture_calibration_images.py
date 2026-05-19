"""Capture still images from an OpenCV camera for calibration.

Controls:
  SPACE / c : save current frame
  q / ESC   : quit
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", default="/dev/xlerobot_head_camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--fourcc", default=None, help="Optional FOURCC, e.g. YUYV or MJPG.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real/calibration/head_camera_images"),
    )
    parser.add_argument("--prefix", default="head_camera")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture one frame after warmup and exit. Useful on headless systems.",
    )
    return parser


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    source: int | str = int(args.camera_index) if str(args.camera_index).isdigit() else args.camera_index
    cap = cv2.VideoCapture(source)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def save_frame(frame, output_dir: Path, prefix: str, index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{index:04d}_{timestamp}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"failed to save image: {path}")
    return path


def main() -> int:
    args = build_parser().parse_args()
    cap = open_camera(args)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera {args.camera_index}")
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)).strip()

    print("Camera:")
    print(f"  requested: {args.camera_index} {args.width}x{args.height}@{args.fps:g}")
    print(f"  actual:    {actual_width}x{actual_height}@{actual_fps:.2f} fourcc={fourcc or 'unknown'}")
    print(f"Output dir: {args.output_dir}")

    for _ in range(max(args.warmup_frames, 0)):
        cap.read()

    ok, frame = cap.read()
    if not ok:
        cap.release()
        print("ERROR: failed to read initial frame")
        return 1

    index = args.start_index
    if args.no_preview:
        path = save_frame(frame, args.output_dir, args.prefix, index)
        print(f"Saved: {path}")
        cap.release()
        return 0

    print("\nControls: SPACE/c save, q/ESC quit")
    window = "capture_calibration_images"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: failed to read frame")
                continue

            display = frame
            if args.preview_scale != 1.0:
                display = cv2.resize(
                    frame,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )

            overlay = display.copy()
            text = f"{actual_width}x{actual_height}  saved={index - args.start_index}  SPACE/c save  q quit"
            cv2.putText(
                overlay,
                text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("c")):
                path = save_frame(frame, args.output_dir, args.prefix, index)
                print(f"Saved: {path}")
                index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {index - args.start_index} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
