"""Sanity check for real D435i red cube grounding.

Captures aligned RealSense RGB-D frames, runs the repo real red cube detector,
and prints a GroundedObject-style result in the robot base frame.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from xlerobot_rl.real.camera_geometry import DEFAULT_EXTRINSICS, DEFAULT_INTRINSICS, RealCameraGeometry
from xlerobot_rl.real.red_cube_detector import detect_red_cube_rgbd, draw_red_cube_debug


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--cube-size-m", type=float, default=0.03)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument(
        "--no-center-correction",
        action="store_true",
        help="Disable adding cube_size/2 along the camera ray.",
    )
    parser.add_argument(
        "--expected-base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Optional measured cube center in base frame for error reporting, meters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real/sanity/red_cube_detector"),
    )
    parser.add_argument("--prefix", default="red_cube")
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture one frame after warmup, print result, save outputs, and exit.",
    )
    return parser


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def object_result(obj, debug: dict, expected_base: np.ndarray | None) -> dict:
    result = dict(debug)
    if obj is not None:
        result.update(
            {
                "object": {
                    "name": obj.name,
                    "bbox": list(obj.bbox),
                    "confidence": float(obj.confidence),
                    "pos_camera_m": obj.pos_camera.tolist(),
                    "pos_base_m": obj.pos_world.tolist(),
                    "attributes": _json_ready(obj.attributes),
                    "detection_method": obj.detection_method,
                }
            }
        )
        if expected_base is not None:
            result["expected_base_m"] = expected_base.tolist()
            result["error_m"] = float(np.linalg.norm(obj.pos_world - expected_base))
            result["error_xyz_m"] = (obj.pos_world - expected_base).tolist()
    return _json_ready(result)


def save_outputs(
    output_dir: Path,
    prefix: str,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    debug_bgr: np.ndarray,
    mask: np.ndarray | None,
    result: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{ts}"
    cv2.imwrite(str(output_dir / f"{stem}_rgb.png"), color_bgr)
    cv2.imwrite(str(output_dir / f"{stem}_debug.png"), debug_bgr)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(output_dir / f"{stem}_depth_mm.png"), depth_mm)
    if mask is not None:
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), np.asarray(mask, dtype=np.uint8) * 255)
    (output_dir / f"{stem}_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print(f"Saved outputs to {output_dir} ({stem}_*)")


def run_once(args, pipeline, align, depth_scale: float, geometry: RealCameraGeometry):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("failed to get aligned color/depth frames")

    color_bgr = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)

    obj, debug = detect_red_cube_rgbd(
        frame_bgr=color_bgr,
        depth_m=depth_m,
        geometry=geometry,
        cube_size_m=args.cube_size_m,
        min_area=args.min_area,
        center_correction=not args.no_center_correction,
    )
    expected = np.asarray(args.expected_base, dtype=np.float64) if args.expected_base else None
    mask = obj.mask if obj is not None else None
    debug_bgr = draw_red_cube_debug(color_bgr, obj, mask=mask, expected_base=expected)
    result = object_result(obj, debug, expected)
    return color_bgr, depth_m, mask, debug_bgr, result


def main() -> int:
    args = build_parser().parse_args()
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print("ERROR: pyrealsense2 is not installed.")
        print("Install the real hardware dependencies, e.g. `pip install pyrealsense2`, then rerun.")
        print(f"Underlying import error: {exc}")
        return 1

    geometry = RealCameraGeometry.from_config(args.intrinsics, args.extrinsics)

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    print("Real red cube grounding sanity")
    print(f"  color/depth: {args.color_width}x{args.color_height}@{args.fps}")
    print(f"  depth_scale: {depth_scale}")
    print(f"  cube_size_m: {args.cube_size_m}")
    print(f"  T_base_camera translation: {np.round(geometry.T_base_camera[:3, 3], 4).tolist()}")

    try:
        for _ in range(max(args.warmup_frames, 0)):
            pipeline.wait_for_frames()

        window = "detect_real_red_cube"
        if not args.no_preview:
            print("\nControls: SPACE/c capture, q/ESC quit")
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            color_bgr, depth_m, mask, debug_bgr, result = run_once(
                args, pipeline, align, depth_scale, geometry
            )
            if args.no_preview:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                if args.save:
                    save_outputs(args.output_dir, args.prefix, color_bgr, depth_m, debug_bgr, mask, result)
                return 0 if result.get("success") else 2

            display = debug_bgr
            if args.preview_scale != 1.0:
                display = cv2.resize(
                    debug_bgr,
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
                print(json.dumps(result, indent=2, ensure_ascii=False))
                if args.save:
                    save_outputs(args.output_dir, args.prefix, color_bgr, depth_m, debug_bgr, mask, result)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
