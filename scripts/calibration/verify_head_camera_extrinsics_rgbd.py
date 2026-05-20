"""Verify head-camera extrinsics with a red cube and RealSense RGB-D.

This script reads aligned D435i color+depth frames, detects a red cube with HSV,
back-projects the cube mask center into the OpenCV camera frame, and transforms
it into the robot base frame using configs/calibration/head_camera_extrinsics.yaml.

It is a sanity check, not a new calibration routine.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_INTRINSICS = Path("configs/calibration/head_camera_intrinsics_1280x720.yaml")
DEFAULT_EXTRINSICS = Path("configs/calibration/head_camera_extrinsics.yaml")


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
    parser.add_argument(
        "--no-center-correction",
        action="store_true",
        help="Disable adding cube_size/2 along the camera ray. By default, depth is treated as front face.",
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
        default=Path("data/real/calibration/extrinsics_rgbd_verify"),
    )
    parser.add_argument("--prefix", default="red_cube_verify")
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--preview-scale", type=float, default=0.75)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture one frame after warmup, print result, save debug images, and exit.",
    )
    return parser


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    K = np.asarray(cfg["intrinsic_matrix"]["data"], dtype=np.float64)
    dist = np.asarray(cfg["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def load_extrinsics(path: Path) -> np.ndarray:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return np.asarray(cfg["T_base_camera"]["data"], dtype=np.float64)


def detect_red_cube(frame_bgr: np.ndarray, min_area: int) -> tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 70, 50], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    h, w = mask.shape
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            # Avoid selecting red objects cut by the image border.
            continue
        candidates.append((area, label, x, y, bw, bh))

    if not candidates:
        return mask, {}

    area, label, x, y, bw, bh = max(candidates, key=lambda item: item[0])
    cx, cy = centroids[label]
    component_mask = (labels == label).astype(np.uint8) * 255
    info = {
        "label": int(label),
        "area_px": int(area),
        "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
        "centroid_px": [float(cx), float(cy)],
        "mask": component_mask,
    }
    return component_mask, info


def median_depth_for_mask(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    values = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0)]
    if values.size == 0:
        return None
    lo, hi = np.percentile(values, [10, 90])
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        trimmed = values
    return float(np.median(trimmed))


def pixel_to_camera_point(
    pixel_xy: tuple[float, float],
    depth_z_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    pts = np.asarray([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float64)
    normalized = cv2.undistortPoints(pts, K, dist).reshape(2)
    return np.array([normalized[0] * depth_z_m, normalized[1] * depth_z_m, depth_z_m], dtype=np.float64)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (T @ np.r_[p, 1.0])[:3]


def draw_debug(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    info: dict,
    p_base_center: np.ndarray | None,
    expected_base: np.ndarray | None,
) -> np.ndarray:
    vis = frame_bgr.copy()
    if info:
        x1, y1, x2, y2 = info["bbox"]
        cx, cy = info["centroid_px"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 255), -1)
        overlay = vis.copy()
        overlay[mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    if p_base_center is not None:
        text = "base xyz: " + ", ".join(f"{v:+.3f}" for v in p_base_center)
        cv2.putText(vis, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    if p_base_center is not None and expected_base is not None:
        err = float(np.linalg.norm(p_base_center - expected_base))
        text = f"expected err: {err * 100:.1f} cm"
        cv2.putText(vis, text, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def save_outputs(
    args: argparse.Namespace,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    debug_bgr: np.ndarray,
    result: dict,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.prefix}_{ts}"
    cv2.imwrite(str(args.output_dir / f"{stem}_rgb.png"), color_bgr)
    cv2.imwrite(str(args.output_dir / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(args.output_dir / f"{stem}_debug.png"), debug_bgr)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(args.output_dir / f"{stem}_depth_mm.png"), depth_mm)
    (args.output_dir / f"{stem}_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print(f"Saved debug outputs to {args.output_dir} ({stem}_*)")


def run_once(args: argparse.Namespace, pipeline, align, depth_scale: float, K: np.ndarray, dist: np.ndarray, T_base_camera: np.ndarray):
    import pyrealsense2 as rs  # noqa: F401

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("failed to get aligned color/depth frames")

    color_bgr = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)

    mask, info = detect_red_cube(color_bgr, args.min_area)
    if not info:
        debug = draw_debug(color_bgr, mask, info, None, None)
        return color_bgr, depth_m, mask, debug, {"success": False, "reason": "red cube not detected"}

    depth_z = median_depth_for_mask(depth_m, mask)
    if depth_z is None:
        debug = draw_debug(color_bgr, mask, info, None, None)
        return color_bgr, depth_m, mask, debug, {"success": False, "reason": "no valid depth in cube mask"}

    u, v = info["centroid_px"]
    p_camera_surface = pixel_to_camera_point((u, v), depth_z, K, dist)
    ray = p_camera_surface / (np.linalg.norm(p_camera_surface) + 1e-12)
    if args.no_center_correction:
        p_camera_center = p_camera_surface
    else:
        p_camera_center = p_camera_surface + ray * (args.cube_size_m / 2.0)

    p_base_surface = transform_point(T_base_camera, p_camera_surface)
    p_base_center = transform_point(T_base_camera, p_camera_center)
    expected = np.asarray(args.expected_base, dtype=np.float64) if args.expected_base else None
    debug = draw_debug(color_bgr, mask, info, p_base_center, expected)

    result = {
        "success": True,
        "cube_size_m": args.cube_size_m,
        "center_correction_m": 0.0 if args.no_center_correction else args.cube_size_m / 2.0,
        "pixel_centroid": [float(u), float(v)],
        "mask_area_px": info["area_px"],
        "bbox": info["bbox"],
        "depth_median_m": float(depth_z),
        "p_camera_surface_m": p_camera_surface.tolist(),
        "p_camera_center_m": p_camera_center.tolist(),
        "p_base_surface_m": p_base_surface.tolist(),
        "p_base_center_m": p_base_center.tolist(),
    }
    if expected is not None:
        result["expected_base_m"] = expected.tolist()
        result["error_m"] = float(np.linalg.norm(p_base_center - expected))
        result["error_xyz_m"] = (p_base_center - expected).tolist()
    return color_bgr, depth_m, mask, debug, result


def main() -> int:
    args = build_parser().parse_args()
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print("ERROR: pyrealsense2 is not installed.")
        print("Install the real hardware dependencies, e.g. `pip install pyrealsense2`, then rerun.")
        print(f"Underlying import error: {exc}")
        return 1

    K, dist = load_intrinsics(args.intrinsics)
    T_base_camera = load_extrinsics(args.extrinsics)

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

    print("RealSense RGB-D extrinsics verification")
    print(f"  color: {args.color_width}x{args.color_height}@{args.fps}")
    print(f"  depth: {args.depth_width}x{args.depth_height}@{args.fps}, scale={depth_scale}")
    print(f"  cube_size_m: {args.cube_size_m}")
    print(f"  T_base_camera translation: {np.round(T_base_camera[:3, 3], 4).tolist()}")

    try:
        for _ in range(max(args.warmup_frames, 0)):
            pipeline.wait_for_frames()

        window = "verify_head_camera_extrinsics_rgbd"
        if not args.no_preview:
            print("\nControls: SPACE/c capture, q/ESC quit")
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            color_bgr, depth_m, mask, debug, result = run_once(
                args, pipeline, align, depth_scale, K, dist, T_base_camera
            )
            if args.no_preview:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                if args.save:
                    save_outputs(args, color_bgr, depth_m, mask, debug, result)
                return 0 if result.get("success") else 2

            display = debug
            if args.preview_scale != 1.0:
                display = cv2.resize(
                    debug,
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
                    save_outputs(args, color_bgr, depth_m, mask, debug, result)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
