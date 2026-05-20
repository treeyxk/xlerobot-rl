"""Real RGB-D red cube detector.

This is intentionally simple for the current hardware milestone:
- HSV red segmentation in BGR camera frames
- connected component filtering
- RGB-D grounding through RealCameraGeometry
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xlerobot_rl.perception.data_types import GroundedObject
from xlerobot_rl.real.camera_geometry import RealCameraGeometry


@dataclass(frozen=True)
class RedCubeDetection:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    area_px: int
    confidence: float


def _red_mask_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 70, 50], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_red_cube_bgr(
    frame_bgr: np.ndarray,
    min_area: int = 300,
    reject_border: bool = True,
) -> RedCubeDetection | None:
    """Detect the largest red connected component in a BGR frame."""

    mask = _red_mask_bgr(frame_bgr)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    candidates: list[tuple[int, int, int, int, int, int]] = []

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
        if reject_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue
        candidates.append((area, label, x, y, bw, bh))

    if not candidates:
        return None

    area, label, x, y, bw, bh = max(candidates, key=lambda item: item[0])
    cx, cy = centroids[label]
    component_mask = labels == label
    confidence = float(np.clip(area / 5000.0, 0.1, 1.0))
    return RedCubeDetection(
        mask=component_mask,
        bbox=(int(x), int(y), int(x + bw), int(y + bh)),
        centroid_px=(float(cx), float(cy)),
        area_px=int(area),
        confidence=confidence,
    )


def detect_red_cube_rgbd(
    frame_bgr: np.ndarray,
    depth_m: np.ndarray,
    geometry: RealCameraGeometry,
    cube_size_m: float = 0.03,
    min_area: int = 300,
    center_correction: bool = True,
    object_id: int = 0,
) -> tuple[GroundedObject | None, dict]:
    """Detect and ground a red cube in base coordinates.

    Returns:
        (object, debug). object is None if RGB or depth grounding fails.
    """

    detection = detect_red_cube_bgr(frame_bgr, min_area=min_area)
    if detection is None:
        return None, {"success": False, "reason": "red cube not detected"}

    try:
        geom = geometry.mask_depth_to_base(
            detection.mask,
            depth_m,
            centroid_px=detection.centroid_px,
            object_size_m=cube_size_m,
            center_correction=center_correction,
        )
    except ValueError as exc:
        return None, {
            "success": False,
            "reason": str(exc),
            "bbox": list(detection.bbox),
            "centroid_px": list(detection.centroid_px),
            "area_px": detection.area_px,
        }

    attrs = {
        "color": "red",
        "cube_size_m": float(cube_size_m),
        "area_px": detection.area_px,
        "centroid_px": list(detection.centroid_px),
        "depth_median_m": float(geom["depth_median_m"]),
        "center_correction_m": float(geom["center_correction_m"]),
        "p_camera_surface_m": np.asarray(geom["p_camera_surface_m"]).tolist(),
        "p_base_surface_m": np.asarray(geom["p_base_surface_m"]).tolist(),
    }
    obj = GroundedObject(
        object_id=object_id,
        name="red_cube",
        bbox=detection.bbox,
        mask=detection.mask.astype(bool),
        pos_camera=np.asarray(geom["p_camera_center_m"], dtype=np.float64),
        pos_world=np.asarray(geom["p_base_center_m"], dtype=np.float64),
        confidence=detection.confidence,
        attributes=attrs,
        is_target=True,
        detection_method="hsv+realsense_depth",
    )

    debug = {
        "success": True,
        "bbox": list(detection.bbox),
        "centroid_px": list(detection.centroid_px),
        "area_px": detection.area_px,
        "depth_median_m": float(geom["depth_median_m"]),
        "p_camera_center_m": obj.pos_camera.tolist(),
        "p_base_center_m": obj.pos_world.tolist(),
        "p_camera_surface_m": attrs["p_camera_surface_m"],
        "p_base_surface_m": attrs["p_base_surface_m"],
        "center_correction_m": attrs["center_correction_m"],
    }
    return obj, debug


def draw_red_cube_debug(
    frame_bgr: np.ndarray,
    obj: GroundedObject | None,
    mask: np.ndarray | None = None,
    expected_base: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a compact visual debug overlay."""

    vis = frame_bgr.copy()
    if obj is not None:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = obj.attributes.get("centroid_px", [(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 255, 255), -1)
        mask = obj.mask if mask is None else mask
        text = "base xyz: " + ", ".join(f"{v:+.3f}" for v in obj.pos_world)
        cv2.putText(vis, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if expected_base is not None:
            err = float(np.linalg.norm(obj.pos_world - expected_base))
            cv2.putText(
                vis,
                f"expected err: {err * 100:.1f} cm",
                (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    if mask is not None:
        overlay = vis.copy()
        overlay[np.asarray(mask) > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    return vis
