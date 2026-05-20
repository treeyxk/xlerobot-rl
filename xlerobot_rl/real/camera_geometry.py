"""Real camera geometry helpers for the XLeRobot head camera.

Coordinate conventions:
- camera frame: OpenCV optical frame, X right, Y down, Z forward
- base frame: robot base/world frame used by the real right arm config
- T_base_camera maps camera-frame points into base-frame points
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_INTRINSICS = Path("configs/calibration/head_camera_intrinsics_1280x720.yaml")
DEFAULT_EXTRINSICS = Path("configs/calibration/head_camera_extrinsics.yaml")


@dataclass(frozen=True)
class CameraIntrinsics:
    """OpenCV pinhole intrinsics."""

    K: np.ndarray
    dist: np.ndarray
    image_width: int | None = None
    image_height: int | None = None


def load_camera_intrinsics(path: Path | str = DEFAULT_INTRINSICS) -> CameraIntrinsics:
    """Load OpenCV intrinsics from the repo calibration YAML."""

    path = Path(path)
    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    K = np.asarray(cfg["intrinsic_matrix"]["data"], dtype=np.float64)
    dist = np.asarray(cfg.get("distortion_coefficients", []), dtype=np.float64).reshape(-1, 1)

    image_width = int(cfg["image_width"]) if cfg.get("image_width") is not None else None
    image_height = int(cfg["image_height"]) if cfg.get("image_height") is not None else None

    image_size = cfg.get("image_size")
    if (image_width is None or image_height is None) and isinstance(image_size, dict):
        image_width = int(image_size.get("width")) if image_size.get("width") is not None else None
        image_height = int(image_size.get("height")) if image_size.get("height") is not None else None
    elif (image_width is None or image_height is None) and isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        image_height = int(image_size[0])
        image_width = int(image_size[1])

    return CameraIntrinsics(K=K, dist=dist, image_width=image_width, image_height=image_height)


def load_camera_extrinsics(path: Path | str = DEFAULT_EXTRINSICS) -> np.ndarray:
    """Load T_base_camera from the repo hand-eye calibration YAML."""

    path = Path(path)
    with path.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return np.asarray(cfg["T_base_camera"]["data"], dtype=np.float64)


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Transform one 3D point by a homogeneous 4x4 matrix."""

    point = np.asarray(point, dtype=np.float64).reshape(3)
    return (np.asarray(T, dtype=np.float64) @ np.r_[point, 1.0])[:3]


def pixel_depth_to_camera(
    pixel_xy: tuple[float, float] | list[float] | np.ndarray,
    depth_z_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Back-project an image pixel and metric Z depth to camera coordinates."""

    if depth_z_m <= 0 or not np.isfinite(depth_z_m):
        raise ValueError(f"depth_z_m must be a positive finite value, got {depth_z_m!r}")

    pixel = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    pts = np.asarray([[[pixel[0], pixel[1]]]], dtype=np.float64)
    normalized = cv2.undistortPoints(pts, np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64))
    x_norm, y_norm = normalized.reshape(2)
    return np.asarray([x_norm * depth_z_m, y_norm * depth_z_m, depth_z_m], dtype=np.float64)


def depth_median_for_mask(depth_m: np.ndarray, mask: np.ndarray) -> float | None:
    """Return a robust median depth for valid pixels inside a mask."""

    mask_bool = np.asarray(mask) > 0
    depth = np.asarray(depth_m)
    values = depth[mask_bool & np.isfinite(depth) & (depth > 0)]
    if values.size == 0:
        return None

    lo, hi = np.percentile(values, [10, 90])
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        trimmed = values
    return float(np.median(trimmed))


def centroid_for_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the pixel centroid of a non-empty mask as (u, v)."""

    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


@dataclass(frozen=True)
class RealCameraGeometry:
    """Configured geometry for RealSense RGB-D points in the robot base frame."""

    K: np.ndarray
    dist: np.ndarray
    T_base_camera: np.ndarray

    @classmethod
    def from_config(
        cls,
        intrinsics_path: Path | str = DEFAULT_INTRINSICS,
        extrinsics_path: Path | str = DEFAULT_EXTRINSICS,
    ) -> "RealCameraGeometry":
        intrinsics = load_camera_intrinsics(intrinsics_path)
        T_base_camera = load_camera_extrinsics(extrinsics_path)
        return cls(K=intrinsics.K, dist=intrinsics.dist, T_base_camera=T_base_camera)

    def pixel_depth_to_camera(
        self,
        pixel_xy: tuple[float, float] | list[float] | np.ndarray,
        depth_z_m: float,
    ) -> np.ndarray:
        return pixel_depth_to_camera(pixel_xy, depth_z_m, self.K, self.dist)

    def camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        return transform_point(self.T_base_camera, point_camera)

    def pixel_depth_to_base(
        self,
        pixel_xy: tuple[float, float] | list[float] | np.ndarray,
        depth_z_m: float,
    ) -> np.ndarray:
        return self.camera_to_base(self.pixel_depth_to_camera(pixel_xy, depth_z_m))

    def mask_depth_to_camera_surface(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        centroid_px: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        """Estimate a visible surface point from mask centroid and mask depth."""

        if centroid_px is None:
            centroid_px = centroid_for_mask(mask)
        if centroid_px is None:
            raise ValueError("mask is empty")

        depth_z = depth_median_for_mask(depth_m, mask)
        if depth_z is None:
            raise ValueError("no valid positive depth inside mask")

        p_camera = self.pixel_depth_to_camera(centroid_px, depth_z)
        return p_camera, centroid_px, depth_z

    def mask_depth_to_base(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        centroid_px: tuple[float, float] | None = None,
        object_size_m: float | None = None,
        center_correction: bool = True,
    ) -> dict[str, np.ndarray | tuple[float, float] | float]:
        """Estimate object surface/center in camera and base frames from RGB-D mask.

        RealSense depth on a cube mask usually lands on the visible/front face.
        If object_size_m is provided, center_correction adds half that size along
        the camera ray to approximate the cube center.
        """

        p_camera_surface, centroid, depth_z = self.mask_depth_to_camera_surface(
            mask, depth_m, centroid_px=centroid_px
        )
        p_camera_center = p_camera_surface.copy()
        center_correction_m = 0.0
        if object_size_m is not None and center_correction:
            ray = p_camera_surface / (np.linalg.norm(p_camera_surface) + 1e-12)
            center_correction_m = float(object_size_m) / 2.0
            p_camera_center = p_camera_surface + ray * center_correction_m

        return {
            "centroid_px": centroid,
            "depth_median_m": float(depth_z),
            "center_correction_m": float(center_correction_m),
            "p_camera_surface_m": p_camera_surface,
            "p_camera_center_m": p_camera_center,
            "p_base_surface_m": self.camera_to_base(p_camera_surface),
            "p_base_center_m": self.camera_to_base(p_camera_center),
        }
