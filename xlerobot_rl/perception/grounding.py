"""M1.2 Semantic Grounding: HSV + SAM2 refinement + 3D reconstruction.

Pipeline:
  1. HSVColorDetector: rgb → coarse color mask (CPU, numpy)
  2. SAM2Refiner: rgb + coarse mask → refined mask (GPU, PyTorch)
  3. depth + refined_mask + K → 3D position in camera frame
  4. T_world_camera @ pos_cam → pos_world

Borrowed from grasp_demo but rewritten for sim integration.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from xlerobot_rl.perception.data_types import GroundedObject


# Default SAM2-tiny config + weights
_DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_DEFAULT_SAM2_WEIGHTS = Path(__file__).resolve().parents[2] / "data/models/sam2/sam2.1_hiera_tiny.pt"


# ============================================================
# Stage 1: HSV Detector (CPU, numpy)
# ============================================================
class HSVColorDetector:
    """Detect colored cube regions via HSV thresholding.
    
    This is the M1 v0 color-block baseline from the project plan. It is deliberately
    limited to red/blue/green blocks, not open-vocabulary grounding.
    """
    
    # OpenCV HSV: H in [0, 179], S/V in [0, 255]
    HSV_RANGES = {
        "red": [
            (np.array([0, 80, 80]), np.array([10, 255, 255])),
            (np.array([165, 80, 80]), np.array([179, 255, 255])),
        ],
        "blue": [
            (np.array([95, 80, 80]), np.array([130, 255, 255])),
        ],
        "green": [
            (np.array([40, 80, 80]), np.array([85, 255, 255])),
        ],
    }
    
    def __init__(
        self,
        min_area_pixels: int = 30,    # 过滤太小的检测 (噪点)
        max_area_pixels: int = 2500,  # 过滤车身/桌面等大色块
        max_bbox_area_pixels: int = 3000,
    ):
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.max_bbox_area_pixels = max_bbox_area_pixels
    
    def detect_all(self, rgb: np.ndarray, color: str) -> list[dict]:
        """Detect largest region for a supported color.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            color: "red", "blue", or "green"
        
        Returns:
            list of dicts with 'mask', 'bbox', 'centroid', and 'area', sorted by area descending.
        """
        assert rgb.dtype == np.uint8, f"expected uint8, got {rgb.dtype}"
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected (H,W,3), got {rgb.shape}"
        if color not in self.HSV_RANGES:
            raise ValueError(f"unsupported HSV color: {color}")
        
        # RGB → HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for low, high in self.HSV_RANGES[color]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        
        # Morphology cleanup (close small holes, remove small noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Connected components. M1 v0 的目标是桌面小色块, 不能直接取最大
        # component, 否则蓝色车身会压过蓝色 cube。
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Skip label 0 (background)
        if num_labels <= 1:
            return []
        
        candidates = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            bbox_area = w * h
            if not (self.min_area_pixels <= area <= self.max_area_pixels):
                continue
            if bbox_area > self.max_bbox_area_pixels:
                continue
            candidates.append((area, label_idx, x, y, w, h))

        if not candidates:
            return []

        results = []
        for area, label_idx, x, y, w, h in sorted(candidates, reverse=True):
            bbox = (int(x), int(y), int(x + w), int(y + h))
            centroid = (float(centroids[label_idx, 0]), float(centroids[label_idx, 1]))
            component_mask = (labels == label_idx).astype(bool)
            results.append(
                {
                    "mask": component_mask,
                    "bbox": bbox,
                    "centroid": centroid,
                    "area": int(area),
                }
            )
        return results

    def detect(self, rgb: np.ndarray, color: str) -> Optional[dict]:
        """Detect the largest valid region for a supported color."""
        detections = self.detect_all(rgb, color)
        return detections[0] if detections else None


class HSVRedDetector(HSVColorDetector):
    """Backward-compatible red-only detector used by older sanity scripts."""

    def detect(self, rgb: np.ndarray) -> Optional[dict]:
        return super().detect(rgb, "red")


# ============================================================
# Stage 2: SAM2 Refiner (GPU, PyTorch)
# ============================================================
class SAM2Refiner:
    """Refine HSV-detected mask using SAM2-tiny.
    
    Uses bbox + centroid as SAM2 prompt to get cleaner mask.
    """
    
    def __init__(
        self,
        config_name: str = _DEFAULT_SAM2_CONFIG,
        weights_path: str | Path = _DEFAULT_SAM2_WEIGHTS,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        weights_path = Path(weights_path)
        assert weights_path.exists(), f"SAM2 weights not found: {weights_path}"
        
        print(f"[SAM2Refiner] Loading SAM2 from {weights_path}")
        print(f"[SAM2Refiner] Using config: {config_name}")
        print(f"[SAM2Refiner] Device: {device}")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_model = build_sam2(config_name, str(weights_path), device=device)
        self.predictor = SAM2ImagePredictor(sam_model)
    
    def refine(
        self,
        rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
        centroid: tuple[float, float],
    ) -> np.ndarray:
        """Refine mask using SAM2 with bbox + centroid prompt.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            bbox: (x1, y1, x2, y2)
            centroid: (cx, cy)
        
        Returns:
            (H, W) bool mask
        """
        self.predictor.set_image(rgb)
        
        # SAM2 wants box as np.array((4,)) and point as np.array((N, 2))
        box_arr = np.array(bbox, dtype=np.float32)
        point_coords = np.array([centroid], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)   # 1 = foreground
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=False,
        )
        
        # masks: (1, H, W) bool. Take first
        refined_mask = masks[0].astype(bool)
        return refined_mask


# ============================================================
# Stage 3+4: 3D Reconstruction + Frame Transform
# ============================================================
def mask_to_3d_position(
    mask: np.ndarray,
    depth_meters: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    use_median: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3D position from mask + depth.
    
    Args:
        mask: (H, W) bool, True = object pixels
        depth_meters: (H, W) float, depth in meters (0 = invalid)
        K: (3, 3) camera intrinsic matrix
        T_world_camera: (4, 4) camera-to-world transform
        use_median: if True, use median depth; else mean (median is more robust to depth noise)
    
    Returns:
        pos_camera: (3,) in camera optical frame, meters
        pos_world: (3,) in world frame, meters
    """
    assert mask.dtype == bool
    assert mask.shape == depth_meters.shape, f"shape mismatch: {mask.shape} vs {depth_meters.shape}"
    
    # Valid pixels: in mask AND has depth
    valid = mask & (depth_meters > 0.0) & np.isfinite(depth_meters)
    if not valid.any():
        raise ValueError("No valid depth pixels in mask")
    
    # Pixel coordinates of valid mask pixels
    ys, xs = np.where(valid)
    depths = depth_meters[valid]
    
    # Aggregate depth: median is robust to noise, mean is smooth
    if use_median:
        z = float(np.median(depths))
    else:
        z = float(np.mean(depths))
    
    # Pixel center (use mask centroid in image)
    cx_pix = float(np.mean(xs))
    cy_pix = float(np.mean(ys))
    
    # Back-project from pixel → camera optical frame
    # K @ [x_c, y_c, z_c, 1].T → s * [u, v, 1, 1].T
    # x_c = (u - K[0,2]) * z / K[0,0]
    # y_c = (v - K[1,2]) * z / K[1,1]
    fx, fy = K[0, 0], K[1, 1]
    cx_K, cy_K = K[0, 2], K[1, 2]
    
    x_cam = (cx_pix - cx_K) * z / fx
    y_cam = (cy_pix - cy_K) * z / fy
    z_cam = z
    
    pos_camera = np.array([x_cam, y_cam, z_cam], dtype=np.float32)
    
    # Transform to world frame
    pos_cam_h = np.append(pos_camera, 1.0)              # (4,) homogeneous
    pos_world_h = T_world_camera @ pos_cam_h            # (4,)
    pos_world = pos_world_h[:3].astype(np.float32)
    
    return pos_camera, pos_world


# ============================================================
# Main Pipeline
# ============================================================
class GroundingPipeline:
    """End-to-end grounding: rgb + depth + camera params → GroundedObject."""
    
    def __init__(
        self,
        use_sam2: bool = True,
        sam2_weights_path: Optional[str | Path] = None,
        hsv_min_area: int = 30,
        hsv_max_area: int = 2500,
    ):
        self.use_sam2 = use_sam2
        self.hsv_detector = HSVColorDetector(
            min_area_pixels=hsv_min_area,
            max_area_pixels=hsv_max_area,
        )
        if use_sam2:
            weights = sam2_weights_path or _DEFAULT_SAM2_WEIGHTS
            self.sam2_refiner = SAM2Refiner(weights_path=weights)
        else:
            self.sam2_refiner = None
    
    def detect_red_cube(
        self,
        rgb: np.ndarray,
        depth_meters: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
        object_id: int = 0,
    ) -> Optional[GroundedObject]:
        """Detect a red cube and return its 3D pose.
        
        Args:
            rgb: (H, W, 3) uint8 RGB
            depth_meters: (H, W) float, depth in meters
            K: (3, 3) camera intrinsic
            T_world_camera: (4, 4) cam-to-world transform
            object_id: integer ID for tracking
        
        Returns:
            GroundedObject or None (if no red detected)
        """
        return self.detect_color_cube(
            rgb=rgb,
            depth_meters=depth_meters,
            K=K,
            T_world_camera=T_world_camera,
            color="red",
            object_id=object_id,
            target_color="red",
        )

    def detect_color_cube(
        self,
        rgb: np.ndarray,
        depth_meters: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
        color: str,
        object_id: int = 0,
        target_color: str | None = None,
    ) -> Optional[GroundedObject]:
        """Detect one colored cube and return its grounded object record."""
        hsv_results = self.hsv_detector.detect_all(rgb, color)
        for hsv_result in hsv_results:
            return self._ground_hsv_result(
                rgb=rgb,
                depth_meters=depth_meters,
                K=K,
                T_world_camera=T_world_camera,
                color=color,
                object_id=object_id,
                target_color=target_color,
                hsv_result=hsv_result,
            )
        return None

    def detect_colored_cubes(
        self,
        rgb: np.ndarray,
        depth_meters: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
        target_color: str,
        colors: tuple[str, ...] = ("red", "blue", "green"),
    ) -> list[GroundedObject]:
        """Detect supported colored cubes and mark target vs distractors."""
        objects: list[GroundedObject] = []
        for color in colors:
            obj = self.detect_color_cube(
                rgb=rgb,
                depth_meters=depth_meters,
                K=K,
                T_world_camera=T_world_camera,
                color=color,
                object_id=len(objects),
                target_color=target_color,
            )
            if obj is not None:
                objects.append(obj)
        return objects

    def _refine_mask(self, rgb: np.ndarray, hsv_result: dict) -> tuple[np.ndarray, str]:
        if self.use_sam2:
            try:
                refined_mask = self.sam2_refiner.refine(
                    rgb,
                    bbox=hsv_result["bbox"],
                    centroid=hsv_result["centroid"],
                )
                return refined_mask, "hsv+sam2"
            except Exception as e:
                print(f"[GroundingPipeline] SAM2 failed: {e}, fallback to HSV")
                return hsv_result["mask"], "hsv-only-fallback"
        return hsv_result["mask"], "hsv-only"

    def _ground_hsv_result(
        self,
        rgb: np.ndarray,
        depth_meters: np.ndarray,
        K: np.ndarray,
        T_world_camera: np.ndarray,
        color: str,
        object_id: int,
        target_color: str | None,
        hsv_result: dict,
    ) -> Optional[GroundedObject]:
        refined_mask, detection_method = self._refine_mask(
            rgb=rgb,
            hsv_result=hsv_result,
        )

        try:
            pos_cam, pos_world = mask_to_3d_position(
                mask=refined_mask,
                depth_meters=depth_meters,
                K=K,
                T_world_camera=T_world_camera,
            )
        except ValueError as e:
            print(f"[GroundingPipeline] 3D reconstruction failed for {color}: {e}")
            return None

        ys, xs = np.where(refined_mask)
        if len(xs) == 0:
            return None
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        confidence = float(refined_mask.sum() / max(bbox_area, 1))
        is_target = target_color is not None and color == target_color

        return GroundedObject(
            object_id=object_id,
            name=f"{color}_cube",
            bbox=bbox,
            mask=refined_mask,
            pos_camera=pos_cam,
            pos_world=pos_world,
            confidence=confidence,
            attributes={"color": color, "category": "cube"},
            is_target=is_target,
            detection_method=detection_method,
        )
