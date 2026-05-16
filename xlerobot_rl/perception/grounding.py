"""M1.2 Semantic Grounding: HSV + SAM2 refinement + 3D reconstruction.

Pipeline:
  1. HSVRedDetector: rgb → coarse red mask (CPU, numpy)
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

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from xlerobot_rl.perception.data_types import GroundedObject


# Default SAM2-tiny config + weights
_DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_DEFAULT_SAM2_WEIGHTS = Path(__file__).resolve().parents[2] / "data/models/sam2/sam2.1_hiera_tiny.pt"


# ============================================================
# Stage 1: HSV Detector (CPU, numpy)
# ============================================================
class HSVRedDetector:
    """Detect red regions via HSV thresholding.
    
    Red wraps around H=0/180 in OpenCV, so we use 2 ranges.
    """
    
    # OpenCV HSV: H in [0, 179], S/V in [0, 255]
    # Red has two ranges due to wraparound
    RED_HSV_LOW_1 = np.array([0, 80, 80])
    RED_HSV_HIGH_1 = np.array([10, 255, 255])
    RED_HSV_LOW_2 = np.array([165, 80, 80])
    RED_HSV_HIGH_2 = np.array([179, 255, 255])
    
    def __init__(
        self,
        min_area_pixels: int = 30,    # 过滤太小的检测 (噪点)
        max_area_pixels: int = 50000, # 过滤太大的检测 (背景误检)
    ):
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
    
    def detect(self, rgb: np.ndarray) -> Optional[dict]:
        """Detect largest red region.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
        
        Returns:
            dict with 'mask' (bool), 'bbox' (x1,y1,x2,y2), 'centroid' (x,y), 'area'
            None if no valid detection.
        """
        assert rgb.dtype == np.uint8, f"expected uint8, got {rgb.dtype}"
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected (H,W,3), got {rgb.shape}"
        
        # RGB → HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # 2 ranges for red
        mask1 = cv2.inRange(hsv, self.RED_HSV_LOW_1, self.RED_HSV_HIGH_1)
        mask2 = cv2.inRange(hsv, self.RED_HSV_LOW_2, self.RED_HSV_HIGH_2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphology cleanup (close small holes, remove small noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Connected components, find largest
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Skip label 0 (background)
        if num_labels <= 1:
            return None
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            return None
        
        # Largest component
        largest_idx = int(np.argmax(areas)) + 1   # +1 because we skipped bg
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]
        
        if not (self.min_area_pixels <= largest_area <= self.max_area_pixels):
            return None
        
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        bbox = (int(x), int(y), int(x + w), int(y + h))
        centroid = (float(centroids[largest_idx, 0]), float(centroids[largest_idx, 1]))
        
        # Component-only mask
        component_mask = (labels == largest_idx).astype(bool)
        
        return {
            "mask": component_mask,
            "bbox": bbox,
            "centroid": centroid,
            "area": int(largest_area),
        }


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
        hsv_max_area: int = 50000,
    ):
        self.use_sam2 = use_sam2
        self.hsv_detector = HSVRedDetector(
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
        # Stage 1: HSV
        hsv_result = self.hsv_detector.detect(rgb)
        if hsv_result is None:
            return None
        
        # Stage 2: SAM2 refine (optional)
        if self.use_sam2:
            try:
                refined_mask = self.sam2_refiner.refine(
                    rgb,
                    bbox=hsv_result["bbox"],
                    centroid=hsv_result["centroid"],
                )
                detection_method = "hsv+sam2"
            except Exception as e:
                print(f"[GroundingPipeline] SAM2 failed: {e}, fallback to HSV")
                refined_mask = hsv_result["mask"]
                detection_method = "hsv-only-fallback"
        else:
            refined_mask = hsv_result["mask"]
            detection_method = "hsv-only"
        
        # Stage 3+4: 3D reconstruction
        try:
            pos_cam, pos_world = mask_to_3d_position(
                mask=refined_mask,
                depth_meters=depth_meters,
                K=K,
                T_world_camera=T_world_camera,
            )
        except ValueError as e:
            print(f"[GroundingPipeline] 3D reconstruction failed: {e}")
            return None
        
        # Re-compute bbox from refined mask
        ys, xs = np.where(refined_mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        
        # Confidence: ratio of refined mask pixels to bbox area (compactness proxy)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        confidence = float(refined_mask.sum() / max(bbox_area, 1))
        
        return GroundedObject(
            object_id=object_id,
            name="red_cube",
            bbox=bbox,
            mask=refined_mask,
            pos_camera=pos_cam,
            pos_world=pos_world,
            confidence=confidence,
            attributes={"color": "red"},
            is_target=True,
            detection_method=detection_method,
        )
