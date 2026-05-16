"""Data structures for perception module.

Aligned with v2.2 SEL design doc §4 (Semantic Grounding).
GroundedObject is the unit of structured perception output, consumed by:
  - M1.3 Tracker (for smoothing / re-id)
  - M1.4 Selector (for skill choice)
  - M5 Orchestrator (for handoff check)
  - π_arm policy (if vision-based)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class GroundedObject:
    """Single grounded object detection.
    
    Convention:
    - All positions in METERS
    - pos_camera: in camera optical frame (Z forward, X right, Y down)
    - pos_world: in sim world frame (= base frame, since base is fixed at origin)
    - mask: bool array, True = object pixels
    - bbox: (x1, y1, x2, y2) pixel coordinates, integer
    """
    object_id: int                          # unique ID, M1.3 用来 track
    name: str                               # "red_cube" 等语义标签
    bbox: tuple[int, int, int, int]         # (x1, y1, x2, y2) in image pixels
    mask: np.ndarray                        # (H, W) bool, True = object
    pos_camera: np.ndarray                  # (3,) float, in camera optical frame, meters
    pos_world: np.ndarray                   # (3,) float, in world/base frame, meters
    confidence: float                       # 0.0 - 1.0
    
    # Optional fields (Phase B 扩展时填)
    attributes: dict = field(default_factory=dict)   # {"color": "red", "size_cm": 3, ...}
    is_target: bool = True                  # 这个 object 是否是当前 target
    detection_method: str = "hsv+sam2"      # 检测方式 (debug/logging)
    
    def __repr__(self) -> str:
        return (
            f"GroundedObject(name='{self.name}', "
            f"pos_world={self.pos_world.round(3).tolist()}, "
            f"bbox={self.bbox}, "
            f"conf={self.confidence:.2f})"
        )
