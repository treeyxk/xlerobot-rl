"""Observation schemas: 模块间传递的所有 obs 数据结构定义。

设计原则:
- 所有 obs 都是 dataclass, 自带 type hints
- 单位严格遵循 docs/interface_contract.md §2
- 坐标系命名遵循 §1
- 所有 timestamps 是 UTC seconds (float)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np


# 单位说明 (与 interface_contract.md §2 一致):
#   - 长度: 米 (m)
#   - 角度: 弧度 (rad)
#   - 时间: 秒 (s)
#   - 图像坐标: 像素 (int)


@dataclass
class TargetObservation:
    """VLM 输出的目标物体观测。

    所有下游模块 (π_nav / π_arm / S predictor / orchestrator)
    都消费这个 schema。是 sim 和 real 的统一接口。
    """

    # 视觉
    mask: np.ndarray              # (H, W) bool 或 uint8, binary segmentation
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels

    # 3D 位置
    pos_base: np.ndarray          # (3,) xyz in base frame, meters
    pos_camera: np.ndarray        # (3,) xyz in head_cam frame, meters

    # 元信息
    confidence: float             # [0, 1]
    timestamp: float              # UTC seconds
    source: Literal["sim_gt", "vlm", "tracked"] = "vlm"
    category: str = "target"      # 物体类别 (red_cube, cup, ...)

    # 可选: 完整 6D pose (有些 VLM/算法能输出, mask 反投影没有)
    pose_base: Optional[np.ndarray] = None  # (4, 4) SE(3) 矩阵

    def __post_init__(self):
        # 基本 shape 校验, 防止下游崩
        assert self.mask.ndim == 2, f"mask must be 2D, got shape {self.mask.shape}"
        assert self.pos_base.shape == (3,), f"pos_base must be (3,), got {self.pos_base.shape}"
        assert self.pos_camera.shape == (3,), f"pos_camera must be (3,), got {self.pos_camera.shape}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence out of range: {self.confidence}"
        if self.pose_base is not None:
            assert self.pose_base.shape == (4, 4), f"pose_base must be (4, 4)"


@dataclass
class RobotProprioception:
    """机器人本体感知 (joint states + base odometry)。

    Sim 和 real 都遵循这个 schema。
    """

    # Base
    base_pose: np.ndarray         # (3,) [x_world, y_world, theta_world], meters & rad

    # Arms (双臂; 单臂使用时另一个填零或 None)
    arm_qpos_left: np.ndarray     # (6,) joint angles in rad
    arm_qpos_right: Optional[np.ndarray] = None

    arm_qvel_left: Optional[np.ndarray] = None    # (6,) joint velocities
    arm_qvel_right: Optional[np.ndarray] = None

    # Gripper (0 = closed, 1 = open)
    gripper_left: float = 0.0
    gripper_right: float = 0.0

    timestamp: float = 0.0

    def __post_init__(self):
        assert self.base_pose.shape == (3,), f"base_pose must be (3,), got {self.base_pose.shape}"
        assert self.arm_qpos_left.shape == (6,), f"arm_qpos_left must be (6,)"
        if self.arm_qpos_right is not None:
            assert self.arm_qpos_right.shape == (6,)


@dataclass
class CameraData:
    """单个相机的一帧数据。"""

    rgb: np.ndarray               # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None      # (H, W) float32, meters, NaN = invalid
    intrinsics: Optional[np.ndarray] = None  # (3, 3) camera intrinsic matrix
    frame_name: str = ""          # "head_cam" / "wrist_cam_left" / ...
    timestamp: float = 0.0

    def __post_init__(self):
        assert self.rgb.ndim == 3 and self.rgb.shape[2] == 3
        if self.depth is not None:
            assert self.depth.shape == self.rgb.shape[:2]


@dataclass
class RobotObservation:
    """完整的机器人观测 (一个 step 的所有传感器数据)。

    这是 sim env 和 real robot 都返回的统一 obs container。
    """

    proprioception: RobotProprioception
    cameras: dict[str, CameraData] = field(default_factory=dict)
    # 例: cameras["head_cam"], cameras["wrist_cam_left"]

    target: Optional[TargetObservation] = None  # 可能为 None (VLM 没检测到)

    # Extra info (debugging / sim-only ground truth)
    extra: dict = field(default_factory=dict)

    timestamp: float = 0.0