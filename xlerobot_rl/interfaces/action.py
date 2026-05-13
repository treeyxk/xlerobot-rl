"""Action schemas + bounds 约束。

所有 policy 输出的 action 都通过这里定义的 dataclass 传递。
Bounds 与 docs/interface_contract.md §6 (TBD) 对应, 当前为草案。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np


# ============================================================
# Action bounds (草案, 见 interface_contract.md §6 TBD)
# 数值上 TBD, 但 schema 先冻结
# ============================================================

class ActionBounds:
    """硬性 action bounds, 用于 clamp 任何 policy 输出。"""

    # Base velocity (m/s, rad/s)
    BASE_VX_MAX = 0.3
    BASE_VY_MAX = 0.3
    BASE_OMEGA_MAX = 0.5

    # Base sub-goal (in base frame, meters / rad)
    BASE_DX_MAX = 0.3
    BASE_DY_MAX = 0.3
    BASE_DTHETA_MAX = 0.5

    # Arm joint delta (rad/step)
    ARM_JOINT_DELTA_MAX = 0.1

    # Arm EE delta (meters / rad per step)
    EE_POS_DELTA_MAX = 0.05
    EE_ROT_DELTA_MAX = 0.1

    # Gripper (continuous [0, 1])
    GRIPPER_MIN = 0.0
    GRIPPER_MAX = 1.0


# ============================================================
# Action dataclasses
# ============================================================

@dataclass
class BaseVelocityAction:
    """底盘速度控制 (low-level)。"""
    vx: float = 0.0   # m/s, in base frame
    vy: float = 0.0   # m/s
    omega: float = 0.0  # rad/s

    def clip(self) -> "BaseVelocityAction":
        return BaseVelocityAction(
            vx=float(np.clip(self.vx, -ActionBounds.BASE_VX_MAX, ActionBounds.BASE_VX_MAX)),
            vy=float(np.clip(self.vy, -ActionBounds.BASE_VY_MAX, ActionBounds.BASE_VY_MAX)),
            omega=float(np.clip(self.omega, -ActionBounds.BASE_OMEGA_MAX, ActionBounds.BASE_OMEGA_MAX)),
        )


@dataclass
class BaseSubGoalAction:
    """底盘 sub-goal 控制 (mid-level, π_nav 推荐输出)。

    底层 controller 跟踪到这个 (dx, dy, dtheta) 偏移点。
    """
    dx: float = 0.0       # meters, in current base frame
    dy: float = 0.0       # meters
    dtheta: float = 0.0   # rad

    def clip(self) -> "BaseSubGoalAction":
        return BaseSubGoalAction(
            dx=float(np.clip(self.dx, -ActionBounds.BASE_DX_MAX, ActionBounds.BASE_DX_MAX)),
            dy=float(np.clip(self.dy, -ActionBounds.BASE_DY_MAX, ActionBounds.BASE_DY_MAX)),
            dtheta=float(np.clip(self.dtheta, -ActionBounds.BASE_DTHETA_MAX, ActionBounds.BASE_DTHETA_MAX)),
        )


@dataclass
class ArmJointAction:
    """机械臂 joint-space action (delta)。"""
    delta_qpos: np.ndarray   # (6,) joint angle deltas in rad
    gripper: float = 0.0     # [0, 1], 0 = closed

    def __post_init__(self):
        assert self.delta_qpos.shape == (6,)

    def clip(self) -> "ArmJointAction":
        return ArmJointAction(
            delta_qpos=np.clip(
                self.delta_qpos,
                -ActionBounds.ARM_JOINT_DELTA_MAX,
                ActionBounds.ARM_JOINT_DELTA_MAX,
            ),
            gripper=float(np.clip(self.gripper, ActionBounds.GRIPPER_MIN, ActionBounds.GRIPPER_MAX)),
        )


@dataclass
class ArmEEAction:
    """机械臂 EE-space action (delta), sim2real 友好。"""
    delta_pos: np.ndarray    # (3,) xyz delta in ee frame, meters
    delta_rot: np.ndarray    # (3,) rpy delta in ee frame, rad
    gripper: float = 0.0     # [0, 1]

    def __post_init__(self):
        assert self.delta_pos.shape == (3,)
        assert self.delta_rot.shape == (3,)

    def clip(self) -> "ArmEEAction":
        return ArmEEAction(
            delta_pos=np.clip(self.delta_pos, -ActionBounds.EE_POS_DELTA_MAX, ActionBounds.EE_POS_DELTA_MAX),
            delta_rot=np.clip(self.delta_rot, -ActionBounds.EE_ROT_DELTA_MAX, ActionBounds.EE_ROT_DELTA_MAX),
            gripper=float(np.clip(self.gripper, ActionBounds.GRIPPER_MIN, ActionBounds.GRIPPER_MAX)),
        )


@dataclass
class WholeBodyAction:
    """完整一个 step 的 action: base + arm (单臂或双臂)。"""

    # Base 部分: BaseVelocityAction 或 BaseSubGoalAction, 二选一
    base: Optional[BaseVelocityAction | BaseSubGoalAction] = None

    # Arm 部分: ArmJointAction 或 ArmEEAction, 单臂或双臂
    arm_left: Optional[ArmJointAction | ArmEEAction] = None
    arm_right: Optional[ArmJointAction | ArmEEAction] = None

    def clip(self) -> "WholeBodyAction":
        return WholeBodyAction(
            base=self.base.clip() if self.base else None,
            arm_left=self.arm_left.clip() if self.arm_left else None,
            arm_right=self.arm_right.clip() if self.arm_right else None,
        )