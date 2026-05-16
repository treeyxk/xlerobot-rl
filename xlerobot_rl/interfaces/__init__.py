"""xlerobot_rl.interfaces: M0 接口契约的代码实现。

Usage:
    from xlerobot_rl.interfaces import (
        TargetObservation, RobotObservation, CameraData, RobotProprioception,
        WholeBodyAction, BaseSubGoalAction, BaseVelocityAction,
        ArmJointAction, ArmEEAction, ActionBounds,
        RobotInterface, CameraInterface,
        TargetState, SemanticExecutiveState,
    )
"""

from xlerobot_rl.interfaces.observation import (
    TargetObservation,
    RobotObservation,
    CameraData,
    RobotProprioception,
)
from xlerobot_rl.interfaces.action import (
    WholeBodyAction,
    BaseVelocityAction,
    BaseSubGoalAction,
    ArmJointAction,
    ArmEEAction,
    ActionBounds,
)
from xlerobot_rl.interfaces.robot import RobotInterface
from xlerobot_rl.interfaces.camera import CameraInterface
from xlerobot_rl.interfaces.semantic import (
    TargetState,
    SemanticExecutiveState,
)

__all__ = [
    "TargetObservation",
    "RobotObservation",
    "CameraData",
    "RobotProprioception",
    "WholeBodyAction",
    "BaseVelocityAction",
    "BaseSubGoalAction",
    "ArmJointAction",
    "ArmEEAction",
    "ActionBounds",
    "RobotInterface",
    "CameraInterface",
    "TargetState",
    "SemanticExecutiveState",
]
