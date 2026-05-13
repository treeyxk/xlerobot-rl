"""RobotInterface ABC: sim 和 real 共享的统一接口。

sim 和 real 的具体实现 (SimRobot, RealRobot) 都继承这个 ABC。
上层代码 (orchestrator, evaluation harness) 只 import 这个 ABC, 不关心是 sim 还是 real。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from xlerobot_rl.interfaces.observation import RobotObservation
from xlerobot_rl.interfaces.action import WholeBodyAction


class RobotInterface(ABC):
    """机器人统一接口 (sim / real 都遵守)。

    设计上类似 gym.Env 但精简, 没有 reward / done / info (那些由 task env 包装)。

    使用方式:
        robot = SimRobot(env)   # 或 RealRobot()
        obs = robot.reset()
        for _ in range(N):
            action = policy(obs)
            obs = robot.step(action)
    """

    @abstractmethod
    def reset(self) -> RobotObservation:
        """重置机器人到初始状态, 返回第一帧 observation。"""
        ...

    @abstractmethod
    def step(self, action: WholeBodyAction) -> RobotObservation:
        """执行 action, 返回下一帧 observation。"""
        ...

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """获取当前 observation, 不 step。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭机器人 (释放硬件 / 关闭 sim env)。"""
        ...

    @property
    @abstractmethod
    def control_freq_hz(self) -> float:
        """控制频率, sim 和 real 可能不同。"""
        ...

    @property
    @abstractmethod
    def is_real(self) -> bool:
        """True 表示真机, False 表示 sim。"""
        ...