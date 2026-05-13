r"""CameraInterface ABC: sim 和 real 相机的统一接口。"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from xlerobot_rl.interfaces.observation import CameraData


class CameraInterface(ABC):
    """相机统一接口。"""

    @abstractmethod
    def get_frame(self) -> CameraData:
        """获取当前一帧。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭相机。"""
        ...

    @property
    @abstractmethod
    def frame_name(self) -> str:
        """相机的坐标系名 (如 'head_cam')。"""
        ...

    @property
    @abstractmethod
    def has_depth(self) -> bool:
        """是否提供深度数据。"""
        ...