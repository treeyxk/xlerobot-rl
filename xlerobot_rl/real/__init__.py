"""Real hardware utilities."""

from xlerobot_rl.real.camera_geometry import RealCameraGeometry
from xlerobot_rl.real.red_cube_detector import detect_red_cube_bgr, detect_red_cube_rgbd

__all__ = [
    "RealCameraGeometry",
    "detect_red_cube_bgr",
    "detect_red_cube_rgbd",
]
