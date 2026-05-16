"""Semantic Executive Layer - Perception Module (M1.2)

Implements grounding (HSV + SAM2) for target detection in head_camera RGB-D.
Outputs GroundedObject for downstream RL policy / S predictor.
"""
from xlerobot_rl.perception.data_types import GroundedObject
from xlerobot_rl.perception.grounding import (
    HSVRedDetector,
    SAM2Refiner,
    GroundingPipeline,
)

__all__ = [
    "GroundedObject",
    "HSVRedDetector",
    "SAM2Refiner",
    "GroundingPipeline",
]
