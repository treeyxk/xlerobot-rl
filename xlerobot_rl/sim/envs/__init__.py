"""Side-effect import to register all custom envs with gymnasium."""
from xlerobot_rl.sim.envs import static_arm_grasp  # noqa: F401
from xlerobot_rl.sim.envs import target_conditioned_arm_grasp  # noqa: F401

__all__ = []
