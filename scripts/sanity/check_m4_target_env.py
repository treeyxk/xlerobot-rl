"""Smoke check for TargetConditionedArmGrasp-v0.

This script verifies that the M4 target-conditioned env exposes the expected
extra observation fields and wrong-object evaluation metrics.
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401

import xlerobot_rl.sim.envs  # noqa: F401


OBS_KEYS = [
    "target_pos_base",
    "target_color_id",
    "distractor_pos_base",
    "distractor_color_ids",
    "skill_id",
    "tcp_to_target",
]

INFO_KEYS = [
    "target_is_grasped",
    "target_lift_height",
    "wrong_object_grasped",
    "wrong_object_lifted",
    "wrong_object_failure",
    "success",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.target_color is not None:
        kwargs["target_color"] = args.target_color

    env = gym.make(
        "TargetConditionedArmGrasp-v0",
        num_envs=args.num_envs,
        obs_mode="state_dict",
        sim_backend=args.sim_backend,
        **kwargs,
    ).unwrapped

    obs, _ = env.reset(seed=args.seed)
    print(f"target_color: {env.target_color}")
    print(f"extra keys: {sorted(obs['extra'].keys())}")

    missing_obs = [key for key in OBS_KEYS if key not in obs["extra"]]
    if missing_obs:
        print(f"missing obs keys: {missing_obs}")
        return 1

    print("\nextra observation fields:")
    for key in OBS_KEYS:
        value = obs["extra"][key]
        print(f"  {key}: shape={tuple(value.shape)}, value={value}")

    action = env.action_space.sample()
    _, reward, terminated, truncated, info = env.step(action)

    missing_info = [key for key in INFO_KEYS if key not in info]
    if missing_info:
        print(f"missing info keys: {missing_info}")
        return 1

    print("\none-step result:")
    print(f"  reward: {reward}")
    print(f"  terminated: {terminated}")
    print(f"  truncated: {truncated}")
    for key in INFO_KEYS:
        print(f"  {key}: {info[key]}")

    print("\nM4 target-conditioned env check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
