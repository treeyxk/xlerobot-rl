"""跑 N 次 episode (random action) 看 env 不崩 + success rate baseline。

用法:
    python scripts/eval/eval_env_random.py --n-episodes 50

预期 random action 的 success rate < 5%, 这是个 baseline.
Day 6 我们写 Oracle scripted policy, 目标 ≥ 80%.
"""
from __future__ import annotations
import argparse
import time

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
import xlerobot_rl.sim.envs  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=4, help="parallel envs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        "StaticArmGrasp-v0",
        num_envs=args.num_envs,
        obs_mode="state",   # 不渲染 camera 加速
        sim_backend="gpu",
    ).unwrapped

    n_episodes_per_env = (args.n_episodes + args.num_envs - 1) // args.num_envs
    # max_episode_steps 在 ManiSkill 不一定通过 env.spec 暴露
    # 直接读 register_env 的元数据, 失败就 hardcode 64
    max_steps = getattr(env, "max_episode_steps", None)
    if max_steps is None:
        max_steps = 64  # 跟 @register_env 注册时一致
    print(f"Running {n_episodes_per_env} episodes × {args.num_envs} envs = "
          f"{n_episodes_per_env * args.num_envs} total episodes")
    print(f"max_episode_steps = {max_steps}")

    successes = []
    final_lift_heights = []
    final_grasps = []

    t0 = time.time()
    for ep in range(n_episodes_per_env):
        obs, _ = env.reset(seed=args.seed + ep * 1000)

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)

        # 取每个 env 的最终状态
        successes.extend(info["success"].cpu().numpy().tolist())
        final_lift_heights.extend(info["lift_height"].cpu().numpy().tolist())
        final_grasps.extend(info["is_grasped"].cpu().numpy().tolist())

    elapsed = time.time() - t0
    n_total = len(successes)
    succ_rate = np.mean(successes) * 100
    grasp_rate = np.mean(final_grasps) * 100
    avg_lift = np.mean(final_lift_heights)

    print(f"\n{'='*50}")
    print(f"Results over {n_total} episodes (random actions):")
    print(f"  Success rate:        {succ_rate:.1f}%")
    print(f"  Final grasp rate:    {grasp_rate:.1f}%")
    print(f"  Avg final lift:      {avg_lift:+.4f}m")
    print(f"  Wall time:           {elapsed:.1f}s ({n_total/elapsed:.1f} ep/s)")
    print(f"{'='*50}")
    print(f"\nRandom baseline 预期 success < 5%. Day 6 Oracle 目标 ≥ 80%.")


if __name__ == "__main__":
    main()
