"""可视化 StaticArmGrasp-v0 env, 用 random action 跑 + 弹 viewer 看。

用法:
    python scripts/viz/visualize_env.py
"""
from __future__ import annotations
import gymnasium as gym

import mani_skill.envs  # noqa: F401
import xlerobot_rl.sim.envs  # noqa: F401, 注册自定义 envs


def main():
    env = gym.make(
        "StaticArmGrasp-v0",
        num_envs=1,
        obs_mode="rgb",
        sim_backend="gpu",
        render_mode="human",
    ).unwrapped

    obs, _ = env.reset(seed=0)
    print(f"✓ Env reset OK")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs keys: {list(obs.keys())}")
    print(f"  Sensor data keys: {list(obs.get('sensor_data', {}).keys())}")

    print(f"\nRunning random actions. Watch the viewer.")
    print(f"按 ESC 或关窗口退出.")

    for step in range(200):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        env.render_human()

        if step % 50 == 0:
            # rew 是 tensor
            rew_val = rew.item() if hasattr(rew, "item") else float(rew)
            print(f"  step={step}, reward={rew_val:+.3f}, "
                  f"is_grasped={info['is_grasped'][0].item()}, "
                  f"lift_height={info['lift_height'][0].item():+.4f}m")

    print("\n✓ 跑完 200 步, env 稳定. 关闭 viewer 退出.")
    while True:
        env.render_human()


if __name__ == "__main__":
    main()
