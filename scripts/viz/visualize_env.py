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
        obs_mode="rgbd",
        sim_backend="gpu",
        render_mode="human",
    ).unwrapped

    obs, _ = env.reset(seed=0)
    print(f"✓ Env reset OK, obs_mode={env.obs_mode}")

    # 检查 sensor data
    if "sensor_data" in obs:
        for cam_name, cam_data in obs["sensor_data"].items():
            print(f"  Camera '{cam_name}':")
            for key, value in cam_data.items():
                if hasattr(value, "shape"):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    {key}: {value}")

    # 保存一张 head_camera RGB 和 depth 图到 data/debug
    import os
    os.makedirs("data/debug", exist_ok=True)
    import cv2
    import numpy as np

    head_rgb = obs["sensor_data"]["head_camera"]["rgb"][0].cpu().numpy()
    head_depth = obs["sensor_data"]["head_camera"]["depth"][0].cpu().numpy()

    # RGB
    cv2.imwrite("data/debug/head_rgb_d415.png", cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR))

    # Depth 可视化 (归一化到 0-255)
    depth_vis = (head_depth.squeeze() / head_depth.max() * 255).astype(np.uint8)
    cv2.imwrite("data/debug/head_depth_d415.png", depth_vis)
    print(f"\n✓ Saved RGB to data/debug/head_rgb_d415.png")
    print(f"✓ Saved depth to data/debug/head_depth_d415.png")
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
