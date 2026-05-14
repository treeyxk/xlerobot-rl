"""Day 4 验证脚本: 在 sim 里加载 XLeRobot, 弹 viewer 看, 验证 joints + camera。

期望产出:
    1. GUI 窗口里 XLeRobot 动起来
    2. Terminal 输出所有 joint 信息
    3. Head camera 渲染一张 RGB 保存
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv


DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _to_scalar(x):
    """把 ManiSkill 各种 batched 返回值转成单值,方便打印。"""
    if isinstance(x, (list, tuple)):
        return x[0] if x else x
    if hasattr(x, "cpu"):  # torch tensor
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.flatten()[0] if x.size == 1 else x
    return x


def main():
    env: BaseEnv = gym.make(
        "Empty-v1",
        robot_uids="xlerobot",
        num_envs=1,
        obs_mode="rgb",
        sim_backend="gpu",
        render_mode="human",
    ).unwrapped

    obs, _ = env.reset(seed=0)
    print(f"\n✓ Env reset OK")

    robot = env.agent.robot
    print(f"\n✓ Robot loaded: {robot.name}")
    print(f"  Total links: {len(robot.get_links())}")
    print(f"  Total joints: {len(robot.get_joints())}")
    print(f"  Active joints (controllable): {len(robot.get_active_joints())}")

    print("\n--- Active joints ---")
    for j in robot.get_active_joints():
        try:
            name = _to_scalar(j.name)
            jtype = _to_scalar(j.type)
            limits = j.get_limits()
            # limits 可能是 tensor (B, 2), 取第一个 env
            if hasattr(limits, "cpu"):
                limits = limits.cpu().numpy()
            if isinstance(limits, np.ndarray) and limits.ndim >= 2:
                limits = limits[0]
            print(f"  {str(name):30s} type={str(jtype):12s} limits={limits}")
        except Exception as e:
            print(f"  {j!r} -- {e}")

    # ---- Camera 信息 + 保存一张 head camera RGB ----
    print("\n--- Cameras (from obs) ---")
    if "sensor_data" in obs:
        for cam_name, cam_data in obs["sensor_data"].items():
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                # shape (B, H, W, 3) -> 取第 0 个 env
                if rgb.ndim == 4:
                    rgb = rgb[0]
                from PIL import Image
                save_path = DEBUG_DIR / f"{cam_name}.png"
                Image.fromarray(rgb.astype(np.uint8)).save(save_path)
                print(f"  {cam_name}: shape={rgb.shape}, saved -> {save_path}")
    else:
        print("  (no sensor_data in obs - check obs_mode)")

    # ---- 跑 100 步 random action 让 joints 动起来 ----
    print(f"\n--- Action space: {env.action_space} ---")
    print("Running 100 random-action steps. Watch the viewer.")
    for i in range(100):
        action = env.action_space.sample()
        env.step(action)
        env.render_human()

    print("\n✓ All checks passed.  Close the viewer window to exit.")
    while True:
        env.render_human()


if __name__ == "__main__":
    main()
