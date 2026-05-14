"""Sanity check: 所有依赖能 import + GPU 可用 + ManiSkill 能跑。"""
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

import mani_skill
import gymnasium as gym
print(f"ManiSkill: {mani_skill.__version__}")

import lerobot
print(f"LeRobot imported OK")

# 跑一个 ManiSkill env
env = gym.make("PickCube-v1", num_envs=4, obs_mode="rgb")
obs, _ = env.reset()
print(f"ManiSkill env reset OK, obs keys: {list(obs.keys())}")

import stable_baselines3
print(f"SB3: {stable_baselines3.__version__}")

print("\n✓ All systems go")
