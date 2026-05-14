"""诊断: random action 跑完, robot 各部分有没有动?"""
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa


env = gym.make(
    "Empty-v1",
    robot_uids="xlerobot",
    num_envs=1,
    obs_mode="state",
    sim_backend="gpu",
).unwrapped

obs, _ = env.reset(seed=0)
robot = env.agent.robot

# 初始 joint positions
qpos_init = robot.get_qpos()[0].cpu().numpy()
print(f"\nInitial qpos shape: {qpos_init.shape}")

# 跑 100 步
for _ in range(100):
    action = env.action_space.sample()
    env.step(action)

# 末态 joint positions
qpos_end = robot.get_qpos()[0].cpu().numpy()

# Diff
diff = qpos_end - qpos_init
print(f"\n{'Joint':30s} {'Init':>10s} {'End':>10s} {'Delta':>10s} {'Moved?':>8s}")
print("-" * 75)
for i, j in enumerate(robot.get_active_joints()):
    name = j.name[0] if isinstance(j.name, list) else j.name
    init = qpos_init[i]
    end = qpos_end[i]
    delta = diff[i]
    moved = "YES" if abs(delta) > 0.01 else "no"
    print(f"{str(name):30s} {init:>10.4f} {end:>10.4f} {delta:>+10.4f} {moved:>8s}")

print(f"\n总 active joints: {len(robot.get_active_joints())}")
print(f"动了的 joints (|delta| > 0.01): {sum(1 for d in diff if abs(d) > 0.01)}")
