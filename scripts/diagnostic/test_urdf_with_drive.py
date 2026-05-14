"""验证: 给 arm joints 加 PD drive 后, 机器人能否保持姿态不下垂。

如果加上 drive 后 arm 不再下垂, 证明 URDF 本身 OK,
只是缺 stiffness 配置 (这个由 ManiSkill controller 在运行时设置)。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sapien
from sapien import Pose


URDF_PATH = Path(__file__).parent.parent / "xlerobot_rl/sim/assets/urdf/xlerobot.urdf"


def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    robot = loader.load(str(URDF_PATH))
    robot.set_pose(Pose([0, 0, 0]))

    # ==== 关键: 给所有 arm joints 加 PD drive ====
    arm_joint_names = [
        "Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L",
        "Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R", "Jaw_R",
        "head_pan_joint", "head_tilt_joint",
    ]

    joint_lookup = {j.name: j for j in robot.get_active_joints()}

    print("--- Setting up PD drive on arm joints ---")
    for name in arm_joint_names:
        j = joint_lookup[name]
        # 模拟 ManiSkill controller config 的典型值
        # arm_stiffness=2e4, arm_damping=1e2 (from ManiSkill xlerobot.py)
        j.set_drive_property(stiffness=2e4, damping=1e2, force_limit=250)
        j.set_drive_target(0.0)  # 目标位置: 0
        print(f"  {name:25s} stiff=2e4  damp=1e2  target=0.0")

    # 让 wheels 也加一点 damping 防止滑 (不加 drive, 让它能自由转)
    for name in ["left_wheel_joint", "right_wheel_joint"]:
        j = joint_lookup[name]
        j.set_drive_property(stiffness=0, damping=5.0, force_limit=100)
        # 不 set_drive_target

    # ==== 模拟 500 步 ====
    print("\n--- Simulating 500 steps with PD drive ---")
    print(f"  {'step':>5s} | {'base dxyz':>30s} | {'Pitch_L':>10s} | {'Pitch_R':>10s}")

    init_qpos = robot.get_qpos().copy()
    init_pose = robot.get_pose()

    pitch_l_idx = robot.get_active_joints().index(joint_lookup["Pitch_L"])
    pitch_r_idx = robot.get_active_joints().index(joint_lookup["Pitch_R"])

    for step in range(501):
        scene.step()
        if step % 50 == 0:
            qpos = robot.get_qpos()
            base_p = robot.get_pose().p
            base_d = base_p - init_pose.p
            print(f"  {step:>5d} | "
                  f"dx={base_d[0]:+.4f} dy={base_d[1]:+.4f} dz={base_d[2]:+.4f} | "
                  f"{qpos[pitch_l_idx]:+10.4f} | "
                  f"{qpos[pitch_r_idx]:+10.4f}")

    # 总结
    final_qpos = robot.get_qpos()
    qpos_diff = final_qpos - init_qpos
    print(f"\n--- After 500 steps ---")
    print(f"  Max joint drift: {np.max(np.abs(qpos_diff)):.4f} rad")
    print(f"  Pitch_L drift: {qpos_diff[pitch_l_idx]:.4f} rad  (理想应 < 0.01)")
    print(f"  Pitch_R drift: {qpos_diff[pitch_r_idx]:.4f} rad  (理想应 < 0.01)")

    if np.max(np.abs(qpos_diff)) < 0.05:
        print("\n✓ SUCCESS: URDF + PD drive 配合工作, 机器人稳定. URDF 本身没问题, ManiSkill 接入后会通过 controller 解决.")
    else:
        print("\n✗ Still drifting. URDF 可能有更深层问题 (mass/inertia/CoM 配置错误).")

    # 弹 viewer 让你看效果
    print("\nOpening viewer...")
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-1.5, y=0, z=1.5)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
