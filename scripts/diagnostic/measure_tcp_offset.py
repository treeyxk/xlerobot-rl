"""测量 + 验证 TCP local offset。

策略:
1. 加载 URDF, 让 Fixed_Jaw 处于一个 known pose
2. 加载 Moving_Jaw 也在 known pose
3. 计算两个 jaw 几何中心 (实际 grasp 接触点) 在 Fixed_Jaw frame 下的位置
4. 这就是 TCP local offset
5. 弹 viewer, 用一个小红球标记 TCP 位置, 肉眼验证是否在两指中间
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sapien
from sapien import Pose


URDF_PATH = Path(__file__).parent.parent.parent / "xlerobot_rl/sim/assets/urdf/xlerobot.urdf"
MARKER_RADIUS = 0.01
# TCP offset in Fixed_Jaw / Fixed_Jaw_2 local frame.
# This matches the grasp_demo kinematics constants.
TCP_OFFSET_IN_FIXED_JAW = np.array([0.0, -0.1070, 0.0])
print(f"URDF: {URDF_PATH}")
assert URDF_PATH.exists()


def transform_inverse(T):
    """SE(3) 4x4 矩阵求逆 (rotation transpose + translation 调整)。"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def create_tcp_marker(
    scene: sapien.Scene,
    name: str,
    position: np.ndarray,
    color: list[float],
    radius: float = MARKER_RADIUS,
):
    """Create a bright visual marker at a TCP candidate position."""
    material = sapien.render.RenderMaterial(base_color=color)
    material.emission = color

    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=material)
    builder.initial_pose = sapien.Pose(p=position)
    marker = builder.build_kinematic(name=name)
    marker.set_pose(sapien.Pose(p=position))
    return marker


def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_ground(0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True  # ← 固定 base 在原点, 让我们好定位
    robot = loader.load(str(URDF_PATH))
    robot.set_pose(Pose([0, 0, 0]))

    # 给 arm joints 加 PD drive 让它保持姿态
    arm_joints = [
        "Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L",
        "Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R", "Jaw_R",
        "head_pan_joint", "head_tilt_joint",
    ]
    joint_lookup = {j.name: j for j in robot.get_active_joints()}
    for name in arm_joints:
        joint_lookup[name].set_drive_property(stiffness=2e4, damping=1e2, force_limit=250)
        joint_lookup[name].set_drive_target(0.0)

    # ==== 让 gripper 处于闭合状态, 方便比对两指之间的 TCP ====
    # 在 URDF 里 Jaw 是 revolute, 不是 prismatic
    # Jaw_R 限制是 [0, 1.7], 0 = 闭合, 1.7 = 完全张开
    joint_lookup["Jaw_R"].set_drive_target(0.0)
    joint_lookup["Jaw_L"].set_drive_target(0.0)

    # 让 physics 稳定下来
    for _ in range(200):
        scene.step()

    # ==== 取出两个 jaw link 的 world pose ====
    links_map = {l.name: l for l in robot.get_links()}
    fixed_jaw_R = links_map["Fixed_Jaw_2"]
    moving_jaw_R = links_map["Moving_Jaw_2"]

    fixed_pose = fixed_jaw_R.get_pose()  # sapien.Pose
    moving_pose = moving_jaw_R.get_pose()

    print(f"\n--- Fixed_Jaw_2 (right arm fixed finger) world pose ---")
    print(f"  position: {fixed_pose.p}")
    print(f"  quat (wxyz): {fixed_pose.q}")

    print(f"\n--- Moving_Jaw_2 (right arm moving finger) world pose ---")
    print(f"  position: {moving_pose.p}")
    print(f"  quat (wxyz): {moving_pose.q}")

    # ==== 计算 TCP: grasp_demo 使用的 Fixed_Jaw 局部 offset ====
    T_world_fixed = fixed_pose.to_transformation_matrix()
    tcp_world = (T_world_fixed @ np.r_[TCP_OFFSET_IN_FIXED_JAW, 1.0])[:3]
    print(f"\n--- TCP from Fixed_Jaw_2 local offset ---")
    print(f"  TCP local offset: {TCP_OFFSET_IN_FIXED_JAW}")
    print(f"  TCP world pos:    {tcp_world}")

    # ==== 转换 TCP 到 Fixed_Jaw 的 local frame ====
    # T_world_fixed = fixed_pose
    # T_fixed_tcp = T_world_fixed^-1 @ T_world_tcp
    T_fixed_world = transform_inverse(T_world_fixed)

    tcp_world_h = np.array([tcp_world[0], tcp_world[1], tcp_world[2], 1.0])
    tcp_in_fixed = (T_fixed_world @ tcp_world_h)[:3]

    print(f"\n*** TCP local offset in Fixed_Jaw_2 frame: {tcp_in_fixed} ***")

    # ==== 同样测左臂 ====
    fixed_jaw_L = links_map["Fixed_Jaw"]
    fp_L = fixed_jaw_L.get_pose()
    T_world_fixed_L = fp_L.to_transformation_matrix()
    tcp_L_world = (T_world_fixed_L @ np.r_[TCP_OFFSET_IN_FIXED_JAW, 1.0])[:3]

    T_fixed_L_world = transform_inverse(T_world_fixed_L)
    tcp_L_h = np.array([tcp_L_world[0], tcp_L_world[1], tcp_L_world[2], 1.0])
    tcp_in_fixed_L = (T_fixed_L_world @ tcp_L_h)[:3]
    print(f"\n*** 左臂 TCP local offset in Fixed_Jaw frame: {tcp_in_fixed_L} ***")
    print(f"    (两臂应该一致, 因为是镜像的同款 SO101)")

    # ==== 在 TCP 位置画 marker, 用 viewer 看 ====
    # 2cm marker + emission 比 5mm 普通材质更适合肉眼验证。
    tcp_marker = create_tcp_marker(
        scene,
        name="tcp_marker_right",
        position=tcp_world,
        color=[1.0, 0.0, 0.0, 1.0],
    )
    tcp_marker_L = create_tcp_marker(
        scene,
        name="tcp_marker_left",
        position=tcp_L_world,
        color=[0.0, 1.0, 0.0, 1.0],
    )

    print(f"\n--- Marker poses ---")
    print(f"  right marker pose: {tcp_marker.get_pose().p}, radius={MARKER_RADIUS}m")
    print(f"  left marker pose:  {tcp_marker_L.get_pose().p}, radius={MARKER_RADIUS}m")

    print(f"\n--- Opening viewer ---")
    print(f"  红球 = 右臂 TCP, 绿球 = 左臂 TCP")
    print(f"  TCP offset = {TCP_OFFSET_IN_FIXED_JAW}")
    print(f"  如果在指尖外或者远离 gripper, 说明 TCP offset 算错了")

    viewer = scene.create_viewer()
    # 原来 x=-0.5 离 TCP 只有约 13cm, 两个 marker 很容易被近裁剪或落到视野外。
    viewer.set_camera_xyz(x=-1.2, y=0.0, z=1.25)
    viewer.set_camera_rpy(r=0, p=-0.25, y=0)
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    main()
