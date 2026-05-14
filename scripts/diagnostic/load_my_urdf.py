"""Day 4 补做: 验证修改版 URDF 能被 SAPIEN/ManiSkill 加载。

这个脚本不走 ManiSkill agent 系统, 直接用 SAPIEN URDF loader,
能避开所有 joint naming / controller 配置的差异问题。
如果连这步都失败, 说明 URDF 本身有结构问题 (mesh 路径 / link 引用错 / 物理参数无效等)。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import sapien
from sapien import Pose


URDF_PATH = Path(__file__).parent.parent / "xlerobot_rl/sim/assets/urdf/xlerobot.urdf"
assert URDF_PATH.exists(), f"URDF not found: {URDF_PATH}"
print(f"URDF: {URDF_PATH}")
print(f"URDF size: {URDF_PATH.stat().st_size / 1024:.1f} KB")


def main():
    # 创建一个 SAPIEN scene
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)

    # 加一些光
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)

    # ---- 关键: 用 SAPIEN URDF loader 加载你的 URDF ----
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False   # base_footprint 是 root, 不固定它 (让机器人能动)
    loader.scale = 1.0

    try:
        robot = loader.load(str(URDF_PATH))
        print(f"\n✓ URDF loaded successfully")
    except Exception as e:
        print(f"\n✗ URDF load FAILED: {e}")
        raise

    # 设置初始 pose (让机器人站在地上)
    robot.set_pose(Pose([0, 0, 0]))

    # 检查 robot 信息
    print(f"\n--- Robot Info ---")
    print(f"Name: {robot.name}")
    print(f"Total links: {len(robot.get_links())}")
    print(f"Active joints (controllable): {len(robot.get_active_joints())}")

    print(f"\n--- Active joints ---")
    for j in robot.get_active_joints():
        limits = j.get_limits()
        print(f"  {j.name:30s} type={j.type:12s} limits={limits[0] if len(limits) > 0 else 'N/A'}")

    print(f"\n--- All links (前 30 个) ---")
    for link in robot.get_links()[:30]:
        print(f"  {link.name}")

    # ---- 开 viewer ----
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-1.5, y=0, z=1.5)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    print(f"\n✓ Viewer opened. Press 'q' or close window to exit.")
    print(f"  鼠标拖拽旋转视角, 滚轮缩放, 右键平移")

    # 让 scene 跑起来, 模拟物理 (重力会让机器人放下到地面)
    step_count = 0
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()
        step_count += 1
        if step_count % 100 == 0:
            # 每 100 步打印一下 base 高度,看是否在合理位置
            base_pose = robot.get_pose()
            print(f"  step={step_count}, base z={base_pose.p[2]:.3f}m")


if __name__ == "__main__":
    main()
