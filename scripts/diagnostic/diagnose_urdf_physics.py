"""诊断: 加载 URDF 后, 不施加任何 action, 看物理量如何演化。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sapien
from sapien import Pose


URDF_PATH = Path(__file__).parent.parent / "xlerobot_rl/sim/assets/urdf/xlerobot.urdf"


def safe_get_attr(obj, attr_names, default="N/A"):
    """尝试一组属性名,返回第一个能取到的。"""
    for name in attr_names:
        try:
            v = getattr(obj, name)
            if callable(v):
                v = v()
            return v
        except Exception:
            continue
    return default


def main():
    scene = sapien.Scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    robot = loader.load(str(URDF_PATH))
    robot.set_pose(Pose([0, 0, 0]))

    joints = robot.get_active_joints()
    joint_names = [j.name for j in joints]
    print(f"Active joints: {len(joints)}\n")

    # ---- Joint 物理参数 (用新版 API) ----
    print("--- Joint dynamics (新版 SAPIEN API) ---")
    for j in joints:
        # 在新版 SAPIEN 这些是 property 不是 method
        stiffness = safe_get_attr(j, ["stiffness"], 0)
        damping = safe_get_attr(j, ["damping"], 0)
        friction = safe_get_attr(j, ["friction"], 0)
        force_limit = safe_get_attr(j, ["force_limit"], "N/A")
        armature = safe_get_attr(j, ["armature"], 0)
        print(f"  {j.name:25s} "
              f"stiff={stiffness:>8.2f}  "
              f"damp={damping:>8.2f}  "
              f"fric={friction:>6.3f}  "
              f"force_lim={force_limit}  "
              f"armature={armature}")

    # ---- Link 质量 ----
    print(f"\n--- Link masses (非零质量) ---")
    total_mass = 0.0
    for link in robot.get_links():
        m = link.mass
        total_mass += m
        if m > 0.01:
            cm = safe_get_attr(link, ["cmass_local_pose"], None)
            cm_str = f"  CoM={cm.p if cm else 'N/A'}"
            print(f"  {link.name:35s} mass={m:6.3f} kg{cm_str}")
    print(f"  TOTAL: {total_mass:.2f} kg")

    # ---- 初始状态 ----
    init_pose = robot.get_pose()
    init_qpos = robot.get_qpos().copy()
    print(f"\n--- Initial state ---")
    print(f"  Base pose.p: {init_pose.p}")
    print(f"  Base pose.q: {init_pose.q}")
    print(f"  qpos[:5]:    {init_qpos[:5]}")

    # ---- 模拟 500 step (5 秒), 看 drift ----
    print(f"\n--- Simulating 500 steps with no action ---")
    print(f"  {'step':>5s} | {'base dx':>9s} {'dy':>9s} {'dz':>9s} | "
          f"{'qmax_drift':>10s} | drifting_joint")
    print(f"  " + "-" * 80)

    for step in range(501):
        scene.step()
        if step % 50 == 0:
            qpos = robot.get_qpos()
            base_p = robot.get_pose().p
            qpos_diff = qpos - init_qpos
            base_dxyz = base_p - init_pose.p
            max_drift_idx = int(np.argmax(np.abs(qpos_diff)))
            max_drift = qpos_diff[max_drift_idx]
            print(f"  {step:>5d} | "
                  f"{base_dxyz[0]:>+9.4f} {base_dxyz[1]:>+9.4f} {base_dxyz[2]:>+9.4f} | "
                  f"{max_drift:>+10.4f} | "
                  f"{joint_names[max_drift_idx]}")

    print(f"\n--- Final state ---")
    final_qpos = robot.get_qpos()
    final_pose = robot.get_pose()
    qpos_diff = final_qpos - init_qpos
    base_drift = final_pose.p - init_pose.p

    print(f"\n  Base displacement: {base_drift}, magnitude = {np.linalg.norm(base_drift):.4f} m")
    print(f"\n  Top 8 most drifted joints:")
    sorted_idx = np.argsort(np.abs(qpos_diff))[::-1][:8]
    for idx in sorted_idx:
        print(f"    {joint_names[idx]:25s} drift = {qpos_diff[idx]:+.4f}")


if __name__ == "__main__":
    main()
