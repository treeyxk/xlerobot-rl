"""XLeRobot 2-wheel agent class for ManiSkill.

继承 BaseAgent, 适配修改版 URDF (xlerobot.urdf, 已 calib):
  - 16 active joints (2 wheels + 12 arm + 2 head, 不含官方 holonomic 的 3 virtual joints)
  - Joint 命名: `_L` / `_R` suffix
  - Differential drive base (2 wheels)
  - TCP local offset = 10.7cm (经 viewer 实测, 见 docs/decisions.md)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent, REGISTERED_AGENTS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


# URDF 在 repo 内, 用 absolute path 避免 ManiSkill ASSET_DIR 假设
_REPO_ROOT = Path(__file__).resolve().parents[3]  # xlerobot_rl/sim/robots/xlerobot_agent.py -> repo root
URDF_PATH = _REPO_ROOT / "xlerobot_rl/sim/assets/urdf/xlerobot.urdf"
assert URDF_PATH.exists(), f"URDF not found: {URDF_PATH}"


# 避免重复注册 (jupyter / 多次 import 时)
if "xlerobot_2wheels" in REGISTERED_AGENTS:
    del REGISTERED_AGENTS["xlerobot_2wheels"]


@register_agent()
class XLeRobot2Wheels(BaseAgent):
    """XLeRobot 0.4.0 双臂 + 2-wheel 移动底盘的 ManiSkill agent。"""

    uid = "xlerobot_2wheels"
    urdf_path = str(URDF_PATH)

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
        ),
        link=dict(
            Fixed_Jaw_2=dict(material="gripper", patch_radius=0.01, min_patch_radius=0.01),
            Moving_Jaw_2=dict(material="gripper", patch_radius=0.01, min_patch_radius=0.01),
            Fixed_Jaw=dict(material="gripper", patch_radius=0.01, min_patch_radius=0.01),
            Moving_Jaw=dict(material="gripper", patch_radius=0.01, min_patch_radius=0.01),
        ),
    )

    # ==================================================
    # Keyframe 'rest'
    # IK 校准过让右臂 EE 在 spawn 中心上方 (来自旧 grasp_demo 项目)
    # qpos 顺序对照见下方 _qpos_idx_map 注释
    # ==================================================
    #   [0,1]  wheels (left/right)
    #   [2,3]  Rotation_L, Rotation_R
    #   [4]    head_pan
    #   [5,6]  Pitch_L, Pitch_R
    #   [7]    head_tilt
    #   [8,9]  Elbow_L, Elbow_R
    #   [10,11] Wrist_Pitch_L, Wrist_Pitch_R
    #   [12,13] Wrist_Roll_L, Wrist_Roll_R
    #   [14,15] Jaw_L, Jaw_R
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0.0, 0.0,                # [0,1]   wheels
                -1.57079633, 0.0,        # [2,3]   Rotation_L, Rotation_R
                -0.60,                   # [4]     head_pan (低头看桌面)
                0.0, -0.365,             # [5,6]   Pitch_L, Pitch_R
                1.11,                    # [7]     head_tilt (低头到 cube 方向)
                0.0, -0.3,               # [8,9]   Elbow_L, Elbow_R
                0.0, +1.331937,          # [10,11] Wrist_Pitch_L, Wrist_Pitch_R
                0.0, +1.57079633,        # [12,13] Wrist_Roll_L, Wrist_Roll_R
                1.2, -0.3,               # [14,15] Jaw_L (开), Jaw_R (闭)
            ], dtype=np.float32),
            pose=sapien.Pose(),
        ),
    )

    LEFT_ARM_JOINTS = [
        "Rotation_L", "Pitch_L", "Elbow_L",
        "Wrist_Pitch_L", "Wrist_Roll_L", "Jaw_L",
    ]
    RIGHT_ARM_JOINTS = [
        "Rotation_R", "Pitch_R", "Elbow_R",
        "Wrist_Pitch_R", "Wrist_Roll_R", "Jaw_R",
    ]
    HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
    WHEEL_JOINTS = ["left_wheel_joint", "right_wheel_joint"]

    # TCP local offset 经 viewer 实测验证 (scripts/diagnostic/measure_tcp_offset.py)
    # Fixed_Jaw_2 frame: 沿 -Y 延伸 10.7cm 到两指中心
    TCP_LOCAL_OFFSET_FIXED = (0.0, -0.107, 0.0)

    # ==================================================
    # Controller config: 只控右臂 6 维 (单臂 task)
    # ==================================================
    @property
    def _controller_configs(self):
        arm_joints = self.RIGHT_ARM_JOINTS  # 6 维: 5 arm + 1 gripper
        lower, upper = [], []
        for name in arm_joints:
            if name.startswith("Jaw"):
                lower.append(-0.2); upper.append(0.2)
            else:
                lower.append(-0.05); upper.append(0.05)

        arm_delta = PDJointPosControllerConfig(
            arm_joints,
            lower=lower, upper=upper,
            stiffness=1e3, damping=1e2, force_limit=50,
            use_delta=True, use_target=True,
            normalize_action=True,
        )

        return dict(
            pd_joint_target_delta_pos=dict(arm=arm_delta),
        )

    # ==================================================
    # Articulation loading
    # ==================================================
    def _load_articulation(self, initial_pose=None):
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid
        loader.disable_self_collisions = False
        parsed_dict = loader.parse(self.urdf_path)
        builder = parsed_dict["articulation_builders"][0]
        builder.initial_pose = (
            initial_pose if initial_pose is not None else sapien.Pose([0, 0, 0])
        )
        self.robot = builder.build()
        return self.robot

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        links_map = {l.name: l for l in self.robot.get_links()}
        # 右臂 gripper 两指 (单臂 task 主用)
        self.finger1_link = links_map["Fixed_Jaw_2"]
        self.finger2_link = links_map["Moving_Jaw_2"]
        self.finger1_tip = self.finger1_link
        self.finger2_tip = self.finger2_link

    # ==================================================
    # TCP (Tool Center Point)
    # ==================================================
    @property
    def tcp_pos(self):
        """TCP 位置 (世界系), Fixed_Jaw_2 沿 -Y 延伸 10.7cm。"""
        device = self.finger1_link.pose.p.device
        off = torch.tensor(
            self.TCP_LOCAL_OFFSET_FIXED, device=device, dtype=torch.float32,
        )
        R = self.finger1_link.pose.to_transformation_matrix()[..., :3, :3]
        world_off = torch.einsum("...ij,j->...i", R, off)
        return self.finger1_link.pose.p + world_off

    @property
    def tcp_pose(self):
        """TCP 完整 SE(3) pose (位置 + Fixed_Jaw 朝向)。"""
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    # ==================================================
    # Grasping check
    # ==================================================
    def is_grasping(self, object: Actor, min_force=0.2, max_angle=None):
        """两指都接触 object 且接触力 >= min_force。"""
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        lflag = lforce >= min_force
        rflag = rforce >= min_force
        if max_angle is None:
            return torch.logical_and(lflag, rflag)

        # 检查接触力方向是否在 jaw 张合方向 ± max_angle 内
        jaw_axis = self.finger2_link.pose.p - self.finger1_link.pose.p
        jaw_axis = jaw_axis / (torch.linalg.norm(jaw_axis, dim=1, keepdim=True) + 1e-6)
        langle = common.compute_angle_between(jaw_axis, l_contact_forces)
        rangle = common.compute_angle_between(-jaw_axis, r_contact_forces)
        lflag = torch.logical_and(lflag, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rflag, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

    # ==================================================
    # Sensor configs (cameras)
    # ==================================================
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),   # 不加 rotation, 看默认朝向
                width=320, height=240,
                fov=np.deg2rad(69),
                near=0.16,
                far=10.0,
                mount=self.robot.find_link_by_name("head_camera_link"),  # ← sensor body
            ),
            CameraConfig(
                uid="right_wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128, height=128,
                fov=np.deg2rad(52),
                near=0.01,
                far=10.0,
                mount=self.robot.find_link_by_name("Right_Arm_Camera"),
            ),
        ]