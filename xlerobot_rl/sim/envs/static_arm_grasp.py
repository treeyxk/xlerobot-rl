"""StaticArmGrasp-v0: 静态 base + 单臂抓取红色 cube。

v2 plan §5 的最小 task env. Base 固定在原点, 右臂从 rest 姿态出发,
抓取桌面上随机位置的红色 cube。

Action space: 6 维 (右臂 5 joint delta + 1 gripper)
Observation: head_camera RGB + right_wrist_camera RGB + arm proprio + GT object pose
Success: cube 抬高 > 5cm 持续 10 步
"""
from __future__ import annotations
import numpy as np
import sapien
import torch
import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import SimConfig

# 确保 agent class 被注册 (side-effect import)
from xlerobot_rl.sim.robots.xlerobot_agent import XLeRobot2Wheels  # noqa: F401


@register_env("StaticArmGrasp-v0", max_episode_steps=64)
class StaticArmGraspEnv(BaseEnv):
    """单臂抓取任务. Base 静止, 右臂从 rest 出发抓桌面 cube。"""

    SUPPORTED_ROBOTS = ["xlerobot_2wheels"]
    agent: XLeRobot2Wheels

    # 场景参数 (跟旧 grasp_demo 项目对齐, spawn 范围已验证可达)
    TABLE_HALF_SIZES = [0.3, 0.8, 0.35]
    TABLE_TOP_Z = 0.70
    CUBE_HALF_SIZE = 0.015  # 边长 3cm
    SPAWN_CENTER = (-0.36, 0.09)
    SPAWN_HALF_RANGE = (0.11, 0.08)

    # Success criteria
    LIFT_HEIGHT_THRESHOLD = 0.05    # 5cm
    LIFT_HOLD_STEPS = 10            # 持续 10 步

    def __init__(
        self,
        *args,
        robot_uids="xlerobot_2wheels",
        control_mode="pd_joint_target_delta_pos",
        **kwargs,
    ):
        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode, **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_human_render_camera_configs(self):
        return [
            CameraConfig(
                uid="third_person",
                pose=sapien_utils.look_at(
                    eye=[0.2, 0.5, 1.0],
                    target=[-0.4, 0.1, 0.7],
                ),
                width=768, height=768,
                fov=np.deg2rad(45),
                near=0.01, far=10.0,
            ),
        ]

    # ==================================================
    # Load scene
    # ==================================================
    def _load_scene(self, options: dict):
        self.floor = build_ground(self.scene)

        # 桌子: 静态 box, top 在 z=0.70
        self.table = actors.build_box(
            self.scene,
            half_sizes=self.TABLE_HALF_SIZES,
            color=[0.4, 0.3, 0.2, 1],
            name="table",
            body_type="static",
            initial_pose=sapien.Pose([-0.525, 0, 0.35]),
        )

        # 红色 cube, 加高摩擦防止滑动
        builder = self.scene.create_actor_builder()
        cube_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        )
        builder.add_box_collision(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=cube_material, density=200,
        )
        builder.add_box_visual(
            half_size=[self.CUBE_HALF_SIZE] * 3,
            material=sapien.render.RenderMaterial(base_color=[0.9, 0.1, 0.1, 1]),
        )
        builder.initial_pose = sapien.Pose(
            p=[0, 0, self.TABLE_TOP_Z + self.CUBE_HALF_SIZE]
        )
        self.cube = builder.build(name="cube")

        # 缓存 rest qpos 给 reset 用
        self.rest_qpos = torch.as_tensor(
            self.agent.keyframes["rest"].qpos,
            dtype=torch.float32, device=self.device,
        )

        # Lift hold counter (per env), 用于持续 N 步判定
        self._lift_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ==================================================
    # Initialize episode
    # ==================================================
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)

        # Robot base 固定在原点
        self.agent.robot.set_pose(sapien.Pose([0, 0, 0]))

        # 重置到 rest qpos + 给右臂加点小 noise (避免每个 episode 一模一样)
        init_qpos = self.rest_qpos.unsqueeze(0).repeat(b, 1).clone()
        noise = torch.randn_like(init_qpos) * 0.02
        noise_mask = torch.zeros_like(init_qpos)
        # 右臂索引: Rotation_R=3, Pitch_R=6, Elbow_R=9, Wrist_Pitch_R=11, Wrist_Roll_R=13
        right_arm_idxs = [3, 6, 9, 11, 13]
        noise_mask[:, right_arm_idxs] = 1.0
        init_qpos = init_qpos + noise * noise_mask
        self.agent.robot.set_qpos(init_qpos)

        # Cube 随机 spawn 在桌面可达区域
        center = torch.tensor(self.SPAWN_CENTER, device=self.device)
        half_range = torch.tensor(self.SPAWN_HALF_RANGE, device=self.device)
        xy_noise = (torch.rand((b, 2), device=self.device) - 0.5) * 2 * half_range
        xyz = torch.zeros((b, 3), device=self.device)
        xyz[:, :2] = center + xy_noise
        xyz[:, 2] = self.TABLE_TOP_Z + self.CUBE_HALF_SIZE + 1e-3
        self.cube.set_pose(Pose.create_from_pq(p=xyz))

        # 记录 cube 初始位置 (用于检测 xy drift)
        if (not hasattr(self, "cube_init_xy")
                or self.cube_init_xy.shape[0] != self.num_envs):
            self.cube_init_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.cube_init_xy[env_idx] = xyz[:, :2]

        # 重置 lift hold counter
        self._lift_hold_counter[env_idx] = 0

    # ==================================================
    # Evaluate
    # ==================================================
    def evaluate(self):
        # TCP 离 cube 距离
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pos, axis=-1,
        )
        reached = tcp_to_obj_dist < 0.03

        # 是否抓住
        is_grasped = self.agent.is_grasping(self.cube)

        # Cube 抬高高度
        cube_z = self.cube.pose.p[..., -1]
        lift_height = cube_z - (self.TABLE_TOP_Z + self.CUBE_HALF_SIZE)
        is_lifted = lift_height >= self.LIFT_HEIGHT_THRESHOLD

        # 持续 N 步抬起 = success
        # 注意: counter 在 step 之间累积, 这里只算"当前是否在抬"
        currently_lifting = torch.logical_and(is_grasped, is_lifted)
        self._lift_hold_counter = torch.where(
            currently_lifting,
            self._lift_hold_counter + 1,
            torch.zeros_like(self._lift_hold_counter),
        )
        success = self._lift_hold_counter >= self.LIFT_HOLD_STEPS

        return dict(
            tcp_to_obj_dist=tcp_to_obj_dist,
            reached=reached,
            is_grasped=is_grasped,
            lift_height=lift_height,
            is_lifted=is_lifted,
            success=success,
        )

    # ==================================================
    # Reward (v2 plan §5.2 简单版)
    # ==================================================
    def compute_dense_reward(self, obs, action, info):
        # 三段式 reward, 跟 v2 design doc 对齐
        # r = α·(-d_tcp_to_obj) + β·grasp_success + γ·lift_height - δ·|action| - ε·step
        reward = -0.5 * info["tcp_to_obj_dist"]                    # dense: 接近物体
        reward = reward + 1.0 * info["is_grasped"].float()         # sparse: 抓住
        reward = reward + 5.0 * torch.clamp(info["lift_height"], min=0.0)  # 抬起越高越好
        reward = reward + 10.0 * info["success"].float()           # 任务成功大奖
        # action smoothness penalty
        if action is not None:
            reward = reward - 0.001 * torch.linalg.norm(action, dim=-1)
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        # Normalize 到 [~-1, ~1] 量级方便训练
        return self.compute_dense_reward(obs, action, info) / 12.0

    # ==================================================
    # Obs extra (GT object info, 暂代 VLM)
    # ==================================================
    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pos=self.agent.tcp_pos,
            tcp_to_obj=self.cube.pose.p - self.agent.tcp_pos,
        )

        if self.obs_mode_struct.state:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                is_grasped=info["is_grasped"].float().unsqueeze(-1),
                lift_height=info["lift_height"].float().unsqueeze(-1),
            )
        return obs
