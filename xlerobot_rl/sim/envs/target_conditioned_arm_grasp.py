"""TargetConditionedArmGrasp-v0: M4 target-conditioned single-arm grasp env.

This is the Month 2 entry point for target-conditioned manipulation:
- three colored cubes are present on the table
- one target color is sampled per episode
- success only means lifting the selected target
- lifting a distractor is reported as wrong-object failure

The action/controller remains the same as StaticArmGrasp-v0: right-arm 6D joint
delta control. Visual target masks are still provided by M1/perception; this env
exposes sim GT target/distractor positions for early BC/RL plumbing.
"""
from __future__ import annotations

import torch

from mani_skill.utils.registration import register_env

from xlerobot_rl.sim.envs.static_arm_grasp import StaticArmGraspEnv


@register_env("TargetConditionedArmGrasp-v0", max_episode_steps=64)
class TargetConditionedArmGraspEnv(StaticArmGraspEnv):
    """Single-arm tabletop grasp with target/distractor semantics."""

    COLORS = ("red", "blue", "green")
    COLOR_TO_ID = {"red": 0, "blue": 1, "green": 2}

    def __init__(self, *args, target_color: str | None = None, **kwargs):
        self.fixed_target_color = target_color
        self.target_color = target_color or "red"
        if self.fixed_target_color is not None and self.fixed_target_color not in self.COLORS:
            raise ValueError(f"unsupported target_color: {self.fixed_target_color}")
        super().__init__(*args, include_distractors=True, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # M4 v0 uses a single target color for all parallel envs in this reset.
        # Per-env target randomization can be added once policies consume batched
        # semantic state cleanly.
        if self.fixed_target_color is not None:
            self.target_color = self.fixed_target_color
        else:
            target_idx = int(torch.randint(len(self.COLORS), (1,), device=self.device).item())
            self.target_color = self.COLORS[target_idx]

        self.cube = self.cubes[self.target_color]

    @property
    def target_cube(self):
        return self.cubes[self.target_color]

    @property
    def distractor_cubes(self):
        return {
            color: cube
            for color, cube in self.cubes.items()
            if color != self.target_color
        }

    def evaluate(self):
        target_cube = self.target_cube
        tcp_to_target_dist = torch.linalg.norm(
            target_cube.pose.p - self.agent.tcp_pos, axis=-1,
        )
        reached = tcp_to_target_dist < 0.03

        target_is_grasped = self.agent.is_grasping(target_cube)
        target_z = target_cube.pose.p[..., -1]
        target_lift_height = target_z - (self.TABLE_TOP_Z + self.CUBE_HALF_SIZE)
        target_is_lifted = target_lift_height >= self.LIFT_HEIGHT_THRESHOLD

        wrong_object_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        wrong_object_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        max_distractor_lift_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for cube in self.distractor_cubes_values():
            distractor_is_grasped = self.agent.is_grasping(cube)
            distractor_lift_height = cube.pose.p[..., -1] - (
                self.TABLE_TOP_Z + self.CUBE_HALF_SIZE
            )
            wrong_object_grasped = torch.logical_or(wrong_object_grasped, distractor_is_grasped)
            wrong_object_lifted = torch.logical_or(
                wrong_object_lifted,
                distractor_lift_height >= self.LIFT_HEIGHT_THRESHOLD,
            )
            max_distractor_lift_height = torch.maximum(
                max_distractor_lift_height,
                distractor_lift_height,
            )

        currently_lifting_target = torch.logical_and(target_is_grasped, target_is_lifted)
        self._lift_hold_counter = torch.where(
            currently_lifting_target,
            self._lift_hold_counter + 1,
            torch.zeros_like(self._lift_hold_counter),
        )
        success = torch.logical_and(
            self._lift_hold_counter >= self.LIFT_HOLD_STEPS,
            torch.logical_not(wrong_object_lifted),
        )

        return dict(
            tcp_to_obj_dist=tcp_to_target_dist,
            tcp_to_target_dist=tcp_to_target_dist,
            reached=reached,
            is_grasped=target_is_grasped,
            target_is_grasped=target_is_grasped,
            lift_height=target_lift_height,
            target_lift_height=target_lift_height,
            is_lifted=target_is_lifted,
            target_is_lifted=target_is_lifted,
            wrong_object_grasped=wrong_object_grasped,
            wrong_object_lifted=wrong_object_lifted,
            wrong_object_failure=wrong_object_lifted,
            max_distractor_lift_height=max_distractor_lift_height,
            success=success,
        )

    def distractor_cubes_values(self):
        return self.distractor_cubes.values()

    def compute_dense_reward(self, obs, action, info):
        reward = -0.5 * info["tcp_to_target_dist"]
        reward = reward + 1.0 * info["target_is_grasped"].float()
        reward = reward + 5.0 * torch.clamp(info["target_lift_height"], min=0.0)
        reward = reward + 10.0 * info["success"].float()
        reward = reward - 5.0 * info["wrong_object_lifted"].float()
        reward = reward - 1.0 * info["wrong_object_grasped"].float()
        if action is not None:
            reward = reward - 0.001 * torch.linalg.norm(action, dim=-1)
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 12.0

    def _get_obs_extra(self, info: dict):
        target_cube = self.target_cube
        target_color_id = self.COLOR_TO_ID[self.target_color]
        target_pos = target_cube.pose.p

        distractor_positions = []
        distractor_color_ids = []
        for color in self.COLORS:
            if color == self.target_color:
                continue
            distractor_positions.append(self.cubes[color].pose.p)
            distractor_color_ids.append(self.COLOR_TO_ID[color])

        obs = dict(
            tcp_pos=self.agent.tcp_pos,
            tcp_to_obj=target_pos - self.agent.tcp_pos,
            tcp_to_target=target_pos - self.agent.tcp_pos,
            target_pos_base=target_pos,
            target_color_id=torch.full(
                (self.num_envs, 1),
                target_color_id,
                dtype=torch.long,
                device=self.device,
            ),
            skill_id=torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device),
            distractor_pos_base=torch.cat(distractor_positions, dim=1),
            distractor_color_ids=torch.tensor(
                distractor_color_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).repeat(self.num_envs, 1),
        )

        if self.obs_mode_struct.state:
            obs.update(
                obj_pose=target_cube.pose.raw_pose,
                is_grasped=info["target_is_grasped"].float().unsqueeze(-1),
                lift_height=info["target_lift_height"].float().unsqueeze(-1),
                wrong_object_failure=info["wrong_object_failure"].float().unsqueeze(-1),
            )
        return obs
