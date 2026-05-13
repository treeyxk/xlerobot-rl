# Logging Schema (M0)

定义所有实验的 logging 字段和存储约定。**字段名一旦冻结不要随便改**, 改动需更新本文档且在 ADR 中记录。

**Status**: 🟡 Draft v1 (字段会在前期 iteration 中调整, Week 2 末锁定 v1.0)

---

## 1. 通用约定

### 1.1 命名规范

- 所有字段名用 `snake_case`
- Boolean 字段用 `is_*` / `has_*` 前缀: `is_success`, `has_collision`
- Rate 字段用 `_rate` 后缀: `success_rate`, `collision_rate`
- Count 字段用 `n_*` 前缀: `n_episodes`, `n_steps`
- 时间字段统一带单位后缀: `*_seconds`, `*_steps`
- 距离用米: `*_meters` 或不带后缀 (默认 m, 见 interface_contract §2)

### 1.2 存储后端

| 数据类型              | 存储位置                     | 格式            |
|---------------------|-----------------------------|----------------|
| 训练时序数据 (loss, reward) | wandb                       | wandb log      |
| Eval episode summary | wandb + 本地 HDF5            | HDF5           |
| Trajectory 详细数据   | 本地 HDF5 (`data/rollouts/`) | HDF5           |
| 视频                 | 本地 + wandb upload         | mp4            |
| Config              | 自动 Hydra dump + wandb     | YAML           |

### 1.3 时间戳

所有 step/event 必须带:

- `timestamp`: UTC seconds (float)
- `step_idx`: episode 内 step 索引 (int)
- `episode_idx`: 全 run 内 episode 索引 (int)

---

## 2. 训练时 (per-step logging via wandb)

每个 PPO update step 记录:

| 字段                       | 类型    | 说明                            |
|---------------------------|--------|--------------------------------|
| `train/reward_mean`        | float  | 当前 batch 平均 reward           |
| `train/reward_std`         | float  | reward 标准差                   |
| `train/episode_length_mean` | float | 平均 episode 长度 (steps)        |
| `train/success_rate`       | float  | sliding window (100 episode)    |
| `train/loss_policy`        | float  | PPO policy loss                |
| `train/loss_value`         | float  | PPO value loss                 |
| `train/loss_entropy`       | float  | entropy bonus                  |
| `train/kl_divergence`      | float  | 新旧 policy KL                  |
| `train/learning_rate`      | float  | 当前 lr                        |
| `train/n_env_steps`        | int    | 累计 env steps                  |
| `train/n_grad_steps`       | int    | 累计 gradient updates           |
| `train/action_clipping_rate` | float | action 触碰 bound 的比例        |
| `train/wall_time_seconds`  | float  | 累计训练时间                    |

---

## 3. Eval 时 (per-episode logging)

每个 eval episode 结束后记录:

### 3.1 通用字段

| 字段                  | 类型    | 说明                            |
|----------------------|--------|--------------------------------|
| `episode_idx`         | int    |                                |
| `task_name`           | str    | env 名称, 如 `StaticArmGrasp-v0` |
| `seed`                | int    | env 随机种子                    |
| `is_success`          | bool   | 任务整体是否成功                 |
| `n_steps`             | int    | episode 长度                    |
| `wall_time_seconds`   | float  | episode 耗时                    |
| `failure_reason`      | str    | 失败原因 code (见 §6, TBD)      |

### 3.2 导航 (π_nav)

仅当 episode 涉及导航时记录:

| 字段                          | 类型    | 说明                            |
|------------------------------|--------|--------------------------------|
| `nav/stop_success`            | bool   | 停下时 mean(S) > τ_s            |
| `nav/has_collision`           | bool   |                                |
| `nav/path_length_meters`      | float  | base 累计移动距离                |
| `nav/time_to_stop_seconds`    | float  | 从 episode 开始到停下的时间      |
| `nav/final_S_mean`            | float  | 停下时 ensemble mean(S)         |
| `nav/final_S_std`             | float  | 停下时 ensemble std(S)          |
| `nav/final_object_distance`   | float  | 停下时 EE 到 object 距离        |

### 3.3 操作 (π_arm)

仅当 episode 涉及操作时记录:

| 字段                          | 类型    | 说明                            |
|------------------------------|--------|--------------------------------|
| `arm/grasp_success`           | bool   | 抓取成功 (物体被夹住)            |
| `arm/lift_success`            | bool   | 抓起 > 5cm 持续 10 步           |
| `arm/has_drop`                | bool   | 抓起后掉落                      |
| `arm/ee_to_object_final_distance` | float | 最终 EE 到物体距离              |
| `arm/action_clipping_rate`    | float  | arm action 触碰 bound 比例      |

### 3.4 S Predictor (训练 / eval 时单独评估)

| 字段                          | 类型    | 说明                            |
|------------------------------|--------|--------------------------------|
| `s/val_accuracy`              | float  |                                |
| `s/ece`                       | float  | Expected Calibration Error      |
| `s/brier_score`               | float  |                                |
| `s/auc`                       | float  |                                |
| `s/false_positive_rate_at_0.8` | float | **核心 metric**                |
| `s/mean_uncertainty`          | float  | ensemble std 的平均             |

### 3.5 VLM / Perception

| 字段                          | 类型    | 说明                            |
|------------------------------|--------|--------------------------------|
| `vlm/detection_success`       | bool   | 是否成功检测到目标               |
| `vlm/mask_iou_vs_gt`          | float  | 若有 GT (仅 sim)                |
| `vlm/3d_position_error_meters` | float | 若有 GT                         |
| `vlm/latency_ms`              | float  | 单次 inference 耗时             |
| `vlm/confidence`              | float  | 最终输出置信度                  |

### 3.6 End-to-end

| 字段                          | 类型    | 说明                            |
|------------------------------|--------|--------------------------------|
| `e2e/total_success`           | bool   |                                |
| `e2e/total_time_seconds`      | float  |                                |
| `e2e/failure_stage`           | str    | `nav` / `arm` / `handoff` / `vlm` / `none` |
| `e2e/handoff_failure`         | bool   | nav 停了但 arm 失败             |

---

## 4. Trajectory 详细数据 (per-step, 仅 eval)

存到本地 HDF5 (`data/rollouts/{run_id}/episode_{idx}.h5`), 用于 failure debug 和 paper 可视化。

### 4.1 必存字段

每个 step:

```python
{
    "timestamp": float,
    "step_idx": int,
    "obs": {
        "head_rgb": np.ndarray,         # (H, W, 3) uint8
        "head_depth": np.ndarray,       # (H, W) float32, in meters (real / sim)
        "wrist_rgb_left": np.ndarray,   # optional
        "wrist_rgb_right": np.ndarray,  # optional
        "base_pose": np.ndarray,        # (3,) [x, y, theta]
        "arm_qpos_left": np.ndarray,    # (6,) joint angles in rad
        "arm_qpos_right": np.ndarray,   # optional
        "gripper_state_left": float,
        "gripper_state_right": float,   # optional
    },
    "action": np.ndarray,               # shape depends on action space
    "reward": float,
    "info": {
        "vlm_target": dict,             # TargetObservation 内容 (见 contract §4 TBD)
        "S_mean": float,
        "S_std": float,
        "current_state": str,           # orchestrator 状态名
    }
}
```

### 4.2 存储策略

- 默认: 仅存 **失败的 episode** (减少存储压力)
- 训练阶段抽样: 每 50 episode 存 1 个完整 trajectory (用于 debug)
- 真机阶段: **全部存** (真机 trial 贵, 不能丢)

---

## 5. Run-level Metadata

每个 wandb run 必须 tag 这些信息, 用于后期 filter 和 ablation:

```yaml
tags:
  - module: "arm" | "nav" | "s" | "e2e"
  - phase: "sim" | "real"
  - experiment_type: "baseline" | "ablation" | "production"
  - git_commit: <commit hash>      # 自动从 git 读取
  - hardware: "5070ti" | "rtx4090" | ...
```

Config dump (Hydra 自动) 必须包含:

- 完整 hyperparameters
- Env config
- Network architecture
- Random seeds (env, torch, numpy)

---

## 6. 待补充 (Section TBD)

以下字段定义等接口和实验设计进一步成熟后补充:

- [ ] §6 Failure mode codes 完整列表 (Week 2-3 在 oracle baseline 阶段定义)
- [ ] Sim2real 相关字段 (calibration_error, latency, odometry_drift) - Week 8+
- [ ] Closed-loop retraining 相关字段 - Week 11+

未定义之前, log 字段用 `tbd_*` 前缀临时命名, 不要污染主 schema。

