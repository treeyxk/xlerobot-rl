# BC 数据集 Schema

**状态**: Draft v0

本文档定义 M4 target-conditioned manipulation 的第一版行为克隆数据集契约。范围刻意收窄:
右臂、桌面、单一 skill (`top_grasp`)、红/蓝/绿色块抓取。

目标是让 sim demo、LeRobot 主从采集 demo、以及后续 dataset converter 暴露同一组训练字段。

## 范围

包含:

- 仅右臂 follower。
- 底盘静止或近似静止。
- 一个目标物体,零个或多个 distractor。
- 仅 `top_grasp` skill。
- v0 使用 joint-space action label。

不包含:

- 多 skill 标签。
- 导航 action。
- 端到端 M2/M3/M5 trajectory。
- 除原始 instruction 文本外的开放词汇语言理解。
- 除简单成功/失败元数据外的 learned reward classifier label。

## 存储布局

LeRobot 原始录制保持原生目录:

```text
data/real/lerobot/<dataset_name>/
```

训练侧转换后的数据应暴露本文档定义的 schema。HDF5 是 v0 参考格式,因为它易检查且不依赖
LeRobot 内部实现:

```text
data/bc/m4_target_grasp_v0/
  dataset_info.yaml
  episode_000000.h5
  episode_000001.h5
  ...
```

Sim 生成的 BC 或 oracle 数据也使用同一 schema:

```text
data/bc/sim_m4_target_grasp_v0/
  dataset_info.yaml
  episode_000000.h5
```

## Dataset 级元数据

`dataset_info.yaml`:

```yaml
schema_version: "bc_m4_v0"
source: "real_lerobot"  # "real_lerobot" | "sim_oracle" | "sim_replay"
env_id: "TargetConditionedArmGrasp-v0"
robot: "XLeRobot 0.4.0"
arm: "right"
intended_control_hz: 15
action_space: "right_arm_joint_delta_6d"
joint_order:
  - shoulder_pan
  - shoulder_lift
  - elbow_flex
  - wrist_flex
  - wrist_roll
gripper_convention: "0=closed, 1=open"
skill_id_map:
  top_grasp: 0
color_id_map:
  red: 0
  blue: 1
  green: 2
created_at: "YYYY-MM-DDTHH:MM:SSZ"
git_commit: "<commit>"
notes: ""
```

## Episode 元数据

每个 `episode_XXXXXX.h5` 必须包含 `meta` group 或等价 attributes:

| 字段 | 类型 | 含义 |
|------|------|------|
| `episode_idx` | int | 数据集内 episode 索引。 |
| `instruction` | str | 原始任务文本,例如 `"Pick up the red cube with the right arm"`。 |
| `target_color_id` | int | `red=0`, `blue=1`, `green=2`。 |
| `target_name` | str | 例如 `"red_cube"`。 |
| `skill_id` | int | `top_grasp=0`。 |
| `sim_seed` | int or null | sim episode 的 `env.reset(seed=...)` 种子;真机数据为 null。 |
| `spawn_xy` | array or null | sim 中 cube 实际生成的 xy 位置,用于 debug/replay;真机数据为 null。 |
| `is_success` | bool | episode 级任务成功:目标物体被抓起并保持。 |
| `wrong_object_failure` | bool | 是否抓起了任何 distractor。 |
| `n_steps` | int | 记录的总 step 数。 |
| `actual_control_hz` | float | episode 平均实际控制频率。 |
| `intended_control_hz` | float | 目标控制频率;若全 dataset 一致则从 dataset metadata 复制。 |
| `source` | str | `"real_lerobot"` / `"sim_oracle"` / `"sim_replay"`。 |

`n_steps` 必须等于 `T`,也就是该 episode 内所有 per-step array 的长度。

## Per-Step 字段

所有数组以时间为第一维。`T` 是 episode 长度。

### 必需 Observation 字段

| HDF5 path | Shape | Dtype | 单位 / 含义 |
|-----------|-------|-------|-------------|
| `obs/timestamp_monotonic` | `(T,)` | float64 | 单调时间秒,例如 `time.monotonic()` 或 simulator monotonic time。 |
| `obs/wall_time` | `(T,)` | float64 | 可选 wall-clock UTC seconds,例如 `time.time()`,用于日志关联。 |
| `obs/step_idx` | `(T,)` | int64 | episode 内 step 索引。 |
| `obs/right_arm_qpos` | `(T, 5)` | float32 | 右臂非 gripper 关节角,单位 rad。 |
| `obs/right_arm_qvel` | `(T, 5)` | float32 | 右臂非 gripper 关节速度;v0 不可用时允许填 0。 |
| `obs/gripper_width` | `(T, 1)` | float32 | gripper 物理开口宽度,单位米;未标定时允许 NaN。 |
| `obs/gripper_command` | `(T, 1)` | float32 | 归一化 gripper 状态/命令,`[0, 1]`, `0=closed`, `1=open`。 |
| `obs/target_pos_base` | `(T, 3)` | float32 | target 在 base/world frame 下的位置,单位米。来源规则见下文。 |
| `obs/target_visible` | `(T, 1)` | bool | 当前帧 grounding 是否产生有效 target observation。 |
| `obs/target_color_id` | `(T, 1)` | int64 | `red=0`, `blue=1`, `green=2`。 |
| `obs/skill_id` | `(T, 1)` | int64 | `top_grasp=0`。 |
| `obs/prev_action` | `(T, 6)` | float32 | 上一步 action;step 0 填 0。 |

### 可选 Observation 字段

| HDF5 path | Shape | Dtype | 含义 |
|-----------|-------|-------|------|
| `obs/head_rgb` | `(T, H, W, 3)` | uint8 | 头部相机 RGB。 |
| `obs/head_depth` | `(T, H, W)` | float32 | 深度,单位米。 |
| `obs/head_depth_mm` | `(T, H, W)` | uint16 | 可选 raw depth,单位毫米,用于存储/debug。 |
| `obs/wrist_rgb_right` | `(T, H, W, 3)` | uint8 | 右腕相机 RGB。 |
| `obs/target_mask` | `(T, H, W)` | bool/uint8 | M1 输出的 target mask,坐标系为 policy 使用的图像 frame。 |
| `obs/distractor_mask` | `(T, H, W)` | bool/uint8 | 非目标物体 union mask。 |
| `obs/distractor_pos_base` | `(T, 6)` | float32 | 两个 distractor 位置,展平为 `[x1,y1,z1,x2,y2,z2]`。 |
| `obs/ee_pose_base` | `(T, 7)` | float32 | 若 FK 可用,为 EE xyz + quaternion in base frame。 |

当前 M4 v0 sim env 已暴露:

```text
target_pos_base
target_color_id
distractor_pos_base
distractor_color_ids
skill_id
tcp_to_target
```

### Target Position 来源

`obs/target_pos_base` 是 policy 输入。它在 sim 和 real 中应尽量保持同分布。canonical 来源是
**M1.2 grounding output**,不是特权 sim GT。

| Dataset source | `target_pos_base` 规则 |
|----------------|------------------------|
| `sim_oracle` | 对每个 recorded step 的 sim head RGB-D 重新跑 M1.2 grounding。Sim GT 可另存作 debug。 |
| `sim_replay` | 同 `sim_oracle`: policy 输入来自 grounding output。 |
| `real_lerobot` | 来自真实 RGB-D 上的 M1/VLM/SAM2 grounding output。 |
| smoke-only real data | 若该数据不用于 BC training,可以为空/NaN。 |

如果临时为了 ablation/debug 使用 privileged sim GT,必须存为 `debug/target_pos_base_gt`,
默认不得喂给 BC policy。

当当前帧 grounding 失败时,设置 `obs/target_visible=false`。M4 v0 中
`obs/target_pos_base` 可以保存 last valid target position 或 NaN;policy 必须使用
`target_visible` 区分“当前看见的 target”和“过期/无效位置”。M1.3 tracker 后续会正式定义
last-seen 行为。

### Action 字段

v0 action label 是右臂 joint delta:

| HDF5 path | Shape | Dtype | 含义 |
|-----------|-------|-------|------|
| `action/right_arm_delta_qpos` | `(T, 5)` | float32 | 非 gripper 关节 delta command,单位 rad/step。 |
| `action/gripper_command` | `(T, 1)` | float32 | 归一化 gripper command,`[0, 1]`, `0=closed`, `1=open`。 |
| `action/right_arm_target_qpos` | `(T, 5)` | float32 | 如果 source 提供,为可选绝对非 gripper target qpos。 |
| `action/is_intervention` | `(T,)` | bool | 人类介入标记;teleop demo 全部为 `True`。 |

canonical joint order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
```

Sim mapping:

```text
Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R
```

gripper 从 arm qpos/action 中拆出。这样避免把 radian joint position 和 normalized open/close command
混在一起,也让 gripper 标定保持显式。

### Label / Info 字段

| HDF5 path | Shape | Dtype | 含义 |
|-----------|-------|-------|------|
| `label/is_success` | `(T,)` | bool | per-step task success flag。Episode-level success 是 `meta/is_success`。 |
| `label/target_is_grasped` | `(T,)` | bool | target grasp flag。 |
| `label/target_lift_height` | `(T,)` | float32 | target lift height,单位米。 |
| `label/wrong_object_failure` | `(T,)` | bool | 是否抓起任何 distractor。 |
| `label/failure_reason` | `(T,)` or episode attr | str/int | 可选 failure code。 |

## Real LeRobot 映射

`lerobot-record` 原始数据可以保持 LeRobot 原生格式。本项目有两条可选路径:

1. converter 写出上述 HDF5 reference schema。
2. LeRobot Dataset API adapter 在训练时暴露等价字段。

关键要求不是文件格式,而是 training loader 看到的 schema 和单位一致。

右臂 follower 的最小映射:

| LeRobot source | BC 字段 |
|----------------|---------|
| follower joint observations | `obs/right_arm_qpos` |
| follower joint velocity, if available | `obs/right_arm_qvel` |
| follower action / target | `action/right_arm_delta_qpos` or `action/right_arm_target_qpos` |
| follower gripper state/command | `obs/gripper_command`, `action/gripper_command` |
| camera frame | `obs/head_rgb` or `obs/wrist_rgb_right` |
| task text | episode `instruction` |
| M1 output / manual label | `obs/target_pos_base`, `obs/target_mask`, `target_color_id` |

第一批 smoke dataset 可以缺少 `target_mask`、`distractor_mask`、`target_pos_base`,只用于验证录制完整性。
正式 BC training data 必须填充 target-conditioned 字段,或能可靠转换出这些字段。

### Smoke 录制入口

项目提供 `scripts/deploy/record_bc_demo.py` 作为真实 LeRobot smoke/BC 录制入口。默认 dry-run,
只生成 `data/bc/<dataset_name>/dataset_info.yaml` 并打印校准、teleop、record 命令。

```bash
python scripts/deploy/record_bc_demo.py \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyACM1 \
  --target-color red \
  --num-episodes 2 \
  --episode-time-s 15 \
  --camera-index /dev/video6
```

确认校准和 10 秒 teleop smoke test 通过后,添加 `--run-record`。脚本会在真正执行
`lerobot-record` 前打印 checklist,并等待人工确认:

- 按 `Space` 开始录制。
- 按 `q` 取消,不会启动 `lerobot-record`。
- 自动化脚本可加 `--no-ready-prompt` 跳过该确认。

建议使用 `/dev/video*` 设备路径作为 `--camera-index`,而不是数字 index,避免相机枚举顺序变化。
可通过 `lerobot-find-cameras opencv` 查看可用相机。录制脚本默认传入
`--dataset.vcodec=h264`,这样视频比 LeRobot 默认 `libsvtav1` 更容易在本地用 OpenCV 或播放器检查。

原始 LeRobot 数据写入:

```text
data/real/lerobot/<dataset_name>/
```

伴随的项目 metadata 模板写入:

```text
data/bc/<dataset_name>/dataset_info.yaml
```

## 数据质量 Gate

BC 训练前必须检查:

- 每个 episode 都有 instruction 和 target color。
- `meta/n_steps` 等于 `T`,即每个 per-step array 的长度。
- `obs/right_arm_qpos` 和 action arrays 有相同的 `T`。
- qpos/action/target position 字段没有 NaN/Inf。
- 必须存在 `obs/target_visible`。若 `target_visible=false`,允许 target position 为 NaN。
- gripper convention 记录为 `0=closed`, `1=open`。
- 如果有视频帧,不得有大的 timestamp gap。
- 每个 episode 至少有一个 success/failure label。
- hard negative 必须标记 `is_success=false`,并尽量记录 failure reason。

## Smoke Demo vs BC Demo

Smoke demo:

- 目的:验证硬件、录制、相机和 action timestamp。
- 数量:2-10 条短 episode。
- 可以缺少完整 target conditioning 字段。
- 不用于 policy training。

BC demo:

- 目的:训练初始 `π_arm`。
- 数量:第一阶段目标是 20-50 条成功 demo + 10-20 条 hard negative。
- 必须包含 target-conditioned 字段,或能转换出这些字段。
- 必须通过上面的数据质量 gate。
