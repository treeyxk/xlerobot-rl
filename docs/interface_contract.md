# Interface Contract (M0)

本文档定义项目所有模块间的接口契约。**一旦冻结,改动需走 ADR 流程,所有模块同步更新。**

**Status**: 🟡 Partial (Part 1: 坐标系 + 单位 + 频率 已锁定; Part 2: M1 v0 Semantic Executive schema 已冻结)

待补充部分:

- [ ] TargetObservation dataclass schema
- [x] M1 v0 Semantic Executive input/output schema
- [ ] M2/M3/M4/M5 input/output schema
- [ ] Action bounds (具体数值)
- [ ] Failure mode codes

---

## 1. 坐标系约定

### 1.1 命名规范

所有坐标系用小写 snake_case 命名。Transform 表示用 `T_<parent>_<child>` 格式,语义为"child 在 parent frame 下的位姿"。

例: `T_base_object` = 物体相对于 base frame 的位姿。

### 1.2 标准坐标系

| Frame 名称       | 描述                              | 备注                          |
|-----------------|----------------------------------|------------------------------|
| `world`         | 世界坐标系                        | 与机器人初始 base pose 重合,Z 朝上 |
| `base`          | 机器人底盘坐标系                   | X 前进方向, Y 左, Z 上          |
| `head_cam`      | 头部相机坐标系                     | 用于 VLM 输入和全局观测         |
| `wrist_cam_left`  | 左臂腕部相机坐标系               | 抓取阶段近距视野               |
| `wrist_cam_right` | 右臂腕部相机坐标系               | 抓取阶段近距视野               |
| `ee_left`       | 左臂末端执行器坐标系               | EE-space action 参考          |
| `ee_right`      | 右臂末端执行器坐标系               | EE-space action 参考          |
| `object`        | 目标物体坐标系                     | 由 VLM 输出,通常 origin 在物体几何中心 |

### 1.3 坐标系朝向 (Right-handed)

所有坐标系遵循右手系。`base` frame 的朝向定义为黄金标准:

- **X 轴**: 机器人正前方 (forward)
- **Y 轴**: 机器人左侧 (left)
- **Z 轴**: 垂直向上 (up)

相机坐标系遵循 OpenCV 约定:

- **X 轴**: 图像右方
- **Y 轴**: 图像下方
- **Z 轴**: 相机光轴方向 (depth 增长方向)

### 1.4 关键 Transform

以下 transform 必须 well-defined,代码中提供常数或标定数据:

| Transform              | 来源          | 备注                                |
|-----------------------|---------------|------------------------------------|
| `T_world_base`        | 运行时 (odometry) | π_nav 的 proprioception          |
| `T_base_head_cam`     | URDF + 标定   | 真机需要外参标定                    |
| `T_base_wrist_cam_*`  | URDF + 标定   | 同上                               |
| `T_base_ee_*`         | URDF + FK     | 通过正向运动学计算                  |
| `T_head_cam_object`   | VLM 输出      | mask + depth 反投影                |
| `T_base_object`       | 运行时计算    | `T_base_head_cam @ T_head_cam_object` |

---

## 2. 单位约定

**统一使用 SI 单位**, 不允许混用。下游代码假设所有数值都是 SI。

| 物理量       | 单位      | 示例                          |
|-------------|----------|------------------------------|
| 长度         | 米 (m)   | `object.pos[2] = 0.75` (75cm 高) |
| 角度         | 弧度 (rad) | `joint_angle = 1.57` (~90°)   |
| 时间         | 秒 (s)   | `dt = 0.05` (50ms)           |
| 线速度       | m/s      | base velocity                |
| 角速度       | rad/s    | base angular velocity        |
| 力           | N        | (暂未用)                      |
| 力矩         | N·m      | (暂未用)                      |
| 质量         | kg       | URDF 中物体质量               |

**例外: 像素**

图像坐标使用像素 (int), 但 mask 和 bbox 等始终标注 "in pixels"。

```python
# 正确
bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
position: np.ndarray              # (3,) in meters

# 错误 - 混用 cm 和 m 是禁止的
distance_cm = 75  # ❌
```

---

## 3. 控制频率约定

不同模块运行在不同频率上。**所有跨模块通信通过最新值传递,不要求严格同步。**

| 模块             | 频率      | 备注                                   |
|-----------------|----------|--------------------------------------|
| Sim 物理引擎     | 100-200 Hz | ManiSkill 内部                       |
| 真机底层 control | 50 Hz    | LeRobot API 实际下发频率              |
| π_arm inference  | 20 Hz    | 抓取需要较快反应                       |
| π_nav inference  | 10 Hz    | 导航不需要很高频率                     |
| VLM 检测         | 5 Hz     | VLM inference 较慢, 异步运行          |
| S predictor      | 与 π_nav 同步 | 每次 nav step 调用一次              |
| 编排器状态切换    | 5 Hz     | 决策粒度不需要太细                     |
| Logging          | per step / per episode | 见 logging_schema.md     |

### 3.1 频率失配处理

由于模块频率不一致,需要明确"过期数据"如何处理:

- **VLM 输出过期**: 若超过 0.5s 未收到新 detection, 触发重新检测 (orchestrator 处理)
- **S 输出**: 总是使用最新值, 不缓存
- **Action 下发**: 若 policy inference 超时 (> 1.5 / control_freq), 重复上一次 action (watchdog)

### 3.2 时间戳要求

所有 observation / action / event 在产生时必须带 timestamp (UTC seconds, float)。Logging 系统依赖这个字段对齐数据。

```python
@dataclass
class TimedEvent:
    timestamp: float  # UTC seconds since epoch
    # ... 其他字段
```

---

## Part 1 结束

---

## 4. M1 v0 Semantic Executive Schema

**Status**: 冻结 v0。用于 Month 1 / W2 色块 demo。后续 tracker、semantic nav、open-vocabulary grounding
若需要改字段名或字段语义,需走 ADR 并同步所有消费方。

M1 v0 的边界:

- 支持输入色块抓取指令,例如 `"抓红色色块"` / `"抓蓝色色块"` / `"抓绿色块"`。
- 支持 HSV/SAM2 grounding 输出 target 与 distractors。
- 输出 `top_grasp` skill proposal 与 `READY_TO_GRASP` navigation mode。
- 不负责真正导航、抓取控制、状态机审批或安全限幅;这些属于 M2/M4/M5。

代码定义位置:

```python
from xlerobot_rl.interfaces import TargetState, SemanticExecutiveState
```

M1 v0 模块实现位置:

```python
from xlerobot_rl.modules.semantic_executive import ColorBlockSemanticExecutive
```

### 4.1 TargetState

单个语义接地物体实例。M1 v0 中 target 和 distractor 都使用同一 schema。

```python
@dataclass
class TargetState:
    object_id: int
    name: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    mask: np.ndarray                # (H, W) bool
    pos_camera: np.ndarray          # (3,) xyz in head_cam frame, meters
    pos_base: np.ndarray            # (3,) xyz in base/world frame, meters
    confidence: float               # [0, 1]
    attributes: dict                # e.g. {"color": "red", "category": "cube"}
    is_target: bool
    detection_method: str           # "hsv-only" / "hsv+sam2" / ...
    last_seen_timestamp: float      # UTC seconds
```

字段约定:

| 字段 | 说明 |
|------|------|
| `object_id` | M1 当前帧内唯一 ID。M1.3 tracker 引入前,不保证跨帧稳定。 |
| `bbox` | 像素坐标,闭开区间语义 `(x1, y1, x2, y2)`。 |
| `mask` | binary mask,序列化到 JSON 时默认不包含,避免日志过大。 |
| `pos_camera` | OpenCV camera optical frame,单位米。 |
| `pos_base` | base/world frame,单位米。当前 sim 中 world 与 base 重合。 |
| `is_target` | 当前用户指令指定的目标实例为 `True`;其他候选物体为 distractor。 |

### 4.2 SemanticExecutiveState

M1 v0 的唯一正式输出。M5/M2/M3/M4 后续消费 M1 结果时应依赖这个 schema,而不是解析临时 JSON。

```python
NavigationMode = Literal[
    "SEARCH", "SEMANTIC_NAV", "SKILL_AWARE_APPROACH",
    "READY_TO_GRASP", "GRASP", "DONE", "SAFE_STOP",
]

ExecutionStatus = Literal[
    "idle", "tracking", "ready", "executing", "success", "failure",
]

FailureReason = Literal[
    "none", "target_not_found", "low_confidence", "ambiguous_target",
    "wrong_object", "target_lost", "unsafe", "timeout",
]

@dataclass
class SemanticExecutiveState:
    instruction: str
    task_graph: list[str]
    current_subgoal: str
    target: TargetState | None
    scene_objects: list[TargetState]
    candidate_skills: list[str]
    selected_skill: str | None
    navigation_mode: NavigationMode
    execution_status: ExecutionStatus
    failure_reason: FailureReason
    search_goal: np.ndarray | None
    local_nav_goal: np.ndarray | None
    success_scores: dict[str, float]
    uncertainty_scores: dict[str, float]
    timestamp: float
```

M1 v0 默认值:

| 字段 | M1 v0 约定 |
|------|------------|
| `task_graph` | `["find_target", "navigate_to_skill_success_region", "grasp_target", "verify_success"]` |
| `current_subgoal` | `"find_target"` |
| `candidate_skills` | `["top_grasp"]` |
| `selected_skill` | `"top_grasp"` |
| `navigation_mode` | 目标已检测到时为 `"READY_TO_GRASP"` |
| `execution_status` | 正常输出为 `"ready"` |
| `failure_reason` | 正常输出为 `"none"` |

M1 v0 failure behavior:

| 场景 | `target` | `navigation_mode` | `execution_status` | `failure_reason` |
|------|----------|-------------------|--------------------|------------------|
| 指令中无法解析红/蓝/绿目标颜色 | `None` | `"SAFE_STOP"` | `"failure"` | `"ambiguous_target"` |
| 指令解析成功,但目标颜色未检测到 | `None` | `"SAFE_STOP"` | `"failure"` | `"target_not_found"` |
| 目标颜色候选多于 1 个 | `None` | `"SAFE_STOP"` | `"failure"` | `"ambiguous_target"` |
| 唯一目标候选置信度低于 `0.25` | `None` | `"SAFE_STOP"` | `"failure"` | `"low_confidence"` |

M1 v0 不抛出裸解析/检测异常给 M5。M5 只消费结构化 `SemanticExecutiveState` 并根据
`failure_reason` 决定 re-detect、replan 或 safe-stop。复杂 referring expression 由后续 VLM
planner/verifier 处理,不属于 M1 v0。

M1 v0 的最小 grounding gate 定义在:

```python
ColorBlockSemanticExecutive.evaluate_grounding_result()
```

该 gate 只检查 target 是否存在、是否唯一、confidence 是否达标;不做 VLM 语义验证、目标跟踪或桌面
几何过滤。

### 4.3 JSON 序列化约定

`SemanticExecutiveState.to_json_dict()` 用于 debug/logging。默认不序列化 `mask`;如需保存 mask,
使用单独图像/HDF5 字段。

M1 v0 sanity 命令:

```bash
python scripts/sanity/demo_m1_v0.py --instruction "抓红色色块"
```

输出:

- `data/debug/m1_v0_semantic_state.json`
- `data/debug/m1_v0_grounding_overlay.png`

以下为后续待写部分,**作为占位符标识尚未冻结的接口**:

- TargetObservation dataclass
- M2/M3/M4/M5 input/output schema
- Action bounds 具体数值
- Failure mode codes 完整列表

未冻结之前,代码中不要硬编码相关假设。
