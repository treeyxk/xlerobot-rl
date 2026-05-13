# Interface Contract (M0)

本文档定义项目所有模块间的接口契约。**一旦冻结,改动需走 ADR 流程,所有模块同步更新。**

**Status**: 🟡 Partial (Part 1: 坐标系 + 单位 + 频率 已锁定)

待补充部分:

- [ ] TargetObservation dataclass schema
- [ ] 各模块 input/output schema
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

以下为后续待写部分,**作为占位符标识尚未冻结的接口**:

- §4 TargetObservation dataclass
- §5 各模块 I/O schema
- §6 Action bounds
- §7 Failure mode codes

未冻结之前,代码中不要硬编码相关假设。
