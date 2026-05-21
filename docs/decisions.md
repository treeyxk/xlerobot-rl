# Architecture Decision Records

本文档记录项目中的重要架构决策。每条 ADR 一旦写完不要修改 —— 决策变了写新的 ADR 并把旧的 status 改为 `Superseded`。

## ADR 索引

| #   | Title                          | Date       | Status   |
|-----|--------------------------------|------------|----------|
| 001 | 不使用 ROS 作为通信中间件        | 2026-05-13 | Accepted |
| 002 | Reward classifier 使用持续时间序列判定 | 2026-05-21 | Accepted |
| 003 | 真机头部相机统一使用 D435i 720p 配置 | 2026-05-21 | Accepted |

---

## ADR-001: 不使用 ROS 作为通信中间件

**Date**: 2026-05-13
**Status**: Accepted

### Context

需要决定整个 sim2real pipeline 的通信架构和系统集成方式。可选方案:

- **ROS / ROS2**: Robotics 主流，生态丰富 (SLAM, MoveIt, nav2 等现成)
- **Pure Python**: 直接调用 Python API,不引入额外 middleware (LeRobot 风格)
- **自造轮子**: zmq / gRPC 等通用 IPC

项目实际需求:

- Sim 阶段用 ManiSkill (Python-native, GPU 并行)
- 真机阶段用 LeRobot API 控制 SO101 + Lekiwi (Python-native, USB serial / WiFi)
- 控制频率 20-50 Hz 量级,非工业级 real-time 需求
- 不涉及 SLAM 建图、复杂 motion planning 或多机协同
- 单人开发,需要降低系统复杂度

### Decision

**整个 stack 使用 Python-native 架构,不引入 ROS。**

具体来说:

- Sim: ManiSkill,所有 observation / action 通过 Python tensor 流动
- Real: LeRobot Python API,直接控制硬件
- 通过抽象的 `RobotInterface` (M0) 让 sim/real 在上层代码中保持一致
- 一个 Python 主进程跑所有模块 (VLM, π_nav, π_arm, S, orchestrator)
- 相机使用 `pyrealsense2` / `opencv` 直接读取

### Consequences

**优点**:

- 无 IPC overhead, control loop 简洁
- Debug 只需要 attach 一个 Python process
- 不依赖 ROS distro / Ubuntu 版本绑定
- 与 LeRobot / ManiSkill 设计哲学契合
- 避免 ROS 学习曲线,降低单人项目复杂度

**缺点**:

- 无法直接复用 ROS 生态 (cartographer, MoveIt, nav2)
- 未来若接入 LiDAR 等只有 ROS driver 的传感器,需要自己适配
- 多机器人协同场景需重新评估通信方案
- 失去 rosbag 等成熟工具 —— 需要自己实现 trajectory recorder (作为 §10 logging 的一部分)

### Reversal Conditions

以下情况应重新考虑此决策:

- 项目需要加入 SLAM 建图 (室内 mapping, long-range navigation)
- 接入多机器人协同任务
- 部署到产品级 robot (工业部署链路通常依赖 ROS2)
- 接入只有 ROS driver 的关键传感器/硬件

### References

- LeRobot 设计哲学: https://github.com/huggingface/lerobot
- ManiSkill 文档: https://maniskill.readthedocs.io/


- **TCP local offset 数值**: `[0.0, -0.107, 0.0]` in Fixed_Jaw frame (沿 -Y 10.7cm)。来源: `scripts/diagnostic/measure_tcp_offset.py` 在 viewer 中可视化验证。旧 grasp_demo 项目用过 9cm, 实测应为 10.7cm。差 1.7cm 接近一个 cube 半径, 必须用 10.7cm 否则 grasp 失败率飙升。

---

## ADR-002: Reward classifier 使用持续时间序列判定

**Date**: 2026-05-21
**Status**: Accepted

### Context

M4 真实红块任务需要一个最小 reward/success classifier,用于区分:

- `success`: 红块被稳定抓住/带起。
- `fail_miss_target`: 接近但未夹到红块。
- `fail_wrong_object`: 抓到蓝/绿 distractor。
- `fail_knock_push`: 推走或撞偏红块。
- `fail_grasp_drop`: 中途夹到或半夹到,但最终掉落。

第一版单窗口 classifier (`scripts/train/train_reward_classifier.py`) 在训练集内部验证很快达到
100%,但独立 holdout 上对 success 泛化差:

- 末尾窗口版本: failure 8/8 正确,success 0/3 正确。
- 末尾前约 1.5s offset 版本: failure 8/8 正确,success 1/3 正确。
- 末尾前约 2.0s offset 版本: failure 8/8 正确,success 1/3 正确。
- 末尾前约 3.3s offset 版本: success 3/3 正确,但 failure 仅 3/8 正确。

根因是 final/current observation 不一定直接可见红块是否仍在夹爪中:红块可能被夹爪或机械臂遮挡,
也可能短暂离开画面。单帧/单窗口 reward 对这种视觉遮挡过于脆弱。

同时不能用 "整段视频中任意一帧像成功就判成功" 的 ever-grasped 逻辑,因为
`fail_grasp_drop` 在中途也可能出现抓住红块的帧,但 episode 最终仍然失败。

offset 扫描进一步确认:窗口越往前,success 越容易识别,但 `fail_grasp_drop` 和
`fail_knock_push` 也越容易被误判为 success。单窗口模型无法同时表达
"曾经接触/夹到" 和 "最终持续成功" 这两个不同概念。

### Decision

Reward classifier v0 使用**有序时间序列**判定稳定成功,而不是单帧判定或 ever-grasped 判定。

当前实现:

- 训练脚本: `scripts/train/train_reward_sequence_classifier.py`
- 评估脚本: `scripts/eval/eval_reward_sequence_classifier.py`
- 数据入口: `configs/reward/reward_dataset_v0.json`
- holdout 入口: `configs/reward/reward_holdout_v0.json`
- 默认建议: 从整条 episode 均匀抽取 32 帧,覆盖接近、夹取、带起和结束状态。
- 模型 v0: 每帧 image+state 编码,经过 GRU 聚合,输出 episode-level success/failure。

标签语义固定为:

```text
success = 后段持续稳定完成红块抓取/带起
failure = miss / wrong_object / knock_push / grasp_drop 任一失败类型
```

### Consequences

优点:

- 能利用动作过程中的时序证据,缓解最终单帧遮挡。
- 能区分 "中途夹到过" 和 "后段持续成功",避免把 `fail_grasp_drop` 误当成功。
- 与当前真实数据的采集方式更一致:人类判断 episode 成败本来就是看一段过程,不是只看一帧。

缺点:

- 在线 RL 使用时需要维护最近 2-4s observation history buffer。
- reward 会有时间窗口/低频判定属性,不适合作为无历史的瞬时 dense reward。
- 计算量和数据质量要求高于单帧模型。
- 如果作为整条 episode outcome reward 使用,它不能直接提供每一步 dense reward;后续需要 rollout buffer 或 episode-end credit assignment。

### Reversal Conditions

以下情况可以重新评估此决策:

- 后续加入可靠的 3D object pose / tactile / gripper contact signal,可以直接判定当前是否稳定抓住目标。
- VLM 或 grounding 模块能稳定解决遮挡和目标身份问题。
- RL 实验证明序列 reward 的延迟过大,需要回退到更严格采集约束下的 current-state classifier。

---

## ADR-003: 真机头部相机统一使用 D435i 720p 配置

**Date**: 2026-05-21
**Status**: Accepted

### Context

项目早期按 D415 做过占位配置和测试。后续实际硬件更换为 RealSense D435i,并完成:

- udev 固定设备名: `/dev/xlerobot_head_camera`
- RGB 录制 smoke: 1280x720@30, h264, OpenCV decode OK
- 内参导入: `configs/calibration/head_camera_intrinsics_1280x720.yaml`
- 外参/红块 RGB-D 验证脚本: `scripts/calibration/verify_head_camera_extrinsics_rgbd.py`
- 真实采集脚本默认相机: `scripts/deploy/record_bc_continuous.py` 和 `scripts/deploy/record_bc_demo.py`

### Decision

真实 BC/reward 数据采集默认使用 D435i 头部相机:

```text
device: /dev/xlerobot_head_camera
resolution: 1280x720
fps: 30
video codec: h264
```

D415 相关配置不再作为当前主线默认值。后续脚本和文档里如果需要真实头部相机,默认应引用
`/dev/xlerobot_head_camera` 和 `configs/calibration/head_camera_intrinsics_1280x720.yaml`。

### Consequences

优点:

- 720p 对红块、AprilTag、夹爪遮挡判断更稳,也与当前采集数据一致。
- 固定 udev 名避免 `/dev/video*` 编号漂移导致录错相机。
- 后续 BC、reward、RGB-D grounding 可以共享同一套 intrinsics/extrinsics。

缺点:

- 720p 训练和视频解码开销高于 640x480。
- 若以后做高频在线视觉 policy,可能需要 resize 或降采样输入。
- 如果更换相机或 USB 拓扑,必须重新验证 udev、intrinsics、extrinsics 和数据集 schema。

### Reversal Conditions

以下情况可以重新评估:

- 在线推理延迟超过可接受范围,且降采样仍不能满足控制频率。
- D435i 安装位姿或硬件损坏导致必须换相机。
- 后续引入 wrist camera 作为主视觉源,head camera 只做辅助观测。
