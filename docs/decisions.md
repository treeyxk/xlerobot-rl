# Architecture Decision Records

本文档记录项目中的重要架构决策。每条 ADR 一旦写完不要修改 —— 决策变了写新的 ADR 并把旧的 status 改为 `Superseded`。

## ADR 索引

| #   | Title                          | Date       | Status   |
|-----|--------------------------------|------------|----------|
| 001 | 不使用 ROS 作为通信中间件        | 2026-05-13 | Accepted |

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

