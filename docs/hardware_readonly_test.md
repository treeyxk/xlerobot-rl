# 真机只读测试 Plan

**Status**: 📋 Plan (Week 2-3 执行)
**目的**: 在不部署任何 policy 的前提下,摸清 XLeRobot 真机的硬约束,为 sim 的 domain randomization 参数提供依据。
**前置条件**: 已完成 Day 1-2 (sim 端环境跑通)。
**最终产出**: `docs/hardware_readonly_report.md` (含定量数据 + 对 sim 参数的建议)

---

## 0. 测试前准备

- [ ] XLeRobot 通电检查 (battery, motor 温度, emergency stop 位置)
- [ ] Head camera (RealSense) 物理装配到位
- [ ] (可选) Wrist camera 装配到位
- [ ] PC 与机器人通信链路确认 (WiFi / USB)
- [ ] LeRobot Python API 能连上机器人 (只读模式, 不发 command)
- [ ] 准备外部测量工具: 卷尺 / 棋盘格 / 二维码标记

**安全前置**: 即使是只读测试,机器人也会通电运动 (测 odometry 和 latency 时)。准备好 emergency stop, 周围 1m 范围无人无物。

---

## 1. RealSense RGB-D 稳定性测试

**目的**: 量化深度噪声, 为 sim depth augmentation 提供参考。

**步骤**:

1. Head camera 装到位, 机器人静止
2. 摆放一个**已知形状物体** (比如平面挡板, 距 0.5m) 在 camera 前
3. 用 `pyrealsense2` 连续采集 **1 小时** RGB-D 数据 (30Hz, 累计 ~108000 帧)
4. 保存到 `data/real/depth_stability_<date>.h5`

**测量指标**:

| 指标 | 计算方式 | 预期数量级 |
|------|---------|-----------|
| Depth noise (单帧 std) | 同一像素位置, 同一物体, 单帧内 depth std | < 5mm @ 0.5m |
| Depth temporal drift | 同一像素的 depth 时序均值随时间的漂移 | < 2mm/hour |
| Frame drop rate | 实际收到帧数 / 期望帧数 (期望 = 时长 × 30Hz) | < 1% |
| Invalid depth pixel rate | depth = 0 或 inf 的像素比例 | < 5% |

**对 sim 的影响**: 上述噪声分布将作为 sim 的 depth observation noise model 参数。

---

## 2. 头部相机外参标定

**目的**: 算出 `T_base_head_cam` (head camera 相对于 base 的位姿), 让 sim 和真机几何一致。

**步骤**:

1. 在机器人 base frame 已知位置贴标定棋盘格 (比如 base origin 正前方 0.5m, 高度已知)
2. 用 head camera 拍棋盘格
3. OpenCV `cv2.solvePnP` 算出棋盘格在 camera frame 下的位姿
4. 通过已知的"棋盘格在 base frame 下的位置"反推 `T_base_head_cam`
5. 同时用 `cv2.calibrateCamera` 标定 intrinsics (fx, fy, cx, cy, 畸变系数)

**测量指标**:

| 指标 | 验收标准 |
|------|---------|
| Intrinsics 标定 reprojection error | < 0.5 pixels |
| 外参重测一致性 (拍 3 张照片分别算, 看一致性) | xyz 误差 < 5mm, rpy 误差 < 2° |

**输出**:

- `data/real/calibration/head_cam_intrinsics.yaml`
- `data/real/calibration/head_cam_extrinsics.yaml`
- (这两份文件未来在 sim 中用,让 sim camera 匹配真机)

---

## 3. Base Odometry 漂移测试

**目的**: 量化 odometry 累积误差, 决定 π_nav 能依赖 odometry 多久。

**步骤**:

1. 机器人静止在已知起点 (用胶带标记)
2. 通过 LeRobot API 让 base 走一个**预定义路径** (比如 1m × 1m 方形, 重复 5 圈)
3. 持续 10 分钟
4. 记录每个时刻的 odometry 报告位置
5. 结束后**手动测量**实际位置,对比

**测量指标**:

| 指标 | 计算方式 | 预期 |
|------|---------|------|
| 累积位置漂移 | 实际位置 vs odometry 报告位置, 单位 m/min | < 0.1 m/min |
| 旋转漂移 | 朝向角偏差 | < 1°/min |
| 起点-终点闭环误差 | 回到起点时 odometry 报告的位置 vs 实际 (0,0) | < 0.2m total |

**对 sim 的影响**: 漂移率 → π_nav observation 中 base pose 的 noise 强度。

**安全提示**: 这是第一次让真机自己动。**手动 hold emergency stop 全程监督**。

---

## 4. Action Latency 测试

**目的**: 测量 LeRobot API 下发 action → actuator 真正执行的延迟,决定 control loop 频率上限。

**步骤**:

1. 在 arm 上贴一个**视觉标记** (ArUco 码)
2. 用外部高速 camera (≥ 60Hz, 用手机即可) 录制 arm 运动
3. 通过 LeRobot 发送一个**阶跃 action** (joint 突然移动 10°)
4. 同步记录 API 发送时刻 + camera 看到 arm 开始动的时刻
5. 重复 20 次, 取中位数

**测量指标**:

| 指标 | 测量内容 | 预期 |
|------|---------|------|
| Command-to-motion latency | API send → arm 开始动 | < 50ms |
| 不同频率下的执行准确度 | 10/20/50 Hz 下下发 sin 波 action, 看实际跟踪 | RMS 误差 < 5° |
| 最高可靠控制频率 | 提高频率直到 actuator 跟不上 | TBD |

**对 sim 的影响**: 延迟值 → sim 训练时加入 action delay 模拟 (DR 一项)。

---

## 5. Wrist Camera 线缆干涉测试

**目的**: 确认 wrist camera 装配后, arm 全 range 运动不会拉断线缆。

**步骤**:

1. 装好 wrist camera + 走线
2. **缓慢**移动 arm 通过整个 joint range (每个 joint 单独测,然后组合)
3. 全程录视频
4. 检查关键风险点: 线缆是否被夹、过度弯折、拉拽

**验收标准**:

- 无线缆断裂 / 接头脱落
- 全 range 内 camera 不被遮挡 (lens 朝向正确)
- (定性观察) 线缆走向不会随时间疲劳

**修复方案** (若有问题):

- 重新走线 (用走线槽 / 扎带固定)
- 必要时改 camera mounting

---

## 6. Control Frequency 上限测试

**目的**: LeRobot 控制循环实际能稳定跑多快, 决定 sim 训练时 action frequency 上限。

**步骤**:

1. 写一个最小 loop: `while True: send_action(hold_pose); time.sleep(1/freq)`
2. 测试不同 freq (10 / 20 / 50 / 100 Hz)
3. 每个频率跑 5 分钟
4. 监控:
   - 实际下发频率 (vs 期望)
   - CPU / 通信带宽是否饱和
   - actuator 温度是否上升

**测量指标**:

| Freq | 实际频率达成 | 稳定性 (jitter std) | 状态 |
|------|------------|-------------------|------|
| 10 Hz | TBD | TBD | TBD |
| 20 Hz | TBD | TBD | TBD |
| 50 Hz | TBD | TBD | TBD |
| 100 Hz | TBD | TBD | TBD |

**对 contract 的影响**: 决定 `docs/interface_contract.md §3` 中真机 control freq 的具体数值。

---

## 7. Sim vs Real 视觉 Gap (定性)

**目的**: 直观感受 sim render 和真机 RGB 的差距, 为后续 visual domain randomization 提供方向。

**步骤**:

1. 在 sim 里搭一个**简单场景** (机器人 + 桌子 + 一个红色 cube), 用与真机一致的相机外参
2. 真机摆同样的场景 (真桌子 + 真红色 cube)
3. 各拍一张 head camera RGB, 存到 `data/real/visual_gap/`

**输出**:

- `sim_render.png`
- `real_photo.png`
- `gap_notes.md`: 列出差异 (光照 / 阴影 / 颜色饱和度 / 纹理 / 反光)

**对 sim 的影响**: 差异列表 → DR 重点应该加什么 (texture noise / lighting jitter / color shift)。

---

## Report 模板

测试全部完成后, 写 `docs/hardware_readonly_report.md`,包含:

```markdown
# Hardware Readonly Test Report

**Date**: YYYY-MM-DD
**Tester**: <name>
**Hardware**: XLeRobot 0.4.0 + RealSense D435i

## Summary

(1-2 段总结: 哪些假设被验证, 哪些被推翻)

## Quantitative Findings

| 测试项 | 关键数值 | 对 sim 的建议 |
|-------|---------|--------------|
| Depth noise | X mm @ 0.5m | ... |
| Odometry drift | X m/min | ... |
| Action latency | X ms | ... |
| Max control freq | X Hz | ... |

## Critical Issues Found

(任何 blocker, 需要硬件层修复的)

## Recommendations for Sim DR

(具体数值建议, 可直接填到 ManiSkill DR config 里)
```

---

## 执行优先级

如果时间紧, 按这个顺序做:

1. ⭐ §3 Odometry 漂移 (影响 nav design)
2. ⭐ §4 Action latency (影响 control loop design)
3. ⭐ §2 头部相机外参 (影响 sim/real 几何一致性)
4. §1 Depth 稳定性
5. §6 Control freq 上限
6. §5 线缆干涉 (机械问题, 不影响算法)
7. §7 Sim vs real 视觉 gap (定性,有时间再做)

