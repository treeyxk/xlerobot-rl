# Right-arm master-slave bring-up

**Status**: Day bring-up plan

目标: 后续训练只控制右臂。物理上拆下左臂后, 左臂作为 SO101 leader
主臂, 右臂作为 XLeRobot 上的 SO101 follower 从臂, 用 LeRobot 先跑通
teleoperation, 再采集 imitation 数据。

## 1. 命名约定

| 名称 | 含义 | LeRobot type | 备注 |
|------|------|--------------|------|
| `right_follower` | 右臂, 装在 XLeRobot 上, 训练/执行对象 | `so101_follower` | 会接收 action, 必须清空工作空间 |
| `left_leader` | 拆下的左臂, 手动操作主臂 | `so101_leader` | 只读关节位置, torque disabled |

项目里的 sim env 也遵循这个约定: `StaticArmGrasp-v0` 当前只控制右臂 6 维
joint delta。

## 2. 安全前置

1. 右臂安装牢固, 左臂已从机器人机械结构上拆下。
2. 右臂周围至少 0.5m 无障碍物, 桌面没有易碎物。
3. 右臂上电前先把 gripper 和各关节放到中间安全姿态。
4. 准备好断电/急停手段。
5. 第一次 teleop 必须使用短时长, 例如 15 Hz, 10 秒。

## 3. 今日最小路径

### 3.1 找串口

```bash
python scripts/deploy/right_arm_master_slave.py ports
```

如果没有列出 `/dev/ttyACM*` 或 `/dev/ttyUSB*`, 先检查 USB 线、权限和设备上电。

LeRobot 也提供交互式端口识别:

```bash
lerobot-find-port
```

分别拔插 leader 和 follower 控制板, 记录两个 port。

### 3.2 生成命令

把真实端口填进去:

```bash
python scripts/deploy/right_arm_master_slave.py commands \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyACM1 \
  --camera-index 0
```

脚本会打印校准、短时 teleop 和短 episode 录制命令。

### 3.3 校准

先校准 leader:

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=left_leader
```

再校准 follower:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=right_follower
```

校准时 LeRobot 会要求把关节移动到中间位置、记录运动范围。不要跳过这一步,
否则主从角度映射不可信。

### 3.4 短时 teleop smoke test

第一次只跑 10 秒。今天实测 `max_relative_target=15` + `fps=15` 能让
`shoulder_lift` 在负载下及时抬起; 之前的 `5` + `10 Hz` 过于保守, 会导致从臂追踪太慢。

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=right_follower \
  --robot.max_relative_target=15 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=left_leader \
  --fps=15 \
  --teleop_time_s=10 \
  --display_data=true
```

通过条件:

- 右臂跟随方向正确。
- 没有突然大幅跳变。
- gripper 开合方向正确。
- loop 频率接近目标频率。

如果方向反了, 先停止, 不要靠手硬掰 follower。需要重新检查校准和左右臂装配方向。

### 3.5 采集一条最小数据

确认 teleop 正常后, 采 2 个 15 秒 episode:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=right_follower \
  --robot.max_relative_target=15 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=left_leader \
  --dataset.repo_id=local/xlerobot_right_arm_smoke \
  --dataset.root=data/real/lerobot/xlerobot_right_arm_smoke \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick up the red cube with the right arm" \
  --dataset.push_to_hub=false \
  --dataset.video=true \
  --display_data=true
```

## 4. 与训练接口的关系

训练侧只消费右臂 6 维 action:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

仿真侧目前对应:

```text
Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R, Jaw_R
```

后续需要补一个 dataset converter, 把 LeRobot 的 follower observation/action 映射到
`xlerobot_rl.interfaces` 中的右臂 schema。今天先不做策略训练, 只验通硬件采集链路。
