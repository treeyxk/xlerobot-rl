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

当前已验证的固定设备名:

```text
left_leader     -> /dev/xlerobot_left_leader
right_follower  -> /dev/xlerobot_right_follower
head_camera     -> /dev/xlerobot_head_camera
camera profile  -> 1280x720@30, h264
```

项目记录配置见 `configs/real/xlerobot_right_arm_720p.yaml`。

### 3.0 固定 udev 设备名

本机已验证的硬件 serial:

| 设备 | 稳定路径 | serial / 匹配条件 |
|------|----------|-------------------|
| right follower | `/dev/xlerobot_right_follower` | `5B14028939` |
| left leader | `/dev/xlerobot_left_leader` | `5B14112340` |
| RealSense head RGB | `/dev/xlerobot_head_camera` | `123423024637`, USB interface `03`, V4L capture |

串口规则:

```bash
sudo tee /etc/udev/rules.d/99-xlerobot-serial.rules >/dev/null <<'EOF'
KERNEL=="ttyACM*", SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B14028939", SYMLINK+="xlerobot_right_follower", MODE="0660", GROUP="dialout", TAG+="uaccess"
KERNEL=="ttyACM*", SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B14112340", SYMLINK+="xlerobot_left_leader", MODE="0660", GROUP="dialout", TAG+="uaccess"
EOF
```

相机规则:

```bash
sudo tee /etc/udev/rules.d/99-xlerobot-camera.rules >/dev/null <<'EOF'
KERNEL=="video*", SUBSYSTEM=="video4linux", ENV{ID_VENDOR_ID}=="8086", ENV{ID_MODEL_ID}=="0ad3", ENV{ID_SERIAL_SHORT}=="123423024637", ENV{ID_USB_INTERFACE_NUM}=="03", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="xlerobot_head_camera", MODE="0660", GROUP="video", TAG+="uaccess"
EOF
```

刷新并重新插拔 USB:

```bash
sudo rm -f /dev/xlerobot_left_leader /dev/xlerobot_right_follower /dev/xlerobot_head_camera
sudo udevadm control --reload-rules
sudo udevadm trigger
```

验证:

```bash
ls -l /dev/xlerobot_*
```

预期:

```text
/dev/xlerobot_left_leader -> ttyACM?
/dev/xlerobot_right_follower -> ttyACM?
/dev/xlerobot_head_camera -> video?
```

注意: udev rule 必须一条规则写在一整行。不要在 nano 中把同一条 rule 拆成多行,
否则可能会错误地创建到 `bus/usb/...` 而不是 `ttyACM?`。

本轮综合 smoke 通过的数据集:

```text
data/real/lerobot/m4_target_grasp_v0_720p_fixed_devices_smoke_run2
```

检查结果:

```text
episodes: 1
frames: 300
duration: 10s
video: 1280x720 h264
OpenCV decode: OK
action/state shape: 6
```

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
  --leader-port /dev/xlerobot_left_leader \
  --follower-port /dev/xlerobot_right_follower \
  --camera-index /dev/xlerobot_head_camera \
  --camera-width 1280 \
  --camera-height 720
```

脚本会打印校准、短时 teleop 和短 episode 录制命令。

### 3.3 校准

先校准 leader:

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/xlerobot_left_leader \
  --teleop.id=left_leader
```

再校准 follower:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/xlerobot_right_follower \
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
  --robot.port=/dev/xlerobot_right_follower \
  --robot.id=right_follower \
  --robot.max_relative_target=15 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/xlerobot_left_leader \
  --teleop.id=left_leader \
  --fps=15 \
  --teleop_time_s=10 \
  --display_data=false
```

通过条件:

- 右臂跟随方向正确。
- 没有突然大幅跳变。
- gripper 开合方向正确。
- loop 频率接近目标频率。

如果方向反了, 先停止, 不要靠手硬掰 follower。需要重新检查校准和左右臂装配方向。

### 3.5 采集一条最小数据

确认 teleop 正常后, 用项目脚本采 2 个 15 秒 episode。先 dry-run 检查命令和 metadata:

```bash
python scripts/deploy/record_bc_demo.py \
  --target-color red \
  --dataset-name m4_target_grasp_v0_720p_smoke \
  --num-episodes 1 \
  --episode-time-s 10
```

实际录制时添加 `--run-record`。脚本会在启动 `lerobot-record` 前打印 checklist,
等待按 `Space` 开始;按 `q` 会取消录制。默认视频编码为 `h264`,便于本地检查。

录制结束后检查数据是否完整 finalize:

```bash
python scripts/sanity/check_lerobot_dataset.py \
  --dataset-root data/real/lerobot/<dataset_name> \
  --expect-episodes 1 \
  --expect-fps 30 \
  --expect-width 1280 \
  --expect-height 720
```

检查通过时应看到 `Result: PASS`。如果缺少 `meta/episodes`、`meta/stats.json` 或标准
`videos/.../file-000.mp4`,说明该 run 没有完成 finalize,不要用于训练。

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
