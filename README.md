# XLeRobot VLM-Guided Mobile Manipulation

> **Skill-aware navigation for VLM-guided mobile manipulation on low-cost hardware.**
> 让 mobile base 停在"当前 manipulator policy 实际能成功抓取"的位置，而不只是几何上靠近物体。

[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()
[![Target](https://img.shields.io/badge/target-IROS%202027-blue)]()

## 项目概述

基于 XLeRobot 0.4.0 硬件的 VLM 引导移动抓取系统。主要技术栈:

- **VLM 高层**: 物体检测 + 语义理解 (frozen pretrained)
- **导航 RL** (π_nav): 学习 skill-aware stopping
- **可达性预测器** (S): Ensemble-based success predictor with uncertainty
- **操作 RL** (π_arm): IL bootstrap + PPO fine-tune
- **任务编排器**: Uncertainty-aware state machine


## 当前状态

| 阶段 | 状态 | 备注 |
|------|------|------|
| Day 1: M0 接口冻结 | 🟡 进行中 | repo + env setup 已完成 |
| Day 2: 基础环境 | ✅ 完成 | ManiSkill + PyTorch cu128 sanity check 通过 |
| Day 3: 真机只读测试 plan | ⬜ 待做 | |
| Day 4: URDF 接入 ManiSkill | 🟡 进行中 | URDF + right-arm grasp env 已接入 |
| Day 5: 右臂主从采集链路 | 🟡 进行中 | 左臂拆下作为 SO101 leader, 右臂作为 follower |
| ... | | |


## 环境要求

- Python 3.11
- CUDA 12.8+ (Blackwell GPU 如 5070 Ti 需要)
- Ubuntu 22.04+ (其他 Linux 应该也行，未测试)
- NVIDIA driver ≥ 560
- ≥ 12GB VRAM (训练时)

## 安装

```bash
# 1. Clone
git clone git@github.com:treeyxk/xlerobot-rl.git
cd xlerobot-rl

# 2. 创建 conda 环境
conda create -n xlerobot-rl python=3.11 -y
conda activate xlerobot-rl

# 3. 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 安装项目 + dev + 真机依赖
pip install -e ".[dev,real]"

# 5. 验证安装
python scripts/sanity/test_setup.py
```

最后一步预期输出 `✓ All systems go`。

## 今日右臂主从链路

```bash
# 1. 枚举 USB 串口
python scripts/deploy/right_arm_master_slave.py ports

# 2. 生成校准 / teleop / 采集命令
python scripts/deploy/right_arm_master_slave.py commands \
  --leader-port /dev/ttyACM0 \
  --follower-port /dev/ttyACM1 \
  --camera-index 0
```

详细步骤见 `docs/right_arm_master_slave.md`。
