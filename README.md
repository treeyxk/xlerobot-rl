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

## M1 v0 色块语义执行层

M1 v0 是当前最小语义 demo: 规则解析红/蓝/绿色块指令,用 HSV/SAM2 grounding 输出
`SemanticExecutiveState`,并标注 target / distractors、`top_grasp` 和 `READY_TO_GRASP`。

```bash
# HSV baseline
python scripts/sanity/demo_m1_v0.py --instruction "抓红色色块" --check

# 使用 SAM2 refinement
python scripts/sanity/demo_m1_v0.py --instruction "抓红色色块" --use-sam2 --check
```

输出文件:

- `data/debug/m1_v0_semantic_state.json`
- `data/debug/m1_v0_grounding_overlay.png`

当前限制:

- 只支持红/蓝/绿色块,不支持开放词汇物体或复杂 referring expression。
- HSV baseline 可能误检同色机器人部件;这属于 M1 v0 已知限制,后续用 VLM verifier / detector /
  tracker 替换或增强。
- 不做目标跟踪、语义导航、抓取执行和安全审批;这些分别属于后续 M1.3、M2、M4、M5。

## M4 target-conditioned 抓取环境

```bash
python scripts/sanity/check_m4_target_env.py --target-color red
python scripts/eval/eval_env_random.py --env-id TargetConditionedArmGrasp-v0 --n-episodes 20 --num-envs 4
```

环境设计见 `docs/m4_target_conditioned_env.md`。BC 数据格式见
`docs/bc_dataset_schema.md` / `docs/bc_dataset_schema_zh.md`。

## 今日右臂主从链路

当前真机 smoke 已固定并验证:

- `leader`: `/dev/xlerobot_left_leader`
- `follower`: `/dev/xlerobot_right_follower`
- `head camera`: `/dev/xlerobot_head_camera`
- camera profile: `1280x720@30`, `h264`

```bash
# 1. 枚举 USB 串口
python scripts/deploy/right_arm_master_slave.py ports

# 2. 生成校准 / teleop / 采集命令
python scripts/deploy/right_arm_master_slave.py commands \
  --leader-port /dev/xlerobot_left_leader \
  --follower-port /dev/xlerobot_right_follower \
  --camera-index /dev/xlerobot_head_camera \
  --camera-width 1280 \
  --camera-height 720
```

详细步骤见 `docs/right_arm_master_slave.md`。

## BC demo 录制流程

已验证配置见 `configs/real/xlerobot_right_arm_720p.yaml`。默认 dry-run 会使用固定设备名和
`1280x720@30 h264`,生成 metadata 模板并打印 LeRobot 命令:

```bash
python scripts/deploy/record_bc_demo.py \
  --target-color red \
  --dataset-name m4_target_grasp_v0_720p_smoke
```

可用下面命令检查相机枚举:

```bash
lerobot-find-cameras opencv
```

确认校准和 10s teleop smoke test 通过后,才添加 `--run-record` 实际录制:

```bash
python scripts/deploy/record_bc_demo.py \
  --target-color red \
  --dataset-name m4_target_grasp_v0_720p_smoke_runN \
  --num-episodes 1 \
  --episode-time-s 10 \
  --run-record
```

实际录制前脚本会打印准备 checklist,并等待按键确认:

- 按 `Space` 开始录制。
- 按 `q` 取消,不会启动 `lerobot-record`。
- 如需自动化跳过确认,添加 `--no-ready-prompt`。

按 `Space` 后,脚本会先抓取一帧 D435i RGB-D target snapshot,将红块的
`target_pos_base_initial_m` 写入 `data/bc/<dataset_name>/dataset_info.yaml`,并保存 debug 图到
`data/bc/<dataset_name>/target_snapshots/`。如果这一步必须成功才允许录制,添加:

```bash
--require-target-snapshot
```

连续采集多条 demo 时,使用交互模式:

```bash
python scripts/deploy/record_bc_continuous.py \
  --target-color red \
  --episode-time-s 20
```

交互控制:

- setup 阶段: 移动左臂,右臂跟随;`Space` 锁住并记录 ready pose。
- 每条 demo 前: 左右臂自动回 ready pose 并锁住;`Space` 开始录制。
- 录制中: `Space` 结束当前 demo。
- `y`: 保存刚录完的一条。
- `n`: 丢弃刚录完的一条。
- `q`: 退出连续采集。

不传 `--dataset-name` 时会自动生成 `m4_target_grasp_v0_bc_session_<timestamp>`。这个脚本在同一个
Python 进程里保持 leader/follower/camera/dataset 连接,避免外部 `lerobot-record` 反复启停造成的
torque gap。

保存的数据会写入同一个 LeRobot dataset root:

```text
data/real/lerobot/m4_target_grasp_v0_bc_session_<timestamp>/
data/bc/m4_target_grasp_v0_bc_session_001/continuous_session.json
```

脚本默认使用 `h264`,便于后续用 OpenCV 或常规播放器检查视频。
原始 LeRobot 数据写入 `data/real/lerobot/<dataset_name>/`,伴随 metadata 模板写入
`data/bc/<dataset_name>/dataset_info.json`。

录制结束后运行数据完整性检查:

```bash
python scripts/sanity/check_lerobot_dataset.py \
  --dataset-root data/real/lerobot/m4_target_grasp_v0_720p_smoke_runN \
  --expect-episodes 1 \
  --expect-fps 30 \
  --expect-width 1280 \
  --expect-height 720
```

合并多个本地 LeRobot dataset 时使用:

```bash
python scripts/data_collection/merge_lerobot_datasets.py \
  --dataset-name m4_target_grasp_v0_bc_red_25ep_merged_20260520 \
  --source-roots \
    data/real/lerobot/m4_target_grasp_v0_bc_session_A \
    data/real/lerobot/m4_target_grasp_v0_bc_session_B
```

合并脚本会重新解码已保存 episode 并重写视频,可以清理早期采集脚本中 discarded trial
图片混入 mp4 的问题。合并后再次运行 `check_lerobot_dataset.py`。

## BC overfit 训练

BC overfit 用来验证“数据读取 -> 图像/状态输入 -> action 输出 -> loss 下降”链路是否打通。
它不是泛化训练,而是先让一个小模型背下当前数据集。

```bash
python scripts/train/train_bc_overfit.py \
  --dataset-root data/real/lerobot/m4_target_grasp_v0_bc_red_25ep_merged_20260520 \
  --output-dir outputs/bc_overfit/red_25ep_v0 \
  --epochs 30 \
  --batch-size 32
```

输出:

- `outputs/bc_overfit/red_25ep_v0/config.json`
- `outputs/bc_overfit/red_25ep_v0/metrics.csv`
- `outputs/bc_overfit/red_25ep_v0/checkpoint_last.pt`

判断标准: `loss_norm_mse` 应明显下降,`action_mae_deg` 应随 epoch 降低。若 overfit 都降不下去,
先检查视频/action 是否错位、任务是否混杂、或 action/state 维度是否对应。

## BC policy 离线/真机测试

离线测试用训练好的 checkpoint 在独立测试集上只计算误差,不控制机械臂:

```bash
python scripts/eval/eval_bc_policy.py \
  --checkpoint outputs/bc_overfit/red_62ep_final_v0/checkpoint_last.pt \
  --dataset-root data/real/lerobot/m4_target_grasp_v0_bc_red_test_10ep_merged_20260521 \
  --output-dir outputs/bc_eval/red_62ep_on_test_10ep_20260521
```

真机测试先 dry-run。dry-run 会读取右臂状态和头部相机,运行 policy 并写日志,但不发动作:

```bash
python scripts/deploy/run_bc_policy.py \
  --checkpoint outputs/bc_overfit/red_62ep_final_v0/checkpoint_last.pt \
  --duration-s 5 \
  --action-scale 0.3 \
  --max-delta-deg 2.0
```

dry-run 正常后再短时低速执行:

```bash
python scripts/deploy/run_bc_policy.py \
  --checkpoint outputs/bc_overfit/red_62ep_final_v0/checkpoint_last.pt \
  --duration-s 5 \
  --action-scale 0.2 \
  --max-delta-deg 1.0 \
  --execute
```

执行模式必须保持急停/断电可触达。先空桌测试,再放红块做短程测试。

当前固定的 BC v0 baseline:

```text
outputs/bc_baselines/bc_v0_red_cube_20260521/checkpoint.pt
```

该 checkpoint 已完成 62ep 训练、10ep 独立测试和低速真机 rollout。它作为后续
reward-guided fine-tune / RL 的初始化,不是最终抓取策略。失败数据补采计划见
`docs/reward_dataset_collection_zh.md`。

## 真机红块 grounding sanity

头部 D435i 内参和 hand-eye 外参固定后,可用真实 RGB-D 直接输出红块在 base frame 下的位置:

```bash
python scripts/sanity/detect_real_red_cube.py --cube-size-m 0.03
```

结果会保存到 `data/real/sanity/red_cube_detector/`,并输出 `GroundedObject` 风格的
`pos_camera_m` / `pos_base_m`。同一套几何逻辑在 `xlerobot_rl.real.camera_geometry` 和
`xlerobot_rl.real.red_cube_detector` 中复用。
