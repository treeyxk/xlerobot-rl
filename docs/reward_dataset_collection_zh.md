# Reward / Success Classifier 数据采集计划

目标是为后续 `BC v0 -> reward-guided fine-tune / RL` 提供最小可用的成功/失败判别数据。
这些数据不直接混进 BC 成功示范训练集。

## 当前固定基线

- BC baseline: `outputs/bc_baselines/bc_v0_red_cube_20260521/checkpoint.pt`
- Train dataset: `data/real/lerobot/m4_target_grasp_v0_bc_red_62ep_final_20260521`
- Eval dataset: `data/real/lerobot/m4_target_grasp_v0_bc_red_test_10ep_merged_20260521`
- 真机状态: 能朝红块移动,有夹爪闭合趋势,但抓取/抬起不稳定。

## 数据目标

第一版 reward/success classifier 先做二分类:

| label | 目标条数 | 当前可用 | 还需补采 |
| --- | ---: | ---: | ---: |
| success | 25 | 25 | 0 |
| fail_miss_target | 10 | 10 | 0 |
| fail_wrong_object | 10 | 10 | 0 |
| fail_knock_push | 10 | 10 | 0 |
| fail_grasp_drop | 10 | 10 | 0 |

训练 manifest:

`configs/reward/reward_dataset_v0.json`

已有干净 success 数据:

`data/real/lerobot/reward_dataset/clean/success_25ep_20260521`

来源: 初始 10 条成功 BC 测试 episode + 后续补采 15 条 success patch episode。

已有干净 miss 数据:

`data/real/lerobot/reward_dataset/clean/fail_miss_target_10ep_20260521`

已有干净 wrong-object 数据:

`data/real/lerobot/reward_dataset/clean/fail_wrong_object_10ep_20260521`

已有干净 knock/push 数据:

`data/real/lerobot/reward_dataset/clean/fail_knock_push_10ep_20260521`

已有干净 grasp/drop 数据:

`data/real/lerobot/reward_dataset/clean/fail_grasp_drop_10ep_20260521`

原始采集数据统一放在:

`data/real/lerobot/reward_dataset/raw/`

筛选/拼接中间产物统一放在:

`data/real/lerobot/reward_dataset/intermediate/`

最终可训练版本统一放在:

`data/real/lerobot/reward_dataset/clean/`

独立 holdout 测试集统一放在:

`data/real/lerobot/reward_dataset/holdout/clean/`

holdout manifest:

`configs/reward/reward_holdout_v0.json`

当前 holdout:

| label | 条数 | 路径 |
| --- | ---: | --- |
| success | 3 | `data/real/lerobot/reward_dataset/holdout/clean/success_3ep_20260521` |
| fail_miss_target | 2 | `data/real/lerobot/reward_dataset/holdout/clean/fail_miss_target_2ep_20260521` |
| fail_wrong_object | 2 | `data/real/lerobot/reward_dataset/holdout/clean/fail_wrong_object_2ep_20260521` |
| fail_knock_push | 2 | `data/real/lerobot/reward_dataset/holdout/clean/fail_knock_push_2ep_20260522` |
| fail_grasp_drop | 2 | `data/real/lerobot/reward_dataset/holdout/clean/fail_grasp_drop_2ep_20260521` |

## 当前模型决策

### 已放弃: 单帧/单窗口作为最终 reward 判定

第一版 `scripts/train/train_reward_classifier.py` 使用 episode 末尾附近的固定窗口做
binary classifier。该方法在训练集内部验证能很快到 100%,但 holdout 上暴露出明显问题:

| 实验 | 训练数据 | holdout 结论 |
| --- | --- | --- |
| `red_cube_v0_success25` | success 25 + failure 40, episode 末尾窗口 | failure 8/8 正确,success 0/3 正确 |
| `red_cube_v0_success25_offset45` | success 25 + failure 40,末尾前约 1.5s 窗口 | failure 8/8 正确,success 1/3 正确 |
| `red_cube_v0_success25_offset60` | success 25 + failure 40,末尾前约 2.0s 窗口 | failure 8/8 正确,success 1/3 正确 |
| `red_cube_v0_success25_offset100` | success 25 + failure 40,末尾前约 3.3s 窗口 | success 3/3 正确,但 failure 仅 3/8 正确 |
| `red_cube_v0_seq16_span120` | success 25 + failure 40,后段约 4s 有序序列 | failure 8/8 正确,success 0/3 正确 |
| `red_cube_v0_seq32_full` | success 25 + failure 40,整条 episode 均匀 32 帧 | failure 8/8 正确,success 0/3 正确 |

原因: 成功 episode 的最终几帧里,红块可能被夹爪、机械臂或视角遮挡,单帧不一定能直接看清
"红块是否仍被稳定抓住"。这不是单纯多采几条数据能完全解决的问题。

同时不采用 "整段视频只要某一帧像成功就判成功" 的 `max(prob)` 逻辑,因为
`fail_grasp_drop` 会在中途出现夹住红块的帧,但最终任务仍然失败。

offset 扫描结论:

- `offset60`: failure 概率整体低,四类 failure 全对;但 success 概率为
  `[0.101, 0.645, 0.135]`,仅 1/3 过 0.5。说明结束前约 2s 仍容易因遮挡或目标不可见漏掉成功。
- `offset100`: success 概率为 `[0.728, 0.967, 0.567]`,3/3 正确;但
  `fail_grasp_drop` 概率 `[0.684, 0.809]`、`fail_knock_push` 概率
  `[0.854, 0.910]`,全部被误判为成功。说明更早的单窗口只学到"中途像接触/夹到",
  不能判断最终是否稳定成功。

因此单窗口不能作为最终 reward:窗口太靠后会漏 success,窗口太靠前会把 drop/push 当 success。

### 当前采用: 持续时间序列 success classifier

当前 reward classifier v0 改为 ordered sequence 方案。后段窗口实验仍然漏掉 holdout success,
因此优先使用 `--sequence-scope full` 从整条 episode 均匀抽帧,让模型看到接近、夹取、
带起和结束状态的完整过程。

注意: `red_cube_v0_seq32_full` 首次 holdout 结果仍为 failure 8/8、success 0/3。
这说明仅把抽帧范围扩展到整条 episode 还不够;当前训练/holdout success 分布仍不一致,
或当前视觉+state 模型容量/标签定义不足以稳定识别 holdout success。该结果先记录为负结果,
暂不继续修改模型。

- 脚本: `scripts/train/train_reward_sequence_classifier.py`
- 评估: `scripts/eval/eval_reward_sequence_classifier.py`
- 输入: 整条 episode 的有序帧序列,默认建议 32 帧。
- 模型: 每帧 image+state 编码后进入 GRU,输出 episode-level success/failure。
- 标签语义: `success=1` 表示后段持续稳定完成任务,不是"曾经夹到过"。
- 对 RL 的影响: 如果在线使用该 reward,需要维护最近 2-4s observation history buffer;它更适合作为 episode/outcome reward 或低频 delayed reward,不应直接当作无历史的瞬时 dense reward。

推荐训练命令:

```bash
python scripts/train/train_reward_sequence_classifier.py \
  --manifest configs/reward/reward_dataset_v0.json \
  --output-dir outputs/reward_sequence_classifier/red_cube_v0_seq32_full \
  --epochs 25 \
  --batch-size 4 \
  --sequence-length 32 \
  --sequence-scope full \
  --device cuda
```

推荐 holdout 评估命令:

```bash
python scripts/eval/eval_reward_sequence_classifier.py \
  --checkpoint outputs/reward_sequence_classifier/red_cube_v0_seq32_full/checkpoint_best.pt \
  --manifest configs/reward/reward_holdout_v0.json \
  --output-dir outputs/reward_sequence_classifier/red_cube_v0_seq32_full_holdout_eval \
  --batch-size 4 \
  --device cuda
```

## 失败类别定义

### fail_miss_target

夹爪靠近红块但没有夹到。红块最终基本不移动,或只有极轻微擦碰。

当前已完成: 10 条。

### fail_wrong_object

任务仍然是抓红块,但实际去抓蓝块或绿块。红/蓝/绿都必须在画面中。

当前已完成: 10 条。

### fail_knock_push

碰到红块,但没有成功夹起,而是把红块推走、撞偏、撞倒。

当前已完成: 10 条。

### fail_grasp_drop

夹住或半夹住红块,但抬起/移动过程中掉落,最终没有稳定离桌。

当前已完成: 10 条。

## 采集要求

- 每条从相同 ready pose 开始。
- 红块位置要在当前成功 demo 覆盖范围内。
- 每 3-5 条换一次红块位置。
- 蓝/绿 distractor 尽量都在场,避免 classifier 学到“有没有 distractor”这种捷径。
- 不要把难到超出机械臂可达范围的位置当 failure。
- 失败原因应由动作导致,不是由目标不可见、光照极差、遮挡严重导致。
- 对 success: 重点是后段持续稳定抓住/带起红块;最终单帧不强制要求无遮挡,但整段后段必须能提供足够时序证据。

## 状态级 reward classifier 采集

后续 reward classifier 改为 HIL-SERL 风格的 frame/state-level 标注:

```text
reward.success = 1  当前帧已经处于稳定任务成功状态
reward.success = 0  当前帧不是稳定成功状态
```

注意这不是 episode-level label。成功 episode 的接近、碰到、未抓稳、刚夹住但还没带起等阶段
仍然是 `0`。只有红块已经被稳定抓住/带起/保持时才标 `1`。

项目脚本:

```bash
python scripts/deploy/record_reward_classifier_states.py \
  --dataset-name reward_red_state_labels_v0_001 \
  --target-color red \
  --episode-time-s 60 \
  --max-episodes 8
```

默认设备:

```text
leader:        /dev/xlerobot_left_leader
follower:      /dev/xlerobot_right_follower
head camera:   /dev/xlerobot_head_camera
camera profile: 1280x720@30
control fps:   15
```

脚本会打开实时相机预览窗口。录制时按键:

| 按键 | 作用 |
| --- | --- |
| `p` | 切换 positive 标注 ON/OFF。ON 时写入 `reward.success=1`。 |
| `0` | 立即回到 negative,写入 `reward.success=0`。 |
| `SPACE` | 结束当前 episode。 |
| `q` / `ESC` | 退出当前录制流程。 |

采集目标:

| 类型 | 目标数量 | 要求 |
| --- | ---: | --- |
| positive state | 300-600 帧 | 红块已经被稳定抓住/带起/保持时打开 `p`。 |
| negative / hard negative state | 1000-2000 帧 | 默认就是 negative;重点覆盖容易误判成成功的状态。 |

建议采集内容:

- 正样本: 抓起红块后保持 2-4 秒,期间 `p` 打开;每条可以多次开关。
- 负样本: 接近红块、碰到但未夹住、半夹住、推走、掉落、抓错蓝/绿、夹爪闭合但没有红块。
- 每 2-3 个 episode 换一次红块位置,但不要放到明显不可达区域。
- 预览窗口里如果红块/夹爪关系完全看不清,这段不要标 positive;宁可少标,不要给脏正样本。
- 结束一条 episode 后脚本会询问是否保存。明显误标或撞到桌子的 episode 直接丢弃。

输出:

```text
data/real/lerobot/<dataset_name>/      LeRobot 数据,含 reward.success 字段
data/reward/<dataset_name>/            采集元数据和 session 摘要
```

## 推荐采集命令

每一类单独采一个 LeRobot dataset。采集完后先 sanity check 和抽帧检查,再合并成 classifier 数据集。

补 `miss_target`:

```bash
python scripts/deploy/record_bc_continuous.py \
  --dataset-name reward_red_fail_miss_target_patch_20260521 \
  --target-color red \
  --episode-time-s 20 \
  --max-episodes 4
```

采 `wrong_object`:

```bash
python scripts/deploy/record_bc_continuous.py \
  --dataset-name reward_red_fail_wrong_object_20260521 \
  --target-color red \
  --episode-time-s 20 \
  --max-episodes 10
```

采 `knock_push`:

```bash
python scripts/deploy/record_bc_continuous.py \
  --dataset-name reward_red_fail_knock_push_20260521 \
  --target-color red \
  --episode-time-s 20 \
  --max-episodes 10
```

采 `grasp_drop`:

```bash
python scripts/deploy/record_bc_continuous.py \
  --dataset-name reward_red_fail_grasp_drop_20260521 \
  --target-color red \
  --episode-time-s 20 \
  --max-episodes 10
```

## 检查命令

```bash
python scripts/sanity/check_lerobot_dataset.py \
  --dataset-root data/real/lerobot/<dataset_name> \
  --expect-fps 30 \
  --expect-width 1280 \
  --expect-height 720
```

如果某类中混入了别的失败类型,先不要删除原始数据。用
`scripts/data_collection/merge_lerobot_datasets.py --include-episodes ...` 生成 clean 版本。

检查训练集 manifest:

```bash
python scripts/sanity/check_reward_dataset.py \
  --manifest configs/reward/reward_dataset_v0.json
```

检查 holdout manifest:

```bash
python scripts/sanity/check_reward_dataset.py \
  --manifest configs/reward/reward_holdout_v0.json
```
