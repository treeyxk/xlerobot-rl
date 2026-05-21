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
| success | 10 | 10 | 0 |
| fail_miss_target | 10 | 8 | 2 |
| fail_wrong_object | 10 | 0 | 10 |
| fail_knock_push | 10 | 0 | 10 |
| fail_grasp_drop | 10 | 0 | 10 |

`success` 暂时使用独立测试集:

`data/real/lerobot/m4_target_grasp_v0_bc_red_test_10ep_merged_20260521`

已有干净 miss 数据:

`data/real/lerobot/reward_red_fail_miss_target_clean_8ep_20260521`

## 失败类别定义

### fail_miss_target

夹爪靠近红块但没有夹到。红块最终基本不移动,或只有极轻微擦碰。

补采目标: 2-4 条。后续合并时保留最干净的 2 条即可。

### fail_wrong_object

任务仍然是抓红块,但实际去抓蓝块或绿块。红/蓝/绿都必须在画面中。

目标: 10 条。

### fail_knock_push

碰到红块,但没有成功夹起,而是把红块推走、撞偏、撞倒。

目标: 10 条。

### fail_grasp_drop

夹住或半夹住红块,但抬起/移动过程中掉落,最终没有稳定离桌。

目标: 10 条。

## 采集要求

- 每条从相同 ready pose 开始。
- 红块位置要在当前成功 demo 覆盖范围内。
- 每 3-5 条换一次红块位置。
- 蓝/绿 distractor 尽量都在场,避免 classifier 学到“有没有 distractor”这种捷径。
- 不要把难到超出机械臂可达范围的位置当 failure。
- 失败原因应由动作导致,不是由目标不可见、光照极差、遮挡严重导致。

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
