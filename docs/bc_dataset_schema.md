# BC Dataset Schema

**Status**: Draft v0

This document defines the first behavior cloning dataset contract for M4
target-conditioned manipulation. It is intentionally narrow: right-arm, tabletop,
single-skill (`top_grasp`) demos for red/blue/green cube picking.

The goal is to make sim demos, LeRobot leader-follower demos, and future dataset
converters produce the same training-facing fields.

## Scope

In scope:

- Right follower arm only.
- Static or near-static mobile base.
- One target object and zero or more distractors.
- `top_grasp` skill only.
- Joint-space action labels for v0.

Out of scope for v0:

- Multi-skill labels.
- Navigation actions.
- End-to-end M2/M3/M5 trajectories.
- Open-vocabulary language beyond the stored instruction text.
- Learned reward classifier labels beyond simple success/failure metadata.

## Storage Layout

Raw LeRobot recordings stay in their native layout:

```text
data/real/lerobot/<dataset_name>/
```

Training-facing converted datasets should expose this schema. HDF5 is the v0
reference format because it is simple to inspect and independent of LeRobot
internals:

```text
data/bc/m4_target_grasp_v0/
  dataset_info.yaml
  episode_000000.h5
  episode_000001.h5
  ...
```

Sim-generated BC or oracle data should use the same schema:

```text
data/bc/sim_m4_target_grasp_v0/
  dataset_info.yaml
  episode_000000.h5
```

## Dataset-Level Metadata

`dataset_info.yaml`:

```yaml
schema_version: "bc_m4_v0"
source: "real_lerobot"  # "real_lerobot" | "sim_oracle" | "sim_replay"
env_id: "TargetConditionedArmGrasp-v0"
robot: "XLeRobot 0.4.0"
arm: "right"
intended_control_hz: 15
action_space: "right_arm_joint_delta_6d"
joint_order:
  - shoulder_pan
  - shoulder_lift
  - elbow_flex
  - wrist_flex
  - wrist_roll
gripper_convention: "0=closed, 1=open"
skill_id_map:
  top_grasp: 0
color_id_map:
  red: 0
  blue: 1
  green: 2
created_at: "YYYY-MM-DDTHH:MM:SSZ"
git_commit: "<commit>"
notes: ""
```

Real smoke/BC recordings may also include a pre-record target snapshot. This is
dataset-level metadata, not a replacement for per-step grounding, but it lets
the first real demos carry a target-conditioned pose immediately:

```yaml
target_grounding:
  enabled: true
  detector: "hsv_rgbd_red_cube_v0"
  intrinsics: "configs/calibration/head_camera_intrinsics_1280x720.yaml"
  extrinsics: "configs/calibration/head_camera_extrinsics.yaml"
  cube_size_m: 0.03
  min_area_px: 300
target_snapshot:
  status: "captured"  # "captured" | "failed" | "not_captured"
  source: "real_rgbd_hsv"
  target_pos_base_initial_m: [-0.381470, 0.001665, 0.772978]
  target_visible_initial: true
  debug_image: "data/bc/<dataset_name>/target_snapshots/pre_record_target_debug.png"
  result_json: "data/bc/<dataset_name>/target_snapshots/pre_record_target_result.json"
```

Formal converted BC episodes should still expose `obs/target_pos_base` and
`obs/target_visible` per step. For the first real smoke data, converters may use
`target_snapshot.target_pos_base_initial_m` as a constant episode target when
the object is not moved during the demo.

## Episode Metadata

Each `episode_XXXXXX.h5` must include a `meta` group or equivalent attributes:

| Field | Type | Meaning |
|-------|------|---------|
| `episode_idx` | int | Dataset episode index. |
| `instruction` | str | Original task text, e.g. `"Pick up the red cube with the right arm"`. |
| `target_color_id` | int | `red=0`, `blue=1`, `green=2`. |
| `target_name` | str | e.g. `"red_cube"`. |
| `skill_id` | int | `top_grasp=0`. |
| `sim_seed` | int or null | `env.reset(seed=...)` seed for sim episodes; null for real data. |
| `spawn_xy` | array or null | Actual generated cube XY positions for sim debug/replay; null for real data. |
| `is_success` | bool | Target lifted and held according to task success rule. |
| `wrong_object_failure` | bool | Any distractor was lifted. |
| `n_steps` | int | Number of recorded steps. |
| `actual_control_hz` | float | Measured episode average control frequency. |
| `intended_control_hz` | float | Intended control frequency copied from dataset metadata if unchanged. |
| `source` | str | `"real_lerobot"` / `"sim_oracle"` / `"sim_replay"`. |

`n_steps` must equal `T`, the length of every per-step array in the episode.

## Per-Step Fields

All arrays are time-major. `T` is episode length.

### Required Observation Fields

| HDF5 path | Shape | Dtype | Units / Meaning |
|-----------|-------|-------|-----------------|
| `obs/timestamp_monotonic` | `(T,)` | float64 | Monotonic seconds, e.g. `time.monotonic()` or simulator monotonic time. |
| `obs/wall_time` | `(T,)` | float64 | Optional wall-clock UTC seconds, e.g. `time.time()`, for log correlation. |
| `obs/step_idx` | `(T,)` | int64 | Step index inside episode. |
| `obs/right_arm_qpos` | `(T, 5)` | float32 | Right arm non-gripper joint positions in radians. |
| `obs/right_arm_qvel` | `(T, 5)` | float32 | Right arm non-gripper joint velocities, zeros allowed for v0 if unavailable. |
| `obs/gripper_width` | `(T, 1)` | float32 | Physical gripper opening width in meters when calibrated; NaN allowed if unavailable. |
| `obs/gripper_command` | `(T, 1)` | float32 | Normalized gripper state/command, `[0, 1]`, `0=closed`, `1=open`. |
| `obs/target_pos_base` | `(T, 3)` | float32 | Target position in base/world frame, meters. See source rule below. |
| `obs/target_visible` | `(T, 1)` | bool | Whether current-frame grounding produced a valid target observation. |
| `obs/target_color_id` | `(T, 1)` | int64 | `red=0`, `blue=1`, `green=2`. |
| `obs/skill_id` | `(T, 1)` | int64 | `top_grasp=0`. |
| `obs/prev_action` | `(T, 6)` | float32 | Previous action, zeros at step 0. |

### Optional Observation Fields

| HDF5 path | Shape | Dtype | Meaning |
|-----------|-------|-------|---------|
| `obs/head_rgb` | `(T, H, W, 3)` | uint8 | Head camera RGB. |
| `obs/head_depth` | `(T, H, W)` | float32 | Depth in meters. |
| `obs/head_depth_mm` | `(T, H, W)` | uint16 | Optional raw depth in millimeters for storage/debug. |
| `obs/wrist_rgb_right` | `(T, H, W, 3)` | uint8 | Right wrist camera RGB. |
| `obs/target_mask` | `(T, H, W)` | bool/uint8 | M1 target mask in the image frame used by policy. |
| `obs/distractor_mask` | `(T, H, W)` | bool/uint8 | Union mask for non-target objects. |
| `obs/distractor_pos_base` | `(T, 6)` | float32 | Two distractor positions flattened as `[x1,y1,z1,x2,y2,z2]`. |
| `obs/ee_pose_base` | `(T, 7)` | float32 | EE xyz + quaternion in base frame if FK is available. |

For M4 v0 sim data, the current env already exposes:

```text
target_pos_base
target_color_id
distractor_pos_base
distractor_color_ids
skill_id
tcp_to_target
```

### Target Position Source

`obs/target_pos_base` is a policy input and must follow the same distribution in
sim and real as much as possible. The canonical source is **M1.2 grounding
output**, not privileged sim GT.

| Dataset source | `target_pos_base` rule |
|----------------|------------------------|
| `sim_oracle` | Re-run M1.2 grounding on sim head RGB-D at each recorded step. Sim GT may be stored separately for debugging. |
| `sim_replay` | Same as `sim_oracle`: derive policy input from grounding output. |
| `real_lerobot` | Use M1/VLM/SAM2 grounding output from real RGB-D. |
| smoke-only real data | May leave empty/NaN if the dataset is not used for BC training. |

If privileged sim GT is used temporarily for ablation or debugging, store it as
`debug/target_pos_base_gt` and do not feed it to the BC policy by default.

When current-frame grounding fails, set `obs/target_visible=false`. For M4 v0,
`obs/target_pos_base` may hold the last valid target position or NaNs; the policy
must use `target_visible` to distinguish an observed target from a stale/invalid
position. M1.3 tracker will formalize last-seen behavior later.

### Action Fields

V0 action labels are right-arm joint deltas:

| HDF5 path | Shape | Dtype | Meaning |
|-----------|-------|-------|---------|
| `action/right_arm_delta_qpos` | `(T, 5)` | float32 | Non-gripper joint delta command in radians per step. |
| `action/gripper_command` | `(T, 1)` | float32 | Normalized gripper command, `[0, 1]`, `0=closed`, `1=open`. |
| `action/right_arm_target_qpos` | `(T, 5)` | float32 | Optional absolute non-gripper target qpos if source provides it. |
| `action/is_intervention` | `(T,)` | bool | Human intervention flag; all `True` for teleop demos. |

The canonical joint order is:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
```

Sim mapping:

```text
Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R
```

The gripper is intentionally separated from arm qpos/action. This avoids mixing
radian joint positions with normalized open/close commands and keeps gripper
calibration explicit.

### Label / Info Fields

| HDF5 path | Shape | Dtype | Meaning |
|-----------|-------|-------|---------|
| `label/is_success` | `(T,)` | bool | Per-step task success flag. Episode-level success is `meta/is_success`. |
| `label/target_is_grasped` | `(T,)` | bool | Target grasp flag. |
| `label/target_lift_height` | `(T,)` | float32 | Target lift height in meters. |
| `label/wrong_object_failure` | `(T,)` | bool | Any distractor lifted. |
| `label/failure_reason` | `(T,)` or episode attr | str/int | Optional failure code. |

## Real LeRobot Mapping

Raw `lerobot-record` data can remain in LeRobot's native format. The project
will use one of two paths:

1. A converter writes the reference HDF5 schema above.
2. A LeRobot Dataset API adapter exposes equivalent fields at training time.

The key requirement is not the file format; it is that the training loader sees
the same schema and units.

Minimum mapping for right-arm follower:

| LeRobot source | BC field |
|----------------|----------|
| follower joint observations | `obs/right_arm_qpos` |
| follower joint velocity, if available | `obs/right_arm_qvel` |
| follower action / target | `action/right_arm_delta_qpos` or `action/right_arm_target_qpos` |
| follower gripper state/command | `obs/gripper_command`, `action/gripper_command` |
| camera frame | `obs/head_rgb` or `obs/wrist_rgb_right` |
| task text | episode `instruction` |
| M1 output / manual label | `obs/target_pos_base`, `obs/target_mask`, `target_color_id` |

For the first smoke dataset, it is acceptable to leave `target_mask`,
`distractor_mask`, and `target_pos_base` empty if the purpose is only to verify
recording integrity. Formal BC training data must fill target conditioning fields.

### Smoke Recording Entry Point

Use `scripts/deploy/record_bc_demo.py` as the real LeRobot smoke/BC recording
entry point. By default it is a dry run: it writes
`data/bc/<dataset_name>/dataset_info.yaml` and prints the calibration, teleop,
and record commands.

```bash
python scripts/deploy/record_bc_demo.py \
  --target-color red \
  --dataset-name m4_target_grasp_v0_720p_smoke \
  --num-episodes 1 \
  --episode-time-s 10
```

After calibration and the 10s teleop smoke test pass, add `--run-record`. Before
starting `lerobot-record`, the script prints a checklist and waits for operator
confirmation:

- Press `Space` to start recording.
- Press `q` to cancel before `lerobot-record` starts.

After `Space`, the script captures one RGB-D target snapshot before launching
`lerobot-record`. Use `--require-target-snapshot` to abort recording if this
snapshot fails.

For repeated collection, use `--continuous-record`. It records one LeRobot
dataset per accepted demo:

```bash
python scripts/deploy/record_bc_demo.py \
  --target-color red \
  --episode-time-s 60 \
  --run-record \
  --continuous-record \
  --require-target-snapshot \
  --auto-return-ready
```

Controls:

- `Space`: start one demo.
- `Space`: end the current demo. The wrapper sends LeRobot's Right Arrow finish
  signal; press Right Arrow manually if the synthetic key is blocked.
- `y`: keep the just-recorded demo.
- `n`: discard it.
- `q`: quit.

If `--dataset-name` is omitted, the wrapper generates
`m4_target_grasp_v0_bc_session_<timestamp>`.

With `--auto-return-ready`, the wrapper first starts in-process leader-follower
teleop for ready-pose setup. The operator moves the left leader arm, the right
follower tracks it, and pressing `Space` locks both arms before reading both
ready poses. Before every demo, the wrapper returns both arms to those poses
with low-speed joint interpolation and keeps them locked. Pressing `Space`
releases the serial ports and starts recording.

Each kept trial is stored under names like
`data/real/lerobot/m4_target_grasp_v0_bc_session_001_ep000/`, with matching
`data/bc/..._ep000/dataset_info.yaml`. The session root stores
`continuous_session.json` listing kept/discarded trial dataset names.
- Add `--no-ready-prompt` for non-interactive automation.

The verified real-hardware device names are recorded in
`configs/real/xlerobot_right_arm_720p.yaml`:

```text
leader: /dev/xlerobot_left_leader
follower: /dev/xlerobot_right_follower
camera: /dev/xlerobot_head_camera
camera profile: 1280x720@30 h264
```

Use `lerobot-find-cameras opencv` to list camera candidates. The helper passes
`--dataset.vcodec=h264` by default so the resulting videos are easier to inspect
locally than LeRobot's default `libsvtav1`.

Raw LeRobot data is written to:

```text
data/real/lerobot/<dataset_name>/
```

The project-side metadata template is written to:

```text
data/bc/<dataset_name>/dataset_info.yaml
```

After recording, run the dataset checker to ensure the raw LeRobot dataset was
finalized:

```bash
python scripts/sanity/check_lerobot_dataset.py \
  --dataset-root data/real/lerobot/<dataset_name> \
  --expect-episodes 1 \
  --expect-fps 30 \
  --expect-width 1280 \
  --expect-height 720
```

## Data Quality Gates

Before a dataset is used for BC:

- Every episode has an instruction and target color.
- `meta/n_steps` equals `T`, the length of every per-step array.
- `obs/right_arm_qpos` and action arrays have the same `T`.
- No NaN/Inf in qpos/action/target position fields.
- `obs/target_visible` is present. If `target_visible=false`, target position NaNs are allowed.
- Gripper convention is documented as `0=closed`, `1=open`.
- Video frames, if present, have no large timestamp gaps.
- At least one success/failure label exists per episode.
- Hard negatives are marked with `is_success=false` and a failure reason when available.

## Smoke vs BC Demo

Smoke demo:

- Purpose: verify hardware, recording, cameras, and action timestamps.
- Count: 2-10 short episodes.
- May lack complete target conditioning fields.
- Not used for policy training.

BC demo:

- Purpose: train the initial `π_arm`.
- Count: first target is 20-50 successful demos plus 10-20 hard negatives.
- Must include target-conditioned fields or be convertible to them.
- Must pass the data quality gates above.
