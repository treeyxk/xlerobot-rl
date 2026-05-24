"""Record frame-level reward-classifier labels with live camera preview.

This is an in-process leader-follower recorder for HIL-SERL-style reward data.
Frames are negative by default. During recording, press ``p`` to toggle positive
labeling for the current success-state window.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep


REPO_ROOT = Path(__file__).resolve().parents[2]
BC_RECORDER_PATH = REPO_ROOT / "scripts" / "deploy" / "record_bc_continuous.py"
spec = importlib.util.spec_from_file_location("xlerobot_record_bc_continuous", BC_RECORDER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"failed to load {BC_RECORDER_PATH}")
bc_recorder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bc_recorder
spec.loader.exec_module(bc_recorder)


DEFAULT_LEADER_PORT = bc_recorder.DEFAULT_LEADER_PORT
DEFAULT_FOLLOWER_PORT = bc_recorder.DEFAULT_FOLLOWER_PORT
DEFAULT_CAMERA_INDEX = bc_recorder.DEFAULT_CAMERA_INDEX
DEFAULT_TASK = "Reward classifier state labeling for red cube grasp"
LABEL_KEY = "reward.success"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def auto_session_name() -> str:
    return f"reward_red_state_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def git_commit() -> str:
    return bc_recorder.git_commit()


def create_reward_dataset(args: argparse.Namespace, follower) -> LeRobotDataset:
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=follower.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=follower.observation_features),
            use_videos=True,
        ),
        {
            LABEL_KEY: {
                "dtype": "float32",
                "shape": (1,),
                "names": ["success"],
            }
        },
    )
    return LeRobotDataset.create(
        repo_id=f"local/{args.dataset_name}",
        fps=args.fps,
        root=args.raw_dataset_root,
        robot_type=follower.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        batch_encoding_size=1,
        vcodec=args.vcodec,
    )


def write_session_info(args: argparse.Namespace) -> None:
    args.reward_dataset_root.mkdir(parents=True, exist_ok=True)
    info = {
        "schema_version": "reward_state_labels_v0",
        "source": "real_lerobot_inprocess",
        "dataset_name": args.dataset_name,
        "raw_lerobot_root": str(args.raw_dataset_root),
        "target_color": args.target_color,
        "instruction": args.instruction,
        "label_key": LABEL_KEY,
        "label_semantics": {
            "0": "not currently in a stable task-success state",
            "1": "red cube is currently stably grasped/lifted/held as task success",
        },
        "controls": {
            "ready_setup": "move left leader, right follower tracks, SPACE captures ready pose",
            "record_start": "SPACE starts an episode",
            "positive_toggle": "p toggles positive labeling on/off during recording",
            "negative": "0 forces negative labeling",
            "end_episode": "SPACE ends the current episode",
            "quit": "q quits before/after episodes; ESC aborts current recording loop",
        },
        "leader_port": args.leader_port,
        "follower_port": args.follower_port,
        "camera": {
            "type": "opencv",
            "index_or_path": str(args.camera_index),
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
        },
        "control_fps": args.fps,
        "created_at": now_utc(),
        "git_commit": git_commit(),
        "notes": (
            "Frame-level reward classifier data. Negative by default; operator toggles "
            "positive only for stable current success states."
        ),
    }
    (args.reward_dataset_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )


def safe_disconnect(name: str, device) -> None:
    try:
        if getattr(device, "is_connected", False):
            device.disconnect()
    except Exception as exc:
        print(f"WARNING: failed to disconnect {name}: {exc}")


def image_to_rgb(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def draw_preview(
    rgb: np.ndarray,
    *,
    positive: bool,
    frames: int,
    positives: int,
    elapsed_s: float,
    dataset_name: str,
    display_width: int,
) -> np.ndarray:
    if display_width > 0 and rgb.shape[1] != display_width:
        scale = display_width / rgb.shape[1]
        rgb = cv2.resize(rgb, (display_width, max(1, int(rgb.shape[0] * scale))))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = (0, 220, 0) if positive else (0, 0, 220)
    label = "POSITIVE success=1" if positive else "NEGATIVE success=0"
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 88), (0, 0, 0), -1)
    cv2.putText(bgr, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        bgr,
        f"frames={frames} pos={positives} t={elapsed_s:5.1f}s",
        (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        "p: toggle positive   0: negative   SPACE: end episode   q/ESC: quit",
        (14, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        dataset_name,
        (14, bgr.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr


def setup_terminal_cbreak():
    if not sys.stdin.isatty():
        return None, None
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old_settings


def restore_terminal(fd, old_settings) -> None:
    if fd is not None and old_settings is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def record_one_episode(args: argparse.Namespace, leader, follower, dataset: LeRobotDataset) -> dict[str, int | bool]:
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    print("\nRecording reward labels.")
    print("  p toggles positive label ON/OFF.")
    print("  0 forces negative label.")
    print("  SPACE ends this episode.")
    print("  q or ESC aborts this episode loop.")
    fd, old_settings = setup_terminal_cbreak()
    window_name = "reward classifier state labeling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    positive = False
    frames = 0
    positives = 0
    aborted = False
    period = 1.0 / args.fps
    start_t = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()
            obs = bc_recorder.get_follower_observation(args, follower)
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            leader_action = bc_recorder.get_leader_action(args, leader)
            action_values = teleop_action_processor((leader_action, obs))
            robot_action = robot_action_processor((action_values, obs))
            bc_recorder.send_follower_action(args, follower, robot_action)

            label_value = 1.0 if positive else 0.0
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            dataset.add_frame(
                {
                    **observation_frame,
                    **action_frame,
                    LABEL_KEY: np.array([label_value], dtype=np.float32),
                    "task": args.instruction,
                }
            )
            frames += 1
            positives += int(positive)

            rgb = image_to_rgb(observation_frame["observation.images.front"])
            preview = draw_preview(
                rgb,
                positive=positive,
                frames=frames,
                positives=positives,
                elapsed_s=time.perf_counter() - start_t,
                dataset_name=args.dataset_name,
                display_width=args.display_width,
            )
            cv2.imshow(window_name, preview)

            key = bc_recorder.poll_key(0.0)
            cv_key = cv2.waitKey(1) & 0xFF
            if key == "p" or cv_key == ord("p"):
                positive = not positive
                print(f"Label -> {'POSITIVE' if positive else 'NEGATIVE'}")
            elif key == "0" or cv_key == ord("0"):
                positive = False
                print("Label -> NEGATIVE")
            elif key == " " or cv_key == ord(" "):
                print("space")
                break
            elif key == "q" or cv_key in (ord("q"), 27):
                aborted = True
                print("Abort requested.")
                break
            if time.perf_counter() - start_t >= args.episode_time_s:
                print("Episode time limit reached.")
                break
            precise_sleep(max(period - (time.perf_counter() - loop_start), 0.0))
    finally:
        restore_terminal(fd, old_settings)
        cv2.destroyWindow(window_name)
    return {"frames": frames, "positives": positives, "aborted": aborted}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT)
    parser.add_argument("--leader-id", default="left_leader")
    parser.add_argument("--follower-id", default="right_follower")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--raw-dataset-root", type=Path, default=None)
    parser.add_argument("--reward-dataset-root", type=Path, default=None)
    parser.add_argument("--target-color", choices=["red", "blue", "green"], default="red")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-relative-target", type=float, default=15.0)
    parser.add_argument("--camera-index", default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--return-step-deg", type=float, default=3.0)
    parser.add_argument("--return-fps", type=float, default=15.0)
    parser.add_argument("--ready-lock-s", type=float, default=0.5)
    parser.add_argument("--torque-num-retry", type=int, default=5)
    parser.add_argument("--torque-attempts", type=int, default=3)
    parser.add_argument("--comm-attempts", type=int, default=8)
    parser.add_argument("--comm-retry-sleep-s", type=float, default=0.08)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_name is None:
        args.dataset_name = auto_session_name()
    if args.raw_dataset_root is None:
        args.raw_dataset_root = Path("data/real/lerobot") / args.dataset_name
    if args.reward_dataset_root is None:
        args.reward_dataset_root = Path("data/reward") / args.dataset_name
    if args.raw_dataset_root.exists():
        print(f"ERROR: dataset root already exists: {args.raw_dataset_root}")
        return 1

    leader, follower = bc_recorder.make_devices(args)
    dataset = None
    kept = 0
    discarded = 0
    total_frames = 0
    total_positives = 0
    try:
        print("Connecting leader/follower/camera...")
        leader.connect()
        follower.connect()
        dataset = create_reward_dataset(args, follower)
        write_session_info(args)
        ready_pose = bc_recorder.run_ready_setup(args, leader, follower)

        while args.max_episodes is None or kept < args.max_episodes:
            bc_recorder.return_and_lock_ready(args, leader, follower, ready_pose)
            key = bc_recorder.wait_key("\nReady. Press SPACE to start reward episode, or q to quit: ", {" ", "q"})
            if key == "q":
                break
            while not bc_recorder.set_leader_torque(
                leader,
                False,
                num_retry=args.torque_num_retry,
                attempts=args.torque_attempts,
            ):
                if not bc_recorder.retry_or_quit("Failed to release leader torque. Check USB/power if needed."):
                    raise KeyboardInterrupt("leader torque release cancelled")
            result = record_one_episode(args, leader, follower, dataset)
            choice = bc_recorder.wait_key(
                f"Episode captured ({result['frames']} frames, {result['positives']} positive). Save? [y/n/q]: ",
                {"y", "n", "q"},
            )
            if choice == "y":
                dataset.save_episode()
                kept += 1
                total_frames += int(result["frames"])
                total_positives += int(result["positives"])
                print(f"Saved episode {kept - 1}.")
            else:
                dataset._wait_image_writer()
                dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)
                discarded += 1
                print("Discarded episode.")
                if choice == "q":
                    break
            summary = {
                "dataset_name": args.dataset_name,
                "updated_at": now_utc(),
                "kept_episodes": kept,
                "discarded_episodes": discarded,
                "total_frames": total_frames,
                "total_positive_frames": total_positives,
                "ready_pose": ready_pose,
            }
            (args.reward_dataset_root / "reward_label_session.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
            if result["aborted"]:
                break
    finally:
        if dataset is not None:
            dataset.finalize()
        safe_disconnect("follower", follower)
        safe_disconnect("leader", leader)
        cv2.destroyAllWindows()

    print(f"\nDone. Saved episodes: {kept}, discarded: {discarded}")
    print(f"Total saved frames: {total_frames}, positive frames: {total_positives}")
    print(f"Dataset: {args.raw_dataset_root}")
    print(f"Metadata: {args.reward_dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
