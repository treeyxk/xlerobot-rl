"""Check reward classifier dataset manifest and LeRobot roots."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("configs/reward/reward_dataset_v0.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expect-episodes-per-category", type=int, default=10)
    parser.add_argument("--expect-fps", type=int, default=30)
    parser.add_argument("--expect-width", type=int, default=1280)
    parser.add_argument("--expect-height", type=int, default=720)
    return parser


def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    return json.loads(info_path.read_text())


def main() -> int:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text())
    image_key = manifest.get("image_key", "observation.images.front")

    issues: list[str] = []
    totals = defaultdict(int)
    print(f"Manifest: {args.manifest}")
    print(f"Name: {manifest.get('name')}")
    print(f"Image key: {image_key}\n")

    for entry in manifest["datasets"]:
        category = entry["category"]
        label = int(entry["label"])
        root = Path(entry["root"])
        print(f"[{category}] label={label} root={root}")
        try:
            info = load_info(root)
        except Exception as exc:
            issues.append(f"{category}: failed to load info: {exc}")
            continue

        episodes = int(info.get("total_episodes", -1))
        frames = int(info.get("total_frames", -1))
        fps = int(info.get("fps", -1))
        features = info.get("features", {})
        image_feature = features.get(image_key)
        print(f"  episodes={episodes} frames={frames} fps={fps}")

        expected_episodes = int(entry.get("expected_episodes", args.expect_episodes_per_category))
        if episodes != expected_episodes:
            issues.append(
                f"{category}: expected {expected_episodes} episodes, got {episodes}"
            )
        if fps != args.expect_fps:
            issues.append(f"{category}: expected fps {args.expect_fps}, got {fps}")
        if image_feature is None:
            issues.append(f"{category}: missing image feature {image_key}")
        else:
            shape = image_feature.get("shape")
            print(f"  {image_key}: shape={shape}, dtype={image_feature.get('dtype')}")
            if shape != [args.expect_height, args.expect_width, 3]:
                issues.append(
                    f"{category}: expected image shape "
                    f"[{args.expect_height}, {args.expect_width}, 3], got {shape}"
                )

        for key in ("observation.state", "action"):
            if key not in features:
                issues.append(f"{category}: missing feature {key}")
        totals[label] += episodes

    print("\nEpisode totals by binary label:")
    for label, count in sorted(totals.items()):
        print(f"  label {label}: {count}")

    print("\nChecks:")
    if issues:
        for issue in issues:
            print(f"  WARN: {issue}")
        print("\nResult: WARN")
        return 2
    print("  PASS: no issues found")
    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
