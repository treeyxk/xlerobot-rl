"""Sanity test for M1.2 Grounding Pipeline.

Validates:
1. HSV detector finds red cube in head_camera RGB
2. SAM2 refines the mask
3. 3D position matches GT cube.pose.p within tolerance

Tolerance for sim (perfect depth + perfect camera):
- expected error < 2 cm
- if error > 5 cm, something is fundamentally wrong (frame transform bug?)
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import gymnasium as gym
import cv2

import mani_skill.envs  # noqa: F401
import xlerobot_rl.sim.envs  # noqa: F401

from xlerobot_rl.perception import GroundingPipeline, GroundedObject


def main():
    print("=" * 60)
    print("M1.2 Grounding Sanity Test")
    print("=" * 60)
    
    # Create env
    print("\n[1/5] Creating StaticArmGrasp-v0 env...")
    env = gym.make(
        "StaticArmGrasp-v0",
        num_envs=1,
        obs_mode="rgbd",
        sim_backend="gpu",
    ).unwrapped
    
    obs, _ = env.reset(seed=0)
    print(f"      ✓ Env reset OK")
    
    # GT cube position
    gt_cube_pos = env.cube.pose.p[0].cpu().numpy()
    print(f"      GT cube position (world): {gt_cube_pos.round(3).tolist()}")
    
    # Extract head_camera data
    print("\n[2/5] Extracting head_camera RGB-D + intrinsics + extrinsics...")
    
    rgb_tensor = obs["sensor_data"]["head_camera"]["rgb"][0]      # (H, W, 3) uint8
    depth_tensor = obs["sensor_data"]["head_camera"]["depth"][0]  # (H, W, 1) int16 mm
    
    rgb = rgb_tensor.cpu().numpy()
    depth_mm = depth_tensor.cpu().numpy().squeeze()   # (H, W) int16
    depth_meters = depth_mm.astype(np.float32) / 1000.0
    
    print(f"      RGB: shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"      Depth: shape={depth_meters.shape}, dtype={depth_meters.dtype}")
    print(f"      Depth range: [{depth_meters[depth_meters>0].min():.3f}, "
          f"{depth_meters.max():.3f}] meters")
    
    # Camera intrinsics + extrinsics from sensor_param
    sensor_param = obs["sensor_param"]["head_camera"]
    K = sensor_param["intrinsic_cv"][0].cpu().numpy()    # (3, 3)
    
    # Extrinsic: ManiSkill provides 'extrinsic_cv' which is T_camera_world (cv convention)
    # We want T_world_camera (cam → world). Take inverse.
    extrinsic_cv = sensor_param["extrinsic_cv"][0].cpu().numpy()   # (3, 4) or (4, 4)?
    if extrinsic_cv.shape == (3, 4):
        T_cam_world = np.eye(4, dtype=np.float32)
        T_cam_world[:3, :] = extrinsic_cv
    else:
        T_cam_world = extrinsic_cv
    
    # Invert: T_world_cam = T_cam_world^-1
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]
    T_world_cam = np.eye(4, dtype=np.float32)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = -R.T @ t
    
    print(f"      K (intrinsic):\n{K.round(1)}")
    print(f"      T_world_camera:\n{T_world_cam.round(3)}")
    
    # Save RGB for debug
    os.makedirs("data/debug", exist_ok=True)
    cv2.imwrite("data/debug/grounding_input_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"      ✓ Saved RGB to data/debug/grounding_input_rgb.png")
    
    # Build grounding pipeline
    print("\n[3/5] Building GroundingPipeline (HSV + SAM2)...")
    pipeline = GroundingPipeline(use_sam2=True)
    print(f"      ✓ Pipeline ready")
    
    # Detect
    print("\n[4/5] Running detection...")
    detection = pipeline.detect_red_cube(
        rgb=rgb,
        depth_meters=depth_meters,
        K=K,
        T_world_camera=T_world_cam,
    )
    
    if detection is None:
        print("      ✗ No detection! Check:")
        print("        - Is cube visible in RGB?")
        print("        - Is cube color in HSV red range?")
        print("        - Saved RGB at data/debug/grounding_input_rgb.png")
        sys.exit(1)
    
    print(f"      ✓ Detection: {detection}")
    
    # Save mask viz
    mask_vis = np.zeros_like(rgb)
    mask_vis[detection.mask] = [255, 0, 0]
    blended = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)
    
    # Draw bbox + centroid
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.circle(blended, (cx, cy), 5, (255, 255, 0), -1)
    
    cv2.imwrite("data/debug/grounding_result.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"      ✓ Saved viz to data/debug/grounding_result.png")
    
    # Validate vs GT
    print("\n[5/5] Validating vs GT...")
    detected_pos = detection.pos_world
    error_3d = np.linalg.norm(detected_pos - gt_cube_pos)
    error_xy = np.linalg.norm(detected_pos[:2] - gt_cube_pos[:2])
    error_z = abs(detected_pos[2] - gt_cube_pos[2])
    
    print(f"      Detected (world): {detected_pos.round(3).tolist()}")
    print(f"      GT       (world): {gt_cube_pos.round(3).tolist()}")
    print(f"      Error 3D:  {error_3d * 100:.1f} cm")
    print(f"      Error XY:  {error_xy * 100:.1f} cm")
    print(f"      Error Z:   {error_z * 100:.1f} cm")
    
    print("\n" + "=" * 60)
    if error_3d < 0.02:
        print("✓ EXCELLENT (error < 2cm)")
    elif error_3d < 0.05:
        print("✓ OK (error < 5cm, acceptable for sim)")
    else:
        print(f"✗ FAIL (error {error_3d*100:.1f}cm > 5cm, debug needed)")
        print("  Possible issues:")
        print("  - T_world_camera invert wrong (check sensor_param.extrinsic_cv format)")
        print("  - Depth unit wrong (int16 mm vs float meters?)")
        print("  - K matrix wrong format")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
