#!/usr/bin/env python3
"""
Convert physics-rollout HDF5 files to the task-space format expected by dataset.py.

Input schema (per clip):
  body_pos:     (T, 24, 3)  world-frame joint positions from Isaac Gym
  body_rot:     (T, 24, 4)  world-frame quaternions, xyzw convention
  body_ang_vel: (T, 24, 3)  not used

Output schema (per clip):
  joint_positions: (T, 24, 3)   same as body_pos
  task_space:      (T,  9)      [vx, vy, pelvis_z, l_hand_xyz, r_hand_xyz]
  pose_aa_root:    (T,  3)      pelvis axis-angle (for motion_frame='start')

Usage:
    python scripts/convert_physics_rollout_to_taskspace.py \\
        final_h5_files/physics_rollout/accad_straight_walk_30fps_physics_rollout.h5 \\
        final_h5_files/physics_rollout/accad_straight_walk_physics_rollout_taskspace.h5
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

L_HAND_IDX = 18
R_HAND_IDX = 23
PELVIS_IDX = 0


def extract_task_space(body_pos: np.ndarray, fps: float) -> np.ndarray:
    """Returns (T, 9): [vx, vy, pelvis_z, l_hand_xyz, r_hand_xyz]."""
    T = body_pos.shape[0]
    dt = 1.0 / fps
    pelvis = body_pos[:, PELVIS_IDX, :]

    vel_xy = np.empty((T, 2), dtype=np.float32)
    vel_xy[1:-1] = (pelvis[2:, :2] - pelvis[:-2, :2]) / (2.0 * dt)
    vel_xy[0]    = (pelvis[1, :2]  - pelvis[0, :2])   / dt
    vel_xy[-1]   = (pelvis[-1, :2] - pelvis[-2, :2])  / dt

    pelvis_z  = pelvis[:, 2:3].astype(np.float32)
    left_eef  = body_pos[:, L_HAND_IDX, :].astype(np.float32)
    right_eef = body_pos[:, R_HAND_IDX, :].astype(np.float32)

    return np.concatenate([vel_xy, pelvis_z, left_eef, right_eef], axis=1)


def body_rot_to_pose_aa_root(body_rot: np.ndarray) -> np.ndarray:
    """Convert pelvis global quaternion (xyzw) to axis-angle. Returns (T, 3)."""
    pelvis_quat = body_rot[:, PELVIS_IDX, :].astype(np.float64)  # (T, 4) xyzw
    return Rotation.from_quat(pelvis_quat).as_rotvec().astype(np.float32)


def convert(in_path: Path, out_path: Path, fps: float) -> None:
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        fout.attrs["source_file"]     = str(in_path)
        fout.attrs["converted_from"]  = "physics_rollout"
        fout.attrs["fps"]             = fps
        fout.attrs["joint_names"]     = [
            "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
            "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
            "Torso", "Spine", "Chest", "Neck", "Head",
            "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
            "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
        ]
        fout.attrs["task_space_cols"] = [
            "root_vel_x", "root_vel_y", "pelvis_height",
            "left_hand_x", "left_hand_y", "left_hand_z",
            "right_hand_x", "right_hand_y", "right_hand_z",
        ]

        clips_grp = fout.create_group("clips")
        clip_names = list(fin["clips"].keys())
        print(f"  {len(clip_names)} clips")

        for clip_name in clip_names:
            grp = fin[f"clips/{clip_name}"]
            body_pos = grp["body_pos"][:].astype(np.float32)  # (T, 24, 3)
            body_rot = grp["body_rot"][:]                      # (T, 24, 4)
            T = body_pos.shape[0]

            task_space   = extract_task_space(body_pos, fps)
            pose_aa_root = body_rot_to_pose_aa_root(body_rot)
            pelvis = body_pos[:, PELVIS_IDX, :]

            vel_mag = np.linalg.norm(task_space[:, :2], axis=1)
            print(
                f"    {clip_name}: T={T} "
                f"pelvis_z=[{pelvis[:,2].min():.3f},{pelvis[:,2].max():.3f}] "
                f"speed=[{vel_mag.min():.3f},{vel_mag.max():.3f}]"
            )

            cg = clips_grp.create_group(clip_name)
            cg.attrs["clip_name"] = clip_name
            cg.attrs["fps"]       = fps
            cg.attrs["n_frames"]  = T
            cg.create_dataset("joint_positions", data=body_pos,
                              dtype="f4", compression="gzip", compression_opts=4)
            cg.create_dataset("task_space", data=task_space,
                              dtype="f4", compression="gzip", compression_opts=4)
            cg.create_dataset("pose_aa_root", data=pose_aa_root,
                              dtype="f4", compression="gzip", compression_opts=4)

    print(f"  → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert physics rollout HDF5 to taskspace format")
    p.add_argument("input",  help="Input physics rollout .h5 file")
    p.add_argument("output", help="Output taskspace .h5 file")
    p.add_argument("--fps",  type=float, default=30.0, help="Simulation fps (default: 30)")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        sys.exit(f"ERROR: input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting {in_path.name} ...")
    convert(in_path, out_path, args.fps)


if __name__ == "__main__":
    main()
