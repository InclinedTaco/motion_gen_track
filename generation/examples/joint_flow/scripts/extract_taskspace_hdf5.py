#!/usr/bin/env python3
"""
Extract task-space quantities from AMASS motion PKL files and save to HDF5.
Uses poselib FK (same as Isaac Gym / PHC training) so positions match the
humanoid exactly — no smplx dependency.

Usage:
    python scripts/extract_taskspace_hdf5.py <input.pkl> <output.h5>
    python scripts/extract_taskspace_hdf5.py <input.pkl> <output.h5> \\
        --humanoid-xml final_humanoid.xml
"""
import argparse
import sys
from pathlib import Path

import sys, os
sys.path.insert(0, os.getcwd())   # makes poselib/ and phc/ importable when run from repo root

import h5py
import joblib
import numpy as np
import torch

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# ---------------------------------------------------------------------------
# Joint layout — poselib / Isaac Gym MJCF DFS order (matches final_humanoid.xml)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    'Pelvis',     #  0
    'L_Hip',      #  1
    'L_Knee',     #  2
    'L_Ankle',    #  3
    'L_Toe',      #  4
    'R_Hip',      #  5
    'R_Knee',     #  6
    'R_Ankle',    #  7
    'R_Toe',      #  8
    'Torso',      #  9
    'Spine',      # 10
    'Chest',      # 11
    'Neck',       # 12
    'Head',       # 13
    'L_Thorax',   # 14
    'L_Shoulder', # 15
    'L_Elbow',    # 16
    'L_Wrist',    # 17
    'L_Hand',     # 18  ← left  EEF
    'R_Thorax',   # 19
    'R_Shoulder', # 20
    'R_Elbow',    # 21
    'R_Wrist',    # 22
    'R_Hand',     # 23  ← right EEF
]
N_JOINTS   = 24
PELVIS_IDX = 0
L_HAND_IDX = 18
R_HAND_IDX = 23

# task_space column layout (T, 9)
# 0:2  root_vel_xy
# 2    pelvis_height
# 3:6  left_eef_xyz
# 6:9  right_eef_xyz


# ---------------------------------------------------------------------------
# Poselib FK
# ---------------------------------------------------------------------------

def compute_joint_positions(clip: dict, sk_tree: SkeletonTree) -> np.ndarray:
    """
    Run poselib FK using pose_quat_global + root_trans_offset.
    Returns float32 (T, 24, 3) in world frame, poselib joint order.
    """
    pqg = clip['pose_quat_global']                      # (T, 24, 4)
    if not isinstance(pqg, torch.Tensor):
        pqg = torch.tensor(pqg, dtype=torch.float32)
    else:
        pqg = pqg.float()

    trans = clip['root_trans_offset']                   # (T, 3)
    if not isinstance(trans, torch.Tensor):
        trans = torch.tensor(trans, dtype=torch.float32)
    else:
        trans = trans.float()

    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, pqg, trans, is_local=False
    )
    return sk_state.global_translation.numpy().astype(np.float32)  # (T, 24, 3)


# ---------------------------------------------------------------------------
# Task-space extraction
# ---------------------------------------------------------------------------

def extract_task_space(joint_positions: np.ndarray, fps: float) -> np.ndarray:
    """
    Returns (T, 9):
        cols 0-1  root_vel_xy    XY velocity of Pelvis (central differences)
        col  2    pelvis_height  Z of Pelvis
        cols 3-5  left_eef_xyz   world pos of L_Hand
        cols 6-8  right_eef_xyz  world pos of R_Hand
    """
    T  = joint_positions.shape[0]
    dt = 1.0 / fps

    root_pos = joint_positions[:, PELVIS_IDX, :]

    root_vel_xy = np.empty((T, 2), dtype=np.float32)
    root_vel_xy[1:-1] = (root_pos[2:, :2] - root_pos[:-2, :2]) / (2.0 * dt)
    root_vel_xy[0]    = (root_pos[1, :2]  - root_pos[0, :2])   / dt
    root_vel_xy[-1]   = (root_pos[-1, :2] - root_pos[-2, :2])  / dt

    pelvis_height = root_pos[:, 2:3].astype(np.float32)
    left_eef      = joint_positions[:, L_HAND_IDX, :].astype(np.float32)
    right_eef     = joint_positions[:, R_HAND_IDX, :].astype(np.float32)

    return np.concatenate([root_vel_xy, pelvis_height, left_eef, right_eef], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract task-space features from AMASS PKL to HDF5 (poselib FK)'
    )
    parser.add_argument('input',  type=str, help='Input .pkl file (joblib format)')
    parser.add_argument('output', type=str, help='Output .h5 file')
    parser.add_argument(
        '--humanoid-xml', type=str, default='final_humanoid.xml',
        help='MJCF used to build the poselib skeleton tree (default: final_humanoid.xml)',
    )
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        sys.exit(f'ERROR: input file not found: {in_path}')
    if not Path(args.humanoid_xml).exists():
        sys.exit(f'ERROR: humanoid XML not found: {args.humanoid_xml}')

    print(f'Building skeleton tree from {args.humanoid_xml} …')
    sk_tree = SkeletonTree.from_mjcf(args.humanoid_xml)
    print(f'  joints: {sk_tree.num_joints}')

    print(f'Loading {in_path} …')
    data = joblib.load(str(in_path))
    print(f'  → {len(data)} clips\n')

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), 'w') as hf:
        clips_grp = hf.create_group('clips')
        hf.attrs['source_file']    = str(in_path)
        hf.attrs['humanoid_xml']   = args.humanoid_xml
        hf.attrs['fk_backend']     = 'poselib'
        hf.attrs['joint_names']    = JOINT_NAMES
        hf.attrs['task_space_cols'] = [
            'root_vel_x', 'root_vel_y',
            'pelvis_height',
            'left_hand_x', 'left_hand_y', 'left_hand_z',
            'right_hand_x', 'right_hand_y', 'right_hand_z',
        ]

        for i, (clip_name, clip) in enumerate(data.items(), 1):
            print(f'[{i:3d}/{len(data)}] {clip_name}')

            joint_positions = compute_joint_positions(clip, sk_tree)  # (T, 24, 3)
            fps             = float(clip['fps'])
            task_space      = extract_task_space(joint_positions, fps)
            T               = joint_positions.shape[0]

            pelvis_z = joint_positions[:, PELVIS_IDX,  2]
            l_hand_z = joint_positions[:, L_HAND_IDX,  2]
            r_hand_z = joint_positions[:, R_HAND_IDX,  2]
            vel_mag  = np.linalg.norm(task_space[:, :2], axis=1)

            print(f'       frames={T}  fps={fps}')
            print(f'       pelvis_height : [{pelvis_z.min():.3f}, {pelvis_z.max():.3f}] m')
            print(f'       L_hand_height : [{l_hand_z.min():.3f}, {l_hand_z.max():.3f}] m')
            print(f'       R_hand_height : [{r_hand_z.min():.3f}, {r_hand_z.max():.3f}] m')
            print(f'       root_speed_xy : [{vel_mag.min():.3f}, {vel_mag.max():.3f}] m/s')

            # pose_aa_body: 23 body joints (no root) in MuJoCo order, axis-angle.
            # Use pose_quat (local, MuJoCo-ordered) NOT pose_aa (which is SMPL-ordered).
            if 'pose_quat' in clip:
                from scipy.spatial.transform import Rotation as _sRot
                pq = np.asarray(clip['pose_quat'], dtype=np.float32)[:T]     # (T, 24, 4) xyzw local MuJoCo
                pose_aa_mj = _sRot.from_quat(pq.reshape(-1, 4)).as_rotvec().reshape(T, 24, 3)
                pose_aa_body = pose_aa_mj[:, 1:, :].reshape(T, 69).astype(np.float32)  # body joints
                pose_aa_root = pose_aa_mj[:, 0, :].astype(np.float32)               # (T, 3) pelvis
            else:
                pose_aa_body = None
                pose_aa_root = None

            safe_name = clip_name.replace('/', '_').replace('\\', '_')
            grp = clips_grp.create_group(safe_name)

            grp.create_dataset('task_space',
                               data=task_space,
                               dtype='f4',
                               compression='gzip', compression_opts=4)
            grp.create_dataset('joint_positions',
                               data=joint_positions,
                               dtype='f4',
                               compression='gzip', compression_opts=4)
            if pose_aa_body is not None:
                grp.create_dataset('pose_aa_body',
                                   data=pose_aa_body,
                                   dtype='f4',
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('pose_aa_root',
                                   data=pose_aa_root,
                                   dtype='f4',
                                   compression='gzip', compression_opts=4)

            grp.attrs['clip_name'] = clip_name
            grp.attrs['fps']       = fps
            grp.attrs['n_frames']  = T

    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
