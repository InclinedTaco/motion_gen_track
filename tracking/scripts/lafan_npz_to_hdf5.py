#!/usr/bin/env python3
"""
Convert lafan_to_smplx NPZ files to flow-model HDF5 training format.

Run from PHC repo root in the phc conda env:
    conda activate phc && cd <PROJECT_ROOT>/tracking
    python scripts/lafan_npz_to_hdf5.py \
        --input_dir /tmp/lafan_npz \
        --output data/amass/hdf5/lafan_locomotion_taskspace.h5

The NPZ files are produced by:
    conda activate gmr && cd /home/marmot/Shyam/lafan_to_smplx
    python lafan_to_smplx.py --bvh_file <clip>.bvh \
        --smplx_model_path <PROJECT_ROOT>/tracking/data \
        --model_type smpl --output_file <clip>.npz

NPZ format:
    poses:  (T, 24, 3) float32 - SMPL axis-angle in SMPL joint order, already Z-up
    trans:  (T, 3)     float32 - world pelvis translation, Z-up, meters
    betas:  (10,)      float32
    gender: "neutral"
    mocap_frame_rate: 30
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

# ---------------------------------------------------------------------------
# Joint layout (poselib / Isaac Gym MJCF DFS order, matches final_humanoid.xml)
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
    'L_Hand',     # 18  ← left EEF
    'R_Thorax',   # 19
    'R_Shoulder', # 20
    'R_Elbow',    # 21
    'R_Wrist',    # 22
    'R_Hand',     # 23  ← right EEF
]
PELVIS_IDX = 0
L_HAND_IDX = 18
R_HAND_IDX = 23

# SMPL standard bone order → MuJoCo joint order remapping
# SMPL: Pelvis L_Hip R_Hip Spine1 L_Knee R_Knee Spine2 L_Ankle R_Ankle Spine3
#       L_Foot R_Foot Neck L_Collar R_Collar Head L_Shoulder R_Shoulder
#       L_Elbow R_Elbow L_Wrist R_Wrist L_Hand R_Hand
SMPL_2_MUJOCO = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]


def extract_task_space(joint_positions: np.ndarray, fps: float) -> np.ndarray:
    T = joint_positions.shape[0]
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


def npz_to_joint_positions(npz, sk_tree):
    """Convert lafan_to_smplx NPZ to (T,24,3) world-frame joint positions.

    The NPZ is already Z-up — no upright rotation correction needed (unlike
    AMASS data which comes in Y-up and needs convert_amass_isaac treatment).
    """
    poses = npz['poses'].astype(np.float64)   # (T, 24, 3) SMPL axis-angle, Z-up
    trans = npz['trans'].astype(np.float64)   # (T, 3) world pelvis pos, Z-up, meters
    T = poses.shape[0]

    # Flatten to (T, 72) and reorder from SMPL → MuJoCo joint order
    pose_aa_smpl = poses.reshape(T, 24, 3)
    pose_aa_mj   = pose_aa_smpl[:, SMPL_2_MUJOCO, :]  # (T, 24, 3)

    # Convert axis-angle → local quaternions (xyzw)
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(T, 24, 4)

    # Align root translation to poselib skeleton (same adjustment as convert_amass_isaac)
    root_trans_offset = torch.from_numpy(trans) + sk_tree.local_translation[0]

    # Build skeleton state with local rotations
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )

    joint_positions = sk_state.global_translation.numpy().astype(np.float32)  # (T, 24, 3)
    local_quat      = sk_state.local_rotation.numpy().astype(np.float32)      # (T, 24, 4)

    fps = float(npz.get('mocap_frame_rate', 30))
    return joint_positions, local_quat, float(fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Directory of NPZ files from lafan_to_smplx')
    parser.add_argument('--output', required=True, help='Output HDF5 path')
    parser.add_argument('--humanoid_xml', default='final_humanoid.xml',
                        help='MJCF used to build poselib skeleton tree')
    args = parser.parse_args()

    if not Path(args.humanoid_xml).exists():
        sys.exit(f'ERROR: humanoid XML not found: {args.humanoid_xml}')

    print(f'Building skeleton tree from {args.humanoid_xml} …')
    sk_tree = SkeletonTree.from_mjcf(args.humanoid_xml)
    print(f'  joints: {sk_tree.num_joints}')

    npz_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npz')))
    if not npz_files:
        sys.exit(f'No NPZ files found in {args.input_dir}')
    print(f'Found {len(npz_files)} NPZ files\n')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), 'w') as hf:
        clips_grp = hf.create_group('clips')
        hf.attrs['source']       = 'LAFAN1'
        hf.attrs['joint_names']  = JOINT_NAMES
        hf.attrs['task_space_cols'] = [
            'root_vel_x', 'root_vel_y', 'pelvis_height',
            'left_hand_x', 'left_hand_y', 'left_hand_z',
            'right_hand_x', 'right_hand_y', 'right_hand_z',
        ]

        for npz_path in tqdm(npz_files):
            clip_name = 'LAFAN_' + Path(npz_path).stem
            print(f'  {clip_name}')

            npz = np.load(npz_path, allow_pickle=True)
            joint_positions, local_quat, fps = npz_to_joint_positions(npz, sk_tree)
            T = joint_positions.shape[0]
            task_space = extract_task_space(joint_positions, fps)

            pelvis_z = joint_positions[:, PELVIS_IDX, 2]
            vel_mag  = np.linalg.norm(task_space[:, :2], axis=1)
            print(f'    frames={T}  fps={fps}')
            print(f'    pelvis_z : [{pelvis_z.min():.3f}, {pelvis_z.max():.3f}] m')
            print(f'    speed_xy : [{vel_mag.min():.3f}, {vel_mag.max():.3f}] m/s')

            # pose_aa_body: 23 body joints (no pelvis) in MuJoCo order, axis-angle
            pose_aa_mj   = sRot.from_quat(local_quat.reshape(-1, 4)).as_rotvec().reshape(T, 24, 3)
            pose_aa_body = pose_aa_mj[:, 1:, :].reshape(T, 69).astype(np.float32)
            pose_aa_root = pose_aa_mj[:, 0, :].astype(np.float32)

            grp = clips_grp.create_group(clip_name)
            grp.create_dataset('task_space',      data=task_space,      dtype='f4', compression='gzip', compression_opts=4)
            grp.create_dataset('joint_positions', data=joint_positions, dtype='f4', compression='gzip', compression_opts=4)
            grp.create_dataset('pose_aa_body',    data=pose_aa_body,    dtype='f4', compression='gzip', compression_opts=4)
            grp.create_dataset('pose_aa_root',    data=pose_aa_root,    dtype='f4', compression='gzip', compression_opts=4)
            grp.attrs['clip_name'] = clip_name
            grp.attrs['fps']       = fps
            grp.attrs['n_frames']  = T

    print(f'\nSaved → {args.output}')


if __name__ == '__main__':
    main()
