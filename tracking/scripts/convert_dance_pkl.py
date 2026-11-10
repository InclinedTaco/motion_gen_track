"""Convert dance sample pkl to PHC format (adds pose_quat_global, pose_quat, etc.)"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

robot_cfg = {
    "mesh": False,
    "model": "smpl",
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}

smpl_local_robot = LocalRobot(robot_cfg, data_dir="data/smpl")
beta = np.zeros(10)
gender_number = [0]
gender = "neutral"
smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
smpl_local_robot.write_xml("/tmp/smpl_humanoid_tmp.xml")
skeleton_tree = SkeletonTree.from_mjcf("/tmp/smpl_humanoid_tmp.xml")

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
                      'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                      'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]

input_files = [
    ("sample_data/dance_sample_h1.pkl", "sample_data/dance_sample_h1_phc.pkl"),
    ("sample_data/dance_sample_g1.pkl", "sample_data/dance_sample_g1_phc.pkl"),
]

for in_path, out_path in input_files:
    if not os.path.exists(in_path):
        print(f"Skipping {in_path} (not found)")
        continue

    print(f"Converting {in_path} ...")
    raw = joblib.load(in_path)
    full_motion_dict = {}

    for key_name in tqdm(raw.keys()):
        entry = raw[key_name]
        pose_aa = entry['pose_aa'].copy()
        root_trans_offset = torch.from_numpy(entry['root_trans_offset'].copy()) if isinstance(entry['root_trans_offset'], np.ndarray) else entry['root_trans_offset'].clone()
        fps = entry.get('fps', 30.0)
        B = pose_aa.shape[0]

        # normalize to (B, 72) flat then back to (B, 24, 3) in mujoco order
        pose_aa_flat = pose_aa.reshape(B, -1)[:, :72]  # handle (B,25,3) or (B,72)
        pose_aa = np.concatenate([pose_aa_flat[:, :66], np.zeros((B, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(B, 24, 4)

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        # dance pkl is already in Z-up space, skip the Y->Z upright rotation
        pose_quat_global = new_sk_state.global_rotation.numpy().reshape(B, -1, 4)
        pose_quat = new_sk_state.local_rotation.numpy()

        full_motion_dict[key_name] = {
            'pose_quat_global': pose_quat_global,
            'pose_quat': pose_quat,
            'trans_orig': root_trans_offset.numpy(),
            'root_trans_offset': root_trans_offset,
            'beta': beta,
            'gender': gender,
            'pose_aa': pose_aa,
            'fps': fps,
        }

    joblib.dump(full_motion_dict, out_path)
    print(f"Saved to {out_path}")
