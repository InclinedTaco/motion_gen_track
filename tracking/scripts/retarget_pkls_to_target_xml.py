import argparse
import glob
import os
import os.path as osp
import sys

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES


MUJOCO_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

UPRIGHT_INV = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_pose_aa(pose_aa):
    pose_aa = _to_numpy(pose_aa).copy()
    batch = pose_aa.shape[0]
    pose_aa_flat = pose_aa.reshape(batch, -1)[:, :72]
    if pose_aa_flat.shape[1] < 72:
        pose_aa_flat = np.pad(pose_aa_flat, ((0, 0), (0, 72 - pose_aa_flat.shape[1])))
    # Keep the PHC SMPL convention: 24 bodies with zeroed hand pose tail when needed.
    pose_aa_flat = np.concatenate([pose_aa_flat[:, :66], np.zeros((batch, 6))], axis=1)
    return pose_aa_flat


def retarget_entry(entry, skeleton_tree, smpl_2_mujoco, upright_start):
    pose_aa = _normalize_pose_aa(entry["pose_aa"] if "pose_aa" in entry else entry["poses"])
    trans_orig = _to_numpy(entry.get("trans_orig", entry["root_trans_offset"]))

    batch = pose_aa.shape[0]
    pose_aa_mj = pose_aa.reshape(batch, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch, 24, 4)

    root_trans_offset = torch.from_numpy(trans_orig).float() + skeleton_tree.local_translation[0]
    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat).float(),
        root_trans_offset,
        is_local=True,
    )

    if upright_start:
        pose_quat_global = (
            sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * UPRIGHT_INV
        ).as_quat().reshape(batch, -1, 4)
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global).float(),
            root_trans_offset,
            is_local=False,
        )

    out = dict(entry)
    out["pose_quat_global"] = sk_state.global_rotation.numpy()
    out["pose_quat"] = sk_state.local_rotation.numpy()
    out["trans_orig"] = trans_orig
    out["root_trans_offset"] = root_trans_offset
    out["pose_aa"] = pose_aa
    return out


def retarget_file(in_file, out_file, skeleton_tree, smpl_2_mujoco, upright_start):
    motions = joblib.load(in_file)
    converted = {}

    for key, entry in tqdm(motions.items(), desc=osp.basename(in_file), leave=False):
        converted[key] = retarget_entry(entry, skeleton_tree, smpl_2_mujoco, upright_start)

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    joblib.dump(converted, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", default="data/amass/pkls/*.pkl")
    parser.add_argument("--target_xml", default="final_humanoid.xml")
    parser.add_argument("--out_dir", default="data/amass/pkls/final_humanoid")
    parser.add_argument("--upright_start", action="store_true", default=True)
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.input_glob}")

    skeleton_tree = SkeletonTree.from_mjcf(args.target_xml)
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(name) for name in MUJOCO_JOINT_NAMES]

    print(f"Target XML: {args.target_xml}")
    print(f"Output dir: {args.out_dir}")
    print(f"Files: {len(files)}")

    for in_file in tqdm(files, desc="Retargeting files"):
        out_file = osp.join(args.out_dir, osp.basename(in_file))
        retarget_file(in_file, out_file, skeleton_tree, smpl_2_mujoco, args.upright_start)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
