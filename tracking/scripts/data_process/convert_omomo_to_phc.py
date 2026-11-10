"""
Convert OMOMO motion sequences (mac_media_sequences.p) to PHC pkl format.

Usage (from PHC root, with phc conda env):
    python scripts/data_process/convert_omomo_to_phc.py \
        --in_file /home/marmot/mac_motion_seqs/mac_media_sequences.p \
        --out_file data/amass/pkls/omomo_isaac_upright.pkl

Output format matches convert_amass_isaac.py:
    pose_quat_global: (T, 24, 4)
    pose_quat:        (T, 24, 4)
    root_trans_offset:(T, 3)
    trans_orig:       (T, 3)
    pose_aa:          (T, 72)
    beta:             (10,)
    gender:           str
    fps:              float
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import joblib
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

sys.path.append(os.getcwd())

from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

MUJOCO_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
]

SMPL_DATA_DIR = "data/smpl"
SMPL_XML_PATH = "phc/data/assets/mjcf/smpl_humanoid_1.xml"

# rotation that converts AMASS y-up rest pose to PHC upright (z-up)
UPRIGHT_QUAT = sRot.from_quat([0.5, 0.5, 0.5, 0.5])


def fix_floor_height(smpl_parser, pose_aa_t, betas_t, trans_t, gender):
    """Shift trans_z so the lowest body vertex in frame 0 is at z=0."""
    with torch.no_grad():
        verts, _ = smpl_parser.get_joints_verts(
            pose_aa_t[0:1], betas_t[0:1], th_trans=trans_t[0:1]
        )
    gp = torch.min(verts[:, :, 2]).item()
    trans_t = trans_t.clone()
    trans_t[:, 2] -= gp
    return trans_t


def get_smpl_parser(gender, smpl_data_dir):
    return SMPL_Parser(
        model_path=smpl_data_dir,
        gender=gender,
        use_pca=False,
        create_transl=False,
    )


def convert_sequence(seq_name, smpl_data, smpl_local_robot, smpl_parsers,
                     smpl_2_mujoco, fps=30.0):
    root_orient = smpl_data["root_orient"]   # (T, 3)
    pose_body = smpl_data["pose_body"]       # (T, 63)
    trans = smpl_data["trans"]               # (T, 3)
    betas_raw = smpl_data["betas"]           # (1, 16) or (16,)
    gender_raw = smpl_data["gender"]

    T = root_orient.shape[0]

    # -- gender string --
    if isinstance(gender_raw, np.ndarray):
        gender = str(gender_raw.item())
    else:
        gender = str(gender_raw)
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"

    # -- betas: take first 10 dims --
    betas = betas_raw.flatten()[:10].astype(np.float32)

    # -- pose_aa: (T, 72) = root(3) + body(63) + zeros(6) --
    pose_aa = np.concatenate(
        [root_orient, pose_body, np.zeros((T, 6))], axis=1
    ).astype(np.float32)

    # -- reorder joints to MuJoCo order --
    pose_aa_mj = pose_aa.reshape(T, 24, 3)[:, smpl_2_mujoco, :]

    # -- tensors --
    pose_aa_t = torch.from_numpy(pose_aa).float()
    pose_aa_mj_t = torch.from_numpy(pose_aa_mj).float()
    trans_t = torch.from_numpy(trans.astype(np.float32))
    betas_t = torch.from_numpy(betas[None]).float()  # (1, 10)

    # -- fix floor height using SMPL FK --
    smpl_parser = smpl_parsers[gender]
    trans_t = fix_floor_height(smpl_parser, pose_aa_t, betas_t, trans_t, gender)

    # -- build MuJoCo SMPL robot for this beta/gender --
    gender_number = {"neutral": [0], "male": [1], "female": [2]}[gender]
    smpl_local_robot.load_from_skeleton(
        betas=betas_t, gender=gender_number, objs_info=None
    )
    smpl_local_robot.write_xml(SMPL_XML_PATH)
    skeleton_tree = SkeletonTree.from_mjcf(SMPL_XML_PATH)

    # -- root translation with pelvis offset --
    root_trans_offset = trans_t + skeleton_tree.local_translation[0]

    # -- local quats from MuJoCo-ordered axis-angle --
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj_t.reshape(-1, 3).numpy())
        .as_quat()
        .reshape(T, 24, 4)
    )

    # -- build SkeletonState and convert to global (upright) frame --
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True,
    )

    pose_quat_global = (
        sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
        * UPRIGHT_QUAT.inv()
    ).as_quat().reshape(T, 24, 4)

    new_sk_state2 = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        root_trans_offset,
        is_local=False,
    )
    pose_quat = new_sk_state2.local_rotation.numpy()

    return {
        "pose_quat_global": pose_quat_global.astype(np.float32),
        "pose_quat":        pose_quat.astype(np.float32),
        "trans_orig":       trans,
        "root_trans_offset": root_trans_offset.float(),  # must be torch tensor (.clone() is called on it)
        "beta":             betas,
        "gender":           gender,
        "pose_aa":          pose_aa,
        "fps":              fps,
        "seq_name":         seq_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",  default="/home/marmot/mac_motion_seqs/mac_media_sequences.p")
    parser.add_argument("--out_file", default="data/amass/pkls/omomo_isaac_upright.pkl")
    parser.add_argument("--fps",      type=float, default=30.0,
                        help="OMOMO capture frame rate (default 30 Hz)")
    args = parser.parse_args()

    print(f"Loading OMOMO data from {args.in_file} ...")
    omomo_data = joblib.load(args.in_file)
    print(f"  {len(omomo_data)} sequences found.")

    robot_cfg = {
        "mesh": False, "model": "smpl", "upright_start": True,
        "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {},
    }
    smpl_local_robot = LocalRobot(robot_cfg, data_dir=SMPL_DATA_DIR)

    # pre-build SMPL parsers for each gender
    smpl_parsers = {
        g: get_smpl_parser(g, SMPL_DATA_DIR)
        for g in ("neutral", "male", "female")
    }

    smpl_2_mujoco = [joint_names.index(q) for q in MUJOCO_JOINT_NAMES if q in joint_names]

    results = {}
    failed = []

    for seq_name, smpl_data in tqdm(omomo_data.items()):
        try:
            entry = convert_sequence(
                seq_name, smpl_data, smpl_local_robot, smpl_parsers,
                smpl_2_mujoco, fps=args.fps,
            )
            results[seq_name] = entry
        except Exception as e:
            print(f"  FAILED {seq_name}: {e}")
            failed.append(seq_name)

    print(f"\nConverted {len(results)}/{len(omomo_data)} sequences. Failed: {len(failed)}")
    if failed:
        print("  Failed sequences:", failed)

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, args.out_file)
    print(f"Saved to {args.out_file}")


if __name__ == "__main__":
    main()
