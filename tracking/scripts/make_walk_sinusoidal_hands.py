"""
Take a straight walking clip and replace arm rotations with a sinusoidal
T-pose swing: arms extended sideways, hands oscillating up/down, 180 deg out of phase.

Output: data/amass/pkls/walk_sinusoidal_hands.pkl
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from tqdm import tqdm

# ─── Skeleton ────────────────────────────────────────────────────────────────
robot_cfg = {
    'mesh': False, 'model': 'smpl', 'upright_start': True,
    'body_params': {}, 'joint_params': {}, 'geom_params': {}, 'actuator_params': {},
}
robot = LocalRobot(robot_cfg, data_dir='data/smpl')
robot.load_from_skeleton(betas=torch.zeros(1, 10), gender=[0], objs_info=None)
robot.write_xml('/tmp/smpl_sinhand.xml')
sk = SkeletonTree.from_mjcf('/tmp/smpl_sinhand.xml')

NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
]
J = {n: i for i, n in enumerate(NAMES)}

# Rest-pose global positions (all local rotations = identity)
rp = SkeletonState.zero_pose(sk).global_translation.numpy()  # (24, 3)

def _norm(v): return v / np.linalg.norm(v)

# Arm axis in the shoulder's local frame (= world direction in rest pose)
L_ARM_AXIS = _norm(rp[J['L_Elbow']] - rp[J['L_Shoulder']])
R_ARM_AXIS = _norm(rp[J['R_Elbow']] - rp[J['R_Shoulder']])

# Horizontal projections — base T-pose direction per frame
L_ARM_H = _norm(L_ARM_AXIS * np.array([1., 1., 0.]))
R_ARM_H = _norm(R_ARM_AXIS * np.array([1., 1., 0.]))

# Total arm length (shoulder → hand, straight)
def _bone(a, b): return np.linalg.norm(rp[J[b]] - rp[J[a]])
L_ARM_LEN = _bone('L_Shoulder', 'L_Elbow') + _bone('L_Elbow', 'L_Wrist') + _bone('L_Wrist', 'L_Hand')
R_ARM_LEN = _bone('R_Shoulder', 'R_Elbow') + _bone('R_Elbow', 'R_Wrist') + _bone('R_Wrist', 'R_Hand')

print(f'L arm: {L_ARM_LEN:.3f}m   R arm: {R_ARM_LEN:.3f}m')

# ─── Load motion ─────────────────────────────────────────────────────────────
PKL_IN  = 'data/amass/pkls/accad_isaac_upright_slim_filtered.pkl'
PKL_OUT = 'data/amass/pkls/walk_sinusoidal_hands.pkl'

data = joblib.load(PKL_IN)
WALK_KEY = 'ACCAD_s007_QkWalk1_poses'
entry = data[WALK_KEY]

def _np(v): return v.numpy() if hasattr(v, 'numpy') else np.array(v)

# Original data — keep root_trans_offset EXACTLY as-is (tensor, for PHC compat)
pose_qg        = _np(entry['pose_quat_global']).copy()  # (B, 24, 4) xyzw, global rots
pose_quat      = _np(entry['pose_quat']).copy()          # (B, 24, 4) local rots
pose_aa        = _np(entry['pose_aa']).copy()            # (B, 72)
root_trans_off = entry['root_trans_offset']              # keep as original (torch tensor)
fps            = float(entry.get('fps', 30.0))
B              = pose_qg.shape[0]
print(f'Motion: {WALK_KEY}   B={B}  fps={fps}')

# ─── Sinusoidal parameters ───────────────────────────────────────────────────
FREQ      = 1.0   # Hz
AMPLITUDE = 0.15  # m

t_arr = np.arange(B) / fps

CHEST = J['Chest']

ARM_CFG = [
    # SHLDR            ELBW             WRST             HAND            THORAX           arm_axis  arm_h    arm_len  phase
    (J['L_Shoulder'], J['L_Elbow'], J['L_Wrist'], J['L_Hand'], J['L_Thorax'], L_ARM_AXIS, L_ARM_H, L_ARM_LEN, 0.0),
    (J['R_Shoulder'], J['R_Elbow'], J['R_Wrist'], J['R_Hand'], J['R_Thorax'], R_ARM_AXIS, R_ARM_H, R_ARM_LEN, np.pi),
]

# ─── Main loop: modify global rotations for arm joints ───────────────────────
for f in tqdm(range(B)):
    R_chest = sRot.from_quat(pose_qg[f, CHEST])

    for (SHLDR, ELBW, WRST, HAND, THORAX, arm_axis, arm_h, arm_len, phase) in ARM_CFG:
        body_side = R_chest.apply(arm_h)
        body_side[2] = 0.0
        body_side = _norm(body_side)

        z_off      = AMPLITUDE * np.sin(2.0 * np.pi * FREQ * t_arr[f] + phase)
        h_reach    = np.sqrt(max(0.0, arm_len**2 - z_off**2))
        target_dir = _norm(body_side * h_reach + np.array([0., 0., z_off]))

        R_shldr    = sRot.from_quat(pose_qg[f, SHLDR])
        arm_world  = R_shldr.apply(arm_axis)

        cross = np.cross(arm_world, target_dir)
        cn    = np.linalg.norm(cross)
        if cn < 1e-7:
            R_shldr_new = R_shldr
        else:
            angle       = np.arccos(np.clip(np.dot(arm_world, target_dir), -1.0, 1.0))
            R_shldr_new = sRot.from_rotvec(cross / cn * angle) * R_shldr

        # Update global rotations (for pose_qg)
        for jj in [SHLDR, ELBW, WRST, HAND]:
            pose_qg[f, jj] = R_shldr_new.as_quat()

        # Directly update LOCAL rotations in pose_quat/pose_aa
        # Shoulder local = R_thorax_global_inv @ R_shoulder_new_global
        R_thorax      = sRot.from_quat(pose_qg[f, THORAX])
        R_shldr_local = R_thorax.inv() * R_shldr_new
        pose_quat[f, SHLDR] = R_shldr_local.as_quat()
        pose_aa[f, SHLDR*3 : SHLDR*3+3] = R_shldr_local.as_rotvec()

        # Elbow, wrist, hand: straight arm → local rotation = identity
        for child in [ELBW, WRST, HAND]:
            pose_quat[f, child] = sRot.identity().as_quat()
            pose_aa[f, child*3 : child*3+3] = 0.0

# ─── Save — root_trans_offset kept identical to original ─────────────────────
new_entry = {
    'pose_quat_global':  pose_qg,
    'pose_quat':         pose_quat,
    'trans_orig':        _np(entry.get('trans_orig', _np(root_trans_off))),
    'root_trans_offset': root_trans_off,   # original torch tensor, untouched
    'beta':              _np(entry['beta']),
    'gender':            entry['gender'],
    'pose_aa':           pose_aa,
    'fps':               fps,
}
joblib.dump({'walk_sinusoidal_hands': new_entry}, PKL_OUT)
print(f'\nSaved → {PKL_OUT}')
