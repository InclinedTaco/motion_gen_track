"""
Take a crouch/squat clip and freeze both arms in T-pose throughout:
arms extended straight sideways, elbows/wrists/hands identity rotation.

Output: data/amass/pkls/squat_tpose_hands_final.pkl
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from tqdm import tqdm

# ─── Skeleton ────────────────────────────────────────────────────────────────
TARGET_XML = 'final_humanoid.xml'
sk = SkeletonTree.from_mjcf(TARGET_XML)

NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
]
J = {n: i for i, n in enumerate(NAMES)}

# Rest-pose (T-pose) global rotations — all local rotations = identity
zero_state = SkeletonState.zero_pose(sk)
rest_global_rot = zero_state.global_rotation.numpy()  # (24, 4) xyzw

# ─── Load motion ─────────────────────────────────────────────────────────────
PKL_IN  = 'data/amass/pkls/accad_isaac_upright_slim.pkl'
PKL_OUT = 'data/amass/pkls/squat_tpose_hands_final.pkl'

data = joblib.load(PKL_IN)
SQUAT_KEY = 'ACCAD_Male1General_c3d_General A7 - Crouch_poses'
entry = data[SQUAT_KEY]

def _np(v): return v.numpy() if hasattr(v, 'numpy') else np.array(v)

pose_qg        = _np(entry['pose_quat_global']).copy()  # (B, 24, 4) xyzw global
pose_quat      = _np(entry['pose_quat']).copy()          # (B, 24, 4) local
pose_aa        = _np(entry['pose_aa']).copy()            # (B, 72)
root_trans_off = entry['root_trans_offset']
fps            = float(entry.get('fps', 30.0))
B              = pose_qg.shape[0]
print(f'Motion: {SQUAT_KEY}   B={B}  fps={fps}')
print(f'Target XML: {TARGET_XML}')

# Joints to freeze to T-pose
ARM_JOINTS = [
    # (shoulder,       elbow,        wrist,        hand,         thorax)
    (J['L_Shoulder'], J['L_Elbow'], J['L_Wrist'], J['L_Hand'], J['L_Thorax']),
    (J['R_Shoulder'], J['R_Elbow'], J['R_Wrist'], J['R_Hand'], J['R_Thorax']),
]

# T-pose shoulder global rotation (from zero/rest pose)
L_SHLDR_REST = sRot.from_quat(rest_global_rot[J['L_Shoulder']])
R_SHLDR_REST = sRot.from_quat(rest_global_rot[J['R_Shoulder']])
REST_ROTS = [L_SHLDR_REST, R_SHLDR_REST]

for f in tqdm(range(B)):
    for (SHLDR, ELBW, WRST, HAND, THORAX), R_shldr_tpose in zip(ARM_JOINTS, REST_ROTS):
        # Set shoulder global rotation = T-pose shoulder rotation
        pose_qg[f, SHLDR] = R_shldr_tpose.as_quat()

        # Propagate: elbow/wrist/hand global = same as shoulder (straight arm)
        for jj in [ELBW, WRST, HAND]:
            pose_qg[f, jj] = R_shldr_tpose.as_quat()

        # Shoulder local = R_thorax_global_inv @ R_shoulder_tpose
        R_thorax = sRot.from_quat(pose_qg[f, THORAX])
        R_shldr_local = R_thorax.inv() * R_shldr_tpose
        pose_quat[f, SHLDR] = R_shldr_local.as_quat()
        pose_aa[f, SHLDR*3 : SHLDR*3+3] = R_shldr_local.as_rotvec()

        # Elbow, wrist, hand: straight arm → local = identity
        for child in [ELBW, WRST, HAND]:
            pose_quat[f, child] = sRot.identity().as_quat()
            pose_aa[f, child*3 : child*3+3] = 0.0

# ─── Save ─────────────────────────────────────────────────────────────────────
new_entry = {
    'pose_quat_global':  pose_qg,
    'pose_quat':         pose_quat,
    'trans_orig':        _np(entry.get('trans_orig', _np(root_trans_off))),
    'root_trans_offset': root_trans_off,
    'beta':              _np(entry['beta']),
    'gender':            entry['gender'],
    'pose_aa':           pose_aa,
    'fps':               fps,
}
joblib.dump({'squat_tpose_hands': new_entry}, PKL_OUT)
print(f'\nSaved → {PKL_OUT}')
