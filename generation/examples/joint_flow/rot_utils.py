"""
6D rotation representation utilities (Zhou et al. 2019).

6D encoding: [col0, col1] of the rotation matrix — 6 values per joint.
Decoding: Gram-Schmidt orthonormalization to recover the full 3×3 matrix.

These are used when the flow model is trained with --use-6d to output
body-joint rotations directly (skipping the gradient-based retargeter).
"""

import numpy as np
from scipy.spatial.transform import Rotation as sRot


def pose_aa_to_6d(pose_aa_body: np.ndarray) -> np.ndarray:
    """
    Convert body-joint axis-angles to 6D rotation representation.

    pose_aa_body : (T, 23, 3) or (T, 69)  — local axis-angle, MuJoCo order
    Returns      : (T, 138)
    """
    if pose_aa_body.ndim == 2 and pose_aa_body.shape[1] == 69:
        T = pose_aa_body.shape[0]
        pose_aa_body = pose_aa_body.reshape(T, 23, 3)
    T, J, _ = pose_aa_body.shape
    R = sRot.from_rotvec(pose_aa_body.reshape(-1, 3)).as_matrix()  # (T*J, 3, 3)
    r6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)       # (T*J, 6)
    return r6d.reshape(T, J * 6).astype(np.float32)                # (T, 138)


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """
    Gram-Schmidt orthonormalization: (*, 6) → (*, 3, 3).

    r6d[..., :3] = first  column of R
    r6d[..., 3:] = second column of R (before orthogonalization)
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (a2 * b1).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns = basis vectors


def rot6d_to_pose_aa(r6d_flat: np.ndarray) -> np.ndarray:
    """
    Convert 6D body-joint output back to axis-angle.

    r6d_flat : (T, 138)
    Returns  : (T, 69)  — 23 body joints × 3 axis-angle
    """
    T = r6d_flat.shape[0]
    r6d = r6d_flat.reshape(T, 23, 6)
    R = rot6d_to_matrix(r6d)                      # (T, 23, 3, 3)
    R_flat = R.reshape(-1, 3, 3)
    # Replace degenerate (NaN/Inf) matrices with identity before converting
    bad = ~np.isfinite(R_flat).all(axis=(-2, -1))
    if bad.any():
        R_flat = R_flat.copy()
        R_flat[bad] = np.eye(3, dtype=R_flat.dtype)
    aa = sRot.from_matrix(R_flat).as_rotvec()     # (T*23, 3)
    return aa.reshape(T, 69).astype(np.float32)


def velocity_to_root_rotvec(vel_x: float, vel_y: float) -> np.ndarray:
    """
    Estimate pelvis/root axis-angle from the horizontal velocity direction.

    Convention (final_humanoid.xml, Z-up): local +X = forward, +Y = left.
    When speed ≈ 0 the character faces +X (world forward).

    Returns: (3,) axis-angle
    """
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    if speed > 1e-3:
        forward = np.array([vel_x, vel_y, 0.0], dtype=np.float64) / speed
    else:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    left = np.array([-forward[1], forward[0], 0.0])
    up   = np.array([0.0, 0.0, 1.0])
    R    = np.stack([forward, left, up], axis=-1)                # columns
    return sRot.from_matrix(R).as_rotvec().astype(np.float32)
