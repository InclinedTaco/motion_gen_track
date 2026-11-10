"""
flow_retarget.py

Gradient-based retargeting of flow-model joint POSITIONS to SMPL-humanoid POSE
(axis-angle `pose_aa` + global/local joint quaternions), using the *same*
differentiable forward-kinematics model PHC uses internally (`Humanoid_Batch`).

Why this exists
---------------
The flow model emits 24 world-frame joint positions.  The previous converter
(`positions_to_global_quats` in flow_to_pkl.py) used per-joint
`scipy.align_vectors`, which only recovers each bone's *direction* — the
roll/twist about the bone axis is left arbitrary and is solved independently
per frame.  Positions reconstruct to ~1 cm, but the *rotations* jitter wildly,
so PHC's motion lib derives garbage angular velocities / DOF targets and the
tracking policy flails ("random limbs, correct root velocity").

This module instead fits `pose_aa` by gradient descent so that FK joint
positions match the targets, with a small pose prior (resolves the roll
ambiguity toward minimal rotation) and temporal smoothing (removes jitter).
The output rotations are anatomically consistent and smooth, and `pose_aa` is
real (not zeros), so PHC's foot-height fix also works.

Two entry points:
  * fit_sequence(...)  — batch fit a whole (T,24,3) motion (offline pkl).
  * StreamFitter       — per-frame warm-started fit for the real-time server.

Joint order is MuJoCo DFS (Pelvis, L_Hip, L_Knee, ... R_Hand) — identical to the
flow model's output order, so the correspondence is the identity.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

PHC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHC_ROOT))

from phc.utils.torch_humanoid_batch import Humanoid_Batch

ROBOT_CFG = PHC_ROOT / "phc" / "data" / "cfg" / "robot" / "smpl_humanoid.yaml"
PELVIS_IDX = 0
L_HIP_IDX = 1
R_HIP_IDX = 5
TORSO_IDX = 9


def build_fk(humanoid_xml, device=torch.device("cpu")):
    """Instantiate the differentiable SMPL forward-kinematics model.

    humanoid_xml must be the *same* XML used by PHC at eval time (e.g.
    final_humanoid.xml) so bone lengths and the rest frame match.
    """
    cfg = OmegaConf.load(ROBOT_CFG)
    cfg.asset.assetFileName = str(humanoid_xml)
    cfg.extend_config = []
    fk = Humanoid_Batch(cfg, device=device)
    assert fk.num_bodies == 24, f"expected 24 bodies, got {fk.num_bodies}"
    return fk


def _smooth_time(x, kernel=5, sigma=1.0):
    """Gaussian smoothing along the time axis of (1, T, J, 3)."""
    T = x.shape[1]
    if T < 3:
        return x
    half = kernel // 2
    coords = torch.arange(kernel, dtype=x.dtype, device=x.device) - half
    w = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    w = w / w.sum()
    w = w.view(1, 1, kernel)
    B, T, J, C = x.shape
    xf = x.permute(0, 2, 3, 1).reshape(B * J * C, 1, T)
    xf = torch.nn.functional.pad(xf, (half, half), mode="replicate")
    xf = torch.nn.functional.conv1d(xf, w)
    return xf.reshape(B, J, C, T).permute(0, 3, 1, 2).contiguous()


def _estimate_root_rotvecs(positions):
    """
    Estimate pelvis/root orientation from the target joint positions.

    In final_humanoid.xml's upright convention, local +X is forward, +Y is
    left, and +Z is up. Joint positions alone do not constrain twist perfectly,
    so this gives the optimiser a consistent facing direction instead of
    allowing a 180-degree root flip with compensating body rotations.
    """
    pos = np.asarray(positions, dtype=np.float32)
    left = pos[:, L_HIP_IDX] - pos[:, R_HIP_IDX]
    up = pos[:, TORSO_IDX] - pos[:, PELVIS_IDX]

    def normalize(v, fallback):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        out = np.broadcast_to(np.asarray(fallback, dtype=np.float32), v.shape).copy()
        valid = n[..., 0] > 1e-6
        out[valid] = v[valid] / n[valid]
        return out

    y_axis = normalize(left, [0.0, 1.0, 0.0])
    z_axis = normalize(up - (up * y_axis).sum(-1, keepdims=True) * y_axis, [0.0, 0.0, 1.0])
    x_axis = normalize(np.cross(y_axis, z_axis), [1.0, 0.0, 0.0])
    # Re-orthogonalise after the cross product for a proper right-handed frame.
    y_axis = normalize(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])

    mats = np.stack([x_axis, y_axis, z_axis], axis=-1)
    from scipy.spatial.transform import Rotation as _sRot
    return _sRot.from_matrix(mats).as_rotvec().astype(np.float32)


def _fit_sequence_single(positions, fk, device=torch.device("cpu"),
                         iters=500, lr=0.05, w_prior=0.02, w_smooth=2.0,
                         smooth_every=25, verbose=True, w_root_facing=0.05):
    """
    Fit SMPL pose to one contiguous motion chunk.

    positions : (T, 24, 3) world-frame joint positions, MuJoCo order (Z-up).
    fk        : Humanoid_Batch from build_fk().

    """
    T = positions.shape[0]
    positions_np = np.asarray(positions, dtype=np.float32)
    target = torch.from_numpy(positions_np).float().to(device)[None]            # (1,T,24,3)
    root_trans = target[:, :, 0, :].clone()                                     # (1,T,3)
    root_init = torch.from_numpy(_estimate_root_rotvecs(positions_np)).to(device).view(1, T, 1, 3)

    # variables: root orientation (1,T,1,3) + 23 body rotations (1,T,23,3), axis-angle
    root_rot = root_init.clone().detach().requires_grad_(True)
    body_rot = torch.zeros(1, T, 23, 3, device=device, requires_grad=True)
    root_off = torch.zeros(1, 1, 3, device=device, requires_grad=True)

    opt = torch.optim.Adam([root_rot, body_rot, root_off], lr=lr)

    for it in range(iters):
        pose_aa = torch.cat([root_rot, body_rot], dim=2)                        # (1,T,24,3)
        fkret = fk.fk_batch(pose_aa, root_trans + root_off)                     # convert_to_mat
        pred = fkret.global_translation                                         # (1,T,24,3)

        loss_pos = (pred - target).norm(dim=-1).mean()
        loss_prior = body_rot.square().mean()                                   # toward minimal rotation
        loss_root = (root_rot - root_init).square().mean()
        # temporal smoothness on body rotations
        loss_smooth = (body_rot[:, 1:] - body_rot[:, :-1]).square().mean() if T > 1 else 0.0
        loss = loss_pos + w_prior * loss_prior + w_root_facing * loss_root + w_smooth * loss_smooth

        opt.zero_grad()
        loss.backward()
        opt.step()

        # periodic temporal smoothing of the body rotations (like fit_smpl_motion)
        if smooth_every and (it + 1) % smooth_every == 0 and T > 2:
            with torch.no_grad():
                body_rot.data = _smooth_time(body_rot.data, kernel=5, sigma=1.0)

        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f"  iter {it:4d}  pos={loss_pos.item()*100:.2f}cm  "
                  f"smooth={float(loss_smooth)*1e3:.3f}  prior={loss_prior.item():.3f}  "
                  f"root={loss_root.item():.3f}")

    with torch.no_grad():
        pose_aa = torch.cat([root_rot, body_rot], dim=2)
        # return_full=False avoids Humanoid_Batch's CUDA-incompatible velocity calc;
        # we only need positions + global rotations here.
        fkret = fk.fk_batch(pose_aa, root_trans + root_off, return_full=False)
        pred = fkret.global_translation
        err = (pred - target).norm(dim=-1).mean().item()

        pose_aa_np = pose_aa[0].reshape(T, 72).cpu().numpy().astype(np.float32)
        pose_quat_global = fkret.global_rotation[0].cpu().numpy().astype(np.float32)   # xyzw
        # local joint quats: direct axis-angle -> quat (xyzw), matches fk's local_rotation
        from scipy.spatial.transform import Rotation as _sRot
        pose_quat_local = _sRot.from_rotvec(
            pose_aa_np.reshape(T * 24, 3)).as_quat().reshape(T, 24, 4).astype(np.float32)
        root_trans_offset = (root_trans + root_off)[0].cpu().numpy().astype(np.float32)

    return {
        "pose_aa": pose_aa_np,
        "pose_quat_global": pose_quat_global,
        "pose_quat": pose_quat_local,
        "root_trans_offset": root_trans_offset,
        "fit_err_m": err,
    }


def _merge_chunk_results(chunks, T):
    pose_aa = np.empty((T, 72), dtype=np.float32)
    pose_quat_global = np.empty((T, 24, 4), dtype=np.float32)
    pose_quat = np.empty((T, 24, 4), dtype=np.float32)
    root_trans_offset = np.empty((T, 3), dtype=np.float32)
    weighted_err = 0.0
    total = 0

    for chunk in chunks:
        s, e = chunk["keep"]
        cs = s - chunk["start"]
        ce = cs + (e - s)
        fit = chunk["fit"]
        pose_aa[s:e] = fit["pose_aa"][cs:ce]
        pose_quat_global[s:e] = fit["pose_quat_global"][cs:ce]
        pose_quat[s:e] = fit["pose_quat"][cs:ce]
        root_trans_offset[s:e] = fit["root_trans_offset"][cs:ce]
        weighted_err += fit["fit_err_m"] * (e - s)
        total += e - s

    return {
        "pose_aa": pose_aa,
        "pose_quat_global": pose_quat_global,
        "pose_quat": pose_quat,
        "root_trans_offset": root_trans_offset,
        "fit_err_m": weighted_err / max(total, 1),
    }


def fit_sequence(positions, fk, device=torch.device("cpu"),
                 iters=500, lr=0.05, w_prior=0.02, w_smooth=2.0,
                 smooth_every=25, verbose=True, chunk_size=240,
                 chunk_overlap=20, w_root_facing=0.05):
    """
    Fit SMPL pose to a full motion.

    positions : (T, 24, 3) world-frame joint positions, MuJoCo order (Z-up).
    fk        : Humanoid_Batch from build_fk().

    Long sequences are fit in overlapping chunks. A single 50s sequence at 30fps
    creates 1500 independent pose variables; fitting all of them at once can
    leave the sequence boundary in a poor twist/pose local minimum even when the
    later walk looks fine. Chunking keeps the optimisation local and makes the
    first replay frames match the same quality as the rest of the motion.

    Returns dict with:
      pose_aa           (T, 72)   axis-angle, root + 23 joints (+ zeroed hand tail)
      pose_quat_global  (T, 24, 4) xyzw
      pose_quat         (T, 24, 4) xyzw, local
      root_trans_offset (T, 3)
      fit_err_m         scalar mean per-joint position error (m)
    """
    T = positions.shape[0]
    if chunk_size is None or chunk_size <= 0 or T <= chunk_size:
        return _fit_sequence_single(
            positions, fk, device=device, iters=iters, lr=lr,
            w_prior=w_prior, w_smooth=w_smooth, smooth_every=smooth_every,
            verbose=verbose, w_root_facing=w_root_facing,
        )

    chunks = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        fit_s = max(0, s - chunk_overlap)
        fit_e = min(T, e + chunk_overlap)
        if verbose:
            print(f"  chunk {s:4d}:{e:4d}  fit window {fit_s:4d}:{fit_e:4d}")
        fit = _fit_sequence_single(
            positions[fit_s:fit_e], fk, device=device, iters=iters, lr=lr,
            w_prior=w_prior, w_smooth=w_smooth, smooth_every=smooth_every,
            verbose=verbose, w_root_facing=w_root_facing,
        )
        chunks.append({"start": fit_s, "keep": (s, e), "fit": fit})

    return _merge_chunk_results(chunks, T)


class StreamFitter:
    """
    Per-frame warm-started fit for the real-time server.

    Maintains the previous frame's pose as the initialisation, so only a handful
    of optimiser steps are needed per new target frame.

    Real-time guidance (measured): the Humanoid_Batch FK loops over 24 joints in
    Python, so per-iter overhead dominates at batch=1. Use **CPU** for streaming:
      cpu  iters=5  -> ~15 ms/frame   (well within the 33 ms @30fps budget)
      cpu  iters=10 -> ~29 ms/frame
    CUDA is slower here (kernel-launch overhead): ~30 ms/frame at iters=5.
    Warm-starting keeps 5-8 iters sufficient between consecutive frames.

    Usage:
        sf = StreamFitter(fk, torch.device("cpu"), iters=5)
        out = sf.step(pos24)          # pos24: (24,3) world Z-up MuJoCo order
        # out: dict with pose_aa(72,), pose_quat_global(24,4), root_trans(3,)
    """

    def __init__(self, fk, device=torch.device("cpu"), iters=5, lr=0.1,
                 w_prior=0.02, w_temporal=5.0, w_root_facing=0.05):
        self.fk = fk
        self.device = device
        self.iters = iters
        self.lr = lr
        self.w_prior = w_prior
        self.w_temporal = w_temporal
        self.w_root_facing = w_root_facing
        self.root_rot = torch.zeros(1, 1, 1, 3, device=device)
        self.body_rot = torch.zeros(1, 1, 23, 3, device=device)
        self.prev_body_rot = self.body_rot.clone()

    def step(self, pos24):
        target = torch.from_numpy(np.asarray(pos24)).float().to(self.device).view(1, 1, 24, 3)
        root_trans = target[:, :, 0, :].clone()
        root_init = torch.from_numpy(_estimate_root_rotvecs(np.asarray(pos24)[None])).to(self.device).view(1, 1, 1, 3)

        if torch.linalg.norm(self.root_rot).item() < 1e-6:
            root_rot = root_init.clone().requires_grad_(True)
        else:
            root_rot = self.root_rot.clone().requires_grad_(True)
        body_rot = self.body_rot.clone().requires_grad_(True)
        opt = torch.optim.Adam([root_rot, body_rot], lr=self.lr)
        prev = self.prev_body_rot

        for _ in range(self.iters):
            pose_aa = torch.cat([root_rot, body_rot], dim=2)
            pred = self.fk.fk_batch(pose_aa, root_trans).global_translation
            loss = ((pred - target).norm(dim=-1).mean()
                    + self.w_prior * body_rot.square().mean()
                    + self.w_temporal * (body_rot - prev).square().mean()
                    + self.w_root_facing * (root_rot - root_init).square().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            self.root_rot = root_rot.detach()
            self.body_rot = body_rot.detach()
            self.prev_body_rot = self.body_rot.clone()
            pose_aa = torch.cat([self.root_rot, self.body_rot], dim=2)
            fkret = self.fk.fk_batch(pose_aa, root_trans)               # return_full=False
            pose_aa_np = pose_aa[0, 0].reshape(24, 3).cpu().numpy().astype(np.float32)
            from scipy.spatial.transform import Rotation as _sRot
            pose_quat_local = _sRot.from_rotvec(pose_aa_np).as_quat().astype(np.float32)
            return {
                "pose_aa": pose_aa_np.reshape(72),
                "pose_quat_global": fkret.global_rotation[0, 0].cpu().numpy().astype(np.float32),
                "pose_quat": pose_quat_local,
                "root_trans": root_trans[0, 0].cpu().numpy().astype(np.float32),
            }
