"""
Train x15 horizon-flow model.

Condition: K past state frames (K×76D) + task_phase (5D) + desired_heading (2D) + active_goal (3D).
Target:    H future state frames (H×76D).

Smoke test:
    python examples/joint_flow/train_horizon_x15.py \
        --hdf5 artifacts/omomo_filtered/omomo_portable_new_holdout_taskspace.h5 \
        --outdir /tmp/x15_smoke \
        --epochs 1 --batch 64 --hidden 128 --layers 2 \
        --max-clips-per-source 2 --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from animate import BONES
from dataset_horizon_x15 import (
    _J_XY_X,
    _J_XY_Y,
    CONDITION_SCHEMA_LOCO_MANIP,
    CONDITION_SCHEMA_PHASE,
    JointFlowX15Dataset,
    validate_x15_stats_compatible,
)
from flow_matching.path import CondOTProbPath
from model import (
    JointFlowConcatMLP,
    JointFlowCrossTransformer,
    JointFlowFullCrossTransformer,
    JointFlowHistoryTransformer,
    JointFlowMLP,
    JointFlowTransformer,
)
from torch.utils.data import DataLoader, Subset

# Bone pairs in root-relative joint space (pelvis dropped: original idx i → stored at i-1).
# Pelvis (idx 0) is the implicit origin [0,0,0]; bones touching pelvis use a sentinel of -1.
_BONE_PARENT = torch.tensor(
    [p - 1 for (p, c) in BONES], dtype=torch.long
)  # -1 = pelvis
_BONE_CHILD = torch.tensor([c - 1 for (p, c) in BONES], dtype=torch.long)

# Foot joints (L_Ankle, L_Toe, R_Ankle, R_Toe = 3,4,7,8) → pelvis-dropped rel indices.
_FOOT_REL = torch.tensor([2, 3, 6, 7], dtype=torch.long)
_CONTACT_VEL_THRESH = 0.02  # m/frame: GT foot displacement below this = planted
_CONTACT_Z_THRESH = 0.15  # m: GT foot height below this = near floor

ARCHS = {
    "film": JointFlowMLP,
    "concat": JointFlowConcatMLP,
    "transformer": JointFlowTransformer,
    "history_transformer": JointFlowHistoryTransformer,
    "cross_transformer": JointFlowCrossTransformer,
    "full_cross_transformer": JointFlowFullCrossTransformer,
}


def save_checkpoint_atomic(payload: dict, path: Path) -> None:
    """Write a checkpoint without corrupting the previous file on failed saves."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class GpuAugmentSampler:
    """Batched θ-augmentation on the GPU over the precomputed aug base cache.

    Replicates JointFlowX15Dataset._build_sample_augmented as vectorized
    tensor ops: one random yaw θ per sample, applied to root_delta_xy,
    joint XY pairs, orientation quat, desired heading, and active goal.
    Yields normalized (cond, target) batches already on the device.
    """

    def __init__(
        self,
        ds: JointFlowX15Dataset,
        indices: list[int],
        batch_size: int,
        device: torch.device,
        drop_last: bool = False,
        next_map: "torch.Tensor | None" = None,
    ):
        self.frames = torch.from_numpy(ds._aug_frames).to(device)  # (N, KH, 76)
        self.condition_schema = ds.condition_schema
        # phase schema
        self.task_phase = (
            torch.from_numpy(ds._aug_task_phase).to(device)
            if ds._aug_task_phase is not None else None
        )
        self.goal = (
            torch.from_numpy(ds._aug_active_goal_base).to(device)
            if ds._aug_active_goal_base is not None else None
        )
        # loco_manip schema
        self.loco_manip_mode = (
            torch.from_numpy(ds._aug_loco_manip_mode).to(device)
            if ds._aug_loco_manip_mode is not None else None
        )
        self.pickup_goal = (
            torch.from_numpy(ds._aug_pickup_goal_base).to(device)
            if ds._aug_pickup_goal_base is not None else None
        )
        self.place_goal = (
            torch.from_numpy(ds._aug_place_goal_base).to(device)
            if ds._aug_place_goal_base is not None else None
        )
        self.heading = torch.from_numpy(ds._aug_desired_heading_base).to(device)  # (N, 2)
        self.target_speed = torch.from_numpy(ds._aug_target_speed).to(device) if ds._aug_target_speed is not None else None
        self.use_target_speed = ds.use_target_speed
        self.cond_mean = torch.from_numpy(ds.cond_mean).to(device)
        self.cond_std = torch.from_numpy(ds.cond_std).to(device)
        self.target_mean = torch.from_numpy(ds.target_mean).to(device)
        self.target_std = torch.from_numpy(ds.target_std).to(device)
        self.jx = torch.from_numpy(_J_XY_X).to(device)
        self.jy = torch.from_numpy(_J_XY_Y).to(device)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.K = ds.past_frames
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.next_map = next_map.to(device) if next_map is not None else None

    def __len__(self) -> int:
        n = len(self.indices)
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )

    def _augment_batch(self, idx: torch.Tensor, theta: torch.Tensor):
        frames = self.frames[idx].clone()  # (B, KH, 76)
        c = torch.cos(-theta)[:, None]  # rotation by -theta, as in CPU path
        s = torch.sin(-theta)[:, None]

        x, y = frames[..., 0].clone(), frames[..., 1].clone()
        frames[..., 0] = c * x - s * y
        frames[..., 1] = s * x + c * y

        jx = frames[:, :, self.jx].clone()  # (B, KH, 23)
        jy = frames[:, :, self.jy].clone()
        frames[:, :, self.jx] = c[..., None] * jx - s[..., None] * jy
        frames[:, :, self.jy] = s[..., None] * jx + c[..., None] * jy

        # q_new = quat_z(-theta) * q_base  (xyzw), positive-w hemisphere
        ht = theta[:, None] * 0.5
        sz, cz = -torch.sin(ht), torch.cos(ht)
        q = frames[..., 3:7]
        qn = torch.stack(
            [
                cz * q[..., 0] + sz * q[..., 1],
                cz * q[..., 1] - sz * q[..., 0],
                cz * q[..., 2] + sz * q[..., 3],
                cz * q[..., 3] - sz * q[..., 2],
            ],
            dim=-1,
        )
        qn = torch.where(qn[..., 3:4] < 0, -qn, qn)
        frames[..., 3:7] = qn

        B = frames.shape[0]
        past = frames[:, : self.K].reshape(B, -1)
        target = frames[:, self.K :].reshape(B, -1)

        heading = self.heading[idx].clone()
        hx, hy = heading[:, 0].clone(), heading[:, 1].clone()
        heading[:, 0] = c[:, 0] * hx - s[:, 0] * hy
        heading[:, 1] = s[:, 0] * hx + c[:, 0] * hy
        desired_heading = heading

        extras = [self.target_speed[idx]] if self.use_target_speed and self.target_speed is not None else []

        if self.condition_schema == CONDITION_SCHEMA_PHASE:
            goal = self.goal[idx].clone()
            gx, gy = goal[:, 0].clone(), goal[:, 1].clone()
            goal[:, 0] = c[:, 0] * gx - s[:, 0] * gy
            goal[:, 1] = s[:, 0] * gx + c[:, 0] * gy
            task_parts = [self.task_phase[idx], desired_heading, goal]
        else:
            pickup = self.pickup_goal[idx].clone()
            px, py = pickup[:, 0].clone(), pickup[:, 1].clone()
            pickup[:, 0] = c[:, 0] * px - s[:, 0] * py
            pickup[:, 1] = s[:, 0] * px + c[:, 0] * py
            place = self.place_goal[idx].clone()
            lx, ly = place[:, 0].clone(), place[:, 1].clone()
            place[:, 0] = c[:, 0] * lx - s[:, 0] * ly
            place[:, 1] = s[:, 0] * lx + c[:, 0] * ly
            task_parts = [self.loco_manip_mode[idx], desired_heading, pickup, place]

        cond = torch.cat([past] + task_parts + extras, dim=1)
        cond = (cond - self.cond_mean) / self.cond_std
        target = (target - self.target_mean) / self.target_std
        return cond, target

    def __iter__(self):
        perm = self.indices[torch.randperm(len(self.indices), device=self.device)]
        for i in range(0, len(perm), self.batch_size):
            idx = perm[i : i + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            theta = torch.rand(len(idx), device=self.device) * (2 * torch.pi)
            if self.next_map is None:
                yield self._augment_batch(idx, theta)
            else:
                nxt = self.next_map[idx]
                valid = nxt >= 0
                # Paired windows must use the same yaw gauge. Independent theta
                # values put current predictions, next-window conditions, and
                # overlap targets in incompatible coordinate frames.
                cond_nx, target_nx = self._augment_batch(
                    torch.where(valid, nxt, idx), theta
                )
                yield (*self._augment_batch(idx, theta), cond_nx, target_nx, valid)


class GpuCachedSampler:
    """No-augmentation sampler: precomputed normalized samples held on the GPU."""

    def __init__(
        self,
        ds: JointFlowX15Dataset,
        indices: list[int],
        batch_size: int,
        device: torch.device,
        drop_last: bool = False,
        next_map: "torch.Tensor | None" = None,
    ):
        self.cond = torch.from_numpy(ds._cond_cache).to(device)
        self.target = torch.from_numpy(ds._target_cache).to(device)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.next_map = next_map.to(device) if next_map is not None else None

    def __len__(self) -> int:
        n = len(self.indices)
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        perm = self.indices[torch.randperm(len(self.indices), device=self.device)]
        for i in range(0, len(perm), self.batch_size):
            idx = perm[i : i + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            if self.next_map is None:
                yield self.cond[idx], self.target[idx]
            else:
                nxt = self.next_map[idx]
                valid = nxt >= 0
                safe = torch.where(valid, nxt, idx)
                yield self.cond[idx], self.target[idx], self.cond[safe], self.target[
                    safe
                ], valid


def split_clips_for_validation(ds: JointFlowX15Dataset, val_frac: float, seed: int = 0):
    n_clips = ds.num_clips
    if n_clips < 2:
        raise ValueError("Need at least two clips for clip-level train/val split")
    clip_lengths = [ds.clip_len(i) for i in range(n_clips)]
    target_val = max(1, int(sum(clip_lengths) * val_frac))
    perm = torch.randperm(
        n_clips, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    val_clips, val_frames = [], 0
    for ci in perm:
        if val_frames >= target_val and val_clips:
            break
        val_clips.append(ci)
        val_frames += clip_lengths[ci]
        if len(val_clips) == n_clips - 1:
            break
    val_set = set(val_clips)
    train_clips = [i for i in range(n_clips) if i not in val_set]
    if not train_clips:
        train_clips = [val_clips.pop()]
    return train_clips, val_clips


def repeated_train_frame_indices(
    ds: JointFlowX15Dataset, train_clips: list[int], factors: list[int]
) -> list[int]:
    train_set = set(train_clips)
    result = []
    for fi, (ci, _) in enumerate(ds._frame_index):
        if ci in train_set:
            result.extend([fi] * factors[ds.clip_source_id(ci)])
    return result


def validation_frame_indices(
    ds: JointFlowX15Dataset, val_clips: list[int]
) -> list[int]:
    val_set = set(val_clips)
    return [fi for fi, (ci, _) in enumerate(ds._frame_index) if ci in val_set]


def _bone_lengths(joints_rel: torch.Tensor) -> torch.Tensor:
    """Compute bone lengths from root-relative joint positions.

    joints_rel: (..., 23, 3) — pelvis dropped, pelvis position = [0,0,0].
    Returns (..., num_bones).
    """
    bp = _BONE_PARENT.to(joints_rel.device)
    bc = _BONE_CHILD.to(joints_rel.device)
    # Pelvis sentinel (-1) → zero vector (origin)
    zeros = joints_rel.new_zeros(joints_rel.shape[:-2] + (3,))
    parent_pts = torch.where(
        (bp < 0).reshape(*([1] * (joints_rel.dim() - 2)), -1, 1),
        zeros.unsqueeze(-2).expand_as(joints_rel[..., : len(bp), :]),
        joints_rel[..., bp.clamp(min=0), :],
    )
    child_pts = joints_rel[..., bc, :]
    return (child_pts - parent_pts).norm(dim=-1)  # (..., num_bones)


def foot_contact_loss(
    pred_x1: torch.Tensor,
    true_x1: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    horizon: int,
    frame_dim: int,
) -> torch.Tensor:
    """MDM-style contact loss: penalize predicted foot velocity on frames where
    the GT foot is planted (low and not moving). Computed in raw units, with the
    contact mask derived on the fly from the GT horizon frames.

    Frame layout: root_delta_xy + root_z(0:3) + quat(3:7) + joint_rel(7:).
    Foot position in the anchor frame = root(0:3) + joint_rel[foot].
    """
    B = pred_x1.shape[0]
    pred = pred_x1.reshape(B, horizon, frame_dim) * target_std + target_mean
    true = true_x1.reshape(B, horizon, frame_dim) * target_std + target_mean

    foot = _FOOT_REL.to(pred_x1.device)

    def foot_pos(frames: torch.Tensor) -> torch.Tensor:
        jrel = frames[:, :, 7:].reshape(B, horizon, 23, 3)
        return frames[:, :, None, 0:3] + jrel[:, :, foot, :]  # (B, H, 4, 3)

    pos_pred = foot_pos(pred)
    pos_true = foot_pos(true)
    vel_pred = pos_pred[:, 1:] - pos_pred[:, :-1]  # (B, H-1, 4, 3)
    vel_true = pos_true[:, 1:] - pos_true[:, :-1]
    contact = (vel_true.norm(dim=-1) < _CONTACT_VEL_THRESH) & (
        pos_true[:, :-1, :, 2] < _CONTACT_Z_THRESH
    )  # (B, H-1, 4)
    n = contact.sum()
    if n == 0:
        return pred_x1.new_tensor(0.0)
    return (vel_pred.pow(2).sum(dim=-1) * contact).sum() / (n * 3)


def horizon_aux_loss(
    pred_x1: torch.Tensor,
    true_x1: torch.Tensor,
    horizon: int,
    frame_dim: int,
    root_weight: float,
    heading_weight: float,
    joint_weight: float,
    smooth_weight: float,
    bone_weight: float = 0.0,
) -> torch.Tensor:
    # Frame layout: root_delta_xy + root_z(0:3) + quat_xyzw(3:7) + joint_rel(7:)
    pred = pred_x1.reshape(pred_x1.shape[0], horizon, frame_dim)
    true = true_x1.reshape(true_x1.shape[0], horizon, frame_dim)
    mse = nn.functional.mse_loss
    loss = pred_x1.new_tensor(0.0)
    if root_weight > 0.0:
        loss = loss + root_weight * mse(pred[:, :, 0:3], true[:, :, 0:3])
    if heading_weight > 0.0:
        loss = loss + heading_weight * mse(pred[:, :, 3:7], true[:, :, 3:7])
    if joint_weight > 0.0:
        loss = loss + joint_weight * mse(pred[:, :, 7:], true[:, :, 7:])
    if smooth_weight > 0.0 and horizon > 1:
        loss = loss + smooth_weight * mse(
            pred[:, 1:] - pred[:, :-1], true[:, 1:] - true[:, :-1]
        )
    if bone_weight > 0.0:
        pred_j = pred[:, :, 7:].reshape(pred.shape[0], horizon, 23, 3)
        true_j = true[:, :, 7:].reshape(true.shape[0], horizon, 23, 3)
        loss = loss + bone_weight * mse(_bone_lengths(pred_j), _bone_lengths(true_j))
    return loss


def _quat_to_zyx(q: torch.Tensor):
    """xyzw quat (built as ZYX euler of relative angles) → (yaw, pitch, roll)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return yaw, pitch, roll


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def reanchor_to_index(f: torch.Tensor, anchor_index: int) -> torch.Tensor:
    """Re-canonicalize raw-unit state frames (B, T, 76) to one frame in the sequence.

    Mirrors the inference-time window update (_state_frame with the newest frame
    as anchor): root_delta_xy and joint XY rotate by -yaw of the anchor; the
    quat holds component-wise relative ZYX eulers, so re-anchoring is euler
    subtraction; root_z is absolute and unchanged.
    """
    if anchor_index < 0:
        anchor_index = f.shape[1] + anchor_index
    if anchor_index < 0 or anchor_index >= f.shape[1]:
        raise ValueError(
            f"anchor_index {anchor_index} out of range for {f.shape[1]} frames"
        )
    jx_idx = torch.from_numpy(_J_XY_X).to(f.device)
    jy_idx = torch.from_numpy(_J_XY_Y).to(f.device)

    yaw_f, pitch_f, roll_f = _quat_to_zyx(f[..., 3:7])  # (B, K)
    yaw_a = yaw_f[:, anchor_index]
    c = torch.cos(-yaw_a)[:, None]
    s = torch.sin(-yaw_a)[:, None]
    out = f.clone()

    dx = f[..., 0] - f[:, anchor_index : anchor_index + 1, 0]
    dy = f[..., 1] - f[:, anchor_index : anchor_index + 1, 1]
    out[..., 0] = c * dx - s * dy
    out[..., 1] = s * dx + c * dy

    jx = f[:, :, jx_idx]
    jy = f[:, :, jy_idx]
    out[:, :, jx_idx] = c[..., None] * jx - s[..., None] * jy
    out[:, :, jy_idx] = s[..., None] * jx + c[..., None] * jy

    hy = _wrap(yaw_f - yaw_f[:, anchor_index : anchor_index + 1]) * 0.5
    hp = _wrap(pitch_f - pitch_f[:, anchor_index : anchor_index + 1]) * 0.5
    hr = _wrap(roll_f - roll_f[:, anchor_index : anchor_index + 1]) * 0.5
    cy, sy = torch.cos(hy), torch.sin(hy)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cr, sr = torch.cos(hr), torch.sin(hr)
    qn = torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )  # xyzw, ZYX(yaw_rel, pitch_rel, roll_rel)
    qn = torch.where(qn[..., 3:4] < 0, -qn, qn)
    out[..., 3:7] = qn
    return out


def reanchor_to_last(f: torch.Tensor) -> torch.Tensor:
    """Re-canonicalize raw-unit state frames (B, K, 76) to their own last frame."""
    return reanchor_to_index(f, -1)


def init_wandb(args, stats: dict, n_params: int):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it or rerun without --wandb."
        ) from exc

    config = dict(vars(args))
    config.update(
        {
            "cond_dim": len(stats["cond_mean"]),
            "target_dim": len(stats["target_mean"]),
            "n_params": n_params,
            "representation": stats.get("representation"),
        }
    )
    if args.wandb_mode:
        wandb_settings = {"mode": args.wandb_mode}
    else:
        wandb_settings = {}
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config=config,
        dir=str(args.outdir),
        **wandb_settings,
    )


def log_wandb_artifact(run, path: Path, name: str, artifact_type: str) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, nargs="+")
    p.add_argument("--oversample", type=int, nargs="+", default=None)
    p.add_argument("--outdir", default="checkpoints/joint_flow_x15_horizon")
    p.add_argument("--stats-in", default=None, help="Reuse precomputed x15 stats.json")
    p.add_argument("--loss-log", default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many microbatches before optimizer.step(). "
        "Effective batch size is --batch * --grad-accum-steps.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument(
        "--arch",
        choices=sorted(ARCHS),
        default="concat",
        help="concat: residual MLP; transformer: DiT-style horizon-token model "
        "(x15, past frames flattened into FiLM cond); history_transformer: "
        "x16, past frames attended over as prepended tokens; cross_transformer: "
        "x16c, past frames cross-attended as read-only memory; "
        "full_cross_transformer: x16f, past frames AND each task group "
        "cross-attended as memory tokens (adaLN = timestep only); film: legacy FiLM",
    )
    p.add_argument(
        "--heads", type=int, default=8, help="Attention heads for --arch transformer"
    )
    p.add_argument(
        "--frame-dim",
        type=int,
        default=76,
        help="Per-frame target dimension for --arch transformer",
    )
    p.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
        help="Transformer feed-forward expansion ratio",
    )
    p.add_argument(
        "--dropout", type=float, default=0.0, help="Transformer attention/MLP dropout"
    )
    p.add_argument("--root-loss-weight", type=float, default=2.0)
    p.add_argument("--heading-loss-weight", type=float, default=1.0)
    p.add_argument("--joint-loss-weight", type=float, default=0.25)
    p.add_argument("--smooth-loss-weight", type=float, default=0.1)
    p.add_argument(
        "--bone-loss-weight",
        type=float,
        default=0.0,
        help="Weight for bone-length consistency loss (penalizes predicted bone "
        "lengths deviating from GT). Fixes varying link-length artifacts.",
    )
    p.add_argument(
        "--contact-loss-weight",
        type=float,
        default=0.0,
        help="MDM-style foot-contact loss: penalize predicted foot velocity on "
        "frames where the GT foot is planted. Fixes foot sliding/gliding.",
    )
    p.add_argument(
        "--past-dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing the past-frames block of the condition per "
        "sample (PARC-style). Forces the model to follow task commands "
        "instead of overfitting to past frames; enables guidance blending "
        "and native cold starts at inference.",
    )
    p.add_argument("--past-frames", type=int, default=2)
    p.add_argument(
        "--condition-schema",
        choices=[CONDITION_SCHEMA_PHASE, CONDITION_SCHEMA_LOCO_MANIP],
        default=CONDITION_SCHEMA_PHASE,
        help="Conditioning schema: 'phase' (5D one-hot + heading + active_goal) or "
        "'loco_manip' ([stand,loco,loco_manip] + heading + pickup_goal + place_goal). "
        "loco_manip removes pickup/carry/place phase distinctions; the model infers manipulation "
        "state from past frames and persistent goals.",
    )
    p.add_argument("--use-target-speed", action="store_true",
                   help="Add GT mean root speed over the horizon as a conditioning scalar (locomotion frames only).")
    p.add_argument(
        "--slow-loco-mode",
        choices=["relabel", "drop", "keep"],
        default="relabel",
        help="How to handle locomotion windows whose horizon speed is below the stand threshold. "
        "relabel preserves current behavior; drop removes them from training; keep leaves them as locomotion.",
    )
    p.add_argument(
        "--motion-fps",
        type=float,
        default=None,
        help="Generation/reference FPS. Used with --policy-hz to derive the "
        "past-frame stride for policy-rate observations.",
    )
    p.add_argument(
        "--policy-hz",
        type=float,
        default=None,
        help="Policy/control observation frequency. With --motion-fps 120 and "
        "--policy-hz 30, past condition frames are spaced by 4 motion frames.",
    )
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Deprecated/no-op. Training only updates best.pt.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Train without yaw augmentation (stats and samples from raw clips)",
    )
    p.add_argument(
        "--mirror",
        action="store_true",
        help="Double dataset by adding left-right mirrored clips (L/R joint swap + negate lateral Y)",
    )
    p.add_argument(
        "--mirror-sources",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Mirror only these HDF5 source indices (0-based). Overrides --mirror.",
    )
    p.add_argument(
        "--cold-start-pad",
        type=int,
        default=0,
        help="Prepend this many copies of frame 0 to every clip so windows near "
        "the start form consistent rest→motion pairs (replaces the old "
        "--cold-start-prob substitution, which paired rest pasts with "
        "arbitrary mid-clip targets and caused gliding).",
    )
    p.add_argument(
        "--filter-loco-hand-speed",
        type=float,
        default=0.0,
        help="Drop non-object locomotion samples whose target horizon overlaps "
        "pelvis-relative hand speed above this m/s. 0 disables this criterion.",
    )
    p.add_argument(
        "--filter-loco-hand-height",
        type=float,
        default=0.0,
        help="Drop non-object locomotion samples whose target horizon overlaps "
        "a hand higher than this pelvis-relative Z height in meters. "
        "0 disables this criterion.",
    )
    p.add_argument(
        "--filter-loco-hand-window",
        type=int,
        default=1,
        help="Require this many sustained bad hand-motion frames before "
        "--filter-loco-hand-speed/height removes samples.",
    )
    p.add_argument(
        "--past-dropout-warmup-epochs",
        type=int,
        default=0,
        help="Keep past dropout disabled for this many epochs before ramping.",
    )
    p.add_argument(
        "--past-dropout-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp past dropout from 0 to --past-dropout over this many epochs after warmup. 0 keeps legacy constant dropout.",
    )
    p.add_argument(
        "--past-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std (normalized space) added to the past-frames "
        "block of the condition during training. Simulates AR rollout / "
        "robot state-estimation error to reduce mean-reversion stalling.",
    )
    p.add_argument(
        "--past-noise-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp past-frame noise from 0 to --past-noise-std over "
        "this many epochs. 0 keeps the legacy constant-noise behavior.",
    )
    p.add_argument(
        "--rollout-prob",
        type=float,
        default=0.0,
        help="Stride-matched rollout consistency probability at full ramp "
        "(0=off). For selected samples, predict window A, build window "
        "B at t+rollout_stride from A[stride-K:stride], and train "
        "low-weight B/overlap/splice losses.",
    )
    p.add_argument(
        "--rollout-stride",
        type=int,
        default=5,
        help="Number of predicted frames consumed before replanning in the "
        "rollout-consistency branch. Must satisfy K <= stride < H.",
    )
    p.add_argument(
        "--rollout-ramp-epochs",
        type=int,
        default=500,
        help="Linearly ramp rollout-prob from 0 to --rollout-prob over this "
        "many epochs after --rollout-warmup-epochs.",
    )
    p.add_argument(
        "--rollout-warmup-epochs",
        type=int,
        default=300,
        help="Keep rollout consistency off for this many epochs.",
    )
    p.add_argument(
        "--rollout-gt-weight",
        type=float,
        default=0.25,
        help="Weight for the stride-matched B-window GT loss.",
    )
    p.add_argument(
        "--overlap-loss-weight",
        type=float,
        default=0.5,
        help="Weight for A[stride:H] vs B[0:H-stride] overlap consistency.",
    )
    p.add_argument(
        "--splice-vel-weight",
        type=float,
        default=0.1,
        help="Weight for the first B frame matching the GT one-step splice "
        "from the consumed A segment.",
    )
    p.add_argument("--no-precompute", action="store_true")
    p.add_argument("--drop-last", action="store_true")
    p.add_argument("--max-clips-per-source", type=int, default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (faster epochs; ~1 min one-time compile)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to reduce activation memory",
    )
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument("--wandb-project", default="joint-flow")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional wandb mode override",
    )
    p.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Upload best checkpoint, stats.json, and loss_log.txt to wandb",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def derive_past_stride(motion_fps, policy_hz) -> int:
    if motion_fps is None and policy_hz is None:
        return 1
    if motion_fps is None or policy_hz is None:
        raise ValueError("--motion-fps and --policy-hz must be passed together")
    if motion_fps <= 0 or policy_hz <= 0:
        raise ValueError("--motion-fps and --policy-hz must be positive")
    ratio = motion_fps / policy_hz
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(
            f"--motion-fps / --policy-hz must be an integer frame stride; "
            f"got {motion_fps:g} / {policy_hz:g} = {ratio:g}"
        )
    return stride


def main():
    args = parse_args()
    args.past_stride = derive_past_stride(args.motion_fps, args.policy_hz)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
    device = torch.device(args.device)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    factors = args.oversample or [1] * len(args.hdf5)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.past_dropout_warmup_epochs < 0:
        raise ValueError("--past-dropout-warmup-epochs must be >= 0")
    if args.past_dropout_ramp_epochs < 0:
        raise ValueError("--past-dropout-ramp-epochs must be >= 0")
    if args.past_noise_ramp_epochs < 0:
        raise ValueError("--past-noise-ramp-epochs must be >= 0")
    if args.filter_loco_hand_speed < 0.0:
        raise ValueError("--filter-loco-hand-speed must be >= 0")
    if args.filter_loco_hand_height < 0.0:
        raise ValueError("--filter-loco-hand-height must be >= 0")
    if args.filter_loco_hand_window < 1:
        raise ValueError("--filter-loco-hand-window must be >= 1")
    if len(factors) != len(args.hdf5):
        raise ValueError(
            f"--oversample has {len(factors)} values but --hdf5 has {len(args.hdf5)}"
        )
    if args.rollout_prob > 0.0:
        if args.past_stride != 1:
            raise ValueError(
                "--rollout-prob with policy-rate past frames is not supported yet; "
                "the rollout branch assumes consecutive motion-frame history"
            )
        if args.rollout_stride < args.past_frames:
            raise ValueError("--rollout-stride must be >= --past-frames")
        if args.rollout_stride >= args.horizon:
            raise ValueError(
                "--rollout-stride must be < --horizon so overlap frames exist"
            )

    loss_log_path = Path(args.loss_log) if args.loss_log else out / "loss_log.txt"
    loss_log_f = open(loss_log_path, "w")
    loss_log_f.write(
        "epoch\ttrain_loss\ttrain_flow\ttrain_aux\tval_loss\tval_flow\tval_aux"
        "\ttrain_rollout\ttrain_overlap\ttrain_splice\n"
    )
    loss_log_f.flush()

    print(
        f"Loading x15 dataset (past={args.past_frames}, "
        f"past_stride={args.past_stride}, horizon={args.horizon})"
    )
    if args.motion_fps is not None:
        print(
            f"  motion_fps={args.motion_fps:g} policy_hz={args.policy_hz:g} "
            f"-> past_stride={args.past_stride}"
        )
    for path, factor in zip(args.hdf5, factors):
        print(f"  x{factor}  {path}")

    existing_stats = out / "stats.json"
    if args.stats_in:
        stats = JointFlowX15Dataset.load_stats(args.stats_in)
        print(f"  stats loaded from --stats-in: {args.stats_in}")
    elif existing_stats.exists():
        stats = JointFlowX15Dataset.load_stats(str(existing_stats))
        print(f"  stats loaded from existing {existing_stats}")
    else:
        # Stats are computed from the train dataset below so that
        # normalization matches the training distribution (augmented or not).
        stats = None
        src = "raw" if args.no_augment else "augmented"
        print(f"  no existing stats.json found — computing from {src} data")

    if stats is not None:
        validate_x15_stats_compatible(stats, require_current=True)

    # Resolve per-source mirror flags
    n_sources = len(args.hdf5) if isinstance(args.hdf5, list) else 1
    if args.mirror_sources is not None:
        mirror_flags = [i in args.mirror_sources for i in range(n_sources)]
    else:
        mirror_flags = args.mirror  # bool — applied uniformly

    # Train dataset; also computes stats and the clip-level split
    ds_train = JointFlowX15Dataset(
        args.hdf5,
        past_frames=args.past_frames,
        past_stride=args.past_stride,
        horizon=args.horizon,
        stats=stats,
        max_clips_per_source=args.max_clips_per_source,
        precompute_samples=args.no_augment,
        augment=not args.no_augment,
        mirror=mirror_flags,
        cold_start_pad=args.cold_start_pad,
        filter_loco_hand_speed=args.filter_loco_hand_speed,
        filter_loco_hand_height=args.filter_loco_hand_height,
        filter_loco_hand_window=args.filter_loco_hand_window,
        use_target_speed=args.use_target_speed,
        condition_schema=args.condition_schema,
        slow_loco_mode=args.slow_loco_mode,
    )
    ds_train.stats["train_args"] = vars(args)
    ds_train.save_stats(str(out / "stats.json"))
    stats = ds_train.stats
    print(
        f"  {len(ds_train):,} samples | clips={ds_train.num_clips:,} | cond_dim={ds_train.cond_dim} target_dim={ds_train.target_dim}"
    )
    slow_loco_stats = ds_train.slow_loco_stats
    print(
        "  slow loco: "
        f"mode={slow_loco_stats['mode']} "
        f"threshold={slow_loco_stats['speed_threshold']:.3g} "
        f"relabelled={slow_loco_stats['relabelled_frames']:,} "
        f"dropped={slow_loco_stats['dropped_frames']:,}"
    )
    filter_stats = ds_train.filter_loco_hand_stats
    if filter_stats["enabled"]:
        print(
            "  loco hand filter: "
            f"dropped={filter_stats['dropped_frames']:,} "
            f"kept={filter_stats['kept_frames']:,} "
            f"speed>{filter_stats['speed_threshold']:.3g} "
            f"height>{filter_stats['height_threshold']:.3g} "
            f"window={filter_stats['window']}"
        )

    train_clips, val_clips = split_clips_for_validation(ds_train, args.val_frac, seed=0)
    print(f"  split: train_clips={len(train_clips)} val_clips={len(val_clips)}")

    # Val dataset: no augmentation, optionally precomputed
    ds_val = JointFlowX15Dataset(
        args.hdf5,
        past_frames=args.past_frames,
        past_stride=args.past_stride,
        horizon=args.horizon,
        stats=stats,
        max_clips_per_source=args.max_clips_per_source,
        precompute_samples=not args.no_precompute,
        augment=False,
        mirror=mirror_flags,
        cold_start_pad=args.cold_start_pad,
        filter_loco_hand_speed=args.filter_loco_hand_speed,
        filter_loco_hand_height=args.filter_loco_hand_height,
        filter_loco_hand_window=args.filter_loco_hand_window,
        use_target_speed=args.use_target_speed,
        condition_schema=args.condition_schema,
        slow_loco_mode=args.slow_loco_mode,
    )

    train_indices = repeated_train_frame_indices(ds_train, train_clips, factors)
    val_indices = validation_frame_indices(ds_train, val_clips)
    print(f"  train_samples={len(train_indices):,} val_samples={len(val_indices):,}")

    if ds_train._aug_frames is not None:
        cache_gb = ds_train._aug_frames.nbytes / (1024**3)
        print(f"  train aug-base cache: {cache_gb:.2f} GiB")
    elif ds_train._cond_cache is not None:
        cache_gb = (ds_train._cond_cache.nbytes + ds_train._target_cache.nbytes) / (
            1024**3
        )
        print(f"  train cache: {cache_gb:.2f} GiB")

    sampler_cls = GpuCachedSampler if args.no_augment else GpuAugmentSampler
    paired_step = args.rollout_stride if args.rollout_prob > 0.0 else None
    next_map = (
        torch.from_numpy(ds_train.next_window_indices(paired_step))
        if paired_step is not None
        else None
    )
    train_loader = sampler_cls(
        ds_train,
        train_indices,
        batch_size=args.batch,
        device=device,
        drop_last=args.drop_last,
        next_map=next_map,
    )
    val_loader = DataLoader(
        Subset(ds_val, val_indices),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"  microbatch={args.batch:,} grad_accum={args.grad_accum_steps} "
        f"effective_batch={args.batch * args.grad_accum_steps:,}"
    )

    model_kwargs = {
        "joint_dim": ds_train.target_dim,
        "cond_dim": ds_train.cond_dim,
        "hidden_size": args.hidden,
        "n_layers": args.layers,
    }
    if args.arch in (
        "transformer",
        "history_transformer",
        "cross_transformer",
        "full_cross_transformer",
    ):
        model_kwargs.update(
            {
                "frame_dim": args.frame_dim,
                "n_heads": args.heads,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
            }
        )
        if args.arch in (
            "history_transformer",
            "cross_transformer",
            "full_cross_transformer",
        ):
            model_kwargs["past_frames"] = args.past_frames
        if args.arch == "full_cross_transformer":
            if args.condition_schema == CONDITION_SCHEMA_LOCO_MANIP:
                # mode(3) + heading(2) + pickup_goal(3) + place_goal(3) [+ speed(1)]
                base = (3, 2, 3, 3)
            else:
                # phase(5) + heading(2) + goal(3) [+ speed(1)]
                base = (5, 2, 3)
            model_kwargs["task_groups"] = base + ((1,) if ds_train.use_target_speed else ())
    model = ARCHS[args.arch](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  arch={args.arch} params={n_params:,}")
    if args.compile:
        model = torch.compile(model)

    amp_enabled = args.amp and device.type == "cuda"
    if args.amp and not amp_enabled:
        print("  --amp requested but CUDA is not active; running in fp32")
    print(f"  amp={'on' if amp_enabled else 'off'}")
    wandb_run = init_wandb(args, stats, n_params)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    prob_path = CondOTProbPath()
    mse = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    K = args.past_frames
    fdim = ds_train.target_frame_dim

    # Stats for replacing the next window's normalized past block with
    # stride-matched predicted history.
    past_cond_mean = torch.tensor(
        ds_train.cond_mean[: K * fdim], dtype=torch.float32, device=device
    ).reshape(1, K, fdim)
    past_cond_std = torch.tensor(
        ds_train.cond_std[: K * fdim], dtype=torch.float32, device=device
    ).reshape(1, K, fdim)

    # Full per-dim target stats reshaped per frame, for raw-space contact loss.
    tgt_mean = torch.tensor(
        stats["target_mean"], dtype=torch.float32, device=device
    ).reshape(1, args.horizon, fdim)
    tgt_std = torch.tensor(
        stats["target_std"], dtype=torch.float32, device=device
    ).reshape(1, args.horizon, fdim)

    def denorm_target(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], args.horizon, fdim) * tgt_std + tgt_mean

    def pose_frame_slice(x: torch.Tensor) -> torch.Tensor:
        """Root position/orientation plus body joints; excludes no dimensions in x15."""
        return x

    def aux_total(x1_hat: torch.Tensor, x1_true: torch.Tensor) -> torch.Tensor:
        loss = horizon_aux_loss(
            x1_hat,
            x1_true,
            args.horizon,
            ds_train.target_frame_dim,
            args.root_loss_weight,
            args.heading_loss_weight,
            args.joint_loss_weight,
            args.smooth_loss_weight,
            args.bone_loss_weight,
        )
        if args.contact_loss_weight > 0.0:
            loss = loss + args.contact_loss_weight * foot_contact_loss(
                x1_hat,
                x1_true,
                tgt_mean,
                tgt_std,
                args.horizon,
                fdim,
            )
        return loss

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = train_flow = train_aux = train_n = 0
        train_rollout = train_rollout_n = 0
        train_overlap = train_overlap_n = 0
        train_splice = train_splice_n = 0

        if args.rollout_prob > 0 and epoch > args.rollout_warmup_epochs:
            ramp = (epoch - args.rollout_warmup_epochs) / max(
                1, args.rollout_ramp_epochs
            )
            rollout_p = min(args.rollout_prob, args.rollout_prob * ramp)
        else:
            rollout_p = 0.0
        if args.past_noise_ramp_epochs > 0:
            past_noise_std = args.past_noise_std * min(
                1.0, epoch / args.past_noise_ramp_epochs
            )
        else:
            past_noise_std = args.past_noise_std
        if epoch <= args.past_dropout_warmup_epochs:
            past_dropout = 0.0
        elif args.past_dropout_ramp_epochs > 0:
            ramp_epoch = epoch - args.past_dropout_warmup_epochs
            past_dropout = args.past_dropout * min(
                1.0, ramp_epoch / args.past_dropout_ramp_epochs
            )
        else:
            past_dropout = args.past_dropout

        optimizer.zero_grad(set_to_none=True)
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            if len(batch) == 5:
                cond, x1, cond_nx, x1_nx, rollout_valid = batch
            else:
                cond, x1 = batch
                cond_nx = x1_nx = rollout_valid = None
            cond_clean = cond
            if past_noise_std > 0.0:
                cond = cond.clone()
                cond[:, : K * fdim] += past_noise_std * torch.randn(
                    len(cond), K * fdim, device=cond.device
                )
            if past_dropout > 0.0:
                cond = cond.clone()
                drop = torch.rand(len(cond), device=cond.device) < past_dropout
                cond[drop, : K * fdim] = 0.0

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], device=device)
                ps = prob_path.sample(x_0=x0, x_1=x1, t=t)
                pred_v = model(ps.x_t, ps.t, cond)
                flow_loss = mse(pred_v, ps.dx_t)
                x1_hat = ps.x_t + (1.0 - ps.t[:, None]) * pred_v
                aux_loss = aux_total(x1_hat, x1)
                loss = flow_loss + aux_loss

                if rollout_p > 0.0 and rollout_valid is not None:
                    rollout_mask = (
                        torch.rand(len(cond), device=device) < rollout_p
                    ) & rollout_valid
                    n_roll = int(rollout_mask.sum())
                    if n_roll > 0:
                        stride = args.rollout_stride
                        overlap_len = args.horizon - stride
                        hist_start = stride - K
                        with torch.no_grad():
                            # Approximate inference by sampling a clean A horizon from
                            # pure noise, then using the last K consumed frames as B's
                            # predicted history.
                            x0_a = torch.randn(n_roll, x1.shape[1], device=device)
                            v_a = model(
                                x0_a,
                                torch.zeros(n_roll, device=device),
                                cond_clean[rollout_mask],
                            )
                            a_est = x0_a + v_a
                            a_raw = denorm_target(a_est)

                            hist_raw = a_raw[:, hist_start:stride]
                            hist_raw = reanchor_to_last(hist_raw)
                            hist_norm = (hist_raw - past_cond_mean) / past_cond_std

                            cond_roll = cond_nx[rollout_mask].clone()
                            cond_roll[:, : K * fdim] = hist_norm.reshape(n_roll, -1)

                            # Re-express A's unused future in B's predicted anchor
                            # (A[stride-1]) before comparing with B[0:overlap_len].
                            a_common = reanchor_to_index(a_raw[:, hist_start:], K - 1)
                            a_overlap = a_common[:, K : K + overlap_len]

                        x0_roll = torch.randn(n_roll, x1.shape[1], device=device)
                        t_roll = torch.rand(n_roll, device=device)
                        x1_roll = x1_nx[rollout_mask]
                        ps_roll = prob_path.sample(x_0=x0_roll, x_1=x1_roll, t=t_roll)
                        pred_v_roll = model(ps_roll.x_t, ps_roll.t, cond_roll)
                        flow_roll = mse(pred_v_roll, ps_roll.dx_t)
                        x1_hat_roll = (
                            ps_roll.x_t + (1.0 - ps_roll.t[:, None]) * pred_v_roll
                        )
                        aux_roll = aux_total(x1_hat_roll, x1_roll)
                        rollout_gt = flow_roll + aux_roll

                        b_raw = denorm_target(x1_hat_roll)
                        true_b_raw = denorm_target(x1_roll)
                        overlap_loss = mse(
                            pose_frame_slice(b_raw[:, :overlap_len]),
                            pose_frame_slice(a_overlap),
                        )
                        splice_loss = mse(
                            pose_frame_slice(b_raw[:, 0]),
                            pose_frame_slice(true_b_raw[:, 0]),
                        )
                        rollout_loss = (
                            args.rollout_gt_weight * rollout_gt
                            + args.overlap_loss_weight * overlap_loss
                            + args.splice_vel_weight * splice_loss
                        )
                        loss = loss + (n_roll / len(cond)) * rollout_loss
                        train_rollout += rollout_gt.item() * n_roll
                        train_rollout_n += n_roll
                        train_overlap += overlap_loss.item() * n_roll
                        train_overlap_n += n_roll
                        train_splice += splice_loss.item() * n_roll
                        train_splice_n += n_roll

            scaler.scale(loss / args.grad_accum_steps).backward()
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == n_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            n = len(cond)
            train_loss += (flow_loss.item() + aux_loss.item()) * n
            train_flow += flow_loss.item() * n
            train_aux += aux_loss.item() * n
            train_n += n

        if train_n == 0:
            raise ValueError(
                "No training batches. Reduce --batch or disable --drop-last."
            )
        scheduler.step()
        train_loss /= train_n
        train_flow /= train_n
        train_aux /= train_n
        train_rollout = train_rollout / train_rollout_n if train_rollout_n > 0 else 0.0
        train_overlap = train_overlap / train_overlap_n if train_overlap_n > 0 else 0.0
        train_splice = train_splice / train_splice_n if train_splice_n > 0 else 0.0

        model.eval()
        val_loss = val_flow = val_aux = val_n = 0
        with torch.no_grad():
            for cond, x1 in val_loader:
                cond, x1 = cond.to(device), x1.to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    x0 = torch.randn_like(x1)
                    t = torch.rand(x1.shape[0], device=device)
                    ps = prob_path.sample(x_0=x0, x_1=x1, t=t)
                    pred_v = model(ps.x_t, ps.t, cond)
                    fl = mse(pred_v, ps.dx_t)
                    x1_hat_val = ps.x_t + (1.0 - ps.t[:, None]) * pred_v
                    al = aux_total(x1_hat_val, x1)
                n = len(cond)
                val_loss += (fl + al).item() * n
                val_flow += fl.item() * n
                val_aux += al.item() * n
                val_n += n
        val_loss /= max(1, val_n)
        val_flow /= max(1, val_n)
        val_aux /= max(1, val_n)

        # Unwrap torch.compile so checkpoint keys stay loadable by inference scripts.
        save_sd = getattr(model, "_orig_mod", model).state_dict()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model": save_sd,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                out / "best.pt",
            )

        loss_log_f.write(
            f"{epoch}\t{train_loss:.6f}\t{train_flow:.6f}\t{train_aux:.6f}"
            f"\t{val_loss:.6f}\t{val_flow:.6f}\t{val_aux:.6f}"
            f"\t{train_rollout:.6f}\t{train_overlap:.6f}\t{train_splice:.6f}\n"
        )
        loss_log_f.flush()
        epoch_sec = time.time() - t0
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/flow": train_flow,
                    "train/aux": train_aux,
                    "train/rollout_gt": train_rollout,
                    "train/overlap": train_overlap,
                    "train/splice": train_splice,
                    "train/rollout_prob": rollout_p,
                    "val/loss": val_loss,
                    "val/flow": val_flow,
                    "val/aux": val_aux,
                    "val/best_loss": best_val,
                    "optim/lr": scheduler.get_last_lr()[0],
                    "time/epoch_sec": epoch_sec,
                },
                step=epoch,
            )
        rollout_str = (
            f" rollout={train_rollout:.4f}/{train_overlap:.4f}/{train_splice:.4f}@p{rollout_p:.2f}"
            if train_rollout_n > 0
            else ""
        )
        print(
            f"epoch {epoch:4d}/{args.epochs}  train={train_loss:.4f} "
            f"(flow={train_flow:.4f} aux={train_aux:.4f}){rollout_str}  "
            f"val={val_loss:.4f} (flow={val_flow:.4f} aux={val_aux:.4f})  "
            f"best={best_val:.4f}  ({epoch_sec:.1f}s)",
            flush=True,
        )

    loss_log_f.close()
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        if args.wandb_log_artifacts:
            log_wandb_artifact(
                wandb_run, out / "stats.json", f"{out.name}-stats", "stats"
            )
            log_wandb_artifact(
                wandb_run, out / "loss_log.txt", f"{out.name}-loss-log", "log"
            )
            best_path = out / "best.pt"
            if best_path.exists():
                log_wandb_artifact(
                    wandb_run, best_path, f"{out.name}-best", "checkpoint"
                )
        wandb_run.finish()
    print(f"Done. best_val={best_val:.5f} -> {out}/")


if __name__ == "__main__":
    main()
