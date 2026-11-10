"""
X15 horizon dataset — standalone, no X14 dependency.

Condition = K past state frames (K×76D) + task_phase (5D) + desired_heading (2D) + active_goal (3D).
Target    = H future state frames (H×76D).

State frame (76D) = root_delta_xy (anchor frame) + root_z (absolute, world)
+ orientation quat xyzw (anchor-relative) + 23 root-relative joints.
desired_heading is the travel direction over the horizon: the unit vector
from anchor_xy to root_xy[t+H-1], expressed in anchor frame (same
representational space as root_delta_xy).  This makes the condition
prescriptive: at inference, set it to the direction you want the character
to travel.  Stand and locomotion phases only; goal phases
(pickup/carry/place) are steered by active_goal alone and get the null
token (0, 0).  Near-stationary windows (displacement < 1 cm) also return
(0, 0) so the model doesn't learn spurious directions from numerical noise.

The planner provides only 8 numbers:
  task_phase (5D one-hot-ish): [stand, locomotion, pickup, carry, place].
  OMOMO merges approach into pickup, then uses carry and place.
  Stand vs locomotion is labeled per window (v10): locomotion-phase windows
  whose GT net root speed over the target horizon is below
  STAND_SPEED_THRESH are relabeled stand. This removes the stand/walk
  ambiguity under a single locomotion token (the cause of freeze attractors
  and failed walk onsets in open-loop rollout).
  active_goal (3D):
    pickup (including merged approach) → pickup goal in anchor frame
    (world XY rotated, world Z)
    carry and place   → place goal in anchor frame (carry walks toward it)

Non-object clips (ACCAD/CMU) use task_phase=[0,1,0,0,0] (or stand) and
active_goal=zeros.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import torch

from dataset import JointFlowDataset  # for task goal / phase helpers (pre-X14)
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


PELVIS_IDX = 0
N_JOINTS = 24
JOINT_REL_DIM = (N_JOINTS - 1) * 3  # 69
STATE_DIM = (
    2 + 1 + 4 + JOINT_REL_DIM
)  # 76: root_delta_xy + root_z + quat (xyzw) + joint_rel
TARGET_FRAME_DIM = STATE_DIM  # 76
TASK_PHASE_DIM = 5
DESIRED_HEADING_DIM = 2
ACTIVE_GOAL_DIM = 3
TARGET_SPEED_DIM = 1  # mean ||root_vel|| over next H frames; 0 for non-locomotion

# loco_manip conditioning schema dims
LOCO_MANIP_MODE_DIM = 3  # [stand, locomotion, loco_manip]
PICKUP_GOAL_DIM = 3
PLACE_GOAL_DIM = 3

X15_REPRESENTATION = "x15_horizon_position_v10"
X15_REPRESENTATION_PREFIX = "x15_horizon_position"
X16_LOCO_MANIP_REPRESENTATION = "x16_loco_manip_condition_v1"
X16_LOCO_MANIP_REPRESENTATION_PREFIX = "x16_loco_manip_condition"

CONDITION_SCHEMA_PHASE = "phase"
CONDITION_SCHEMA_LOCO_MANIP = "loco_manip"
CONDITION_SCHEMAS = {CONDITION_SCHEMA_PHASE, CONDITION_SCHEMA_LOCO_MANIP}


def validate_x15_stats_compatible(
    stats: dict, *, require_current: bool = False
) -> None:
    """Validate stats for this 76D X15 implementation."""
    representation = str(stats.get("representation", ""))
    current_representations = {X15_REPRESENTATION, X16_LOCO_MANIP_REPRESENTATION}
    if require_current and representation not in current_representations:
        raise ValueError(
            f"stats representation {representation!r} is not one of "
            f"{sorted(current_representations)!r}; "
            "delete the stale stats.json or pass a matching --stats-in"
        )
    if not representation.startswith(
        (X15_REPRESENTATION_PREFIX, X16_LOCO_MANIP_REPRESENTATION_PREFIX)
    ):
        raise ValueError(
            f"stats representation {representation!r} is not a supported X15/X16 checkpoint"
        )
    state_dim = int(stats.get("state_dim", -1))
    target_frame_dim = int(stats.get("target_frame_dim", -1))
    if state_dim != STATE_DIM or target_frame_dim != TARGET_FRAME_DIM:
        raise ValueError(
            f"stats use state_dim={state_dim}, target_frame_dim={target_frame_dim}; "
            f"this code supports only {STATE_DIM}D X15 checkpoints. "
            "Older 75D v4 checkpoints need the archived 75D loader."
        )

# Phase order: [stand, locomotion, pickup, carry, place]
MODE_STAND = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
MODE_LOCOMOTION = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Net root speed over the target horizon below this (m/s, 30 Hz data) labels
# a locomotion-phase window as stand.
STAND_SPEED_THRESH = 0.2
SLOW_LOCO_RELABEL = "relabel"
SLOW_LOCO_DROP = "drop"
SLOW_LOCO_KEEP = "keep"
SLOW_LOCO_MODES = {SLOW_LOCO_RELABEL, SLOW_LOCO_DROP, SLOW_LOCO_KEEP}
FPS = 30.0
L_HAND_REL_IDX = 17
R_HAND_REL_IDX = 22

# Pre-built index arrays for rotating joint XY in-place (shape: (23,))
_J_XY_X = np.arange(7, STATE_DIM, 3)  # [7, 10, 13, ..., 73]
_J_XY_Y = _J_XY_X + 1  # [8, 11, 14, ..., 74]

# Left-right joint swap pairs (0-indexed in the 23-joint vector, i.e. body joint index - 1).
# Lateral axis is Y (L_Hip is at +Y, R_Hip at -Y relative to pelvis in world frame).
_LR_SWAP_PAIRS = [
    (0, 4),   # L_Hip  ↔ R_Hip
    (1, 5),   # L_Knee ↔ R_Knee
    (2, 6),   # L_Ankle ↔ R_Ankle
    (3, 7),   # L_Toe  ↔ R_Toe
    (13, 18), # L_Thorax ↔ R_Thorax
    (14, 19), # L_Shoulder ↔ R_Shoulder
    (15, 20), # L_Elbow ↔ R_Elbow
    (16, 21), # L_Wrist ↔ R_Wrist
    (17, 22), # L_Hand ↔ R_Hand
]


# ---------------------------------------------------------------------------
# Anchor-frame geometry helpers
# ---------------------------------------------------------------------------


def _rot_to_anchor(vec_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    """Rotate world-frame XY into the anchor heading frame (inverse rotation)."""
    c, s = np.cos(anchor_heading), np.sin(anchor_heading)
    x = c * vec_xy[..., 0] + s * vec_xy[..., 1]
    y = -s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def _angle_rel(angle: np.ndarray, anchor_heading: float) -> np.ndarray:
    rel = angle - anchor_heading
    return np.arctan2(np.sin(rel), np.cos(rel)).astype(np.float32)


def _quat_mul_z(theta: float, q: np.ndarray) -> np.ndarray:
    """Multiply a Z-axis rotation quat_z(-theta) by each row of q (N×4, xyzw).

    quat_z(-theta) = [0, 0, -sin(t/2), cos(t/2)]
    This applies an additional yaw of -theta to each orientation, equivalent
    to adding theta to the anchor heading for all frames at once.
    """
    ht = theta * 0.5
    sz, cz = -np.sin(ht), np.cos(ht)  # x1=0, y1=0, z1=sz, w1=cz
    x2, y2, z2, w2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty_like(q)
    out[:, 0] = cz * x2 - sz * y2  # w1*x2 - z1*y2  (x1=y1=0)
    out[:, 1] = cz * y2 + sz * x2  # w1*y2 + z1*x2  (with sign)
    out[:, 2] = cz * z2 + sz * w2  # w1*z2 + z1*w2
    out[:, 3] = cz * w2 - sz * z2  # w1*w2 - z1*z2
    # Enforce positive-w hemisphere
    neg = out[:, 3] < 0
    out[neg] = -out[neg]
    return out


# ---------------------------------------------------------------------------
# Clip-level task goal derivation (mirrors dataset_horizon.py, X14)
# ---------------------------------------------------------------------------


def _derive_world_task_goals(
    ts: np.ndarray, jp: np.ndarray, obj_rel: np.ndarray, pose_aa_root: np.ndarray
) -> np.ndarray:
    """Return (T, 9): [pickup_world(3), place_world(3), hold_rel(3)]."""
    T = len(ts)
    pelvis_world = jp[:, PELVIS_IDX, :].astype(np.float32)
    obj_world = JointFlowDataset._heading_rel_to_world(
        obj_rel, pelvis_world, pose_aa_root
    )
    obj_world_z = obj_world[:, 2]

    obj_step = np.zeros(T, dtype=np.float32)
    if T > 1:
        obj_step[1:] = np.linalg.norm(np.diff(obj_rel, axis=0), axis=1)

    carry_mask = JointFlowDataset._carry_mask(ts, jp, obj_rel, pose_aa_root)
    desired_hold_rel = np.median(obj_rel[carry_mask], axis=0).astype(np.float32)

    stable = obj_step < 0.05
    low_thresh = np.percentile(obj_world_z, 40)
    low = obj_world_z <= low_thresh
    first = np.arange(T) < max(1, int(0.35 * T))
    last = np.arange(T) >= max(0, int(0.65 * T))

    def goal_world(mask, fallback):
        if mask.sum() >= 5:
            return np.median(obj_world[mask], axis=0).astype(np.float32)
        return np.median(obj_world[fallback], axis=0).astype(np.float32)

    pickup = np.tile(
        goal_world(first & low & stable, slice(0, max(1, int(0.15 * T))))[None], (T, 1)
    )
    place = np.tile(
        goal_world(last & low & stable, slice(max(0, int(0.85 * T)), T))[None], (T, 1)
    )
    hold = np.tile(desired_hold_rel[None], (T, 1))
    return np.concatenate([pickup, place, hold], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class JointFlowX15Dataset(Dataset):
    """X15 receding-horizon dataset."""

    def __init__(
        self,
        hdf5_path: "str | list[str]",
        past_frames: int = 2,
        past_stride: int = 1,
        horizon: int = 15,
        stats: Optional[dict] = None,
        default_state: Optional[np.ndarray] = None,
        max_clips_per_source: Optional[int] = None,
        precompute_samples: bool = False,
        augment: bool = False,
        mirror: "bool | list[bool]" = False,
        cold_start_pad: int = 0,
        filter_loco_hand_speed: float = 0.0,
        filter_loco_hand_height: float = 0.0,
        filter_loco_hand_window: int = 1,
        use_target_speed: bool = False,
        condition_schema: str = CONDITION_SCHEMA_PHASE,
        slow_loco_mode: str = SLOW_LOCO_RELABEL,
    ):
        if past_frames < 1:
            raise ValueError("past_frames must be >= 1")
        if past_stride < 1:
            raise ValueError("past_stride must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if augment and precompute_samples:
            raise ValueError(
                "augment=True is incompatible with precompute_samples=True (use default precompute_samples=False)"
            )
        if filter_loco_hand_speed < 0.0:
            raise ValueError("filter_loco_hand_speed must be >= 0")
        if filter_loco_hand_height < 0.0:
            raise ValueError("filter_loco_hand_height must be >= 0")
        if filter_loco_hand_window < 1:
            raise ValueError("filter_loco_hand_window must be >= 1")
        if slow_loco_mode not in SLOW_LOCO_MODES:
            raise ValueError(f"slow_loco_mode must be one of {sorted(SLOW_LOCO_MODES)}, got {slow_loco_mode!r}")

        self.past_frames = past_frames
        self.past_stride = past_stride
        self.horizon = horizon
        self.augment = augment
        self.filter_loco_hand_speed = float(filter_loco_hand_speed)
        self.filter_loco_hand_height = float(filter_loco_hand_height)
        self.filter_loco_hand_window = int(filter_loco_hand_window)
        self.slow_loco_mode = slow_loco_mode
        self.slow_loco_stats = {
            "mode": self.slow_loco_mode,
            "speed_threshold": STAND_SPEED_THRESH,
            "relabelled_frames": 0,
            "dropped_frames": 0,
        }
        self.filter_loco_hand_stats = {
            "enabled": bool(filter_loco_hand_speed > 0.0 or filter_loco_hand_height > 0.0),
            "speed_threshold": self.filter_loco_hand_speed,
            "height_threshold": self.filter_loco_hand_height,
            "window": self.filter_loco_hand_window,
            "dropped_frames": 0,
            "kept_frames": 0,
        }
        self.condition_schema = (
            condition_schema
            if stats is None
            else stats.get("condition_schema", CONDITION_SCHEMA_PHASE)
        )
        if self.condition_schema not in CONDITION_SCHEMAS:
            raise ValueError(
                f"condition_schema must be one of {sorted(CONDITION_SCHEMAS)}, "
                f"got {self.condition_schema!r}"
            )
        self.use_target_speed = (
            use_target_speed
            if stats is None
            else bool(stats.get("use_target_speed", False))
        )
        self.state_dim = STATE_DIM
        self.target_frame_dim = TARGET_FRAME_DIM
        if self.condition_schema == CONDITION_SCHEMA_PHASE:
            task_cond_dim = TASK_PHASE_DIM + DESIRED_HEADING_DIM + ACTIVE_GOAL_DIM
        else:
            task_cond_dim = (
                LOCO_MANIP_MODE_DIM
                + DESIRED_HEADING_DIM
                + PICKUP_GOAL_DIM
                + PLACE_GOAL_DIM
            )
        self.cond_dim = (
            past_frames * STATE_DIM
            + task_cond_dim
            + (TARGET_SPEED_DIM if self.use_target_speed else 0)
        )
        self.target_dim = horizon * TARGET_FRAME_DIM

        paths = [hdf5_path] if isinstance(hdf5_path, str) else hdf5_path
        self._clips: list[dict] = []
        self._frame_index: list[tuple[int, int]] = []
        self._clip_source_ids: list[int] = []

        for source_id, path in enumerate(paths):
            do_mirror = mirror[source_id] if isinstance(mirror, list) else mirror
            with h5py.File(path, "r") as f:
                clip_names = sorted(f["clips"].keys())
                if max_clips_per_source is not None:
                    clip_names = clip_names[:max_clips_per_source]
                for clip_name in clip_names:
                    clip = self._load_clip(
                        f[f"clips/{clip_name}"],
                        clip_name,
                        path,
                        cold_start_pad,
                        horizon,
                    )
                    self._clips.append(clip)
                    self._clip_source_ids.append(source_id)
                    clip_idx = len(self._clips) - 1
                    self._append_clip_frame_indices(clip_idx, clip)
                    if do_mirror:
                        mclip = self._mirror_clip(clip)
                        self._clips.append(mclip)
                        self._clip_source_ids.append(source_id)
                        mclip_idx = len(self._clips) - 1
                        self._append_clip_frame_indices(mclip_idx, mclip)

        self.default_state = (
            np.asarray(default_state, dtype=np.float32)
            if default_state is not None
            else self._derive_default_state()
        )
        if self.default_state.shape != (STATE_DIM,):
            raise ValueError(f"default_state must have shape ({STATE_DIM},)")

        if stats is None:
            cond_mean, cond_std, target_mean, target_std = self._compute_stats()
            self.stats = {
                "cond_mean": cond_mean.tolist(),
                "cond_std": cond_std.tolist(),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
                "past_frames": past_frames,
                "past_stride": past_stride,
                "horizon": horizon,
                "state_dim": STATE_DIM,
                "target_frame_dim": TARGET_FRAME_DIM,
                "condition_schema": self.condition_schema,
                "task_phase_dim": TASK_PHASE_DIM,
                "desired_heading_dim": DESIRED_HEADING_DIM,
                "active_goal_dim": ACTIVE_GOAL_DIM,
                "loco_manip_mode_dim": (
                    LOCO_MANIP_MODE_DIM
                    if self.condition_schema == CONDITION_SCHEMA_LOCO_MANIP
                    else 0
                ),
                "pickup_goal_dim": (
                    PICKUP_GOAL_DIM
                    if self.condition_schema == CONDITION_SCHEMA_LOCO_MANIP
                    else 0
                ),
                "place_goal_dim": (
                    PLACE_GOAL_DIM
                    if self.condition_schema == CONDITION_SCHEMA_LOCO_MANIP
                    else 0
                ),
                "use_target_speed": self.use_target_speed,
                "filter_loco_hand": self.filter_loco_hand_stats,
                "slow_loco": self.slow_loco_stats,
                "default_state": self.default_state.tolist(),
                "representation": (
                    X16_LOCO_MANIP_REPRESENTATION
                    if self.condition_schema == CONDITION_SCHEMA_LOCO_MANIP
                    else X15_REPRESENTATION
                ),
            }
        else:
            self.stats = stats

        self.cond_mean = np.asarray(self.stats["cond_mean"], dtype=np.float32)
        self.cond_std = np.asarray(self.stats["cond_std"], dtype=np.float32)
        self.target_mean = np.asarray(self.stats["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(self.stats["target_std"], dtype=np.float32)
        stats_past_stride = int(self.stats.get("past_stride", 1))
        if stats_past_stride != self.past_stride:
            raise ValueError(
                f"stats use past_stride={stats_past_stride}, "
                f"but dataset was requested with past_stride={self.past_stride}"
            )

        self._cond_cache: Optional[np.ndarray] = None
        self._target_cache: Optional[np.ndarray] = None
        # Augmented base cache — built once, θ applied cheaply in __getitem__
        self._aug_frames: Optional[np.ndarray] = None
        self._aug_task_phase: Optional[np.ndarray] = None
        self._aug_loco_manip_mode: Optional[np.ndarray] = None
        self._aug_desired_heading_base: Optional[np.ndarray] = None
        self._aug_active_goal_base: Optional[np.ndarray] = None
        self._aug_pickup_goal_base: Optional[np.ndarray] = None
        self._aug_place_goal_base: Optional[np.ndarray] = None
        self._aug_target_speed: Optional[np.ndarray] = None

        if augment:
            self._precompute_augment_base()
        elif precompute_samples:
            self._precompute_samples()

    def _loco_hand_keep_mask(self, clip: dict) -> np.ndarray:
        T = len(clip["root_xy"])
        keep = np.ones(T, dtype=bool)
        if (
            clip["has_obj"]
            or (
                self.filter_loco_hand_speed <= 0.0
                and self.filter_loco_hand_height <= 0.0
            )
        ):
            return keep

        loco = clip["mode"][:, 1] > 0.5
        if not loco.any():
            return keep

        joint_rel = clip["joint_rel"].reshape(T, 23, 3)
        hand_rel = joint_rel[:, [L_HAND_REL_IDX, R_HAND_REL_IDX], :]
        bad = np.zeros(T, dtype=bool)
        if self.filter_loco_hand_speed > 0.0 and T > 1:
            hand_speed = np.zeros((T, 2), dtype=np.float32)
            hand_speed[1:] = np.linalg.norm(np.diff(hand_rel, axis=0), axis=2) * FPS
            bad |= hand_speed.max(axis=1) > self.filter_loco_hand_speed
        if self.filter_loco_hand_height > 0.0:
            hand_height = hand_rel[:, :, 2]
            bad |= hand_height.max(axis=1) > self.filter_loco_hand_height

        bad &= loco
        if self.filter_loco_hand_window > 1 and bad.any():
            k = self.filter_loco_hand_window
            run = np.convolve(bad.astype(np.int32), np.ones(k, dtype=np.int32), "same")
            bad &= run >= k

        if not bad.any():
            return keep

        # Drop a sample if any target frame in its horizon overlaps the bad run.
        bad_i = bad.astype(np.int32)
        csum = np.concatenate([[0], np.cumsum(bad_i)])
        for t in range(T):
            end = min(t + self.horizon, T)
            keep[t] = (csum[end] - csum[t]) == 0
        return keep

    def _append_clip_frame_indices(self, clip_idx: int, clip: dict) -> None:
        keep = self._loco_hand_keep_mask(clip)
        valid_frames = clip.get("valid_frames")
        if valid_frames is not None:
            keep = keep & valid_frames
        dropped = int((~keep).sum())
        self.filter_loco_hand_stats["dropped_frames"] += dropped
        self.filter_loco_hand_stats["kept_frames"] += int(keep.sum())
        for t, ok in enumerate(keep):
            if ok:
                self._frame_index.append((clip_idx, t))

    # ------------------------------------------------------------------
    # Clip loading
    # ------------------------------------------------------------------

    @staticmethod
    def _mirror_clip(clip: dict) -> dict:
        """Return a left-right mirrored copy of a clip dict.

        Mirrors across the XZ plane (negates world-frame Y):
        - joint_rel: swap L/R joint pairs, negate Y for all joints
        - root_xy: negate Y (world-frame lateral)
        - root_heading / root_roll: negate (antisymmetric under L/R flip)
        - root_pitch: unchanged
        - task_goals: negate Y for all three goal positions
        """
        joint_rel = clip["joint_rel"].reshape(-1, 23, 3).copy()
        for a, b in _LR_SWAP_PAIRS:
            joint_rel[:, [a, b]] = joint_rel[:, [b, a]]
        joint_rel[:, :, 1] *= -1  # negate lateral (Y) for all joints

        root_xy = clip["root_xy"].copy()
        root_xy[:, 1] *= -1

        root_vel = clip["root_vel"].copy()
        root_vel[:, 1] *= -1  # lateral velocity negates under L/R mirror

        task_goals = clip["task_goals"].copy()  # (T, 9): pickup(3) place(3) hold(3)
        task_goals[:, 1] *= -1  # pickup Y
        task_goals[:, 4] *= -1  # place Y
        task_goals[:, 7] *= -1  # hold Y

        return {
            "root_xy": root_xy,
            "root_z": clip["root_z"].copy(),
            "root_heading": -clip["root_heading"],
            "root_pitch": clip["root_pitch"].copy(),
            "root_roll": -clip["root_roll"],
            "joint_rel": joint_rel.reshape(-1, 69).astype(np.float32),
            "root_vel": root_vel,
            "clip_mean_speed": clip["clip_mean_speed"],
            "task_goals": task_goals,
            "mode": clip["mode"].copy(),
            "has_obj": clip["has_obj"],
        }

    @staticmethod
    def _load_clip(
        grp, clip_name: str, path: str, cold_start_pad: int = 0, horizon: int = 15
    ) -> dict:
        for key in ("task_space", "joint_positions"):
            if key not in grp:
                raise ValueError(f"Clip '{clip_name}' in {path} has no '{key}'")

        def _pad(arr: np.ndarray) -> np.ndarray:
            if cold_start_pad <= 0:
                return arr
            return np.concatenate([np.repeat(arr[:1], cold_start_pad, axis=0), arr])

        fps = float(grp.attrs.get("fps", grp.file.attrs.get("fps", FPS)))
        ts = _pad(grp["task_space"][:].astype(np.float32))
        jp = _pad(grp["joint_positions"][:].astype(np.float32))
        if cold_start_pad > 0:
            ts[
                :cold_start_pad, :2
            ] = 0.0  # padded frames are at rest: zero root velocity

        root_xy = jp[:, PELVIS_IDX, :2].astype(np.float32)
        root_z = jp[:, PELVIS_IDX, 2].astype(np.float32)
        joint_rel = (jp[:, 1:, :] - jp[:, PELVIS_IDX : PELVIS_IDX + 1, :]).reshape(
            len(jp), -1
        )

        pose_aa_root = (
            _pad(grp["pose_aa_root"][:].astype(np.float32))
            if "pose_aa_root" in grp
            else None
        )
        if pose_aa_root is not None:
            euler_zyx = R.from_rotvec(pose_aa_root).as_euler("ZYX").astype(np.float32)
            root_heading = euler_zyx[:, 0]
            root_pitch = euler_zyx[:, 1]
            root_roll = euler_zyx[:, 2]
        else:
            # fallback: derive heading from velocity; pitch/roll assumed zero
            vel = ts[:, :2]
            heading = np.zeros(len(ts), dtype=np.float32)
            last = 0.0
            for t in range(len(ts)):
                if np.linalg.norm(vel[t]) > 0.05:
                    last = float(np.arctan2(vel[t, 1], vel[t, 0]))
                heading[t] = last
            root_heading = heading
            root_pitch = np.zeros(len(ts), dtype=np.float32)
            root_roll = np.zeros(len(ts), dtype=np.float32)

        has_obj = "obj_com_pos_rel" in grp
        if has_obj:
            if pose_aa_root is None:
                raise ValueError(
                    f"Clip '{clip_name}' in {path} has object data but no 'pose_aa_root'"
                )
            obj_rel = _pad(grp["obj_com_pos_rel"][:].astype(np.float32))
            task_goals = _derive_world_task_goals(ts, jp, obj_rel, pose_aa_root)
            task_phase = JointFlowDataset._derive_task_phase(
                ts, jp, obj_rel, pose_aa_root, task_phase_dim=4
            )
            # Embed legacy 4D [locomotion, pickup, carry, place] into 5D
            # [stand, locomotion, pickup, carry, place].
            mode = np.zeros((len(ts), TASK_PHASE_DIM), dtype=np.float32)
            mode[:, 1:] = task_phase.astype(np.float32)
        else:
            task_goals = np.zeros((len(ts), 9), dtype=np.float32)
            mode = np.tile(MODE_LOCOMOTION[None], (len(ts), 1))

        # v10 stand/locomotion split: slow locomotion windows can be relabeled,
        # dropped, or kept. The default relabel mode preserves existing behavior.
        root_xy_clip = jp[:, PELVIS_IDX, :2].astype(np.float32)
        valid_frames = np.ones(len(ts), dtype=bool)
        T = len(ts)
        for t in range(T):
            if mode[t, 1] <= 0.5:
                continue
            end = min(t + horizon - 1, T - 1)
            if end == t:
                continue
            speed = (
                np.linalg.norm(root_xy_clip[end] - root_xy_clip[t]) * fps / (end - t)
            )
            if speed < STAND_SPEED_THRESH:
                if self.slow_loco_mode == SLOW_LOCO_RELABEL:
                    mode[t] = MODE_STAND
                    self.slow_loco_stats["relabelled_frames"] += 1
                elif self.slow_loco_mode == SLOW_LOCO_DROP:
                    valid_frames[t] = False
                    self.slow_loco_stats["dropped_frames"] += 1

        # clip-level mean locomotion speed — stable scalar for synthetic generation
        loco_mask = mode[:, 1] > 0.5
        if loco_mask.any():
            clip_mean_speed = float(np.linalg.norm(ts[loco_mask, :2], axis=1).mean())
        else:
            clip_mean_speed = 0.0

        return {
            "root_xy": root_xy,
            "root_z": root_z,
            "root_heading": root_heading,
            "root_pitch": root_pitch,
            "root_roll": root_roll,
            "joint_rel": joint_rel.astype(np.float32),
            "root_vel": ts[:, :2].copy(),
            "clip_mean_speed": clip_mean_speed,
            "task_goals": task_goals,
            "mode": mode,
            "valid_frames": valid_frames,
            "has_obj": has_obj,
        }

    def _derive_default_state(self) -> np.ndarray:
        """Median standing pose and root height as the default padding state."""
        clips = [c for c in self._clips if not c["has_obj"]] or self._clips
        joint_rel = np.median(
            np.concatenate([c["joint_rel"] for c in clips], axis=0), axis=0
        )
        root_z = np.median(np.concatenate([c["root_z"] for c in clips]))
        return np.concatenate(
            [
                np.zeros(2, dtype=np.float32),  # root_delta_xy = 0
                np.array([root_z], dtype=np.float32),  # absolute root height
                np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                ),  # identity quaternion [x,y,z,w]
                joint_rel.astype(np.float32),
            ]
        )

    # ------------------------------------------------------------------
    # State and target frame construction
    # ------------------------------------------------------------------

    def _state_frame(
        self,
        clip: dict,
        frame_idx: int,
        anchor_xy: np.ndarray,
        anchor_heading: float,
        anchor_pitch: float,
        anchor_roll: float,
    ) -> np.ndarray:
        if frame_idx < 0:
            return self.default_state
        frame_idx = min(frame_idx, len(clip["root_xy"]) - 1)
        root_delta = _rot_to_anchor(
            clip["root_xy"][frame_idx] - anchor_xy, anchor_heading
        )
        heading_rel = _angle_rel(clip["root_heading"][frame_idx], anchor_heading)
        pitch_rel = _angle_rel(clip["root_pitch"][frame_idx], anchor_pitch)
        roll_rel = _angle_rel(clip["root_roll"][frame_idx], anchor_roll)
        q = R.from_euler(
            "ZYX", [heading_rel, pitch_rel, roll_rel]
        ).as_quat()  # [x,y,z,w]
        if q[3] < 0:
            q = -q
        # Rotate joint XY into anchor heading frame so joints are body-frame aligned.
        joint_rel = clip["joint_rel"][frame_idx].reshape(23, 3).copy()
        joint_rel[:, :2] = _rot_to_anchor(joint_rel[:, :2], anchor_heading)
        return np.concatenate(
            [
                root_delta,
                clip["root_z"][frame_idx : frame_idx + 1],
                q.astype(np.float32),
                joint_rel.reshape(-1),
            ]
        ).astype(np.float32)

    def _target_frame(
        self,
        clip: dict,
        frame_idx: int,
        anchor_xy: np.ndarray,
        anchor_heading: float,
        anchor_pitch: float,
        anchor_roll: float,
    ) -> np.ndarray:
        return self._state_frame(
            clip, frame_idx, anchor_xy, anchor_heading, anchor_pitch, anchor_roll
        )

    # ------------------------------------------------------------------
    # Task conditioning
    # ------------------------------------------------------------------

    def _desired_heading(
        self, clip: dict, t: int, anchor_heading: float, anchor_xy: np.ndarray
    ) -> np.ndarray:
        """2D desired heading as travel direction over the horizon, anchor frame.

        Computed as the unit vector from anchor_xy to root_xy[t+H-1], rotated
        into anchor frame.  Stand and locomotion phases only; goal phases and
        near-stationary windows return (0, 0).
        """
        mode = clip["mode"][min(t, len(clip["mode"]) - 1)]
        if mode[0] + mode[1] <= 0.5:
            return np.zeros(2, dtype=np.float32)
        frame_idx = min(t + self.horizon - 1, len(clip["root_xy"]) - 1)
        disp = clip["root_xy"][frame_idx] - anchor_xy  # world-frame displacement
        dist = float(np.linalg.norm(disp))
        if dist < 0.01:  # < 1 cm — nearly stationary, no reliable direction
            return np.zeros(2, dtype=np.float32)
        disp_anchor = _rot_to_anchor(disp / dist, anchor_heading)
        return disp_anchor.astype(np.float32)

    def _active_goal(
        self,
        clip: dict,
        t: int,
        anchor_xy: np.ndarray,
        anchor_z: float,
        anchor_heading: float,
    ) -> np.ndarray:
        """3D active goal in anchor frame for the task phase at frame t."""
        if not clip["has_obj"]:
            return np.zeros(3, dtype=np.float32)
        frame_idx = min(t, len(clip["root_xy"]) - 1)

        # task_goals: [pickup_world(3), place_world(3), hold_rel(3)]
        goals = clip["task_goals"][frame_idx].reshape(3, 3).copy()
        goals[:2, :2] = _rot_to_anchor(goals[:2, :2] - anchor_xy, anchor_heading)
        # Subtract anchor Z from pickup and place so all axes are anchor-relative.
        goals[:2, 2] -= anchor_z

        mode = clip["mode"][frame_idx]  # OMOMO: pickup/approach, carry, place
        if mode[3] > 0.5 or mode[4] > 0.5:  # carry walks toward the place goal
            return goals[1].copy()
        else:  # pickup / merged approach
            return goals[0].copy()

    def _loco_manip_condition(
        self,
        clip: dict,
        t: int,
        anchor_xy: np.ndarray,
        anchor_z: float,
        anchor_heading: float,
    ) -> list:
        """loco_manip schema: [stand, loco, loco_manip] + heading + pickup + place."""
        mode5 = clip["mode"][min(t, len(clip["mode"]) - 1)]
        if clip["has_obj"]:
            mode = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            desired_heading = np.zeros(2, dtype=np.float32)
        elif mode5[0] > 0.5:
            mode = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            desired_heading = np.zeros(2, dtype=np.float32)
        else:
            mode = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            desired_heading = self._desired_heading(clip, t, anchor_heading, anchor_xy)

        if not clip["has_obj"]:
            return [
                mode,
                desired_heading,
                np.zeros(3, dtype=np.float32),  # pickup goal
                np.zeros(3, dtype=np.float32),  # place goal
            ]

        frame_idx = min(t, len(clip["root_xy"]) - 1)
        goals = clip["task_goals"][frame_idx].reshape(3, 3).copy()
        goals[:2, :2] = _rot_to_anchor(goals[:2, :2] - anchor_xy, anchor_heading)
        goals[:2, 2] -= anchor_z
        return [
            mode,
            desired_heading,
            goals[0].astype(np.float32),  # pickup goal, anchor frame
            goals[1].astype(np.float32),  # place goal, anchor frame
        ]

    def _target_speed(self, clip: dict, t: int) -> np.ndarray:
        """Clip-level mean locomotion speed; 0 for non-locomotion frames."""
        mode = clip["mode"][min(t, len(clip["mode"]) - 1)]
        if mode[1] <= 0.5:
            return np.zeros(1, dtype=np.float32)
        return np.array([clip["clip_mean_speed"]], dtype=np.float32)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _build_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        clip_idx, t = self._frame_index[idx]
        clip = self._clips[clip_idx]
        anchor_idx = max(0, t - 1)

        # Compute anchor values once. Add random yaw offset during training so the
        # model sees all heading directions and learns to respond to desired_heading.
        anchor_xy = clip["root_xy"][anchor_idx]
        anchor_z = float(clip["root_z"][anchor_idx])
        anchor_heading = float(clip["root_heading"][anchor_idx])
        anchor_pitch = float(clip["root_pitch"][anchor_idx])
        anchor_roll = float(clip["root_roll"][anchor_idx])
        if self.augment:
            anchor_heading += np.random.uniform(0, 2 * np.pi)

        past_indices = [
            t - 1 - self.past_stride * (self.past_frames - 1 - i)
            for i in range(self.past_frames)
        ]
        past = [
            self._state_frame(
                clip, frame_idx, anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            for frame_idx in past_indices
        ]
        if self.condition_schema == CONDITION_SCHEMA_PHASE:
            task_parts = [
                clip["mode"][min(t, len(clip["mode"]) - 1)].copy(),  # (5,)
                self._desired_heading(clip, t, anchor_heading, anchor_xy),  # (2,)
                self._active_goal(clip, t, anchor_xy, anchor_z, anchor_heading),  # (3,)
            ]
        else:
            task_parts = self._loco_manip_condition(
                clip, t, anchor_xy, anchor_z, anchor_heading
            )
        targets = [
            self._target_frame(
                clip, frame_idx, anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            for frame_idx in range(t, t + self.horizon)
        ]

        extras = [self._target_speed(clip, t)] if self.use_target_speed else []
        cond = np.concatenate(past + task_parts + extras).astype(np.float32)
        target = np.concatenate(targets).astype(np.float32)
        return cond, target

    # ------------------------------------------------------------------
    # Stats and caching
    # ------------------------------------------------------------------

    def _compute_stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cond_sum = np.zeros(self.cond_dim, dtype=np.float64)
        cond_sumsq = np.zeros(self.cond_dim, dtype=np.float64)
        target_sum = np.zeros(self.target_dim, dtype=np.float64)
        target_sumsq = np.zeros(self.target_dim, dtype=np.float64)
        n = 0
        for idx in range(len(self)):
            cond, target = self._build_sample(idx)
            c64, t64 = cond.astype(np.float64), target.astype(np.float64)
            cond_sum += c64
            cond_sumsq += c64 * c64
            target_sum += t64
            target_sumsq += t64 * t64
            n += 1
        cond_mean = cond_sum / n
        target_mean = target_sum / n
        cond_std = np.sqrt(np.maximum(cond_sumsq / n - cond_mean**2, 0.0)).clip(1e-6)
        target_std = np.sqrt(np.maximum(target_sumsq / n - target_mean**2, 0.0)).clip(
            1e-6
        )
        return (
            cond_mean.astype(np.float32),
            cond_std.astype(np.float32),
            target_mean.astype(np.float32),
            target_std.astype(np.float32),
        )

    def _precompute_samples(self) -> None:
        cc = np.empty((len(self), self.cond_dim), dtype=np.float32)
        tc = np.empty((len(self), self.target_dim), dtype=np.float32)
        for idx in range(len(self)):
            cond, target = self._build_sample(idx)
            cc[idx] = (cond - self.cond_mean) / self.cond_std
            tc[idx] = (target - self.target_mean) / self.target_std
        self._cond_cache = cc
        self._target_cache = tc

    def _precompute_augment_base(self) -> None:
        """Precompute per-sample base frames at θ=0 for fast augmented getitem.

        Scipy is called here once; __getitem__ then applies the random θ via
        cheap numpy ops (2×2 rotation + quaternion multiply).
        """
        N = len(self)
        KH = self.past_frames + self.horizon
        self._aug_frames = np.empty((N, KH, STATE_DIM), dtype=np.float32)
        self._aug_desired_heading_base = np.empty((N, 2), dtype=np.float32)
        self._aug_target_speed = np.empty((N, 1), dtype=np.float32)
        if self.condition_schema == CONDITION_SCHEMA_PHASE:
            self._aug_task_phase = np.empty((N, TASK_PHASE_DIM), dtype=np.float32)
            self._aug_active_goal_base = np.empty((N, 3), dtype=np.float32)
        else:
            self._aug_loco_manip_mode = np.empty((N, LOCO_MANIP_MODE_DIM), dtype=np.float32)
            self._aug_pickup_goal_base = np.empty((N, 3), dtype=np.float32)
            self._aug_place_goal_base = np.empty((N, 3), dtype=np.float32)

        for idx in range(N):
            clip_idx, t = self._frame_index[idx]
            clip = self._clips[clip_idx]
            anchor_idx = max(0, t - 1)
            anchor_xy = clip["root_xy"][anchor_idx]
            anchor_z = float(clip["root_z"][anchor_idx])
            anchor_heading = float(clip["root_heading"][anchor_idx])
            anchor_pitch = float(clip["root_pitch"][anchor_idx])
            anchor_roll = float(clip["root_roll"][anchor_idx])

            past_indices = [
                t - 1 - self.past_stride * (self.past_frames - 1 - i)
                for i in range(self.past_frames)
            ]
            target_indices = list(range(t, t + self.horizon))
            for fi, frame_idx in enumerate(past_indices + target_indices):
                self._aug_frames[idx, fi] = self._state_frame(
                    clip,
                    frame_idx,
                    anchor_xy,
                    anchor_heading,
                    anchor_pitch,
                    anchor_roll,
                )

            task_phase = clip["mode"][min(t, len(clip["mode"]) - 1)]
            fi_t = min(t, len(clip["root_xy"]) - 1)

            if self.condition_schema == CONDITION_SCHEMA_PHASE:
                self._aug_task_phase[idx] = task_phase
                self._aug_desired_heading_base[idx] = self._desired_heading(
                    clip, t, anchor_heading, anchor_xy
                )
                if clip["has_obj"]:
                    goals = clip["task_goals"][fi_t].reshape(3, 3).copy()
                    goals[:2, :2] = _rot_to_anchor(
                        goals[:2, :2] - anchor_xy, anchor_heading
                    )
                    goals[:2, 2] -= anchor_z
                    mode = task_phase
                    if mode[3] > 0.5 or mode[4] > 0.5:
                        self._aug_active_goal_base[idx] = goals[1]
                    else:
                        self._aug_active_goal_base[idx] = goals[0]
                else:
                    self._aug_active_goal_base[idx] = np.zeros(3, dtype=np.float32)
            else:
                parts = self._loco_manip_condition(
                    clip, t, anchor_xy, anchor_z, anchor_heading
                )
                (
                    self._aug_loco_manip_mode[idx],
                    self._aug_desired_heading_base[idx],
                    self._aug_pickup_goal_base[idx],
                    self._aug_place_goal_base[idx],
                ) = parts

            self._aug_target_speed[idx] = self._target_speed(clip, t)

    def _build_sample_augmented(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Fast augmented sample: apply random θ to precomputed base frames."""
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(-theta), np.sin(-theta)

        frames = self._aug_frames[idx].copy()  # (KH, 76)

        # Rotate root_delta_xy (0:2) by -theta
        dx, dy = frames[:, 0].copy(), frames[:, 1].copy()
        frames[:, 0] = c * dx - s * dy
        frames[:, 1] = s * dx + c * dy

        # Rotate joint XY pairs (body-frame) by -theta
        jx = frames[:, _J_XY_X].copy()
        jy = frames[:, _J_XY_Y].copy()
        frames[:, _J_XY_X] = c * jx - s * jy
        frames[:, _J_XY_Y] = s * jx + c * jy

        # Update quaternion: q_new = quat_z(-theta) * q_base  (exact, not approximation)
        frames[:, 3:7] = _quat_mul_z(theta, frames[:, 3:7])

        past = frames[: self.past_frames].reshape(-1)
        target = frames[self.past_frames :].reshape(-1)

        # Desired travel direction: rotate anchor-frame vector by -theta.
        hx, hy = self._aug_desired_heading_base[idx]
        desired_heading = np.array([c * hx - s * hy, s * hx + c * hy], dtype=np.float32)

        extras = [self._aug_target_speed[idx]] if self.use_target_speed else []

        if self.condition_schema == CONDITION_SCHEMA_PHASE:
            active_goal = self._aug_active_goal_base[idx].copy()
            gx, gy = active_goal[0], active_goal[1]
            active_goal[0] = c * gx - s * gy
            active_goal[1] = s * gx + c * gy
            task_parts = [self._aug_task_phase[idx], desired_heading, active_goal]
        else:
            pickup = self._aug_pickup_goal_base[idx].copy()
            px, py = pickup[0], pickup[1]
            pickup[0] = c * px - s * py
            pickup[1] = s * px + c * py
            place = self._aug_place_goal_base[idx].copy()
            lx, ly = place[0], place[1]
            place[0] = c * lx - s * ly
            place[1] = s * lx + c * ly
            task_parts = [
                self._aug_loco_manip_mode[idx],
                desired_heading,
                pickup,
                place,
            ]

        cond = np.concatenate([past] + task_parts + extras)
        return cond.astype(np.float32), target.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int):
        if self._cond_cache is not None:
            return torch.from_numpy(self._cond_cache[idx]), torch.from_numpy(
                self._target_cache[idx]
            )
        if self._aug_frames is not None:
            cond, target = self._build_sample_augmented(idx)
        else:
            cond, target = self._build_sample(idx)
        cond = (cond - self.cond_mean) / self.cond_std
        target = (target - self.target_mean) / self.target_std
        return torch.from_numpy(cond), torch.from_numpy(target)

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    def clip_len(self, clip_idx: int) -> int:
        return len(self._clips[clip_idx]["root_xy"])

    def clip_source_id(self, clip_idx: int) -> int:
        return self._clip_source_ids[clip_idx]

    def frame_indices_for_clips(self, clip_indices: list[int]) -> list[int]:
        clip_set = set(clip_indices)
        return [fi for fi, (ci, _) in enumerate(self._frame_index) if ci in clip_set]

    def next_window_indices(self, k: int) -> np.ndarray:
        """Sample index of the window k frames later in the same clip, or -1.

        Window (clip, t+k)'s past frames are GT frames t+k-K..t+k-1, which are
        target frames 0..K-1 of window (clip, t) when k == K. _frame_index is
        built with t consecutive per clip, so that window sits at idx + k.
        """
        n = len(self._frame_index)
        nxt = np.full(n, -1, dtype=np.int64)
        for i, (ci, t) in enumerate(self._frame_index):
            j = i + k
            if j < n and self._frame_index[j] == (ci, t + k):
                nxt[i] = j
        return nxt

    def save_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @staticmethod
    def load_stats(path: str) -> dict:
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, nargs="+")
    p.add_argument("--past-frames", type=int, default=2)
    p.add_argument("--motion-fps", type=float, default=None)
    p.add_argument("--policy-hz", type=float, default=None)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--max-clips-per-source", type=int, default=None)
    p.add_argument("--stats-out", default=None)
    p.add_argument("--samples", type=int, default=3)
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
    past_stride = derive_past_stride(args.motion_fps, args.policy_hz)
    ds = JointFlowX15Dataset(
        args.hdf5,
        past_frames=args.past_frames,
        past_stride=past_stride,
        horizon=args.horizon,
        max_clips_per_source=args.max_clips_per_source,
    )
    print(f"clips={ds.num_clips:,}  samples={len(ds):,}")
    print(
        f"cond_dim={ds.cond_dim}  ({ds.past_frames}×{STATE_DIM} past, "
        f"stride={ds.past_stride} + {TASK_PHASE_DIM} phase + "
        f"{DESIRED_HEADING_DIM} heading + {ACTIVE_GOAL_DIM} goal)"
    )
    print(f"target_dim={ds.target_dim}  ({ds.horizon}×{TARGET_FRAME_DIM})")
    for idx in np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int):
        cond, target = ds[idx]
        print(
            f"  sample {idx}: cond={tuple(cond.shape)} target={tuple(target.shape)} "
            f"cond_std={cond.float().std().item():.3f} target_std={target.float().std().item():.3f}"
        )
    if args.stats_out:
        ds.save_stats(args.stats_out)
        print(f"saved stats -> {args.stats_out}")


if __name__ == "__main__":
    main()
