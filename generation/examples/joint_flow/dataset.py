"""
HDF5 dataset for joint-flow training.

Each sample is a (cond, joints) pair where:
  cond   (K*9,) — last K frames of root-relative task-space (sliding window)
  joints (72,)  — root-relative joint positions at the current frame

Root-relative normalization:
  joint_positions: subtract pelvis (joint 0) so pelvis = [0,0,0]
  EEF xyz in task_space (cols 3-8): subtract pelvis world XYZ per frame
  root_vel_xy (cols 0-1) can stay in world frame or be rotated into the
  start-heading frame; pelvis_height (col 2) is kept as-is.

Window boundary: frames before the clip start are padded by repeating frame 0.

Stats (mean/std per feature, computed on single-frame cond and joints) are saved
alongside the checkpoint so inference can normalize/denormalize correctly.
"""

import json
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PELVIS_IDX = 0
N_JOINTS = 24
COND_DIM = 9
JOINT_DIM = N_JOINTS * 3  # 72
JOINT_DIM_6D = 23 * 6  # 138 — 6D rotation per body joint
JOINT_DIM_6D_ROOT = 24 * 6  # 144 — 6D rotation per body joint + pelvis
TASK_PHASE_DIM = 5
TASK_PHASE_DIM_LEGACY = 4
OBJ_SENTINEL = np.array(
    [2.0, 2.0, 2.0], dtype=np.float32
)  # out-of-range dummy for non-OMOMO clips


class JointFlowDataset(Dataset):
    """
    Sliding-window dataset.

    Args:
        hdf5_path (str | list[str]): Path or paths to task-space HDF5 files.
        window (int): Number of past frames, including the current frame, in
            each condition window.
        stats (Optional[dict]): Precomputed normalization statistics. When
            omitted, statistics are computed from the loaded data.
        use_prev_frame (bool): Append the previous target pose to the
            conditioning vector.
        no_eef (bool): Drop end-effector coordinates from the task-space
            condition.
        use_6d (bool): Use 6D joint rotations instead of joint positions as
            training targets.
        predict_root (bool): Include pelvis rotation in the 6D target. Only
            applies when `use_6d` is enabled.
        use_obj (bool): Append object relative position features when present.
        use_task_goals (bool): Append pickup, place, and desired-hold goal
            features.
        motion_frame (str): `world` keeps locomotion XY in world axes; `start`
            rotates it into the frame defined by frame-0 pelvis yaw.
        motion_repr (str): `velocity` keeps per-frame root_vel_xy; `future_dxdy`
            replaces it with future root displacement over a short horizon.
        motion_horizon_s (float): Horizon in seconds for `future_dxdy`.
        task_goal_frame (str): `current` makes pickup/place relative to the
            current pelvis heading; `start` makes them relative to frame 0.
        use_carry_phase (bool): Append a 1D carrying indicator derived from
            OMOMO object height and motion; non-object clips get zero.
        use_task_phase (bool): Append task phase one-hot.
        task_phase_dim (int): `4` for [pickup, carry_walk, place] with
            approach merged into pickup; `5` for [pickup, carry_stand,
            carry_walk, place] with approach merged into pickup.
        scheduled_sampling (bool): Also return the condition that predicts the
            previous frame, so the training loop can feed the model its own
            one-step prediction as the prev-frame (autoregressive training).

    Raises:
        ValueError: If `use_6d` or `use_task_goals` requires pose datasets that
            are missing from an input HDF5 clip.
    """

    def __init__(
        self,
        hdf5_path: "str | list[str]",
        window: int = 10,
        stats: Optional[dict] = None,
        use_prev_frame: bool = False,
        no_eef: bool = False,
        use_6d: bool = False,
        predict_root: bool = False,
        use_obj: bool = False,
        use_task_goals: bool = False,
        motion_frame: str = "world",
        motion_repr: str = "velocity",
        motion_horizon_s: float = 0.5,
        task_goal_frame: str = "current",
        use_carry_phase: bool = False,
        use_task_phase: bool = False,
        task_phase_dim: int = TASK_PHASE_DIM_LEGACY,
        scheduled_sampling: bool = False,
    ):
        if task_goal_frame not in {"current", "start"}:
            raise ValueError("--task-goal-frame must be 'current' or 'start'")
        if motion_frame not in {"world", "start"}:
            raise ValueError("--motion-frame must be 'world' or 'start'")
        if motion_repr not in {"velocity", "future_dxdy"}:
            raise ValueError("--motion-repr must be 'velocity' or 'future_dxdy'")
        if motion_horizon_s <= 0.0:
            raise ValueError("--motion-horizon-s must be > 0")
        if task_phase_dim not in {TASK_PHASE_DIM_LEGACY, TASK_PHASE_DIM}:
            raise ValueError("--task-phase-dim must be 4 or 5")
        self.window = window
        self.scheduled_sampling = scheduled_sampling
        self.use_prev_frame = use_prev_frame
        self.no_eef = no_eef
        self.use_6d = use_6d
        self.predict_root = predict_root and use_6d  # only meaningful with 6D
        self.use_obj = use_obj
        self.use_task_goals = use_task_goals
        self.motion_frame = motion_frame
        self.motion_repr = motion_repr
        self.motion_horizon_s = motion_horizon_s
        self.task_goal_frame = task_goal_frame
        self.use_carry_phase = use_carry_phase
        self.use_task_phase = use_task_phase
        self.task_phase_dim = task_phase_dim
        self.ts_dim = (
            (3 if no_eef else 9)
            + (3 if use_obj else 0)
            + (9 if use_task_goals else 0)
            + (1 if use_carry_phase else 0)
            + (task_phase_dim if use_task_phase else 0)
        )
        paths = [hdf5_path] if isinstance(hdf5_path, str) else hdf5_path

        if use_6d:
            from rot_utils import pose_aa_to_6d

        # per-clip arrays (root-relative, un-standardized)
        self._clip_conds: list[np.ndarray] = []  # each (T, 9)
        self._clip_joints: list[np.ndarray] = []  # each (T, 69) or (T, 138)
        self._clip_has_obj: list[bool] = []  # True if clip has real obj data
        self._clip_source_ids: list[int] = []  # source HDF5 index per clip
        self._raw_obj: list = []  # (T,3) real obj or None
        self._task_goals: list = []  # (T,9) pickup/place/hold or None
        self._carry_phase: list = []  # (T,1) real carry indicator or None
        self._task_phase: list = []  # (T,4/5) real phase one-hot or None

        for source_id, path in enumerate(paths):
            with h5py.File(path, "r") as f:
                for clip_name in f["clips"]:
                    grp = f[f"clips/{clip_name}"]
                    ts = grp["task_space"][:]  # (T, 9)
                    if use_6d:
                        if "pose_aa_body" not in grp:
                            raise ValueError(
                                f"Clip '{clip_name}' in {path} has no 'pose_aa_body'. "
                                "Re-run extract_taskspace_hdf5.py to add it."
                            )
                        pa_body = grp["pose_aa_body"][:]  # (T, 69)
                        body_6d = pose_aa_to_6d(
                            pa_body.reshape(len(pa_body), 23, 3)
                        )  # (T, 138)
                        if self.predict_root:
                            if "pose_aa_root" not in grp:
                                raise ValueError(
                                    f"Clip '{clip_name}' in {path} has no 'pose_aa_root'. "
                                    "Re-run extract_taskspace_hdf5.py to add it."
                                )
                            pa_root = grp["pose_aa_root"][:]  # (T, 3)
                            root_6d = pose_aa_to_6d(
                                pa_root.reshape(len(pa_root), 1, 3)
                            )  # (T, 6)
                            joints = np.concatenate(
                                [root_6d, body_6d], axis=1
                            )  # (T, 144)
                        else:
                            joints = body_6d  # (T, 138)
                        cond, _ = self._normalize_clip(ts, None)
                    else:
                        jp = grp["joint_positions"][:]  # (T, 24, 3)
                        cond, joints = self._normalize_clip(ts, jp)
                    if motion_repr == "future_dxdy":
                        fps = float(grp.attrs["fps"])
                        cond[:, :2] = self._future_dxdy_from_world_vel(
                            cond[:, :2], fps, motion_horizon_s
                        )
                    if motion_frame == "start":
                        if "pose_aa_root" not in grp:
                            raise ValueError(
                                f"Clip '{clip_name}' in {path} has no 'pose_aa_root'. "
                                "Start-frame motion conditioning requires root orientations."
                            )
                        pose_aa_root_vel = grp["pose_aa_root"][:].astype(np.float32)
                        cond[:, :2] = self._world_xy_to_start_heading_rel(
                            cond[:, :2], pose_aa_root_vel
                        )
                    if no_eef:
                        cond = cond[:, :3]
                    has_obj = False
                    raw_obj = None
                    task_goals = None
                    carry_phase = None
                    task_phase = None
                    if (
                        use_obj
                        or use_task_goals
                        or use_carry_phase
                        or use_task_phase
                    ) and "obj_com_pos_rel" in grp:
                        raw_obj = grp["obj_com_pos_rel"][:].astype(np.float32)  # (T, 3)
                        has_obj = True
                    if (use_task_goals or use_carry_phase or use_task_phase) and has_obj:
                        if "joint_positions" not in grp or "pose_aa_root" not in grp:
                            raise ValueError(
                                f"Clip '{clip_name}' in {path} needs 'joint_positions' and "
                                "'pose_aa_root' for task-goal/phase features."
                            )
                        jp_for_goals = grp["joint_positions"][:].astype(np.float32)
                        pose_aa_root = grp["pose_aa_root"][:].astype(np.float32)
                        if use_task_goals:
                            task_goals = self._derive_task_goals(
                                ts,
                                jp_for_goals,
                                raw_obj,
                                pose_aa_root,
                                task_goal_frame=task_goal_frame,
                            )
                        if use_carry_phase:
                            carry_phase = self._derive_carry_phase(
                                ts, jp_for_goals, raw_obj, pose_aa_root
                            )
                        if use_task_phase:
                            task_phase = self._derive_task_phase(
                                ts,
                                jp_for_goals,
                                raw_obj,
                                pose_aa_root,
                                task_phase_dim=task_phase_dim,
                            )
                    self._clip_conds.append(cond)
                    self._clip_joints.append(joints)
                    self._clip_has_obj.append(has_obj)
                    self._clip_source_ids.append(source_id)
                    self._raw_obj.append(raw_obj)  # real (T,3) or None
                    self._task_goals.append(task_goals)  # real (T,9) or None
                    self._carry_phase.append(carry_phase)  # real (T,1) or None
                    self._task_phase.append(task_phase)  # real (T,5) or None

        # For object/task fields: real OMOMO values or out-of-distribution sentinels
        # for locomotion-only clips. Object and pickup sentinels share a linear
        # ramp; place is a farther linear ramp; hold stays a far constant target.
        # sentinel_end[i] = max_omomo_obj[i] + 0.1 (guaranteed > all real obj values)
        if use_obj or use_task_goals or use_carry_phase or use_task_phase:
            if any(self._clip_has_obj):
                max_obj = np.concatenate(
                    [o for o, has in zip(self._raw_obj, self._clip_has_obj) if has],
                    axis=0,
                ).max(
                    axis=0
                )  # (3,)
            else:
                max_obj = np.array([1.2, 0.5, 1.2], dtype=np.float32)
            sentinel_end = (max_obj + 0.1).astype(np.float32)
            sentinel_start = (sentinel_end * 2.0).astype(np.float32)
            for i, (
                cond,
                raw_obj,
                task_goals,
                carry_phase,
                task_phase,
                has,
            ) in enumerate(
                zip(
                    self._clip_conds,
                    self._raw_obj,
                    self._task_goals,
                    self._carry_phase,
                    self._task_phase,
                    self._clip_has_obj,
                )
            ):
                T = len(cond)
                t_arr = np.linspace(0, 1, T, dtype=np.float32)[:, None]  # (T, 1)
                obj_sentinel = (
                    sentinel_start[None] + (sentinel_end - sentinel_start)[None] * t_arr
                )
                place_sentinel = obj_sentinel + sentinel_end[None]
                hold_sentinel = sentinel_start[None].repeat(T, axis=0)
                # Non-object clips should read as "not carrying": keep x/y far
                # out of distribution, but put the desired hold height at floor
                # level relative to the current pelvis.
                hold_sentinel[:, 2] = -cond[:, 2]
                extra = []
                if use_obj:
                    if has:
                        obj_arr = raw_obj
                    else:
                        obj_arr = obj_sentinel
                    extra.append(obj_arr)
                if use_task_goals:
                    if has:
                        goal_arr = task_goals
                    else:
                        goal_arr = np.concatenate(
                            [obj_sentinel, place_sentinel, hold_sentinel],
                            axis=1,
                        )
                    extra.append(goal_arr)
                if use_carry_phase:
                    if has:
                        phase_arr = carry_phase
                    else:
                        phase_arr = np.zeros((T, 1), dtype=np.float32)
                    extra.append(phase_arr)
                if use_task_phase:
                    if has:
                        task_phase_arr = task_phase
                    else:
                        task_phase_arr = np.zeros((T, task_phase_dim), dtype=np.float32)
                        task_phase_arr[:, 0] = 1.0
                    extra.append(task_phase_arr)
                self._clip_conds[i] = np.concatenate([cond] + extra, axis=1)

        # compute stats on single-frame data (before windowing)
        all_conds = np.concatenate(self._clip_conds, axis=0).astype(np.float32)
        all_joints = np.concatenate(self._clip_joints, axis=0).astype(np.float32)

        if stats is None:
            if (
                use_obj or use_task_goals or use_carry_phase or use_task_phase
            ) and any(self._clip_has_obj):
                # Compute object/task stats only from clips with real object data
                # to avoid sentinel values poisoning the normalization.
                base_dim = 3 if no_eef else 9
                base_conds = all_conds[:, :base_dim]
                obj_conds_real = np.concatenate(
                    [
                        c[:, base_dim:]
                        for c, has in zip(self._clip_conds, self._clip_has_obj)
                        if has
                    ],
                    axis=0,
                )
                cond_mean = np.concatenate([base_conds.mean(0), obj_conds_real.mean(0)])
                cond_std = np.concatenate(
                    [base_conds.std(0).clip(1e-6), obj_conds_real.std(0).clip(1e-6)]
                )
            else:
                cond_mean = all_conds.mean(0)
                cond_std = all_conds.std(0).clip(1e-6)
            self.stats = {
                "cond_mean": cond_mean.tolist(),
                "cond_std": cond_std.tolist(),
                "joint_mean": all_joints.mean(0).tolist(),
                "joint_std": all_joints.std(0).clip(1e-6).tolist(),
                "window": window,
                "use_prev_frame": use_prev_frame,
                "no_eef": no_eef,
                "use_6d": use_6d,
                "predict_root": self.predict_root,
                "use_obj": use_obj,
                "use_task_goals": use_task_goals,
                "motion_frame": motion_frame,
                "motion_repr": motion_repr,
                "motion_horizon_s": motion_horizon_s,
                "task_goal_frame": task_goal_frame,
                "use_carry_phase": use_carry_phase,
                "use_task_phase": use_task_phase,
                "task_phase_dim": task_phase_dim if use_task_phase else 0,
            }
        else:
            self.stats = stats

        cond_mean = np.array(self.stats["cond_mean"], dtype=np.float32)
        cond_std = np.array(self.stats["cond_std"], dtype=np.float32)
        joint_mean = np.array(self.stats["joint_mean"], dtype=np.float32)
        joint_std = np.array(self.stats["joint_std"], dtype=np.float32)

        # standardize per-clip and build flat index
        self._conds_std: list[np.ndarray] = []
        self._joints_std: list[np.ndarray] = []
        self._frame_index: list[Tuple[int, int]] = []  # (clip_idx, frame_idx)

        for ci, (c, j) in enumerate(zip(self._clip_conds, self._clip_joints)):
            c_std = (c.astype(np.float32) - cond_mean) / cond_std
            j_std = (j.astype(np.float32) - joint_mean) / joint_std
            self._conds_std.append(c_std)
            self._joints_std.append(j_std)
            for t in range(len(c)):
                self._frame_index.append((ci, t))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._frame_index)

    def _build_cond(self, ci: int, t: int) -> np.ndarray:
        """Flattened conditioning vector (prev-frame + task-space window) for frame t."""
        c_std = self._conds_std[ci]  # (T, cond_dim)
        j_std = self._joints_std[ci]  # (T, joint_dim)

        # sliding window: [t-K+1 .. t], pad start by repeating frame 0
        K = self.window
        start = t - K + 1
        if start < 0:
            pad = c_std[0:1].repeat(-start, axis=0)  # repeat first frame
            window_cond = np.concatenate([pad, c_std[0 : t + 1]], axis=0)
        else:
            window_cond = c_std[start : t + 1]  # (K, cond_dim)

        cond_flat = window_cond.reshape(-1)  # (K*cond_dim,)

        if self.use_prev_frame:
            prev_dim = j_std.shape[1]  # 144, 138, or 69 depending on flags
            prev = j_std[t - 1] if t > 0 else np.zeros(prev_dim, dtype=np.float32)
            cond_flat = np.concatenate([prev, cond_flat])

        return cond_flat

    def __getitem__(self, idx: int):
        ci, t = self._frame_index[idx]
        cond_flat = self._build_cond(ci, t)
        joints = self._joints_std[ci][t]  # (joint_dim,)

        if self.scheduled_sampling:
            # Also emit the condition that predicts frame t-1, so the training
            # loop can roll the model one step and feed its own output as the
            # prev-frame. Frame 0 has no predecessor; reuse its own condition.
            prev_cond = self._build_cond(ci, t - 1) if t > 0 else cond_flat
            return (
                torch.from_numpy(cond_flat),
                torch.from_numpy(prev_cond),
                torch.from_numpy(joints),
            )

        return torch.from_numpy(cond_flat), torch.from_numpy(joints)

    @property
    def num_clips(self) -> int:
        return len(self._clip_conds)

    def clip_len(self, clip_idx: int) -> int:
        return len(self._clip_conds[clip_idx])

    def clip_source_id(self, clip_idx: int) -> int:
        return self._clip_source_ids[clip_idx]

    def frame_indices_for_clips(self, clip_indices: list[int]) -> list[int]:
        clip_set = set(clip_indices)
        return [
            frame_idx
            for frame_idx, (clip_idx, _) in enumerate(self._frame_index)
            if clip_idx in clip_set
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_clip(
        ts: np.ndarray,  # (T, 9)
        jp: Optional[np.ndarray],  # (T, 24, 3) or None when use_6d=True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        cond = ts.copy()
        if jp is not None:
            pelvis_world = jp[:, PELVIS_IDX, :]
            joints_rel = jp - pelvis_world[:, None, :]
            joints_flat = joints_rel[:, 1:, :].reshape(len(ts), -1)  # (T, 69)
            cond[:, 3:6] -= pelvis_world
            cond[:, 6:9] -= pelvis_world
            return cond, joints_flat
        # use_6d: cond still needs pelvis subtraction — use ts height as proxy (no EEF offset)
        return cond, None

    # ------------------------------------------------------------------
    @staticmethod
    def _heading_rot_mats(pose_aa_root: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        yaws = R.from_rotvec(pose_aa_root).as_euler("ZYX")[:, 0]
        c, s = np.cos(yaws), np.sin(yaws)
        mats = np.zeros((len(yaws), 2, 2), dtype=np.float32)
        mats[:, 0, 0] = c
        mats[:, 0, 1] = -s
        mats[:, 1, 0] = s
        mats[:, 1, 1] = c
        return mats

    @classmethod
    def _world_to_heading_rel(
        cls,
        world_xyz: np.ndarray,
        pelvis_world: np.ndarray,
        pose_aa_root: np.ndarray,
    ) -> np.ndarray:
        """Convert fixed/per-frame world XYZ to pelvis heading-frame relative XYZ."""
        if world_xyz.ndim == 1:
            world_xyz = np.tile(world_xyz[None], (len(pelvis_world), 1))
        delta = (world_xyz - pelvis_world).astype(np.float32)
        mats = cls._heading_rot_mats(pose_aa_root)
        rel_xy = np.einsum("tji,tj->ti", mats, delta[:, :2])
        return np.concatenate([rel_xy, delta[:, 2:3]], axis=1).astype(np.float32)

    @classmethod
    def _heading_rel_to_world(
        cls,
        rel_xyz: np.ndarray,
        pelvis_world: np.ndarray,
        pose_aa_root: np.ndarray,
    ) -> np.ndarray:
        mats = cls._heading_rot_mats(pose_aa_root)
        world_xy = np.einsum("tij,tj->ti", mats, rel_xyz[:, :2]) + pelvis_world[:, :2]
        world_z = rel_xyz[:, 2:3] + pelvis_world[:, 2:3]
        return np.concatenate([world_xy, world_z], axis=1).astype(np.float32)

    @classmethod
    def _world_to_start_heading_rel(
        cls,
        world_xyz: np.ndarray,
        pelvis_world: np.ndarray,
        pose_aa_root: np.ndarray,
    ) -> np.ndarray:
        if world_xyz.ndim == 1:
            world_xyz = np.tile(world_xyz[None], (len(pelvis_world), 1))
        start_pelvis = np.tile(pelvis_world[0:1], (len(pelvis_world), 1))
        start_root = np.tile(pose_aa_root[0:1], (len(pelvis_world), 1))
        return cls._world_to_heading_rel(world_xyz, start_pelvis, start_root)

    @classmethod
    def _world_xy_to_start_heading_rel(
        cls, world_xy: np.ndarray, pose_aa_root: np.ndarray
    ) -> np.ndarray:
        start_root = np.tile(pose_aa_root[0:1], (len(world_xy), 1))
        mats = cls._heading_rot_mats(start_root)
        return np.einsum("tji,tj->ti", mats, world_xy.astype(np.float32)).astype(
            np.float32
        )

    @staticmethod
    def _future_dxdy_from_world_vel(
        world_xy_vel: np.ndarray, fps: float, horizon_s: float
    ) -> np.ndarray:
        dt = 1.0 / fps
        horizon = max(1, int(round(horizon_s * fps)))
        xy = np.cumsum(world_xy_vel.astype(np.float32) * dt, axis=0)
        future = np.empty_like(world_xy_vel, dtype=np.float32)
        last = len(world_xy_vel) - 1
        for t in range(len(world_xy_vel)):
            t1 = min(last, t + horizon)
            future[t] = xy[t1] - xy[t]
        return future

    @classmethod
    def _carry_mask(
        cls,
        ts: np.ndarray,
        jp: np.ndarray,
        obj_rel: np.ndarray,
        pose_aa_root: np.ndarray,
    ) -> np.ndarray:
        T = len(ts)
        pelvis_world = jp[:, PELVIS_IDX, :].astype(np.float32)
        obj_world = cls._heading_rel_to_world(obj_rel, pelvis_world, pose_aa_root)
        obj_world_z = obj_world[:, 2]

        obj_step = np.zeros(T, dtype=np.float32)
        if T > 1:
            obj_step[1:] = np.linalg.norm(np.diff(obj_rel, axis=0), axis=1)

        speed = np.linalg.norm(ts[:, :2], axis=1)
        pelvis_z = ts[:, 2]
        carry_mask = (
            (speed > 0.25)
            & (obj_world_z > 0.95)
            & (obj_step < 0.05)
            & (pelvis_z > 0.85)
        )
        if carry_mask.sum() < 10:
            carry_mask = (speed > 0.25) & (obj_world_z > 0.95) & (pelvis_z > 0.85)
        if carry_mask.sum() < 10:
            carry_mask = obj_world_z > 0.95
        if carry_mask.sum() < 1:
            carry_mask = np.ones(T, dtype=bool)
        return carry_mask

    @classmethod
    def _derive_carry_phase(
        cls,
        ts: np.ndarray,
        jp: np.ndarray,
        obj_rel: np.ndarray,
        pose_aa_root: np.ndarray,
    ) -> np.ndarray:
        return cls._carry_mask(ts, jp, obj_rel, pose_aa_root).astype(np.float32)[:, None]

    @classmethod
    def _derive_task_phase(
        cls,
        ts: np.ndarray,
        jp: np.ndarray,
        obj_rel: np.ndarray,
        pose_aa_root: np.ndarray,
        task_phase_dim: int = TASK_PHASE_DIM,
    ) -> np.ndarray:
        """Return one-hot task phase in 4D or 5D schema."""
        T = len(ts)
        pelvis_world = jp[:, PELVIS_IDX, :].astype(np.float32)
        obj_world = cls._heading_rel_to_world(obj_rel, pelvis_world, pose_aa_root)
        obj_world_z = obj_world[:, 2]

        speed = np.linalg.norm(ts[:, :2], axis=1)

        # Every OMOMO clip is one monotone pickup -> carry -> place, but
        # the pickup and place surfaces can sit at different heights
        # (ground vs table), so a single absolute or percentile height
        # cutoff mislabels one end of the clip. Reference the object's
        # own resting height at each end instead: carry starts when the
        # object first rises clearly above its start rest height and ends
        # the last time it is still clearly above its end rest height.
        k = 7
        pad = k // 2
        z_smooth = np.convolve(np.pad(obj_world_z, pad, mode="edge"), np.ones(k) / k, "valid")
        rest_n = min(15, max(1, T // 10))
        z_rest_start = float(np.median(z_smooth[:rest_n]))
        z_rest_end = float(np.median(z_smooth[-rest_n:]))
        lift_margin = 0.10

        above_start = np.flatnonzero(z_smooth > z_rest_start + lift_margin)
        above_end = np.flatnonzero(z_smooth > z_rest_end + lift_margin)
        c0 = int(above_start[0]) if above_start.size else -1
        c1 = int(above_end[-1]) if above_end.size else -1

        if not (0 <= c0 <= c1):
            # Fallback 1: longest run above the midpoint of the height range.
            z_lo = np.percentile(z_smooth, 10)
            z_hi = np.percentile(z_smooth, 90)
            lifted = z_smooth > (z_lo + z_hi) / 2.0
            runs = []
            start = None
            for t in range(T):
                if lifted[t] and start is None:
                    start = t
                elif not lifted[t] and start is not None:
                    runs.append((start, t - 1))
                    start = None
            if start is not None:
                runs.append((start, T - 1))
            runs = [r for r in runs if r[1] - r[0] + 1 >= 5]
            if runs and (z_hi - z_lo) > 0.05:
                c0, c1 = max(runs, key=lambda r: r[1] - r[0])
            else:
                # Fallback 2: fixed fractions so every clip still has carry.
                c0, c1 = int(0.4 * T), int(0.6 * T)

        idx = np.arange(T)
        pickup = idx < c0
        carry = (idx >= c0) & (idx <= c1)
        place = idx > c1

        carry_walk = carry & (speed > 0.25)
        carry_stand = carry & ~carry_walk

        if task_phase_dim == TASK_PHASE_DIM_LEGACY:
            phase = np.zeros((T, TASK_PHASE_DIM_LEGACY), dtype=np.float32)
            phase[pickup] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            phase[place] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            phase[carry_stand | carry_walk] = np.array(
                [0.0, 0.0, 1.0, 0.0], dtype=np.float32
            )
            return phase

        phase = np.zeros((T, TASK_PHASE_DIM), dtype=np.float32)
        phase[pickup] = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        phase[place] = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        phase[carry_stand] = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        phase[carry_walk] = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return phase

    @classmethod
    def _derive_task_goals(
        cls,
        ts: np.ndarray,
        jp: np.ndarray,
        obj_rel: np.ndarray,
        pose_aa_root: np.ndarray,
        task_goal_frame: str = "current",
    ) -> np.ndarray:
        """Return per-frame [pickup_goal_rel, place_goal_rel, desired_hold_rel]."""
        if task_goal_frame not in {"current", "start"}:
            raise ValueError("task_goal_frame must be 'current' or 'start'")
        T = len(ts)
        pelvis_world = jp[:, PELVIS_IDX, :].astype(np.float32)
        obj_world = cls._heading_rel_to_world(obj_rel, pelvis_world, pose_aa_root)
        obj_world_z = obj_world[:, 2]

        obj_step = np.zeros(T, dtype=np.float32)
        if T > 1:
            obj_step[1:] = np.linalg.norm(np.diff(obj_rel, axis=0), axis=1)

        carry_mask = cls._carry_mask(ts, jp, obj_rel, pose_aa_root)
        desired_hold_rel = np.median(obj_rel[carry_mask], axis=0).astype(np.float32)

        stable = obj_step < 0.05
        low_thresh = np.percentile(obj_world_z, 40)
        low = obj_world_z <= low_thresh
        first = np.arange(T) < max(1, int(0.35 * T))
        last = np.arange(T) >= max(0, int(0.65 * T))

        def goal_world(mask: np.ndarray, fallback_slice: slice) -> np.ndarray:
            if mask.sum() >= 5:
                return np.median(obj_world[mask], axis=0).astype(np.float32)
            return np.median(obj_world[fallback_slice], axis=0).astype(np.float32)

        pickup_world = goal_world(first & low & stable, slice(0, max(1, int(0.15 * T))))
        place_world = goal_world(last & low & stable, slice(max(0, int(0.85 * T)), T))

        if task_goal_frame == "start":
            pickup_rel = cls._world_to_start_heading_rel(
                pickup_world, pelvis_world, pose_aa_root
            )
            place_rel = cls._world_to_start_heading_rel(
                place_world, pelvis_world, pose_aa_root
            )
        else:
            pickup_rel = cls._world_to_heading_rel(
                pickup_world, pelvis_world, pose_aa_root
            )
            place_rel = cls._world_to_heading_rel(
                place_world, pelvis_world, pose_aa_root
            )
        hold_rel = np.tile(desired_hold_rel[None], (T, 1))
        return np.concatenate([pickup_rel, place_rel, hold_rel], axis=1).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    def save_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @staticmethod
    def load_stats(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
