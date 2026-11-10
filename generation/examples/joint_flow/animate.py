"""
Skeleton visualizer / motion generator.

Clip mode — reconstruction vs ground truth:
    python animate.py --checkpoint ... --stats ... --hdf5 ... [--clip N] [--out out.gif]

Motion mode — synthetic generation with a named motion type:
    python animate.py --checkpoint ... --stats ... --motion walk_forward --speed 0.3 [--out out.gif]
    python animate.py --checkpoint ... --stats ... --motion walk_circle   --speed 0.5
    python animate.py --checkpoint ... --stats ... --motion walk_backward --speed 0.2
    python animate.py --checkpoint ... --stats ... --motion stand
    python animate.py --checkpoint ... --stats ... --motion crouch

Raw synthetic mode — constant arbitrary velocity:
    python animate.py --checkpoint ... --stats ... --vel-x 0.3 --vel-y 0.1 --height 0.9

Options common to all synthetic modes: --height --duration --fps --steps --smooth --out
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from dataset import JointFlowDataset, TASK_PHASE_DIM

from flow_matching.solver import ODESolver
from infer import CondWrapper
from model import CVAEPrior, JointFlowMLP
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


BONES = [
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),  # spine + neck/head
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # left leg
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # right leg
    (11, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),  # left arm
    (11, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),  # right arm
]
JOINT_NAMES = [
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


def bone_color(child_idx):
    name = JOINT_NAMES[child_idx]
    if name.startswith("L"):
        return "#4A90D9"
    if name.startswith("R"):
        return "#E05252"
    return "#888888"


def load_clip(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name = clips[clip_idx]
        grp = f[f"clips/{name}"]
        ts = grp["task_space"][:]  # (T, 9)
        jp = grp["joint_positions"][:]  # (T, 24, 3)
        fps = grp.attrs["fps"]
        pose_aa_root = (
            grp["pose_aa_root"][:].astype(np.float32)
            if "pose_aa_root" in grp
            else None
        )
        obj_rel = (
            grp["obj_com_pos_rel"][:].astype(np.float32)
            if "obj_com_pos_rel" in grp
            else None
        )
    print(f"Clip: {name}  |  {len(ts)} frames @ {fps:.0f} fps")
    return ts, jp, float(fps), name, obj_rel, pose_aa_root


def _build_linear_sentinel(omomo_hdf5, T):
    import h5py

    with h5py.File(omomo_hdf5, "r") as f:
        clips = list(f["clips"].keys())
        arrays = [
            f["clips"][c]["obj_com_pos_rel"][:]
            for c in clips
            if "obj_com_pos_rel" in f["clips"][c]
        ]
    max_obj = np.concatenate(arrays, axis=0).max(0).astype(np.float32)
    sentinel_end = max_obj + 0.1
    sentinel_start = sentinel_end * 2.0
    t_arr = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    return sentinel_start[None] + (sentinel_end - sentinel_start)[None] * t_arr


def _build_task_goal_sentinel(omomo_hdf5, T, ts_world=None):
    if omomo_hdf5 is None:
        max_obj = np.array([1.2, 0.5, 1.2], dtype=np.float32)
    else:
        with h5py.File(omomo_hdf5, "r") as f:
            clips = list(f["clips"].keys())
            arrays = [
                f["clips"][c]["obj_com_pos_rel"][:]
                for c in clips
                if "obj_com_pos_rel" in f["clips"][c]
            ]
        max_obj = np.concatenate(arrays, axis=0).max(0).astype(np.float32)
    sentinel_end = (max_obj + 0.1).astype(np.float32)
    sentinel_start = (sentinel_end * 2.0).astype(np.float32)
    t_arr = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    pickup = sentinel_start[None] + (sentinel_end - sentinel_start)[None] * t_arr
    place = pickup + sentinel_end[None]
    hold = sentinel_start[None].repeat(T, axis=0)
    if ts_world is not None:
        hold[:, 2] = -np.asarray(ts_world, dtype=np.float32)[:, 2]
    return np.concatenate([pickup, place, hold], axis=1).astype(np.float32)


def _make_obj_rel(ts_world, obj_wx, obj_wy, obj_wz, fps):
    """
    Convert a fixed world-frame obj position to per-frame pelvis-relative.
    Integrates root_vel to get pelvis XY, then subtracts per frame.
    """
    T = len(ts_world)
    dt = 1.0 / fps
    pelvis_xy = np.zeros((T, 2), dtype=np.float32)
    for t in range(1, T):
        pelvis_xy[t] = pelvis_xy[t - 1] + ts_world[t, :2] * dt
    obj_rel = np.empty((T, 3), dtype=np.float32)
    obj_rel[:, 0] = obj_wx - pelvis_xy[:, 0]
    obj_rel[:, 1] = obj_wy - pelvis_xy[:, 1]
    obj_rel[:, 2] = obj_wz - ts_world[:, 2]  # obj_world_z - pelvis_z
    return obj_rel


def _integrate_pelvis_xy(ts_world, fps, init_pelvis_xy=None):
    T = len(ts_world)
    dt = 1.0 / fps
    pelvis_xy = np.zeros((T, 2), dtype=np.float32)
    if init_pelvis_xy is not None:
        pelvis_xy[0] = init_pelvis_xy
    for t in range(1, T):
        pelvis_xy[t] = pelvis_xy[t - 1] + ts_world[t, :2] * dt
    return pelvis_xy


def _make_task_goals_from_world(
    ts_world,
    fps,
    pickup_world,
    place_world,
    hold_rel,
    init_pelvis_xy=None,
    task_goal_frame="current",
):
    if task_goal_frame not in {"current", "start"}:
        raise ValueError("task_goal_frame must be 'current' or 'start'")
    pelvis_xy = _integrate_pelvis_xy(ts_world, fps, init_pelvis_xy)
    pelvis = np.column_stack([pelvis_xy, ts_world[:, 2]]).astype(np.float32)
    if task_goal_frame == "start":
        start_pelvis = np.tile(pelvis[0:1], (len(ts_world), 1))
        pickup = np.asarray(pickup_world, dtype=np.float32)[None] - start_pelvis
        place = np.asarray(place_world, dtype=np.float32)[None] - start_pelvis
    else:
        pickup = np.asarray(pickup_world, dtype=np.float32)[None] - pelvis
        place = np.asarray(place_world, dtype=np.float32)[None] - pelvis
    hold = np.asarray(hold_rel, dtype=np.float32)[None].repeat(len(ts_world), axis=0)
    return np.concatenate([pickup, place, hold], axis=1).astype(np.float32)


def _load_clip_task_goals(hdf5_path, clip_idx, task_goal_frame="current"):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name = clips[clip_idx]
        grp = f[f"clips/{name}"]
        if "obj_com_pos_rel" not in grp:
            return None
        return JointFlowDataset._derive_task_goals(
            grp["task_space"][:].astype(np.float32),
            grp["joint_positions"][:].astype(np.float32),
            grp["obj_com_pos_rel"][:].astype(np.float32),
            grp["pose_aa_root"][:].astype(np.float32),
            task_goal_frame=task_goal_frame,
        )


def _load_clip_carry_phase(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name = clips[clip_idx]
        grp = f[f"clips/{name}"]
        if "obj_com_pos_rel" not in grp:
            return None
        return JointFlowDataset._derive_carry_phase(
            grp["task_space"][:].astype(np.float32),
            grp["joint_positions"][:].astype(np.float32),
            grp["obj_com_pos_rel"][:].astype(np.float32),
            grp["pose_aa_root"][:].astype(np.float32),
        )


def _load_clip_task_phase(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name = clips[clip_idx]
        grp = f[f"clips/{name}"]
        if "obj_com_pos_rel" not in grp:
            return None
        return JointFlowDataset._derive_task_phase(
            grp["task_space"][:].astype(np.float32),
            grp["joint_positions"][:].astype(np.float32),
            grp["obj_com_pos_rel"][:].astype(np.float32),
            grp["pose_aa_root"][:].astype(np.float32),
        )


def _task_phase_dim(stats):
    if not stats.get("use_task_phase", False):
        return 0
    return stats.get("task_phase_dim", 4)


def _motion_frame(stats):
    return stats.get("motion_frame", "world")


def _motion_repr(stats):
    return stats.get("motion_repr", "velocity")


def _motion_horizon_s(stats):
    return stats.get("motion_horizon_s", 0.5)


def _carry_walk_phase_index(stats):
    phase_dim = _task_phase_dim(stats)
    if phase_dim >= TASK_PHASE_DIM:
        return 3
    return 2


def _build_synthetic_task_phase(phase_dim, t0, t1, t2, t3, t4):
    task_phase = np.zeros((t4, phase_dim), dtype=np.float32)
    task_phase[:, 0] = 1.0
    if phase_dim >= TASK_PHASE_DIM:
        task_phase[t0:t1] = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        task_phase[t1:t2] = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        task_phase[t2:t3] = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        task_phase[t3:t4] = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        task_phase[t0:t1] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        task_phase[t1:t3] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        task_phase[t3:t4] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return task_phase


def _match_task_phase_to_stats(task_phase, stats):
    if task_phase is None or not stats.get("use_task_phase", False):
        return task_phase
    phase_dim = _task_phase_dim(stats)
    if task_phase.shape[1] == phase_dim:
        return task_phase
    if task_phase.shape[1] == TASK_PHASE_DIM and phase_dim == 4:
        merged = np.zeros((len(task_phase), 4), dtype=np.float32)
        merged[:, 0] = task_phase[:, 0]
        merged[:, 1] = task_phase[:, 1]
        merged[:, 2] = task_phase[:, 2] + task_phase[:, 3]
        merged[:, 3] = task_phase[:, 4]
        return merged
    if task_phase.shape[1] == 4 and phase_dim == TASK_PHASE_DIM:
        expanded = np.zeros((len(task_phase), TASK_PHASE_DIM), dtype=np.float32)
        expanded[:, 0] = task_phase[:, 0]
        expanded[:, 1] = task_phase[:, 1]
        expanded[:, 3] = task_phase[:, 2]
        expanded[:, 4] = task_phase[:, 3]
        return expanded
    raise ValueError(
        f"task_phase has dim {task_phase.shape[1]} but stats expect {phase_dim}"
    )


def _match_motion_frame(ts_world, stats, fps, pose_aa_root=None):
    ts_cond = ts_world.copy()
    if _motion_repr(stats) == "future_dxdy":
        ts_cond[:, :2] = JointFlowDataset._future_dxdy_from_world_vel(
            ts_cond[:, :2], fps, _motion_horizon_s(stats)
        )
    if _motion_frame(stats) != "start":
        return ts_cond
    if pose_aa_root is not None:
        ts_cond[:, :2] = JointFlowDataset._world_xy_to_start_heading_rel(
            ts_cond[:, :2], pose_aa_root
        )
    return ts_cond


def _task_goal_markers_world(task_goals, stats, ts_world, fps, pose_aa_root=None):
    if task_goals is None:
        return None, None
    task_goals = np.asarray(task_goals, dtype=np.float32)
    if task_goals.shape[1] < 6:
        return None, None

    pelvis_xy = _integrate_pelvis_xy(ts_world, fps)
    pelvis_world = np.column_stack([pelvis_xy, ts_world[:, 2]]).astype(np.float32)

    if stats.get("task_goal_frame", "current") == "start":
        if pose_aa_root is None:
            pose_aa_root = np.zeros((len(ts_world), 3), dtype=np.float32)
        start_pelvis = np.tile(pelvis_world[0:1], (len(ts_world), 1))
        start_root = np.tile(pose_aa_root[0:1], (len(ts_world), 1))
        pickup_world = JointFlowDataset._heading_rel_to_world(
            task_goals[:, 0:3], start_pelvis, start_root
        )
        place_world = JointFlowDataset._heading_rel_to_world(
            task_goals[:, 3:6], start_pelvis, start_root
        )
    else:
        if pose_aa_root is None:
            return None, None
        pickup_world = JointFlowDataset._heading_rel_to_world(
            task_goals[:, 0:3], pelvis_world, pose_aa_root
        )
        place_world = JointFlowDataset._heading_rel_to_world(
            task_goals[:, 3:6], pelvis_world, pose_aa_root
        )

    return (
        np.median(pickup_world, axis=0).astype(np.float32),
        np.median(place_world, axis=0).astype(np.float32),
    )


def _clip_goal_markers_world(ts_world, jp_gt, obj_rel, pose_aa_root):
    if obj_rel is None or jp_gt is None or pose_aa_root is None:
        return None, None
    pelvis_world = jp_gt[:, 0, :].astype(np.float32)
    obj_world = pelvis_world + obj_rel.astype(np.float32)
    obj_world_z = obj_world[:, 2]

    obj_step = np.zeros(len(ts_world), dtype=np.float32)
    if len(ts_world) > 1:
        obj_step[1:] = np.linalg.norm(np.diff(obj_rel, axis=0), axis=1)

    stable = obj_step < 0.05
    low_thresh = np.percentile(obj_world_z, 40)
    low = obj_world_z <= low_thresh
    first = np.arange(len(ts_world)) < max(1, int(0.35 * len(ts_world)))
    last = np.arange(len(ts_world)) >= max(0, int(0.65 * len(ts_world)))

    def goal_world(mask, fallback_slice):
        if mask.sum() >= 5:
            return np.median(obj_world[mask], axis=0).astype(np.float32)
        return np.median(obj_world[fallback_slice], axis=0).astype(np.float32)

    pickup_world = goal_world(first & low & stable, slice(0, max(1, int(0.15 * len(ts_world)))))
    place_world = goal_world(last & low & stable, slice(max(0, int(0.85 * len(ts_world))), len(ts_world)))
    return pickup_world, place_world


def make_task_space(motion, speed, fps, duration, height=0.938):
    """Build a synthetic (T, 9) task-space array for the given motion type."""
    T = int(duration * fps)
    ts = np.zeros((T, 9), dtype=np.float32)
    ts[:, 2] = height

    t_arr = np.arange(T, dtype=np.float32) / fps

    if motion == "walk_forward":
        ts[:, 0] = speed
    elif motion == "walk_backward":
        ts[:, 0] = -speed
    elif motion == "walk_circle":
        radius = max(speed * 3.0, 0.5)  # tighter circle at lower speeds
        omega = speed / radius
        ts[:, 0] = speed * np.cos(omega * t_arr)
        ts[:, 1] = speed * np.sin(omega * t_arr)
    elif motion == "stand":
        pass  # zero velocity
    elif motion == "crouch":
        half = T // 2
        crouch_z = height * 0.6
        ramp_down = np.linspace(height, crouch_z, half // 2)
        hold = np.full(half - half // 2, crouch_z)
        ramp_up = np.linspace(crouch_z, height, T - half)
        ts[:, 2] = np.concatenate([ramp_down, hold, ramp_up]).astype(np.float32)
    elif motion == "pick_place":
        # phases (fraction of total duration):
        #   0.30 — walk toward box
        #   0.10 — slow to stop
        #   0.15 — crouch down
        #   0.15 — hold crouched (pick up)
        #   0.15 — rise back up
        #   0.15 — walk away
        phases = [0.30, 0.10, 0.15, 0.15, 0.15, 0.15]
        ends = np.cumsum([int(p * T) for p in phases])
        ends[-1] = T  # ensure we fill exactly T frames
        t0, t1, t2, t3, t4, t5 = 0, ends[0], ends[1], ends[2], ends[3], ends[4]

        # velocity profile
        vel = np.zeros(T, dtype=np.float32)
        vel[t0:t1] = speed  # approach
        vel[t1:t2] = np.linspace(speed, 0.0, t2 - t1)  # slow down
        vel[t2:t3] = 0.0  # crouch
        vel[t3:t4] = 0.0  # hold
        vel[t4:t5] = 0.0  # rise
        vel[t5:T] = -speed  # walk away
        ts[:, 0] = vel

        # pelvis height profile
        crouch_z = height * 0.62
        h = np.full(T, height, dtype=np.float32)
        h[t2:t3] = np.linspace(height, crouch_z, t3 - t2)
        h[t3:t4] = crouch_z
        h[t4:t5] = np.linspace(crouch_z, height, t5 - t4)
        ts[:, 2] = h
    else:
        raise ValueError(f"Unknown motion type: {motion}")

    return ts


def run_flow(
    model,
    stats,
    ts_world,
    window,
    device,
    steps=20,
    fps=30.0,
    init_pelvis_xy=None,
    prior=None,
    obj_rel=None,
    obj_sentinel=None,
    task_goals=None,
    carry_phase=None,
    task_phase=None,
    pose_aa_root=None,
):
    """
    Autoregressively predict joint positions for every frame.
    ts_world: (T, 9) world-frame task-space
    obj_rel: (T, 3) object CoM relative to pelvis, or None to use sentinel
    obj_sentinel: (T, 3) explicit sentinel array for locomotion (no-object) clips; overrides
                  the fixed (2,2,2) default. Use a linearly decreasing ramp to match training.
    Returns: (T, 24, 3) predicted joint positions (world-frame)
    """
    T = len(ts_world)
    ts_cond_world = _match_motion_frame(ts_world, stats, fps, pose_aa_root)
    use_prev = stats.get("use_prev_frame", False)
    no_eef = stats.get("no_eef", False)
    use_obj = stats.get("use_obj", False)
    use_task_goals = stats.get("use_task_goals", False)
    use_carry_phase = stats.get("use_carry_phase", False)
    use_task_phase = stats.get("use_task_phase", False)
    phase_dim = _task_phase_dim(stats)
    ts_dim = (
        (3 if no_eef else 9)
        + (3 if use_obj else 0)
        + (9 if use_task_goals else 0)
        + (1 if use_carry_phase else 0)
        + phase_dim
    )
    cond_mean = np.array(stats["cond_mean"], dtype=np.float32)
    cond_std = np.array(stats["cond_std"], dtype=np.float32)
    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)
    joint_std = np.array(stats["joint_std"], dtype=np.float32)

    dt = 1.0 / fps
    pelvis_xy = np.zeros((T, 2), dtype=np.float32)
    if init_pelvis_xy is not None:
        pelvis_xy[0] = init_pelvis_xy
    for t in range(1, T):
        pelvis_xy[t] = pelvis_xy[t - 1] + ts_world[t, :2] * dt

    if use_obj:
        if obj_rel is not None:
            obj_arr = obj_rel
        elif obj_sentinel is not None:
            obj_arr = obj_sentinel
        else:
            obj_arr = np.tile(np.array([2.0, 2.0, 2.0], dtype=np.float32), (T, 1))

    if use_task_goals:
        if task_goals is None:
            task_goals = _build_task_goal_sentinel(None, T, ts_world)
        task_goals = task_goals.astype(np.float32)
    if use_carry_phase:
        if carry_phase is None:
            carry_phase = np.zeros((T, 1), dtype=np.float32)
        carry_phase = carry_phase.astype(np.float32)
    if use_task_phase:
        if task_phase is None:
            task_phase = np.zeros((T, phase_dim), dtype=np.float32)
            task_phase[:, 0] = 1.0
        task_phase = task_phase.astype(np.float32)

    cond_norm = np.empty((T, ts_dim), dtype=np.float32)
    for t in range(T):
        f = ts_cond_world[t].copy()
        if not no_eef:
            ph = np.array([pelvis_xy[t, 0], pelvis_xy[t, 1], f[2]], dtype=np.float32)
            f[3:6] -= ph
            f[6:9] -= ph
        base = f[:3] if no_eef else f[:9]
        row_parts = [base]
        if use_obj:
            row_parts.append(obj_arr[t])
        if use_task_goals:
            row_parts.append(task_goals[t])
        if use_carry_phase:
            row_parts.append(carry_phase[t])
        if use_task_phase:
            row_parts.append(task_phase[t])
        row = np.concatenate(row_parts)
        cond_norm[t] = (row - cond_mean) / cond_std

    predictions = np.empty((T, 24, 3), dtype=np.float32)
    prev_joints_norm = np.zeros(69, dtype=np.float32)

    for t in range(T):
        start = t - window + 1
        if start < 0:
            pad = cond_norm[0:1].repeat(-start, axis=0)
            win = np.concatenate([pad, cond_norm[0 : t + 1]], axis=0)
        else:
            win = cond_norm[start : t + 1]

        if use_prev:
            cond_vec = np.concatenate([prev_joints_norm, win.reshape(-1)])
        else:
            cond_vec = win.reshape(-1)
        cond_flat = torch.tensor(cond_vec, device=device).unsqueeze(0)

        if prior is not None:
            with torch.no_grad():
                mu, _ = prior(cond_flat)
            cond_flat = mu

        x0 = torch.randn(1, 69, device=device)
        wrapped = CondWrapper(model, cond_flat)
        with torch.no_grad():
            solver = ODESolver(velocity_model=wrapped)
            x1 = solver.sample(
                x_init=x0,
                step_size=1.0 / steps,
                method="midpoint",
                time_grid=torch.linspace(0, 1, steps + 1, device=device),
                return_intermediates=False,
            )
        x1_np = np.clip(x1.cpu().numpy()[0], -10, 10)

        if use_prev:
            prev_joints_norm = x1_np

        joints_23 = (x1_np * joint_std + joint_mean).reshape(23, 3)
        joints_rel = np.concatenate(
            [np.zeros((1, 3), dtype=np.float32), joints_23], axis=0
        )
        pelvis_world = np.array(
            [pelvis_xy[t, 0], pelvis_xy[t, 1], ts_world[t, 2]], dtype=np.float32
        )
        predictions[t] = joints_rel + pelvis_world

        if t % 50 == 0:
            print(f"  frame {t}/{T}")

    return predictions


def run_flow_6d(
    model,
    stats,
    ts_world,
    window,
    device,
    steps=20,
    fps=30.0,
    init_pelvis_xy=None,
    obj_rel=None,
    obj_sentinel=None,
    task_goals=None,
    carry_phase=None,
    task_phase=None,
    pose_aa_root=None,
):
    """
    Like run_flow but for 6D-output models (use_6d=True in stats).
    Returns (T, 72) pose_aa and (T, 3) root_trans instead of joint positions.
    pose_aa[:, 0:3] = root axis-angle from velocity direction.
    pose_aa[:, 3:72] = 23 body joints from 6D model output.
    obj_rel: (T, 3) object CoM relative to pelvis, or None to use sentinel
    obj_sentinel: (T, 3) explicit sentinel array; overrides fixed (2,2,2) default
    """
    from rot_utils import rot6d_to_pose_aa, velocity_to_root_rotvec

    T = len(ts_world)
    ts_cond_world = _match_motion_frame(ts_world, stats, fps, pose_aa_root)
    use_prev = stats.get("use_prev_frame", False)
    no_eef = stats.get("no_eef", False)
    use_obj = stats.get("use_obj", False)
    use_task_goals = stats.get("use_task_goals", False)
    use_carry_phase = stats.get("use_carry_phase", False)
    use_task_phase = stats.get("use_task_phase", False)
    predict_root = stats.get("predict_root", False)
    phase_dim = _task_phase_dim(stats)
    ts_dim = (
        (3 if no_eef else 9)
        + (3 if use_obj else 0)
        + (9 if use_task_goals else 0)
        + (1 if use_carry_phase else 0)
        + phase_dim
    )
    cond_mean = np.array(stats["cond_mean"], dtype=np.float32)
    cond_std = np.array(stats["cond_std"], dtype=np.float32)
    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)  # (138,) or (144,)
    joint_std = np.array(stats["joint_std"], dtype=np.float32)  # (138,) or (144,)

    dt = 1.0 / fps
    pelvis_xy = np.zeros((T, 2), dtype=np.float32)
    if init_pelvis_xy is not None:
        pelvis_xy[0] = init_pelvis_xy
    for t in range(1, T):
        pelvis_xy[t] = pelvis_xy[t - 1] + ts_world[t, :2] * dt

    if use_obj:
        if obj_rel is not None:
            obj_arr = obj_rel
        elif obj_sentinel is not None:
            obj_arr = obj_sentinel
        else:
            obj_arr = np.tile(np.array([2.0, 2.0, 2.0], dtype=np.float32), (T, 1))

    if use_task_goals:
        if task_goals is None:
            task_goals = _build_task_goal_sentinel(None, T, ts_world)
        task_goals = task_goals.astype(np.float32)
    if use_carry_phase:
        if carry_phase is None:
            carry_phase = np.zeros((T, 1), dtype=np.float32)
        carry_phase = carry_phase.astype(np.float32)
    if use_task_phase:
        if task_phase is None:
            task_phase = np.zeros((T, phase_dim), dtype=np.float32)
            task_phase[:, 0] = 1.0
        task_phase = task_phase.astype(np.float32)

    cond_norm = np.empty((T, ts_dim), dtype=np.float32)
    for t in range(T):
        f = ts_cond_world[t].copy()
        if not no_eef:
            ph = np.array([pelvis_xy[t, 0], pelvis_xy[t, 1], f[2]], dtype=np.float32)
            f[3:6] -= ph
            f[6:9] -= ph
        base = f[:3] if no_eef else f[:9]
        row_parts = [base]
        if use_obj:
            row_parts.append(obj_arr[t])
        if use_task_goals:
            row_parts.append(task_goals[t])
        if use_carry_phase:
            row_parts.append(carry_phase[t])
        if use_task_phase:
            row_parts.append(task_phase[t])
        row = np.concatenate(row_parts)
        cond_norm[t] = (row - cond_mean) / cond_std

    # smooth velocity for root rotation — EMA prevents instant body flips near zero speed
    alpha = 0.2
    vel_smooth = np.empty((T, 2), dtype=np.float32)
    vel_smooth[0] = ts_world[0, :2]
    for t in range(1, T):
        vel_smooth[t] = alpha * ts_world[t, :2] + (1 - alpha) * vel_smooth[t - 1]

    joint_out_dim = 144 if predict_root else 138
    pose_aa = np.empty((T, 72), dtype=np.float32)
    root_trans = np.empty((T, 3), dtype=np.float32)
    prev_joints_norm = np.zeros(joint_out_dim, dtype=np.float32)

    for t in range(T):
        start = t - window + 1
        if start < 0:
            pad = cond_norm[0:1].repeat(-start, axis=0)
            win = np.concatenate([pad, cond_norm[0 : t + 1]], axis=0)
        else:
            win = cond_norm[start : t + 1]

        if use_prev:
            cond_vec = np.concatenate([prev_joints_norm, win.reshape(-1)])
        else:
            cond_vec = win.reshape(-1)
        cond_flat = torch.tensor(cond_vec, device=device).unsqueeze(0)

        x0 = torch.randn(1, joint_out_dim, device=device)
        wrapped = CondWrapper(model, cond_flat)
        with torch.no_grad():
            solver = ODESolver(velocity_model=wrapped)
            x1 = solver.sample(
                x_init=x0,
                step_size=1.0 / steps,
                method="midpoint",
                time_grid=torch.linspace(0, 1, steps + 1, device=device),
                return_intermediates=False,
            )
        x1_np = np.clip(x1.cpu().numpy()[0], -10, 10)
        if use_prev:
            prev_joints_norm = x1_np

        denormed = x1_np * joint_std + joint_mean
        if predict_root:
            from rot_utils import rot6d_to_matrix
            from scipy.spatial.transform import Rotation as _sRot2

            root_R = rot6d_to_matrix(denormed[:6].reshape(1, 6))[0]  # (3,3)
            root_aa = _sRot2.from_matrix(root_R).as_rotvec().astype(np.float32)
            body_aa = rot6d_to_pose_aa(denormed[6:][None])[0]  # (69,)
        else:
            root_aa = velocity_to_root_rotvec(vel_smooth[t, 0], vel_smooth[t, 1])
            body_aa = rot6d_to_pose_aa(denormed[None])[0]  # (69,)

        pose_aa[t, 0:3] = root_aa
        pose_aa[t, 3:72] = body_aa
        root_trans[t] = [pelvis_xy[t, 0], pelvis_xy[t, 1], ts_world[t, 2]]

        if t % 50 == 0:
            print(f"  frame {t}/{T}")

    return pose_aa, root_trans


def draw_bones(ax, joints, alpha=1.0, lw=2):
    lines = []
    for p, c in BONES:
        (l,) = ax.plot(
            [joints[p, 0], joints[c, 0]],
            [joints[p, 1], joints[c, 1]],
            [joints[p, 2], joints[c, 2]],
            color=bone_color(c),
            linewidth=lw,
            alpha=alpha,
        )
        lines.append(l)
    pts = ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2], c="black", s=15, alpha=alpha, zorder=5
    )
    return lines, pts


def set_skeleton_colors(lines, pts, carry_on):
    if carry_on:
        color = "#18A558"
        for ln in lines:
            ln.set_color(color)
        pts.set_color(color)
        return
    for (p, c), ln in zip(BONES, lines):
        ln.set_color(bone_color(c))
    pts.set_color("black")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)

    # clip mode
    p.add_argument("--hdf5", default=None, help="HDF5 file (required for clip mode)")
    p.add_argument("--clip", type=int, default=0, help="Clip index within HDF5")

    # motion mode
    p.add_argument(
        "--motion",
        default=None,
        choices=[
            "walk_forward",
            "walk_backward",
            "walk_circle",
            "stand",
            "crouch",
            "pick_place",
            "pick_carry_place",
        ],
        help="Named synthetic motion type",
    )
    p.add_argument(
        "--speed", type=float, default=0.3, help="Speed in m/s (motion mode)"
    )

    # raw synthetic mode
    p.add_argument(
        "--vel-x",
        type=float,
        default=None,
        help="Constant forward velocity (raw synthetic)",
    )
    p.add_argument(
        "--vel-y",
        type=float,
        default=0.0,
        help="Constant lateral velocity (raw synthetic)",
    )

    # shared synthetic options
    p.add_argument(
        "--height",
        type=float,
        default=0.938,
        help="Pelvis height in m (synthetic modes)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds (synthetic modes)",
    )

    # synthetic obj conditioning — world-frame position; pelvis-relative computed per frame
    # omit all three to use the sentinel (no-object token = locomotion mode)
    p.add_argument(
        "--obj-x",
        type=float,
        default=None,
        help="Object world X (m); omit for sentinel/locomotion",
    )
    p.add_argument(
        "--obj-y",
        type=float,
        default=None,
        help="Object world Y (m, forward); omit for sentinel/locomotion",
    )
    p.add_argument(
        "--obj-z",
        type=float,
        default=None,
        help="Object world Z (m, height above floor); omit for sentinel/locomotion",
    )

    # task-goal conditioning for --use-task-goals models. Coordinates are world-frame
    # for pickup/place and pelvis-heading-frame relative for hold.
    p.add_argument("--pickup-x", type=float, default=None)
    p.add_argument("--pickup-y", type=float, default=None)
    p.add_argument("--pickup-z", type=float, default=None)
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--hold-x", type=float, default=0.29)
    p.add_argument("--hold-y", type=float, default=0.09)
    p.add_argument("--hold-z", type=float, default=0.13)

    # sentinel mode for use_obj models in locomotion (no object)
    p.add_argument(
        "--linear-sentinel",
        action="store_true",
        help="Use linearly decreasing obj sentinel matching training, instead of fixed (2,2,2)",
    )
    p.add_argument(
        "--omomo-hdf5",
        default=None,
        help="Path to OMOMO HDF5 to compute sentinel bounds; required with --linear-sentinel",
    )

    # shared
    p.add_argument("--steps", type=int, default=20, help="ODE steps per frame")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma over time (0=off)",
    )
    p.add_argument("--out", default=None, help=".gif or .mp4; omit for interactive")
    p.add_argument(
        "--highlight-carry",
        action="store_true",
        help="render skeletons green on carry/carry-walk frames",
    )
    p.add_argument(
        "--humanoid-xml",
        default=None,
        help="Path to humanoid XML (required for 6D models)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)
    window = stats.get("window", 10)
    task_goal_frame = stats.get("task_goal_frame", "current")

    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    use_6d_ = stats.get("use_6d", False)
    predict_root_ = stats.get("predict_root", False)
    _6d_dim = (144 if predict_root_ else 138) if use_6d_ else 69
    prev_dim = _6d_dim if stats.get("use_prev_frame", False) else 0
    phase_dim = _task_phase_dim(stats)
    ts_dim = (
        (3 if stats.get("no_eef", False) else 9)
        + (3 if stats.get("use_obj", False) else 0)
        + (9 if stats.get("use_task_goals", False) else 0)
        + (1 if stats.get("use_carry_phase", False) else 0)
        + phase_dim
    )
    use_cvae = train_args.get("use_cvae", False)
    latent_dim = train_args.get("latent_dim", 64)
    flow_cond_dim = latent_dim if use_cvae else prev_dim + window * ts_dim

    use_6d = stats.get("use_6d", False)
    joint_dim = _6d_dim if use_6d else 69

    model = JointFlowMLP(
        joint_dim=joint_dim,
        cond_dim=flow_cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if use_cvae:
        prior = CVAEPrior(prev_dim + window * ts_dim, latent_dim).to(device)
        prior.load_state_dict(ckpt["prior"])
        prior.eval()
    else:
        prior = None

    # --- determine mode ---
    obj_rel = None  # set for clip mode if obj data available
    task_goals = None
    carry_phase = None
    task_phase = None
    pose_aa_root = None
    if args.motion == "pick_carry_place" and (
        stats.get("use_obj") or stats.get("use_task_goals")
    ):
        # Phase durations from OMOMO statistics (mean):
        #   pickup:       0.82s  speed~0.22  pelvis_z~0.83  obj_rel~(0.43,0.05,-0.34)
        #   carry_stand:  0.72s  speed~0.05  pelvis_z~0.98  obj_rel~(0.11,0.06,+0.36)
        #   carry_walk:   1.50s  speed~0.64  pelvis_z~0.98  obj_rel~(0.32,0.05,+0.13)
        #   place:        1.15s  speed~0.22  pelvis_z~0.91  obj_rel~(0.45,0.03,-0.31)
        #
        # obj_rel convention: heading-canonicalized (x=forward, y=left, z=up), relative to pelvis.
        # pickup: smooth interpolation from standing-near-box to carry position.
        # place: smooth interpolation from carry position back to placed-on-ground.
        # Pelvis dips to 0.70m (deeper than mean min of 0.755) at ~40% through each phase,
        # coordinated with the box lift so the model sees realistic co-variation.
        fps_ = args.fps
        carry_spd = args.speed if args.speed != 0.3 else 0.64
        dur_pu = 0.82
        dur_cs = 0.72
        dur_cw = 1.50
        dur_pl = 1.15
        total_dur = dur_pu + dur_cs + dur_cw + dur_pl  # 4.19s
        scale = args.duration / total_dur
        n_pu = max(4, int(dur_pu * scale * fps_))
        n_cs = max(2, int(dur_cs * scale * fps_))
        n_cw = max(4, int(dur_cw * scale * fps_))
        n_pl = max(4, int(dur_pl * scale * fps_))
        T = n_pu + n_cs + n_cw + n_pl

        t0 = 0
        t1 = n_pu
        t2 = t1 + n_cs
        t3 = t2 + n_cw
        t4 = T

        # --- velocity ---
        vel = np.zeros(T, dtype=np.float32)
        vel[t0:t1] = 0.10  # slow approach / stop
        vel[t1:t2] = 0.0  # stand still after picking up
        ramp = max(1, n_cw // 6)
        vel[t2 : t2 + ramp] = np.linspace(0, carry_spd, ramp)
        vel[t2 + ramp : t3 - ramp] = carry_spd
        vel[t3 - ramp : t3] = np.linspace(carry_spd, 0, ramp)
        # Add gait-like speed oscillation to carry_walk (training windows never have flat velocity)
        t_cw = np.arange(n_cw, dtype=np.float32)
        gait_freq = 1.8  # ~1.8 Hz gait cycle at 30fps
        noise = 0.06 * np.sin(2 * np.pi * gait_freq * t_cw / fps_) + \
                0.03 * np.sin(2 * np.pi * 2 * gait_freq * t_cw / fps_)
        vel[t2:t3] = np.clip(vel[t2:t3] + noise, 0.05, carry_spd * 1.3).astype(np.float32)
        vel[t3:t4] = 0.10  # slow approach to set-down spot

        # --- pelvis height ---
        # Pickup: dip to 0.70 at 40% through (reaching for box), rise back to 0.98 by end.
        # Place:  mirror of pickup.
        pz = np.full(T, args.height, dtype=np.float32)
        crouch_z = 0.70
        pu_dip = max(1, int(0.4 * n_pu))
        pz[t0 : t0 + pu_dip] = np.linspace(args.height, crouch_z, pu_dip)
        pz[t0 + pu_dip : t1] = np.linspace(crouch_z, 0.98, n_pu - pu_dip)
        pz[t1:t3] = 0.98
        pl_dip = max(1, int(0.4 * n_pl))
        pz[t3 : t3 + pl_dip] = np.linspace(0.98, crouch_z, pl_dip)
        pz[t3 + pl_dip : t4] = np.linspace(crouch_z, args.height, n_pl - pl_dip)

        # Add lateral velocity during carry_walk — 97.6% of training carry_walk frames have |vy|>0.05
        vy = np.zeros(T, dtype=np.float32)
        t_cw = np.arange(n_cw, dtype=np.float32)
        vy[t2:t3] = 0.15 * np.sin(2 * np.pi * 0.4 * t_cw / fps_)  # slow lateral sway

        ts_world = np.zeros((T, 9), dtype=np.float32)
        ts_world[:, 0] = vel
        ts_world[:, 1] = vy
        ts_world[:, 2] = pz
        carry_phase = np.zeros((T, 1), dtype=np.float32)
        carry_phase[t1:t3, 0] = 1.0
        task_phase = _build_synthetic_task_phase(phase_dim, t0, t1, t2, t3, t4)
        pelvis_xy = _integrate_pelvis_xy(ts_world, fps_)
        pelvis = np.column_stack([pelvis_xy, ts_world[:, 2]]).astype(np.float32)

        custom_pickup = all(
            v is not None for v in [args.pickup_x, args.pickup_y, args.pickup_z]
        )
        custom_place = all(
            v is not None for v in [args.place_x, args.place_y, args.place_z]
        )
        custom_obj = all(v is not None for v in [args.obj_x, args.obj_y, args.obj_z])

        # --- obj_rel: smooth interpolation between OMOMO phase means ---
        # No hard jumps — each transition phase interpolates continuously.
        # pickup: person standing near box on floor → box lifted to carry_stand position
        pu_pre = np.array(
            [0.43, 0.05, -0.49], dtype=np.float32
        )  # standing, box at floor (~0.49-0.98=-0.49)
        cs_rel = np.array(
            [0.11, 0.06, 0.36], dtype=np.float32
        )  # carry_stand (OMOMO mean)
        cw_rel = np.array(
            [0.32, 0.05, 0.13], dtype=np.float32
        )  # carry_walk  (OMOMO mean)
        pl_post = np.array(
            [0.45, 0.03, -0.31], dtype=np.float32
        )  # placed on ground (OMOMO mean)

        obj_rel = np.empty((T, 3), dtype=np.float32)
        if custom_pickup or custom_place or custom_obj:
            pickup_world = np.array(
                [
                    args.pickup_x if custom_pickup else (args.obj_x if custom_obj else 0.43),
                    args.pickup_y if custom_pickup else (args.obj_y if custom_obj else 0.05),
                    args.pickup_z if custom_pickup else (args.obj_z if custom_obj else 0.49),
                ],
                dtype=np.float32,
            )
            place_world = np.array(
                [
                    args.place_x if custom_place else pelvis_xy[-1, 0] + 0.45,
                    args.place_y if custom_place else pelvis_xy[-1, 1] + 0.03,
                    args.place_z if custom_place else args.height - 0.31,
                ],
                dtype=np.float32,
            )
            hold_rel = np.array(
                [args.hold_x, args.hold_y, args.hold_z], dtype=np.float32
            )
            pickup_rel = pickup_world[None] - pelvis
            place_rel = place_world[None] - pelvis
            alpha_pu = np.linspace(0, 1, n_pu, dtype=np.float32)[:, None]
            alpha_pl = np.linspace(0, 1, n_pl, dtype=np.float32)[:, None]
            obj_rel[t0:t1] = (1 - alpha_pu) * pickup_rel[t0:t1] + alpha_pu * hold_rel
            obj_rel[t1:t3] = hold_rel
            obj_rel[t3:t4] = (1 - alpha_pl) * hold_rel + alpha_pl * place_rel[t3:t4]
        else:
            # pickup: linear from pu_pre → cs_rel
            alpha_pu = np.linspace(0, 1, n_pu, dtype=np.float32)[:, None]
            obj_rel[t0:t1] = (1 - alpha_pu) * pu_pre + alpha_pu * cs_rel
            # carry_stand: constant
            obj_rel[t1:t2] = cs_rel
            # carry_walk: constant
            obj_rel[t2:t3] = cw_rel
            # place: linear from cs_rel → pl_post
            alpha_pl = np.linspace(0, 1, n_pl, dtype=np.float32)[:, None]
            obj_rel[t3:t4] = (1 - alpha_pl) * cs_rel + alpha_pl * pl_post

        if stats.get("use_task_goals"):
            if not (custom_pickup or custom_place or custom_obj):
                pickup_world = np.array([0.43, 0.05, 0.49], dtype=np.float32)
                place_world = np.array(
                    [
                        pelvis_xy[-1, 0] + 0.45,
                        pelvis_xy[-1, 1] + 0.03,
                        args.height - 0.31,
                    ],
                    dtype=np.float32,
                )
            hold_rel = np.array(
                [args.hold_x, args.hold_y, args.hold_z], dtype=np.float32
            )
            task_goals = _make_task_goals_from_world(
                ts_world,
                fps_,
                pickup_world,
                place_world,
                hold_rel,
                task_goal_frame=task_goal_frame,
            )

        jp_gt = None
        anim_fps = fps_
        clip_name = (
            f"pick_carry_place  carry_spd={carry_spd:.2f}  dur={args.duration:.1f}s"
        )
        print(f"Generating 'pick_carry_place'  ({T} frames, {T/fps_:.1f}s) …")
    elif args.motion is not None:
        # motion mode: named synthetic generation
        ts_world = make_task_space(
            args.motion, args.speed, args.fps, args.duration, args.height
        )
        jp_gt = None
        clip_name = f"{args.motion}  speed={args.speed:.2f} m/s  h={args.height:.2f}"
        anim_fps = args.fps
        if stats.get("use_obj") and all(
            v is not None for v in [args.obj_x, args.obj_y, args.obj_z]
        ):
            obj_rel = _make_obj_rel(
                ts_world, args.obj_x, args.obj_y, args.obj_z, args.fps
            )
            clip_name += (
                f"  obj_world=({args.obj_x:.2f},{args.obj_y:.2f},{args.obj_z:.2f})"
            )
        elif stats.get("use_obj"):
            clip_name += "  obj=sentinel(loco)"
        if stats.get("use_task_goals") and all(
            v is not None
            for v in [
                args.pickup_x,
                args.pickup_y,
                args.pickup_z,
                args.place_x,
                args.place_y,
                args.place_z,
            ]
        ):
            task_goals = _make_task_goals_from_world(
                ts_world,
                args.fps,
                np.array(
                    [args.pickup_x, args.pickup_y, args.pickup_z], dtype=np.float32
                ),
                np.array([args.place_x, args.place_y, args.place_z], dtype=np.float32),
                np.array([args.hold_x, args.hold_y, args.hold_z], dtype=np.float32),
                task_goal_frame=task_goal_frame,
            )
            clip_name += "  task_goals=world"
        print(
            f"Generating '{args.motion}' @ {args.speed} m/s  ({len(ts_world)} frames) …"
        )
    elif args.vel_x is not None:
        # raw synthetic mode: constant arbitrary velocity
        T = int(args.duration * args.fps)
        ts_world = np.zeros((T, 9), dtype=np.float32)
        ts_world[:, 0] = args.vel_x
        ts_world[:, 1] = args.vel_y
        ts_world[:, 2] = args.height
        jp_gt = None
        clip_name = (
            f"synthetic  vel=({args.vel_x:.2f},{args.vel_y:.2f})  h={args.height:.2f}"
        )
        anim_fps = args.fps
        if stats.get("use_obj") and all(
            v is not None for v in [args.obj_x, args.obj_y, args.obj_z]
        ):
            obj_rel = _make_obj_rel(
                ts_world, args.obj_x, args.obj_y, args.obj_z, args.fps
            )
            clip_name += (
                f"  obj_world=({args.obj_x:.2f},{args.obj_y:.2f},{args.obj_z:.2f})"
            )
        elif stats.get("use_obj"):
            clip_name += "  obj=sentinel(loco)"
        if stats.get("use_task_goals") and all(
            v is not None
            for v in [
                args.pickup_x,
                args.pickup_y,
                args.pickup_z,
                args.place_x,
                args.place_y,
                args.place_z,
            ]
        ):
            task_goals = _make_task_goals_from_world(
                ts_world,
                args.fps,
                np.array(
                    [args.pickup_x, args.pickup_y, args.pickup_z], dtype=np.float32
                ),
                np.array([args.place_x, args.place_y, args.place_z], dtype=np.float32),
                np.array([args.hold_x, args.hold_y, args.hold_z], dtype=np.float32),
                task_goal_frame=task_goal_frame,
            )
            clip_name += "  task_goals=world"
        print(f"Generating raw synthetic ({len(ts_world)} frames) …")
    else:
        # clip mode: reconstruct from real data
        if args.hdf5 is None:
            raise ValueError(
                "Provide --hdf5 for clip mode, --motion for motion mode, or --vel-x for raw synthetic"
            )
        ts_world, jp_gt, clip_fps, clip_name, obj_rel, pose_aa_root = load_clip(
            args.hdf5, args.clip
        )
        if stats.get("use_task_goals"):
            task_goals = _load_clip_task_goals(
                args.hdf5, args.clip, task_goal_frame=task_goal_frame
            )
        if stats.get("use_carry_phase"):
            carry_phase = _load_clip_carry_phase(args.hdf5, args.clip)
        if stats.get("use_task_phase"):
            task_phase = _match_task_phase_to_stats(
                _load_clip_task_phase(args.hdf5, args.clip), stats
            )
        anim_fps = args.fps if args.fps > 0 else clip_fps

    # build sentinels if requested and no real object/task data
    obj_sentinel = None
    if args.linear_sentinel and stats.get("use_obj") and obj_rel is None:
        if args.omomo_hdf5 is None:
            raise ValueError("--omomo-hdf5 is required when --linear-sentinel is set")
        obj_sentinel = _build_linear_sentinel(args.omomo_hdf5, len(ts_world))
    if stats.get("use_task_goals") and task_goals is None:
        task_goals = _build_task_goal_sentinel(args.omomo_hdf5, len(ts_world), ts_world)
    if stats.get("use_carry_phase") and carry_phase is None:
        carry_phase = np.zeros((len(ts_world), 1), dtype=np.float32)
    if stats.get("use_task_phase") and task_phase is None:
        task_phase = np.zeros((len(ts_world), phase_dim), dtype=np.float32)
        task_phase[:, 0] = 1.0
    task_phase = _match_task_phase_to_stats(task_phase, stats)

    print("Running flow inference …")
    if use_6d:
        if args.humanoid_xml is None:
            raise ValueError("--humanoid-xml is required for 6D models")
        import sys as _sys

        _phc_root = str(Path(__file__).resolve().parents[3] / "tracking")
        _sys.path.insert(0, str(Path(_phc_root) / "poselib"))
        from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
        from scipy.spatial.transform import Rotation as _sRot

        sk_tree = SkeletonTree.from_mjcf(args.humanoid_xml)
        pose_aa_np, root_trans_np = run_flow_6d(
            model,
            stats,
            ts_world,
            window,
            device,
            steps=args.steps,
            fps=anim_fps,
            init_pelvis_xy=None if jp_gt is None else jp_gt[0, 0, :2],
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
            carry_phase=carry_phase,
            task_phase=task_phase,
            pose_aa_root=pose_aa_root,
        )
        T6 = len(pose_aa_np)
        quats = _sRot.from_rotvec(pose_aa_np.reshape(-1, 3)).as_quat()  # xyzw
        quats_t = torch.from_numpy(quats).float().reshape(T6, 24, 4)
        rt_t = torch.from_numpy(root_trans_np).float()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree, quats_t, rt_t, is_local=True
        )
        pred = sk_state.global_translation.numpy()  # (T, 24, 3)
    else:
        pred = run_flow(
            model,
            stats,
            ts_world,
            window,
            device,
            steps=args.steps,
            fps=anim_fps,
            init_pelvis_xy=None if jp_gt is None else jp_gt[0, 0, :2],
            prior=prior,
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
            carry_phase=carry_phase,
            task_phase=task_phase,
            pose_aa_root=pose_aa_root,
        )
    if args.smooth > 0:
        from scipy.ndimage import gaussian_filter1d

        pred = gaussian_filter1d(pred, sigma=args.smooth, axis=0)
    print("Done.")

    carry_highlight = np.zeros(len(pred), dtype=bool)
    if args.highlight_carry:
        if task_phase is not None:
            carry_highlight = task_phase[:, _carry_walk_phase_index(stats)] > 0.5
        elif carry_phase is not None:
            carry_highlight = carry_phase[:, 0] > 0.5
    pickup_marker, place_marker = _clip_goal_markers_world(
        ts_world, jp_gt, obj_rel, pose_aa_root
    )
    if pickup_marker is None or place_marker is None:
        pickup_marker, place_marker = _task_goal_markers_world(
            task_goals, stats, ts_world, anim_fps, pose_aa_root
        )

    interval = 1000.0 / anim_fps

    if jp_gt is None:
        # single-panel synthetic view
        fig = plt.figure(figsize=(7, 7))
        fig.suptitle(clip_name, fontsize=9)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(0, 2.0)
        ax.view_init(elev=15, azim=-60)
        lines_h, pts_h = draw_bones(ax, pred[0])
        if pickup_marker is not None:
            ax.scatter(
                [pickup_marker[0]], [pickup_marker[1]], [pickup_marker[2]],
                c="#39b54a", s=45, depthshade=False
            )
        if place_marker is not None:
            ax.scatter(
                [place_marker[0]], [place_marker[1]], [place_marker[2]],
                c="#d94343", s=45, depthshade=False
            )

        # for motions with translation, use fixed axes so walking is visible
        _traj_x = pred[:, 0, 0]
        _traj_y = pred[:, 0, 1]
        _x_range = _traj_x.max() - _traj_x.min()
        _y_range = _traj_y.max() - _traj_y.min()
        _use_fixed = max(_x_range, _y_range) > 0.5  # walking motion
        if _use_fixed:
            _pad = 1.0
            _xl = (_traj_x.min() - _pad, _traj_x.max() + _pad)
            _yl = (_traj_y.min() - _pad, _traj_y.max() + _pad)
            ax.set_xlim(*_xl)
            ax.set_ylim(*_yl)

        def update_syn(t):
            j = pred[t]
            for (p, c), ln in zip(BONES, lines_h):
                ln.set_data_3d(
                    [j[p, 0], j[c, 0]], [j[p, 1], j[c, 1]], [j[p, 2], j[c, 2]]
                )
            pts_h._offsets3d = (j[:, 0], j[:, 1], j[:, 2])
            set_skeleton_colors(lines_h, pts_h, carry_highlight[t])
            if not _use_fixed:
                cx, cy = j[0, 0], j[0, 1]
                ax.set_xlim(cx - 1.2, cx + 1.2)
                ax.set_ylim(cy - 1.2, cy + 1.2)
            fig.suptitle(f"{clip_name}  |  frame {t}/{len(pred)}", fontsize=9)
            return lines_h + [pts_h]

        ani = animation.FuncAnimation(
            fig, update_syn, frames=len(pred), interval=interval, blit=False
        )
    else:
        # dual-panel clip reconstruction view
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(clip_name, fontsize=9)
        ax_pred = fig.add_subplot(121, projection="3d")
        ax_gt = fig.add_subplot(122, projection="3d")
        ax_pred.set_title("Flow prediction", fontsize=9)
        ax_gt.set_title("Ground truth", fontsize=9)

        for ax in [ax_pred, ax_gt]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_zlim(0, 2.0)
            ax.view_init(elev=10, azim=-70)
            if pickup_marker is not None:
                ax.scatter(
                    [pickup_marker[0]], [pickup_marker[1]], [pickup_marker[2]],
                    c="#39b54a", s=45, depthshade=False
                )
            if place_marker is not None:
                ax.scatter(
                    [place_marker[0]], [place_marker[1]], [place_marker[2]],
                    c="#d94343", s=45, depthshade=False
                )

        pred_lines, pred_pts = draw_bones(ax_pred, pred[0])
        gt_lines, gt_pts = draw_bones(ax_gt, jp_gt[0])

        def update(t):
            lim = 0.9
            for ax, joints, lines, pts in [
                (ax_pred, pred[t], pred_lines, pred_pts),
                (ax_gt, jp_gt[t], gt_lines, gt_pts),
            ]:
                for (p, c), ln in zip(BONES, lines):
                    ln.set_data_3d(
                        [joints[p, 0], joints[c, 0]],
                        [joints[p, 1], joints[c, 1]],
                        [joints[p, 2], joints[c, 2]],
                    )
                pts._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
                set_skeleton_colors(lines, pts, carry_highlight[t])
                cx, cy = joints[0, 0], joints[0, 1]
                ax.set_xlim(cx - lim, cx + lim)
                ax.set_ylim(cy - lim, cy + lim)
            fig.suptitle(f"{clip_name}  |  frame {t}/{len(pred)}", fontsize=9)
            return pred_lines + gt_lines + [pred_pts, gt_pts]

        ani = animation.FuncAnimation(
            fig, update, frames=len(pred), interval=interval, blit=False
        )

    if args.out:
        writer = "pillow" if args.out.endswith(".gif") else "ffmpeg"
        ani.save(args.out, writer=writer, fps=anim_fps)
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
