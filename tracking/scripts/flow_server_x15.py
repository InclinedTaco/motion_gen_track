"""
flow_server_x15.py: Real-time x15 horizon-flow model server for PHC demo mode.

The x15 model generates a horizon of H future state frames conditioned on
K past state frames + task_phase + desired_heading + active_goal.  The server
runs one inference per stride (default H//2 = 5 frames, matching the
animate_horizon_x15 rollout) and serves the decoded frames one per 30 Hz tick
via GET /get_pose.

Closed-loop only.  Planning is gated on robot feedback: the flow model does NOT
run until PHC has pushed at least K physical states via POST /push_state, and
each chunk is then conditioned on the latest K pushed states (prefetch=0 plans
only after the current chunk is consumed).  Until that history fills — and any
time no fresh feedback has arrived — the server holds the default standing pose
seeded from stats["default_state"]; keyboard/REST commands are stored but have
no effect while holding.  This held stand pose is the bootstrap target the
tracker follows on startup: PHC pushes its (initially standing) state every
tick *before* fetching a pose, so after K ticks the history fills and the model
switches from holding stand to generating motion.  There is no open-loop /
predicted-state rollout in this server.

Usage
-----
Terminal 1 (this server):
  cd <PROJECT_ROOT>/tracking
  python scripts/flow_server_x15.py \\
    --flow-ckpt  ../generation/checkpoints/joint_flow_x15_v13_travel_heading/best.pt \\
    --flow-stats ../generation/checkpoints/joint_flow_x15_v13_travel_heading/stats.json

Terminal 2 (PHC):
  python phc/run_hydra.py \\
    learning=im_mcp exp_name=phc_kp_mcp_iccv \\
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \\
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \\
    env.z_activation=relu env.obs_v=7 \\
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \\
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \\
    env.num_envs=1 headless=False test=True epoch=-1

Keyboard controls (terminal 1):
  1 / 2         stand / locomotion phase
  W / A / S / D set desired heading (North / West / South / East), locomotion only
  Space         zero heading (face forward, +X), stand phase
  G             print current world position
  Esc / Ctrl-C  quit

REST endpoints:
  GET  /get_pose   -> {"j3d": [[24×3]]}  (same format as flow_server.py)
  POST /push_state -> body JSON: {"root_xy":[x,y], "root_z":z,
                                  "heading":h, "pitch":p, "roll":r,
                                  "joint_rel":[69 floats]}
                   Feed one actual robot state into the server's feedback
                   history.  Planning uses ONLY these pushed states; the model
                   stays idle (serving the held stand pose) until K of them
                   have arrived.
  GET  /set_cmd?phase=loco&heading=<rad>&gx=0&gy=0&gz=0
                   Set command programmatically (phase: stand/loco/pickup/carry/place).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tty
import termios
import random
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R

PHC_ROOT       = Path(__file__).resolve().parents[1]
FLOW_ROOT      = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))

from dataset_horizon_x15 import STATE_DIM
from flow_server import mujoco_zup_to_server  # reuse coord-transform util

# ---------------------------------------------------------------------------
# Inlined from animate_horizon_x15 to avoid pulling in matplotlib / dataset.py
# ---------------------------------------------------------------------------
from flow_matching.solver import ODESolver
from infer import CondWrapper
from model import (
    JointFlowConcatMLP,
    JointFlowCrossTransformer,
    JointFlowFullCrossTransformer,
    JointFlowHistoryTransformer,
    JointFlowMLP,
    JointFlowTransformer,
)


def _rot_to_world(vec_xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    x = c * vec_xy[..., 0] - s * vec_xy[..., 1]
    y = s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def _rot_to_anchor(vec_xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    x = c * vec_xy[..., 0] + s * vec_xy[..., 1]
    y = -s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def _angle_rel(angle, anchor_heading):
    rel = angle - anchor_heading
    return float(np.arctan2(np.sin(rel), np.cos(rel)))


def local_state_from_world(state, anchor_xy, anchor_heading, anchor_pitch, anchor_roll):
    root_delta = _rot_to_anchor(state["root_xy"] - anchor_xy, anchor_heading)
    h_rel = _angle_rel(float(state["heading"]), anchor_heading)
    p_rel = _angle_rel(float(state["pitch"]), anchor_pitch)
    r_rel = _angle_rel(float(state["roll"]), anchor_roll)
    q = R.from_euler("ZYX", [h_rel, p_rel, r_rel]).as_quat()
    if q[3] < 0:
        q = -q
    joint_rel = state["joint_rel"].reshape(23, 3).copy()
    joint_rel[:, :2] = _rot_to_anchor(joint_rel[:, :2], anchor_heading)
    return np.concatenate([
        root_delta,
        np.array([state["root_z"]], dtype=np.float32),
        q.astype(np.float32),
        joint_rel.reshape(-1),
    ]).astype(np.float32)


def state_from_target_frame(frame, anchor_xy, anchor_heading, anchor_pitch, anchor_roll):
    root_xy = anchor_xy + _rot_to_world(frame[:2], anchor_heading)
    root_z = float(frame[2])
    q = frame[3:7].astype(np.float64)
    q = q / max(np.linalg.norm(q), 1e-6)
    h_rel, p_rel, r_rel = R.from_quat(q).as_euler("ZYX")
    joint_rel = frame[7:].astype(np.float32).reshape(23, 3).copy()
    joint_rel[:, :2] = _rot_to_world(joint_rel[:, :2], anchor_heading)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = np.array([root_xy[0], root_xy[1], root_z], dtype=np.float32)
    joints[1:] = joint_rel + joints[0][None]
    return {
        "root_xy": root_xy.astype(np.float32),
        "root_z": np.float32(root_z),
        "heading": np.float32(anchor_heading + h_rel),
        "pitch": np.float32(anchor_pitch + p_rel),
        "roll": np.float32(anchor_roll + r_rel),
        "joint_rel": joint_rel.reshape(-1),
        "joints": joints,
    }


def sample_horizon(model, cond, stats, device, steps):
    cond_norm = (cond - np.asarray(stats["cond_mean"], dtype=np.float32)) / np.asarray(
        stats["cond_std"], dtype=np.float32
    )
    cond_t = torch.tensor(cond_norm, device=device).unsqueeze(0)
    x0 = torch.randn(1, len(stats["target_mean"]), device=device)
    wrapped = CondWrapper(model, cond_t)
    with torch.no_grad():
        solver = ODESolver(velocity_model=wrapped)
        x1 = solver.sample(
            x_init=x0,
            step_size=1.0 / steps,
            method="midpoint",
            time_grid=torch.linspace(0, 1, steps + 1, device=device),
            return_intermediates=False,
        )
    raw = x1.cpu().numpy()[0]
    result = raw * np.asarray(stats["target_std"], dtype=np.float32) + np.asarray(
        stats["target_mean"], dtype=np.float32
    )
    result = np.clip(np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    return result.reshape(stats["horizon"], -1)


def sample_horizon_batch(model, conds, stats, device, steps):
    cond_mean = np.asarray(stats["cond_mean"], dtype=np.float32)
    cond_std = np.asarray(stats["cond_std"], dtype=np.float32)
    cond_norm = (conds - cond_mean[None]) / cond_std[None]
    cond_t = torch.tensor(cond_norm, device=device)
    x0 = torch.randn(cond_t.shape[0], len(stats["target_mean"]), device=device)
    wrapped = CondWrapper(model, cond_t)
    with torch.no_grad():
        solver = ODESolver(velocity_model=wrapped)
        x1 = solver.sample(
            x_init=x0,
            step_size=1.0 / steps,
            method="midpoint",
            time_grid=torch.linspace(0, 1, steps + 1, device=device),
            return_intermediates=False,
        )
    raw = x1.cpu().numpy()
    result = raw * np.asarray(stats["target_std"], dtype=np.float32)[None] + np.asarray(
        stats["target_mean"], dtype=np.float32
    )[None]
    result = np.clip(np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    return result.reshape(cond_t.shape[0], stats["horizon"], -1)


def load_model(checkpoint, stats, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    arch_name = saved_args.get("arch", "film")
    arch = {
        "film": JointFlowMLP,
        "concat": JointFlowConcatMLP,
        "transformer": JointFlowTransformer,
        "history_transformer": JointFlowHistoryTransformer,
        "cross_transformer": JointFlowCrossTransformer,
        "full_cross_transformer": JointFlowFullCrossTransformer,
    }[arch_name]
    model_kwargs = {
        "joint_dim": len(stats["target_mean"]),
        "cond_dim": len(stats["cond_mean"]),
        "hidden_size": saved_args.get("hidden", 1024),
        "n_layers": saved_args.get("layers", 5),
    }
    if arch_name in ("transformer", "history_transformer", "cross_transformer", "full_cross_transformer"):
        model_kwargs.update({
            "frame_dim": saved_args.get("frame_dim", 76),
            "n_heads": saved_args.get("heads", 8),
            "mlp_ratio": saved_args.get("mlp_ratio", 4.0),
            "dropout": saved_args.get("dropout", 0.0),
        })
        if arch_name in ("history_transformer", "cross_transformer", "full_cross_transformer"):
            model_kwargs["past_frames"] = int(stats["past_frames"])
        if arch_name == "full_cross_transformer":
            schema = stats.get("condition_schema", "phase")
            base = (3, 2, 3, 3) if schema == "loco_manip" else (5, 2, 3)
            model_kwargs["task_groups"] = base + ((1,) if stats.get("use_target_speed") else ())
    model = arch(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Command constants (phase indices: stand/loco/pickup/carry/place)
# ---------------------------------------------------------------------------
PHASE_STAND = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PHASE_LOCO  = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared server state
# ---------------------------------------------------------------------------
class ServerState:
    def __init__(
        self,
        default_world_state: dict,
        fps: float,
        initial_phase: np.ndarray,
        loco_warm_start: list[dict] | None = None,
        target_speed: float = 1.3,
        align_to_robot: bool = True,
        align_z: bool = False,
        auto_walk_schedule: bool = False,
        stand_steps: int = 30,
        curriculum_warmup_epochs: int = 100,
        curriculum_ramp_end_epochs: int = 700,
        curriculum_episode_length: int = 300,
    ):
        self.lock = threading.Lock()
        self.fps = fps
        self.dt  = 1.0 / fps

        # Rolling history of K world-state dicts (seed with default)
        self._history: list = []  # filled by flow_loop before first serve
        self._default = default_world_state

        # Planner commands
        self.task_phase      = initial_phase.copy()
        self.desired_heading = 0.0          # world-frame angle (rad)
        self.active_goal     = np.zeros(3, dtype=np.float32)  # world frame (phase schema)
        self.pickup_goal     = np.zeros(3, dtype=np.float32)  # world frame (loco_manip schema)
        self.place_goal      = np.zeros(3, dtype=np.float32)  # world frame (loco_manip schema)
        self.target_speed    = float(target_speed)
        self.align_to_robot  = bool(align_to_robot)
        self.align_z         = bool(align_z)
        self.auto_walk_schedule = bool(auto_walk_schedule)
        self.stand_steps = int(stand_steps)
        self.curriculum_warmup_epochs = int(curriculum_warmup_epochs)
        self.curriculum_ramp_end_epochs = int(curriculum_ramp_end_epochs)
        self.curriculum_episode_length = max(1, int(curriculum_episode_length))
        self._latest_episode_step = 0
        self._latest_episode_idx = 0
        self._push_count = 0
        self._output_offset  = np.zeros(3, dtype=np.float32)
        self._output_aligned = False
        self._loco_warm_start = copy_world_history(loco_warm_start) if loco_warm_start else None
        self._pending_loco_warm_start = (
            copy_world_history(loco_warm_start)
            if loco_warm_start and initial_phase[1] > 0.5
            else None
        )

        # Latest j3d to serve: (1, 24, 3) in PHC format
        standing = np.zeros((1, 24, 3), dtype=np.float32)
        standing[0, 0, 2] = float(default_world_state["root_z"])
        self.current_j3d = mujoco_zup_to_server(standing)

        # Closed-loop feedback from robot/PHC.  The next chunk should be
        # conditioned on the latest K physical states, not predicted refs.
        self._feedback_history: list[dict] = []
        self._feedback_version = 0

        self.running = True

    # -- command setters (called from keyboard / REST) -----------------------
    def set_phase(self, phase: np.ndarray):
        with self.lock:
            was_loco = self.task_phase[1] > 0.5
            will_loco = phase[1] > 0.5
            self.task_phase = phase.copy()
            if will_loco and not was_loco and self._loco_warm_start:
                self._pending_loco_warm_start = copy_world_history(self._loco_warm_start)
                print("  queued loco warm start from physics rollout", flush=True)

    def set_heading(self, angle: float):
        with self.lock:
            self.desired_heading = float(angle)

    def set_active_goal(self, goal: np.ndarray):
        with self.lock:
            self.active_goal = np.array(goal, dtype=np.float32)

    def set_pickup_goal(self, goal: np.ndarray):
        with self.lock:
            self.pickup_goal = np.array(goal, dtype=np.float32)

    def set_place_goal(self, goal: np.ndarray):
        with self.lock:
            self.place_goal = np.array(goal, dtype=np.float32)

    def set_target_speed(self, speed: float):
        with self.lock:
            self.target_speed = max(0.0, float(speed))

    def _apply_auto_walk_schedule(self, step: int):
        if not self.auto_walk_schedule:
            return
        phase = PHASE_STAND if step < self.stand_steps else PHASE_LOCO
        self.task_phase = phase.copy()

    def feedback_alpha(self) -> float:
        with self.lock:
            epoch = self._latest_episode_idx
            if epoch < self.curriculum_warmup_epochs:
                return 0.0
            if epoch >= self.curriculum_ramp_end_epochs:
                return 1.0
            denom = max(1, self.curriculum_ramp_end_epochs - self.curriculum_warmup_epochs)
            return float((epoch - self.curriculum_warmup_epochs) / denom)

    def consume_pending_loco_warm_start(self) -> list[dict] | None:
        with self.lock:
            warm = self._pending_loco_warm_start
            self._pending_loco_warm_start = None
            return warm

    # -- robot state feedback ------------------------------------------------
    def push_state(self, state: dict):
        """Called by /push_state endpoint. Records a physical robot state."""
        with self.lock:
            self._push_count += 1
            step_raw = state.get("episode_step")
            episode_idx_raw = state.get("episode_idx")
            step = int(step_raw) if step_raw is not None else self._push_count % self.curriculum_episode_length
            episode_idx = (
                int(episode_idx_raw)
                if episode_idx_raw is not None
                else self._push_count // self.curriculum_episode_length
            )
            self._latest_episode_step = step
            self._latest_episode_idx = episode_idx
            self._apply_auto_walk_schedule(step)
            if self.align_to_robot and not self._output_aligned:
                root0 = np.array([
                    float(self._default["root_xy"][0]),
                    float(self._default["root_xy"][1]),
                    float(self._default["root_z"]),
                ], dtype=np.float32)
                robot0 = np.array([
                    float(state["root_xy"][0]),
                    float(state["root_xy"][1]),
                    float(state["root_z"]),
                ], dtype=np.float32)
                self._output_offset[:2] = robot0[:2] - root0[:2]
                if self.align_z:
                    self._output_offset[2] = robot0[2] - root0[2]
                self._output_aligned = True
                print(
                    f"Aligned generated reference to robot: offset="
                    f"({self._output_offset[0]:+.3f}, {self._output_offset[1]:+.3f}, "
                    f"{self._output_offset[2]:+.3f})",
                    flush=True,
                )
            self._feedback_history.append(state)
            self._feedback_version += 1
            if len(self._feedback_history) > 128:
                self._feedback_history = self._feedback_history[-128:]

    def get_feedback_snapshot(self, k: int) -> tuple[list, int]:
        with self.lock:
            return list(self._feedback_history[-k:]), self._feedback_version

    # -- j3d access ----------------------------------------------------------
    def joints_to_j3d(self, joints):
        with self.lock:
            shifted = joints + self._output_offset
        return mujoco_zup_to_server(shifted)

    def set_j3d(self, j3d):
        with self.lock:
            self.current_j3d = j3d

    def get_j3d(self):
        with self.lock:
            return self.current_j3d.tolist(), self.task_phase.tolist()

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running


# ---------------------------------------------------------------------------
# Helper: convert a world-state dict to (24, 3) Z-up MuJoCo joint positions
# ---------------------------------------------------------------------------
def world_state_to_joints(s: dict) -> np.ndarray:
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [s["root_xy"][0], s["root_xy"][1], float(s["root_z"])]
    joints[1:] = s["joint_rel"].reshape(23, 3) + joints[0][None]
    return joints


def copy_world_state(s: dict) -> dict:
    return {
        "root_xy": np.array(s["root_xy"], dtype=np.float32).copy(),
        "root_z": np.float32(s["root_z"]),
        "heading": np.float32(s["heading"]),
        "pitch": np.float32(s.get("pitch", 0.0)),
        "roll": np.float32(s.get("roll", 0.0)),
        "joint_rel": np.array(s["joint_rel"], dtype=np.float32).copy(),
        "joints": np.array(s["joints"], dtype=np.float32).copy(),
    }


def copy_world_history(history: list[dict] | None) -> list[dict] | None:
    if history is None:
        return None
    return [copy_world_state(s) for s in history]


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def rebase_history_to_last_state(states: list[dict]) -> list[dict]:
    """Move a walking clip window so its last frame is at XY=(0,0), yaw=0."""
    last = states[-1]
    origin_xy = np.asarray(last["root_xy"], dtype=np.float32)
    origin_heading = float(last["heading"])
    out = []
    for s in states:
        xy = _rot_to_anchor(np.asarray(s["root_xy"], dtype=np.float32) - origin_xy, origin_heading)
        joint_rel = np.asarray(s["joint_rel"], dtype=np.float32).reshape(23, 3).copy()
        joint_rel[:, :2] = _rot_to_anchor(joint_rel[:, :2], origin_heading)
        root = np.array([xy[0], xy[1], float(s["root_z"])], dtype=np.float32)
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0] = root
        joints[1:] = joint_rel + root[None]
        out.append({
            "root_xy": xy.astype(np.float32),
            "root_z": np.float32(s["root_z"]),
            "heading": np.float32(_wrap_angle(float(s["heading"]) - origin_heading)),
            "pitch": np.float32(s.get("pitch", 0.0)),
            "roll": np.float32(s.get("roll", 0.0)),
            "joint_rel": joint_rel.reshape(-1),
            "joints": joints,
        })
    return out


def load_loco_warm_start_history(
    hdf5_path: str | None,
    clip_idx: int,
    start_frame: int,
    stats: dict,
) -> list[dict] | None:
    if not hdf5_path:
        return None

    import h5py

    path = Path(hdf5_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"loco warm-start HDF5 not found: {path}")

    K = int(stats["past_frames"])
    past_stride = int(stats.get("past_stride", 1))
    min_history = 1 + past_stride * (K - 1)

    with h5py.File(path, "r") as f:
        clips_group = f["clips"] if "clips" in f else f
        clip_names = sorted(clips_group.keys())
        if not (0 <= clip_idx < len(clip_names)):
            raise IndexError(f"--loco-warm-start-clip {clip_idx} out of range; {path} has {len(clip_names)} clips")
        clip_name = clip_names[clip_idx]
        grp = clips_group[clip_name]
        joint_positions = grp["joint_positions"][:].astype(np.float32)
        if start_frame + min_history > len(joint_positions):
            raise ValueError(
                f"warm-start window [{start_frame}, {start_frame + min_history}) exceeds "
                f"clip '{clip_name}' length {len(joint_positions)}"
            )
        frame_ids = range(start_frame, start_frame + min_history)
        root_xy = joint_positions[:, 0, :2]
        root_z = joint_positions[:, 0, 2]
        joint_rel = (joint_positions[:, 1:, :] - joint_positions[:, 0:1, :]).reshape(len(joint_positions), -1)
        if "pose_aa_root" in grp:
            euler_zyx = R.from_rotvec(grp["pose_aa_root"][:].astype(np.float32)).as_euler("ZYX").astype(np.float32)
            root_heading = euler_zyx[:, 0]
            root_pitch = euler_zyx[:, 1]
            root_roll = euler_zyx[:, 2]
        else:
            task_space = grp["task_space"][:].astype(np.float32)
            root_heading = np.zeros(len(task_space), dtype=np.float32)
            last_heading = 0.0
            for i, vel in enumerate(task_space[:, :2]):
                if np.linalg.norm(vel) > 0.05:
                    last_heading = float(np.arctan2(vel[1], vel[0]))
                root_heading[i] = last_heading
            root_pitch = np.zeros(len(task_space), dtype=np.float32)
            root_roll = np.zeros(len(task_space), dtype=np.float32)

    states = []
    for i in frame_ids:
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0] = [root_xy[i, 0], root_xy[i, 1], root_z[i]]
        joints[1:] = joint_rel[i].reshape(23, 3) + joints[0][None]
        states.append({
            "root_xy": root_xy[i].astype(np.float32),
            "root_z": np.float32(root_z[i]),
            "heading": np.float32(root_heading[i]),
            "pitch": np.float32(root_pitch[i]),
            "roll": np.float32(root_roll[i]),
            "joint_rel": joint_rel[i].astype(np.float32),
            "joints": joints,
        })

    history = rebase_history_to_last_state(states)
    print(
        f"Loaded loco warm start: {path} clip={clip_name!r} frames "
        f"[{start_frame}, {start_frame + min_history})",
        flush=True,
    )
    return history


# ---------------------------------------------------------------------------
# Flow inference loop
# ---------------------------------------------------------------------------
def flow_loop(state: ServerState, model, stats: dict, device, ode_steps: int,
              stride: int, prefetch_ticks: int, feedback_source: str):
    K          = int(stats["past_frames"])
    past_stride = int(stats.get("past_stride", 1))
    min_history = 1 + past_stride * (K - 1)
    default_st = stats["default_state"]
    init_z     = float(np.asarray(default_st)[2])
    init_jr    = np.asarray(default_st)[7:].astype(np.float32)

    def make_default_world() -> dict:
        xy = np.zeros(2, dtype=np.float32)
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0] = [0.0, 0.0, init_z]
        joints[1:] = init_jr.reshape(23, 3) + joints[0][None]
        return {
            "root_xy":  xy,
            "root_z":   np.float32(init_z),
            "heading":  np.float32(0.0),
            "pitch":    np.float32(0.0),
            "roll":     np.float32(0.0),
            "joint_rel": init_jr.copy(),
            "joints":   joints,
        }

    history = [make_default_world() for _ in range(min_history)]
    buffer: list = []  # decoded frames pending to be served, one per tick
    last_plan_feedback_version = -1

    def serve_hold_frame(feedback_tail: list):
        hold_state = feedback_tail[-1] if feedback_tail else history[-1]
        state.set_j3d(state.joints_to_j3d(hold_state["joints"][np.newaxis]))

    def infer_stride(hist_tail: list) -> list:
        """One inference + decode of `stride` frames (matches animate_horizon_x15
        generate()).  hist_tail is the K frames preceding the new anchor."""
        with state.lock:
            task_phase      = state.task_phase.copy()
            desired_heading = float(state.desired_heading)
            active_goal     = state.active_goal.copy()
            pickup_goal     = state.pickup_goal.copy()
            place_goal      = state.place_goal.copy()
            target_speed    = float(state.target_speed)

        anchor           = hist_tail[-1]
        anchor_xy        = anchor["root_xy"]
        anchor_z         = float(anchor["root_z"])
        anchor_heading   = float(anchor["heading"])
        anchor_pitch     = float(anchor["pitch"])
        anchor_roll      = float(anchor["roll"])

        past = [
            local_state_from_world(
                hist_tail[-1 - past_stride * (K - 1 - i)],
                anchor_xy, anchor_heading, anchor_pitch, anchor_roll,
            )
            for i in range(K)
        ]

        schema = stats.get("condition_schema", "phase")

        # desired_heading: null token for non-loco phases; travel-direction unit vector for loco
        if task_phase[1] > 0.5:
            h_rel = desired_heading - anchor_heading
            h_rel = float(np.arctan2(np.sin(h_rel), np.cos(h_rel)))
            heading_vec = np.array([np.cos(h_rel), np.sin(h_rel)], dtype=np.float32)
        else:
            heading_vec = np.zeros(2, dtype=np.float32)

        def _world_to_anchor(world_xyz: np.ndarray) -> np.ndarray:
            c, s_val = np.cos(anchor_heading), np.sin(anchor_heading)
            dxy = world_xyz[:2] - anchor_xy
            return np.array([
                c * dxy[0] + s_val * dxy[1],
                -s_val * dxy[0] + c * dxy[1],
                world_xyz[2] - anchor_z,
            ], dtype=np.float32)

        if schema == "loco_manip":
            is_manip = task_phase[2] > 0.5 or task_phase[3] > 0.5 or task_phase[4] > 0.5
            is_loco  = task_phase[1] > 0.5
            if is_manip:
                mode = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            elif is_loco:
                mode = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                mode = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if is_manip:
                pu_anchor = _world_to_anchor(pickup_goal)
                pl_anchor = _world_to_anchor(place_goal)
            else:
                pu_anchor = np.zeros(3, dtype=np.float32)
                pl_anchor = np.zeros(3, dtype=np.float32)
            task_parts = [mode, heading_vec, pu_anchor, pl_anchor]
        else:
            # phase schema: [task_phase(5D), heading(2D), active_goal(3D)]
            if np.any(active_goal != 0.0):
                goal_anchor = _world_to_anchor(active_goal)
            else:
                goal_anchor = np.zeros(3, dtype=np.float32)
            task_parts = [task_phase, heading_vec, goal_anchor]

        extras = []
        if stats.get("use_target_speed", False):
            speed = target_speed if task_phase[1] > 0.5 else 0.0
            extras = [np.array([speed], dtype=np.float32)]

        cond = np.concatenate(past + task_parts + extras).astype(np.float32)
        target = sample_horizon(model, cond, stats, device, ode_steps)  # (H, 76D)

        return [
            state_from_target_frame(
                target[j], anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            for j in range(int(stats["horizon"]))
        ]

    # With closed-loop robot feedback, prefetch=0 means the next chunk is
    # planned only after the current chunk is consumed, so the latest K
    # physical states can be used as history.
    worker: threading.Thread | None = None
    worker_out: list = []
    planning_history = copy_world_history(history)

    target_dt = state.dt
    while state.is_running():
        t0 = time.perf_counter()

        if worker is None and len(buffer) <= prefetch_ticks:
            warm_tail = state.consume_pending_loco_warm_start()
            feedback_tail, feedback_version = state.get_feedback_snapshot(min_history)
            if warm_tail is not None:
                hist_tail = warm_tail
                history = copy_world_history(warm_tail)
                planning_history = copy_world_history(warm_tail)
                last_plan_feedback_version = feedback_version
                print("  using loco warm-start history for next chunk", flush=True)
            elif feedback_source == "prediction":
                hist_tail = copy_world_history(planning_history)
            elif feedback_source == "curriculum" and (
                len(feedback_tail) < min_history
                or feedback_version == last_plan_feedback_version
                or random.random() >= state.feedback_alpha()
            ):
                hist_tail = copy_world_history(planning_history)
            elif len(feedback_tail) < min_history or feedback_version == last_plan_feedback_version:
                serve_hold_frame(feedback_tail)
                elapsed = time.perf_counter() - t0
                sleep_t = target_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue
            else:
                hist_tail = feedback_tail
                last_plan_feedback_version = feedback_version
            worker = threading.Thread(
                target=lambda ht=hist_tail: worker_out.append(infer_stride(ht)),
                daemon=True,
            )
            worker.start()

        if not buffer:
            worker.join()
            planned = worker_out.pop()
            worker = None
            buffer = planned[:stride]
            if feedback_source in ("prediction", "curriculum"):
                planning_history.extend(planned[:stride])
                planning_history = planning_history[-min_history:]

        next_state = buffer.pop(0)
        if feedback_source == "robot":
            history.append(next_state)
            if len(history) > min_history:
                history.pop(0)

        pos_world = next_state["joints"][np.newaxis]     # (1, 24, 3) Z-up MuJoCo
        j3d = state.joints_to_j3d(pos_world)
        state.set_j3d(j3d)

        elapsed = time.perf_counter() - t0
        sleep_t = target_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        elif elapsed > stride * target_dt:
            print(f"  WARNING: step took {elapsed*1000:.1f} ms (inference budget {stride*target_dt*1000:.0f} ms)", flush=True)


# ---------------------------------------------------------------------------
# Flask REST server
# ---------------------------------------------------------------------------
app = Flask(__name__)
_state: ServerState = None


@app.route("/get_pose")
def get_pose():
    j3d, phase = _state.get_j3d()
    return jsonify({"j3d": j3d, "task_phase": phase})


@app.route("/push_state", methods=["POST"])
def push_state():
    """Closed-loop feedback: robot/PHC sends its current state.

    Body JSON:
      {
        "root_xy": [x, y],
        "root_z": z,
        "heading": h,          // yaw, radians
        "pitch": p,
        "roll": r,
        "joint_rel": [69 floats]  // 23 root-relative joints, world frame
      }
    """
    data = request.get_json(force=True)
    ws = {
        "root_xy":  np.array(data["root_xy"], dtype=np.float32),
        "root_z":   np.float32(data["root_z"]),
        "heading":  np.float32(data["heading"]),
        "pitch":    np.float32(data.get("pitch", 0.0)),
        "roll":     np.float32(data.get("roll", 0.0)),
        "joint_rel": np.array(data["joint_rel"], dtype=np.float32),
        "episode_step": int(data["episode_step"]) if "episode_step" in data else None,
        "episode_idx": int(data["episode_idx"]) if "episode_idx" in data else None,
    }
    # joints column for completeness
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [ws["root_xy"][0], ws["root_xy"][1], float(ws["root_z"])]
    joints[1:] = ws["joint_rel"].reshape(23, 3) + joints[0][None]
    ws["joints"] = joints
    _state.push_state(ws)
    return jsonify({"ok": True})


@app.route("/set_cmd")
def set_cmd():
    """Programmatic command override.

    Query params (all schemas):
      phase  : stand | loco | pickup | carry | place | manip
      heading: float (radians, world frame)
    Phase schema only:
      gx, gy, gz: active goal world position
    Loco_manip schema only:
      pux, puy, puz: pickup goal world position
      plx, ply, plz: place goal world position
    """
    phase_map = {
        "stand":  np.array([1, 0, 0, 0, 0], dtype=np.float32),
        "loco":   np.array([0, 1, 0, 0, 0], dtype=np.float32),
        "pickup": np.array([0, 0, 1, 0, 0], dtype=np.float32),
        "carry":  np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "place":  np.array([0, 0, 0, 0, 1], dtype=np.float32),
        "manip":  np.array([0, 0, 1, 0, 0], dtype=np.float32),  # loco_manip alias
    }
    p = request.args.get("phase", "loco")
    if p in phase_map:
        _state.set_phase(phase_map[p])
    h = request.args.get("heading")
    if h is not None:
        _state.set_heading(float(h))
    # phase schema goal
    gx = request.args.get("gx")
    if gx is not None:
        _state.set_active_goal(np.array([float(gx),
                                          float(request.args.get("gy", 0.0)),
                                          float(request.args.get("gz", 0.0))], dtype=np.float32))
    # loco_manip schema goals
    pux = request.args.get("pux")
    if pux is not None:
        _state.set_pickup_goal(np.array([float(pux),
                                          float(request.args.get("puy", 0.0)),
                                          float(request.args.get("puz", 0.0))], dtype=np.float32))
    plx = request.args.get("plx")
    if plx is not None:
        _state.set_place_goal(np.array([float(plx),
                                         float(request.args.get("ply", 0.0)),
                                         float(request.args.get("plz", 0.0))], dtype=np.float32))
    speed = request.args.get("speed")
    if speed is not None:
        _state.set_target_speed(float(speed))
    return jsonify({"phase": p, "heading": h, "speed": speed})


def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8080, threaded=True)


def kill_port_listener(port: int):
    """Best-effort cleanup for stale local Flask/debug servers."""
    def listeners():
        try:
            proc = subprocess.run(
                ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("lsof not found; cannot auto-clear port", flush=True)
            return []
        pids = []
        for line in proc.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)
        return sorted(set(pids))

    for attempt in range(5):
        pids = listeners()
        if not pids:
            return
        sig = signal.SIGTERM if attempt == 0 else signal.SIGKILL
        action = "killing" if sig == signal.SIGTERM else "force-killing"
        print(f"Port {port} is busy; {action} listener PID(s): {pids}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        time.sleep(0.5)

    pids = listeners()
    if pids:
        raise RuntimeError(f"Port {port} is still busy after cleanup; listener PID(s): {pids}")


# ---------------------------------------------------------------------------
# Keyboard control
# ---------------------------------------------------------------------------
HEADING_STEP = 0.2  # rad per key press

HEADING_DIRS = {
    "w": 0.0,           # +X (forward)
    "s": np.pi,         # -X (backward)
    "a": np.pi / 2,     # +Y (left)
    "d": -np.pi / 2,    # -Y (right)
}


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def print_status(state: ServerState, schema: str = "phase"):
    with state.lock:
        tp = state.task_phase
        hd = state.desired_heading
        if schema == "loco_manip":
            is_manip = tp[2] > 0.5 or tp[3] > 0.5 or tp[4] > 0.5
            ph = "manip" if is_manip else ("loco" if tp[1] > 0.5 else "stand")
        else:
            ph = "stand" if tp[0] > 0.5 else \
                 "loco"  if tp[1] > 0.5 else \
                 "pickup" if tp[2] > 0.5 else \
                 "carry" if tp[3] > 0.5 else "place"
    sys.stdout.write(
        f"\r  phase={ph:6s}  heading={hd:+.2f} rad ({np.degrees(hd):+.0f}°)   "
    )
    sys.stdout.flush()


def keyboard_loop(state: ServerState, schema: str = "phase"):
    PHASE_MANIP = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    print("\n=== Flow Server X15 running on port 8080 ===")
    if schema == "loco_manip":
        print("  1/2/3  stand/loco/manip    W/A/S/D  set heading (loco mode)")
    else:
        print("  1/2  stand/loco    W/A/S/D  set heading (loco mode)")
    print("  Space  stand + zero heading")
    print("  G  print current world position")
    print("  Esc / Ctrl-C  quit\n")

    while state.is_running():
        ch = getch()
        if ch in ("\x1b", "\x03"):
            state.stop()
            break
        elif ch == "1":
            state.set_phase(PHASE_STAND)
        elif ch == "2":
            state.set_phase(PHASE_LOCO)
        elif ch == "3" and schema == "loco_manip":
            state.set_phase(PHASE_MANIP)
        elif ch in HEADING_DIRS:
            state.set_phase(PHASE_LOCO)
            state.set_heading(HEADING_DIRS[ch])
        elif ch == " ":
            state.set_phase(PHASE_STAND)
            state.set_heading(0.0)
        elif ch in ("g", "G"):
            pass  # position logged by print_status
        print_status(state, schema)

    print("\nShutting down …")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt",  required=True)
    p.add_argument("--flow-stats", required=True)
    p.add_argument("--fps",      type=float, default=30.0)
    p.add_argument("--ode-steps", type=int,  default=10,
                   help="ODE steps per inference (10 ≈ 11 ms on GPU)")
    p.add_argument("--stride",   type=int,   default=None,
                   help="Frames consumed per inference (default: horizon // 2, "
                        "matching animate_horizon_x15)")
    p.add_argument("--prefetch-ticks", type=int, default=0,
                   help="Start the next chunk when this many current-buffer frames remain. "
                        "Use 0 to condition on the latest physical robot states at chunk boundary.")
    p.add_argument("--no-kill-port", action="store_true", default=False,
                   help="Do not kill an existing listener on port 8080 before starting Flask")
    p.add_argument("--feedback-source", choices=["robot", "prediction", "curriculum"], default="robot",
                   help="Condition chunks on PHC-pushed robot states, server predictions, or a prediction->robot curriculum")
    p.add_argument("--warmup",   type=int,   default=30,
                   help="Frames to run before accepting PHC connections")
    p.add_argument("--initial-phase", choices=["stand", "loco"], default="stand",
                   help="Phase used on startup before keyboard/REST commands")
    p.add_argument("--target-speed", type=float, default=1.3,
                   help="Target speed token for checkpoints trained with --use-target-speed")
    p.add_argument("--auto-walk-schedule", action="store_true", default=False,
                   help="Use pushed PHC episode step to serve stand for --stand-steps, then loco")
    p.add_argument("--stand-steps", type=int, default=30,
                   help="Episode steps to keep stand phase before switching to loco under --auto-walk-schedule")
    p.add_argument("--curriculum-warmup-epochs", type=int, default=100,
                   help="For --feedback-source curriculum, keep robot feedback probability at 0 before this episode index")
    p.add_argument("--curriculum-ramp-end-epochs", type=int, default=700,
                   help="For --feedback-source curriculum, reach robot feedback probability 1 at this episode index")
    p.add_argument("--curriculum-episode-length", type=int, default=300,
                   help="Fallback episode length used to interpret pushed global steps when episode_idx is absent")
    p.add_argument("--no-align-to-robot", action="store_true", default=False,
                   help="Do not shift served references to the first PHC-pushed robot root XY")
    p.add_argument("--align-z", action="store_true", default=False,
                   help="Also shift served reference root height to the first PHC-pushed robot root")
    p.add_argument("--loco-warm-start-hdf5", default=None,
                   help="Physics-rollout HDF5 clip used to seed the next loco chunk on stand->loco")
    p.add_argument("--loco-warm-start-clip", type=int, default=0,
                   help="Sorted clip index inside --loco-warm-start-hdf5")
    p.add_argument("--loco-warm-start-frame", type=int, default=10,
                   help="First frame of the physics-rollout warm-start window")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    global _state
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.flow_stats) as f:
        stats = json.load(f)

    model = load_model(args.flow_ckpt, stats, device)
    stride = args.stride if args.stride is not None else max(1, int(stats["horizon"]) // 2)
    print(f"X15 model loaded  (device={device}  K={stats['past_frames']}  "
          f"H={stats['horizon']}  stride={stride})")

    # Build default world state from stats
    ds  = np.asarray(stats["default_state"], dtype=np.float32)
    init_z = float(ds[2])
    init_jr = ds[7:].copy()
    joints0 = np.zeros((24, 3), dtype=np.float32)
    joints0[0] = [0.0, 0.0, init_z]
    joints0[1:] = init_jr.reshape(23, 3) + joints0[0][None]
    default_ws = {
        "root_xy":  np.zeros(2, dtype=np.float32),
        "root_z":   np.float32(init_z),
        "heading":  np.float32(0.0),
        "pitch":    np.float32(0.0),
        "roll":     np.float32(0.0),
        "joint_rel": init_jr,
        "joints":   joints0,
    }

    warm_history = load_loco_warm_start_history(
        args.loco_warm_start_hdf5,
        args.loco_warm_start_clip,
        args.loco_warm_start_frame,
        stats,
    )

    initial_phase = PHASE_STAND if args.initial_phase == "stand" else PHASE_LOCO
    align_to_robot = not args.no_align_to_robot
    if args.feedback_source in ("robot", "curriculum") and align_to_robot:
        align_to_robot = False
        print("Disabled output alignment for robot-conditioned feedback.", flush=True)

    _state = ServerState(
        default_ws,
        fps=args.fps,
        initial_phase=initial_phase,
        loco_warm_start=warm_history,
        target_speed=args.target_speed,
        align_to_robot=align_to_robot,
        align_z=args.align_z,
        auto_walk_schedule=args.auto_walk_schedule,
        stand_steps=args.stand_steps,
        curriculum_warmup_epochs=args.curriculum_warmup_epochs,
        curriculum_ramp_end_epochs=args.curriculum_ramp_end_epochs,
        curriculum_episode_length=args.curriculum_episode_length,
    )

    flow_thread = threading.Thread(
        target=flow_loop,
        args=(
            _state,
            model,
            stats,
            device,
            args.ode_steps,
            stride,
            args.prefetch_ticks,
            args.feedback_source,
        ),
        daemon=True,
    )
    flow_thread.start()

    schema = stats.get("condition_schema", "phase")
    print(
        f"Initial phase: {args.initial_phase}  schema={schema}  "
        f"prefetch_ticks={args.prefetch_ticks}  feedback_source={args.feedback_source}"
    )
    if args.feedback_source == "curriculum":
        print(
            f"Feedback curriculum: alpha=0 until epoch {args.curriculum_warmup_epochs}, "
            f"ramps to 1 by epoch {args.curriculum_ramp_end_epochs}."
        )
    if args.auto_walk_schedule:
        print(f"Auto walk schedule: stand for {args.stand_steps} steps, then loco.")
    if warm_history:
        print("Loco warm-start enabled; next stand->loco transition will seed from physics rollout.")
    print(f"Warming up ({args.warmup} frames) …", end="", flush=True)
    time.sleep(args.warmup / args.fps)
    print(" done.")

    if not args.no_kill_port:
        kill_port_listener(8080)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        if sys.stdin.isatty():
            keyboard_loop(_state, schema)
        else:
            print("stdin is not a tty — control via REST:")
            print("  curl 'http://localhost:8080/set_cmd?phase=loco&heading=0.0'")
            while _state.is_running():
                time.sleep(0.5)
    except KeyboardInterrupt:
        _state.stop()


if __name__ == "__main__":
    main()
