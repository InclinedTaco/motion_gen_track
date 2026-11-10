"""
Animate x15 (v6) receding-horizon flow generation.

Loads an x15 checkpoint and generates synthetic planner-command sequences.
Commands are: task_phase (5D: stand/locomotion/pickup/carry/place)
+ desired_heading (2D unit vector, travel direction in anchor frame)
+ active_goal (3D, anchor frame).  Root height is predicted by the model.

desired_heading steers locomotion only; goal phases (pickup/carry/place) are
steered by active_goal and get the null heading token (0, 0).  command["heading"]
is the desired travel direction angle in world frame (0 = forward/+X convention).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # joint_flow

import matplotlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from animate import BONES, draw_bones
from dataset_horizon_x15 import (
    STATE_DIM,
    TARGET_FRAME_DIM,
    validate_x15_stats_compatible,
)
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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def rot_to_world(vec_xy: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    x = c * vec_xy[..., 0] - s * vec_xy[..., 1]
    y = s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def rot_to_anchor(vec_xy: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    x = c * vec_xy[..., 0] + s * vec_xy[..., 1]
    y = -s * vec_xy[..., 0] + c * vec_xy[..., 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def angle_rel(angle: float, anchor_heading: float) -> float:
    rel = angle - anchor_heading
    return float(np.arctan2(np.sin(rel), np.cos(rel)))


# ---------------------------------------------------------------------------
# State representation (mirrors JointFlowX15Dataset._state_frame, 76D)
# Frame layout: root_delta_xy(0:2) + root_z(2) + quat_xyzw(3:7) + joint_rel(7:)
# ---------------------------------------------------------------------------


def local_state_from_world(
    state: dict,
    anchor_xy: np.ndarray,
    anchor_heading: float,
    anchor_pitch: float,
    anchor_roll: float,
) -> np.ndarray:
    root_delta = rot_to_anchor(state["root_xy"] - anchor_xy, anchor_heading)
    h_rel = angle_rel(float(state["heading"]), anchor_heading)
    p_rel = angle_rel(float(state["pitch"]), anchor_pitch)
    r_rel = angle_rel(float(state["roll"]), anchor_roll)
    q = R.from_euler("ZYX", [h_rel, p_rel, r_rel]).as_quat()  # [x,y,z,w]
    if q[3] < 0:
        q = -q
    joint_rel = state["joint_rel"].reshape(23, 3).copy()
    joint_rel[:, :2] = rot_to_anchor(joint_rel[:, :2], anchor_heading)
    return np.concatenate(
        [
            root_delta,
            np.array([state["root_z"]], dtype=np.float32),
            q.astype(np.float32),
            joint_rel.reshape(-1),
        ]
    ).astype(np.float32)


def state_from_target_frame(
    frame: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_heading: float,
    anchor_pitch: float,
    anchor_roll: float,
) -> dict:
    root_xy = anchor_xy + rot_to_world(frame[:2], anchor_heading)
    root_z = float(frame[2])
    q = frame[3:7].astype(np.float64)
    q = q / max(np.linalg.norm(q), 1e-6)
    h_rel, p_rel, r_rel = R.from_quat(q).as_euler("ZYX")
    joint_rel = frame[7:].astype(np.float32).reshape(23, 3).copy()
    joint_rel[:, :2] = rot_to_world(joint_rel[:, :2], anchor_heading)
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


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def get_active_goal_local(
    command: dict, t: int, anchor_xy: np.ndarray, anchor_z: float, anchor_heading: float
) -> np.ndarray:
    """Convert the world-space active goal to the anchor frame at time t."""
    phase = command["task_phase"][min(t, len(command["task_phase"]) - 1)]
    goals = command["goals"]
    if goals is None:
        return np.zeros(3, dtype=np.float32)

    # Training: only goal phases (pickup/carry/place) carry active_goal; stand
    # and locomotion phases get the null (zero) goal token.
    if not (phase[2] > 0.5 or phase[3] > 0.5 or phase[4] > 0.5):
        return np.zeros(3, dtype=np.float32)

    if phase[3] > 0.5 or phase[4] > 0.5:  # carry walks toward the place goal
        g = goals["place"]
    else:  # approach or pickup
        g = goals["pickup"]
    xy = rot_to_anchor(g[:2] - anchor_xy, anchor_heading)
    return np.array([xy[0], xy[1], g[2] - anchor_z], dtype=np.float32)


def make_commands(args) -> dict:
    T = int(round(args.duration * args.fps))
    # Phase order: [stand, locomotion, pickup, carry, place]
    task_phase = np.zeros((T, 5), dtype=np.float32)
    heading = np.zeros(T, dtype=np.float32)  # desired world heading (stand/locomotion)
    goals = None
    place_frames = None

    if args.motion == "stand":
        task_phase[:, 0] = 1.0

    elif args.motion == "constant_walk":
        task_phase[:, 1] = 1.0

    elif args.motion == "curved_walk":
        # Locomotion with a constant commanded yaw rate: world heading ramps
        # linearly so the receding-horizon command is a steady turn.
        task_phase[:, 1] = 1.0
        omega = np.radians(args.turn_rate)  # rad/s
        heading = (omega * np.arange(T) / args.fps).astype(np.float32)

    elif args.motion == "stop_start_walk":
        # Alternate walk (3 s) and stand (2 s) segments.
        n_walk = int(3.0 * args.fps)
        n_stand = int(2.0 * args.fps)
        t = 0
        while t < T:
            end = min(t + n_walk, T)
            task_phase[t:end, 1] = 1.0
            t = end
            end = min(t + n_stand, T)
            task_phase[t:end, 0] = 1.0
            t = end

    elif args.motion == "walk_pcp":
        # Walk forward (locomotion, heading-steered) then pick → carry → place
        # (goal-steered). Goals sit ahead of the walk endpoint.
        fps = args.fps
        n_walk = max(4, int(args.walk_secs * fps))
        n_pu = max(4, int(args.pickup_secs * fps))
        n_carry = max(4, int(args.carry_secs * fps))
        n_place = max(4, int(args.place_secs * fps))
        T = n_walk + n_pu + n_carry + n_place
        task_phase = np.zeros((T, 5), dtype=np.float32)
        heading = np.zeros(T, dtype=np.float32)  # walk straight forward

        b = n_walk
        c = b + n_pu
        d = c + n_carry
        task_phase[0:b, 1] = 1.0                  # locomotion
        task_phase[b:c] = [0.0, 0.0, 1.0, 0.0, 0.0]  # pickup
        task_phase[c:d] = [0.0, 0.0, 0.0, 1.0, 0.0]  # carry
        task_phase[d:T] = [0.0, 0.0, 0.0, 0.0, 1.0]  # place

        # Goals scale with intrinsic forward speed (~1.05 m/s): pickup lands just
        # ahead of the walk endpoint; place lands a carry-distance further on.
        walk_dist = 1.05 * args.walk_secs           # ~1.05 m/s straight walk
        pickup_x = walk_dist + 0.30
        goals = {
            "pickup": np.array([pickup_x, 0.05, 0.08], dtype=np.float32),
            # carry advances at ~0.6 m/s; place lands where carry actually ends.
            "place": np.array([pickup_x + 0.60 * args.carry_secs, 0.05, 0.18],
                              dtype=np.float32),
        }
        place_frames = n_place

    elif args.motion == "pick_carry_place":
        fps = args.fps
        n_loco = max(4, int(args.walk_secs * fps))   # loco budget; proximity-terminated
        n_pu = max(4, int(args.pickup_secs * fps))
        n_carry = max(4, int(args.carry_secs * fps))
        n_pl = max(4, int(args.place_secs * fps))
        T = n_loco + n_pu + n_carry + n_pl
        task_phase = np.zeros((T, 5), dtype=np.float32)
        heading = np.zeros(T, dtype=np.float32)

        pickup_x = 1.05 * args.walk_secs + 0.30
        goals = {
            "pickup": np.array([pickup_x, 0.05, 0.08], dtype=np.float32),
            "place": np.array([pickup_x + 0.60 * args.carry_secs, 0.05, 0.18], dtype=np.float32),
        }
        task_phase[0:n_loco, 1] = 1.0                              # loco
        task_phase[n_loco:n_loco + n_pu, 2] = 1.0                  # pickup
        task_phase[n_loco + n_pu:n_loco + n_pu + n_carry, 3] = 1.0  # carry
        task_phase[n_loco + n_pu + n_carry:T, 4] = 1.0             # place
        # walk heading toward pickup goal
        heading[:n_loco] = float(np.arctan2(goals["pickup"][1], goals["pickup"][0]))

        pickup_frames = n_pu
        carry_frames = n_carry
        place_frames = n_pl

    else:
        raise ValueError(f"Unknown motion: {args.motion}")

    # target_speed: desired mean root speed (m/s); 0 for non-locomotion frames
    target_speed = np.where(task_phase[:, 1] > 0.5, float(args.speed), 0.0).astype(np.float32)

    return {
        "name": args.motion,
        "T": T,
        "task_phase": task_phase,
        "heading": heading,
        "goals": goals,
        "target_speed": target_speed,
        "place_frames": place_frames,
        "pickup_frames": locals().get("pickup_frames"),
        "carry_frames": locals().get("carry_frames"),
    }


# ---------------------------------------------------------------------------
# Model loading and ODE sampling
# ---------------------------------------------------------------------------


def load_model(checkpoint: str, stats: dict, device):
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
    if arch_name in (
        "transformer",
        "history_transformer",
        "cross_transformer",
        "full_cross_transformer",
    ):
        model_kwargs.update(
            {
                "frame_dim": saved_args.get("frame_dim", 76),
                "n_heads": saved_args.get("heads", 8),
                "mlp_ratio": saved_args.get("mlp_ratio", 4.0),
                "dropout": saved_args.get("dropout", 0.0),
            }
        )
        if arch_name in (
            "history_transformer",
            "cross_transformer",
            "full_cross_transformer",
        ):
            model_kwargs["past_frames"] = int(stats["past_frames"])
        if arch_name == "full_cross_transformer":
            schema = stats.get("condition_schema", "phase")
            if schema == "loco_manip":
                base = (3, 2, 3, 3)
            else:
                base = (5, 2, 3)
            model_kwargs["task_groups"] = base + ((1,) if stats.get("use_target_speed") else ())
    model = arch(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


class _PastBlendWrapper(CondWrapper):
    """PARC-style guidance: blend the velocity field conditioned with and
    without past frames. s weights the no-past (command-only) branch.
    Requires a model trained with --past-dropout."""

    def __init__(
        self, model, cond_full: torch.Tensor, cond_null: torch.Tensor, s: float
    ):
        super().__init__(model, cond_full)
        self.cond_null = cond_null
        self.s = s

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        t2 = t.reshape(1).expand(2 * B)
        v = self.model(
            x.repeat(2, 1), t2, torch.cat([self.cond, self.cond_null], dim=0)
        )
        return (1.0 - self.s) * v[:B] + self.s * v[B:]


def sample_horizon(
    model, cond: np.ndarray, stats: dict, device, steps: int, past_guidance: float = 0.0
) -> np.ndarray:
    cond_norm = (cond - np.asarray(stats["cond_mean"], dtype=np.float32)) / np.asarray(
        stats["cond_std"], dtype=np.float32
    )
    cond_t = torch.tensor(cond_norm, device=device).unsqueeze(0)
    x0 = torch.randn(1, len(stats["target_mean"]), device=device)
    if past_guidance > 0.0:
        past_dim = int(stats["past_frames"]) * int(stats["state_dim"])
        cond_null = cond_t.clone()
        cond_null[:, :past_dim] = 0.0
        wrapped = _PastBlendWrapper(model, cond_t, cond_null, past_guidance)
    else:
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
    result = np.clip(
        np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0
    )
    return result.reshape(stats["horizon"], TARGET_FRAME_DIM)


# ---------------------------------------------------------------------------
# Receding-horizon generation
# ---------------------------------------------------------------------------


def rebase_history(states: list) -> list:
    """Re-express world states so the last one sits at the origin with heading 0
    (matching the synthetic command frame). Inter-frame motion is preserved."""
    last = states[-1]
    origin_xy = last["root_xy"]
    th = float(last["heading"])
    out = []
    for s in states:
        xy = rot_to_anchor(s["root_xy"] - origin_xy, th)
        jr = s["joint_rel"].reshape(23, 3).copy()
        jr[:, :2] = rot_to_anchor(jr[:, :2], th)
        root = np.array([xy[0], xy[1], s["root_z"]], dtype=np.float32)
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0] = root
        joints[1:] = jr + root[None]
        out.append(
            {
                "root_xy": xy.astype(np.float32),
                "root_z": np.float32(s["root_z"]),
                "heading": np.float32(angle_rel(float(s["heading"]), th)),
                "pitch": np.float32(s["pitch"]),
                "roll": np.float32(s["roll"]),
                "joint_rel": jr.reshape(-1),
                "joints": joints,
            }
        )
    return out


def generate(
    model,
    stats: dict,
    command: dict,
    device,
    steps: int,
    stride: int,
    past_guidance: float = 0.0,
    init_history: list = None,
) -> np.ndarray:
    past_frames = int(stats["past_frames"])
    past_stride = int(stats.get("past_stride", 1))
    # minimum history length so the oldest past frame exists
    min_history = 1 + past_stride * (past_frames - 1)
    default_state = np.asarray(stats["default_state"], dtype=np.float32)
    if default_state.shape != (STATE_DIM,):
        raise ValueError(
            f"default_state has shape {default_state.shape}, expected ({STATE_DIM},)"
        )
    if stride is None:
        H = int(stats["horizon"])
        stride = H if past_stride > 1 else max(1, H // 2)

    if init_history is not None:
        history = list(init_history)
    else:
        init_joint_rel = default_state[7:].copy()
        init_z = float(default_state[2])
        init_state = {
            "root_xy": np.zeros(2, dtype=np.float32),
            "root_z": np.float32(init_z),
            "heading": np.float32(0.0),
            "pitch": np.float32(0.0),
            "roll": np.float32(0.0),
            "joint_rel": init_joint_rel,
            "joints": np.concatenate(
                [
                    np.array([[0.0, 0.0, init_z]], dtype=np.float32),
                    init_joint_rel.reshape(23, 3) + np.array([[0.0, 0.0, init_z]]),
                ],
                axis=0,
            ),
        }
        history = [init_state.copy() for _ in range(min_history)]

    pred = []
    t = 0
    T = command["T"]

    while t < T:
        anchor = history[-1]
        anchor_xy = anchor["root_xy"]
        anchor_z = float(anchor["root_z"])
        anchor_heading = float(anchor["heading"])
        anchor_pitch = float(anchor["pitch"])
        anchor_roll = float(anchor["roll"])

        # mirrors dataset_horizon_x15.py past_indices formula exactly:
        # past_indices[i] = t - 1 - past_stride * (past_frames - 1 - i)
        past = [
            local_state_from_world(
                history[-1 - past_stride * (past_frames - 1 - i)],
                anchor_xy, anchor_heading, anchor_pitch, anchor_roll,
            )
            for i in range(past_frames)
        ]

        frame_t = min(t, T - 1)

        # Dynamic loco→pickup: switch when root arrives within 0.5 m of pickup goal.
        goals = command.get("goals")
        if goals is not None and command["task_phase"][frame_t, 1] > 0.5:
            pickup_xy = goals["pickup"][:2]
            dist = float(np.linalg.norm(anchor_xy - pickup_xy))
            if dist < 0.5:
                n_pu = command.get("pickup_frames") or max(4, int(0.9 * 30))
                n_carry = command.get("carry_frames") or max(4, int(2.5 * 30))
                n_pl = command.get("place_frames") or max(4, int(1.5 * 30))
                T = frame_t + n_pu + n_carry + n_pl
                new_phase = np.zeros((T, 5), dtype=np.float32)
                new_phase[:frame_t] = command["task_phase"][:frame_t]
                new_phase[frame_t:frame_t + n_pu, 2] = 1.0
                new_phase[frame_t + n_pu:frame_t + n_pu + n_carry, 3] = 1.0
                new_phase[frame_t + n_pu + n_carry:, 4] = 1.0
                command["task_phase"] = new_phase
                command["T"] = T
                frame_t = min(t, T - 1)

        # Dynamic carry→place: switch as soon as root reaches place goal XY.
        if goals is not None and command["task_phase"][frame_t, 3] > 0.5:
            place_xy = goals["place"][:2]
            dist = float(np.linalg.norm(anchor_xy - place_xy))
            if dist < 0.5:
                n_place = command.get("place_frames", max(4, int(1.5 * 30)))
                T = frame_t + n_place
                new_phase = np.zeros((T, 5), dtype=np.float32)
                new_phase[:frame_t] = command["task_phase"][:frame_t]
                new_phase[frame_t:, 4] = 1.0  # place
                command["task_phase"] = new_phase
                command["T"] = T
                frame_t = min(t, T - 1)

        task_phase = command["task_phase"][frame_t].copy()  # (5,)

        # Desired travel direction steers locomotion; stand/goal phases get the null token.
        # Training semantics: travel direction unit vector in anchor frame.
        # command["heading"] is the desired travel direction angle in world frame.
        if task_phase[1] > 0.5:
            h_idx = min(t + int(stats["horizon"]) - 1, T - 1)
            h_rel = angle_rel(float(command["heading"][h_idx]), anchor_heading)
            desired_heading = np.array([np.cos(h_rel), np.sin(h_rel)], dtype=np.float32)
        else:
            desired_heading = np.zeros(2, dtype=np.float32)

        active_goal = get_active_goal_local(
            command, frame_t, anchor_xy, anchor_z, anchor_heading
        )

        extras = []
        if stats.get("use_target_speed", False):
            extras = [np.array([command["target_speed"][frame_t]], dtype=np.float32)]
        cond = np.concatenate(past + [task_phase, desired_heading, active_goal] + extras).astype(
            np.float32
        )
        target = sample_horizon(model, cond, stats, device, steps, past_guidance)

        take = min(stride, T - t)
        for j in range(take):
            state = state_from_target_frame(
                target[j], anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            history.append(state)
            # Display only at 30fps boundaries (every past_stride-th frame); S==1 display all
            if past_stride == 1 or (j + 1) % past_stride == 0:
                pred.append(state["joints"])
        t += take

    return np.stack(pred, axis=0)


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------


_PHASE_COLORS = {
    0: "#888888",  # stand  — gray
    1: "#4477CC",  # loco   — blue
    2: "#44AA44",  # pickup — green
    3: "#DDBB00",  # carry  — yellow
    4: "#CC3333",  # place  — red
}


def render_gif(
    path: str,
    pred: np.ndarray,
    title: str,
    fps: float,
    render_fps: float = None,
    phase_colors: np.ndarray = None,  # int array (T,), one phase index per pred frame
    goals: dict = None,               # {"pickup": (3,), "place": (3,)} world-frame positions
) -> None:
    from PIL import Image as PILImage

    if render_fps is None:
        render_fps = fps
    step = max(1, round(fps / render_fps))
    frames = pred[::step]
    actual_fps = fps / step
    colors = phase_colors[::step] if phase_colors is not None else None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(0, 2.0)
    ax.view_init(elev=15, azim=-60)
    lines, pts = draw_bones(ax, frames[0])

    # Fixed world-frame markers for pickup and place goals
    if goals is not None:
        pu = goals["pickup"]
        pl = goals["place"]
        ax.scatter([pu[0]], [pu[1]], [pu[2]], marker="*", s=200, c="#44AA44",
                   zorder=10, label="pickup")
        ax.scatter([pl[0]], [pl[1]], [pl[2]], marker="*", s=200, c="#CC3333",
                   zorder=10, label="place")
        ax.legend(loc="upper left", fontsize=8)

    pil_frames = []
    for fi, j in enumerate(frames):
        for (p, c), line in zip(BONES, lines):
            line.set_data_3d([j[p, 0], j[c, 0]], [j[p, 1], j[c, 1]], [j[p, 2], j[c, 2]])
            if colors is not None:
                line.set_color(_PHASE_COLORS[int(colors[fi])])
        pts._offsets3d = (j[:, 0], j[:, 1], j[:, 2])
        if colors is not None:
            pts.set_color(_PHASE_COLORS[int(colors[fi])])
        cx, cy = float(np.nan_to_num(j[0, 0])), float(np.nan_to_num(j[0, 1]))
        ax.set_xlim(cx - 1.5, cx + 1.5)
        ax.set_ylim(cy - 1.5, cy + 1.5)
        fig.suptitle(f"{title} | frame {fi}/{len(frames)}", fontsize=9)
        fig.canvas.draw()
        # tostring_argb is ARGB; convert to RGB by dropping channel 0 and reordering
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        argb = buf.reshape(h, w, 4)
        rgb = argb[:, :, [1, 2, 3]]  # drop A, keep R G B
        pil_frames.append(PILImage.fromarray(rgb))

    plt.close(fig)
    duration_ms = int(1000.0 / actual_fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=1,  # play once, don't loop
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument(
        "--motion",
        choices=["stand", "constant_walk", "curved_walk", "stop_start_walk",
                 "pick_carry_place", "walk_pcp"],
        default="constant_walk",
    )
    p.add_argument("--turn-rate", type=float, default=30.0,
                   help="Commanded yaw rate (deg/s) for --motion curved_walk")
    p.add_argument("--walk-secs", type=float, default=2.5,
                   help="Locomotion duration (s) before pcp for --motion walk_pcp")
    p.add_argument("--pickup-secs", type=float, default=2.5,
                   help="Pickup duration (s) for --motion walk_pcp")
    p.add_argument("--carry-secs", type=float, default=3.0,
                   help="Carry duration (s) for --motion walk_pcp")
    p.add_argument("--place-secs", type=float, default=2.5,
                   help="Place duration (s) for --motion walk_pcp")
    p.add_argument("--out", default="gif/x15_horizon.gif")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fps", type=float, default=None,
                   help="Motion FPS (default: read from stats train_args, fallback 30)")
    p.add_argument("--render-fps", type=float, default=30.0,
                   help="GIF output FPS; frames are subsampled from motion FPS (default 30)")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--speed", type=float, default=1.3,
                   help="Desired root speed m/s for locomotion phases (default 1.3)")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument(
        "--past-guidance",
        type=float,
        default=0.0,
        help="PARC-style blend weight for the no-past-frames branch "
        "(0=off; ~0.65 typical). Requires --past-dropout training.",
    )
    p.add_argument(
        "--warm-start-hdf5",
        default=None,
        help="Seed the past-frame history from K consecutive GT frames of a "
        "real clip (re-based to origin/heading 0) instead of default_state.",
    )
    p.add_argument("--warm-start-clip", type=int, default=0)
    p.add_argument(
        "--warm-start-frame",
        type=int,
        default=0,
        help="First GT frame of the K-frame warm-start window.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)
    validate_x15_stats_compatible(stats)

    # resolve motion fps: CLI > stats train_args > fallback 30
    if args.fps is None:
        train_args = stats.get("train_args") or {}
        args.fps = float(train_args.get("motion_fps") or 30.0)

    model = load_model(args.checkpoint, stats, device)
    command = make_commands(args)

    init_history = None
    if args.warm_start_hdf5:
        from dataset_horizon_x15 import JointFlowX15Dataset

        ds = JointFlowX15Dataset(
            [args.warm_start_hdf5],
            past_frames=int(stats["past_frames"]),
            horizon=int(stats["horizon"]),
            stats=stats,
        )
        clip = ds._clips[args.warm_start_clip]
        K = int(stats["past_frames"])
        ps = int(stats.get("past_stride", 1))
        min_hist = 1 + ps * (K - 1)  # frames needed to satisfy strided past indexing
        f0 = args.warm_start_frame
        if f0 + min_hist > len(clip["root_xy"]):
            raise ValueError(
                f"warm-start window [{f0}, {f0 + min_hist}) exceeds clip length {len(clip['root_xy'])}"
            )
        states = [
            {
                "root_xy": clip["root_xy"][f],
                "root_z": float(clip["root_z"][f]),
                "heading": float(clip["root_heading"][f]),
                "pitch": float(clip["root_pitch"][f]),
                "roll": float(clip["root_roll"][f]),
                "joint_rel": clip["joint_rel"][f],
            }
            for f in range(f0, f0 + min_hist)
        ]
        init_history = rebase_history(states)
        print(
            f"warm start: clip {args.warm_start_clip} frames [{f0}, {f0 + K}) of {args.warm_start_hdf5}"
        )

    pred = generate(
        model,
        stats,
        command,
        device,
        args.steps,
        args.stride,
        args.past_guidance,
        init_history,
    )
    if args.motion == "curved_walk":
        pelvis = pred[:, 0, :2]
        seg = np.diff(pelvis, axis=0)
        step = np.linalg.norm(seg, axis=1)
        ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        actual_turn = float(np.degrees(ang[-1] - ang[0]))
        cmd_turn = args.turn_rate * args.duration
        path_len = float(step.sum())
        print(
            f"curved_walk: commanded turn {cmd_turn:+.1f} deg ({args.turn_rate:+.1f} deg/s), "
            f"actual {actual_turn:+.1f} deg ({actual_turn/args.duration:+.1f} deg/s), "
            f"ratio {actual_turn/cmd_turn:.2f} | "
            f"mean speed {path_len/args.duration:.2f} m/s (cmd {args.speed:.2f})"
        )

    if args.motion in ("walk_pcp", "pick_carry_place"):
        names = ["stand", "loco", "pickup", "carry", "place"]
        labels = command["task_phase"].argmax(1)
        n = min(len(pred), len(labels))
        pelvis = pred[:n, 0, :]
        lab = labels[:n]
        # contiguous phase segments in temporal order
        bounds = [0] + [i for i in range(1, n) if lab[i] != lab[i - 1]] + [n]
        print("per-phase motion (xy displacement | mean pelvis z):")
        for a, b in zip(bounds[:-1], bounds[1:]):
            disp = float(np.linalg.norm(pelvis[b - 1, :2] - pelvis[a, :2]))
            print(f"  {names[lab[a]]:7s} f[{a:3d}:{b:3d}] "
                  f"disp={disp:.2f}m  z=[{pelvis[a:b,2].min():.2f},{pelvis[a:b,2].max():.2f}]")

    # Build per-frame phase color index from command task_phase (5D one-hot).
    # pred may be shorter than T if past_stride > 1; clamp to pred length.
    phase_labels = command["task_phase"].argmax(1)  # (T,) int
    phase_labels = phase_labels[: len(pred)]
    if len(phase_labels) < len(pred):
        phase_labels = np.pad(phase_labels, (0, len(pred) - len(phase_labels)), mode="edge")

    render_gif(args.out, pred, args.motion, args.fps, args.render_fps,
               phase_colors=phase_labels, goals=command.get("goals"))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
