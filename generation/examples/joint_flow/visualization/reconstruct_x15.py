"""
Reconstruct a clip with an x15 (v7) checkpoint: GT warm start + GT-derived
commands, autoregressive receding-horizon rollout, predicted vs ground truth
rendered together (GT in gray).

Usage:
    python examples/joint_flow/reconstruct_x15.py \
        --checkpoint checkpoints/joint_flow_x15_horizon_position_v7/best.pt \
        --stats checkpoints/joint_flow_x15_horizon_position_v7/stats.json \
        --hdf5 final_h5_files/cmu_straight_walk_taskspace_with_root.h5 \
        --clip-idx 0 --out gif/x15_recon_cmu.gif
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # joint_flow

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from animate import BONES, draw_bones
from animate_horizon_x15 import (
    load_model, local_state_from_world, rot_to_anchor,
    sample_horizon, state_from_target_frame,
)
from dataset_horizon_x15 import JointFlowX15Dataset, validate_x15_stats_compatible


def clip_world_state(clip: dict, f: int) -> dict:
    return {
        "root_xy": clip["root_xy"][f],
        "root_z": float(clip["root_z"][f]),
        "heading": float(clip["root_heading"][f]),
        "pitch": float(clip["root_pitch"][f]),
        "roll": float(clip["root_roll"][f]),
        "joint_rel": clip["joint_rel"][f],
    }


def gt_joints(clip: dict, f: int) -> np.ndarray:
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [clip["root_xy"][f, 0], clip["root_xy"][f, 1], clip["root_z"][f]]
    joints[1:] = clip["joint_rel"][f].reshape(23, 3) + joints[0][None]
    return joints


def reconstruct(model, stats: dict, clip: dict, device, steps: int, stride: int,
                teacher_forced: bool = False):
    K = int(stats["past_frames"])
    S = int(stats.get("past_stride", 1))
    H = int(stats["horizon"])
    if stride is None:
        stride = H if S > 1 else max(1, H // 2)
    T = len(clip["root_xy"])

    # history must be long enough so the oldest strided frame exists
    min_history = 1 + S * (K - 1)
    history = [clip_world_state(clip, f) for f in range(min_history)]  # GT warm start
    pred = []
    gt_display_frames = []
    t = min_history
    while t < T:
        if teacher_forced:
            history = [clip_world_state(clip, f) for f in range(t - min_history, t)]
        anchor = history[-1]
        anchor_xy = anchor["root_xy"]
        anchor_z = float(anchor["root_z"])
        anchor_heading = float(anchor["heading"])
        anchor_pitch = float(anchor["pitch"])
        anchor_roll = float(anchor["roll"])

        # mirrors dataset_horizon_x15.py: t-1-S*(K-1-i) for i in range(K)
        past = [
            local_state_from_world(
                history[-1 - S * (K - 1 - i)],
                anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            for i in range(K)
        ]
        mode = clip["mode"][min(t, T - 1)].copy()
        if mode[0] > 0.5 or mode[1] > 0.5:  # stand or locomotion
            h_idx = min(t + H - 1, T - 1)
            disp = clip["root_xy"][h_idx] - anchor_xy
            dist = float(np.linalg.norm(disp))
            if dist < 0.01:
                desired_heading = np.zeros(2, dtype=np.float32)
            else:
                desired_heading = rot_to_anchor(disp / dist, anchor_heading)
        else:
            desired_heading = np.zeros(2, dtype=np.float32)
        if clip["has_obj"]:
            goals = clip["task_goals"][min(t, T - 1)].reshape(3, 3).copy()
            goals[:2, :2] = rot_to_anchor(goals[:2, :2] - anchor_xy, anchor_heading)
            goals[:2, 2] -= anchor_z
            active_goal = goals[1] if (mode[3] > 0.5 or mode[4] > 0.5) else goals[0]
        else:
            active_goal = np.zeros(3, dtype=np.float32)

        extras = []
        if stats.get("use_target_speed", False):
            spd = clip["clip_mean_speed"] if mode[1] > 0.5 else 0.0
            extras = [np.array([spd], dtype=np.float32)]
        cond = np.concatenate(past + [mode, desired_heading, active_goal] + extras).astype(np.float32)
        target = sample_horizon(model, cond, stats, device, steps)

        take = min(stride, T - t)
        for j in range(take):
            state = state_from_target_frame(
                target[j], anchor_xy, anchor_heading, anchor_pitch, anchor_roll
            )
            history.append(state)
            # Display only at 30fps boundaries (every S-th frame); for S==1 display all
            if S == 1 or (j + 1) % S == 0:
                pred.append(state["joints"])
                gt_display_frames.append(t + j)
        t += take

    gt = np.stack([gt_joints(clip, f) for f in gt_display_frames], axis=0)

    # Debug: log world-frame place goal vs predicted/GT agent positions at place phase
    if clip["has_obj"]:
        place_world = clip["task_goals"][0].reshape(3, 3)[1, :2]  # place XY in world frame
        print(f"  place goal world XY:  {place_world.round(3)}")
        pred_stack = np.stack(pred, axis=0)
        gt_stack = gt
        # find frames in place phase (mode[4] > 0.5)
        place_frames = [i for i in range(len(pred_stack))
                        if clip["mode"][min(i + K, T - 1)][4] > 0.5]
        if place_frames:
            pred_pelvis_place = pred_stack[place_frames[0], 0, :2]
            gt_pelvis_place   = gt_stack[place_frames[0], 0, :2]
            print(f"  first place-phase frame: {place_frames[0]+K} / {T}")
            print(f"  pred pelvis XY at place: {pred_pelvis_place.round(3)}")
            print(f"    GT pelvis XY at place: {gt_pelvis_place.round(3)}")
            # predicted agent → place goal distance
            dist_pred = np.linalg.norm(pred_stack[place_frames[-1], 0, :2] - place_world)
            dist_gt   = np.linalg.norm(gt_stack[place_frames[-1], 0, :2] - place_world)
            print(f"  dist pred-agent → place goal (last place frame): {dist_pred*100:.1f} cm")
            print(f"  dist   gt-agent → place goal (last place frame): {dist_gt*100:.1f} cm")
        else:
            print("  no place-phase frames found in this clip")

    return np.stack(pred, axis=0), gt


def render_gif(path: str, pred: np.ndarray, gt: np.ndarray, title: str, fps: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_zlim(0, 2.0)
    ax.view_init(elev=15, azim=-60)
    allxy = np.concatenate([pred[:, 0, :2], gt[:, 0, :2]], axis=0)
    ax.set_xlim(float(allxy[:, 0].min() - 1.0), float(allxy[:, 0].max() + 1.0))
    ax.set_ylim(float(allxy[:, 1].min() - 1.0), float(allxy[:, 1].max() + 1.0))

    gt_lines = []
    for p, c in BONES:
        (l,) = ax.plot([], [], [], color="0.65", linewidth=2, alpha=0.8)
        gt_lines.append(l)
    lines, pts = draw_bones(ax, pred[0])

    def update(fi):
        j = pred[fi]
        g = gt[fi]
        for (p, c), line, gl in zip(BONES, lines, gt_lines):
            line.set_data_3d([j[p, 0], j[c, 0]], [j[p, 1], j[c, 1]], [j[p, 2], j[c, 2]])
            gl.set_data_3d([g[p, 0], g[c, 0]], [g[p, 1], g[c, 1]], [g[p, 2], g[c, 2]])
        pts._offsets3d = (j[:, 0], j[:, 1], j[:, 2])
        fig.suptitle(f"{title} | frame {fi}/{len(pred)} | gray = GT", fontsize=9)
        return lines + gt_lines + [pts]

    ani = animation.FuncAnimation(fig, update, frames=len(pred), interval=1000.0 / fps, blit=False)
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--clip-idx", type=int, default=0)
    p.add_argument("--out", default="gif/x15_recon.gif")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--teacher-forced", action="store_true",
                   help="Rebuild history from GT at every replan (no autoregressive feedback)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-fps", type=float, default=None,
                   help="GIF display FPS; frames subsampled from --fps (default: same as --fps)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.stats) as f:
        stats = json.load(f)
    validate_x15_stats_compatible(stats)

    ds = JointFlowX15Dataset(
        [args.hdf5],
        past_frames=int(stats["past_frames"]),
        past_stride=int(stats.get("past_stride", 1)),
        horizon=int(stats["horizon"]),
        stats=stats,
    )
    clip = ds._clips[args.clip_idx]
    with h5py.File(args.hdf5, "r") as f:
        all_names = sorted(f["clips"].keys())
    clip_name = all_names[args.clip_idx] if args.clip_idx < len(all_names) else f"clip_{args.clip_idx}"
    if args.max_frames and len(clip["root_xy"]) > args.max_frames:
        clip = {k: (v[: args.max_frames] if isinstance(v, np.ndarray) else v) for k, v in clip.items()}
    print(f"clip {args.clip_idx} ({clip_name}): {len(clip['root_xy'])} frames, has_obj={clip['has_obj']}")

    model = load_model(args.checkpoint, stats, device)
    pred, gt = reconstruct(model, stats, clip, device, args.steps, args.stride,
                           teacher_forced=args.teacher_forced)

    joint_l2 = np.linalg.norm(pred - gt, axis=-1).mean()
    root_err = np.linalg.norm(pred[:, 0, :2] - gt[:, 0, :2], axis=-1)
    print(f"mean joint L2: {joint_l2 * 100:.2f} cm")
    print(f"root XY drift: mean {root_err.mean() * 100:.1f} cm  final {root_err[-1] * 100:.1f} cm")
    print(f"GT travel: {np.linalg.norm(gt[-1, 0, :2] - gt[0, 0, :2]):.2f} m  "
          f"pred travel: {np.linalg.norm(pred[-1, 0, :2] - pred[0, 0, :2]):.2f} m")

    render_fps = args.render_fps if args.render_fps else args.fps
    if render_fps < args.fps:
        step = max(1, round(args.fps / render_fps))
        pred = pred[::step]
        gt = gt[::step]
    render_gif(args.out, pred, gt, clip_name, render_fps)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
