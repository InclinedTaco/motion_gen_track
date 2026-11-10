"""
Side-by-side comparison of stride 1 vs stride N for an x15 checkpoint.

Modes:
  recon   -- GT-commanded reconstruction of an OMOMO clip
  walk    -- synthetic constant-walk generation

Usage:
    python examples/joint_flow/compare_strides_x15.py recon \
        --checkpoint checkpoints/.../best.pt \
        --stats checkpoints/.../stats.json \
        --hdf5 final_h5_files/physics_rollout/omomo_holdout_physics.h5 \
        --clip-idx 34 \
        --stride-b 5 \
        --out gif/compare_stride_recon.gif

    python examples/joint_flow/compare_strides_x15.py walk \
        --checkpoint checkpoints/.../best.pt \
        --stats checkpoints/.../stats.json \
        --duration 30 \
        --stride-b 5 \
        --out gif/compare_stride_walk.gif
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
from animate_horizon_x15 import generate, load_model, make_commands
from dataset_horizon_x15 import JointFlowX15Dataset, validate_x15_stats_compatible
from reconstruct_x15 import gt_joints, reconstruct


# ---------------------------------------------------------------------------
# Side-by-side renderer
# ---------------------------------------------------------------------------


def render_side_by_side(
    path: str,
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    fps: float,
    gt_left: np.ndarray = None,
    gt_right: np.ndarray = None,
) -> None:
    """Render two animations side by side. Optionally overlay GT in gray."""
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    if gt_left is not None:
        gt_left = gt_left[:n]
    if gt_right is not None:
        gt_right = gt_right[:n]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    def _setup_ax(ax, frames, gt=None):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_zlim(0, 2.0)
        ax.view_init(elev=15, azim=-60)
        if gt is not None:
            allxy = np.concatenate([frames[:, 0, :2], gt[:, 0, :2]], axis=0)
        else:
            allxy = frames[:, 0, :2]
        ax.set_xlim(allxy[:, 0].min() - 1.0, allxy[:, 0].max() + 1.0)
        ax.set_ylim(allxy[:, 1].min() - 1.0, allxy[:, 1].max() + 1.0)

    _setup_ax(ax_l, left, gt_left)
    _setup_ax(ax_r, right, gt_right)

    lines_l, pts_l = draw_bones(ax_l, left[0])
    lines_r, pts_r = draw_bones(ax_r, right[0])

    gt_lines_l, gt_lines_r = [], []
    if gt_left is not None:
        for p, c in BONES:
            (l,) = ax_l.plot([], [], [], color="0.65", linewidth=2, alpha=0.8)
            gt_lines_l.append(l)
    if gt_right is not None:
        for p, c in BONES:
            (l,) = ax_r.plot([], [], [], color="0.65", linewidth=2, alpha=0.8)
            gt_lines_r.append(l)

    ax_l.set_title(left_title, fontsize=9)
    ax_r.set_title(right_title, fontsize=9)

    def update(fi):
        jl, jr = left[fi], right[fi]
        for (p, c), line in zip(BONES, lines_l):
            line.set_data_3d([jl[p, 0], jl[c, 0]], [jl[p, 1], jl[c, 1]], [jl[p, 2], jl[c, 2]])
        pts_l._offsets3d = (jl[:, 0], jl[:, 1], jl[:, 2])
        for (p, c), line in zip(BONES, lines_r):
            line.set_data_3d([jr[p, 0], jr[c, 0]], [jr[p, 1], jr[c, 1]], [jr[p, 2], jr[c, 2]])
        pts_r._offsets3d = (jr[:, 0], jr[:, 1], jr[:, 2])

        # Follow camera for walk mode (no fixed global view)
        if gt_left is None:
            cx, cy = float(np.nan_to_num(jl[0, 0])), float(np.nan_to_num(jl[0, 1]))
            ax_l.set_xlim(cx - 1.5, cx + 1.5)
            ax_l.set_ylim(cy - 1.5, cy + 1.5)
        if gt_right is None:
            cx, cy = float(np.nan_to_num(jr[0, 0])), float(np.nan_to_num(jr[0, 1]))
            ax_r.set_xlim(cx - 1.5, cx + 1.5)
            ax_r.set_ylim(cy - 1.5, cy + 1.5)

        if gt_left is not None:
            gl = gt_left[fi]
            for (p, c), line in zip(BONES, gt_lines_l):
                line.set_data_3d([gl[p, 0], gl[c, 0]], [gl[p, 1], gl[c, 1]], [gl[p, 2], gl[c, 2]])
        if gt_right is not None:
            gr = gt_right[fi]
            for (p, c), line in zip(BONES, gt_lines_r):
                line.set_data_3d([gr[p, 0], gr[c, 0]], [gr[p, 1], gr[c, 1]], [gr[p, 2], gr[c, 2]])

        fig.suptitle(f"frame {fi}/{n}", fontsize=8)
        return lines_l + lines_r + gt_lines_l + gt_lines_r + [pts_l, pts_r]

    ani = animation.FuncAnimation(fig, update, frames=n, interval=1000.0 / fps, blit=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["recon", "walk"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride-a", type=int, default=1, help="Left panel stride (default 1)")
    p.add_argument("--stride-b", type=int, required=True, help="Right panel stride")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--render-fps", type=float, default=30.0)
    # recon args
    p.add_argument("--hdf5")
    p.add_argument("--clip-idx", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=300)
    # walk args
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)
    validate_x15_stats_compatible(stats)

    if args.fps is None:
        train_args = stats.get("train_args") or {}
        args.fps = float(train_args.get("motion_fps") or 30.0)

    model = load_model(args.checkpoint, stats, device)

    render_fps = args.render_fps if args.render_fps else args.fps
    step = max(1, round(args.fps / render_fps))

    if args.mode == "recon":
        if not args.hdf5:
            raise ValueError("--hdf5 required for recon mode")
        ds = JointFlowX15Dataset(
            [args.hdf5],
            past_frames=int(stats["past_frames"]),
            past_stride=int(stats.get("past_stride", 1)),
            horizon=int(stats["horizon"]),
            stats=stats,
        )
        clip = ds._clips[args.clip_idx]
        with h5py.File(args.hdf5, "r") as f:
            clip_name = sorted(f["clips"].keys())[args.clip_idx]
        if args.max_frames and len(clip["root_xy"]) > args.max_frames:
            clip = {k: (v[: args.max_frames] if isinstance(v, np.ndarray) else v) for k, v in clip.items()}
        print(f"clip {args.clip_idx} ({clip_name}): {len(clip['root_xy'])} frames")

        print(f"Running stride {args.stride_a} ...")
        pred_a, gt_a = reconstruct(model, stats, clip, device, args.steps, args.stride_a)
        print(f"Running stride {args.stride_b} ...")
        pred_b, gt_b = reconstruct(model, stats, clip, device, args.steps, args.stride_b)

        # align lengths
        n = min(len(pred_a), len(pred_b))
        pred_a, pred_b = pred_a[:n:step], pred_b[:n:step]
        gt_a, gt_b = gt_a[:n:step], gt_b[:n:step]

        for tag, pred, gt in [("stride1", pred_a, gt_a), (f"stride{args.stride_b}", pred_b, gt_b)]:
            l2 = np.linalg.norm(pred - gt, axis=-1).mean()
            drift = np.linalg.norm(pred[:, 0, :2] - gt[:, 0, :2], axis=-1).mean()
            print(f"  {tag}: L2={l2*100:.2f}cm  root_drift={drift*100:.1f}cm")

        render_side_by_side(
            args.out,
            pred_a, pred_b,
            f"stride {args.stride_a} (L=left)", f"stride {args.stride_b} (R=right)",
            render_fps,
            gt_left=gt_a, gt_right=gt_b,
        )

    elif args.mode == "walk":
        # Build a constant-walk command dict by patching args
        class _WalkArgs:
            motion = "constant_walk"
            duration = args.duration
            speed = args.speed
            fps = args.fps
        command = make_commands(_WalkArgs())

        print(f"Running stride {args.stride_a} ...")
        pred_a = generate(model, stats, command, device, args.steps, args.stride_a, 0.0, None)
        print(f"Running stride {args.stride_b} ...")
        pred_b = generate(model, stats, command, device, args.steps, args.stride_b, 0.0, None)

        n = min(len(pred_a), len(pred_b))
        pred_a, pred_b = pred_a[:n:step], pred_b[:n:step]

        render_side_by_side(
            args.out,
            pred_a, pred_b,
            f"stride {args.stride_a}", f"stride {args.stride_b}",
            render_fps,
        )


if __name__ == "__main__":
    main()
