"""
Save a grid of frame-by-frame pred vs GT comparisons.

Usage:
    python examples/joint_flow/compare.py \
        --checkpoint checkpoints/joint_flow/best.pt \
        --stats      checkpoints/joint_flow/stats.json \
        --hdf5       /path/to/accad_research_filtered_taskspace.h5 \
        --clip       58 \
        --n-frames   12 \       # how many frames to sample across the clip
        --out        compare.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from animate import bone_color, BONES, load_clip, run_flow
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from report import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--clip", type=int, default=58)
    p.add_argument("--n-frames", type=int, default=12, help="frames to sample evenly")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="compare.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def draw_skeleton_ax(ax, joints, lw=1.5, alpha=1.0):
    for p, c in BONES:
        ax.plot(
            [joints[p, 0], joints[c, 0]],
            [joints[p, 1], joints[c, 1]],
            [joints[p, 2], joints[c, 2]],
            color=bone_color(c),
            linewidth=lw,
            alpha=alpha,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="black", s=8, zorder=5)


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)
    window = stats.get("window", 10)

    model, window, prior = load_model(args.checkpoint, stats, device)

    ts_world, jp_gt, clip_fps, clip_name, obj_rel = load_clip(args.hdf5, args.clip)

    print("Running flow inference …")
    pred = run_flow(
        model,
        stats,
        ts_world,
        window,
        device,
        steps=args.steps,
        prior=prior,
        obj_rel=obj_rel,
    )
    print("Done.")

    # root-relative (pelvis-centered) for both
    pred_rel = pred - pred[:, 0:1, :]
    gt_rel = jp_gt - jp_gt[:, 0:1, :]

    # evenly spaced frame indices
    T = len(pred)
    indices = np.linspace(0, T - 1, args.n_frames, dtype=int)

    ncols = args.n_frames
    fig = plt.figure(figsize=(ncols * 2.5, 6))
    fig.suptitle(clip_name, fontsize=9, y=1.01)

    for col, t in enumerate(indices):
        lim = 0.8

        # --- prediction (already pelvis-centered) ---
        ax_p = fig.add_subplot(2, ncols, col + 1, projection="3d")
        draw_skeleton_ax(ax_p, pred_rel[t])
        ax_p.set_xlim(-lim, lim)
        ax_p.set_ylim(-lim, lim)
        ax_p.set_zlim(-0.2, 1.8)
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_zticks([0, 1])
        ax_p.set_title(f"f={t}", fontsize=7)
        if col == 0:
            ax_p.set_ylabel("Pred", fontsize=8, labelpad=10)
        ax_p.view_init(elev=10, azim=-70)

        # --- ground truth (pelvis-centered) ---
        ax_g = fig.add_subplot(2, ncols, ncols + col + 1, projection="3d")
        draw_skeleton_ax(ax_g, gt_rel[t])
        ax_g.set_xlim(-lim, lim)
        ax_g.set_ylim(-lim, lim)
        ax_g.set_zlim(-0.2, 1.8)
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        ax_g.set_zticks([0, 1])
        if col == 0:
            ax_g.set_ylabel("GT", fontsize=8, labelpad=10)
        ax_g.view_init(elev=10, azim=-70)

    # MPJPE in root-relative space
    mpjpe_per_frame = np.linalg.norm(pred_rel - gt_rel, axis=-1).mean(axis=-1)  # (T,)
    mean_mpjpe = mpjpe_per_frame.mean() * 1000  # m → mm

    fig.text(
        0.5,
        -0.01,
        f"Mean MPJPE: {mean_mpjpe:.1f} mm  |  top row = flow prediction, bottom = GT",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")

    txt_path = Path(args.out).with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"clip:       {clip_name}\n")
        f.write(f"n_frames:   {T}\n")
        f.write(f"mean_mpjpe: {mean_mpjpe:.2f} mm\n\n")
        f.write("frame, mpjpe_mm\n")
        for i, v in enumerate(mpjpe_per_frame):
            f.write(f"{i:5d}, {v*1000:.2f}\n")

    print(f"Saved → {args.out}  |  {txt_path}  |  Mean MPJPE: {mean_mpjpe:.1f} mm")


if __name__ == "__main__":
    main()
