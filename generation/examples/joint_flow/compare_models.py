"""
Compare two checkpoints on the same clip: frame-by-frame L2 error and smoothness.

Usage:
    python examples/joint_flow/compare_models.py \
        --ckpt-a  checkpoints/joint_flow_singleframe/best.pt \
        --stats-a checkpoints/joint_flow_singleframe/stats.json \
        --label-a "Reference" \
        --ckpt-b  checkpoints/joint_flow_prevframe/best.pt \
        --stats-b checkpoints/joint_flow_prevframe/stats.json \
        --label-b "Prev-Frame Conditioned" \
        --hdf5    /path/to/dataset.h5 \
        --clip    470 \
        --out     comparison_kit183.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import JointFlowMLP, CVAEPrior
from animate import load_clip, run_flow, draw_bones, BONES


def load_model(ckpt_path, stats, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args", {})
    window     = stats.get("window", 10)
    prev_dim   = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim     = (3 if stats.get("no_eef", False) else 9) + (3 if stats.get("use_obj", False) else 0)
    use_cvae   = train_args.get("use_cvae", False)
    latent_dim = train_args.get("latent_dim", 64)
    flow_cond_dim = latent_dim if use_cvae else prev_dim + window * ts_dim

    model = JointFlowMLP(
        joint_dim=69,
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

    return model, window, prior


def run_inference(ckpt_path, stats_path, device, hdf5, clip_idx, steps, seed=42):
    with open(stats_path) as f:
        stats = json.load(f)
    model, window, prior = load_model(ckpt_path, stats, device)
    ts_world, jp_gt, clip_fps, clip_name, obj_rel = load_clip(hdf5, clip_idx)
    torch.manual_seed(seed)
    pred = run_flow(model, stats, ts_world, window, device,
                    steps=steps, fps=clip_fps, init_pelvis_xy=jp_gt[0, 0, :2],
                    prior=prior, obj_rel=obj_rel)
    return pred, jp_gt, clip_fps, clip_name


def per_frame_l2(pred, gt):
    """Mean L2 across all 24 joints per frame. Returns (T,)."""
    diff = pred - gt                          # (T, 24, 3)
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)  # (T,)


def smoothness_score(seq):
    """
    Mean frame-to-frame joint displacement (lower = smoother).
    seq: (T, 24, 3)
    Returns scalar in metres.
    """
    deltas = np.linalg.norm(seq[1:] - seq[:-1], axis=-1)  # (T-1, 24)
    return float(deltas.mean())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a",  required=True)
    p.add_argument("--stats-a", required=True)
    p.add_argument("--label-a", default="Model A")
    p.add_argument("--ckpt-b",  required=True)
    p.add_argument("--stats-b", required=True)
    p.add_argument("--label-b", default="Model B")
    p.add_argument("--ckpt-c",  default=None)
    p.add_argument("--stats-c", default=None)
    p.add_argument("--label-c", default="Model C")
    p.add_argument("--ckpt-d",  default=None)
    p.add_argument("--stats-d", default=None)
    p.add_argument("--label-d", default="Model D")
    p.add_argument("--hdf5",    required=True)
    p.add_argument("--clip",    type=int, default=0)
    p.add_argument("--steps",   type=int, default=20)
    p.add_argument("--out",     default="comparison.png")
    p.add_argument("--gif",     default=None, help="also save a 3-panel GIF to this path")
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Running {args.label_a} …")
    pred_a, jp_gt, clip_fps, clip_name = run_inference(
        args.ckpt_a, args.stats_a, device, args.hdf5, args.clip, args.steps)

    print(f"Running {args.label_b} …")
    pred_b, _, _, _ = run_inference(
        args.ckpt_b, args.stats_b, device, args.hdf5, args.clip, args.steps)

    use_c = args.ckpt_c is not None
    if use_c:
        print(f"Running {args.label_c} …")
        pred_c, _, _, _ = run_inference(
            args.ckpt_c, args.stats_c, device, args.hdf5, args.clip, args.steps)

    use_d = args.ckpt_d is not None
    if use_d:
        print(f"Running {args.label_d} …")
        pred_d, _, _, _ = run_inference(
            args.ckpt_d, args.stats_d, device, args.hdf5, args.clip, args.steps)

    models = [(args.label_a, pred_a, "#E05252"),
              (args.label_b, pred_b, "#4A90D9")]
    if use_c:
        models.append((args.label_c, pred_c, "#52C05A"))
    if use_d:
        models.append((args.label_d, pred_d, "#E0A020"))

    T     = len(pred_a)
    t_sec = np.arange(T) / clip_fps

    smooth_gt = smoothness_score(jp_gt)

    # ---- print summary ----
    print(f"\nClip: {clip_name}  ({T} frames @ {clip_fps:.0f} fps)")
    col_w = 22
    header = f"\n{'Metric':<30} {'GT':>10}" + "".join(f"{lbl:>{col_w}}" for lbl, _, _ in models)
    print(header)
    print("-" * (40 + col_w * len(models)))

    errs = [per_frame_l2(pred, jp_gt) for _, pred, _ in models]
    smooths = [smoothness_score(pred) for _, pred, _ in models]

    print(f"{'Mean L2 (m)':<30} {'—':>10}" + "".join(f"{e.mean():>{col_w}.4f}" for e in errs))
    print(f"{'Median L2 (m)':<30} {'—':>10}" + "".join(f"{np.median(e):>{col_w}.4f}" for e in errs))
    print(f"{'Max L2 (m)':<30} {'—':>10}" + "".join(f"{e.max():>{col_w}.4f}" for e in errs))
    print(f"{'Smoothness (m/frame)':<30} {smooth_gt:>10.4f}" + "".join(f"{s:>{col_w}.4f}" for s in smooths))
    ratio_str = "  |  ".join(f"{lbl} {s/smooth_gt:.3f}x" for (lbl, _, _), s in zip(models, smooths))
    print(f"\nSmoothness ratio vs GT:  {ratio_str}  (1.0 = GT smooth)")

    # ---- per-joint mean L2 ----
    from animate import JOINT_NAMES
    JOINT_GROUPS = {
        "Legs":  [1, 2, 3, 4, 5, 6, 7, 8],
        "Spine": [9, 10, 11, 12, 13],
        "Arms":  [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    }
    print(f"\n{'Per-joint mean L2 (cm)':<26}" + "".join(f"{lbl:>{col_w}}" for lbl, _, _ in models))
    print("-" * (26 + col_w * len(models)))
    for group, idxs in JOINT_GROUPS.items():
        # mean over frames and joints in group
        group_errs = [np.linalg.norm(pred[:, idxs, :] - jp_gt[:, idxs, :], axis=-1).mean() * 100
                      for _, pred, _ in models]
        print(f"  {group:<24}" + "".join(f"{e:>{col_w}.2f}" for e in group_errs))
    print()
    for ji, jname in enumerate(JOINT_NAMES):
        joint_errs = [np.linalg.norm(pred[:, ji, :] - jp_gt[:, ji, :], axis=-1).mean() * 100
                      for _, pred, _ in models]
        print(f"  {jname:<24}" + "".join(f"{e:>{col_w}.2f}" for e in joint_errs))

    # ---- plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{clip_name}", fontsize=11)

    ax = axes[0]
    for (lbl, pred, color), err in zip(models, errs):
        ax.plot(t_sec, err * 100, label=lbl, color=color, linewidth=0.8, alpha=0.85)
        ax.axhline(err.mean() * 100, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_ylabel("Mean L2 error (cm)")
    ax.legend(fontsize=9)
    ax.set_title("Frame-by-frame mean L2 error (all joints)", fontsize=9)
    ax.grid(True, alpha=0.3)

    disp_gt = np.linalg.norm(jp_gt[1:] - jp_gt[:-1], axis=-1).mean(axis=-1) * 100
    ax2 = axes[1]
    ax2.plot(t_sec[1:], disp_gt, label="Ground truth", color="#888888", linewidth=0.8, alpha=0.7)
    for lbl, pred, color in models:
        disp = np.linalg.norm(pred[1:] - pred[:-1], axis=-1).mean(axis=-1) * 100
        ax2.plot(t_sec[1:], disp, label=lbl, color=color, linewidth=0.8, alpha=0.85)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mean joint displacement (cm/frame)")
    ax2.set_title("Frame-to-frame joint displacement (smoothness — lower & closer to GT = better)", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")

    if args.gif:
        panels = models + [("Ground truth", jp_gt, "#888888")]
        n_panels = len(panels)
        # pre-compute per-frame L2 for model panels (GT has no error vs itself)
        panel_errs = [per_frame_l2(seq, jp_gt) for _, seq, _ in models] + [None]
        panel_mean_l2 = [e.mean() * 100 if e is not None else None for e in panel_errs]

        fig2 = plt.figure(figsize=(5 * n_panels, 6))
        axes_gif = [fig2.add_subplot(1, n_panels, i + 1, projection="3d") for i in range(n_panels)]
        ax_titles = []
        for ax, (lbl, _, _), mean_l2 in zip(axes_gif, panels, panel_mean_l2):
            title = f"{lbl}\nmean={mean_l2:.2f} cm" if mean_l2 is not None else lbl
            t_obj = ax.set_title(title, fontsize=9)
            ax_titles.append(t_obj)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_zlim(0, 2.0)
            ax.view_init(elev=10, azim=-70)

        panel_lines = []
        panel_pts   = []
        for ax, (_, seq, _) in zip(axes_gif, panels):
            ln, pt = draw_bones(ax, seq[0])
            panel_lines.append(ln)
            panel_pts.append(pt)

        def update_gif(t):
            lim = 0.9
            all_artists = []
            for ax, (lbl, seq, _), lines, pts, err, mean_l2, title_obj in zip(
                    axes_gif, panels, panel_lines, panel_pts, panel_errs, panel_mean_l2, ax_titles):
                joints = seq[t]
                for (p, c), ln in zip(BONES, lines):
                    ln.set_data_3d([joints[p,0],joints[c,0]],
                                   [joints[p,1],joints[c,1]],
                                   [joints[p,2],joints[c,2]])
                pts._offsets3d = (joints[:,0], joints[:,1], joints[:,2])
                cx, cy = joints[0,0], joints[0,1]
                ax.set_xlim(cx-lim, cx+lim); ax.set_ylim(cy-lim, cy+lim)
                if err is not None:
                    title_obj.set_text(f"{lbl}\nmean={mean_l2:.2f} cm  |  now={err[t]*100:.2f} cm")
                all_artists += lines + [pts]
            fig2.suptitle(f"{clip_name}  |  frame {t}/{T}", fontsize=9)
            return all_artists

        ani = animation.FuncAnimation(fig2, update_gif, frames=T,
                                      interval=1000.0/args.fps, blit=False)
        writer = "pillow" if args.gif.endswith(".gif") else "ffmpeg"
        ani.save(args.gif, writer=writer, fps=args.fps)
        print(f"Saved GIF → {args.gif}")


if __name__ == "__main__":
    main()
