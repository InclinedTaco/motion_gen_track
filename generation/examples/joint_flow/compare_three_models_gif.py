"""
4-panel side-by-side GIF: ground truth + three flow models on the SAME ACCAD clip.

Panels: GT | prevframe-EEF (pos) | no-EEF (world pos) | no-EEF 6D

All models are driven by the clip's own task-space stream (T,9). Each model
slices/transforms it as it needs (EEF model uses pelvis-relative 9d; no-EEF
models use the first 3 dims). The 6D model's axis-angle output is run through
the SMPL SkeletonTree to recover global joint positions, so every panel ends up
as (T,24,3) world-frame joints in the SAME coordinate frame as GT.

Usage (phc env — has h5py, torch, poselib):
    python examples/joint_flow/compare_three_models_gif.py --clip 58 --frames 360 \
        --out compare_three_accad.gif
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from animate import run_flow, run_flow_6d, draw_bones, BONES
from model import JointFlowMLP

CKPT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"
HDF5 = str(Path(__file__).resolve().parents[3] / "tracking/data/amass/hdf5/accad_research_filtered_taskspace.h5")
HUMANOID_XML = str(Path(__file__).resolve().parents[3] / "tracking/final_humanoid.xml")  # matches 6D model's Z-up convention

MODELS = [
    ("joint_flow_prevframe",       "prevframe EEF (pos)"),
    ("joint_flow_noeef_prevframe", "no-EEF (world pos)"),
    ("joint_flow_6d_rootpred",     "no-EEF 6D+root"),
]


def load_model(name, device):
    ckpt_path = CKPT_ROOT / name / "best.pt"
    stats_path = CKPT_ROOT / name / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window   = stats.get("window", 10)
    use_6d       = stats.get("use_6d", False)
    predict_root = stats.get("predict_root", False)
    joint_dim    = (144 if predict_root else 138) if use_6d else 69
    prev_dim     = joint_dim if stats.get("use_prev_frame", False) else 0
    ts_dim    = 3 if stats.get("no_eef", False) else 9
    cond_dim  = prev_dim + window * ts_dim
    model = JointFlowMLP(
        joint_dim=joint_dim,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stats, window


def predict_positions(name, model, stats, window, ts_world, init_xy, device, steps, fps):
    """Return (T,24,3) world-frame joint positions for any model variant."""
    if stats.get("use_6d", False):
        _phc_root = (Path(__file__).resolve().parents[3] / "tracking")
        sys.path.insert(0, str(_phc_root / "poselib"))
        from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
        from scipy.spatial.transform import Rotation as sRot

        sk_tree = SkeletonTree.from_mjcf(HUMANOID_XML)
        pose_aa, root_trans = run_flow_6d(
            model, stats, ts_world, window, device, steps=steps, fps=fps,
            init_pelvis_xy=init_xy,
        )
        T = len(pose_aa)
        quats = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat()  # xyzw
        quats_t = torch.from_numpy(quats).float().reshape(T, 24, 4)
        rt_t = torch.from_numpy(root_trans).float()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree, quats_t, rt_t, is_local=True
        )
        return sk_state.global_translation.numpy()
    return run_flow(model, stats, ts_world, window, device, steps=steps, fps=fps,
                    init_pelvis_xy=init_xy)


def main():
    p = argparse.ArgumentParser()
    # clip mode
    p.add_argument("--hdf5",   default=HDF5)
    p.add_argument("--clip",   type=int, default=58)
    p.add_argument("--frames", type=int, default=360, help="trim to first N frames (0 = all)")
    # synthetic mode
    p.add_argument("--vel-x",    type=float, default=None, help="constant forward velocity (synthetic mode)")
    p.add_argument("--vel-y",    type=float, default=0.0)
    p.add_argument("--height",   type=float, default=0.938)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps",      type=float, default=30.0)
    # shared
    p.add_argument("--steps",  type=int, default=20)
    p.add_argument("--out",    default="compare_three_accad.gif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.vel_x is not None:
        # synthetic mode — no GT panel
        fps = args.fps
        T   = int(args.duration * fps)
        dt  = 1.0 / fps
        ts_world = np.zeros((T, 9), dtype=np.float32)
        ts_world[:, 0] = args.vel_x
        ts_world[:, 1] = args.vel_y
        ts_world[:, 2] = args.height

        # EEF cols must be world-frame hand positions (run_flow subtracts pelvis internally).
        # Use the EEF model's cond_mean to get typical pelvis-relative hand offsets,
        # then add the pelvis trajectory so the model sees in-distribution inputs.
        _eef_stats_path = CKPT_ROOT / MODELS[0][0] / "stats.json"
        with open(_eef_stats_path) as _f:
            _eef_cm = np.array(json.load(_f)["cond_mean"], dtype=np.float32)
        left_rel  = _eef_cm[3:6]   # typical left  hand offset from pelvis
        right_rel = _eef_cm[6:9]   # typical right hand offset from pelvis
        pelvis_x  = np.arange(T, dtype=np.float32) * dt * args.vel_x
        pelvis_y  = np.arange(T, dtype=np.float32) * dt * args.vel_y
        ts_world[:, 3] = pelvis_x + left_rel[0]
        ts_world[:, 4] = pelvis_y + left_rel[1]
        ts_world[:, 5] = args.height + left_rel[2]
        ts_world[:, 6] = pelvis_x + right_rel[0]
        ts_world[:, 7] = pelvis_y + right_rel[1]
        ts_world[:, 8] = args.height + right_rel[2]

        jp_gt  = None
        cname  = f"synthetic  vel=({args.vel_x:.2f},{args.vel_y:.2f})  h={args.height:.2f}"
        init_xy = np.zeros(2, dtype=np.float32)
        print(f"Synthetic mode: {cname}  |  {T} frames @ {fps:.0f} fps")
    else:
        with h5py.File(args.hdf5, "r") as f:
            clips = sorted(f["clips"].keys())
            cname = clips[args.clip]
            ts_world = f[f"clips/{cname}/task_space"][:].astype(np.float32)
            jp_gt    = f[f"clips/{cname}/joint_positions"][:].astype(np.float32)
            fps      = float(f[f"clips/{cname}"].attrs["fps"])
        if args.frames and args.frames < len(ts_world):
            ts_world = ts_world[: args.frames]
            jp_gt    = jp_gt[: args.frames]
        init_xy = jp_gt[0, 0, :2]
        print(f"Clip {args.clip}: {cname}  |  {len(ts_world)} frames @ {fps:.0f} fps")

    preds = []
    for name, label in MODELS:
        print(f"\n=== {label}  ({name}) ===")
        model, stats, window = load_model(name, device)
        pos = predict_positions(name, model, stats, window, ts_world, init_xy,
                                 device, args.steps, fps)
        preds.append(pos)

    panels = ([] if jp_gt is None else [("Ground truth", jp_gt)]) + \
             [(lbl, pr) for (_, lbl), pr in zip(MODELS, preds)]
    n_panels = len(panels)
    T = min(len(p[1]) for p in panels)
    interval = 1000.0 / fps

    fig = plt.figure(figsize=(5.5 * n_panels, 5.5))
    axes = []
    for i, (title, _) in enumerate(panels):
        ax = fig.add_subplot(1, n_panels, i + 1, projection="3d")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_zlim(0, 2.0)
        ax.view_init(elev=10, azim=-70)
        axes.append(ax)

    artists = [draw_bones(ax, panel[0]) for ax, (_, panel) in zip(axes, panels)]

    def update(t):
        out = []
        for ax, (_, panel), (lines, pts) in zip(axes, panels, artists):
            j = panel[t]
            for (a, b), ln in zip(BONES, lines):
                ln.set_data_3d([j[a,0],j[b,0]], [j[a,1],j[b,1]], [j[a,2],j[b,2]])
            pts._offsets3d = (j[:,0], j[:,1], j[:,2])
            cx, cy = j[0,0], j[0,1]
            ax.set_xlim(cx-0.9, cx+0.9); ax.set_ylim(cy-0.9, cy+0.9)
            out += lines + [pts]
        fig.suptitle(f"{cname}  |  frame {t}/{T}", fontsize=10)
        return out

    print("\nRendering GIF …")
    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    ani.save(args.out, writer="pillow", fps=fps)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
