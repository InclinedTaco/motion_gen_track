"""
Visualize the ODE flow trajectory in 2D (UMAP projection of 69d joint space).

Shows:
  - Real data distribution (grey cloud) — normalized 69d joint vectors from the clip
  - Flow trajectories from noise (t=0) → prediction (t=1) for N conditions
  - Animated across ODE steps

Usage:
    python examples/joint_flow/flow_viz.py \
        --checkpoint checkpoints/joint_flow_singleframe/best.pt \
        --stats      checkpoints/joint_flow_singleframe/stats.json \
        --hdf5       /path/to/dataset.h5 \
        --clip       58 \
        --n-samples  8 \
        --steps      20 \
        --out        flow_viz.gif
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
import umap

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from model import JointFlowMLP


class CondWrapper(ModelWrapper):
    def __init__(self, model, cond):
        super().__init__(model)
        self.cond = cond
    def forward(self, x, t, **kwargs):
        return self.model(x, t.reshape(1).expand(x.shape[0]), self.cond)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats",      required=True)
    p.add_argument("--hdf5",       required=True, nargs="+")
    p.add_argument("--n-motions",  type=int, default=6,   help="subplots / clips to sample")
    p.add_argument("--n-particles",type=int, default=6,   help="ODE particles per subplot")
    p.add_argument("--n-data",     type=int, default=2000, help="real data points for cloud")
    p.add_argument("--steps",      type=int, default=20)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--clip-idx",   type=int, default=None, help="fix a single clip by combined-list index (overrides random)")
    p.add_argument("--out",        default="flow_viz.gif")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)

    ckpt       = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    window     = train_args.get("window", 10)

    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=window * 9,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)  # (69,)
    joint_std  = np.array(stats["joint_std"],  dtype=np.float32)
    cond_mean  = np.array(stats["cond_mean"],  dtype=np.float32)  # (9,)
    cond_std   = np.array(stats["cond_std"],   dtype=np.float32)

    # --- collect all clips across all HDF5 files ---
    all_clips = []  # list of (hdf5_path, clip_name)
    for path in args.hdf5:
        with h5py.File(path, "r") as f:
            for name in sorted(f["clips"].keys()):
                all_clips.append((path, name))
    print(f"Total clips across all datasets: {len(all_clips)}")

    N_MOTIONS   = args.n_motions
    N_PARTICLES = args.n_particles

    rng = np.random.default_rng(args.seed)
    if args.clip_idx is not None:
        chosen = [args.clip_idx]
        N_MOTIONS = 1
    else:
        chosen = rng.choice(len(all_clips), size=N_MOTIONS, replace=False)

    # --- build real data cloud ---
    # if a specific clip is chosen, use only that clip; otherwise sample all clips
    rng2 = np.random.default_rng(args.seed + 1)
    if args.clip_idx is not None:
        cloud_clips = [all_clips[args.clip_idx]]
    else:
        cloud_clips = all_clips
    n_per_clip = max(1, args.n_data // len(cloud_clips))
    real_cloud_parts = []
    for path, cname in cloud_clips:
        with h5py.File(path, "r") as f:
            jp = f[f"clips/{cname}/joint_positions"][:]
        pw = jp[:, 0, :]
        jr = jp[:, 1:, :] - pw[:, None, :]
        jf = jr.reshape(len(jp), -1)
        jn = (jf - joint_mean) / joint_std
        idx = rng2.choice(len(jn), min(n_per_clip, len(jn)), replace=False)
        real_cloud_parts.append(jn[idx])
    real_cloud = np.concatenate(real_cloud_parts, axis=0)
    print(f"  {len(real_cloud)} points from {len(cloud_clips)} clip(s)")

    # conds[i] = list of N_PARTICLES conditions, one per evenly-spaced frame
    all_conds   = []   # (N_MOTIONS * N_PARTICLES, 90)
    subplot_labels = []

    for ci in chosen:
        path, name = all_clips[ci]
        with h5py.File(path, "r") as f:
            ts_world = f[f"clips/{name}/task_space"][:]
            jp_gt    = f[f"clips/{name}/joint_positions"][:]
        T = len(ts_world)
        pelvis_world = jp_gt[:, 0, :]

        # normalize task-space
        cond_raw = ts_world.copy()
        cond_raw[:, 3:6] -= pelvis_world
        cond_raw[:, 6:9] -= pelvis_world
        cond_norm = (cond_raw - cond_mean) / cond_std

        # pick N_PARTICLES evenly-spaced frames — spread across the clip
        frames = np.linspace(window, T - 1, N_PARTICLES, dtype=int)
        for t in frames:
            start = t - window + 1
            if start < 0:
                pad = cond_norm[0:1].repeat(-start, axis=0)
                window_cond = np.concatenate([pad, cond_norm[0:t+1]], axis=0)
            else:
                window_cond = cond_norm[start:t+1]
            all_conds.append(window_cond.reshape(-1))

        short = name[:30] + "…" if len(name) > 30 else name
        subplot_labels.append(short)
        print(f"  Subplot {len(subplot_labels)}: {name}  frames={frames.tolist()}")

    print("Running ODE …")
    cond_tensor = torch.tensor(np.stack(all_conds), dtype=torch.float32, device=device)  # (N_MOTIONS*N_PARTICLES, 90)
    x0 = torch.randn(N_MOTIONS * N_PARTICLES, 69, device=device)
    wrapped = CondWrapper(model, cond_tensor)
    with torch.no_grad():
        solver = ODESolver(velocity_model=wrapped)
        x_traj = solver.sample(
            x_init=x0,
            step_size=1.0 / args.steps,
            method="midpoint",
            time_grid=torch.linspace(0, 1, args.steps + 1, device=device),
            return_intermediates=True,
        )  # (steps+1, N_MOTIONS*N_PARTICLES, 69)

    traj_np = x_traj.cpu().numpy()  # (steps+1, N*P, 69)
    n_steps = traj_np.shape[0]
    # reshape → (steps+1, N_MOTIONS, N_PARTICLES, 69)
    traj_np = traj_np.reshape(n_steps, N_MOTIONS, N_PARTICLES, 69)

    # --- fit UMAP on real data + all trajectory points ---
    print("Fitting UMAP …")
    all_traj   = traj_np.reshape(-1, 69)                 # (steps*N*P, 69)
    all_points = np.concatenate([real_cloud, all_traj], axis=0)
    reducer    = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    all_2d     = reducer.fit_transform(all_points)

    real_2d = all_2d[:len(real_cloud)]
    traj_2d = all_2d[len(real_cloud):].reshape(n_steps, N_MOTIONS, N_PARTICLES, 2)

    # global axis limits — show full UMAP
    pad  = (all_2d.max() - all_2d.min()) * 0.03
    xlim = (all_2d[:, 0].min() - pad, all_2d[:, 0].max() + pad)
    ylim = (all_2d[:, 1].min() - pad, all_2d[:, 1].max() + pad)

    # --- subplots ---
    n_cols = N_MOTIONS
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if N_MOTIONS == 1:
        axes = [axes]
    p_colors = plt.cm.tab10(np.linspace(0, 1, N_PARTICLES))

    def update(s):
        for n, ax in enumerate(axes):
            ax.clear()
            ax.scatter(real_2d[:, 0], real_2d[:, 1],
                       c="lightgrey", s=5, alpha=0.3)
            for p in range(N_PARTICLES):
                ax.plot(traj_2d[:s+1, n, p, 0], traj_2d[:s+1, n, p, 1],
                        color=p_colors[p], linewidth=1.5, alpha=0.8)
                ax.scatter(traj_2d[0, n, p, 0], traj_2d[0, n, p, 1],
                           color="black", s=30, zorder=6, marker="x")
                ax.scatter(traj_2d[s, n, p, 0], traj_2d[s, n, p, 1],
                           color=p_colors[p], s=60, zorder=5)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(subplot_labels[n] + f"\nt={s/(n_steps-1):.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"ODE flow in 69d pose space (UMAP) — step {s}/{n_steps-1}", fontsize=10)

    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=200)
    plt.tight_layout()
    ani.save(args.out, writer="pillow")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
