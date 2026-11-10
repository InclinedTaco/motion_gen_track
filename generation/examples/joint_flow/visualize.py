"""
Quick matplotlib 3D skeleton visualizer for joint_flow inference output.

Usage:
    python examples/joint_flow/visualize.py \
        --checkpoint checkpoints/joint_flow/best.pt \
        --stats      checkpoints/joint_flow/stats.json \
        --cond "0.5 0.0 0.95 0.2 0.3 1.1 -0.2 0.3 1.1" \
        --out        pose.png          # optional, shows interactive window if omitted
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from flow_matching.solver import ODESolver

from model import JointFlowMLP
from infer import CondWrapper, build_cond_window


JOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip",  "R_Knee","R_Ankle","R_Toe",
    "Torso",  "Spine", "Chest",  "Neck",    "Head",
    "L_Thorax","L_Shoulder","L_Elbow","L_Wrist","L_Hand",
    "R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",
]

# (parent, child) index pairs that form bones
BONES = [
    # spine
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    # left leg
    (0, 1), (1, 2), (2, 3), (3, 4),
    # right leg
    (0, 5), (5, 6), (6, 7), (7, 8),
    # left arm
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),
    # right arm
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),
]


def run_inference(checkpoint: str, stats: dict, cond_str: str, steps: int, device: torch.device):
    ckpt       = torch.load(checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    window     = train_args.get("window", 10)

    cond_values = np.array([float(v) for v in cond_str.split()], dtype=np.float32)
    pelvis_height = cond_values[2] if len(cond_values) == 9 else cond_values[-7]

    task_norm  = build_cond_window(cond_values, window, stats)
    joint_hist = np.zeros(window * 69, dtype=np.float32)
    cond_norm  = np.concatenate([task_norm, joint_hist])
    cond_t     = torch.tensor(cond_norm, device=device).unsqueeze(0)

    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=window * (9 + 69),
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapped = CondWrapper(model, cond_t)
    x0 = torch.randn(1, 69, device=device)

    with torch.no_grad():
        solver = ODESolver(velocity_model=wrapped)
        x1 = solver.sample(
            x_init=x0,
            step_size=1.0 / steps,
            method="midpoint",
            time_grid=torch.linspace(0, 1, steps + 1, device=device),
            return_intermediates=False,
        )

    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)
    joint_std  = np.array(stats["joint_std"],  dtype=np.float32)
    joints_23  = (x1.cpu().numpy()[0] * joint_std + joint_mean).reshape(23, 3)
    joints_rel = np.concatenate([np.zeros((1, 3), dtype=np.float32), joints_23], axis=0)
    pelvis_world = np.array([0.0, 0.0, pelvis_height], dtype=np.float32)
    joints_world = joints_rel + pelvis_world

    return joints_world


def draw_skeleton(joints: np.ndarray, title: str, out: str | None):
    """joints: (24, 3)  world-frame XYZ"""
    fig = plt.figure(figsize=(6, 9))
    ax  = fig.add_subplot(111, projection="3d")

    # bones
    for p, c in BONES:
        xs = [joints[p, 0], joints[c, 0]]
        ys = [joints[p, 1], joints[c, 1]]
        zs = [joints[p, 2], joints[c, 2]]
        color = "#4A90D9" if JOINT_NAMES[c].startswith("L") else \
                "#E05252" if JOINT_NAMES[c].startswith("R") else "#888888"
        ax.plot(xs, ys, zs, color=color, linewidth=2)

    # joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c="black", s=20, zorder=5)

    # label a few key joints
    for idx in [0, 13, 18, 23]:
        ax.text(joints[idx, 0], joints[idx, 1], joints[idx, 2],
                f" {JOINT_NAMES[idx]}", fontsize=6, color="dimgray")

    lim = 0.8
    cx, cy, cz = joints[0]
    ax.set_xlim(cx - lim, cx + lim)
    ax.set_ylim(cy - lim, cy + lim)
    ax.set_zlim(0, 2.0)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=10, azim=-70)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats",      required=True)
    p.add_argument("--cond",       required=True)
    p.add_argument("--steps",  type=int, default=50)
    p.add_argument("--out",    default=None, help=".png path; omit for interactive window")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)

    joints = run_inference(args.checkpoint, stats, args.cond, args.steps, device)

    cond_vals = [float(v) for v in args.cond.split()]
    title = (f"vel=({cond_vals[0]:.2f},{cond_vals[1]:.2f})  "
             f"h={cond_vals[2]:.2f}  "
             f"L_hand=({cond_vals[3]:.2f},{cond_vals[4]:.2f},{cond_vals[5]:.2f})  "
             f"R_hand=({cond_vals[6]:.2f},{cond_vals[7]:.2f},{cond_vals[8]:.2f})")

    draw_skeleton(joints, title, args.out)


if __name__ == "__main__":
    main()
