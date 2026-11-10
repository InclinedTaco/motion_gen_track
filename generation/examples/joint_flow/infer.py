"""
Inference: given a K-frame task-space condition window, generate full-body joint positions.

Usage:
    # Single frame (will be repeated K times to fill the window):
    python examples/joint_flow/infer.py \
        --checkpoint checkpoints/joint_flow/best.pt \
        --stats      checkpoints/joint_flow/stats.json \
        --cond "0.5 0.0 0.95 0.2 0.3 1.1 -0.2 0.3 1.1"

    # Explicit K-frame window (K*9 values, oldest first):
    python examples/joint_flow/infer.py \
        --checkpoint checkpoints/joint_flow/best.pt \
        --stats      checkpoints/joint_flow/stats.json \
        --cond "0.4 0.0 0.94 ... 0.5 0.0 0.95 ..."   # K*9 floats

    Each frame is 9 world-frame floats:
        [root_vel_x, root_vel_y, pelvis_height,
         left_hand_x,  left_hand_y,  left_hand_z,
         right_hand_x, right_hand_y, right_hand_z]

    The script normalizes each frame to root-relative, runs the ODE, then prints
    the 24×3 world-frame joint positions (pelvis assumed at world origin of last frame).

Output:
    Prints the (24, 3) joint positions and optionally saves to --out (numpy .npy).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from model import JointFlowMLP


JOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip",  "R_Knee","R_Ankle","R_Toe",
    "Torso",  "Spine", "Chest",  "Neck",    "Head",
    "L_Thorax","L_Shoulder","L_Elbow","L_Wrist","L_Hand",
    "R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",
]


# ---------------------------------------------------------------------------
# ModelWrapper so ODESolver can call model(x, t, cond=cond)
# ---------------------------------------------------------------------------

class CondWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module, cond: torch.Tensor):
        super().__init__(model)
        self.cond = cond  # (B, 9) — fixed during ODE solve

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        t_batch = t.reshape(1).expand(x.shape[0])  # scalar → (B,)
        return self.model(x, t_batch, self.cond)


# ---------------------------------------------------------------------------

def normalize_single_frame(frame: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert one world-frame 9D command to root-relative standardized form.
    frame: (9,) — [root_vel_xy(2), pelvis_h(1), left_eef_xyz(3), right_eef_xyz(3)]
    """
    f = frame.copy().astype(np.float32)
    pelvis_xyz = np.array([0.0, 0.0, f[2]], dtype=np.float32)
    f[3:6] -= pelvis_xyz
    f[6:9] -= pelvis_xyz
    mean = np.array(stats["cond_mean"], dtype=np.float32)
    std  = np.array(stats["cond_std"],  dtype=np.float32)
    return (f - mean) / std


def build_cond_window(values: np.ndarray, window: int, stats: dict) -> np.ndarray:
    """
    Build a (window*9,) normalized condition vector.

    values: either (9,) for a single frame (repeated K times)
            or (K*9,) for an explicit K-frame window (oldest first).
    """
    n = len(values)
    if n == 9:
        frames = np.stack([normalize_single_frame(values, stats)] * window)  # (K, 9)
    elif n == window * 9:
        raw_frames = values.reshape(window, 9)
        frames = np.stack([normalize_single_frame(f, stats) for f in raw_frames])
    else:
        raise ValueError(
            f"--cond must have 9 values (single frame) or {window*9} values ({window}-frame window), got {n}"
        )
    return frames.reshape(-1).astype(np.float32)  # (K*9,)


def denormalize_joints(joints_norm: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.array(stats["joint_mean"], dtype=np.float32)
    std  = np.array(stats["joint_std"],  dtype=np.float32)
    return joints_norm * std + mean


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats",      required=True, help="stats.json saved during training")
    p.add_argument("--cond",       required=True,
                   help="9 floats: root_vel_x root_vel_y pelvis_h lx ly lz rx ry rz")
    p.add_argument("--steps",  type=int,   default=50,   help="ODE solver steps")
    p.add_argument("--n-samples", type=int, default=1,   help="how many poses to generate")
    p.add_argument("--out",    default=None, help="optional .npy output path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- load stats ---
    with open(args.stats) as f:
        stats = json.load(f)

    # --- load model (read window size from checkpoint args) ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    window   = train_args.get("window", 10)
    cond_dim = window * (9 + 69)  # task-space + joint history

    # --- parse condition ---
    cond_values = np.array([float(v) for v in args.cond.split()], dtype=np.float32)
    if len(cond_values) == 9:
        pelvis_height = cond_values[2]
    else:
        pelvis_height = cond_values[-7]

    task_norm  = build_cond_window(cond_values, window, stats)       # (K*9,)
    joint_hist = np.zeros(window * 69, dtype=np.float32)             # no prior history
    cond_norm  = np.concatenate([task_norm, joint_hist])             # (K*78,)
    cond_t = torch.tensor(cond_norm, device=device).unsqueeze(0).expand(args.n_samples, -1)

    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- ODE solve ---
    wrapped = CondWrapper(model, cond_t)
    x0 = torch.randn(args.n_samples, 69, device=device)

    with torch.no_grad():
        solver = ODESolver(velocity_model=wrapped)
        x1 = solver.sample(
            x_init=x0,
            step_size=1.0 / args.steps,
            method="midpoint",
            time_grid=torch.linspace(0, 1, args.steps + 1, device=device),
            return_intermediates=False,
        )

    # --- denormalize ---
    joints_norm = x1.cpu().numpy()                            # (N, 72)
    joints_23   = denormalize_joints(joints_norm, stats).reshape(args.n_samples, 23, 3)
    pelvis      = np.zeros((args.n_samples, 1, 3), dtype=np.float32)
    joints_rel  = np.concatenate([pelvis, joints_23], axis=1)  # (N, 24, 3)

    # add back pelvis world position (assumed at x=0, y=0, z=pelvis_height)
    pelvis_world = np.array([0.0, 0.0, pelvis_height], dtype=np.float32)
    joints_world = joints_rel + pelvis_world[None, None, :]   # (N, 24, 3)

    # --- print ---
    for s in range(args.n_samples):
        print(f"\n--- Sample {s+1} ---")
        for j, name in enumerate(JOINT_NAMES):
            xyz = joints_world[s, j]
            print(f"  {name:12s}  x={xyz[0]:+.4f}  y={xyz[1]:+.4f}  z={xyz[2]:+.4f}")

    if args.out:
        np.save(args.out, joints_world)
        print(f"\nSaved ({args.n_samples}, 24, 3) → {args.out}")


if __name__ == "__main__":
    main()
