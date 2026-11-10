"""
flow_to_pkl.py

Generate a PHC-compatible motion PKL from the no-EEF flow model.

The flow model generates joint positions conditioned on a commanded
(vel_x, vel_y, pelvis_height) trajectory.  This script retargets the
predicted world-frame joint positions to SMPL pose (axis-angle pose_aa +
joint quaternions) by gradient-fitting the same differentiable FK model PHC
uses internally (Humanoid_Batch), with a pose prior + temporal smoothing,
and saves a PKL that PHC can track directly.  See flow_retarget.py.

Run from the PHC directory:

  python scripts/flow_to_pkl.py \\
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \\
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json \\
    --vel-x 0.5 --vel-y 0.0 --height 0.9 \\
    --duration 4.0 \\
    --out data/generated/walk_forward.pkl

Then track with PHC:

  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \\
    env=env_im_getup_mcp robot=smpl_humanoid \\
    robot.freeze_hand=True robot.box_body=False \\
    env.z_activation=relu env.obs_v=7 \\
    env.motion_file=data/generated/walk_forward.pkl \\
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \\
    env.num_envs=1 headless=False test=True epoch=-1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

# --- path setup -----------------------------------------------------------
PHC_ROOT        = Path(__file__).resolve().parents[1]
FLOW_ROOT       = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR  = FLOW_ROOT / "examples" / "joint_flow"

sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))

from animate import run_flow
from model import JointFlowMLP
from flow_retarget import build_fk, fit_sequence


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Flow model → PHC PKL")
    p.add_argument("--flow-ckpt",  required=True, help="Path to best.pt")
    p.add_argument("--flow-stats", required=True, help="Path to stats.json")
    p.add_argument("--vel-x",   type=float, default=0.5,  help="Forward velocity (m/s)")
    p.add_argument("--vel-y",   type=float, default=0.0,  help="Lateral velocity (m/s)")
    p.add_argument("--height",  type=float, default=0.9,  help="Pelvis height (m)")
    p.add_argument("--duration",type=float, default=4.0,  help="Duration (seconds)")
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--ramp",    type=float, default=0.5,
                   help="Velocity ramp-up duration (seconds); avoids abrupt start")
    p.add_argument("--humanoid-xml", default="final_humanoid.xml",
                   help="Path to runtime-generated humanoid XML (must match PHC eval flags)")
    p.add_argument("--out",      required=True, help="Output PKL path")
    p.add_argument("--clip-name",default="flow_generated")
    p.add_argument("--ode-steps",type=int, default=20, help="ODE steps per frame")
    p.add_argument("--fit-iters",type=int, default=500, help="Gradient-fit iterations for retargeting")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------
def build_taskspace(vel_x, vel_y, height, n_frames, fps, ramp_frames):
    """
    Build (T, 9) task-space array.  Only cols 0-2 matter for no-EEF model:
      col 0: vel_x  col 1: vel_y  col 2: pelvis_height
    Velocity is linearly ramped up over ramp_frames to avoid a step discontinuity.
    """
    ts = np.zeros((n_frames, 9), dtype=np.float32)
    ramp = np.ones(n_frames, dtype=np.float32)
    if ramp_frames > 0:
        ramp[:ramp_frames] = np.linspace(0.0, 1.0, ramp_frames)
    ts[:, 0] = vel_x * ramp
    ts[:, 1] = vel_y * ramp
    ts[:, 2] = height
    return ts


# --------------------------------------------------------------------------
def load_flow_model(ckpt_path, stats, device):
    ckpt       = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args", {})
    window     = stats.get("window", 10)
    prev_dim   = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim     = 3  if stats.get("no_eef", False) else 9
    cond_dim   = prev_dim + window * ts_dim

    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device)
    fps    = args.fps
    n_frames    = int(args.duration * fps)
    ramp_frames = int(args.ramp * fps)

    print(f"Generating {n_frames} frames  ({args.duration}s @ {fps:.0f}fps)")
    print(f"  vel=({args.vel_x:.2f}, {args.vel_y:.2f})  height={args.height:.2f}  ramp={args.ramp:.1f}s")

    with open(args.flow_stats) as f:
        stats = json.load(f)

    model = load_flow_model(args.flow_ckpt, stats, device)
    print("Flow model loaded.")

    ts_world = build_taskspace(args.vel_x, args.vel_y, args.height, n_frames, fps, ramp_frames)

    print("Running autoregressive flow inference …")
    pred = run_flow(
        model, stats, ts_world,
        window=stats.get("window", 10),
        device=device,
        steps=args.ode_steps,
        fps=fps,
        init_pelvis_xy=np.zeros(2, dtype=np.float32),
    )  # (T, 24, 3) world-frame positions

    print("Running gradient-based SMPL fit (Humanoid_Batch FK) …")
    fk  = build_fk(args.humanoid_xml, device)
    fit = fit_sequence(pred, fk, device=device, iters=args.fit_iters)
    print(f"Fit done.  mean joint position error: {fit['fit_err_m']*100:.2f} cm")

    motion_dict = {
        args.clip_name: {
            "pose_quat_global": fit["pose_quat_global"],                         # (T,24,4) xyzw
            "pose_quat":        fit["pose_quat"],                                 # (T,24,4) xyzw local
            "root_trans_offset":torch.from_numpy(fit["root_trans_offset"].astype(np.float64)),
            "trans_orig":       fit["root_trans_offset"].astype(np.float32),
            "beta":             np.zeros(10, dtype=np.float64),
            "gender":           "neutral",
            "pose_aa":          fit["pose_aa"],                                   # (T,72) real, not zeros
            "fps":              fps,
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, out_path)
    print(f"\nSaved → {out_path}")
    print("\nTrack with PHC:")
    print(f"  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \\")
    print(f"    env=env_im_getup_mcp robot=smpl_humanoid \\")
    print(f"    robot.freeze_hand=True robot.box_body=False \\")
    print(f"    env.z_activation=relu env.obs_v=7 \\")
    print(f"    env.motion_file={out_path} \\")
    print(f"    \"env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']\" \\")
    print(f"    env.num_envs=1 headless=False test=True epoch=-1")


if __name__ == "__main__":
    main()
