"""
Side-by-side comparison of two x15 checkpoints at the same stride.

Modes:
  recon  -- GT-commanded reconstruction of a clip
  walk   -- synthetic constant-walk generation

Usage:
    python examples/joint_flow/compare_models_x15.py recon \
        --ckpt-a checkpoints/A/best.pt --stats-a checkpoints/A/stats.json \
        --ckpt-b checkpoints/B/best.pt --stats-b checkpoints/B/stats.json \
        --hdf5 final_h5_files/... --clip-idx 529 --stride 5 \
        --label-a "h15" --label-b "h5" \
        --out gif/compare_models_recon.gif

    python examples/joint_flow/compare_models_x15.py walk \
        --ckpt-a ... --stats-a ... --ckpt-b ... --stats-b ... \
        --stride 5 --duration 30 --out gif/compare_models_walk.gif
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # joint_flow

import h5py
import numpy as np
import torch

from animate_horizon_x15 import generate, load_model, make_commands
from compare_strides_x15 import render_side_by_side
from dataset_horizon_x15 import JointFlowX15Dataset, validate_x15_stats_compatible
from reconstruct_x15 import reconstruct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["recon", "walk"])
    p.add_argument("--ckpt-a", required=True)
    p.add_argument("--stats-a", required=True)
    p.add_argument("--ckpt-b", required=True)
    p.add_argument("--stats-b", required=True)
    p.add_argument("--label-a", default="model-a")
    p.add_argument("--label-b", default="model-b")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--render-fps", type=float, default=30.0)
    # recon args
    p.add_argument("--hdf5")
    p.add_argument("--clip-idx", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=300)
    # walk args
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load(ckpt, stats_path, device):
    with open(stats_path) as f:
        stats = json.load(f)
    validate_x15_stats_compatible(stats)
    model = load_model(ckpt, stats, device)
    ta = stats.get("train_args") or {}
    fps = float(ta.get("motion_fps") or 30.0)
    return model, stats, fps


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model_a, stats_a, fps_a = load(args.ckpt_a, args.stats_a, device)
    model_b, stats_b, fps_b = load(args.ckpt_b, args.stats_b, device)

    step_a = max(1, round(fps_a / args.render_fps))
    step_b = max(1, round(fps_b / args.render_fps))

    if args.mode == "recon":
        if not args.hdf5:
            raise ValueError("--hdf5 required for recon mode")

        def load_clip(stats):
            ds = JointFlowX15Dataset(
                [args.hdf5],
                past_frames=int(stats["past_frames"]),
                past_stride=int(stats.get("past_stride", 1)),
                horizon=int(stats["horizon"]),
                stats=stats,
            )
            clip = ds._clips[args.clip_idx]
            if args.max_frames and len(clip["root_xy"]) > args.max_frames:
                clip = {k: (v[: args.max_frames] if isinstance(v, np.ndarray) else v) for k, v in clip.items()}
            return clip

        with h5py.File(args.hdf5, "r") as f:
            clip_name = sorted(f["clips"].keys())[args.clip_idx]
        print(f"clip {args.clip_idx} ({clip_name})")

        clip_a = load_clip(stats_a)
        clip_b = load_clip(stats_b)

        print(f"Running {args.label_a} ...")
        pred_a, gt_a = reconstruct(model_a, stats_a, clip_a, device, args.steps, args.stride)
        print(f"Running {args.label_b} ...")
        pred_b, gt_b = reconstruct(model_b, stats_b, clip_b, device, args.steps, args.stride)

        n = min(len(pred_a), len(pred_b))
        pred_a, pred_b = pred_a[:n:step_a], pred_b[:n:step_b]
        gt_a, gt_b = gt_a[:n:step_a], gt_b[:n:step_b]

        for tag, pred, gt in [(args.label_a, pred_a, gt_a), (args.label_b, pred_b, gt_b)]:
            l2 = np.linalg.norm(pred - gt, axis=-1).mean()
            drift = np.linalg.norm(pred[:, 0, :2] - gt[:, 0, :2], axis=-1).mean()
            print(f"  {tag}: L2={l2*100:.2f}cm  root_drift={drift*100:.1f}cm")

        render_side_by_side(
            args.out, pred_a, pred_b,
            f"{args.label_a} | stride={args.stride}", f"{args.label_b} | stride={args.stride}",
            args.render_fps, gt_left=gt_a, gt_right=gt_b,
        )

    elif args.mode == "walk":
        class _WalkArgs:
            motion = "constant_walk"
            duration = args.duration
            speed = args.speed

        class WalkA:
            motion = "constant_walk"
            duration = args.duration
            speed = args.speed
            fps = fps_a

        class WalkB:
            motion = "constant_walk"
            duration = args.duration
            speed = args.speed
            fps = fps_b

        cmd_a = make_commands(WalkA())
        cmd_b = make_commands(WalkB())

        print(f"Running {args.label_a} ...")
        pred_a = generate(model_a, stats_a, cmd_a, device, args.steps, args.stride, 0.0, None)
        print(f"Running {args.label_b} ...")
        pred_b = generate(model_b, stats_b, cmd_b, device, args.steps, args.stride, 0.0, None)

        n = min(len(pred_a), len(pred_b))
        pred_a, pred_b = pred_a[:n:step_a], pred_b[:n:step_b]

        render_side_by_side(
            args.out, pred_a, pred_b,
            f"{args.label_a} | stride={args.stride}", f"{args.label_b} | stride={args.stride}",
            args.render_fps,
        )


if __name__ == "__main__":
    main()
