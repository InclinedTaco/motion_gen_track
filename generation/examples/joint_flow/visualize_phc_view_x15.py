#!/usr/bin/env python3
"""Render the downsampled x15 motion that PHC would receive.

The x15 generator can run at a higher motion FPS than the downstream policy /
tracker. This script generates at --motion-fps, keeps every
motion_fps / policy_hz frame, and renders the kept sequence at --policy-hz.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from animate_horizon_x15 import generate, load_model, make_commands, render_gif
from dataset_horizon_x15 import validate_x15_stats_compatible


def derive_downsample_factor(motion_fps: float, policy_hz: float) -> int:
    if motion_fps <= 0 or policy_hz <= 0:
        raise ValueError("--motion-fps and --policy-hz must be positive")
    ratio = motion_fps / policy_hz
    factor = int(round(ratio))
    if factor < 1 or abs(ratio - factor) > 1e-6:
        raise ValueError(
            f"--motion-fps / --policy-hz must be an integer frame factor; "
            f"got {motion_fps:g} / {policy_hz:g} = {ratio:g}"
        )
    return factor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument(
        "--motion",
        choices=["stand", "constant_walk", "stop_start_walk", "pick_carry_place"],
        default="constant_walk",
    )
    p.add_argument("--out", default="gif/x15_phc_view.gif")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--motion-fps", type=float, default=120.0)
    p.add_argument("--policy-hz", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Generator stride in motion-FPS frames. Must be a multiple of "
        "motion_fps / policy_hz. Defaults to that factor.",
    )
    p.add_argument(
        "--past-guidance",
        type=float,
        default=0.0,
        help="PARC-style blend weight for the no-past-frames branch.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    factor = derive_downsample_factor(args.motion_fps, args.policy_hz)
    if args.stride is None:
        args.stride = factor
    if args.stride % factor != 0:
        raise ValueError(
            f"--stride must be a multiple of downsample factor {factor}; "
            f"got --stride {args.stride}. Use {factor}, {2 * factor}, "
            f"{3 * factor}, ... so PHC samples the same local horizon phase "
            "from every generated chunk."
        )
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)
    validate_x15_stats_compatible(stats)

    model = load_model(args.checkpoint, stats, device)
    command_args = SimpleNamespace(
        motion=args.motion,
        duration=args.duration,
        fps=args.motion_fps,
    )
    command = make_commands(command_args)
    pred_motion = generate(
        model,
        stats,
        command,
        device,
        args.steps,
        args.stride,
        args.past_guidance,
        init_history=None,
    )
    pred_policy = pred_motion[::factor]
    render_gif(args.out, pred_policy, f"{args.motion} PHC view", args.policy_hz)
    print(
        f"Generated {len(pred_motion)} frames at {args.motion_fps:g}fps; "
        f"kept every {factor} -> {len(pred_policy)} frames at {args.policy_hz:g}Hz"
    )
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
