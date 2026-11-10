#!/usr/bin/env python3
"""
Filter a root-aware locomotion HDF5 to forward-walking clips (any direction).

Keeps clips that are:
  - Walking at human walking speeds (not running, not standing)
  - Primarily forward locomotion (not backward walking)
  - Actually moving (not idle poses)
  - Upright (pelvis height in normal walking range)

Turns and curved paths are explicitly allowed — that is the point.

Example:
    python examples/joint_flow/scripts/filter_locomotion_hdf5.py \
        --input  final_h5_files/cmu_locomotion_taskspace_with_root.h5 \
        --output final_h5_files/cmu_mixed_walk_taskspace_with_root.h5 \
        --dry-run

    python examples/joint_flow/scripts/filter_locomotion_hdf5.py \
        --input  final_h5_files/accad_research_filtered_taskspace_with_root.h5 \
        --output final_h5_files/accad_mixed_walk_taskspace_with_root.h5 \
        --accad-name-filter --dry-run
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


# ACCAD clips to reject by name token.
# "turn" and "side" are intentionally NOT here — we want those.
# "run" rejects running; "backward"/"backwards" rejects backward walks.
# "crouch"/"stand"/"box"/"pick" reject non-locomotion.
BAD_ACCAD_TOKENS = (
    "run",
    "backward",
    "backwards",
    "crouch",
    "stand",
    "box",
    "pick",
)


def wrap_angle(delta):
    return np.arctan2(np.sin(delta), np.cos(delta))


def yaw_from_root_aa(pose_aa_root: np.ndarray) -> np.ndarray:
    return R.from_rotvec(pose_aa_root).as_euler("ZYX")[:, 0].astype(np.float32)


def compute_metrics(jp: np.ndarray, pose_aa_root: Optional[np.ndarray]) -> dict:
    """Compute per-clip locomotion quality metrics from joint positions."""
    root_xy = jp[:, 0, :2].astype(np.float32)
    root_z  = jp[:, 0, 2].astype(np.float32)

    deltas   = np.diff(root_xy, axis=0)          # (T-1, 2)
    step     = np.linalg.norm(deltas, axis=1)     # (T-1,)
    speed    = step * 30.0                         # m/s at 30 fps
    path_len = float(step.sum())
    disp     = float(np.linalg.norm(root_xy[-1] - root_xy[0]))

    moving = step > 1e-4
    speed_mean   = float(speed.mean()) if len(speed) else 0.0
    speed_p95    = float(np.percentile(speed, 95)) if len(speed) else 0.0
    moving_frac  = float((speed > 0.2).mean()) if len(speed) else 0.0

    # Forward / side / back fractions relative to pelvis heading.
    if pose_aa_root is not None and np.any(moving):
        yaw      = yaw_from_root_aa(pose_aa_root[1:][moving])  # heading at each moving frame
        vel_dir  = np.arctan2(deltas[moving, 1], deltas[moving, 0])
        diff     = np.abs(wrap_angle(vel_dir - yaw))
        fwd_frac  = float((diff <= math.radians(60.0)).mean())
        side_frac = float(((diff > math.radians(60.0)) & (diff <= math.radians(120.0))).mean())
        back_frac = float((diff > math.radians(120.0)).mean())
    else:
        fwd_frac = side_frac = back_frac = 0.0

    pelvis_z_mean = float(root_z.mean())
    pelvis_z_min  = float(root_z.min())

    return {
        "speed_mean":    speed_mean,
        "speed_p95":     speed_p95,
        "moving_frac":   moving_frac,
        "fwd_frac":      fwd_frac,
        "side_frac":     side_frac,
        "back_frac":     back_frac,
        "displacement":  disp,
        "path_len":      path_len,
        "pelvis_z_mean": pelvis_z_mean,
        "pelvis_z_min":  pelvis_z_min,
    }


def keep_clip(name: str, jp: np.ndarray, pose_aa_root: Optional[np.ndarray],
              args) -> tuple[bool, dict, str]:
    # ACCAD name-based pre-filter: must contain "walk", must not contain bad tokens
    if args.accad_name_filter:
        low = name.lower()
        if "walk" not in low:
            return False, {}, "accad_no_walk_in_name"
        for tok in BAD_ACCAD_TOKENS:
            if tok in low:
                return False, {}, f"accad_name:{tok}"

    m = compute_metrics(jp, pose_aa_root)

    # --- Speed: reject running and stationary ---
    if m["speed_mean"] < args.min_mean_speed:
        return False, m, f"too_slow({m['speed_mean']:.2f})"
    if m["speed_mean"] > args.max_mean_speed:
        return False, m, f"too_fast_mean({m['speed_mean']:.2f})"
    if m["speed_p95"] > args.max_speed_p95:
        return False, m, f"too_fast_p95({m['speed_p95']:.2f})"

    # --- Movement: reject idle / mostly standing ---
    if m["moving_frac"] < args.min_moving_frac:
        return False, m, f"not_moving({m['moving_frac']:.2f})"

    # --- Backward locomotion: reject clips that are primarily backward ---
    if m["back_frac"] > args.max_back_frac:
        return False, m, f"backward({m['back_frac']:.2f})"

    # --- Displacement: must actually travel somewhere ---
    if m["displacement"] < args.min_displacement and m["path_len"] < args.min_path_len:
        return False, m, f"no_displacement(d={m['displacement']:.2f},p={m['path_len']:.2f})"

    # --- Pelvis height: reject crouching / falling / non-upright ---
    if m["pelvis_z_mean"] < args.min_pelvis_z:
        return False, m, f"pelvis_low({m['pelvis_z_mean']:.2f})"
    if m["pelvis_z_mean"] > args.max_pelvis_z:
        return False, m, f"pelvis_high({m['pelvis_z_mean']:.2f})"

    return True, m, "ok"


def copy_clip(src_grp: h5py.Group, dst_parent: h5py.Group, clip_name: str) -> None:
    dst_grp = dst_parent.create_group(clip_name)
    for key, value in src_grp.attrs.items():
        dst_grp.attrs[key] = value
    for key in src_grp.keys():
        src_grp.copy(key, dst_grp, name=key)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be kept/rejected without writing output")

    # Speed thresholds (m/s at 30 fps)
    p.add_argument("--min-mean-speed", type=float, default=0.15,
                   help="Reject clips slower than this (idle/standing)")
    p.add_argument("--max-mean-speed", type=float, default=1.8,
                   help="Reject clips faster than this (running)")
    p.add_argument("--max-speed-p95",  type=float, default=2.2,
                   help="Reject if 95th-percentile speed exceeds this (burst running)")

    # Movement
    p.add_argument("--min-moving-frac", type=float, default=0.5,
                   help="Reject if fewer than this fraction of frames are moving (>0.2 m/s)")

    # Direction
    p.add_argument("--max-back-frac", type=float, default=0.15,
                   help="Reject if more than this fraction of frames are backward (vel >120deg from heading)")

    # Displacement — keep if EITHER displacement OR path_len passes
    p.add_argument("--min-displacement", type=float, default=0.3,
                   help="Minimum end-to-end displacement (m); OR-ed with --min-path-len")
    p.add_argument("--min-path-len",     type=float, default=1.5,
                   help="Minimum total path length (m); OR-ed with --min-displacement")

    # Pelvis height (retargeted PHC humanoid: standing ~0.85-0.95m)
    p.add_argument("--min-pelvis-z", type=float, default=0.60,
                   help="Reject if mean pelvis height below this (crouching/falling)")
    p.add_argument("--max-pelvis-z", type=float, default=1.15,
                   help="Reject if mean pelvis height above this (unrealistic)")

    # ACCAD-specific
    p.add_argument("--accad-name-filter", action="store_true",
                   help="For ACCAD: require 'walk' in clip name; reject run/backward/crouch/stand/box/pick")

    return p.parse_args()


def main():
    args = parse_args()
    in_path  = Path(args.input)
    out_path = Path(args.output)

    results = {"kept": [], "rejected": []}

    with h5py.File(in_path, "r") as src:
        clips_node = src["clips"] if "clips" in src else src
        clip_names = sorted(clips_node.keys())

        if not args.dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dst_file = h5py.File(out_path, "w")
            for key, value in src.attrs.items():
                dst_file.attrs[key] = value
            dst_clips = dst_file.create_group("clips")
        else:
            dst_file = dst_clips = None

        for clip_name in clip_names:
            grp = clips_node[clip_name]
            if "joint_positions" not in grp:
                continue
            pose_aa_root = grp["pose_aa_root"][:] if "pose_aa_root" in grp else None
            jp = grp["joint_positions"][:]

            keep, m, reason = keep_clip(clip_name, jp, pose_aa_root, args)

            if keep:
                results["kept"].append((clip_name, m))
                if dst_clips is not None:
                    copy_clip(grp, dst_clips, clip_name)
            else:
                results["rejected"].append((clip_name, m, reason))

        if dst_file is not None:
            dst_file.close()

    # --- Summary ---
    kept     = results["kept"]
    rejected = results["rejected"]
    total_frames = sum(m["path_len"] / (m["speed_mean"] + 1e-6) * 30
                       for _, m in kept) if kept else 0
    # more accurate: recount from shapes
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results")
    print(f"  input:    {in_path}")
    print(f"  output:   {out_path}")
    print(f"  kept:     {len(kept)}")
    print(f"  rejected: {len(rejected)}")

    if kept:
        speeds = [m["speed_mean"] for _, m in kept]
        backs  = [m["back_frac"]  for _, m in kept]
        print(f"\n  kept speed  mean={np.mean(speeds):.2f} min={np.min(speeds):.2f} max={np.max(speeds):.2f} m/s")
        print(f"  kept back_frac mean={np.mean(backs):.3f}")
        print(f"\n  Kept clips:")
        for name, m in kept:
            print(f"    {name[:55]:<55} "
                  f"spd={m['speed_mean']:.2f} p95={m['speed_p95']:.2f} "
                  f"fwd={m['fwd_frac']:.2f} back={m['back_frac']:.2f} "
                  f"pz={m['pelvis_z_mean']:.2f}")

    if rejected:
        from collections import Counter
        reason_counts = Counter(r for _, _, r in rejected)
        print(f"\n  Rejection reasons:")
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
