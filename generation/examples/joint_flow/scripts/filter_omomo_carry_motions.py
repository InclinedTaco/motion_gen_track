#!/usr/bin/env python3
"""
Filter raw OMOMO object trajectories for normal carry-style motions.

The filter is intentionally conservative: it keeps clips where the object is
lifted, the pelvis moves while the object stays close in XY, and the object is
not carried overhead.

Example:
    python examples/joint_flow/scripts/filter_omomo_carry_motions.py \
        --obj-pkl <PROJECT_ROOT>/tracking/data/amass/pkls/omomo_obj_trajectories.pkl \
        --hdf5 <PROJECT_ROOT>/tracking/data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \
        --out-pkl <PROJECT_ROOT>/generation/artifacts/omomo_filtered/omomo_carry_nonoverhead.pkl \
        --out-csv <PROJECT_ROOT>/generation/artifacts/omomo_filtered/omomo_carry_nonoverhead.csv
"""
import argparse
import csv
from pathlib import Path

import h5py
import joblib
import numpy as np


def object_name(seq_name):
    parts = seq_name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def load_hdf5_names(path):
    if path is None:
        return set()
    with h5py.File(path, "r") as f:
        if "clips" not in f:
            return set()
        return set(f["clips"].keys())


def longest_true_run(mask, fps, min_seconds):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) / fps >= min_seconds:
            runs.append((i, j))
        i = j
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def carry_walk_metrics(carry_obj, carry_pelvis, fps):
    obj_world_z = carry_obj[:, 2]
    pelvis_z = carry_pelvis[:, 2]

    speed = np.zeros(len(carry_pelvis), dtype=np.float32)
    if len(carry_pelvis) > 1:
        speed[1:] = np.linalg.norm(np.diff(carry_pelvis[:, :2], axis=0), axis=1) * fps

    rel = carry_obj - carry_pelvis
    rel_step = np.zeros(len(rel), dtype=np.float32)
    if len(rel) > 1:
        rel_step[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)

    mask = (
        (speed > 0.25)
        & (obj_world_z > 0.95)
        & (rel_step < 0.05)
        & (pelvis_z > 0.85)
    )
    run = longest_true_run(mask, fps, 0.0)
    if run is None:
        return 0.0, 0.0

    start, end = run
    seconds = (end - start) / fps
    pelvis_disp_xy = float(
        np.linalg.norm(carry_pelvis[end - 1, :2] - carry_pelvis[start, :2])
    )
    return seconds, pelvis_disp_xy


def analyze_sequence(seq, args):
    obj = np.asarray(seq.get("obj_com_pos"))
    pelvis = np.asarray(seq.get("trans"))
    if obj.ndim != 2 or pelvis.ndim != 2 or len(obj) < 20:
        return None

    n = min(len(obj), len(pelvis))
    obj = obj[:n]
    pelvis = pelvis[:n]
    fps = float(seq.get("fps", 30.0) or 30.0)

    obj_z = obj[:, 2]
    low_z = float(np.percentile(obj_z, 10))
    high_z = float(np.percentile(obj_z, 90))
    lift_amp = high_z - low_z
    lifted = obj_z > low_z + max(args.min_lift_z, args.lift_frac * lift_amp)
    run = longest_true_run(lifted, fps, args.min_carry_seconds)
    if run is None:
        return None

    start, end = run
    carry_obj = obj[start:end]
    carry_pelvis = pelvis[start:end]
    rel = carry_obj - carry_pelvis

    pickup_to_place_xy = float(np.linalg.norm(obj[end - 1, :2] - obj[start, :2]))
    pelvis_disp_xy = float(np.linalg.norm(pelvis[end - 1, :2] - pelvis[start, :2]))
    rel_xy_std = float(np.mean(np.std(rel[:, :2], axis=0))) if len(rel) > 2 else float("inf")
    obj_z_p95 = float(np.percentile(carry_obj[:, 2], 95))
    rel_z_mean = float(np.mean(rel[:, 2]))
    rel_z_p05 = float(np.percentile(rel[:, 2], 5))
    rel_z_p95 = float(np.percentile(rel[:, 2], 95))
    carry_walk_seconds, carry_walk_pelvis_disp_xy = carry_walk_metrics(
        carry_obj, carry_pelvis, fps
    )

    return {
        "start": start,
        "end": end,
        "fps": fps,
        "carry_seconds": (end - start) / fps,
        "carry_walk_seconds": carry_walk_seconds,
        "carry_walk_pelvis_disp_xy": carry_walk_pelvis_disp_xy,
        "pickup_to_place_xy": pickup_to_place_xy,
        "pelvis_disp_xy": pelvis_disp_xy,
        "rel_xy_std": rel_xy_std,
        "lift_amp": lift_amp,
        "obj_z_p95": obj_z_p95,
        "rel_z_mean": rel_z_mean,
        "rel_z_p05": rel_z_p05,
        "rel_z_p95": rel_z_p95,
    }


def keep_reason(metrics, args):
    if metrics is None:
        return False, "missing_or_no_lifted_run"
    checks = [
        (metrics["pickup_to_place_xy"] >= args.min_pick_place_xy, "short_pick_place_xy"),
        (metrics["carry_seconds"] >= args.min_carry_seconds, "short_carry"),
        (metrics["pelvis_disp_xy"] >= args.min_pelvis_disp_xy, "low_pelvis_motion"),
        (
            metrics["carry_walk_seconds"] >= args.min_carry_walk_seconds,
            "short_carry_walk",
        ),
        (
            metrics["carry_walk_pelvis_disp_xy"] >= args.min_carry_walk_pelvis_disp_xy,
            "low_carry_walk_pelvis_motion",
        ),
        (metrics["lift_amp"] >= args.min_lift_amp, "low_lift_amp"),
        (metrics["rel_xy_std"] <= args.max_rel_xy_std, "loose_object_xy"),
        (metrics["obj_z_p95"] <= args.max_obj_z_p95, "overhead_abs_z"),
        (metrics["rel_z_p95"] <= args.max_rel_z_p95, "overhead_rel_z"),
        (metrics["rel_z_p05"] >= args.min_rel_z_p05, "object_too_low"),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, "keep"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj-pkl", required=True)
    p.add_argument("--hdf5", default=None, help="Existing HDF5 to mark clips already included")
    p.add_argument("--out-pkl", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--objects", nargs="*", default=None, help="Optional object allowlist")
    p.add_argument("--min-pick-place-xy", type=float, default=0.8)
    p.add_argument("--min-carry-seconds", type=float, default=0.7)
    p.add_argument("--min-pelvis-disp-xy", type=float, default=0.4)
    p.add_argument("--min-carry-walk-seconds", type=float, default=0.7)
    p.add_argument("--min-carry-walk-pelvis-disp-xy", type=float, default=0.25)
    p.add_argument("--min-lift-amp", type=float, default=0.25)
    p.add_argument("--min-lift-z", type=float, default=0.12)
    p.add_argument("--lift-frac", type=float, default=0.35)
    p.add_argument("--max-rel-xy-std", type=float, default=0.45)
    p.add_argument("--max-obj-z-p95", type=float, default=1.65)
    p.add_argument("--max-rel-z-p95", type=float, default=0.85)
    p.add_argument("--min-rel-z-p05", type=float, default=-1.0)
    args = p.parse_args()

    seqs = joblib.load(args.obj_pkl)
    hdf5_names = load_hdf5_names(args.hdf5)
    allow_objects = set(args.objects) if args.objects else None

    rows = []
    kept = {}
    for name, seq in seqs.items():
        obj_name = object_name(name)
        metrics = analyze_sequence(seq, args)
        keep, reason = keep_reason(metrics, args)
        if allow_objects is not None and obj_name not in allow_objects:
            keep, reason = False, "object_not_allowed"

        row = {
            "name": name,
            "object": obj_name,
            "keep": int(keep),
            "reason": reason,
            "in_hdf5": int(name in hdf5_names),
        }
        if metrics is not None:
            row.update(metrics)
        rows.append(row)
        if keep:
            kept[name] = seq

    out_pkl = Path(args.out_pkl)
    out_csv = Path(args.out_csv)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(kept, out_pkl)

    fieldnames = [
        "name",
        "object",
        "keep",
        "reason",
        "in_hdf5",
        "start",
        "end",
        "fps",
        "carry_seconds",
        "carry_walk_seconds",
        "carry_walk_pelvis_disp_xy",
        "pickup_to_place_xy",
        "pelvis_disp_xy",
        "rel_xy_std",
        "lift_amp",
        "obj_z_p95",
        "rel_z_mean",
        "rel_z_p05",
        "rel_z_p95",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    kept_names = set(kept)
    print(f"raw_sequences={len(seqs)}")
    print(f"kept={len(kept)}")
    print(f"kept_already_in_hdf5={len(kept_names & hdf5_names)}")
    print(f"kept_left_out={len(kept_names - hdf5_names)}")
    print(f"saved_pkl={out_pkl}")
    print(f"saved_csv={out_csv}")


if __name__ == "__main__":
    main()
