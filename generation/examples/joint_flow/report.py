"""
Frame-by-frame per-joint error report. Works for both position and 6D models.

Usage:
    # Position model
    python examples/joint_flow/report.py \
        --checkpoint checkpoints/joint_flow_noeef_prevframe/best.pt \
        --stats      checkpoints/joint_flow_noeef_prevframe/stats.json \
        --hdf5       data/amass/hdf5/accad_research_filtered_taskspace.h5 \
        --out        report_pos.csv

    # 6D model
    python examples/joint_flow/report.py \
        --checkpoint  checkpoints/joint_flow_6d/best.pt \
        --stats       checkpoints/joint_flow_6d/stats.json \
        --hdf5        data/amass/hdf5/accad_6d_taskspace.h5 \
        --humanoid-xml /path/to/final_humanoid.xml \
        --out         report_6d.csv

Output CSV columns:
    clip, frame, joint_idx, joint_name, l2_error

Summary printed to stdout: mean/median/max L2 per joint + smoothness.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from animate import (
    _build_task_goal_sentinel,
    _load_clip_task_goals,
    load_clip,
    run_flow,
    run_flow_6d,
)

from model import CVAEPrior, JointFlowMLP

JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument(
        "--clip",
        type=int,
        default=None,
        help="Single clip index; omit to run all clips",
    )
    p.add_argument(
        "--n-clips",
        type=int,
        default=None,
        help="Evaluate N evenly-spaced clips instead of all",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument(
        "--humanoid-xml", default=None, help="Required for 6D models (poselib FK)"
    )
    p.add_argument(
        "--linear-sentinel",
        action="store_true",
        help="Use linearly decreasing obj sentinel (matches training) instead of fixed (2,2,2)",
    )
    p.add_argument(
        "--omomo-hdf5",
        default=None,
        help="Path to OMOMO HDF5; required when --linear-sentinel is set",
    )
    p.add_argument("--out", default="report.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(ckpt_path, stats, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window = stats.get("window", 10)
    use_6d = stats.get("use_6d", False)
    predict_root = stats.get("predict_root", False)
    joint_dim = (144 if predict_root else 138) if use_6d else 69
    prev_dim = joint_dim if stats.get("use_prev_frame", False) else 0
    ts_dim = (
        (3 if stats.get("no_eef", False) else 9)
        + (3 if stats.get("use_obj", False) else 0)
        + (9 if stats.get("use_task_goals", False) else 0)
    )
    use_cvae = train_args.get("use_cvae", False)
    latent_dim = train_args.get("latent_dim", 64)
    cond_dim = prev_dim + window * ts_dim
    flow_cond_dim = latent_dim if use_cvae else cond_dim

    model = JointFlowMLP(
        joint_dim=joint_dim,
        cond_dim=flow_cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prior = None
    if use_cvae:
        prior = CVAEPrior(cond_dim, latent_dim).to(device)
        prior.load_state_dict(ckpt["prior"])
        prior.eval()

    return model, window, prior


def build_sk(humanoid_xml):
    import sys as _sys

    _phc_root = str(Path(__file__).resolve().parents[3] / "tracking")
    _sys.path.insert(0, str(Path(_phc_root) / "poselib"))
    from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

    return SkeletonTree.from_mjcf(humanoid_xml), SkeletonState


def positions_from_6d(pose_aa_np, root_trans_np, sk_tree, SkeletonState):
    from scipy.spatial.transform import Rotation as _sRot

    T = len(pose_aa_np)
    quats = _sRot.from_rotvec(pose_aa_np.reshape(-1, 3)).as_quat()
    quats_t = torch.from_numpy(quats).float().reshape(T, 24, 4)
    rt_t = torch.from_numpy(root_trans_np).float()
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, quats_t, rt_t, is_local=True
    )
    return sk_state.global_translation.numpy()  # (T, 24, 3)


def smoothness(positions):
    """Mean per-joint speed (m/frame) — lower = smoother."""
    vel = np.diff(positions, axis=0)  # (T-1, 24, 3)
    return float(np.linalg.norm(vel, axis=-1).mean())


def build_linear_sentinel(omomo_hdf5, T):
    import h5py

    with h5py.File(omomo_hdf5, "r") as f:
        clips = list(f["clips"].keys())
        arrays = [
            f["clips"][c]["obj_com_pos_rel"][:]
            for c in clips
            if "obj_com_pos_rel" in f["clips"][c]
        ]
    max_obj = np.concatenate(arrays, axis=0).max(0).astype(np.float32)
    sentinel_end = max_obj + 0.1
    sentinel_start = sentinel_end * 2.0
    t_arr = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    return sentinel_start[None] + (sentinel_end - sentinel_start)[None] * t_arr


def run_clip(
    clip_idx, model, stats, window, prior, device, args, sk_tree, SkeletonState
):
    use_6d = stats.get("use_6d", False)
    ts_world, jp_gt, clip_fps, clip_name, obj_rel = load_clip(args.hdf5, clip_idx)
    task_goals = (
        _load_clip_task_goals(args.hdf5, clip_idx)
        if stats.get("use_task_goals", False)
        else None
    )

    # build per-clip linear sentinel if requested and clip has no real obj data
    obj_sentinel = None
    if (
        getattr(args, "linear_sentinel", False)
        and stats.get("use_obj", False)
        and obj_rel is None
    ):
        obj_sentinel = build_linear_sentinel(args.omomo_hdf5, len(ts_world))
    if stats.get("use_task_goals", False) and task_goals is None:
        task_goals = _build_task_goal_sentinel(args.omomo_hdf5, len(ts_world), ts_world)

    if use_6d:
        pose_aa_np, root_trans_np = run_flow_6d(
            model,
            stats,
            ts_world,
            window,
            device,
            steps=args.steps,
            fps=clip_fps,
            init_pelvis_xy=jp_gt[0, 0, :2],
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
        )
        pred = positions_from_6d(pose_aa_np, root_trans_np, sk_tree, SkeletonState)
    else:
        pred = run_flow(
            model,
            stats,
            ts_world,
            window,
            device,
            steps=args.steps,
            fps=clip_fps,
            init_pelvis_xy=jp_gt[0, 0, :2],
            prior=prior,
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
        )

    T = min(len(pred), len(jp_gt))
    pred, jp_gt = pred[:T], jp_gt[:T]

    rows = []
    per_frame_mean = []
    for t in range(T):
        frame_errs = []
        for j in range(24):
            l2 = float(np.linalg.norm(pred[t, j] - jp_gt[t, j]))
            frame_errs.append(l2)
            rows.append(
                {
                    "clip": clip_name,
                    "frame": t,
                    "joint_idx": j,
                    "joint_name": JOINT_NAMES[j],
                    "l2_error": f"{l2:.6f}",
                }
            )
        per_frame_mean.append(np.mean(frame_errs))

    return rows, np.array(per_frame_mean), pred, clip_name


def count_clips(hdf5_path):
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        return len(list(f["clips"].keys()))


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats) as f:
        stats = json.load(f)

    use_6d = stats.get("use_6d", False)
    print(f"Model type: {'6D rotation' if use_6d else 'joint position'}")

    model, window, prior = load_model(args.checkpoint, stats, device)

    sk_tree = SkeletonState = None
    if use_6d:
        if args.humanoid_xml is None:
            raise ValueError("--humanoid-xml required for 6D models")
        sk_tree, SkeletonState = build_sk(args.humanoid_xml)

    n_clips = count_clips(args.hdf5)
    if args.clip is not None:
        clip_indices = [args.clip]
    elif args.n_clips is not None:
        step = max(1, n_clips // args.n_clips)
        clip_indices = list(range(0, n_clips, step))[: args.n_clips]
    else:
        clip_indices = list(range(n_clips))
    print(f"Running on {len(clip_indices)}/{n_clips} clips from {args.hdf5}\n")

    all_rows = []
    all_frame_means = []
    all_positions = []

    for ci in clip_indices:
        rows, frame_means, pred, clip_name = run_clip(
            ci, model, stats, window, prior, device, args, sk_tree, SkeletonState
        )
        all_rows.extend(rows)
        all_frame_means.append(frame_means)
        all_positions.append(pred)
        mean_err = frame_means.mean() * 100
        print(f"  [{ci:3d}] {clip_name[:50]:50s}  mean={mean_err:.2f}cm")

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["clip", "frame", "joint_idx", "joint_name", "l2_error"]
        )
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows → {args.out}")

    # global summary
    all_errors = np.concatenate(all_frame_means)
    n_nan = int(np.isnan(all_errors).sum())
    print("\n--- L2 error (mean per-frame across all joints) ---")
    if n_nan:
        print(f"  ({n_nan} NaN frames ignored)")
    print(f"  Mean:   {np.nanmean(all_errors)*100:.2f} cm")
    print(f"  Median: {np.nanmedian(all_errors)*100:.2f} cm")
    print(f"  p95:    {np.nanpercentile(all_errors, 95)*100:.2f} cm")
    print(f"  Max:    {np.nanmax(all_errors)*100:.2f} cm")

    # per-joint summary
    print("\n--- Per-joint mean L2 (cm) ---")
    for j in range(24):
        errs = np.array(
            [float(r["l2_error"]) for r in all_rows if int(r["joint_idx"]) == j]
        )
        print(f"  {JOINT_NAMES[j]:14s}  {np.nanmean(errs)*100:.2f} cm")

    # smoothness
    all_pos = np.concatenate(all_positions, axis=0)
    print("\n--- Smoothness (mean joint speed m/frame, lower=smoother) ---")
    print(f"  {smoothness(all_pos):.5f} m/frame")


if __name__ == "__main__":
    main()
