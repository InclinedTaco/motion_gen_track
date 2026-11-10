"""
eval_flow_quality.py — command-compliance and smoothness eval for x15 checkpoints.

Four checks, printed as a table:
  1. TF joint L2, velocity error, %HJF        (teacher-forced, holdout clips)
  2. AR joint L2, root drift                   (autoregressive, same clips)
  3. Warm-start command compliance             (GT seed → commanded walk speed/heading)
  4. Phase compliance (carry_walk)             (GT seed → carry command → stepping rate)

Usage:
    python examples/joint_flow/eval_flow_quality.py \\
        --checkpoint checkpoints/joint_flow_x15_v12_stride_rollout/best.pt \\
        --stats     checkpoints/joint_flow_x15_v12_stride_rollout/stats.json

Optional flags:
    --walk-hdf5   path to locomotion HDF5   (default: final_h5_files/cmu_straight_walk_taskspace_with_root.h5)
    --omomo-hdf5  path to OMOMO holdout HDF5 (default: artifacts/omomo_filtered/omomo_portable_new_holdout_taskspace.h5)
    --n-walk-clips    N clips for walk eval  (default: 10)
    --n-omomo-clips   N clips for OMOMO eval (default: 10)
    --warmup-frames   GT frames used to seed (default: 30, = 1 s)
    --eval-frames     frames to run AR eval  (default: 150, = 5 s)
    --walk-speed      commanded speed m/s    (default: 0.5)
    --steps           ODE steps              (default: 20)
    --stride          replan stride          (default: 5)
    --seed            random seed            (default: 42)
    --device          cpu/cuda               (default: auto)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from animate_horizon_x15 import (
    load_model, local_state_from_world, rot_to_anchor,
    sample_horizon, state_from_target_frame,
)
from dataset_horizon_x15 import JointFlowX15Dataset, STATE_DIM, X15_REPRESENTATION

FPS = 30.0
# PARC jerk threshold: max observed in mocap (~11 666 m/s³); use a round number
JERK_MAX_MS3 = 10_000.0


# ---------------------------------------------------------------------------
# Shared rollout helpers
# ---------------------------------------------------------------------------

def _clip_world_state(clip: dict, f: int) -> dict:
    return {
        "root_xy":   clip["root_xy"][f],
        "root_z":    float(clip["root_z"][f]),
        "heading":   float(clip["root_heading"][f]),
        "pitch":     float(clip["root_pitch"][f]),
        "roll":      float(clip["root_roll"][f]),
        "joint_rel": clip["joint_rel"][f],
    }


def _build_cond(history: list, mode: np.ndarray, desired_heading: np.ndarray,
                active_goal: np.ndarray, K: int) -> np.ndarray:
    anchor = history[-1]
    anchor_xy      = anchor["root_xy"]
    anchor_heading = float(anchor["heading"])
    anchor_pitch   = float(anchor["pitch"])
    anchor_roll    = float(anchor["roll"])
    past = [
        local_state_from_world(s, anchor_xy, anchor_heading, anchor_pitch, anchor_roll)
        for s in history[-K:]
    ]
    return np.concatenate(past + [mode, desired_heading, active_goal]).astype(np.float32)


def _step_ar(history: list, target: np.ndarray, take: int) -> list:
    """Unpack `take` frames from the horizon prediction, update history."""
    anchor = history[-1]
    new_states = []
    for j in range(take):
        s = state_from_target_frame(
            target[j],
            anchor["root_xy"], float(anchor["heading"]),
            float(anchor["pitch"]), float(anchor["roll"]),
        )
        history.append(s)
        new_states.append(s)
    return new_states


# ---------------------------------------------------------------------------
# Check 1 + 2: TF and AR reconstruction metrics on GT clips
# ---------------------------------------------------------------------------

def _jerk_fraction(positions: np.ndarray) -> float:
    """Fraction of frames with any joint jerk > JERK_MAX_MS3.
    positions: (T, 24, 3) world-frame joint positions, 30 Hz.
    """
    vel  = np.diff(positions, n=1, axis=0) * FPS          # (T-1, 24, 3)
    acc  = np.diff(vel,       n=1, axis=0) * FPS          # (T-2, 24, 3)
    jerk = np.diff(acc,       n=1, axis=0) * FPS          # (T-3, 24, 3)
    mag  = np.linalg.norm(jerk, axis=-1)                  # (T-3, 24)
    high = (mag > JERK_MAX_MS3).any(axis=-1)              # (T-3,)
    return float(high.mean())


def _vel_error(pred_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """Mean per-frame joint velocity error (m/s)."""
    pred_v = np.diff(pred_pos, axis=0) * FPS
    gt_v   = np.diff(gt_pos,   axis=0) * FPS
    return float(np.linalg.norm(pred_v - gt_v, axis=-1).mean())


def eval_reconstruction(model, stats: dict, clips: list, device,
                        steps: int, stride: int) -> dict:
    K  = int(stats["past_frames"])
    H  = int(stats["horizon"])

    tf_l2_all, tf_vel_all, tf_hjf_all = [], [], []
    ar_l2_all, ar_root_mean_all, ar_root_final_all = [], [], []

    for clip in clips:
        T = len(clip["root_xy"])
        if T < K + stride + 3:
            continue

        gt_pos = np.zeros((T, 24, 3), dtype=np.float32)
        gt_pos[:, 0, :] = np.column_stack([clip["root_xy"], clip["root_z"]])
        gt_pos[:, 1:, :] = clip["joint_rel"].reshape(T, 23, 3) + gt_pos[:, :1, :]

        # ---- TF pass -------------------------------------------------------
        tf_pred = []
        for t in range(K, T):
            history_tf = [_clip_world_state(clip, f) for f in range(t - K, t)]
            anchor = history_tf[-1]
            mode = clip["mode"][min(t, T - 1)].copy()
            desired_heading, active_goal = _gt_heading_and_goal(clip, t, T, H, anchor)
            cond = _build_cond(history_tf, mode, desired_heading, active_goal, K)
            target = sample_horizon(model, cond, stats, device, steps)
            s = state_from_target_frame(
                target[0],
                anchor["root_xy"], float(anchor["heading"]),
                float(anchor["pitch"]), float(anchor["roll"]),
            )
            tf_pred.append(s["joints"])

        tf_pred = np.stack(tf_pred, axis=0)           # (T-K, 24, 3)
        gt_slice = gt_pos[K:]
        tf_l2_all.append(np.linalg.norm(tf_pred - gt_slice, axis=-1).mean())
        tf_vel_all.append(_vel_error(tf_pred, gt_slice))
        tf_hjf_all.append(_jerk_fraction(tf_pred))

        # ---- AR pass -------------------------------------------------------
        history_ar = [_clip_world_state(clip, f) for f in range(K)]
        ar_joints = []
        t = K
        while t < T:
            anchor = history_ar[-1]
            mode = clip["mode"][min(t, T - 1)].copy()
            desired_heading, active_goal = _gt_heading_and_goal(clip, t, T, H, anchor)
            cond = _build_cond(history_ar, mode, desired_heading, active_goal, K)
            target = sample_horizon(model, cond, stats, device, steps)
            take = min(stride, T - t)
            new = _step_ar(history_ar, target, take)
            ar_joints.extend([s["joints"] for s in new])
            t += take

        ar_pred = np.stack(ar_joints, axis=0)
        ar_l2_all.append(np.linalg.norm(ar_pred - gt_slice[:len(ar_pred)], axis=-1).mean())
        root_err = np.linalg.norm(
            ar_pred[:, 0, :2] - gt_slice[:len(ar_pred), 0, :2], axis=-1
        )
        ar_root_mean_all.append(float(root_err.mean()))
        ar_root_final_all.append(float(root_err[-1]))

    return {
        "tf_joint_l2_cm":     np.mean(tf_l2_all) * 100,
        "tf_vel_error_cms":   np.mean(tf_vel_all) * 100,
        "tf_hjf_pct":         np.mean(tf_hjf_all) * 100,
        "ar_joint_l2_cm":     np.mean(ar_l2_all) * 100,
        "ar_root_mean_cm":    np.mean(ar_root_mean_all) * 100,
        "ar_root_final_cm":   np.mean(ar_root_final_all) * 100,
        "n_clips":            len(tf_l2_all),
    }


def _gt_heading_and_goal(clip: dict, t: int, T: int, H: int, anchor: dict):
    mode = clip["mode"][min(t, T - 1)]
    anchor_xy      = anchor["root_xy"]
    anchor_z       = float(anchor["root_z"])
    anchor_heading = float(anchor["heading"])

    if mode[0] > 0.5 or mode[1] > 0.5:
        h_idx = min(t + H - 1, T - 1)
        disp = clip["root_xy"][h_idx] - anchor_xy
        dist = float(np.linalg.norm(disp))
        if dist < 0.01:
            desired_heading = np.zeros(2, dtype=np.float32)
        else:
            desired_heading = rot_to_anchor(disp / dist, anchor_heading)
    else:
        desired_heading = np.zeros(2, dtype=np.float32)

    if clip["has_obj"]:
        goals = clip["task_goals"][min(t, T - 1)].reshape(3, 3).copy()
        goals[:2, :2] = rot_to_anchor(goals[:2, :2] - anchor_xy, anchor_heading)
        goals[:2, 2] -= anchor_z
        active_goal = goals[1] if (mode[3] > 0.5 or mode[4] > 0.5) else goals[0]
    else:
        active_goal = np.zeros(3, dtype=np.float32)

    return desired_heading, active_goal


# ---------------------------------------------------------------------------
# Check 3: warm-start command compliance (walk speed & heading)
# ---------------------------------------------------------------------------

def eval_command_compliance(model, stats: dict, clips: list, device,
                            steps: int, stride: int,
                            warmup_frames: int, eval_frames: int,
                            walk_speed: float) -> dict:
    K = int(stats["past_frames"])
    MODE_LOCO = np.array([0, 1, 0, 0, 0], dtype=np.float32)

    achieved_speeds = []
    heading_errors  = []

    for clip in clips:
        T = len(clip["root_xy"])
        if T < warmup_frames + K:
            continue

        # GT warm-start at a frame where the clip is already walking
        seed_end = min(warmup_frames, T)
        history = [_clip_world_state(clip, f) for f in range(max(0, seed_end - K), seed_end)]

        commanded_heading = float(clip["root_heading"][seed_end - 1])  # keep current heading
        desired_heading_vec = np.array([0.0, 1.0], dtype=np.float32)   # straight ahead (anchor frame)

        root_positions = [history[-1]["root_xy"].copy()]

        t = 0
        while t < eval_frames:
            anchor = history[-1]
            anchor_xy      = anchor["root_xy"]
            anchor_heading = float(anchor["heading"])
            anchor_pitch   = float(anchor["pitch"])
            anchor_roll    = float(anchor["roll"])

            active_goal = np.zeros(3, dtype=np.float32)
            cond = _build_cond(history, MODE_LOCO, desired_heading_vec, active_goal, K)
            target = sample_horizon(model, cond, stats, device, steps)
            take = min(stride, eval_frames - t)
            new = _step_ar(history, target, take)
            for s in new:
                root_positions.append(s["root_xy"].copy())
            t += take

        root_pos = np.stack(root_positions, axis=0)           # (eval_frames+1, 2)
        displacements = np.linalg.norm(np.diff(root_pos, axis=0), axis=-1)
        achieved_speed = float(displacements.mean() * FPS)    # m/s

        # Heading compliance: mean cos-similarity between successive displacement vectors
        disp_vecs = np.diff(root_pos, axis=0)                 # (eval_frames, 2)
        norms = np.linalg.norm(disp_vecs, axis=-1, keepdims=True)
        moving = norms[:, 0] > 1e-4
        if moving.sum() > 2:
            disp_unit = disp_vecs[moving] / norms[moving]
            # commanded direction is straight ahead from initial heading
            cmd_dir = np.array([np.sin(commanded_heading), np.cos(commanded_heading)])
            heading_err_deg = float(
                np.degrees(np.arccos(np.clip((disp_unit * cmd_dir).sum(axis=-1), -1, 1))).mean()
            )
        else:
            heading_err_deg = 180.0  # no movement at all

        achieved_speeds.append(achieved_speed)
        heading_errors.append(heading_err_deg)

    return {
        "commanded_speed_ms":   walk_speed,
        "achieved_speed_ms":    float(np.mean(achieved_speeds)),
        "speed_ratio":          float(np.mean(achieved_speeds)) / max(walk_speed, 1e-6),
        "heading_error_deg":    float(np.mean(heading_errors)),
        "n_clips":              len(achieved_speeds),
    }


# ---------------------------------------------------------------------------
# Check 4: carry-phase leg-stepping rate
# ---------------------------------------------------------------------------

def eval_carry_compliance(model, stats: dict, clips: list, device,
                          steps: int, stride: int,
                          warmup_frames: int, eval_frames: int) -> dict:
    """
    Seed with GT frames from OMOMO clips at a carry-walk phase frame.
    Run AR with carry_walk phase for eval_frames.
    Measure alternating-foot contact rate:
      foot joints = index 7 (left foot) and 14 (right foot) in the 24-joint skeleton.
      Proxy contact: foot z < pelvis_z * 0.25 (near ground).
      Alternating = one foot near ground, other is not.
    """
    K = int(stats["past_frames"])
    MODE_CARRY = np.array([0, 0, 0, 1, 0], dtype=np.float32)

    # joint indices (0-based, 24 joints; 0 = pelvis)
    LFOOT_IDX = 7
    RFOOT_IDX = 14

    stepping_rates = []
    root_speeds    = []

    for clip in clips:
        if not clip["has_obj"]:
            continue
        T = len(clip["root_xy"])

        # Find a carry-walk frame to seed from; need at least K frames before first carry frame
        carry_frames = np.where(clip["mode"][:, 3] > 0.5)[0]
        if len(carry_frames) < 5:
            continue
        # Seed at a carry frame far enough into the clip for K history frames
        valid = carry_frames[carry_frames >= K]
        if len(valid) == 0:
            continue
        seed_end = int(valid[0])
        if seed_end < K:
            continue

        history = [_clip_world_state(clip, f) for f in range(seed_end - K, seed_end)]

        # place goal in anchor frame (carry walks toward it)
        anchor = history[-1]
        if clip["has_obj"]:
            from animate_horizon_x15 import rot_to_anchor as rta
            goals = clip["task_goals"][min(seed_end, T - 1)].reshape(3, 3).copy()
            goals[:2, :2] = rta(goals[:2, :2] - anchor["root_xy"], float(anchor["heading"]))
            goals[:2, 2] -= float(anchor["root_z"])
            active_goal = goals[1].copy()  # place goal
        else:
            active_goal = np.zeros(3, dtype=np.float32)

        desired_heading_vec = np.array([0.0, 1.0], dtype=np.float32)

        joints_seq = []
        t = 0
        while t < eval_frames:
            cond = _build_cond(history, MODE_CARRY, desired_heading_vec, active_goal, K)
            target = sample_horizon(model, cond, stats, device, steps)
            take = min(stride, eval_frames - t)
            new = _step_ar(history, target, take)
            for s in new:
                joints_seq.append(s["joints"])
            t += take

        joints = np.stack(joints_seq, axis=0)     # (eval_frames, 24, 3)
        pelvis_z = joints[:, 0, 2]
        lfoot_z  = joints[:, LFOOT_IDX, 2]
        rfoot_z  = joints[:, RFOOT_IDX, 2]

        # Foot is "near ground" if its z < pelvis_z * 0.25
        thresh = pelvis_z * 0.25
        ldown = lfoot_z < thresh
        rdown = rfoot_z < thresh

        # Alternating: one down and not both down / both up
        alternating = np.logical_xor(ldown, rdown)
        stepping_rate = float(alternating.mean())

        # Root speed during the carry walk
        root_pos = joints[:, 0, :2]
        disp = np.linalg.norm(np.diff(root_pos, axis=0), axis=-1)
        root_speed = float(disp.mean() * FPS)

        stepping_rates.append(stepping_rate)
        root_speeds.append(root_speed)

    if not stepping_rates:
        return {"stepping_rate": float("nan"), "carry_root_speed_ms": float("nan"), "n_clips": 0}

    return {
        "stepping_rate":       float(np.mean(stepping_rates)),
        "carry_root_speed_ms": float(np.mean(root_speeds)),
        "n_clips":             len(stepping_rates),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats",      required=True)
    p.add_argument("--walk-hdf5",  default="final_h5_files/cmu_straight_walk_taskspace_with_root.h5")
    p.add_argument("--omomo-hdf5", default="artifacts/omomo_filtered/omomo_portable_new_holdout_taskspace.h5")
    p.add_argument("--n-walk-clips",   type=int,   default=10)
    p.add_argument("--n-omomo-clips",  type=int,   default=10)
    p.add_argument("--warmup-frames",  type=int,   default=30,  help="GT frames to seed warm-start")
    p.add_argument("--eval-frames",    type=int,   default=150, help="frames to evaluate after warmup")
    p.add_argument("--walk-speed",     type=float, default=0.5, help="commanded walk speed m/s")
    p.add_argument("--steps",          type=int,   default=20)
    p.add_argument("--stride",         type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.stats) as f:
        stats = json.load(f)
    if stats.get("representation") != X15_REPRESENTATION:
        raise ValueError(
            f"stats representation {stats.get('representation')!r} != {X15_REPRESENTATION!r}"
        )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()

    model = load_model(args.checkpoint, stats, device)

    # Load clips
    walk_ds = JointFlowX15Dataset(
        [args.walk_hdf5],
        past_frames=int(stats["past_frames"]),
        horizon=int(stats["horizon"]),
        stats=stats,
    )
    np.random.seed(args.seed)
    walk_idx = np.random.choice(len(walk_ds._clips),
                                min(args.n_walk_clips, len(walk_ds._clips)),
                                replace=False).tolist()
    walk_clips = [walk_ds._clips[i] for i in walk_idx]
    print(f"Walk clips: {len(walk_clips)} from {Path(args.walk_hdf5).name}")

    omomo_clips = []
    omomo_path = Path(args.omomo_hdf5)
    if omomo_path.exists():
        omomo_ds = JointFlowX15Dataset(
            [str(omomo_path)],
            past_frames=int(stats["past_frames"]),
            horizon=int(stats["horizon"]),
            stats=stats,
        )
        np.random.seed(args.seed)
        omomo_idx = np.random.choice(len(omomo_ds._clips),
                                     min(args.n_omomo_clips, len(omomo_ds._clips)),
                                     replace=False).tolist()
        omomo_clips = [omomo_ds._clips[i] for i in omomo_idx]
        print(f"OMOMO holdout clips: {len(omomo_clips)} from {omomo_path.name}")
    else:
        print(f"OMOMO holdout not found at {args.omomo_hdf5}, skipping OMOMO checks")

    print()

    # ---- Check 1+2: reconstruction metrics ---------------------------------
    print("=" * 60)
    print("CHECK 1+2: Reconstruction (walk clips)")
    print("=" * 60)
    r_walk = eval_reconstruction(model, stats, walk_clips, device, args.steps, args.stride)
    _print_recon(r_walk)

    if omomo_clips:
        print()
        print("CHECK 1+2: Reconstruction (OMOMO holdout clips)")
        print("=" * 60)
        r_omomo = eval_reconstruction(model, stats, omomo_clips, device, args.steps, args.stride)
        _print_recon(r_omomo)

    # ---- Check 3: command compliance ---------------------------------------
    print()
    print("=" * 60)
    print("CHECK 3: Warm-start command compliance (walk speed + heading)")
    print("=" * 60)
    c_walk = eval_command_compliance(
        model, stats, walk_clips, device, args.steps, args.stride,
        args.warmup_frames, args.eval_frames, args.walk_speed,
    )
    _print_compliance(c_walk)

    # ---- Check 4: carry stepping -------------------------------------------
    if omomo_clips:
        print()
        print("=" * 60)
        print("CHECK 4: Carry-phase leg-stepping rate (OMOMO holdout)")
        print("=" * 60)
        c_carry = eval_carry_compliance(
            model, stats, omomo_clips, device, args.steps, args.stride,
            args.warmup_frames, args.eval_frames,
        )
        _print_carry(c_carry)

    # ---- Gate summary ------------------------------------------------------
    print()
    print("=" * 60)
    print("GATE SUMMARY  (targets before PHC integration)")
    print("=" * 60)
    gate = {
        "TF joint L2 walk < 5 cm":       r_walk["tf_joint_l2_cm"] < 5.0,
        "TF %HJF walk < 5%":             r_walk["tf_hjf_pct"] < 5.0,
        "Walk speed ratio > 0.8":        c_walk["speed_ratio"] > 0.8,
        "Walk heading error < 30 deg":   c_walk["heading_error_deg"] < 30.0,
    }
    if omomo_clips:
        gate["TF joint L2 OMOMO < 6 cm"] = r_omomo["tf_joint_l2_cm"] < 6.0
        gate["AR joint L2 OMOMO < 35 cm"] = r_omomo["ar_joint_l2_cm"] < 35.0
        gate["Carry stepping rate > 0.5"] = c_carry.get("stepping_rate", 0.0) > 0.5

    for label, passed in gate.items():
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}]  {label}")
    print()


def _print_recon(r: dict):
    print(f"  clips:              {r['n_clips']}")
    print(f"  TF joint L2:        {r['tf_joint_l2_cm']:.2f} cm   (target < 5 cm)")
    print(f"  TF velocity error:  {r['tf_vel_error_cms']:.2f} cm/s")
    print(f"  TF %HJF:            {r['tf_hjf_pct']:.1f}%         (target < 5%)")
    print(f"  AR joint L2:        {r['ar_joint_l2_cm']:.2f} cm")
    print(f"  AR root mean drift: {r['ar_root_mean_cm']:.1f} cm")
    print(f"  AR root final drift:{r['ar_root_final_cm']:.1f} cm")


def _print_compliance(c: dict):
    print(f"  clips:              {c['n_clips']}")
    print(f"  commanded speed:    {c['commanded_speed_ms']:.2f} m/s")
    print(f"  achieved speed:     {c['achieved_speed_ms']:.3f} m/s")
    print(f"  speed ratio:        {c['speed_ratio']:.2f}         (target > 0.8)")
    print(f"  heading error:      {c['heading_error_deg']:.1f} deg      (target < 30 deg)")


def _print_carry(c: dict):
    print(f"  clips:              {c['n_clips']}")
    sr = c['stepping_rate']
    sr_str = f"{sr:.2f}" if sr == sr else "n/a"
    print(f"  stepping rate:      {sr_str}           (target > 0.50 = alternating feet)")
    rs = c['carry_root_speed_ms']
    rs_str = f"{rs:.3f} m/s" if rs == rs else "n/a"
    print(f"  carry root speed:   {rs_str}")


if __name__ == "__main__":
    main()
