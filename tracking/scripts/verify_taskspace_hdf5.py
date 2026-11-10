#!/usr/bin/env python3
"""Verification script for extracted task-space HDF5 data."""

import h5py
import numpy as np
import matplotlib
from typing import Dict, List
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from pathlib import Path

HDF5_PATH = "data/amass/hdf5/accad_research_filtered_taskspace.h5"
OUT_DIR = Path("scripts/verify_taskspace_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── column layout ────────────────────────────────────────────────────────────
# task_space (T, 9):
#   0,1  → root velocity XY
#   2    → pelvis height
#   3,4,5 → left_eef_xyz
#   6,7,8 → right_eef_xyz
#
# joint_positions (T, 24, 3):
#   index 22 → L_Hand  → should match left_eef_xyz  (cols 3,4,5)
#   index 23 → R_Hand  → should match right_eef_xyz (cols 6,7,8)

PELVIS_H_COL   = 2
LEFT_EEF_COLS  = slice(3, 6)    # xyz
RIGHT_EEF_COLS = slice(6, 9)    # xyz
LEFT_EEF_Z     = 5
RIGHT_EEF_Z    = 8
ROOT_VEL_COLS  = slice(0, 2)
L_HAND_IDX     = 18   # poselib order (L_Hand)
R_HAND_IDX     = 23   # poselib order (R_Hand)

BOUNDS = dict(
    pelvis_h_min=0.0,
    pelvis_h_max=2.5,
    pelvis_h_standing_min=0.8,
    pelvis_h_standing_max=1.15,
    hand_z_min=0.0,
    hand_z_max=2.5,
    root_vel_max=5.0,
    frame_jump_max=0.1,
)


def load(f: h5py.File, clip: str):
    ts  = f[f"clips/{clip}/task_space"][:]
    jp  = f[f"clips/{clip}/joint_positions"][:]
    return ts, jp


# ── 1. Sanity checks ─────────────────────────────────────────────────────────

def sanity_check(clips: List[str], f: h5py.File) -> Dict[str, List[str]]:
    violations: Dict[str, List[str]] = {}

    for clip in clips:
        ts, _ = load(f, clip)
        flags = []

        pelvis_h = ts[:, PELVIS_H_COL]
        if np.any(pelvis_h < BOUNDS["pelvis_h_min"]):
            flags.append(f"pelvis_height negative (min={pelvis_h.min():.3f})")
        if np.any(pelvis_h > BOUNDS["pelvis_h_max"]):
            flags.append(f"pelvis_height > {BOUNDS['pelvis_h_max']} (max={pelvis_h.max():.3f})")

        for side, col in [("left", LEFT_EEF_Z), ("right", RIGHT_EEF_Z)]:
            z = ts[:, col]
            if np.any(z < BOUNDS["hand_z_min"]):
                flags.append(f"{side}_hand Z below 0 (min={z.min():.3f})")
            if np.any(z > BOUNDS["hand_z_max"]):
                flags.append(f"{side}_hand Z > {BOUNDS['hand_z_max']} (max={z.max():.3f})")

        vel_mag = np.linalg.norm(ts[:, ROOT_VEL_COLS], axis=1)
        if np.any(vel_mag > BOUNDS["root_vel_max"]):
            bad = vel_mag[vel_mag > BOUNDS["root_vel_max"]]
            flags.append(
                f"root_vel exceeds {BOUNDS['root_vel_max']} m/s "
                f"({len(bad)} frames, max={bad.max():.2f})"
            )

        if flags:
            violations[clip] = flags

    return violations


# ── 2. Single-clip visualisation ─────────────────────────────────────────────

def visualise_clip(clip: str, f: h5py.File, out_path: Path):
    ts, _ = load(f, clip)
    T = ts.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Task-space visualisation\n{clip}", fontsize=10)

    axes[0].plot(t, ts[:, PELVIS_H_COL], color="steelblue")
    axes[0].axhline(BOUNDS["pelvis_h_standing_min"], color="gray", ls="--", lw=0.8, label="standing bounds")
    axes[0].axhline(BOUNDS["pelvis_h_standing_max"], color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("Pelvis height (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ts[:, LEFT_EEF_Z],  label="Left hand Z",  color="tomato")
    axes[1].plot(t, ts[:, RIGHT_EEF_Z], label="Right hand Z", color="seagreen")
    axes[1].set_ylabel("Hand Z height (m)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    vel_mag = np.linalg.norm(ts[:, ROOT_VEL_COLS], axis=1)
    axes[2].plot(t, vel_mag, color="darkorange")
    axes[2].axhline(BOUNDS["root_vel_max"], color="red", ls="--", lw=0.8, label=f"limit {BOUNDS['root_vel_max']} m/s")
    axes[2].set_ylabel("Root vel magnitude (m/s)")
    axes[2].set_xlabel("Frame")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved → {out_path}")


# ── 3. Cross-check joint positions vs task_space EEFs ────────────────────────

def cross_check_joints(clips: List[str], f: h5py.File, n_clips: int = 5, n_frames: int = 5):
    rng = random.Random(42)
    sample_clips = rng.sample(clips, min(n_clips, len(clips)))
    threshold = 1e-3  # metres – HDF5 stores float32 so tiny FP errors expected
    mismatches = []

    for clip in sample_clips:
        ts, jp = load(f, clip)
        T = ts.shape[0]
        frames = rng.sample(range(T), min(n_frames, T))

        for fi in frames:
            # L_Hand (22) ↔ left_eef (cols 3,4,5)
            l_hand_jp = jp[fi, L_HAND_IDX, :]
            left_eef  = ts[fi, LEFT_EEF_COLS]
            err_l = np.max(np.abs(l_hand_jp - left_eef))
            if err_l > threshold:
                mismatches.append(
                    f"  {clip}  frame {fi}: "
                    f"L_Hand jp={l_hand_jp.round(4)} vs left_eef={left_eef.round(4)}  Δ={err_l:.4f}"
                )

            # R_Hand (23) ↔ right_eef (cols 6,7,8)
            r_hand_jp = jp[fi, R_HAND_IDX, :]
            right_eef = ts[fi, RIGHT_EEF_COLS]
            err_r = np.max(np.abs(r_hand_jp - right_eef))
            if err_r > threshold:
                mismatches.append(
                    f"  {clip}  frame {fi}: "
                    f"R_Hand jp={r_hand_jp.round(4)} vs right_eef={right_eef.round(4)}  Δ={err_r:.4f}"
                )

    return sample_clips, mismatches


# ── 4. Temporal consistency ───────────────────────────────────────────────────

def temporal_consistency(clips: List[str], f: h5py.File) -> Dict[str, List[str]]:
    discontinuities: Dict[str, List[str]] = {}
    thr = BOUNDS["frame_jump_max"]

    for clip in clips:
        _, jp = load(f, clip)
        flags = []

        # check pelvis position via root joint (index 0) in world-frame joint_positions
        root_pos = jp[:, 0, :]   # (T, 3)
        jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)  # (T-1,)
        bad = np.where(jumps > thr)[0]
        if len(bad):
            flags.append(
                f"root joint jump > {thr} m: {len(bad)} frames, "
                f"max={jumps[bad].max():.4f} at frame {bad[jumps[bad].argmax()]}"
            )

        # also check each hand
        for name, idx in [("L_Hand", L_HAND_IDX), ("R_Hand", R_HAND_IDX)]:
            hand_pos = jp[:, idx, :]
            h_jumps  = np.linalg.norm(np.diff(hand_pos, axis=0), axis=1)
            bad_h    = np.where(h_jumps > thr)[0]
            if len(bad_h):
                flags.append(
                    f"{name} jump > {thr} m: {len(bad_h)} frames, "
                    f"max={h_jumps[bad_h].max():.4f} at frame {bad_h[h_jumps[bad_h].argmax()]}"
                )

        if flags:
            discontinuities[clip] = flags

    return discontinuities


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Opening {HDF5_PATH} …")
    with h5py.File(HDF5_PATH, "r") as f:
        clips = sorted(f["clips"].keys())
        print(f"Found {len(clips)} clips.\n")

        # ── 1. Sanity checks ──────────────────────────────────────────────────
        print("=" * 60)
        print("1. SANITY CHECKS")
        print("=" * 60)
        violations = sanity_check(clips, f)
        if violations:
            print(f"  {len(violations)} clip(s) with violations:")
            for clip, flags in violations.items():
                print(f"\n  [{clip}]")
                for fl in flags:
                    print(f"    ✗ {fl}")
        else:
            print("  All clips pass value bounds. ✓")

        # ── 2. Visualise a Walking clip ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("2. VISUALISATION")
        print("=" * 60)
        walking_clips = [c for c in clips if "Walking" in c]
        if walking_clips:
            vis_clip = next((c for c in walking_clips if "B2 " in c or "B1 " in c), walking_clips[0])
        else:
            vis_clip = clips[0]
            print(f"  No 'Walking' clip found; falling back to: {vis_clip}")
        print(f"  Visualising: {vis_clip}")
        visualise_clip(vis_clip, f, OUT_DIR / "taskspace_walking.png")

        # ── 3. Cross-check joint positions ───────────────────────────────────
        print("\n" + "=" * 60)
        print("3. CROSS-CHECK JOINT POSITIONS vs TASK_SPACE")
        print("=" * 60)
        sample_clips, mismatches = cross_check_joints(clips, f)
        print(f"  Sampled clips: {[c.split('_c3d_')[1] if '_c3d_' in c else c for c in sample_clips]}")
        if mismatches:
            print(f"  {len(mismatches)} mismatch(es) found:")
            for m in mismatches:
                print(m)
        else:
            print("  All sampled frames match within tolerance. ✓")

        # ── 4. Temporal consistency ───────────────────────────────────────────
        print("\n" + "=" * 60)
        print("4. TEMPORAL CONSISTENCY (frame-to-frame jumps)")
        print("=" * 60)
        discontinuities = temporal_consistency(clips, f)
        if discontinuities:
            print(f"  {len(discontinuities)} clip(s) with discontinuities:")
            for clip, flags in discontinuities.items():
                print(f"\n  [{clip}]")
                for fl in flags:
                    print(f"    ✗ {fl}")
        else:
            print("  No discontinuities found. ✓")

        # ── summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total clips          : {len(clips)}")
        print(f"  Sanity violations    : {len(violations)}")
        print(f"  Cross-check failures : {len(mismatches)}")
        print(f"  Discontinuous clips  : {len(discontinuities)}")
        print(f"  Plot saved to        : {OUT_DIR}/taskspace_walking.png")


if __name__ == "__main__":
    main()
