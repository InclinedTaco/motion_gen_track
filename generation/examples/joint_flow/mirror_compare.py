"""
Side-by-side animation: original vs mirrored OMOMO clip.

Usage:
    python examples/joint_flow/mirror_compare.py \
        --h5 final_h5_files/omomo_locomanipulation_crouch_plus_portable_new_train_split_taskspace.h5 \
        --clip sub11_trashcan_000 \
        --out mirror_compare.gif   # omit for interactive window
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# ── joint skeleton ────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip",  "R_Knee","R_Ankle","R_Toe",
    "Torso",  "Spine", "Chest",  "Neck",    "Head",
    "L_Thorax","L_Shoulder","L_Elbow","L_Wrist","L_Hand",
    "R_Thorax","R_Shoulder","R_Elbow","R_Wrist","R_Hand",
]
BONES = [
    (0,9),(9,10),(10,11),(11,12),(12,13),
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (11,14),(14,15),(15,16),(16,17),(17,18),
    (11,19),(19,20),(20,21),(21,22),(22,23),
]
PELVIS_IDX = 0
_LR_SWAP_PAIRS = [
    (0,4),(1,5),(2,6),(3,7),
    (13,18),(14,19),(15,20),(16,21),(17,22),
]


def load_clip(h5path: str, clip_name: str) -> dict:
    with h5py.File(h5path, "r") as f:
        grp = f[f"clips/{clip_name}"]
        jp  = grp["joint_positions"][:].astype(np.float32)   # (T,24,3)
        ts  = grp["task_space"][:].astype(np.float32)
        paa = grp["pose_aa_root"][:].astype(np.float32) if "pose_aa_root" in grp else None
        obj_rel = grp["obj_com_pos_rel"][:].astype(np.float32) if "obj_com_pos_rel" in grp else None
    return {"jp": jp, "ts": ts, "pose_aa_root": paa, "obj_rel": obj_rel}


def mirror_jp(jp: np.ndarray) -> np.ndarray:
    """Return left-right mirrored joint positions (T,24,3)."""
    mjp = jp.copy()
    # swap L/R joint pairs (1-indexed in full 24-joint; LR pairs are +1 offset)
    lr_full = [(a+1, b+1) for a, b in _LR_SWAP_PAIRS]
    for a, b in lr_full:
        mjp[:, [a,b]] = mjp[:, [b,a]]
    # negate world-frame Y for all joints
    mjp[:, :, 1] *= -1
    return mjp


def draw_skeleton(ax, jp_frame: np.ndarray, color_l: str, color_r: str, title: str):
    ax.cla()
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    for pa, ch in BONES:
        name = JOINT_NAMES[ch]
        c = color_l if name.startswith("L") else (color_r if name.startswith("R") else "#888888")
        xs = [jp_frame[pa,0], jp_frame[ch,0]]
        ys = [jp_frame[pa,1], jp_frame[ch,1]]
        zs = [jp_frame[pa,2], jp_frame[ch,2]]
        ax.plot(xs, ys, zs, color=c, linewidth=2)
    ax.scatter(jp_frame[:,0], jp_frame[:,1], jp_frame[:,2], s=8, c="#333333", zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",   default="final_h5_files/omomo_locomanipulation_crouch_plus_portable_new_train_split_taskspace.h5")
    ap.add_argument("--clip", default="sub11_trashcan_000")
    ap.add_argument("--out",  default="mirror_compare.gif")
    ap.add_argument("--fps",  type=int, default=15)
    args = ap.parse_args()

    data   = load_clip(args.h5, args.clip)
    jp_orig = data["jp"]           # (T,24,3)
    jp_mirr = mirror_jp(jp_orig)   # (T,24,3)
    T = len(jp_orig)

    # Center each clip on its pelvis at t=0 for cleaner view
    orig_offset = jp_orig[0, PELVIS_IDX].copy(); orig_offset[2] = 0
    mirr_offset = jp_mirr[0, PELVIS_IDX].copy(); mirr_offset[2] = 0
    jp_orig = jp_orig - orig_offset
    jp_mirr = jp_mirr - mirr_offset

    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    lim = 1.2
    for ax in (ax1, ax2):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 2.0)
        ax.view_init(elev=15, azim=-60)

    def update(t):
        draw_skeleton(ax1, jp_orig[t], "#4A90D9", "#E05252", f"Original  t={t}")
        draw_skeleton(ax2, jp_mirr[t], "#4A90D9", "#E05252", f"Mirrored  t={t}")
        for ax in (ax1, ax2):
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 2.0)
            ax.view_init(elev=15, azim=-60)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000//args.fps, blit=False)
    ani.save(args.out, writer="pillow", fps=args.fps)
    print(f"Saved {args.out}  ({T} frames, clip={args.clip})")


if __name__ == "__main__":
    main()
