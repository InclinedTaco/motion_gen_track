"""
Copy a 6D taskspace HDF5 and add pose_aa_root (T, 3) to each clip.

The root axis-angle comes from pose_quat[:, 0, :] in the source PKL —
the same local quaternion the extractor already uses for pose_aa_body,
just the pelvis joint (index 0) which was previously dropped.

Usage:
    python scripts/patch_hdf5_add_root.py \
        data/amass/hdf5/accad_6d_taskspace.h5 \
        data/amass/hdf5/accad_6d_root_taskspace.h5 \
        --pkl data/amass/pkls/final_humanoid/accad_isaac_upright_slim_research_filtered.pkl

    # or omit --pkl and let it read source_file from the HDF5 attrs
    python scripts/patch_hdf5_add_root.py \
        data/amass/hdf5/cmu_6d_taskspace.h5 \
        data/amass/hdf5/cmu_6d_root_taskspace.h5
"""

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_h5",  help="source 6D HDF5 (read-only)")
    p.add_argument("output_h5", help="destination HDF5 (copy + patched)")
    p.add_argument("--pkl",     default=None,
                   help="source PKL; if omitted, read from HDF5 source_file attr")
    return p.parse_args()


def main():
    args = parse_args()
    in_path  = Path(args.input_h5)
    out_path = Path(args.output_h5)

    if not in_path.exists():
        sys.exit(f"ERROR: input not found: {in_path}")

    # find PKL path
    if args.pkl:
        pkl_path = Path(args.pkl)
    else:
        with h5py.File(in_path, "r") as f:
            pkl_path = Path(f.attrs.get("source_file", ""))
    if not pkl_path.exists():
        sys.exit(f"ERROR: PKL not found: {pkl_path}\n"
                 f"Pass --pkl explicitly if the path has changed.")

    print(f"Input  HDF5 : {in_path}")
    print(f"Output HDF5 : {out_path}")
    print(f"PKL         : {pkl_path}")

    # copy
    print("Copying HDF5 …")
    shutil.copy2(in_path, out_path)

    # load PKL
    print("Loading PKL …")
    data = joblib.load(str(pkl_path))
    # build lookup: sanitized_name → clip_dict
    pkl_lookup = {
        k.replace("/", "_").replace("\\", "_"): v
        for k, v in data.items()
    }

    added = skipped = 0
    with h5py.File(out_path, "a") as f:
        clips = list(f["clips"].keys())
        for clip_name in clips:
            grp = f[f"clips/{clip_name}"]

            if "pose_aa_root" in grp:
                skipped += 1
                continue

            if clip_name not in pkl_lookup:
                print(f"  [WARN] {clip_name} not in PKL — skipping")
                skipped += 1
                continue

            clip = pkl_lookup[clip_name]
            if "pose_quat" not in clip:
                print(f"  [WARN] {clip_name} has no pose_quat — skipping")
                skipped += 1
                continue

            T = grp["task_space"].shape[0]
            pq = np.asarray(clip["pose_quat"], dtype=np.float32)[:T]  # (T, 24, 4) xyzw local
            root_aa = sRot.from_quat(pq[:, 0, :]).as_rotvec().astype(np.float32)  # (T, 3)

            grp.create_dataset("pose_aa_root", data=root_aa,
                               dtype="f4", compression="gzip", compression_opts=4)
            added += 1

    print(f"\nDone.  added={added}  skipped={skipped}  →  {out_path}")


if __name__ == "__main__":
    main()
