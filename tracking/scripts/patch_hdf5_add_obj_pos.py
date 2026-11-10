#!/usr/bin/env python3
"""
Patch the OMOMO HDF5 file to add per-frame object CoM position
relative to pelvis (obj_com_pos_rel, T×3) and object rotation
as quaternion (obj_rot_quat, T×4 xyzw).

Usage (preferred — pre-extracted pkl):
    python scripts/patch_hdf5_add_obj_pos.py \
        --hdf5    data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \
        --obj-pkl data/amass/pkls/omomo_obj_trajectories.pkl

Usage (legacy — reads directly from tar):
    python scripts/patch_hdf5_add_obj_pos.py \
        --hdf5 data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \
        --tar  /path/to/omomo/data.tar.gz

Generate the pkl once with:
    python scripts/extract_omomo_obj_trajs.py \
        --tar /path/to/omomo/data.tar.gz \
        --out data/amass/pkls/omomo_obj_trajectories.pkl
"""
import argparse
import io
import tarfile

import h5py
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_seqs(source, is_tar):
    """Load raw OMOMO sequences from either a tar or a pre-extracted pkl."""
    if is_tar:
        seqs = {}
        with tarfile.open(source, "r:gz") as tar:
            for split in ("train", "test"):
                name = f"data/{split}_diffusion_manip_seq_joints24.p"
                f = tar.extractfile(name)
                d = joblib.load(io.BytesIO(f.read()))
                for v in d.values():
                    seqs[v["seq_name"]] = v
    else:
        seqs = joblib.load(source)
    print(f"Loaded {len(seqs)} sequences from {source}")
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5",    required=True)
    p.add_argument("--obj-pkl", default=None, help="Pre-extracted object trajectories pkl (preferred)")
    p.add_argument("--tar",     default=None, help="OMOMO data.tar.gz (legacy fallback)")
    args = p.parse_args()

    if args.obj_pkl is None and args.tar is None:
        p.error("Provide either --obj-pkl or --tar")

    seqs = load_seqs(args.obj_pkl or args.tar, is_tar=args.tar is not None)

    with h5py.File(args.hdf5, "a") as f:
        clips = list(f["clips"].keys())
        matched, skipped = 0, 0
        for clip in clips:
            if clip not in seqs:
                print(f"  SKIP (no match): {clip}")
                skipped += 1
                continue

            grp = f[f"clips/{clip}"]
            seq = seqs[clip]
            trans   = np.array(seq["trans"],       dtype=np.float32)  # (T, 3)
            obj_pos = np.array(seq["obj_com_pos"], dtype=np.float32)  # (T, 3)
            obj_rot = np.array(seq["obj_rot"],     dtype=np.float32)  # (T, 3, 3)

            T_hdf5 = grp["task_space"].shape[0]
            if T_hdf5 != trans.shape[0]:
                print(f"  SKIP (frame mismatch {T_hdf5} vs {trans.shape[0]}): {clip}")
                skipped += 1
                continue

            rel  = obj_pos - trans
            quat = R.from_matrix(obj_rot).as_quat().astype(np.float32)  # xyzw

            if "obj_com_pos_rel" not in grp:
                grp.create_dataset("obj_com_pos_rel", data=rel)
            if "obj_rot_quat" not in grp:
                grp.create_dataset("obj_rot_quat", data=quat)
            matched += 1

        print(f"\nDone. Patched {matched} clips, skipped {skipped}.")


if __name__ == "__main__":
    main()
