#!/usr/bin/env python3
"""
Extract per-clip object trajectories from the OMOMO data.tar.gz and save
as a standalone joblib pickle in the same format as the tar sequences.

The output pkl maps clip name → the original sequence dict, same as what
load_omomo_seqs() reads from the tar (fields include trans, obj_com_pos,
obj_rot, seq_name, etc.).

Usage:
    python scripts/extract_omomo_obj_trajs.py \
        --tar  /home/marmot/Downloads/data.tar.gz \
        --out  data/amass/pkls/omomo_obj_trajectories.pkl
"""
import argparse
import io
import tarfile

import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar", required=True, help="Path to OMOMO data.tar.gz")
    p.add_argument("--out", required=True, help="Output pkl path")
    args = p.parse_args()

    print(f"Reading {args.tar} ...")
    seqs = {}
    with tarfile.open(args.tar, "r:gz") as tar:
        for split in ("train", "test"):
            name = f"data/{split}_diffusion_manip_seq_joints24.p"
            f = tar.extractfile(name)
            d = joblib.load(io.BytesIO(f.read()))
            for v in d.values():
                seqs[v["seq_name"]] = v
    print(f"  {len(seqs)} sequences loaded")

    joblib.dump(seqs, args.out)
    print(f"Saved {len(seqs)} clips → {args.out}")


if __name__ == "__main__":
    main()
