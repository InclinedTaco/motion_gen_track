# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())


import os
import joblib
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

dict_keys = ["betas", "dmpls", "gender", "mocap_framerate", "poses", "trans"]

# extract SMPL joints from SMPL-H model
joints_to_use = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        37,
    ]
)
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

all_sequences = [
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "DFaust_67",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "SSM_synced",
    "TCD_handMocap",
    "TotalCapture",
    "Transitions_mocap",
    "BMLhandball",
    "DanceDB"
]

# CMU subjects that are primarily locomotion (walk, run, jog, jump, stairs).
# Excludes manipulation subjects: 54-56 (object interaction), 62 (tools),
# 70 (carrying), 79-80 (household tasks), 115 (pick up box),
# 123 (carrying), 141-144 (pick up / toss-catch).
CMU_LOCOMOTION_SUBJECTS = {
    "01", "02", "05", "06", "07", "08", "09", "10", "11", "12", "13", "15",
    "16", "17",
    # 18-23: two-person interaction (handshakes, wrestling, passing objects) — excluded
    # 24-32: nursery rhymes, animal pantomimes — excluded
    "35", "36", "37", "38", "39", "40", "41", "43", "45", "46",
    "47", "49", "60", "61", "63", "64", "73", "74", "75", "76", "77", "78",
    "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "93", "94",
    "102", "103", "105", "106", "107", "108", "111", "113", "118", "120",
    "122", "124", "125", "126", "127", "128", "131", "132", "133", "135",
    "136", "137", "138", "139", "140",
}

def read_data(folder, sequences, subject_filter=None):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    if sequences == "all":
        sequences = all_sequences

    db = {}
    print(folder)
    for seq_name in sequences:
        print(f"Reading {seq_name} sequence...")
        seq_folder = osp.join(folder, seq_name)

        filter_set = subject_filter.get(seq_name) if subject_filter else None
        datas = read_single_sequence(seq_folder, seq_name, subject_filter=filter_set)
        db.update(datas)
        print(seq_name, "number of seqs", len(datas))

    return db


def read_single_sequence(folder, seq_name, subject_filter=None):
    subjects = os.listdir(folder)

    datas = {}

    for subject in tqdm(subjects):
        if not osp.isdir(osp.join(folder, subject)):
            continue
        if subject_filter is not None and subject not in subject_filter:
            continue
        actions = [
            x for x in os.listdir(osp.join(folder, subject)) if x.endswith(".npz")
        ]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith("shape.npz"):
                continue

            data = dict(np.load(fname))
            # data['poses'] = pose = data['poses'][:, joints_to_use]

            # shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)
            # theta = np.concatenate([pose,shape], axis=1)
            vid_name = f"{seq_name}_{subject}_{action[:-4]}"

            datas[vid_name] = data
            # thetas.append(theta)

    return datas


def read_seq_data(folder, nsubjects, fps):
    subjects = os.listdir(folder)
    sequences = {}

    assert nsubjects < len(subjects), "nsubjects should be less than len(subjects)"

    for subject in subjects[:nsubjects]:
        actions = os.listdir(osp.join(folder, subject))

        for action in actions:
            data = np.load(osp.join(folder, subject, action))
            mocap_framerate = int(data["mocap_framerate"])
            sampling_freq = mocap_framerate // fps
            sequences[(subject, action)] = data["poses"][
                0::sampling_freq, joints_to_use
            ]

    train_set = {}
    test_set = {}

    for i, (k, v) in enumerate(sequences.items()):
        if i < len(sequences.keys()) - len(sequences.keys()) // 4:
            train_set[k] = v
        else:
            test_set[k] = v

    return train_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="dataset directory", default="data/amass"
    )
    parser.add_argument(
        "--out_dir", type=str, help="dataset directory", default="out"
    )
    parser.add_argument(
        '--sequences', type=str, nargs='+', help='which sequences to use', default=all_sequences
    )

    parser.add_argument(
        '--cmu_locomotion_only', action='store_true',
        help='For CMU, only include locomotion subjects (excludes manipulation)'
    )

    args = parser.parse_args()
    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True)
    db_file = osp.join(out_path, "amass_db_smplh.pt")

    subject_filter = {"CMU": CMU_LOCOMOTION_SUBJECTS} if args.cmu_locomotion_only else None
    db = read_data(args.dir, sequences=args.sequences, subject_filter=subject_filter)
     
    
    print(f"Saving AMASS dataset to {db_file}")
    joblib.dump(db, db_file)
