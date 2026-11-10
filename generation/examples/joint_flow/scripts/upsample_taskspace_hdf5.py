#!/usr/bin/env python3
"""Upsample joint-flow taskspace HDF5 clips by temporal interpolation."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PELVIS_IDX = 0
L_HAND_IDX = 18
R_HAND_IDX = 23


def interp_linear(arr: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(arr)
    if len(arr) < 2:
        return arr.copy()
    old_t = np.arange(len(arr), dtype=np.float64)
    new_t = np.linspace(0.0, len(arr) - 1, (len(arr) - 1) * factor + 1)
    flat = arr.reshape(len(arr), -1)
    out = np.empty((len(new_t), flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(new_t, old_t, flat[:, i])
    return out.reshape((len(new_t),) + arr.shape[1:]).astype(np.float32)


def interp_rotvec(rotvec: np.ndarray, factor: int) -> np.ndarray:
    if len(rotvec) < 2:
        return rotvec.copy()
    old_t = np.arange(len(rotvec), dtype=np.float64)
    new_t = np.linspace(0.0, len(rotvec) - 1, (len(rotvec) - 1) * factor + 1)
    rots = Rotation.from_rotvec(rotvec.astype(np.float64))
    return Slerp(old_t, rots)(new_t).as_rotvec().astype(np.float32)


def interp_quat_xyzw(quat: np.ndarray, factor: int) -> np.ndarray:
    if len(quat) < 2:
        return quat.copy()
    old_t = np.arange(len(quat), dtype=np.float64)
    new_t = np.linspace(0.0, len(quat) - 1, (len(quat) - 1) * factor + 1)
    rots = Rotation.from_quat(quat.astype(np.float64))
    return Slerp(old_t, rots)(new_t).as_quat().astype(np.float32)


def extract_task_space(joint_positions: np.ndarray, fps: float) -> np.ndarray:
    pelvis = joint_positions[:, PELVIS_IDX, :]
    dt = 1.0 / fps
    vel_xy = np.empty((len(joint_positions), 2), dtype=np.float32)
    if len(joint_positions) == 1:
        vel_xy[:] = 0.0
    else:
        vel_xy[1:-1] = (pelvis[2:, :2] - pelvis[:-2, :2]) / (2.0 * dt)
        vel_xy[0] = (pelvis[1, :2] - pelvis[0, :2]) / dt
        vel_xy[-1] = (pelvis[-1, :2] - pelvis[-2, :2]) / dt
    pelvis_z = pelvis[:, 2:3].astype(np.float32)
    left_eef = joint_positions[:, L_HAND_IDX, :].astype(np.float32)
    right_eef = joint_positions[:, R_HAND_IDX, :].astype(np.float32)
    return np.concatenate([vel_xy, pelvis_z, left_eef, right_eef], axis=1)


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def upsample_file(input_path: Path, output_path: Path, factor: int, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        copy_attrs(fin, fout)
        fout.attrs["fps"] = fps
        fout.attrs["upsampled_from"] = str(input_path)
        fout.attrs["upsample_factor"] = factor

        clips_out = fout.create_group("clips")
        for clip_name in sorted(fin["clips"].keys()):
            gin = fin[f"clips/{clip_name}"]
            gout = clips_out.create_group(clip_name)
            copy_attrs(gin, gout)

            joint_positions = interp_linear(gin["joint_positions"][:], factor)
            gout.create_dataset(
                "joint_positions",
                data=joint_positions,
                dtype="f4",
                compression="gzip",
                compression_opts=4,
            )
            gout.create_dataset(
                "task_space",
                data=extract_task_space(joint_positions, fps),
                dtype="f4",
                compression="gzip",
                compression_opts=4,
            )
            if "pose_aa_root" in gin:
                gout.create_dataset(
                    "pose_aa_root",
                    data=interp_rotvec(gin["pose_aa_root"][:], factor),
                    dtype="f4",
                    compression="gzip",
                    compression_opts=4,
                )
            if "obj_com_pos_rel" in gin:
                gout.create_dataset(
                    "obj_com_pos_rel",
                    data=interp_linear(gin["obj_com_pos_rel"][:], factor),
                    dtype="f4",
                    compression="gzip",
                    compression_opts=4,
                )
            if "obj_rot_quat" in gin:
                gout.create_dataset(
                    "obj_rot_quat",
                    data=interp_quat_xyzw(gin["obj_rot_quat"][:], factor),
                    dtype="f4",
                    compression="gzip",
                    compression_opts=4,
                )

            gout.attrs["fps"] = fps
            gout.attrs["n_frames"] = len(joint_positions)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--factor", type=int, default=4)
    p.add_argument("--fps", type=float, default=120.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.factor < 1:
        raise ValueError("--factor must be >= 1")
    upsample_file(args.input, args.output, args.factor, args.fps)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
