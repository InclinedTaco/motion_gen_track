#!/usr/bin/env python3
"""
Raw HDF5/PKL motion visualizer for joint_flow data.

HDF5 files are expected to contain /clips/<name>/joint_positions with shape
(T, 24, 3). PKL support is limited to clips that already contain joint_positions;
PHC rotation-only PKLs need poselib + the humanoid XML and should be converted to
HDF5 first.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


BONES = [
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (11, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (11, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
]

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


def bone_color(child_idx):
    name = JOINT_NAMES[child_idx]
    if name.startswith("L"):
        return "#4A90D9"
    if name.startswith("R"):
        return "#E05252"
    return "#777777"


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def heading_rel_to_world(rel_xyz, pelvis_world, pose_aa_root):
    from scipy.spatial.transform import Rotation as R

    yaws = R.from_rotvec(pose_aa_root).as_euler("ZYX")[:, 0]
    c, s = np.cos(yaws), np.sin(yaws)
    world_x = c * rel_xyz[:, 0] - s * rel_xyz[:, 1] + pelvis_world[:, 0]
    world_y = s * rel_xyz[:, 0] + c * rel_xyz[:, 1] + pelvis_world[:, 1]
    world_z = rel_xyz[:, 2] + pelvis_world[:, 2]
    return np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)


def clip_names_h5(path):
    import h5py

    with h5py.File(path, "r") as f:
        if "clips" not in f:
            raise ValueError(f"{path} has no /clips group")
        return list(f["clips"].keys())


def resolve_clip(names, clip):
    if clip in names:
        return clip
    try:
        idx = int(clip)
    except ValueError as exc:
        raise ValueError(f"clip {clip!r} is not a name or integer index") from exc
    if idx < 0 or idx >= len(names):
        raise IndexError(f"clip index {idx} out of range for {len(names)} clips")
    return names[idx]


def load_h5_clip(path, clip, show_object):
    import h5py

    with h5py.File(path, "r") as f:
        names = list(f["clips"].keys())
        name = resolve_clip(names, clip)
        grp = f["clips"][name]
        if "joint_positions" not in grp:
            raise ValueError(f"{path}:{name} has no joint_positions dataset")
        joints = grp["joint_positions"][:].astype(np.float32)
        fps = float(grp.attrs.get("fps", 30.0))
        obj_world = None
        if show_object and "obj_com_pos_rel" in grp:
            obj_rel = grp["obj_com_pos_rel"][:].astype(np.float32)
            pelvis = joints[:, 0, :].astype(np.float32)
            if "pose_aa_root" in grp:
                obj_world = heading_rel_to_world(
                    obj_rel, pelvis, grp["pose_aa_root"][:].astype(np.float32)
                )
            else:
                obj_world = pelvis + obj_rel
    return joints, fps, name, obj_world


def pkl_names(path):
    import joblib

    data = joblib.load(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is a {type(data).__name__}, expected dict")
    return list(data.keys()), data


def load_pkl_clip(path, clip):
    names, data = pkl_names(path)
    name = resolve_clip(names, clip)
    item = data[name]
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{name} is a {type(item).__name__}, expected dict")
    if "joint_positions" not in item:
        keys = ", ".join(sorted(item.keys()))
        raise ValueError(
            f"{path}:{name} has no joint_positions. It appears to be a rotation/"
            f"parameter PKL with keys: {keys}. Convert it to HDF5 first with "
            "extract_taskspace_hdf5.py, or add poselib + final_humanoid.xml support."
        )
    joints = as_numpy(item["joint_positions"]).astype(np.float32)
    fps = float(item.get("fps", 30.0))
    return joints, fps, name, None


def list_file(path, limit):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        names = clip_names_h5(path)
        print(f"{path}: {len(names)} clips")
        for i, name in enumerate(names[:limit]):
            print(f"  {i:4d}  {name}")
        if len(names) > limit:
            print(f"  ... {len(names) - limit} more")
        return
    if suffix == ".pkl":
        names, data = pkl_names(path)
        print(f"{path}: {len(names)} clips/items")
        for i, name in enumerate(names[:limit]):
            item = data[name]
            if isinstance(item, dict):
                keys = ", ".join(list(item.keys())[:8])
                print(f"  {i:4d}  {name}  keys=[{keys}]")
            else:
                print(f"  {i:4d}  {name}  type={type(item).__name__}")
        if len(names) > limit:
            print(f"  ... {len(names) - limit} more")
        return
    raise ValueError(f"unsupported file extension: {path.suffix}")


def load_motion(path, clip, show_object):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return load_h5_clip(path, clip, show_object)
    if suffix == ".pkl":
        return load_pkl_clip(path, clip)
    raise ValueError(f"unsupported file extension: {path.suffix}")


def draw_bones(ax, joints):
    lines = []
    for parent, child in BONES:
        (line,) = ax.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=bone_color(child),
            linewidth=2,
        )
        lines.append(line)
    pts = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="black", s=12)
    return lines, pts


def set_axes_equalish(ax, joints, follow):
    if follow:
        center = joints[0, 0, :2]
        ax.set_xlim(center[0] - 1.0, center[0] + 1.0)
        ax.set_ylim(center[1] - 1.0, center[1] + 1.0)
    else:
        xy = joints[:, :, :2].reshape(-1, 2)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        span = max(float((hi - lo).max()), 1.0)
        center = (lo + hi) / 2.0
        pad = 0.15 * span
        half = 0.5 * span + pad
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    z = joints[:, :, 2]
    ax.set_zlim(max(0.0, float(z.min()) - 0.15), float(z.max()) + 0.25)


def render(joints, fps, title, out, follow, obj_world=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=12, azim=-65)
    set_axes_equalish(ax, joints, follow=False)

    lines, pts = draw_bones(ax, joints[0])
    obj_pt = None
    if obj_world is not None:
        obj_pt = ax.scatter(
            [obj_world[0, 0]],
            [obj_world[0, 1]],
            [obj_world[0, 2]],
            c="#39b54a",
            s=50,
            depthshade=False,
        )

    def update(frame):
        current = joints[frame]
        for (parent, child), line in zip(BONES, lines):
            line.set_data_3d(
                [current[parent, 0], current[child, 0]],
                [current[parent, 1], current[child, 1]],
                [current[parent, 2], current[child, 2]],
            )
        pts._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
        if obj_pt is not None:
            obj = obj_world[frame]
            obj_pt._offsets3d = ([obj[0]], [obj[1]], [obj[2]])
        if follow:
            set_axes_equalish(ax, joints[frame : frame + 1], follow=True)
        fig.suptitle(f"{title} | frame {frame + 1}/{len(joints)}", fontsize=9)
        artists = lines + [pts]
        if obj_pt is not None:
            artists.append(obj_pt)
        return artists

    update(0)
    if out is None:
        plt.show()
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
        return

    ani = animation.FuncAnimation(
        fig, update, frames=len(joints), interval=1000.0 / fps, blit=False
    )
    writer = "pillow" if out.suffix.lower() == ".gif" else "ffmpeg"
    ani.save(out, writer=writer, fps=fps)
    print(f"Saved {out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="HDF5 or PKL motion file")
    parser.add_argument("--clip", default="0", help="Clip index or clip name")
    parser.add_argument("--list", action="store_true", help="List clips/items and exit")
    parser.add_argument("--limit", type=int, default=20, help="List limit")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None, help="Override render FPS")
    parser.add_argument(
        "--out", default=None, help=".gif, .mp4, or .png; omit for interactive"
    )
    parser.add_argument("--follow", action="store_true", help="Center camera on pelvis")
    parser.add_argument(
        "--show-object", action="store_true", help="Render obj_com_pos_rel when present"
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        path = Path(args.path)
        if not path.exists():
            raise FileNotFoundError(path)
        if args.stride < 1:
            raise ValueError("--stride must be >= 1")

        if args.list:
            list_file(path, args.limit)
            return

        joints, fps, name, obj_world = load_motion(path, args.clip, args.show_object)
        if joints.ndim != 3 or joints.shape[1:] != (24, 3):
            raise ValueError(f"expected joints shape (T, 24, 3), got {joints.shape}")

        end = min(len(joints), args.max_frames) if args.max_frames else len(joints)
        joints = joints[: end : args.stride]
        if obj_world is not None:
            obj_world = obj_world[: end : args.stride]
        render_fps = float(args.fps if args.fps is not None else fps) / args.stride
        print(f"Clip: {name} | frames={len(joints)} | render_fps={render_fps:.2f}")
        render(
            joints,
            render_fps,
            name,
            Path(args.out) if args.out else None,
            args.follow,
            obj_world,
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
