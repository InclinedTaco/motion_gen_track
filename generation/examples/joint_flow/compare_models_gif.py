"""
Generate a side-by-side GIF comparing two flow models on the same taskspace.

Usage:
    python examples/joint_flow/compare_models_gif.py \
        --ckpt-a  checkpoints/joint_flow_noeef_prevframe/best.pt \
        --stats-a checkpoints/joint_flow_noeef_prevframe/stats.json \
        --label-a "Position model" \
        --ckpt-b  checkpoints/joint_flow_6d/best.pt \
        --stats-b checkpoints/joint_flow_6d/stats.json \
        --label-b "6D model" \
        --vel-x 0.4 --height 0.9 --duration 10.0 \
        --out compare_6d_vs_pos.gif

    # Synthetic pick/carry/place:
    python examples/joint_flow/compare_models_gif.py \
        --ckpt-a ... --stats-a ... --label-a "x13" \
        --ckpt-b ... --stats-b ... --label-b "x13_ss" \
        --motion pick_carry_place --duration 6.0 --speed 0.50 \
        --pickup-x 0.55 --pickup-y -0.20 --pickup-z 0.25 \
        --place-x  1.65 --place-y  0.30 --place-z  0.25 \
        --hold-x 0.29 --hold-y 0.09 --hold-z 0.13 \
        --out compare_synthetic.gif
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from animate import (
    run_flow,
    run_flow_6d,
    draw_bones,
    BONES,
    load_clip,
    _load_clip_task_goals,
    _load_clip_carry_phase,
    _load_clip_task_phase,
    _integrate_pelvis_xy,
    _make_task_goals_from_world,
    _task_phase_dim,
    _carry_walk_phase_index,
    _build_synthetic_task_phase,
    _match_task_phase_to_stats,
    _task_goal_markers_world,
    _clip_goal_markers_world,
    set_skeleton_colors,
)
from model import JointFlowMLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a",   required=True)
    p.add_argument("--stats-a",  required=True)
    p.add_argument("--label-a",  default="Model A")
    p.add_argument("--ckpt-b",   required=True)
    p.add_argument("--stats-b",  required=True)
    p.add_argument("--label-b",  default="Model B")
    p.add_argument("--humanoid-xml", default=None, help="Required if either model is 6D")
    p.add_argument("--vel-x",    type=float, default=0.4)
    p.add_argument("--vel-y",    type=float, default=0.0)
    p.add_argument("--height",   type=float, default=0.9)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps",      type=float, default=30.0)
    p.add_argument("--hdf5",     default=None, help="HDF5 file for clip reconstruction mode")
    p.add_argument("--clip",     type=int, default=0, help="Clip index for --hdf5 mode")
    p.add_argument("--steps",    type=int,   default=20)
    p.add_argument("--out",      default="compare.gif")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--highlight-carry", action="store_true",
                   help="Render both model predictions green on carry/carry-walk frames")
    p.add_argument("--linear-sentinel-a", action="store_true",
                   help="Use linearly decreasing obj sentinel for model A instead of fixed (2,2,2)")
    p.add_argument("--linear-sentinel-b", action="store_true",
                   help="Use linearly decreasing obj sentinel for model B instead of fixed (2,2,2)")
    p.add_argument("--omomo-hdf5", default=None,
                   help="Path to OMOMO HDF5; required when either --linear-sentinel-* is set")
    # synthetic motion args
    p.add_argument("--motion", default=None, choices=["pick_carry_place"],
                   help="Synthetic motion type (currently only pick_carry_place)")
    p.add_argument("--speed",    type=float, default=0.50)
    p.add_argument("--pickup-x", type=float, default=None)
    p.add_argument("--pickup-y", type=float, default=None)
    p.add_argument("--pickup-z", type=float, default=None)
    p.add_argument("--place-x",  type=float, default=None)
    p.add_argument("--place-y",  type=float, default=None)
    p.add_argument("--place-z",  type=float, default=None)
    p.add_argument("--hold-x",   type=float, default=0.29)
    p.add_argument("--hold-y",   type=float, default=0.09)
    p.add_argument("--hold-z",   type=float, default=0.13)
    return p.parse_args()


def load_model(ckpt_path, stats, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window       = stats.get("window", 10)
    use_6d       = stats.get("use_6d", False)
    predict_root = stats.get("predict_root", False)
    joint_dim    = (144 if predict_root else 138) if use_6d else 69
    prev_dim     = joint_dim if stats.get("use_prev_frame", False) else 0
    phase_dim    = _task_phase_dim(stats)
    ts_dim = (
        (3 if stats.get("no_eef", False) else 9)
        + (3 if stats.get("use_obj", False) else 0)
        + (9 if stats.get("use_task_goals", False) else 0)
        + (1 if stats.get("use_carry_phase", False) else 0)
        + phase_dim
    )
    cond_dim  = prev_dim + window * ts_dim
    model = JointFlowMLP(
        joint_dim=joint_dim,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, window


def build_linear_sentinel(omomo_hdf5, T):
    """Compute the linearly decreasing obj sentinel used for locomotion clips during training."""
    import h5py
    with h5py.File(omomo_hdf5, "r") as f:
        clips = list(f["clips"].keys())
        arrays = [f["clips"][c]["obj_com_pos_rel"][:] for c in clips
                  if "obj_com_pos_rel" in f["clips"][c]]
    max_obj = np.concatenate(arrays, axis=0).max(0).astype(np.float32)
    sentinel_end   = max_obj + 0.1
    sentinel_start = sentinel_end * 2.0
    t_arr = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None]
    return sentinel_start[None] + (sentinel_end - sentinel_start)[None] * t_arr


def get_positions(
    model,
    stats,
    ts_world,
    window,
    device,
    steps,
    fps,
    humanoid_xml,
    init_pelvis_xy=None,
    obj_rel=None,
    obj_sentinel=None,
    task_goals=None,
    carry_phase=None,
    task_phase=None,
    pose_aa_root=None,
):
    use_6d = stats.get("use_6d", False)
    if use_6d:
        if humanoid_xml is None:
            raise ValueError("--humanoid-xml required for 6D model")
        import sys as _sys
        _phc_root = str(Path(__file__).resolve().parents[3] / "tracking")
        _sys.path.insert(0, str(Path(_phc_root) / "poselib"))
        from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
        from scipy.spatial.transform import Rotation as _sRot

        sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
        pose_aa_np, root_trans_np = run_flow_6d(
            model, stats, ts_world, window, device, steps=steps, fps=fps,
            init_pelvis_xy=init_pelvis_xy,
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
            carry_phase=carry_phase,
            task_phase=task_phase,
            pose_aa_root=pose_aa_root,
        )
        T = len(pose_aa_np)
        quats = _sRot.from_rotvec(pose_aa_np.reshape(-1, 3)).as_quat()
        quats_t = torch.from_numpy(quats).float().reshape(T, 24, 4)
        rt_t = torch.from_numpy(root_trans_np).float()
        sk_state = SkeletonState.from_rotation_and_root_translation(
            sk_tree, quats_t, rt_t, is_local=True
        )
        return sk_state.global_translation.numpy()
    else:
        return run_flow(
            model,
            stats,
            ts_world,
            window,
            device,
            steps=steps,
            fps=fps,
            init_pelvis_xy=init_pelvis_xy,
            obj_rel=obj_rel,
            obj_sentinel=obj_sentinel,
            task_goals=task_goals,
            carry_phase=carry_phase,
            task_phase=task_phase,
            pose_aa_root=pose_aa_root,
        )


def build_pick_carry_place(args, stats_a, stats_b):
    """Build synthetic ts_world, task_phase, task_goals for pick_carry_place."""
    fps_ = args.fps
    carry_spd = args.speed
    dur_pu, dur_cs, dur_cw, dur_pl = 0.82, 0.72, 1.50, 1.15
    scale = args.duration / (dur_pu + dur_cs + dur_cw + dur_pl)
    n_pu = max(4, int(dur_pu * scale * fps_))
    n_cs = max(2, int(dur_cs * scale * fps_))
    n_cw = max(4, int(dur_cw * scale * fps_))
    n_pl = max(4, int(dur_pl * scale * fps_))
    T = n_pu + n_cs + n_cw + n_pl
    t0, t1, t2, t3, t4 = 0, n_pu, n_pu + n_cs, n_pu + n_cs + n_cw, T

    vel = np.zeros(T, dtype=np.float32)
    vel[t0:t1] = 0.10
    ramp = max(1, n_cw // 6)
    vel[t2 : t2 + ramp] = np.linspace(0, carry_spd, ramp)
    vel[t2 + ramp : t3 - ramp] = carry_spd
    vel[t3 - ramp : t3] = np.linspace(carry_spd, 0, ramp)
    t_cw = np.arange(n_cw, dtype=np.float32)
    gait_freq = 1.8
    noise = 0.06 * np.sin(2 * np.pi * gait_freq * t_cw / fps_) + \
            0.03 * np.sin(2 * np.pi * 2 * gait_freq * t_cw / fps_)
    vel[t2:t3] = np.clip(vel[t2:t3] + noise, 0.05, carry_spd * 1.3).astype(np.float32)
    vel[t3:t4] = 0.10

    pz = np.full(T, args.height, dtype=np.float32)
    crouch_z = 0.70
    pu_dip = max(1, int(0.4 * n_pu))
    pz[t0 : t0 + pu_dip] = np.linspace(args.height, crouch_z, pu_dip)
    pz[t0 + pu_dip : t1] = np.linspace(crouch_z, 0.98, n_pu - pu_dip)
    pz[t1:t3] = 0.98
    pl_dip = max(1, int(0.4 * n_pl))
    pz[t3 : t3 + pl_dip] = np.linspace(0.98, crouch_z, pl_dip)
    pz[t3 + pl_dip : t4] = np.linspace(crouch_z, args.height, n_pl - pl_dip)

    vy = np.zeros(T, dtype=np.float32)
    vy[t2:t3] = 0.15 * np.sin(2 * np.pi * 0.4 * t_cw / fps_)

    ts_world = np.zeros((T, 9), dtype=np.float32)
    ts_world[:, 0] = vel
    ts_world[:, 1] = vy
    ts_world[:, 2] = pz

    carry_phase = np.zeros((T, 1), dtype=np.float32)
    carry_phase[t1:t3, 0] = 1.0

    custom_pickup = all(v is not None for v in [args.pickup_x, args.pickup_y, args.pickup_z])
    custom_place  = all(v is not None for v in [args.place_x,  args.place_y,  args.place_z])
    hold_rel = np.array([args.hold_x, args.hold_y, args.hold_z], dtype=np.float32)

    def _task_goals(stats):
        if not stats.get("use_task_goals", False):
            return None
        pelvis_xy = _integrate_pelvis_xy(ts_world, fps_)
        pickup_world = np.array([
            args.pickup_x if custom_pickup else 0.43,
            args.pickup_y if custom_pickup else 0.05,
            args.pickup_z if custom_pickup else 0.49,
        ], dtype=np.float32)
        place_world = np.array([
            args.place_x if custom_place else pelvis_xy[-1, 0] + 0.45,
            args.place_y if custom_place else pelvis_xy[-1, 1] + 0.03,
            args.place_z if custom_place else args.height - 0.31,
        ], dtype=np.float32)
        return _make_task_goals_from_world(
            ts_world, fps_, pickup_world, place_world, hold_rel,
            task_goal_frame=stats.get("task_goal_frame", "current"),
        )

    def _task_phase(stats):
        if not stats.get("use_task_phase", False):
            return None
        return _build_synthetic_task_phase(_task_phase_dim(stats), t0, t1, t2, t3, t4)

    def _carry_phase(stats):
        return carry_phase if stats.get("use_carry_phase", False) else None

    clip_name = f"pick_carry_place  carry_spd={carry_spd:.2f}  dur={args.duration:.1f}s"
    print(f"Generating 'pick_carry_place'  ({T} frames, {T/fps_:.1f}s) …")
    return ts_world, _task_goals, _task_phase, _carry_phase, clip_name


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats_a) as f: stats_a = json.load(f)
    with open(args.stats_b) as f: stats_b = json.load(f)

    if args.hdf5 is not None:
        ts_world, jp_gt, clip_fps, clip_name, obj_rel, pose_aa_root = load_clip(
            args.hdf5, args.clip
        )
        fps = args.fps if args.fps > 0 else clip_fps
        task_goals_a = (
            _load_clip_task_goals(args.hdf5, args.clip, task_goal_frame=stats_a.get("task_goal_frame", "current"))
            if stats_a.get("use_task_goals", False)
            else None
        )
        task_goals_b = (
            _load_clip_task_goals(args.hdf5, args.clip, task_goal_frame=stats_b.get("task_goal_frame", "current"))
            if stats_b.get("use_task_goals", False)
            else None
        )
        carry_phase_a = _load_clip_carry_phase(args.hdf5, args.clip) if stats_a.get("use_carry_phase", False) else None
        carry_phase_b = _load_clip_carry_phase(args.hdf5, args.clip) if stats_b.get("use_carry_phase", False) else None
        task_phase_a = (
            _match_task_phase_to_stats(_load_clip_task_phase(args.hdf5, args.clip), stats_a)
            if stats_a.get("use_task_phase", False)
            else None
        )
        task_phase_b = (
            _match_task_phase_to_stats(_load_clip_task_phase(args.hdf5, args.clip), stats_b)
            if stats_b.get("use_task_phase", False)
            else None
        )
        init_pelvis_xy = jp_gt[0, 0, :2]
    elif args.motion == "pick_carry_place":
        ts_world, _tg, _tp, _cp, clip_name = build_pick_carry_place(args, stats_a, stats_b)
        jp_gt = None
        obj_rel = None
        fps = args.fps
        task_goals_a   = _tg(stats_a)
        task_goals_b   = _tg(stats_b)
        task_phase_a   = _match_task_phase_to_stats(_tp(stats_a), stats_a)
        task_phase_b   = _match_task_phase_to_stats(_tp(stats_b), stats_b)
        carry_phase_a  = _cp(stats_a)
        carry_phase_b  = _cp(stats_b)
        init_pelvis_xy = None
        pose_aa_root = None
    else:
        T = int(args.duration * args.fps)
        ts_world = np.zeros((T, 9), dtype=np.float32)
        ts_world[:, 0] = args.vel_x
        ts_world[:, 1] = args.vel_y
        ts_world[:, 2] = args.height
        jp_gt = None
        clip_name = None
        obj_rel = None
        fps = args.fps
        task_goals_a = task_goals_b = None
        carry_phase_a = carry_phase_b = None
        task_phase_a = task_phase_b = None
        init_pelvis_xy = None
        pose_aa_root = None

    need_sentinel = (args.linear_sentinel_a or args.linear_sentinel_b)
    linear_sentinel = None
    if need_sentinel:
        if args.omomo_hdf5 is None:
            raise ValueError("--omomo-hdf5 is required when --linear-sentinel-a/b is set")
        linear_sentinel = build_linear_sentinel(args.omomo_hdf5, len(ts_world))

    model_a, window_a = load_model(args.ckpt_a, stats_a, device)
    model_b, window_b = load_model(args.ckpt_b, stats_b, device)

    print(f"Generating {args.label_a} …")
    pred_a = get_positions(
        model_a, stats_a, ts_world, window_a, device, args.steps, fps, args.humanoid_xml,
        init_pelvis_xy=init_pelvis_xy,
        obj_rel=obj_rel,
        obj_sentinel=linear_sentinel if (args.linear_sentinel_a and stats_a.get("use_obj")) else None,
        task_goals=task_goals_a,
        carry_phase=carry_phase_a,
        task_phase=task_phase_a,
        pose_aa_root=pose_aa_root,
    )
    print(f"Generating {args.label_b} …")
    pred_b = get_positions(
        model_b, stats_b, ts_world, window_b, device, args.steps, fps, args.humanoid_xml,
        init_pelvis_xy=init_pelvis_xy,
        obj_rel=obj_rel,
        obj_sentinel=linear_sentinel if (args.linear_sentinel_b and stats_b.get("use_obj")) else None,
        task_goals=task_goals_b,
        carry_phase=carry_phase_b,
        task_phase=task_phase_b,
        pose_aa_root=pose_aa_root,
    )
    print("Done. Rendering GIF …")

    T = min(len(pred_a), len(pred_b))
    interval = 1000.0 / fps
    carry_highlight = np.zeros(T, dtype=bool)
    if args.highlight_carry:
        if task_phase_b is not None:
            carry_highlight = task_phase_b[:T, _carry_walk_phase_index(stats_b)] > 0.5
        elif carry_phase_b is not None:
            carry_highlight = carry_phase_b[:T, 0] > 0.5
        elif task_phase_a is not None:
            carry_highlight = task_phase_a[:T, _carry_walk_phase_index(stats_a)] > 0.5
        elif carry_phase_a is not None:
            carry_highlight = carry_phase_a[:T, 0] > 0.5
    pickup_marker, place_marker = _clip_goal_markers_world(
        ts_world, jp_gt, obj_rel, pose_aa_root
    )
    if pickup_marker is None or place_marker is None:
        marker_task_goals = task_goals_b if task_goals_b is not None else task_goals_a
        marker_stats = stats_b if task_goals_b is not None else stats_a
        pickup_marker, place_marker = _task_goal_markers_world(
            marker_task_goals, marker_stats, ts_world, fps, pose_aa_root
        )

    fig = plt.figure(figsize=(12, 6))
    ax_a = fig.add_subplot(121, projection="3d")
    ax_b = fig.add_subplot(122, projection="3d")
    ax_a.set_title(args.label_a, fontsize=10)
    ax_b.set_title(args.label_b, fontsize=10)
    for ax in [ax_a, ax_b]:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_zlim(0, 2.0)
        ax.view_init(elev=15, azim=-60)
        if pickup_marker is not None:
            ax.scatter(
                [pickup_marker[0]], [pickup_marker[1]], [pickup_marker[2]],
                c="#39b54a", s=45, depthshade=False
            )
        if place_marker is not None:
            ax.scatter(
                [place_marker[0]], [place_marker[1]], [place_marker[2]],
                c="#d94343", s=45, depthshade=False
            )

    lines_a, pts_a = draw_bones(ax_a, pred_a[0])
    lines_b, pts_b = draw_bones(ax_b, pred_b[0])

    def update(t):
        for ax, pred, lines, pts in [
            (ax_a, pred_a, lines_a, pts_a),
            (ax_b, pred_b, lines_b, pts_b),
        ]:
            j = pred[t]
            for (p, c), ln in zip(BONES, lines):
                ln.set_data_3d([j[p,0],j[c,0]], [j[p,1],j[c,1]], [j[p,2],j[c,2]])
            pts._offsets3d = (j[:,0], j[:,1], j[:,2])
            set_skeleton_colors(lines, pts, carry_highlight[t])
            cx, cy = j[0,0], j[0,1]
            ax.set_xlim(cx-1.2, cx+1.2); ax.set_ylim(cy-1.2, cy+1.2)
        title = f"frame {t}/{T}"
        if clip_name is not None:
            title = f"{clip_name}  |  {title}"
        fig.suptitle(title, fontsize=9)
        return lines_a + lines_b + [pts_a, pts_b]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    ani.save(args.out, writer="pillow", fps=fps)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
