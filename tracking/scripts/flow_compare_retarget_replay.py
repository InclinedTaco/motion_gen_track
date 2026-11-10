"""
Compare offline and streaming flow retargeting side-by-side in Isaac Gym.

Generates one flow joint-position motion, retargets it with:
  1. fit_sequence  (offline/chunked, high quality)
  2. StreamFitter  (per-frame warm-started, realtime-style)

Then replays both SMPL humanoids kinematically in the same viewer.

Run from PHC:
  python scripts/flow_compare_retarget_replay.py \
    --vel-x 0.4 --height 0.9 --duration 50.0 --fps 30
"""

import json
import os
import sys
import time
from pathlib import Path

from isaacgym import gymapi, gymutil, gymtorch

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.getcwd())

PHC_ROOT = Path(__file__).resolve().parents[1]
FLOW_ROOT = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_ROOT / "scripts"))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))


def parse_args():
    custom = [
        {"name": "--flow-ckpt", "type": str, "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/best.pt"},
        {"name": "--flow-stats", "type": str, "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/stats.json"},
        {"name": "--humanoid-xml", "type": str, "default": "final_humanoid.xml"},
        {"name": "--vel-x", "type": float, "default": 0.4},
        {"name": "--vel-y", "type": float, "default": 0.0},
        {"name": "--height", "type": float, "default": 0.9},
        {"name": "--duration", "type": float, "default": 50.0},
        {"name": "--fps", "type": float, "default": 30.0},
        {"name": "--ramp", "type": float, "default": 0.5},
        {"name": "--ode-steps", "type": int, "default": 20},
        {"name": "--fit-iters", "type": int, "default": 500},
        {"name": "--stream-iters", "type": int, "default": 8},
        {"name": "--flow-device", "type": str, "default": "cuda"},
        {"name": "--retarget-device", "type": str, "default": "cpu"},
        {"name": "--cache", "type": str, "default": "data/generated/compare_retarget_04.pkl"},
        {"name": "--no-cache", "action": "store_true", "default": False},
        {"name": "--build-only", "action": "store_true", "default": False,
         "help": "Generate/cache both retargeted motions, then exit before opening the viewer"},
    ]
    return gymutil.parse_arguments(
        description="Compare offline fit_sequence vs StreamFitter retargeting",
        custom_parameters=custom,
    )


def build_taskspace(args):
    n_frames = int(args.duration * args.fps)
    ramp_frames = int(args.ramp * args.fps)
    ts_world = np.zeros((n_frames, 9), dtype=np.float32)
    ramp = np.ones(n_frames, dtype=np.float32)
    if ramp_frames > 0:
        ramp[:ramp_frames] = np.linspace(0.0, 1.0, ramp_frames)
    ts_world[:, 0] = args.vel_x * ramp
    ts_world[:, 1] = args.vel_y * ramp
    ts_world[:, 2] = args.height
    return ts_world


def generate_flow_positions(args, device):
    from animate import run_flow
    from model import JointFlowMLP

    with open(args.flow_stats) as f:
        stats = json.load(f)

    ckpt = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window = stats.get("window", 10)
    prev_dim = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim = 3 if stats.get("no_eef", False) else 9
    cond_dim = prev_dim + window * ts_dim

    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ts_world = build_taskspace(args)
    print(f"Generating flow positions: {len(ts_world)} frames @ {args.fps:.0f}fps")
    return run_flow(
        model, stats, ts_world, window=window, device=device,
        steps=args.ode_steps, fps=args.fps,
        init_pelvis_xy=np.zeros(2, dtype=np.float32),
    )


def retarget_offline(positions, fk, args, device):
    from flow_retarget import fit_sequence

    t0 = time.perf_counter()
    fit = fit_sequence(
        positions, fk, device=device, iters=args.fit_iters,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Offline fit: {elapsed:.2f}s total, {elapsed / len(positions) * 1000:.2f} ms/frame, "
          f"err={fit['fit_err_m'] * 100:.2f}cm")
    return fit, elapsed


def retarget_stream(positions, fk, args, device):
    from flow_retarget import StreamFitter

    fitter = StreamFitter(fk, device=device, iters=args.stream_iters)
    pose_aa = np.empty((len(positions), 72), dtype=np.float32)
    root_trans = np.empty((len(positions), 3), dtype=np.float32)
    per_frame_ms = []

    t0 = time.perf_counter()
    for i, pos in enumerate(positions):
        f0 = time.perf_counter()
        out = fitter.step(pos)
        per_frame_ms.append((time.perf_counter() - f0) * 1000.0)
        pose_aa[i] = out["pose_aa"]
        root_trans[i] = out["root_trans"]
        if i % 100 == 0:
            print(f"  stream frame {i}/{len(positions)}")
    elapsed = time.perf_counter() - t0

    with torch.no_grad():
        target = torch.from_numpy(positions).float().to(device)[None]
        pred = fk.fk_batch(
            torch.from_numpy(pose_aa).float().to(device).view(1, len(positions), 24, 3),
            torch.from_numpy(root_trans).float().to(device).view(1, len(positions), 3),
            return_full=False,
        ).global_translation
        err = (pred - target).norm(dim=-1).mean().item()

    fit = {
        "pose_aa": pose_aa,
        "root_trans_offset": root_trans,
        "fit_err_m": err,
    }
    print(f"Stream fit: {elapsed:.2f}s total, {elapsed / len(positions) * 1000:.2f} ms/frame, "
          f"median={np.median(per_frame_ms):.2f} ms, p95={np.percentile(per_frame_ms, 95):.2f} ms, "
          f"err={err * 100:.2f}cm")
    return fit, elapsed, np.asarray(per_frame_ms, dtype=np.float32)


def load_or_build(args):
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.no_cache:
        print(f"Loading cache: {cache_path}")
        return joblib.load(cache_path)

    flow_device = torch.device(args.flow_device if torch.cuda.is_available() else "cpu")
    retarget_device = torch.device(args.retarget_device)

    positions = generate_flow_positions(args, flow_device)

    from flow_retarget import build_fk
    fk = build_fk(args.humanoid_xml, retarget_device)

    offline, offline_sec = retarget_offline(positions, fk, args, retarget_device)
    stream, stream_sec, stream_ms = retarget_stream(positions, fk, args, retarget_device)

    data = {
        "fps": float(args.fps),
        "positions": positions,
        "offline": offline,
        "stream": stream,
        "offline_sec": offline_sec,
        "stream_sec": stream_sec,
        "stream_ms": stream_ms,
        "args": {
            "vel_x": args.vel_x,
            "vel_y": args.vel_y,
            "height": args.height,
            "duration": args.duration,
            "fps": args.fps,
            "ode_steps": args.ode_steps,
            "fit_iters": args.fit_iters,
            "stream_iters": args.stream_iters,
            "humanoid_xml": args.humanoid_xml,
        },
    }

    if not args.no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, cache_path)
        print(f"Saved cache: {cache_path}")
    return data


def pose_to_replay_arrays(fit):
    pose_aa = np.asarray(fit["pose_aa"], dtype=np.float32)
    root_trans = np.asarray(fit["root_trans_offset"], dtype=np.float32)
    root_quat = sRot.from_rotvec(pose_aa[:, 0:3]).as_quat().astype(np.float32)
    dof_pos = pose_aa[:, 3:72].astype(np.float32)
    return root_trans, root_quat, dof_pos


def main():
    args = parse_args()
    data = load_or_build(args)
    if args.build_only:
        print("Build-only requested; skipping Isaac Gym viewer.")
        return
    fps = float(data["fps"])
    dt = 1.0 / fps

    off_root, off_quat, off_dof = pose_to_replay_arrays(data["offline"])
    str_root, str_quat, str_dof = pose_to_replay_arrays(data["stream"])
    T = min(len(off_root), len(str_root))

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = False
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    xml_path = Path(args.humanoid_xml)
    asset_root = str(xml_path.parent) if xml_path.parent != Path("") else "."
    opts = gymapi.AssetOptions()
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    opts.collapse_fixed_joints = False
    opts.disable_gravity = True
    asset = gym.load_asset(sim, asset_root, xml_path.name, opts)
    num_dofs = gym.get_asset_dof_count(asset)

    env = gym.create_env(sim, gymapi.Vec3(-8, -5, 0), gymapi.Vec3(8, 5, 5), 2)
    starts = []
    for y in (1.0, -1.0):
        tr = gymapi.Transform()
        tr.p = gymapi.Vec3(0.0, y, float(off_root[0, 2]))
        tr.r = gymapi.Quat(0, 0, 0, 1)
        starts.append(tr)
    actor_off = gym.create_actor(env, asset, starts[0], "offline_fit", 0, 0, 0)
    actor_str = gym.create_actor(env, asset, starts[1], "stream_fit", 0, 0, 0)

    gym.prepare_sim(sim)
    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    actor_ids = [
        gym.get_actor_index(env, actor_off, gymapi.DOMAIN_SIM),
        gym.get_actor_index(env, actor_str, gymapi.DOMAIN_SIM),
    ]
    actor_ids_t = torch.tensor(actor_ids, dtype=torch.int32)

    off_root_t = torch.from_numpy(off_root[:T]).float()
    off_quat_t = torch.from_numpy(off_quat[:T]).float()
    off_dof_t = torch.from_numpy(off_dof[:T]).float()
    str_root_t = torch.from_numpy(str_root[:T]).float()
    str_quat_t = torch.from_numpy(str_quat[:T]).float()
    str_dof_t = torch.from_numpy(str_dof[:T]).float()

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(4, -6, 2.2), gymapi.Vec3(1.5, 0, 1))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "restart")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

    print("\nSide-by-side replay:")
    print("  y=+1  offline fit_sequence")
    print("  y=-1  StreamFitter")
    print("  Space=pause  R=restart  Q=quit\n")

    frame_idx = 0
    paused = False
    y_offsets = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    while not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0:
                if evt.action == "pause":
                    paused = not paused
                elif evt.action == "restart":
                    frame_idx = 0
                    paused = False
                elif evt.action == "quit":
                    gym.destroy_viewer(viewer)
                    gym.destroy_sim(sim)
                    return

        root_state[actor_ids_t[0], 0:3] = off_root_t[frame_idx] + y_offsets[0]
        root_state[actor_ids_t[0], 3:7] = off_quat_t[frame_idx]
        root_state[actor_ids_t[0], 7:] = 0
        root_state[actor_ids_t[1], 0:3] = str_root_t[frame_idx] + y_offsets[1]
        root_state[actor_ids_t[1], 3:7] = str_quat_t[frame_idx]
        root_state[actor_ids_t[1], 7:] = 0

        dof_state[0:num_dofs, 0] = off_dof_t[frame_idx]
        dof_state[0:num_dofs, 1] = 0
        dof_state[num_dofs:2 * num_dofs, 0] = str_dof_t[frame_idx]
        dof_state[num_dofs:2 * num_dofs, 1] = 0

        gym.set_actor_root_state_tensor_indexed(
            sim, gymtorch.unwrap_tensor(root_state),
            gymtorch.unwrap_tensor(actor_ids_t), len(actor_ids),
        )
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        if not paused:
            frame_idx = (frame_idx + 1) % T

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
