"""
Side-by-side Isaac Gym replay of two flow models on a real HDF5 clip.

Loads a clip from HDF5, runs both models (e.g. EEF and OBJ), retargets each
with fit_sequence (Humanoid_Batch gradient fit), then replays both humanoids
kinematically side-by-side in the same viewer.

Run from PHC directory:
  python scripts/flow_clip_compare.py \
    --hdf5   /path/to/omomo_locomanipulation_crouch_taskspace.h5 \
    --clip   45 \
    --ckpt-a ../generation/checkpoints/joint_flow_eef_omomo_prevframe/best.pt \
    --stats-a ../generation/checkpoints/joint_flow_eef_omomo_prevframe/stats.json \
    --label-a EEF \
    --ckpt-b ../generation/checkpoints/joint_flow_obj_accad_omomo/best.pt \
    --stats-b ../generation/checkpoints/joint_flow_obj_accad_omomo/stats.json \
    --label-b OBJ \
    --humanoid-xml final_humanoid.xml

Controls: Space=pause  R=restart  Q/Esc=quit
"""

import json
import os
import sys
import time
from pathlib import Path

from isaacgym import gymapi, gymutil, gymtorch

import h5py
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

PHC_ROOT = Path(__file__).resolve().parents[1]
FLOW_ROOT = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_ROOT / "scripts"))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))


def parse_args():
    custom = [
        {"name": "--hdf5",       "type": str, "required": True},
        {"name": "--clip",       "type": int, "default": 0},
        {"name": "--ckpt-a",     "type": str, "required": True},
        {"name": "--stats-a",    "type": str, "required": True},
        {"name": "--label-a",    "type": str, "default": "Model A"},
        {"name": "--ckpt-b",     "type": str, "required": True},
        {"name": "--stats-b",    "type": str, "required": True},
        {"name": "--label-b",    "type": str, "default": "Model B"},
        {"name": "--humanoid-xml", "type": str, "default": "final_humanoid.xml"},
        {"name": "--ode-steps",  "type": int,   "default": 20},
        {"name": "--fit-iters",  "type": int,   "default": 500},
        {"name": "--seed",       "type": int,   "default": 42},
        {"name": "--cache",      "type": str,   "default": None,
         "help": "pkl to load/save retargeted results; skips inference+fit if exists"},
        {"name": "--no-cache",   "action": "store_true", "default": False},
        {"name": "--build-only", "action": "store_true", "default": False,
         "help": "Run inference+retarget and save cache, then exit without viewer"},
        {"name": "--flow-device",    "type": str, "default": "cuda"},
        {"name": "--retarget-device","type": str, "default": "cpu"},
    ]
    return gymutil.parse_arguments(
        description="Side-by-side flow model comparison on a real HDF5 clip",
        custom_parameters=custom,
    )


def load_clip(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips = sorted(f["clips"].keys())
        name  = clips[clip_idx]
        grp   = f[f"clips/{name}"]
        ts    = grp["task_space"][:]
        jp    = grp["joint_positions"][:]
        fps   = float(grp.attrs["fps"])
        obj_rel = grp["obj_com_pos_rel"][:].astype(np.float32) if "obj_com_pos_rel" in grp else None
    print(f"Clip [{clip_idx}]: {name}  |  {len(ts)} frames @ {fps:.0f} fps  |  has_obj={obj_rel is not None}")
    return ts, jp, fps, name, obj_rel


def load_model(ckpt_path, stats, device):
    from model import JointFlowMLP
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window  = stats.get("window", 10)
    prev_dim = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim   = (3 if stats.get("no_eef", False) else 9) + (3 if stats.get("use_obj", False) else 0)
    cond_dim = prev_dim + window * ts_dim
    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stats.get("window", 10)


def run_model(ckpt_path, stats_path, ts_world, jp_gt, obj_rel, device, ode_steps, seed):
    from animate import run_flow
    with open(stats_path) as f:
        stats = json.load(f)
    model, window = load_model(ckpt_path, stats, device)
    torch.manual_seed(seed)
    pred = run_flow(
        model, stats, ts_world, window, device,
        steps=ode_steps, fps=30.0,
        init_pelvis_xy=jp_gt[0, 0, :2],
        obj_rel=obj_rel,
    )
    return pred  # (T, 24, 3) world-frame


def retarget(positions, fk, device, iters):
    from flow_retarget import fit_sequence
    t0 = time.perf_counter()
    fit = fit_sequence(positions, fk, device=device, iters=iters, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"  fit done: {elapsed:.1f}s  err={fit['fit_err_m']*100:.2f} cm")
    return fit


def load_or_build(args):
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists() and not args.no_cache:
        print(f"Loading cache: {cache_path}")
        return joblib.load(cache_path)

    flow_device     = torch.device(args.flow_device if torch.cuda.is_available() else "cpu")
    retarget_device = torch.device(args.retarget_device)

    ts_world, jp_gt, fps, clip_name, obj_rel = load_clip(args.hdf5, args.clip)

    print(f"\nRunning {args.label_a} …")
    pos_a = run_model(args.ckpt_a, args.stats_a, ts_world, jp_gt, obj_rel,
                      flow_device, args.ode_steps, args.seed)

    print(f"\nRunning {args.label_b} …")
    pos_b = run_model(args.ckpt_b, args.stats_b, ts_world, jp_gt, obj_rel,
                      flow_device, args.ode_steps, args.seed)

    from flow_retarget import build_fk
    fk = build_fk(args.humanoid_xml, retarget_device)

    print(f"\nRetargeting {args.label_a} …")
    fit_a = retarget(pos_a, fk, retarget_device, args.fit_iters)

    print(f"\nRetargeting {args.label_b} …")
    fit_b = retarget(pos_b, fk, retarget_device, args.fit_iters)

    data = {
        "clip_name": clip_name,
        "fps":       fps,
        "label_a":   args.label_a,
        "label_b":   args.label_b,
        "fit_a":     fit_a,
        "fit_b":     fit_b,
    }

    if cache_path and not args.no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, cache_path)
        print(f"Saved cache: {cache_path}")

    return data


def pose_to_arrays(fit):
    pose_aa    = np.asarray(fit["pose_aa"],           dtype=np.float32)
    root_trans = np.asarray(fit["root_trans_offset"], dtype=np.float32)
    root_quat  = sRot.from_rotvec(pose_aa[:, 0:3]).as_quat().astype(np.float32)
    dof_pos    = pose_aa[:, 3:72].astype(np.float32)
    return root_trans, root_quat, dof_pos


def main():
    args = parse_args()
    data = load_or_build(args)

    if args.build_only:
        print("Build-only; exiting before viewer.")
        return

    fps = float(data["fps"])
    dt  = 1.0 / fps

    root_a, quat_a, dof_a = pose_to_arrays(data["fit_a"])
    root_b, quat_b, dof_b = pose_to_arrays(data["fit_b"])
    T = min(len(root_a), len(root_b))

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

    xml_path   = Path(args.humanoid_xml)
    asset_root = str(xml_path.parent) if xml_path.parent != Path("") else "."
    opts = gymapi.AssetOptions()
    opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    opts.collapse_fixed_joints = False
    opts.disable_gravity = True
    asset = gym.load_asset(sim, asset_root, xml_path.name, opts)
    num_dofs = gym.get_asset_dof_count(asset)

    env = gym.create_env(sim, gymapi.Vec3(-8, -5, 0), gymapi.Vec3(8, 5, 5), 2)
    for y, label, actor_name in [(1.0, data["label_a"], "actor_a"),
                                  (-1.0, data["label_b"], "actor_b")]:
        tr = gymapi.Transform()
        tr.p = gymapi.Vec3(0.0, y, float(root_a[0, 2]))
        tr.r = gymapi.Quat(0, 0, 0, 1)
        gym.create_actor(env, asset, tr, actor_name, 0, 0, 0)

    gym.prepare_sim(sim)
    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state  = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    actor_ids = [
        gym.get_actor_index(env, gym.get_actor_handle(env, 0), gymapi.DOMAIN_SIM),
        gym.get_actor_index(env, gym.get_actor_handle(env, 1), gymapi.DOMAIN_SIM),
    ]
    actor_ids_t = torch.tensor(actor_ids, dtype=torch.int32)

    root_a_t = torch.from_numpy(root_a[:T]).float()
    quat_a_t = torch.from_numpy(quat_a[:T]).float()
    dof_a_t  = torch.from_numpy(dof_a[:T]).float()
    root_b_t = torch.from_numpy(root_b[:T]).float()
    quat_b_t = torch.from_numpy(quat_b[:T]).float()
    dof_b_t  = torch.from_numpy(dof_b[:T]).float()

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(4, -6, 2.2), gymapi.Vec3(1.5, 0, 1))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,  "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,      "restart")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,      "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

    print(f"\nClip: {data['clip_name']}")
    print(f"  y=+1  {data['label_a']}")
    print(f"  y=-1  {data['label_b']}")
    print("  Space=pause  R=restart  Q=quit\n")

    frame_idx = 0
    paused    = False
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

        root_state[actor_ids_t[0], 0:3] = root_a_t[frame_idx] + y_offsets[0]
        root_state[actor_ids_t[0], 3:7] = quat_a_t[frame_idx]
        root_state[actor_ids_t[0], 7:]  = 0
        root_state[actor_ids_t[1], 0:3] = root_b_t[frame_idx] + y_offsets[1]
        root_state[actor_ids_t[1], 3:7] = quat_b_t[frame_idx]
        root_state[actor_ids_t[1], 7:]  = 0

        dof_state[0:num_dofs,          0] = dof_a_t[frame_idx]
        dof_state[0:num_dofs,          1] = 0
        dof_state[num_dofs:2*num_dofs, 0] = dof_b_t[frame_idx]
        dof_state[num_dofs:2*num_dofs, 1] = 0

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
