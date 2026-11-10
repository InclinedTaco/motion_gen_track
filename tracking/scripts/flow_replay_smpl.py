"""
flow_replay_smpl.py

Kinematic replay of flow-model motion on the FULL articulated SMPL humanoid
in Isaac Gym (no policy, no physics tracking — the body is driven directly
from the retargeted pose each frame).

This is the articulated-body analogue of flow_replay.py (which only drew
spheres at joint positions).  It uses the gradient-fit retargeter
(flow_retarget.py): the SMPL humanoid's DOF are exactly the per-joint
axis-angle of pose_aa, and the root is pose_aa[:3] + root translation.

Run from the PHC directory:

  # generate + replay (matches flow_replay.py's interface)
  python scripts/flow_replay_smpl.py --vel-x 0.4 --vel-y 0.0 --height 0.9 --duration 5.0

  # or replay an existing pkl made by flow_to_pkl.py
  python scripts/flow_replay_smpl.py --motion-file data/generated/walk_forward_04.pkl

Controls:
  Space  pause / resume
  R      restart
  Q/Esc  quit
"""

import os
import sys
import json
from pathlib import Path

from isaacgym import gymapi, gymutil, gymtorch

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.getcwd())

PHC_ROOT       = Path(__file__).resolve().parents[1]
FLOW_ROOT      = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_ROOT / "scripts"))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))


def parse_args():
    custom = [
        {"name": "--motion-file", "type": str,   "default": "",
         "help": "Existing pkl from flow_to_pkl.py; if set, --vel/--height ignored"},
        {"name": "--flow-ckpt",  "type": str,   "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/best.pt"},
        {"name": "--flow-stats", "type": str,   "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/stats.json"},
        {"name": "--humanoid-xml", "type": str, "default": "final_humanoid.xml"},
        {"name": "--vel-x",      "type": float, "default": 0.4},
        {"name": "--vel-y",      "type": float, "default": 0.0},
        {"name": "--height",     "type": float, "default": 0.9},
        {"name": "--duration",   "type": float, "default": 5.0},
        {"name": "--fps",        "type": float, "default": 30.0},
        {"name": "--ramp",       "type": float, "default": 0.5},
        {"name": "--ode-steps",  "type": int,   "default": 20},
        {"name": "--fit-iters",  "type": int,   "default": 500},
        {"name": "--flow-device","type": str,   "default": "cuda"},
    ]
    return gymutil.parse_arguments(
        description="Flow model full-body SMPL kinematic replay in Isaac Gym",
        custom_parameters=custom,
    )


def get_pose_sequence(args, flow_device):
    """Return (pose_aa (T,72), root_trans (T,3), fps).

    Either load from a pkl or generate (flow inference + gradient-fit retarget).
    """
    from flow_retarget import build_fk, fit_sequence

    if args.motion_file:
        import joblib
        data = joblib.load(args.motion_file)
        key = list(data.keys())[0]
        m = data[key]
        pose_aa = np.asarray(m["pose_aa"], dtype=np.float32)            # (T,72)
        root_trans = np.asarray(m["root_trans_offset"], dtype=np.float32)  # (T,3)
        fps = float(m.get("fps", args.fps))
        print(f"Loaded pkl '{key}'  ({len(pose_aa)} frames @ {fps:.0f} fps)")
        return pose_aa, root_trans, fps

    # generate
    from model import JointFlowMLP
    from animate import run_flow, run_flow_6d

    with open(args.flow_stats) as f:
        stats = json.load(f)
    ckpt       = torch.load(args.flow_ckpt, map_location=flow_device, weights_only=False)
    train_args = ckpt.get("args", {})
    window     = stats.get("window", 10)
    use_6d     = stats.get("use_6d", False)
    joint_dim  = 138 if use_6d else 69
    prev_dim   = joint_dim if stats.get("use_prev_frame", False) else 0
    ts_dim     = 3 if stats.get("no_eef", False) else 9
    cond_dim   = prev_dim + window * ts_dim
    model = JointFlowMLP(joint_dim=joint_dim, cond_dim=cond_dim,
                         hidden_size=train_args.get("hidden", 512),
                         n_layers=train_args.get("layers", 3)).to(flow_device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"Flow model loaded  (cond_dim={cond_dim}, use_6d={use_6d})")

    fps = args.fps
    n_frames    = int(args.duration * fps)
    ramp_frames = int(args.ramp * fps)
    ts_world    = np.zeros((n_frames, 9), dtype=np.float32)
    ramp        = np.ones(n_frames, dtype=np.float32)
    if ramp_frames > 0:
        ramp[:ramp_frames] = np.linspace(0.0, 1.0, ramp_frames)
    ts_world[:, 0] = args.vel_x * ramp
    ts_world[:, 1] = args.vel_y * ramp
    ts_world[:, 2] = args.height

    print(f"Generating {n_frames} frames  vel=({args.vel_x:.2f},{args.vel_y:.2f})  h={args.height:.2f} …")

    if use_6d:
        pose_aa, root_trans = run_flow_6d(
            model, stats, ts_world, window=window, device=flow_device,
            steps=args.ode_steps, fps=fps, init_pelvis_xy=np.zeros(2, dtype=np.float32),
        )
        print("6D mode: skipping retargeter.")
        return pose_aa.astype(np.float32), root_trans.astype(np.float32), fps

    pred = run_flow(model, stats, ts_world, window=window, device=flow_device,
                    steps=args.ode_steps, fps=fps, init_pelvis_xy=np.zeros(2, dtype=np.float32))

    print("Gradient-fit retargeting to SMPL pose …")
    fk  = build_fk(args.humanoid_xml, flow_device)
    fit = fit_sequence(pred, fk, device=flow_device, iters=args.fit_iters)
    print(f"Done.  fit error {fit['fit_err_m']*100:.2f} cm")
    return fit["pose_aa"].astype(np.float32), fit["root_trans_offset"].astype(np.float32), fps


def main():
    args = parse_args()
    flow_device = torch.device(args.flow_device if torch.cuda.is_available() else "cpu")

    pose_aa, root_trans, fps = get_pose_sequence(args, flow_device)
    T = len(pose_aa)
    dt = 1.0 / fps

    # root quaternion (xyzw) from root axis-angle
    root_quat = sRot.from_rotvec(pose_aa[:, 0:3]).as_quat().astype(np.float32)   # (T,4) xyzw
    # DOF = per-joint axis-angle tail. This is exactly PHC's SMPL convention
    # (_local_rotation_to_dof_smpl = quat_to_exp_map); verified best for Isaac
    # Gym (~2.4cm body error vs FK; euler decompositions are worse on Isaac).
    dof_pos   = pose_aa[:, 3:72].astype(np.float32)                              # (T,69)

    # ── Isaac Gym ────────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt      = dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)          # kinematic: no gravity
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu     = args.use_gpu
    sim_params.use_gpu_pipeline  = False                     # keep states on CPU for easy writes

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)

    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    # load the SMPL humanoid asset
    xml_path  = Path(args.humanoid_xml)
    asset_root = str(xml_path.parent) if xml_path.parent != Path("") else "."
    asset_file = xml_path.name
    a_opts = gymapi.AssetOptions()
    a_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    a_opts.collapse_fixed_joints  = False
    a_opts.disable_gravity        = True
    asset = gym.load_asset(sim, asset_root, asset_file, a_opts)
    num_dofs = gym.get_asset_dof_count(asset)
    print(f"Asset '{asset_file}'  num_dofs={num_dofs}  (pose_aa tail={dof_pos.shape[1]})")
    if num_dofs != dof_pos.shape[1]:
        print("WARNING: DOF count mismatch — check humanoid_xml matches the fit.")

    env = gym.create_env(sim, gymapi.Vec3(-5, -5, 0), gymapi.Vec3(5, 5, 5), 1)
    start = gymapi.Transform()
    start.p = gymapi.Vec3(float(root_trans[0, 0]), float(root_trans[0, 1]), float(root_trans[0, 2]))
    start.r = gymapi.Quat(*root_quat[0])
    actor = gym.create_actor(env, asset, start, "smpl", 0, 0, 0)

    gym.prepare_sim(sim)

    root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))  # (1,13)
    dof_state  = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))         # (num_dofs,2)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    root_quat_t  = torch.from_numpy(root_quat).float()
    root_trans_t = torch.from_numpy(root_trans).float()
    dof_pos_t    = torch.from_numpy(dof_pos).float()

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(3, -3, 2), gymapi.Vec3(0, 0, 1))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,  "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,      "restart")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,      "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

    print(f"\nReplaying {T} frames on the SMPL humanoid. Space=pause R=restart Q=quit\n")
    frame_idx = 0
    paused    = False

    while not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0:
                if evt.action == "pause":   paused = not paused
                elif evt.action == "restart": frame_idx = 0; paused = False
                elif evt.action == "quit":
                    gym.destroy_viewer(viewer); gym.destroy_sim(sim); return

        # write kinematic state for this frame
        root_state[0, 0:3] = root_trans_t[frame_idx]
        root_state[0, 3:7] = root_quat_t[frame_idx]      # xyzw
        root_state[0, 7:]  = 0.0
        dof_state[:, 0]    = dof_pos_t[frame_idx]
        dof_state[:, 1]    = 0.0

        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))

        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        if not paused:
            frame_idx = (frame_idx + 1) % T

    gym.destroy_viewer(viewer); gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
