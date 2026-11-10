"""
flow_replay.py

Replay flow model output in Isaac Gym by placing one sphere per joint
and setting positions directly — no IK, no DOF, exact positions.

Run from PHC directory:

  python scripts/flow_replay.py \\
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \\
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json \\
    --vel-x 0.4 --vel-y 0.0 --height 0.9 --duration 5.0

Controls:
  Space   pause / resume
  R       restart
  Q/Esc   quit
"""

import os
import sys
import json
from pathlib import Path

from isaacgym import gymapi, gymutil, gymtorch

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

FLOW_ROOT      = Path(__file__).resolve().parents[2] / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))

N_JOINTS = 24

# bones: (parent_idx, child_idx) — for coloring left/right
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # left leg
    (0, 5), (5, 6), (6, 7), (7, 8),           # right leg
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),  # spine + head
    (11, 14), (14, 15), (15, 16), (16, 17), (17, 18),  # left arm
    (11, 19), (19, 20), (20, 21), (21, 22), (22, 23),  # right arm
]


def parse_args():
    custom = [
        {"name": "--flow-ckpt",  "type": str,   "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/best.pt"},
        {"name": "--flow-stats", "type": str,   "default":
         "../generation/checkpoints/joint_flow_noeef_prevframe/stats.json"},
        {"name": "--vel-x",      "type": float, "default": 0.5},
        {"name": "--vel-y",      "type": float, "default": 0.0},
        {"name": "--height",     "type": float, "default": 0.9},
        {"name": "--duration",   "type": float, "default": 5.0},
        {"name": "--fps",        "type": float, "default": 30.0},
        {"name": "--ramp",       "type": float, "default": 0.5},
        {"name": "--ode-steps",  "type": int,   "default": 10},
        {"name": "--flow-device","type": str,   "default": "cuda"},
    ]
    return gymutil.parse_arguments(
        description="Flow model kinematic replay in Isaac Gym",
        custom_parameters=custom,
    )


def main():
    args = parse_args()
    fps  = args.fps
    dt   = 1.0 / fps

    # ── generate motion ──────────────────────────────────────────────────────
    from model import JointFlowMLP
    from animate import run_flow

    with open(args.flow_stats) as f:
        stats = json.load(f)

    flow_device = torch.device(args.flow_device if torch.cuda.is_available() else "cpu")
    ckpt        = torch.load(args.flow_ckpt, map_location=flow_device, weights_only=False)
    train_args  = ckpt.get("args", {})
    window      = stats.get("window", 10)
    prev_dim    = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim      = 3  if stats.get("no_eef", False) else 9
    cond_dim    = prev_dim + window * ts_dim

    model = JointFlowMLP(
        joint_dim=69, cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(flow_device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Flow model loaded  (cond_dim={cond_dim})")

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
    positions = run_flow(model, stats, ts_world, window=window,
                         device=flow_device, steps=args.ode_steps, fps=fps,
                         init_pelvis_xy=np.zeros(2, dtype=np.float32))
    # positions: (T, 24, 3) world-frame Z-up
    print("Done.")

    T = len(positions)
    positions_t = torch.from_numpy(positions).float()  # (T, 24, 3)

    # ── Isaac Gym ────────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt      = dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.num_threads     = args.num_threads
    sim_params.physx.use_gpu         = args.use_gpu
    sim_params.use_gpu_pipeline      = args.use_gpu_pipeline

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)

    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ── create one sphere asset per joint ────────────────────────────────────
    sphere_opts = gymapi.AssetOptions()
    sphere_opts.fix_base_link = True   # kinematic — not affected by physics
    joint_asset = gym.create_sphere(sim, 0.04, sphere_opts)

    env = gym.create_env(sim, gymapi.Vec3(-20, 20, 0), gymapi.Vec3(20, 20, 20), N_JOINTS)

    joint_handles = []
    for j in range(N_JOINTS):
        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(float(positions[0, j, 0]),
                             float(positions[0, j, 1]),
                             float(positions[0, j, 2]))
        pose.r = gymapi.Quat(0, 0, 0, 1)
        h = gym.create_actor(env, joint_asset, pose, f"joint_{j}", 0, 0, 0)
        joint_handles.append(h)

        # colour: left=blue, right=red, centre=grey
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        if j in {1,2,3,4, 14,15,16,17,18}:   # left
            color = gymapi.Vec3(0.29, 0.56, 0.85)
        elif j in {5,6,7,8, 19,20,21,22,23}: # right
            color = gymapi.Vec3(0.88, 0.32, 0.32)
        gym.set_rigid_body_color(env, h, 0, gymapi.MESH_VISUAL, color)

    gym.prepare_sim(sim)

    sim_device = (torch.device("cuda", args.compute_device_id)
                  if args.use_gpu_pipeline and torch.cuda.is_available()
                  else torch.device("cpu"))

    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states       = gymtorch.wrap_tensor(root_state_tensor)  # (N_JOINTS, 13)

    gym.refresh_actor_root_state_tensor(sim)

    # get sim-domain index of each joint actor
    joint_ids = torch.tensor(
        [gym.get_actor_index(env, h, gymapi.DOMAIN_SIM) for h in joint_handles],
        dtype=torch.int32, device=sim_device,
    )

    positions_t = positions_t.to(sim_device)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None,
                               gymapi.Vec3(3, -4, 2),
                               gymapi.Vec3(0, 0, 1))

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,  "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,      "restart")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,      "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "quit")

    frame_idx = 0
    paused    = False

    print(f"\nPlaying {T} frames, loops. Space=pause  R=restart  Q=quit\n")

    while not gym.query_viewer_has_closed(viewer):

        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0:
                if evt.action == "pause":
                    paused = not paused
                elif evt.action == "restart":
                    frame_idx = 0; paused = False
                elif evt.action == "quit":
                    gym.destroy_viewer(viewer)
                    gym.destroy_sim(sim)
                    return

        # set each sphere's position directly from predicted joint position
        root_states[joint_ids, 0:3] = positions_t[frame_idx]   # (24, 3)
        root_states[joint_ids, 3]   = 0.0  # quat x
        root_states[joint_ids, 4]   = 0.0  # quat y
        root_states[joint_ids, 5]   = 0.0  # quat z
        root_states[joint_ids, 6]   = 1.0  # quat w
        root_states[joint_ids, 7:]  = 0.0  # zero velocity

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(joint_ids), N_JOINTS,
        )

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
