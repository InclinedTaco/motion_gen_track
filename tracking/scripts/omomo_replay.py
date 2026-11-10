"""
omomo_replay.py

Replay a ground-truth OMOMO clip in Isaac Gym — joints as spheres plus
the manipulated object loaded from its URDF and positioned each frame.

Run from PHC directory:

  python scripts/omomo_replay.py \\
    --hdf5  data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \\
    --clip  sub12_largebox_044 \\
    --obj-dir assets/omomo_objects

Controls:
  Space   pause / resume
  R       restart
  Q/Esc   quit
"""

import os
import sys
import argparse

from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch
import h5py

sys.path.insert(0, os.getcwd())

N_JOINTS = 24

# object name → URDF stem mapping
OBJ_URDF = {
    "largebox":   "largebox_cleaned_simplified",
    "smallbox":   "smallbox_cleaned_simplified",
    "plasticbox": "plasticbox_cleaned_simplified",
    "trashcan":   "trashcan_cleaned_simplified",
    "suitcase":   "suitcase_cleaned_simplified",
}


def parse_args():
    custom = [
        {"name": "--hdf5",    "type": str, "required": True,
         "help": "OMOMO taskspace HDF5 file"},
        {"name": "--clip",    "type": str, "default": None,
         "help": "Clip name (default: first clip in HDF5)"},
        {"name": "--obj-dir", "type": str,
         "default": "assets/omomo_objects",
         "help": "Directory containing object URDFs"},
        {"name": "--fps",     "type": float, "default": 30.0},
    ]
    return gymutil.parse_arguments(
        description="OMOMO GT motion + object replay in Isaac Gym",
        custom_parameters=custom,
    )


def load_clip(hdf5_path, clip_name):
    with h5py.File(hdf5_path, "r") as f:
        clips = list(f["clips"].keys())
        if clip_name is None:
            clip_name = clips[0]
        if clip_name not in f["clips"]:
            raise ValueError(f"Clip '{clip_name}' not found. Available: {clips[:5]} …")

        grp = f[f"clips/{clip_name}"]
        joint_pos = grp["joint_positions"][:]   # (T, 24, 3) world-frame Z-up
        if "obj_com_pos_rel" not in grp:
            raise ValueError(f"Clip '{clip_name}' has no obj_com_pos_rel — run patch_hdf5_add_obj_pos.py first.")
        obj_rel  = grp["obj_com_pos_rel"][:]     # (T, 3) object CoM relative to pelvis
        obj_quat = grp["obj_rot_quat"][:]        # (T, 4) xyzw

    pelvis    = joint_pos[:, 0, :]               # (T, 3)
    obj_world = pelvis + obj_rel                 # (T, 3) object CoM world
    obj_world[:, 1] += 0.234                     # PHC applies -0.234m Y-shift to motion; correct object to match

    return joint_pos, obj_world, obj_quat, clip_name


def obj_name_from_clip(clip_name):
    """Extract object type from clip name, e.g. sub12_largebox_044 → largebox."""
    parts = clip_name.split("_")
    for p in parts:
        if p in OBJ_URDF:
            return p
    return None


def main():
    args = parse_args()
    fps  = args.fps
    dt   = 1.0 / fps

    joint_pos, obj_world, obj_quat, clip_name = load_clip(args.hdf5, args.clip)
    T = len(joint_pos)
    print(f"Loaded clip '{clip_name}'  ({T} frames @ {fps:.0f} fps)")

    obj_type = obj_name_from_clip(clip_name)
    urdf_stem = OBJ_URDF.get(obj_type)
    if urdf_stem is None:
        print(f"WARNING: no URDF mapping for object type '{obj_type}', skipping object.")
    else:
        urdf_path = os.path.join(args.obj_dir, f"{urdf_stem}.urdf")
        print(f"Object URDF: {urdf_path}")

    positions_t = torch.from_numpy(joint_pos.astype(np.float32))   # (T, 24, 3)
    obj_world_t = torch.from_numpy(obj_world.astype(np.float32))   # (T, 3)
    obj_quat_t  = torch.from_numpy(obj_quat.astype(np.float32))    # (T, 4) xyzw

    # ── Isaac Gym ────────────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt      = dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.physx.num_threads  = args.num_threads
    sim_params.physx.use_gpu      = args.use_gpu
    sim_params.use_gpu_pipeline   = args.use_gpu_pipeline

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)

    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ── joint sphere asset ───────────────────────────────────────────────────
    sphere_opts = gymapi.AssetOptions()
    sphere_opts.fix_base_link = True
    joint_asset = gym.create_sphere(sim, 0.04, sphere_opts)

    # ── object asset ─────────────────────────────────────────────────────────
    obj_asset = None
    if urdf_stem:
        obj_opts = gymapi.AssetOptions()
        obj_opts.fix_base_link = True
        obj_opts.disable_gravity = True
        obj_asset = gym.load_asset(sim, args.obj_dir, f"{urdf_stem}.urdf", obj_opts)

    n_actors = N_JOINTS + (1 if obj_asset else 0)
    env = gym.create_env(sim,
                         gymapi.Vec3(-20, 20, 0),
                         gymapi.Vec3(20, 20, 20),
                         n_actors)

    # spawn joint spheres
    joint_handles = []
    for j in range(N_JOINTS):
        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(*joint_pos[0, j].tolist())
        pose.r = gymapi.Quat(0, 0, 0, 1)
        h = gym.create_actor(env, joint_asset, pose, f"joint_{j}", 0, 0, 0)
        joint_handles.append(h)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        if j in {1, 2, 3, 4, 14, 15, 16, 17, 18}:
            color = gymapi.Vec3(0.29, 0.56, 0.85)
        elif j in {5, 6, 7, 8, 19, 20, 21, 22, 23}:
            color = gymapi.Vec3(0.88, 0.32, 0.32)
        gym.set_rigid_body_color(env, h, 0, gymapi.MESH_VISUAL, color)

    # spawn object
    obj_handle = None
    if obj_asset:
        pose   = gymapi.Transform()
        pose.p = gymapi.Vec3(*obj_world[0].tolist())
        q0 = obj_quat[0]  # xyzw
        pose.r = gymapi.Quat(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3]))
        obj_handle = gym.create_actor(env, obj_asset, pose, "object", 0, 0, 0)
        gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL,
                                 gymapi.Vec3(0.85, 0.65, 0.13))

    gym.prepare_sim(sim)

    sim_device = (torch.device("cuda", args.compute_device_id)
                  if args.use_gpu_pipeline and torch.cuda.is_available()
                  else torch.device("cpu"))

    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states       = gymtorch.wrap_tensor(root_state_tensor)

    gym.refresh_actor_root_state_tensor(sim)

    joint_ids = torch.tensor(
        [gym.get_actor_index(env, h, gymapi.DOMAIN_SIM) for h in joint_handles],
        dtype=torch.int32, device=sim_device,
    )
    if obj_handle is not None:
        obj_id = torch.tensor(
            [gym.get_actor_index(env, obj_handle, gymapi.DOMAIN_SIM)],
            dtype=torch.int32, device=sim_device,
        )

    positions_t = positions_t.to(sim_device)
    obj_world_t = obj_world_t.to(sim_device)
    obj_quat_t  = obj_quat_t.to(sim_device)

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
    print(f"\nPlaying {T} frames, loops.  Space=pause  R=restart  Q=quit\n")

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

        # update joint spheres
        root_states[joint_ids, 0:3] = positions_t[frame_idx]
        root_states[joint_ids, 3]   = 0.0
        root_states[joint_ids, 4]   = 0.0
        root_states[joint_ids, 5]   = 0.0
        root_states[joint_ids, 6]   = 1.0
        root_states[joint_ids, 7:]  = 0.0

        # update object position + rotation (xyzw → Isaac xyzw)
        if obj_handle is not None:
            root_states[obj_id, 0:3] = obj_world_t[frame_idx].unsqueeze(0)
            root_states[obj_id, 3:7] = obj_quat_t[frame_idx].unsqueeze(0)  # xyzw
            root_states[obj_id, 7:]  = 0.0
            all_ids = torch.cat([joint_ids, obj_id])
        else:
            all_ids = joint_ids

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(all_ids), len(all_ids),
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
