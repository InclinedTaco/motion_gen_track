"""
Kinematic playback of a single clip from the task-space HDF5 in Isaac Gym.

Draws:
  • green sphere  – left  EEF  (task_space cols 3-5)
  • red   sphere  – right EEF  (task_space cols 6-8)
  • blue  sphere  – pelvis     (joint_positions[:,0,:])

The humanoid pose comes from MotionLibSMPL (same PKL used to build the HDF5),
so the skeleton deforms correctly every frame.

Usage:
    python scripts/vis/play_hdf5_clip.py \\
        --clip "ACCAD_Female1Walking_c3d_B1 - stand to walk_poses"

Controls:
    SPACE  – pause / resume
    LEFT   – step back one frame
    RIGHT  – step forward one frame
    R      – restart clip
    Q / Esc – quit
"""

import os
import sys

sys.path.append(os.getcwd())

# isaacgym MUST be imported before torch
from isaacgym import gymapi, gymutil, gymtorch

import numpy as np
import torch
import joblib
import h5py
from easydict import EasyDict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.flags import flags

flags.test = True
flags.im_eval = True

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_HDF5 = "data/amass/hdf5/accad_research_filtered_taskspace.h5"
DEFAULT_PKL  = "data/amass/pkls/final_humanoid/accad_isaac_upright_slim_research_filtered.pkl"
HUMANOID_XML = "final_humanoid.xml"   # asset_root = "./"
MARKER_URDF  = "traj_marker.urdf"
MARKER_ROOT  = "phc/data/assets/urdf/"
NUM_MARKERS  = 3   # left_eef, right_eef, pelvis – one actor each

# task_space columns (T, 9)
#   0-1  root_vel_xy
#   2    pelvis_height
#   3-5  left_eef_xyz
#   6-8  right_eef_xyz
LEFT_EEF_COLS  = slice(3, 6)
RIGHT_EEF_COLS = slice(6, 9)
PELVIS_H_COL   = 2

# joint_positions: Pelvis is index 0 (world-frame XYZ)
PELVIS_JOINT_IDX = 0

# ── colours ───────────────────────────────────────────────────────────────────
COLORS = [
    gymapi.Vec3(0.0, 0.85, 0.0),   # 0 = left  EEF  → green
    gymapi.Vec3(0.85, 0.0, 0.0),   # 1 = right EEF  → red
    gymapi.Vec3(0.0, 0.4,  0.9),   # 2 = pelvis     → blue
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5_clip(hdf5_path: str, clip_name: str):
    with h5py.File(hdf5_path, "r") as f:
        if "clips" not in f or clip_name not in f["clips"]:
            avail = sorted(f["clips"].keys()) if "clips" in f else []
            raise KeyError(
                f"Clip '{clip_name}' not found in {hdf5_path}.\n"
                f"Available ({len(avail)}): {avail[:10]} …"
            )
        grp = f[f"clips/{clip_name}"]
        task_space      = grp["task_space"][:]       # (T, 9)  float32
        joint_positions = grp["joint_positions"][:]  # (T, 24, 3)  float32
        fps             = float(grp.attrs.get("fps", 30.0))
    return task_space, joint_positions, fps


def load_pkl_clip(pkl_path: str, clip_name: str):
    """Return the clip dict for betas/gender. Index resolved later via motion_lib._motion_data_keys."""
    data = joblib.load(pkl_path)
    if clip_name not in data:
        keys = list(data.keys())
        raise KeyError(
            f"Clip '{clip_name}' not found in PKL.\n"
            f"First 10: {keys[:10]}"
        )
    return data[clip_name]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = gymutil.parse_arguments(
        description="Kinematic HDF5 clip viewer with EEF spheres",
        custom_parameters=[
            {"name": "--clip",  "type": str, "required": True,
             "help": "Clip name (must exist in both HDF5 and PKL)"},
            {"name": "--hdf5",  "type": str, "default": DEFAULT_HDF5},
            {"name": "--pkl",   "type": str, "default": DEFAULT_PKL},
            {"name": "--speed", "type": float, "default": 1.0,
             "help": "Playback speed multiplier"},
        ],
    )

    clip_name = args.clip
    hdf5_path = args.hdf5
    pkl_path  = args.pkl
    speed     = args.speed

    # ── load HDF5 data ───────────────────────────────────────────────────────
    print(f"Loading HDF5 clip: {clip_name}")
    task_space, joint_positions, fps = load_hdf5_clip(hdf5_path, clip_name)
    T   = task_space.shape[0]
    dt  = 1.0 / fps
    print(f"  frames={T}  fps={fps}  duration={T*dt:.2f}s")

    # ── locate clip in PKL ───────────────────────────────────────────────────
    print(f"Locating clip in PKL …")
    pkl_clip    = load_pkl_clip(pkl_path, clip_name)
    clip_betas  = pkl_clip["beta"]
    clip_gender = pkl_clip["gender"]
    if isinstance(clip_gender, bytes):
        clip_gender = clip_gender.decode()
    gender_float = {"neutral": 0.0, "male": 0.0, "female": 1.0}.get(
        clip_gender.lower(), 0.0
    )
    num_betas   = min(len(clip_betas), 16)
    gender_beta = torch.zeros(17)
    gender_beta[0]             = gender_float
    gender_beta[1:num_betas+1] = torch.tensor(clip_betas[:num_betas], dtype=torch.float32)
    print(f"  gender={clip_gender}  betas[:4]={clip_betas[:4].round(3)}")

    # ── Isaac Gym init ───────────────────────────────────────────────────────
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt               = dt
    sim_params.up_axis          = gymapi.UP_AXIS_Z
    sim_params.gravity          = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type            = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads            = args.num_threads
    sim_params.physx.use_gpu                = args.use_gpu
    sim_params.use_gpu_pipeline             = args.use_gpu_pipeline

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        sys.exit("ERROR: gym.create_sim() failed")

    # ground plane
    plane_params        = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # ── load assets ──────────────────────────────────────────────────────────
    humanoid_opts = gymapi.AssetOptions()
    humanoid_opts.use_mesh_materials = True
    humanoid_opts.angular_damping    = 0.01
    humanoid_opts.linear_damping     = 0.01
    humanoid_asset = gym.load_asset(sim, "./", HUMANOID_XML, humanoid_opts)
    if humanoid_asset is None:
        sys.exit(f"ERROR: could not load {HUMANOID_XML}")
    num_dofs   = gym.get_asset_dof_count(humanoid_asset)
    num_bodies = gym.get_asset_rigid_body_count(humanoid_asset)
    print(f"Humanoid: {num_bodies} bodies, {num_dofs} DOFs")

    marker_opts = gymapi.AssetOptions()
    marker_opts.angular_damping         = 0.0
    marker_opts.linear_damping          = 0.0
    marker_opts.max_angular_velocity    = 0.0
    marker_opts.density                 = 0
    marker_opts.fix_base_link           = True
    marker_opts.default_dof_drive_mode  = gymapi.DOF_MODE_NONE
    marker_asset = gym.load_asset(sim, MARKER_ROOT, MARKER_URDF, marker_opts)
    if marker_asset is None:
        sys.exit(f"ERROR: could not load {MARKER_URDF} from {MARKER_ROOT}")

    # ── skeleton tree (for MotionLibSMPL) ────────────────────────────────────
    sk_tree = SkeletonTree.from_mjcf(HUMANOID_XML)

    # ── create env ───────────────────────────────────────────────────────────
    spacing   = 5.0
    env_lower = gymapi.Vec3(-spacing,  spacing, 0)
    env_upper = gymapi.Vec3( spacing,  spacing, spacing)

    env = gym.create_env(sim, env_lower, env_upper, 1)

    # humanoid actor
    start_pose   = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0, 0, 0.93)
    start_pose.r = gymapi.Quat(0, 0, 0, 1)
    humanoid_handle = gym.create_actor(
        env, humanoid_asset, start_pose, "humanoid", 0, 1, 0
    )

    # sphere marker actors (one per EEF + pelvis)
    marker_handles = []
    default_pose   = gymapi.Transform()
    default_pose.p = gymapi.Vec3(0, 0, 1000)   # hide until first frame
    default_pose.r = gymapi.Quat(0, 0, 0, 1)

    label_order = ["left_eef", "right_eef", "pelvis"]
    for mi, label in enumerate(label_order):
        mh = gym.create_actor(
            env, marker_asset, default_pose, label, num_bodies + 10, 1, 0
        )
        gym.set_rigid_body_color(
            env, mh, 0, gymapi.MESH_VISUAL, COLORS[mi]
        )
        marker_handles.append(mh)

    # ── motion library ───────────────────────────────────────────────────────
    print("Loading motion library …")
    device = (
        torch.device("cuda", args.compute_device_id)
        if args.use_gpu_pipeline and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # im_eval=False keeps original PKL insertion order so clip_idx is stable
    motion_lib_cfg = EasyDict({
        "motion_file":        pkl_path,
        "device":             device,
        "fix_height":         FixHeightMode.no_fix,
        "min_length":         -1,
        "max_length":         -1,
        "im_eval":            False,
        "multi_thread":       False,
        "smpl_type":          "smpl",
        "randomrize_heading": False,
        "step_dt":            dt,
    })
    motion_lib = MotionLibSMPL(motion_lib_cfg)

    # Resolve index from the motion lib's own key list (not the raw PKL order)
    # so we're immune to any reordering done inside load_data.
    lib_keys  = list(motion_lib._motion_data_keys)
    if clip_name not in lib_keys:
        sys.exit(f"ERROR: '{clip_name}' not found in motion lib keys")
    clip_idx = lib_keys.index(clip_name)
    print(f"  motion_lib index = {clip_idx}  key = {lib_keys[clip_idx]}")

    motion_lib.load_motions(
        skeleton_trees = [sk_tree],
        gender_betas   = [gender_beta],
        limb_weights   = [np.zeros(10)],
        random_sample  = False,
        start_idx      = clip_idx,
    )
    motion_len = motion_lib.get_motion_length(0).item()
    print(f"Motion loaded: length={motion_len:.2f}s")

    # ── prepare sim + acquire tensors ────────────────────────────────────────
    gym.prepare_sim(sim)

    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states       = gymtorch.wrap_tensor(root_state_tensor)   # (num_actors, 13)

    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_states       = gymtorch.wrap_tensor(dof_state_tensor)     # (num_dofs, 2)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    # actor id tensors for indexed set calls
    humanoid_id    = gym.get_actor_index(env, humanoid_handle, gymapi.DOMAIN_SIM)
    marker_ids_np  = np.array([
        gym.get_actor_index(env, mh, gymapi.DOMAIN_SIM)
        for mh in marker_handles
    ], dtype=np.int32)
    humanoid_id_t = torch.tensor([humanoid_id], dtype=torch.int32, device=device)
    marker_ids_t  = torch.tensor(marker_ids_np, dtype=torch.int32, device=device)

    # ── viewer ───────────────────────────────────────────────────────────────
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        sys.exit("ERROR: gym.create_viewer() failed")

    cam_pos    = gymapi.Vec3(3, -4, 2)
    cam_target = gymapi.Vec3(0,  0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,     "pause")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,      "prev")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT,     "next")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,         "restart")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q,         "quit")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE,    "quit")

    # ── playback state ───────────────────────────────────────────────────────
    time_step = 0.0
    paused    = False
    frame_dt  = dt / speed

    print("\nControls: SPACE=pause  ←/→=step  R=restart  Q/Esc=quit\n")
    print(f"Playing: {clip_name}")

    # ── main loop ────────────────────────────────────────────────────────────
    while not gym.query_viewer_has_closed(viewer):

        # ── keyboard events ──────────────────────────────────────────────────
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0:
                if evt.action == "pause":
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif evt.action == "prev" and paused:
                    time_step = max(0.0, time_step - dt)
                elif evt.action == "next" and paused:
                    time_step = min(motion_len - dt, time_step + dt)
                elif evt.action == "restart":
                    time_step = 0.0
                    paused    = False
                elif evt.action == "quit":
                    gym.destroy_viewer(viewer)
                    gym.destroy_sim(sim)
                    return

        # ── get motion state from motion lib ─────────────────────────────────
        motion_time = time_step % motion_len
        frame_idx   = int(round(motion_time * fps)) % T

        motion_res = motion_lib.get_motion_state(
            torch.tensor([0], device=device),
            torch.tensor([motion_time], device=device),
        )
        root_pos     = motion_res["root_pos"]      # (1, 3)
        root_rot     = motion_res["root_rot"]      # (1, 4)  xyzw
        root_vel     = motion_res["root_vel"]      # (1, 3)
        root_ang_vel = motion_res["root_ang_vel"]  # (1, 3)
        dof_pos      = motion_res["dof_pos"]       # (1, 69)
        dof_vel      = motion_res["dof_vel"]       # (1, 69)

        # ── set humanoid root state ───────────────────────────────────────────
        root_states[humanoid_id, 0:3]  = root_pos[0]
        root_states[humanoid_id, 3:7]  = root_rot[0]
        root_states[humanoid_id, 7:10] = root_vel[0]
        root_states[humanoid_id, 10:13]= root_ang_vel[0]

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(humanoid_id_t),
            1,
        )

        # ── set DOF state ─────────────────────────────────────────────────────
        dof_states_full = torch.zeros(num_dofs, 2, device=device)
        dof_states_full[:, 0] = dof_pos[0]
        dof_states_full[:, 1] = dof_vel[0]
        dof_states.copy_(dof_states_full)

        gym.set_dof_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(dof_states),
            gymtorch.unwrap_tensor(humanoid_id_t),
            1,
        )

        # ── set sphere marker positions from task_space + joint_positions ─────
        ts_row  = task_space[frame_idx]       # (9,)
        jp_row  = joint_positions[frame_idx]  # (24, 3)

        left_eef_pos  = torch.tensor(ts_row[LEFT_EEF_COLS],    dtype=torch.float32, device=device)
        right_eef_pos = torch.tensor(ts_row[RIGHT_EEF_COLS],   dtype=torch.float32, device=device)
        pelvis_pos    = torch.tensor(jp_row[PELVIS_JOINT_IDX], dtype=torch.float32, device=device)

        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        zeros3        = torch.zeros(3, device=device)

        for mi, pos in enumerate([left_eef_pos, right_eef_pos, pelvis_pos]):
            actor_idx = marker_ids_np[mi]
            root_states[actor_idx, 0:3]   = pos
            root_states[actor_idx, 3:7]   = identity_quat
            root_states[actor_idx, 7:10]  = zeros3
            root_states[actor_idx, 10:13] = zeros3

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(marker_ids_t),
            NUM_MARKERS,
        )

        # ── step physics (just FK; no real dynamics needed) ───────────────────
        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.fetch_results(sim, True)

        # ── render ────────────────────────────────────────────────────────────
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # ── advance time ──────────────────────────────────────────────────────
        if not paused:
            time_step += frame_dt

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
