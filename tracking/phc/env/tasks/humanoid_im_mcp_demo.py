
import os
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im_mcp as humanoid_im_mcp
import phc.env.tasks.humanoid_im as humanoid_im
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL 

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from collections import defaultdict
from scipy.spatial.transform import Rotation as sRot
import phc.utils.pytorch3d_transforms as ptr
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState

import aiohttp, cv2, asyncio, json
import requests
from collections import deque
import scipy.ndimage.filters as filters
from smpl_sim.utils.transform_utils import quat_correct_two_batch
import subprocess
import time
from pathlib import Path

SERVER = os.environ.get("PHC_FLOW_SERVER", "127.0.0.1")
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]


class HumanoidImMCPDemo(humanoid_im_mcp.HumanoidImMCP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        ## Debugging
        # self.res_data = joblib.load("/home/zhengyiluo5/dev/meta/HybrIK/ik_res.pkl")
        # self.rot_mat_ref = torch.from_numpy(sRot.from_rotvec(np.array(self.res_data['pose_aa']).reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)).float().to(self.device)
        ## Debugging
        
        self.local_translation_batch = self.skeleton_trees[0].local_translation[None,]
        self.parent_indices = self.skeleton_trees[0].parent_indices
        self.pose_mat = torch.eye(3).repeat(self.num_envs, 24, 1, 1).to(self.device)
        self.trans = torch.zeros(self.num_envs, 3).to(self.device)
        self._flow_command_heading = float(cfg["env"].get("flow_command_heading", cfg["env"].get("robot_start_heading", 0.0)))
        self._align_robot_start_to_flow_heading = bool(
            cfg["env"].get(
                "align_robot_start_to_flow_heading",
                cfg["env"].get("flow_command_heading", None) is not None or cfg["env"].get("robot_start_heading", None) is not None,
            )
        )

        self.prev_ref_body_pos = torch.zeros(self.num_envs, 24, 3).to(self.device)
        self.prev_ref_body_rot = torch.zeros(self.num_envs, 24, 4).to(self.device)
        self._flow_ref_valid = False
        self._flow_episode_idx = 0
        self._flow_ref_mode = cfg["env"].get("flow_ref_mode", "server")
        self._flow_model = None
        if self._flow_ref_mode == "inprocess":
            self._init_inprocess_flow(cfg)

        self.zero_trans = torch.zeros([self.num_envs, 3])
        self.s_dt = 1 / 30

        self.to_isaac_mat = torch.from_numpy(sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()).float()
        self.to_global = torch.from_numpy(sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()).float()

        self.root_pos_acc = deque(maxlen=30)
        self.body_rot_acc = deque(maxlen=30)
        self.body_pos_acc = deque(maxlen=30)

        flags.no_collision_check = True
        flags.show_traj = True
        self.close_distance = 0.5
        self.mean_limb_lengths = np.array([0.1061, 0.3624, 0.4015, 0.1384, 0.1132], dtype=np.float32)[None, :]

        self.flow_debug_log_path = os.environ.get(
            "PHC_FLOW_DEBUG_LOG",
            os.path.join("output", "debug", "flow_ref_robot_world.csv"),
        )
        self.flow_debug_log_every = int(os.environ.get("PHC_FLOW_DEBUG_LOG_EVERY", "0"))
        self._flow_debug_log_file = None
        self._flow_debug_log_count = 0
        if self.flow_debug_log_every > 0:
            print(f"Flow/robot world-position log: {self.flow_debug_log_path}")

        self._body_log_path = os.environ.get(
            "PHC_BODY_LOG",
            os.path.join("output", "debug", "flow_body_log.npz"),
        )
        self._body_log_enabled = os.environ.get("PHC_BODY_LOG_ENABLE", "0") == "1"
        self._body_log_ref = []
        self._body_log_robot = []
        self._body_log_phase = []
        self._body_log_progress = []
        self._body_log_buffer_idx = []
        self._body_log_chunk_start = []
        self._body_log_ref_source = []
        self._body_log_buffer_valid = []
        self._body_log_reset_count = []
        self._body_log_save_every = 300
        if self._body_log_enabled:
            print(f"Flow/robot full-body log: {self._body_log_path}")

    def _init_inprocess_flow(self, cfg):
        import json
        import importlib.util
        import sys

        ckpt = cfg["env"].get("flow_ckpt")
        stats_path = cfg["env"].get("flow_stats")
        if ckpt is None or stats_path is None:
            raise ValueError("env.flow_ckpt and env.flow_stats are required for env.flow_ref_mode=inprocess")

        flow_server_path = Path(__file__).resolve().parents[3] / "scripts" / "flow_server_x15.py"
        flow_scripts_dir = str(flow_server_path.parent)
        if flow_scripts_dir not in sys.path:
            sys.path.insert(0, flow_scripts_dir)
        spec = importlib.util.spec_from_file_location("phc_flow_server_x15", flow_server_path)
        flow_x15 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_x15)

        self._flow_x15 = flow_x15
        with open(Path(stats_path).expanduser()) as f:
            self._flow_stats = json.load(f)

        self._flow_device = torch.device(self.device)
        flow_seed = int(cfg["env"].get("flow_seed", cfg.get("seed", 0)))
        if flow_seed < 0:
            flow_seed = 0
        # Match offline generation semantics: horizon_flow_to_pkl.py seeds
        # before load_model(), whose module construction consumes Torch RNG.
        torch.manual_seed(flow_seed)
        self._flow_model = flow_x15.load_model(ckpt, self._flow_stats, self._flow_device)
        self._flow_torch_generator = torch.Generator(device=self._flow_device)
        if self._flow_device.type == "cuda":
            self._flow_torch_generator.set_state(torch.cuda.get_rng_state(self._flow_device))
        else:
            self._flow_torch_generator.set_state(torch.get_rng_state())
        self._flow_ode_steps = int(cfg["env"].get("flow_ode_steps", 10))
        self._flow_stride = int(cfg["env"].get("flow_stride", max(1, int(self._flow_stats["horizon"]) // 2)))
        self._flow_target_speed = float(cfg["env"].get("flow_target_speed", 1.3))
        self._flow_stand_steps = int(cfg["env"].get("flow_stand_steps", 30))
        self._flow_stand_steps_min = int(cfg["env"].get("flow_stand_steps_min", self._flow_stand_steps))
        self._flow_stand_steps_max = int(cfg["env"].get("flow_stand_steps_max", self._flow_stand_steps))
        self._flow_stand_steps_max = max(self._flow_stand_steps_min, self._flow_stand_steps_max)
        self._flow_hold_stand_ref = bool(cfg["env"].get("flow_hold_stand_ref", False))
        self._flow_seed_default_history = bool(cfg["env"].get("flow_seed_default_history", False))
        self._flow_pure_ar = bool(cfg["env"].get("flow_pure_ar", False))
        self._flow_reset_history_on_env_reset = bool(cfg["env"].get("flow_reset_history_on_env_reset", False))
        self._flow_enable_tracking_reset = bool(cfg["env"].get("flow_enable_tracking_reset", False))
        self._flow_tracking_reset_disable_collision = bool(cfg["env"].get("flow_tracking_reset_disable_collision", False))
        self._flow_random_heading = bool(cfg["env"].get("flow_random_heading", False))
        self._flow_heading_min = float(cfg["env"].get("flow_heading_min", -np.pi))
        self._flow_heading_max = float(cfg["env"].get("flow_heading_max", np.pi))
        self._flow_heading_change_steps_min = int(cfg["env"].get("flow_heading_change_steps_min", 120))
        self._flow_heading_change_steps_max = int(cfg["env"].get("flow_heading_change_steps_max", 240))
        self._flow_heading_change_steps_max = max(self._flow_heading_change_steps_min, self._flow_heading_change_steps_max)
        self._flow_heading_turn_rate = float(cfg["env"].get("flow_heading_turn_rate", np.deg2rad(1.0)))
        self._flow_replay_joints = None
        self._flow_replay_reset_on_loop = bool(cfg["env"].get("flow_replay_reset_on_loop", True))
        replay_file = cfg["env"].get("flow_replay_file")
        if replay_file:
            replay_path = Path(replay_file).expanduser()
            if replay_path.suffix == ".npz":
                replay = np.load(replay_path)
                self._flow_replay_joints = replay["joints"].astype(np.float32)
            else:
                replay = joblib.load(replay_path)
                first = next(iter(replay.values()))
                if "raw_global_translation" in first:
                    self._flow_replay_joints = np.asarray(first["raw_global_translation"], dtype=np.float32)
                else:
                    raise ValueError(f"{replay_path} has no raw_global_translation")
            if self._flow_replay_joints.ndim != 3 or self._flow_replay_joints.shape[1:] != (24, 3):
                raise ValueError(f"flow_replay_file joints must have shape (T, 24, 3), got {self._flow_replay_joints.shape}")
            print(f"Flow replay refs enabled: {replay_path} frames={len(self._flow_replay_joints)}")
        self._flow_curriculum_warmup_epochs = int(cfg["env"].get("flow_curriculum_warmup_epochs", 100))
        self._flow_curriculum_ramp_end_epochs = int(cfg["env"].get("flow_curriculum_ramp_end_epochs", 700))
        self._flow_feedback_source = cfg["env"].get("flow_feedback_source", "curriculum")
        self._flow_history = [[] for _ in range(self.num_envs)]
        self._flow_prediction_history = [[] for _ in range(self.num_envs)]
        self._flow_ref_buffer = torch.zeros(
            self.num_envs,
            self._flow_stride,
            24,
            3,
            device=self.device,
        )
        self._flow_ref_buffer_idx = np.full((self.num_envs,), self._flow_stride, dtype=np.int64)
        self._flow_ref_chunk_start = None
        self._flow_ref_buffer_valid = np.zeros((self.num_envs,), dtype=bool)
        self._flow_ref_source = "init"
        self._flow_reset_count = 0
        self._flow_cached_ref = None
        self._flow_cached_phase = None
        self._flow_cached_progress = None
        self._flow_rng = np.random.default_rng(flow_seed)
        self._flow_stand_steps_env = np.full((self.num_envs,), self._flow_stand_steps, dtype=np.int64)
        self._flow_command_heading_env = np.full((self.num_envs,), self._flow_command_heading, dtype=np.float32)
        self._flow_heading_target_env = self._flow_command_heading_env.copy()
        self._flow_next_heading_change_step = np.full((self.num_envs,), self._flow_heading_change_steps_max, dtype=np.int64)
        self._reset_flow_command_schedule(range(self.num_envs))
        if self._flow_seed_default_history or self._flow_pure_ar:
            self._seed_flow_default_history(range(self.num_envs))
            print("Flow default-state history seed enabled")
        if self._flow_pure_ar:
            print("Flow pure autoregressive mode enabled")
        print(
            f"In-process flow refs enabled: envs={self.num_envs} stride={self._flow_stride} "
            f"ode_steps={self._flow_ode_steps} feedback={self._flow_feedback_source}"
        )

    def _seed_flow_default_history(self, env_ids):
        default_state = np.asarray(self._flow_stats["default_state"], dtype=np.float32)
        min_history = 1 + int(self._flow_stats.get("past_stride", 1)) * (int(self._flow_stats["past_frames"]) - 1)
        init_z = float(default_state[2])
        init_joint_rel = default_state[7:].reshape(23, 3)
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[0] = [0.0, 0.0, init_z]
        joints[1:] = init_joint_rel + joints[0][None]
        default_world = {
            "root_xy": np.zeros(2, dtype=np.float32),
            "root_z": np.float32(init_z),
            "heading": np.float32(0.0),
            "pitch": np.float32(0.0),
            "roll": np.float32(0.0),
            "joint_rel": init_joint_rel.reshape(-1).astype(np.float32),
            "joints": joints,
        }
        for env_id in env_ids:
            env_id = int(env_id)
            self._flow_history[env_id] = [default_world.copy() for _ in range(min_history)]
            self._flow_prediction_history[env_id] = [default_world.copy() for _ in range(min_history)]

    def _sample_flow_heading(self):
        return float(self._flow_rng.uniform(self._flow_heading_min, self._flow_heading_max))

    def _reset_flow_command_schedule(self, env_ids):
        flow_env_ids = self._flow_env_ids_to_list(env_ids)
        for env_id in flow_env_ids:
            if self._flow_stand_steps_max > self._flow_stand_steps_min:
                self._flow_stand_steps_env[env_id] = int(
                    self._flow_rng.integers(self._flow_stand_steps_min, self._flow_stand_steps_max + 1)
                )
            else:
                self._flow_stand_steps_env[env_id] = self._flow_stand_steps_min

            if self._flow_random_heading:
                heading = self._sample_flow_heading()
            else:
                heading = self._flow_command_heading
            self._flow_command_heading_env[env_id] = np.float32(heading)
            self._flow_heading_target_env[env_id] = np.float32(heading)
            self._flow_next_heading_change_step[env_id] = int(
                self._flow_rng.integers(self._flow_heading_change_steps_min, self._flow_heading_change_steps_max + 1)
            )

    def _update_flow_heading_commands(self):
        if not self._flow_random_heading:
            self._flow_command_heading_env[:] = np.float32(self._flow_command_heading)
            self._flow_heading_target_env[:] = np.float32(self._flow_command_heading)
            return

        progress = self.progress_buf.detach().cpu().numpy().astype(np.int64)
        for env_id, step in enumerate(progress):
            if step >= self._flow_next_heading_change_step[env_id]:
                self._flow_heading_target_env[env_id] = np.float32(self._sample_flow_heading())
                interval = int(self._flow_rng.integers(self._flow_heading_change_steps_min, self._flow_heading_change_steps_max + 1))
                self._flow_next_heading_change_step[env_id] = int(step + interval)

        delta = self._flow_heading_target_env - self._flow_command_heading_env
        delta = np.arctan2(np.sin(delta), np.cos(delta)).astype(np.float32)
        step_delta = np.clip(delta, -self._flow_heading_turn_rate, self._flow_heading_turn_rate)
        self._flow_command_heading_env = (
            self._flow_command_heading_env + step_delta
        ).astype(np.float32)

    def _flow_loco_phase(self, env_ids=None):
        if env_ids is None:
            progress = self.progress_buf.detach().cpu().numpy().astype(np.int64)
            return (progress >= self._flow_stand_steps_env).astype(np.float32)
        flow_env_ids = self._flow_env_ids_to_list(env_ids)
        progress = self.progress_buf[flow_env_ids].detach().cpu().numpy().astype(np.int64)
        stand_steps = self._flow_stand_steps_env[flow_env_ids]
        return (progress >= stand_steps).astype(np.float32)

    def _reset_flow_env_streams(self, env_ids):
        flow_env_ids = self._flow_env_ids_to_list(env_ids)
        self._reset_flow_command_schedule(flow_env_ids)
        for env_id in flow_env_ids:
            self._flow_history[int(env_id)].clear()
            self._flow_prediction_history[int(env_id)].clear()
        self._seed_flow_default_history(flow_env_ids)
        self._flow_ref_buffer_idx[flow_env_ids] = self._flow_stride
        self._flow_ref_chunk_start = None
        self._flow_ref_buffer_valid[flow_env_ids] = False
        self._flow_cached_ref = None
        self._flow_cached_phase = None
        self._flow_cached_progress = None

    def _flow_alpha(self):
        epoch = float(self._flow_episode_idx)
        if epoch < self._flow_curriculum_warmup_epochs:
            return 0.0
        if epoch >= self._flow_curriculum_ramp_end_epochs:
            return 1.0
        denom = max(1, self._flow_curriculum_ramp_end_epochs - self._flow_curriculum_warmup_epochs)
        return float((epoch - self._flow_curriculum_warmup_epochs) / denom)

    def _body_to_flow_states(self, body_pos, body_rot):
        body_pos_np = body_pos.detach().cpu().numpy()
        body_rot_np = body_rot.detach().cpu().numpy()
        out = []
        for env_id in range(body_pos_np.shape[0]):
            root = body_pos_np[env_id, 0].astype(np.float32)
            heading, pitch, roll = sRot.from_quat(body_rot_np[env_id, 0]).as_euler("ZYX")
            joint_rel = (body_pos_np[env_id, 1:] - body_pos_np[env_id, 0]).astype(np.float32)
            out.append({
                "root_xy": root[:2].astype(np.float32),
                "root_z": np.float32(root[2]),
                "heading": np.float32(heading),
                "pitch": np.float32(pitch),
                "roll": np.float32(roll),
                "joint_rel": joint_rel.reshape(-1),
                "joints": body_pos_np[env_id].astype(np.float32),
            })
        return out

    def _flow_cond_from_history(self, hist_tail, phase, desired_heading):
        K = int(self._flow_stats["past_frames"])
        past_stride = int(self._flow_stats.get("past_stride", 1))
        anchor = hist_tail[-1]
        anchor_xy = anchor["root_xy"]
        anchor_z = float(anchor["root_z"])
        anchor_heading = float(anchor["heading"])
        anchor_pitch = float(anchor["pitch"])
        anchor_roll = float(anchor["roll"])

        past = [
            self._flow_x15.local_state_from_world(
                hist_tail[-1 - past_stride * (K - 1 - i)],
                anchor_xy, anchor_heading, anchor_pitch, anchor_roll,
            )
            for i in range(K)
        ]

        if phase[1] > 0.5:
            h_rel = float(desired_heading) - anchor_heading
            h_rel = float(np.arctan2(np.sin(h_rel), np.cos(h_rel)))
            heading_vec = np.array([np.cos(h_rel), np.sin(h_rel)], dtype=np.float32)
        else:
            heading_vec = np.zeros(2, dtype=np.float32)

        active_goal = np.zeros(3, dtype=np.float32)
        task_parts = [phase, heading_vec, active_goal]
        extras = []
        if self._flow_stats.get("use_target_speed", False):
            speed = self._flow_target_speed if phase[1] > 0.5 else 0.0
            extras = [np.array([speed], dtype=np.float32)]
        return np.concatenate(past + task_parts + extras).astype(np.float32)

    def _flow_env_ids_to_list(self, env_ids):
        if env_ids is None:
            return list(range(self.num_envs))
        if torch.is_tensor(env_ids):
            return [int(x) for x in env_ids.detach().cpu().numpy().tolist()]
        return [int(x) for x in env_ids]

    def _next_inprocess_flow_ref(self, body_pos, body_rot, env_ids=None):
        flow_env_ids = self._flow_env_ids_to_list(env_ids)
        full_env_batch = len(flow_env_ids) == self.num_envs
        if full_env_batch:
            progress_key = tuple(int(x) for x in self.progress_buf.detach().cpu().numpy().tolist())
            if self._flow_cached_progress == progress_key and self._flow_cached_ref is not None:
                return self._flow_cached_ref, self._flow_cached_phase.copy()
            self._update_flow_heading_commands()
        states = None if self._flow_pure_ar else self._body_to_flow_states(body_pos, body_rot)
        state_count = self.num_envs if states is None else len(states)
        if len(flow_env_ids) != state_count:
            raise ValueError("env_ids length must match in-process flow state batch")
        K = int(self._flow_stats["past_frames"])
        past_stride = int(self._flow_stats.get("past_stride", 1))
        min_history = 1 + past_stride * (K - 1)

        if not self._flow_pure_ar:
            for local_id, env_id in enumerate(flow_env_ids):
                hist = self._flow_history[env_id]
                hist.append(states[local_id])
                if len(hist) > 128:
                    del hist[:-128]

        if len(flow_env_ids) != self.num_envs:
            phases = self._flow_loco_phase(flow_env_ids)
            return body_pos.detach().clone(), phases

        if self._flow_hold_stand_ref and np.all(self._flow_loco_phase() < 0.5):
            self._flow_ref_buffer_idx[:] = self._flow_stride
            self._flow_ref_buffer_valid[:] = False
            self._flow_ref_source = "hold_stand"
            phases = np.zeros((self.num_envs,), dtype=np.float32)
            ref = body_pos.detach().clone()
            self._flow_cached_progress = progress_key
            self._flow_cached_ref = ref
            self._flow_cached_phase = phases.copy()
            return ref, phases

        phases = self._flow_loco_phase()
        ref = torch.zeros((self.num_envs, 24, 3), dtype=torch.float, device=self.device)
        need_plan = np.logical_not(
            np.logical_and(self._flow_ref_buffer_valid, self._flow_ref_buffer_idx < self._flow_stride)
        )
        buffered_env_ids = np.nonzero(~need_plan)[0]
        if len(buffered_env_ids) > 0:
            frame_ids = self._flow_ref_buffer_idx[buffered_env_ids]
            ref[buffered_env_ids] = self._flow_ref_buffer[
                torch.as_tensor(buffered_env_ids, dtype=torch.long, device=self.device),
                torch.as_tensor(frame_ids, dtype=torch.long, device=self.device),
            ]
            self._flow_ref_buffer_idx[buffered_env_ids] += 1
        plan_env_ids = np.nonzero(need_plan)[0].tolist()
        if len(plan_env_ids) == 0:
            self._flow_ref_source = "buffer"
            self._flow_cached_progress = progress_key
            self._flow_cached_ref = ref
            self._flow_cached_phase = phases.copy()
            return ref, phases

        if self._flow_replay_joints is not None:
            buffer_np = np.zeros((len(plan_env_ids), self._flow_stride, 24, 3), dtype=np.float32)
            for local_i, env_id in enumerate(plan_env_ids):
                start = int(self.progress_buf[env_id].item())
                frame_ids = np.arange(start, start + self._flow_stride) % len(self._flow_replay_joints)
                replay = self._flow_replay_joints[frame_ids]
                server_j3d = self._flow_x15.mujoco_zup_to_server(replay)
                buffer_np[local_i] = server_j3d[:, smpl_2_mujoco]
            planned_ref = torch.from_numpy(buffer_np).float().matmul(self.to_isaac_mat.T).to(self.device)
            plan_env_ids_t = torch.as_tensor(plan_env_ids, dtype=torch.long, device=self.device)
            self._flow_ref_buffer[plan_env_ids_t] = planned_ref
            ref[plan_env_ids_t] = planned_ref[:, 0]
            self._flow_ref_buffer_idx[plan_env_ids] = 1
            self._flow_ref_buffer_valid[plan_env_ids] = True
            self._flow_ref_source = "replay"
            self._flow_cached_progress = progress_key
            self._flow_cached_ref = ref
            self._flow_cached_phase = phases.copy()
            return ref, phases

        conds = []
        env_ids = []
        cond_hist_tails = []
        planned_world = {}
        alpha = self._flow_alpha()
        use_robot = self._flow_feedback_source == "robot"
        use_prediction = self._flow_feedback_source == "prediction"

        for env_id in plan_env_ids:
            is_loco = bool(self.progress_buf[env_id].item() >= self._flow_stand_steps_env[env_id])
            phase = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32) if is_loco else np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            robot_hist = self._flow_history[env_id]
            pred_hist = self._flow_prediction_history[env_id]
            if len(robot_hist) < min_history:
                hist = pred_hist if self._flow_pure_ar else robot_hist
                planned_world[env_id] = [hist[-1]] * self._flow_stride
                continue

            if self._flow_pure_ar:
                hist_tail = pred_hist[-min_history:]
            elif use_prediction:
                hist_tail = pred_hist[-min_history:] if len(pred_hist) >= min_history else robot_hist[-min_history:]
            elif use_robot:
                hist_tail = robot_hist[-min_history:]
            else:
                if len(pred_hist) >= min_history and self._flow_rng.random() >= alpha:
                    hist_tail = pred_hist[-min_history:]
                else:
                    hist_tail = robot_hist[-min_history:]

            conds.append(self._flow_cond_from_history(hist_tail, phase, self._flow_command_heading_env[env_id]))
            env_ids.append(env_id)
            cond_hist_tails.append(hist_tail)

        if conds:
            conds = np.stack(conds, axis=0)
            targets = self._flow_x15.sample_horizon_batch(
                self._flow_model,
                conds,
                self._flow_stats,
                self._flow_device,
                self._flow_ode_steps,
                generator=self._flow_torch_generator,
            )
            for batch_i, env_id in enumerate(env_ids):
                hist_tail = cond_hist_tails[batch_i]
                anchor = hist_tail[-1]
                decoded = [
                    self._flow_x15.state_from_target_frame(
                        targets[batch_i, j],
                        anchor["root_xy"],
                        float(anchor["heading"]),
                        float(anchor["pitch"]),
                        float(anchor["roll"]),
                    )
                    for j in range(int(self._flow_stats["horizon"]))
                ]
                planned_world[env_id] = decoded[:self._flow_stride]
                pred_hist = self._flow_prediction_history[env_id]
                pred_hist.extend(decoded[:self._flow_stride])
                if len(pred_hist) > 128:
                    del pred_hist[:-128]

        buffer_np = np.zeros((len(plan_env_ids), self._flow_stride, 24, 3), dtype=np.float32)
        for local_i, env_id in enumerate(plan_env_ids):
            planned = planned_world.get(env_id)
            if planned is None:
                fallback = self._flow_prediction_history[env_id][-1] if self._flow_pure_ar else states[env_id]
                planned = [fallback] * self._flow_stride
            joints = np.stack([p["joints"] for p in planned], axis=0)
            server_j3d = self._flow_x15.mujoco_zup_to_server(joints)
            buffer_np[local_i] = server_j3d[:, smpl_2_mujoco]

        planned_ref = torch.from_numpy(buffer_np).float().matmul(self.to_isaac_mat.T).to(self.device)
        plan_env_ids_t = torch.as_tensor(plan_env_ids, dtype=torch.long, device=self.device)
        self._flow_ref_buffer[plan_env_ids_t] = planned_ref
        ref[plan_env_ids_t] = planned_ref[:, 0]
        self._flow_ref_buffer_idx[plan_env_ids] = 1
        self._flow_ref_buffer_valid[plan_env_ids] = True
        self._flow_ref_source = "generated"
        self._flow_cached_progress = progress_key
        self._flow_cached_ref = ref
        self._flow_cached_phase = np.asarray(phases, dtype=np.float32).copy()
        return ref, np.asarray(phases, dtype=np.float32)
        
    async def talk(self):
        URL = f'http://{SERVER}:8080/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close cmd':
                        await ws.close()
                        break
                    else:
                        print(msg.data)
                        try:
                            msg = json.loads(msg.data)
                            if msg['action'] == 'reset':
                                self.reset()
                            elif msg['action'] == 'start_record':
                                subprocess.Popen(["simplescreenrecorder", "--start-recording"])
                                print("start recording!!!!")
                                # self.recording = True
                            elif msg['action'] == 'end_record':
                                print("end_recording!!!!")
                                if not self.recording:
                                    print("Not recording")
                                else:
                                    self.recording = False
                                    self.recording_state_change = True
                            elif msg['action'] == 'set_env':
                                query = msg['query']
                                env_id = query['env']
                                self.viewing_env_idx = int(env_id)
                                print("view env idx: ", self.viewing_env_idx)
                        except:
                            import ipdb
                            ipdb.set_trace()
                            print("error parsing server message")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    def _update_marker(self):
        if flags.show_traj:
            self._marker_pos[:] = self.ref_body_pos
        else:
            self._marker_pos[:] = 0

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = torch_utils.calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))

        return

    def _log_flow_robot_world_pos(self, robot_body_pos, ref_body_pos):
        if self.flow_debug_log_every <= 0:
            return
        self._flow_debug_log_count += 1
        if self._flow_debug_log_count % self.flow_debug_log_every != 0:
            return

        if self._flow_debug_log_file is None:
            log_dir = os.path.dirname(self.flow_debug_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._flow_debug_log_file = open(self.flow_debug_log_path, "w")
            self._flow_debug_log_file.write(
                "wall_time,step,env,robot_x,robot_y,robot_z,ref_x,ref_y,ref_z,"
                "delta_x,delta_y,delta_z,root_dist,mean_body_dist,max_body_dist\n"
            )

        robot_np = robot_body_pos.detach().cpu().numpy()
        ref_np = ref_body_pos.detach().cpu().numpy()
        delta = ref_np[:, 0] - robot_np[:, 0]
        body_dist = np.linalg.norm(ref_np - robot_np, axis=-1)
        steps = self.progress_buf.detach().cpu().numpy() if hasattr(self, "progress_buf") else np.zeros(robot_np.shape[0], dtype=np.int64)
        now = time.time()

        for env_id in range(robot_np.shape[0]):
            root_dist = float(np.linalg.norm(delta[env_id]))
            self._flow_debug_log_file.write(
                f"{now:.6f},{int(steps[env_id])},{env_id},"
                f"{robot_np[env_id,0,0]:.6f},{robot_np[env_id,0,1]:.6f},{robot_np[env_id,0,2]:.6f},"
                f"{ref_np[env_id,0,0]:.6f},{ref_np[env_id,0,1]:.6f},{ref_np[env_id,0,2]:.6f},"
                f"{delta[env_id,0]:.6f},{delta[env_id,1]:.6f},{delta[env_id,2]:.6f},"
                f"{root_dist:.6f},{float(body_dist[env_id].mean()):.6f},{float(body_dist[env_id].max()):.6f}\n"
            )
        self._flow_debug_log_file.flush()

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs_demo(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs

        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:10].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, 5))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs
        return obs

    def _compute_task_obs_demo(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
        full_env_batch = len(env_ids) == self.num_envs

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        if self.obs_v == 6:
            raise NotImplementedError
            # This part is not as good. use obs_v == 7 instead.
            # ref_rb_pos = self.j3d[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d.shape[0]]
            # ref_body_vel = self.j3d_vel[((self.progress_buf[env_ids] + 1) / 2).long() % self.j3d_vel.shape[0]]
            # pose_mat = self.pose_mat.clone()
            # trans = self.trans.clone()

            # pose_mat = self.rot_mat_ref[((self.progress_buf[env_ids] + 1) / 2).long() % self.rot_mat_ref.shape[0]] # debugging
            pose_res = requests.get(f'http://{SERVER}:8080/get_pose')
            json_data = pose_res.json()
            pose_mat = torch.tensor(json_data["pose_mat"])[None,].float()
            # trans = torch.tensor(json_data["trans"]).to(self.device).float()

            trans = np.array(json_data["trans"]).squeeze()
            s_dt = json_data['dt']
            self.root_pos_acc.append(trans)
            filtered_trans = filters.gaussian_filter1d(self.root_pos_acc, 3, axis=0, mode="mirror")
            trans = torch.tensor(filtered_trans[-1]).float()

            new_root = self.to_isaac_mat.matmul(pose_mat[:, 0])
            pose_mat[:, 0] = new_root
            trans = trans.matmul(self.to_isaac_mat.T)
            _, global_rotation = humanoid_kin.forward_kinematics_batch(pose_mat[:, smpl_2_mujoco], self.zero_trans, self.local_translation_batch, self.parent_indices)

            ref_rb_rot = ptr.matrix_to_quaternion_ijkr(global_rotation.matmul(self.to_global))

            ##################  ##################
            ref_rb_rot_np = ref_rb_rot.numpy()[0]

            if len(self.body_rot_acc) > 0:
                ref_rb_rot_np = quat_correct_two_batch(self.body_rot_acc[-1], ref_rb_rot_np)
                filtered_quats = filters.gaussian_filter1d(np.concatenate([self.body_rot_acc, ref_rb_rot_np[None,]], axis=0), 1, axis=0, mode="mirror")
                new_quat = filtered_quats[-1] / np.linalg.norm(filtered_quats[-1], axis=1)[:, None]
                self.body_rot_acc.append(new_quat)  # add the filtered quat.

                # pose_quat_global = np.array(self.body_rot_acc)
                # select_quats = np.linalg.norm(pose_quat_global[:-1, :] - pose_quat_global[1:, :], axis=2) > np.linalg.norm(pose_quat_global[:-1, :] + pose_quat_global[1:, :], axis=2)
                ref_rb_rot = torch.tensor(new_quat[None,]).float()
            else:
                self.body_rot_acc.append(ref_rb_rot_np)

            ################## ##################

            ref_rb_pos = SkeletonState.from_rotation_and_root_translation(self.skeleton_trees[0], ref_rb_rot, trans, is_local=False).global_translation.to(self.device)  # SLOWWWWWWW
            ref_rb_rot = ref_rb_rot.to(self.device)
            ref_rb_pos = ref_rb_pos.to(self.device)
            ref_body_ang_vel = SkeletonMotion._compute_angular_velocity(torch.stack([self.prev_ref_body_rot, ref_rb_rot], dim=1), time_delta=s_dt, guassian_filter=False)[:, 0]
            ref_body_vel = SkeletonMotion._compute_velocity(torch.stack([self.prev_ref_body_pos, ref_rb_pos], dim=1), time_delta=s_dt, guassian_filter=False)[:, 0]  # this is slow!


            time_steps = 1
            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_rb_rot_subset[zeros_subset, 1:] = body_rot_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                ref_body_ang_vel_subset[zeros_subset, :] = body_ang_vel_subset[zeros_subset, :]

                far_distance = 3  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            obs = humanoid_im.compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)

            self.prev_ref_body_pos = ref_rb_pos
            self.prev_ref_body_rot = ref_rb_rot
        elif self.obs_v == 7:
            if self._flow_ref_mode == "inprocess":
                ref_rb_pos, phase_loco = self._next_inprocess_flow_ref(body_pos, body_rot, env_ids)
                _task_phase = np.array([1.0 - phase_loco[0], phase_loco[0], 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # --- push actual robot state to flow server for closed-loop history ---
                try:
                    _rp = body_pos[0, 0].cpu().numpy()   # pelvis world pos (3,) Z-up
                    _rq = body_rot[0, 0].cpu().numpy()   # root quat xyzw
                    _heading, _pitch, _roll = sRot.from_quat(_rq).as_euler("ZYX")
                    _joint_rel = (body_pos[0, 1:] - body_pos[0, 0]).cpu().numpy().reshape(-1).tolist()
                    requests.post(f'http://{SERVER}:8080/push_state', json={
                        "root_xy": [float(_rp[0]), float(_rp[1])],
                        "root_z": float(_rp[2]),
                        "heading": float(_heading),
                        "pitch": float(_pitch),
                        "roll": float(_roll),
                        "joint_rel": _joint_rel,
                        "episode_step": int(self.progress_buf[0].item()),
                        "episode_idx": int(self._flow_episode_idx),
                    }, timeout=0.005)
                except Exception:
                    pass
                # ----------------------------------------------------------------------
                pose_res = requests.get(f'http://{SERVER}:8080/get_pose')
                json_data = pose_res.json()
                ref_rb_pos_orig = np.array(json_data["j3d"])[:self.num_envs, smpl_2_mujoco]
                _task_phase = np.array(json_data.get("task_phase", [0]*5), dtype=np.float32)
                ref_rb_pos = torch.from_numpy(ref_rb_pos_orig).float().matmul(self.to_isaac_mat.T).cuda()
                phase_loco = np.full((self.num_envs,), float(_task_phase[1] > 0.5), dtype=np.float32)
            s_dt = 1/30

            self._log_flow_robot_world_pos(body_pos, ref_rb_pos)
            self._log_full_body(self._rigid_body_pos[0], ref_rb_pos[0], _task_phase)
            flow_body_dist = torch.norm(ref_rb_pos - body_pos, dim=-1)
            if full_env_batch:
                self.extras["wandb/flow/root_dist"] = flow_body_dist[:, 0].detach()
                self.extras["wandb/flow/mean_body_dist"] = flow_body_dist.mean(dim=-1).detach()
                self.extras["wandb/flow/max_body_dist"] = flow_body_dist.max(dim=-1).values.detach()
                self.extras["wandb/flow/episode_idx"] = torch.full(
                    (self.num_envs,),
                    float(self._flow_episode_idx),
                    device=self.device,
                )
                self.extras["wandb/flow/episode_step"] = self.progress_buf.float().detach()
                self.extras["wandb/flow/is_loco"] = torch.full(
                    (self.num_envs,), 0.0, device=self.device
                )
                self.extras["wandb/flow/is_loco"][:] = torch.from_numpy(phase_loco).float().to(self.device)
                if self._flow_ref_mode == "inprocess":
                    self.extras["wandb/flow/command_heading"] = torch.from_numpy(
                        self._flow_command_heading_env
                    ).float().to(self.device)
                    self.extras["wandb/flow/stand_steps"] = torch.from_numpy(
                        self._flow_stand_steps_env.astype(np.float32)
                    ).to(self.device)

            prev_ref_body_pos = self.prev_ref_body_pos if full_env_batch else self.prev_ref_body_pos[env_ids]
            ref_body_vel = SkeletonMotion._compute_velocity(torch.stack([prev_ref_body_pos, ref_rb_pos], dim=1), time_delta=s_dt, guassian_filter=False)[:, 0]

            time_steps = 1
            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            obs = humanoid_im.compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)

            if full_env_batch:
                self.prev_ref_body_pos = ref_rb_pos
                self.ref_body_pos = ref_rb_pos
                self.ref_body_vel = ref_body_vel
                self.ref_body_rot = body_rot.clone()
                self.ref_body_pos_subset = ref_rb_pos_subset
            else:
                self.prev_ref_body_pos[env_ids] = ref_rb_pos
                self.ref_body_pos[env_ids] = ref_rb_pos
                self.ref_body_vel[env_ids] = ref_body_vel
                self.ref_body_rot[env_ids] = body_rot.clone()
            self._flow_ref_valid = True

        if len(env_ids) == self.num_envs:
            self.ref_body_pos = ref_rb_pos
            if self.obs_v == 7:
                self.ref_body_vel = ref_body_vel
                self.ref_body_rot = body_rot.clone()
                self.ref_body_pos_subset = ref_rb_pos_subset
            else:
                self.ref_body_pos_subset = torch.from_numpy(ref_rb_pos_orig)
            self.ref_pose_aa = None

        return obs

    def _compute_reward(self, actions):
        if self.obs_v != 7 or not self._flow_ref_valid:
            return super()._compute_reward(actions)

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        ref_rb_pos = self.ref_body_pos
        ref_rb_rot = body_rot.detach().clone()
        ref_body_vel = self.ref_body_vel
        ref_body_ang_vel = body_ang_vel.detach().clone()

        def _im_reward(curr_mask=None):
            if self._full_body_reward and self.humanoid_type not in ['h1', 'g1']:
                args = (
                    root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel,
                    ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel,
                )
            else:
                body_pos_subset = body_pos[..., self._track_bodies_id, :]
                body_rot_subset = body_rot[..., self._track_bodies_id, :]
                body_vel_subset = body_vel[..., self._track_bodies_id, :]
                body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]
                ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
                ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
                ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
                ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
                args = (
                    root_pos, root_rot, body_pos_subset, body_rot_subset,
                    body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset,
                    ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset,
                )

            if curr_mask is not None:
                args = tuple(x[curr_mask] for x in args)
            return humanoid_im.compute_imitation_reward(*args, self.reward_specs)

        if self.zero_out_far:
            transition_distance = 0.25
            distance = torch.norm(root_pos - ref_rb_pos[..., 0, :], dim=-1)
            zeros_subset = distance > transition_distance
            self.reward_raw = torch.zeros((self.num_envs, 4)).to(self.device)
            self.rew_buf, self.reward_raw[:, 0] = humanoid_im.compute_point_goal_reward(self._point_goal, distance)
            inside = ~zeros_subset
            if torch.any(inside):
                im_reward, im_reward_raw = _im_reward(inside)
                self.rew_buf[inside] = self.rew_buf[inside] + im_reward * 0.5
                self.reward_raw[inside, :4] = self.reward_raw[inside, :4] + im_reward_raw * 0.5
        else:
            self.rew_buf[:], self.reward_raw = _im_reward()

        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0
            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        return
    
    def _log_full_body(self, robot_body_pos, ref_body_pos, task_phase):
        if not self._body_log_enabled:
            return
        self._body_log_robot.append(robot_body_pos.detach().cpu().numpy())
        self._body_log_ref.append(ref_body_pos.detach().cpu().numpy())
        self._body_log_phase.append(task_phase)
        self._body_log_progress.append(
            int(self.progress_buf[0].item()) if hasattr(self, "progress_buf") else -1
        )
        buffer_idx = getattr(self, "_flow_ref_buffer_idx", -1)
        if isinstance(buffer_idx, np.ndarray):
            buffer_idx = buffer_idx[0] if len(buffer_idx) > 0 else -1
        self._body_log_buffer_idx.append(int(buffer_idx))
        self._body_log_chunk_start.append(int(getattr(self, "_flow_ref_chunk_start", -1) or -1))
        self._body_log_ref_source.append(str(getattr(self, "_flow_ref_source", "unknown")))
        buffer_valid = getattr(self, "_flow_ref_buffer_valid", False)
        if isinstance(buffer_valid, np.ndarray):
            buffer_valid = buffer_valid[0] if len(buffer_valid) > 0 else False
        self._body_log_buffer_valid.append(int(bool(buffer_valid)))
        self._body_log_reset_count.append(int(getattr(self, "_flow_reset_count", -1)))
        os.makedirs(os.path.dirname(self._body_log_path), exist_ok=True)
        np.savez(
            self._body_log_path,
            ref=np.stack(self._body_log_ref),
            robot=np.stack(self._body_log_robot),
            phase=np.stack(self._body_log_phase),
            progress=np.asarray(self._body_log_progress, dtype=np.int64),
            buffer_idx=np.asarray(self._body_log_buffer_idx, dtype=np.int64),
            chunk_start=np.asarray(self._body_log_chunk_start, dtype=np.int64),
            ref_source=np.asarray(self._body_log_ref_source),
            buffer_valid=np.asarray(self._body_log_buffer_valid, dtype=np.int64),
            reset_count=np.asarray(self._body_log_reset_count, dtype=np.int64),
        )

    def _compute_reset(self):
        self.reset_buf[:] = 0
        self._terminate_buf[:] = 0
        if (
            self._flow_ref_mode == "inprocess"
            and self._flow_replay_joints is not None
            and self._flow_replay_reset_on_loop
        ):
            replay_len = int(len(self._flow_replay_joints))
            if replay_len > 0:
                self.reset_buf[:] = self.progress_buf >= (replay_len - 1)
                return
        if self._flow_ref_mode == "inprocess":
            pass_time = self.progress_buf >= self.max_episode_length - 1
            if (
                self._flow_enable_tracking_reset
                and self._flow_ref_valid
                and hasattr(self, "ref_body_pos")
            ):
                body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
                ref_body_pos = self.ref_body_pos[..., self._reset_bodies_id, :].clone()
                self.reset_buf[:], self._terminate_buf[:] = humanoid_im.compute_humanoid_im_reset(
                    self.reset_buf,
                    self.progress_buf,
                    self._contact_forces,
                    self._contact_body_ids,
                    body_pos,
                    ref_body_pos,
                    pass_time,
                    self._enable_early_termination,
                    self._termination_distances[..., self._reset_bodies_id],
                    self._flow_tracking_reset_disable_collision,
                    flags.im_eval and (not self.strict_eval),
                )
            else:
                self.reset_buf[:] = pass_time
            return
        try:
            r = requests.get(f'http://{SERVER}:8080/should_reset', timeout=0.01)
            if r.json().get("reset", False):
                self.reset_buf[:] = 1
        except Exception:
            pass

    def _reset_actors(self, env_ids):
        if (
            hasattr(self, "_flow_reset_history_on_env_reset")
            and self._flow_ref_mode == "inprocess"
            and self._flow_reset_history_on_env_reset
        ):
            self._reset_flow_env_streams(env_ids)
        super()._reset_actors(env_ids)
        if not self._align_robot_start_to_flow_heading or len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids = env_ids.to(self.device)
        flow_env_ids = self._flow_env_ids_to_list(env_ids)
        root_rot = self._humanoid_root_states[env_ids, 3:7]
        current_heading = torch_utils.calc_heading(root_rot)
        desired_heading = torch.tensor(
            self._flow_command_heading_env[flow_env_ids],
            dtype=torch.float,
            device=self.device,
        )
        angles = desired_heading - current_heading
        axis = torch.zeros((len(env_ids), 3), device=self.device)
        axis[:, 2] = 1.0
        yaw = quat_from_angle_axis(angles, axis)
        self._humanoid_root_states[env_ids, 3:7] = quat_mul(yaw, root_rot)

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        if self._flow_ref_mode == "inprocess":
            self._flow_reset_count += 1
            if self._flow_replay_joints is None:
                if self._flow_reset_history_on_env_reset:
                    self._flow_ref_source = "reset_default_history"
                    self._flow_episode_idx += len(env_ids) / float(self.num_envs)
                    self._flow_ref_valid = False
                    return
                self._flow_ref_source = "robot_reset_preserve_flow"
                return
            if torch.is_tensor(env_ids):
                flow_env_ids = env_ids.detach().cpu().numpy().tolist()
            else:
                flow_env_ids = list(env_ids)
            for env_id in flow_env_ids:
                self._flow_history[int(env_id)].clear()
                self._flow_prediction_history[int(env_id)].clear()
            if self._flow_seed_default_history or self._flow_pure_ar:
                self._seed_flow_default_history(flow_env_ids)
            self._flow_ref_buffer_idx[flow_env_ids] = self._flow_stride
            self._flow_ref_chunk_start = None
            self._flow_ref_buffer_valid[flow_env_ids] = False
            self._flow_ref_source = "reset"
            self._flow_cached_ref = None
            self._flow_cached_phase = None
            self._flow_cached_progress = None
            self._flow_episode_idx += len(env_ids) / float(self.num_envs)
        elif len(env_ids) == self.num_envs:
            self._flow_episode_idx += 1
        self._flow_ref_valid = False
        return
        
