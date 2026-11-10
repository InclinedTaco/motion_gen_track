"""
flow_server_clip.py: Real-time flow model server replaying a specific HDF5 clip.

Replays the clip's task-space + obj trajectory frame-by-frame through the flow
model in real-time (StreamFitter for retargeting), serving poses to PHC via
GET /get_pose.  Useful for testing whether real-time inference reproduces the
quality of offline animate.py on a known clip.

Terminal 1 (server):
  cd <PROJECT_ROOT>/tracking
  python scripts/flow_server_clip.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_obj_accad_omomo/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_obj_accad_omomo/stats.json \
    --hdf5       <PROJECT_ROOT>/tracking/data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \
    --clip       45

Terminal 2 (PHC tracking):
  python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1

Controls: Space=pause/resume  R=restart  Esc/Ctrl-C=quit
"""

import argparse
import json
import sys
import threading
import time
import tty
import termios
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
from flask import Flask, jsonify

PHC_ROOT       = Path(__file__).resolve().parents[1]
FLOW_ROOT      = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_ROOT / "scripts"))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))

from flow_matching.solver import ODESolver
from model import JointFlowMLP
from infer import CondWrapper
from flow_retarget import build_fk, StreamFitter
from flow_server import mujoco_zup_to_server   # reuse coord-transform util


# --------------------------------------------------------------------------
def load_clip(hdf5_path, clip_idx):
    with h5py.File(hdf5_path, "r") as f:
        clips   = sorted(f["clips"].keys())
        name    = clips[clip_idx]
        grp     = f[f"clips/{name}"]
        ts      = grp["task_space"][:].astype(np.float32)   # (T, 9)
        jp      = grp["joint_positions"][:].astype(np.float32)  # (T, 24, 3)
        fps     = float(grp.attrs["fps"])
        obj_rel = grp["obj_com_pos_rel"][:].astype(np.float32) if "obj_com_pos_rel" in grp else None
    print(f"Clip [{clip_idx}]: {name}  |  {len(ts)} frames @ {fps:.0f} fps  |  has_obj={obj_rel is not None}")
    return ts, jp, fps, name, obj_rel


def load_model(ckpt_path, stats, device):
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    window     = stats.get("window", 10)
    prev_dim   = 69 if stats.get("use_prev_frame", False) else 0
    ts_dim     = (3 if stats.get("no_eef", False) else 9) + (3 if stats.get("use_obj", False) else 0)
    cond_dim   = prev_dim + window * ts_dim
    model = JointFlowMLP(
        joint_dim=69,
        cond_dim=cond_dim,
        hidden_size=train_args.get("hidden", 512),
        n_layers=train_args.get("layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded  cond_dim={cond_dim}  use_obj={stats.get('use_obj')}  use_prev={stats.get('use_prev_frame')}")
    return model, window


# --------------------------------------------------------------------------
class ClipState:
    def __init__(self, ts_world, jp_gt, obj_rel, fps):
        self.ts_world  = ts_world   # (T, 9)
        self.jp_gt     = jp_gt      # (T, 24, 3)
        self.obj_rel   = obj_rel    # (T, 3) or None
        self.T         = len(ts_world)
        self.fps       = fps
        self.dt        = 1.0 / fps
        self.lock      = threading.Lock()
        self.frame          = 0
        self.paused         = False
        self.running        = True
        self._reset_pending = False
        standing = np.zeros((1, 24, 3), dtype=np.float32)
        standing[:, 0, 2] = ts_world[0, 2]
        self.current_j3d = mujoco_zup_to_server(standing)

    def current_frame(self):
        with self.lock:
            return self.frame

    def advance(self):
        with self.lock:
            if not self.paused:
                next_frame = (self.frame + 1) % self.T
                if next_frame == 0:
                    self._reset_pending = True
                self.frame = next_frame

    def restart(self):
        with self.lock:
            self.frame = 0

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running

    def pop_reset(self):
        """Returns True once after each clip loop-back, then False until next reset."""
        with self.lock:
            v = self._reset_pending
            self._reset_pending = False
            return v

    def set_j3d(self, j3d):
        with self.lock:
            self.current_j3d = j3d

    def get_j3d(self):
        with self.lock:
            return self.current_j3d.tolist()


# --------------------------------------------------------------------------
def flow_loop(state: ClipState, model, stats, window, device, ode_steps, fitter):
    cond_mean  = np.array(stats["cond_mean"],  dtype=np.float32)
    cond_std   = np.array(stats["cond_std"],   dtype=np.float32)
    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)
    joint_std  = np.array(stats["joint_std"],  dtype=np.float32)
    no_eef     = stats.get("no_eef", False)
    use_obj    = stats.get("use_obj", False)
    use_prev   = stats.get("use_prev_frame", False)
    ts_dim     = (3 if no_eef else 9) + (3 if use_obj else 0)

    cond_window      = deque(maxlen=window)
    prev_joints_norm = np.zeros(69, dtype=np.float32)

    # compute pelvis XY by integrating root_vel from clip
    T   = state.T
    dt  = state.dt
    pelvis_xy = np.zeros((T, 2), dtype=np.float32)
    pelvis_xy[0] = state.jp_gt[0, 0, :2]
    for t in range(1, T):
        pelvis_xy[t] = pelvis_xy[t-1] + state.ts_world[t, :2] * dt

    # pre-normalize all cond frames
    sentinel = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    cond_norm = np.empty((T, ts_dim), dtype=np.float32)
    for t in range(T):
        f = state.ts_world[t].copy()
        if not no_eef:
            ph = np.array([pelvis_xy[t, 0], pelvis_xy[t, 1], f[2]], dtype=np.float32)
            f[3:6] -= ph
            f[6:9] -= ph
        base = f[:3] if no_eef else f[:9]
        obj  = state.obj_rel[t] if (use_obj and state.obj_rel is not None) else sentinel
        row  = np.concatenate([base, obj]) if use_obj else base
        cond_norm[t] = (row - cond_mean) / cond_std

    target_dt = dt
    while state.is_running():
        t0 = time.perf_counter()

        t = state.current_frame()
        cond_window.append(cond_norm[t])
        while len(cond_window) < window:
            cond_window.appendleft(cond_window[0])

        win_arr  = np.stack(list(cond_window), axis=0)   # (K, ts_dim)
        cond_vec = np.concatenate([prev_joints_norm, win_arr.reshape(-1)]) if use_prev \
                   else win_arr.reshape(-1)

        cond_t = torch.tensor(cond_vec, device=device).unsqueeze(0)
        x0     = torch.randn(1, 69, device=device)
        with torch.no_grad():
            solver = ODESolver(velocity_model=CondWrapper(model, cond_t))
            x1 = solver.sample(
                x_init=x0,
                step_size=1.0 / ode_steps,
                method="midpoint",
                time_grid=torch.linspace(0, 1, ode_steps + 1, device=device),
                return_intermediates=False,
            )
        x1_np = np.clip(x1.cpu().numpy()[0], -10, 10)

        if use_prev:
            prev_joints_norm = x1_np

        joints_23  = (x1_np * joint_std + joint_mean).reshape(23, 3)
        joints_rel = np.concatenate([np.zeros((1, 3), dtype=np.float32), joints_23], axis=0)
        pos_world  = (joints_rel + np.array([pelvis_xy[t, 0], pelvis_xy[t, 1], state.ts_world[t, 2]],
                                             dtype=np.float32))[np.newaxis]   # (1, 24, 3)

        # StreamFitter: warm-start retargeting for smooth rotations (keeps prev frame state)
        fitter.step(pos_world[0])

        j3d = mujoco_zup_to_server(pos_world)
        state.set_j3d(j3d)
        state.advance()

        elapsed  = time.perf_counter() - t0
        sleep_t  = target_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        elif elapsed > target_dt * 1.5:
            print(f"  WARNING: frame took {elapsed*1000:.1f} ms (budget {target_dt*1000:.0f} ms)")


# --------------------------------------------------------------------------
app    = Flask(__name__)
_state: ClipState = None


@app.route("/get_pose")
def get_pose():
    return jsonify({"j3d": _state.get_j3d()})


@app.route("/should_reset")
def should_reset():
    return jsonify({"reset": _state.pop_reset()})


def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8080, threaded=True)


# --------------------------------------------------------------------------
def getch():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def keyboard_loop(state: ClipState, clip_name: str):
    print(f"\n=== Flow Clip Server  port=8080  clip={clip_name}  T={state.T} ===")
    print("  Space=pause/resume   R=restart   Esc/Ctrl-C=quit\n")
    while state.is_running():
        ch = getch()
        if ch in ("\x1b", "\x03"):
            state.stop()
            break
        elif ch == " ":
            state.toggle_pause()
            paused = state.paused
            print(f"\r  {'PAUSED ' if paused else 'RUNNING'}  frame={state.current_frame()}/{state.T}   ", end="")
        elif ch in ("r", "R"):
            state.restart()
            print(f"\r  RESTART                                  ", end="")
        sys.stdout.write(f"\r  frame={state.current_frame():4d}/{state.T}   ")
        sys.stdout.flush()
    print("\nShutting down …")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt",  required=True)
    p.add_argument("--flow-stats", required=True)
    p.add_argument("--hdf5",       required=True)
    p.add_argument("--clip",       type=int, default=0)
    p.add_argument("--humanoid-xml", default="final_humanoid.xml")
    p.add_argument("--ode-steps",  type=int, default=10)
    p.add_argument("--fit-iters",  type=int, default=8,
                   help="StreamFitter gradient iters per frame (8 = ~22 ms)")
    p.add_argument("--warmup",     type=int, default=30)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--retarget-device", default="cpu")
    return p.parse_args()


def main():
    global _state
    args    = parse_args()
    device  = torch.device(args.device)
    rdevice = torch.device(args.retarget_device)

    with open(args.flow_stats) as f:
        stats = json.load(f)

    model, window = load_model(args.flow_ckpt, stats, device)
    ts_world, jp_gt, fps, clip_name, obj_rel = load_clip(args.hdf5, args.clip)

    fk      = build_fk(args.humanoid_xml, rdevice)
    fitter  = StreamFitter(fk, device=rdevice, iters=args.fit_iters)

    _state  = ClipState(ts_world, jp_gt, obj_rel, fps)

    flow_thread = threading.Thread(
        target=flow_loop,
        args=(_state, model, stats, window, device, args.ode_steps, fitter),
        daemon=True,
    )
    flow_thread.start()

    print(f"Warming up ({args.warmup} frames) …", end="", flush=True)
    time.sleep(args.warmup / fps)
    print(" done.")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        if sys.stdin.isatty():
            keyboard_loop(_state, clip_name)
        else:
            while _state.is_running():
                time.sleep(0.5)
    except KeyboardInterrupt:
        _state.stop()


if __name__ == "__main__":
    main()
