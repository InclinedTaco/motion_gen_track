"""
flow_server.py: Real-time flow model pose server for PHC demo mode.

The flow model generates human joint positions autoregressively from a
commanded velocity.  PHC (obs_v=7) polls GET /get_pose every sim step.

Usage
-----
Terminal 1 (this server):
  cd <PROJECT_ROOT>/tracking
  python scripts/flow_server.py \\
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \\
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json

Terminal 2 (PHC):
  python phc/run_hydra.py \\
    learning=im_mcp exp_name=phc_kp_mcp_iccv \\
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \\
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \\
    env.z_activation=relu env.obs_v=7 \\
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \\
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \\
    env.num_envs=1 headless=False test=True epoch=-1

Keyboard controls (terminal 1):
  W / S       forward / backward velocity
  A / D       left / right velocity
  Q / E       raise / lower pelvis height
  Space       zero velocity (stand still)
  Esc / Ctrl-C  quit
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

import numpy as np
import torch
from flask import Flask, jsonify
from scipy.spatial.transform import Rotation as sRot

# --- path setup -----------------------------------------------------------
PHC_ROOT       = Path(__file__).resolve().parents[1]
FLOW_ROOT      = PHC_ROOT.parent / "generation"
JOINT_FLOW_DIR = FLOW_ROOT / "examples" / "joint_flow"
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(FLOW_ROOT))
sys.path.insert(0, str(JOINT_FLOW_DIR))

from flow_matching.solver import ODESolver
from model import JointFlowMLP
from infer import CondWrapper

# --------------------------------------------------------------------------
# Coordinate system constants
# --------------------------------------------------------------------------

# smpl_2_mujoco[i] = SMPL anatomical index of MuJoCo DFS joint i
# PHC does: ref_rb_pos = j3d[:, smpl_2_mujoco]  (SMPL-order j3d → MuJoCo-order)
SMPL_2_MUJOCO = np.array(
    [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
)

# Our flow model outputs in MuJoCo DFS (Z-up) order.
# PHC applies to_isaac_mat = R_x(-π/2) after receiving, so pre-compensate.
_TO_ISAAC = sRot.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix().astype(np.float32)


def mujoco_zup_to_server(pos_mujoco: np.ndarray) -> np.ndarray:
    """
    Convert (N, 24, 3) Z-up MuJoCo-order positions to the (N, 24, 3)
    SMPL-order camera-frame array expected by PHC's obs_v=7 endpoint.

    PHC pipeline on receipt:
      j3d[:, smpl_2_mujoco]   →  MuJoCo order
      matmul(to_isaac_mat.T)  →  Z-up Isaac frame   ← we want this to equal pos_mujoco
    """
    # Step 1: pre-apply TO_ISAAC so PHC's to_isaac_mat.T undoes it
    pos_cam = pos_mujoco @ _TO_ISAAC          # (N, 24, 3)
    # Step 2: scatter MuJoCo order → SMPL order
    j3d = np.empty_like(pos_cam)
    j3d[:, SMPL_2_MUJOCO, :] = pos_cam       # j3d[smpl_idx] = pos_cam[mujoco_idx]
    return j3d                                # (N, 24, 3)


# --------------------------------------------------------------------------
# Shared state (guarded by a lock)
# --------------------------------------------------------------------------
class FlowState:
    def __init__(self, height_init: float, fps: float):
        self.lock = threading.Lock()
        self.vel_x   = 0.0
        self.vel_y   = 0.0
        self.height  = height_init
        self.fps     = fps
        self.dt      = 1.0 / fps
        self.pelvis_xy = np.zeros(2, dtype=np.float32)
        # autoregressive state
        self.prev_joints_norm = np.zeros(69, dtype=np.float32)
        # latest j3d in server format: (1, 24, 3)
        # initialise with a plausible standing position (pelvis at height)
        standing = np.zeros((1, 24, 3), dtype=np.float32)
        standing[:, 0, 2] = height_init
        self.current_j3d = mujoco_zup_to_server(standing)
        self.running = True

    def get_command(self):
        with self.lock:
            return self.vel_x, self.vel_y, self.height

    def set_vel(self, vx, vy):
        with self.lock:
            self.vel_x = vx
            self.vel_y = vy

    def set_height(self, h):
        with self.lock:
            self.height = max(0.6, min(1.2, h))

    def get_j3d(self):
        with self.lock:
            return self.current_j3d.tolist()

    def stop(self):
        with self.lock:
            self.running = False

    def is_running(self):
        with self.lock:
            return self.running


# --------------------------------------------------------------------------
# Flow model inference thread
# --------------------------------------------------------------------------
def flow_loop(state: FlowState, model: JointFlowMLP, stats: dict,
              window: int, device: torch.device, ode_steps: int):
    cond_mean  = np.array(stats["cond_mean"],  dtype=np.float32)   # (3,)
    cond_std   = np.array(stats["cond_std"],   dtype=np.float32)   # (3,)
    joint_mean = np.array(stats["joint_mean"], dtype=np.float32)   # (69,)
    joint_std  = np.array(stats["joint_std"],  dtype=np.float32)   # (69,)

    cond_window = deque(maxlen=window)  # deque of (3,) normalized cond frames

    use_obj  = stats.get("use_obj", False)
    sentinel = np.array([2.0, 2.0, 2.0], dtype=np.float32)  # no-object token

    def step_once():
        vx, vy, h = state.get_command()

        # normalize current task-space frame
        raw = np.array([vx, vy, h], dtype=np.float32)
        if use_obj:
            raw = np.concatenate([raw, sentinel])
        cond_t = (raw - cond_mean) / cond_std
        cond_window.append(cond_t)

        # pad window if not full yet
        while len(cond_window) < window:
            cond_window.appendleft(cond_window[0])

        # build flat condition vector
        win_arr = np.stack(list(cond_window), axis=0)  # (K, 3)
        cond_vec = np.concatenate([state.prev_joints_norm, win_arr.reshape(-1)])  # (69 + K*3,)

        cond_tensor = torch.tensor(cond_vec, device=device).unsqueeze(0)  # (1, cond_dim)

        x0 = torch.randn(1, 69, device=device)
        wrapped = CondWrapper(model, cond_tensor)
        with torch.no_grad():
            solver = ODESolver(velocity_model=wrapped)
            x1 = solver.sample(
                x_init=x0,
                step_size=1.0 / ode_steps,
                method="midpoint",
                time_grid=torch.linspace(0, 1, ode_steps + 1, device=device),
                return_intermediates=False,
            )
        x1_np = np.clip(x1.cpu().numpy()[0], -10, 10)

        # update autoregressive state
        state.prev_joints_norm = x1_np

        # denormalise → pelvis-relative positions
        joints_23  = (x1_np * joint_std + joint_mean).reshape(23, 3)
        joints_rel = np.concatenate([np.zeros((1, 3), dtype=np.float32), joints_23], axis=0)  # (24, 3)

        # accumulate pelvis XY
        with state.lock:
            state.pelvis_xy += np.array([vx, vy], dtype=np.float32) * state.dt
            px, py = state.pelvis_xy.copy()

        pelvis_world = np.array([px, py, h], dtype=np.float32)
        pos_world = joints_rel + pelvis_world          # (24, 3) Z-up MuJoCo order
        pos_world = pos_world[np.newaxis]              # (1, 24, 3)

        j3d = mujoco_zup_to_server(pos_world)          # (1, 24, 3) server format
        with state.lock:
            state.current_j3d = j3d

    target_dt = state.dt
    while state.is_running():
        t0 = time.perf_counter()
        step_once()
        elapsed = time.perf_counter() - t0
        sleep_t = target_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------
app = Flask(__name__)
_state: FlowState = None   # set before starting


@app.route("/get_pose")
def get_pose():
    return jsonify({"j3d": _state.get_j3d()})


@app.route("/set_vel")
def set_vel():
    """Optional REST endpoint: /set_vel?x=0.5&y=0.0&h=0.9"""
    from flask import request
    vx = float(request.args.get("x", 0.0))
    vy = float(request.args.get("y", 0.0))
    h  = float(request.args.get("h", _state.height))
    _state.set_vel(vx, vy)
    _state.set_height(h)
    return jsonify({"vel_x": vx, "vel_y": vy, "height": h})


def run_flask():
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8080, threaded=True)


# --------------------------------------------------------------------------
# Keyboard control (main thread, raw tty)
# --------------------------------------------------------------------------
VEL_STEP    = 0.2   # m/s per key press
HEIGHT_STEP = 0.05  # m per key press
VEL_MAX     = 2.0


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def print_status(state: FlowState):
    vx, vy, h = state.get_command()
    sys.stdout.write(
        f"\r  vel=({vx:+.2f}, {vy:+.2f})  height={h:.2f}  "
        f"pelvis_xy=({state.pelvis_xy[0]:.1f}, {state.pelvis_xy[1]:.1f})   "
    )
    sys.stdout.flush()


def keyboard_loop(state: FlowState):
    print("\n=== Flow Server running on port 8080 ===")
    print("  W/S  forward/back   A/D  left/right")
    print("  Q/E  raise/lower    Space  stop")
    print("  Esc or Ctrl-C to quit\n")

    vx, vy = 0.0, 0.0
    while state.is_running():
        ch = getch()
        if ch in ("\x1b", "\x03"):   # Esc or Ctrl-C
            state.stop()
            break
        elif ch == "w":
            vx = min(vx + VEL_STEP, VEL_MAX)
        elif ch == "s":
            vx = max(vx - VEL_STEP, -VEL_MAX)
        elif ch == "a":
            vy = min(vy + VEL_STEP, VEL_MAX)
        elif ch == "d":
            vy = max(vy - VEL_STEP, -VEL_MAX)
        elif ch == " ":
            vx, vy = 0.0, 0.0
        elif ch == "q":
            state.set_height(state.height + HEIGHT_STEP)
        elif ch == "e":
            state.set_height(state.height - HEIGHT_STEP)
        state.set_vel(vx, vy)
        print_status(state)

    print("\nShutting down …")


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flow-ckpt",  required=True)
    p.add_argument("--flow-stats", required=True)
    p.add_argument("--height",  type=float, default=0.9, help="Initial pelvis height (m)")
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--ode-steps", type=int, default=10,
                   help="ODE steps per frame (10 is fast enough for real-time)")
    p.add_argument("--warmup",  type=int,   default=30,
                   help="Frames to run before accepting PHC connections")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    global _state
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.flow_stats) as f:
        stats = json.load(f)

    ckpt       = torch.load(args.flow_ckpt, map_location=device,
                            weights_only=False)
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
    print(f"Flow model loaded  (device={device}, cond_dim={cond_dim})")

    _state = FlowState(height_init=args.height, fps=args.fps)

    # start flow inference thread
    flow_thread = threading.Thread(
        target=flow_loop,
        args=(_state, model, stats, window, device, args.ode_steps),
        daemon=True,
    )
    flow_thread.start()

    # warmup: let the model run a few frames before PHC connects
    print(f"Warming up ({args.warmup} frames) …", end="", flush=True)
    time.sleep(args.warmup / args.fps)
    print(" done.")

    # start Flask in daemon thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # keyboard control in main thread (only if running in a real terminal)
    try:
        if sys.stdin.isatty():
            keyboard_loop(_state)
        else:
            print("stdin is not a tty — use REST endpoint to control velocity:")
            print("  curl 'http://localhost:8080/set_vel?x=0.5&y=0.0&h=0.9'")
            while _state.is_running():
                time.sleep(0.5)
    except KeyboardInterrupt:
        _state.stop()


if __name__ == "__main__":
    main()
