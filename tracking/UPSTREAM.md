> Upstream PHC documentation and earlier research notes, retained for attribution and reference. Reported paper metrics are upstream results, not measurements of this integration. Start with [the unified setup](../docs/setup.md).

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/zhengyiluo-phc-badge.png)](https://mseep.ai/app/zhengyiluo-phc)

# Perpetual Humanoid Control for Real-time Simulated Avatars

Official implementation of ICCV 2023 paper: "Perpetual Humanoid Control for Real-time Simulated Avatars". In this paper, we present a physics-based humanoid controller that achieves high-fidelity motion imitation and fail-statue recovery in the presence of noisy input (e.g. pose estimates from video or generated from language) and unexpected falls. No external forces is used. 

[[paper]](https://arxiv.org/abs/2305.06456) [[website]](https://zhengyiluo.github.io/PHC/) [[Video]](https://www.youtube.com/watch?v=zS6Y00EW37A)

<div float="center">
<img src="assets/isaaclab_teaser.gif" />
  <img src="assets/phc_teaser.gif" />
  <img src="assets/h1_phc.gif" />
</div>

## Research Workflow

This section consolidates everything specific to our research setup: the PHC policy configuration, the humanoid XML we use, and the full data pipeline from raw AMASS to task-space HDF5.

---

### PHC Evaluation Setup

**Policy configuration**

| Setting | Value |
|---------|-------|
| Policy family | `learning=im_mcp` |
| Checkpoint | `output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth` |
| Experiment name | `phc_kp_mcp_iccv` |

**Motion-tracking command**

```bash
python phc/run_hydra.py \
  learning=im_mcp \
  exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp \
  robot=smpl_humanoid \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.z_activation=relu \
  env.obs_v=7 \
  env.motion_file=data/amass/pkls/<your_motion>.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 \
  headless=False \
  test=True \
  epoch=-1
```

**Example with squat motion**

```bash
python phc/run_hydra.py \
  learning=im_mcp \
  exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp \
  robot=smpl_humanoid \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.z_activation=relu \
  env.obs_v=7 \
  env.motion_file=data/amass/pkls/squat_tpose_hands_final.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 \
  headless=False \
  test=True \
  epoch=-1
```

---

### Retargeting humanoid — `final_humanoid.xml`

Do **not** retarget against the generic checked-in `phc/data/assets/mjcf/smpl_humanoid.xml`. Use the runtime-generated `final_humanoid.xml` instead, which reflects the exact flags used during evaluation.

**How to generate it:** run the motion-tracking command above once — PHC writes the runtime XML to `final_humanoid.xml` at startup.

**Flags that affect the XML:**

| Flag | Value |
|------|-------|
| `robot` | `smpl_humanoid` |
| `robot.freeze_hand` | `True` |
| `robot.box_body` | `False` |
| `env` | `env_im_getup_mcp` |
| `learning` | `im_mcp` |
| `env.z_activation` | `relu` |

The SMPL builder changes joint parameters, coordinate layout, and hand geometry at runtime. Retargeting against `final_humanoid.xml` guarantees the retarget target matches the body PHC actually controls.

---

### Task-Space Data Pipeline

Full pipeline from raw AMASS motions to the task-space HDF5 used for EEF-conditioned control and BC datasets.

#### Overview

```
AMASS .npz
    └─► convert_amass_isaac.py ──► accad_isaac_upright_slim.pkl
            └─► filter_motion_pkl.py ──► accad_isaac_upright_slim_research_filtered.pkl
                    └─► extract_taskspace_hdf5.py ──► accad_research_filtered_taskspace.h5
```

---

#### Stage 0 — Raw AMASS input

Each AMASS clip is a dict with:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `pose_aa` / `poses` | `(T, 72)` | float32 | SMPL axis-angle, 24 joints × 3 |
| `trans` | `(T, 3)` | float32 | raw root translation in AMASS frame |
| `betas` / `beta` | `(10,)` | float64 | SMPL shape coefficients |
| `gender` | scalar | str/bytes | `"male"` / `"female"` / `"neutral"` |

---

#### Stage 1 — `scripts/data_process/convert_amass_isaac.py`

Converts raw AMASS data into the poselib/Isaac Gym–compatible format.

**1a. SMPL → MuJoCo joint reordering**

SMPL uses anatomical order; Isaac Gym uses MJCF DFS tree order. A permutation index (`smpl_2_mujoco`) reorders joints accordingly.

```
SMPL order:   Pelvis, L_Hip, R_Hip, Spine1, ...
MuJoCo order: Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, ...
```

**1b. Axis-angle → quaternion**

```
pose_aa_mj  (T, 24, 3)  axis-angle in MuJoCo order
pose_quat   (T, 24, 4)  xyzw quaternion via scipy Rotation.from_rotvec()
```

**1c. Apply Pelvis body-frame offset**

This is the most critical step for FK consistency. The MJCF `Pelvis` body has a frame offset `pos="-0.0018 -0.2233 0.0282"` relative to its joint. The raw AMASS translation must be shifted into this body frame:

```python
root_trans_offset = root_trans + skeleton_tree.local_translation[0]
# local_translation[0] = [-0.0018, -0.2233, 0.0282]
```

Skipping this step causes a constant ~0.22 m Y-offset between HDF5 positions and what Isaac Gym actually renders.

**1d. Upright-start rotation**

AMASS motions are in a camera frame (not Z-up). All global quats are rotated by `sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()` to place the humanoid Z-up at rest.

**Output PKL — per-clip keys:**

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `pose_quat_global` | `(T, 24, 4)` | float32 | global joint rotations (xyzw) |
| `pose_quat` | `(T, 24, 4)` | float32 | local joint rotations (used for Isaac Gym DOF targets) |
| `root_trans_offset` | `(T, 3)` | float64 | pelvis world position (MJCF offset applied) |
| `trans_orig` | `(T, 3)` | float32 | original AMASS translation (reference only) |
| `pose_aa` | `(T, 72)` | float32 | original axis-angle, SMPL order, zero-padded to 72 |
| `beta` | `(10,)` | float64 | shape params (forced to neutral zeros) |
| `gender` | scalar | str | forced `"neutral"` |
| `fps` | scalar | float | hardcoded `30.0` |

---

#### Stage 2 — `scripts/filter_motion_pkl.py`

Removes clips unsuitable for research (dynamics tasks, static poses). Any one of three criteria excludes a clip:

**2a. Name-based exclusion**

```python
EXCLUDE_KEYWORDS = [
    "jump", "hop", "leap", "skip", "cart", "flip",
    "crawl", "crouch", "lie", "roll", "martial",
    "run", "sprint", "jog",
]
```

**2b. Height-variance filter** — catches jumps that slip past the name filter

```
excluded if  max(root_z) - min(root_z)  >  0.3 m
```

**2c. No-movement filter** — removes static/calibration poses

```
excluded if  mean(var(pose_quat_global, axis=time))  <  0.02
```

Result for ACCAD: **143 clips** retained.

---

#### Stage 3 — `scripts/extract_taskspace_hdf5.py`

Runs poselib FK on the filtered PKL to get world-frame joint positions, then extracts task-space features.

**3a. Forward kinematics via poselib**

```python
sk_tree = SkeletonTree.from_mjcf("final_humanoid.xml")

sk_state = SkeletonState.from_rotation_and_root_translation(
    sk_tree,
    clip['pose_quat_global'],   # (T, 24, 4)  global rotations
    clip['root_trans_offset'],  # (T, 3)
    is_local=False
)
joint_positions = sk_state.global_translation.numpy()  # (T, 24, 3)  float32
```

This is **byte-for-byte identical** to the FK Isaac Gym uses — same skeleton tree, same inputs, same call. Any other FK source (e.g. smplx) will produce positions that don't match the live simulation.

**3b. Joint index layout (poselib / MJCF DFS order)**

| Index | Name | Role |
|-------|------|------|
| 0 | Pelvis | root |
| 1–4 | L_Hip … L_Toe | left leg |
| 5–8 | R_Hip … R_Toe | right leg |
| 9–13 | Torso … Head | spine/head |
| 14–18 | L_Thorax … **L_Hand** | left arm — **EEF at 18** |
| 19–23 | R_Thorax … **R_Hand** | right arm — **EEF at 23** |

**3c. Task-space feature layout — `(T, 9)` float32**

| Cols | Feature | Computation |
|------|---------|-------------|
| 0–1 | `root_vel_xy` | central differences on `joint_positions[:, 0, :2]`, `dt = 1/fps` |
| 2 | `pelvis_height` | `joint_positions[:, 0, 2]` (world Z) |
| 3–5 | `left_hand_xyz` | `joint_positions[:, 18, :]` |
| 6–8 | `right_hand_xyz` | `joint_positions[:, 23, :]` |

Boundary frames use forward/backward difference to avoid zero-padding.

---

#### HDF5 file structure

```
accad_research_filtered_taskspace.h5
│
├── attrs:
│     source_file       path to input PKL
│     humanoid_xml      "final_humanoid.xml"
│     fk_backend        "poselib"
│     joint_names       list[24]  (MJCF DFS names)
│     task_space_cols   list[9]   (column names)
│
└── clips/
    └── <clip_name>/           (143 groups)
          ├── task_space        (T, 9)     float32  gzip-4
          ├── joint_positions   (T, 24, 3) float32  gzip-4
          └── attrs:
                clip_name   str
                fps         float  (30.0)
                n_frames    int
```

---

#### Verification & visualisation scripts

| Script | Purpose |
|--------|---------|
| `scripts/extract_taskspace_hdf5.py` | Stage 3 extraction; `--humanoid-xml` selects the MJCF |
| `scripts/filter_motion_pkl.py` | Stage 2 filtering |
| `scripts/verify_taskspace_hdf5.py` | Sanity bounds, cross-check joint_positions vs task_space, temporal consistency, plot |
| `scripts/vis/play_hdf5_clip.py` | Isaac Gym kinematic playback with coloured EEF debug spheres |

**Run extraction:**

```bash
python scripts/extract_taskspace_hdf5.py \
    data/amass/pkls/final_humanoid/accad_isaac_upright_slim_research_filtered.pkl \
    data/amass/hdf5/accad_research_filtered_taskspace.h5 \
    --humanoid-xml final_humanoid.xml
```

**Run verification:**

```bash
python scripts/verify_taskspace_hdf5.py
```

**Visualise a clip in Isaac Gym:**

```bash
python scripts/vis/play_hdf5_clip.py \
    --pkl  data/amass/pkls/final_humanoid/accad_isaac_upright_slim_research_filtered.pkl \
    --hdf5 data/amass/hdf5/accad_research_filtered_taskspace.h5 \
    --clip "ACCAD_Female1Walking_c3d_B1 - stand to walk_poses"
```

Viewer controls: `Space` pause/resume · `←/→` step frame · `R` restart · `Q`/`Esc` quit

---

### OMOMO Retargeting

Converts the OMOMO human–object interaction dataset (167 sequences: suitcase, boxes, monitor, tables, trashcan, chair) from raw SMPL format to the PHC pkl format used by Isaac Gym.

**Input:** `mac_media_sequences.p` — SMPL sequences with `root_orient` (T,3), `pose_body` (T,63), `trans` (T,3), `betas` (1,16), `gender`.

**Output:** PHC pkl with `pose_quat_global` (T,24,4), `pose_quat` (T,24,4), `root_trans_offset` (T,3) as torch tensor, `pose_aa` (T,72), `beta` (10,), `gender`, `fps`.

```bash
# from PHC root, with phc conda env
python scripts/data_process/convert_omomo_to_phc.py \
    --in_file  /home/marmot/mac_motion_seqs/mac_media_sequences.p \
    --out_file data/amass/pkls/omomo_locomanipulation_crouch.pkl

# extract task-space HDF5
python scripts/extract_taskspace_hdf5.py \
    data/amass/pkls/omomo_locomanipulation_crouch.pkl \
    data/amass/hdf5/omomo_locomanipulation_crouch_taskspace.h5 \
    --humanoid-xml final_humanoid.xml

# track in Isaac Gym
python phc/run_hydra.py \
    learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp robot=smpl_humanoid \
    robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=data/amass/pkls/omomo_locomanipulation_crouch.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1
```

---

### X16 Receding-Horizon Flow Model

End-to-end pipeline for the current **X16** receding-horizon flow model: collect physics rollouts with PHC, filter + convert them into the training HDF5, train the flow model (in the `flow_matching` repo), then deploy it back into PHC as a real-time closed-loop reference generator.

Everything runs in **joint-position (keypoint) space** — the flow model outputs `24×3` joint positions and PHC's `obs_v=7` tracker consumes them directly as `ref_rb_pos` (the server's `j3d`); there is no joint-angle IK in the loop.

```
AMASS / source motions (.pkl)
   │  1. PHC physics-rollout collection       (run_hydra, env.collect_physics_rollouts=True)
   ▼
raw rollout HDF5  output/debug/physics_rollouts_*.h5      (rollouts/<id>/body_pos …)
   │  2. keep clips PHC tracked successfully  (scripts/filter_rollout_h5_success.py)
   ▼
filtered rollout HDF5  *_physics_rollout.h5               (clips/<clip>/body_pos …)
   │  3. convert to task-space training data  (flow_matching: convert_physics_rollout_to_taskspace.py)
   ▼
training HDF5  final_h5_files/physics_rollout/*_physics.h5
        (clips/<clip>/{joint_positions (T,24,3), task_space (T,9), pose_aa_root (T,3)})
   │  4. train X16 self / cross               (flow_matching: train_horizon_x15.py)
   ▼
checkpoint  best.pt + stats.json
   │  5. deploy: scripts/flow_server_x15.py  ⇄  PHC closed-loop tracker
   ▼
real-time motion on the humanoid
```

#### 1. Collect physics rollouts (PHC)

Roll out the PHC tracking policy on the source motions and record the joint positions it actually achieves — physically feasible motion, not raw mocap. Setting `env.collect_physics_rollouts=True` routes `run()` into collection mode.

```bash
python phc/run_hydra.py \
    learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.collect_physics_rollouts=True \
    env.physics_rollout_out=output/debug/physics_rollouts_my_run.h5 \
    env.physics_rollout_frames=300 \
    env.num_envs=20 headless=True test=True epoch=-1
```

`physics_rollout_out` is the output HDF5, `physics_rollout_frames` caps frames per clip, and `env.physics_rollout_filter_hdf5=<path>` restricts collection to the clip keys present in an existing HDF5.

#### 2. Filter to successfully tracked clips (PHC)

Drop clips where the policy fell or the physics correction was too large to be useful training data.

```bash
python scripts/filter_rollout_h5_success.py \
    output/debug/physics_rollouts_my_run.h5 \
    sample_data/amass_isaac_standing_upright_slim.pkl \
    my_run_physics_rollout.h5 \
    --height-thresh 0.4 --mean-thresh 0.10 --max-thresh 0.30
```

#### 3. Convert to the task-space training format (flow_matching repo)

Maps `body_pos → joint_positions` and derives the 9-D task-space command (`[vx, vy, pelvis_z, l_hand_xyz, r_hand_xyz]`).

```bash
python examples/joint_flow/scripts/convert_physics_rollout_to_taskspace.py \
    my_run_physics_rollout.h5 \
    final_h5_files/physics_rollout/my_run_physics.h5 \
    --fps 30.0
```

#### 4. Train X16 (flow_matching repo)

Both variants share the same data and differ only in `--arch` (and `--outdir`). See the `flow_matching` README for the full flag reference.

```bash
python examples/joint_flow/train_horizon_x15.py \
    --hdf5 final_h5_files/physics_rollout/my_run_physics.h5 \
    --arch cross_transformer \
    --hidden 512 --layers 8 --heads 8 --past-frames 2 --horizon 15 \
    --mirror-sources 0 \
    --outdir checkpoints/joint_flow_x16c_cross_h15 \
    --epochs 100 --batch 2048 --lr 3e-4 --amp
```

Use `--arch history_transformer` for X16 self.

#### 5. Deploy — `scripts/flow_server_x15.py` (X16 closed-loop server)

At deployment the trained model runs as an HTTP server that PHC / Isaac Gym polls at 30 Hz. The flow model is the **reference generator**; PHC is the **physics tracker**. The loop is closed: PHC pushes its *actual tracked pose* back to the server every tick, and each new horizon is conditioned (re-anchored) on the last two tracked frames — not on the model's own predictions — so tracking drift is corrected every stride.

**Design flow**

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Planner: task_phase · desired_heading · active_goal  (keyboard/REST)  │
  └───────────────────────────────────┬──────────────────────────────────┘
                                       │ command
                                       ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ FLOW SERVER  (scripts/flow_server_x15.py, 30 Hz)                      │
  │                                                                      │
  │   last 2 tracked frames ─┐                                           │
  │                          ▼                                           │
  │    cond = [ past×2 | phase(5) | heading(2) | goal(3) ]   (162-D)     │
  │                          │                                           │
  │                  x16 / x16c flow model    (10 ODE steps, ~11 ms GPU) │
  │                          │                                           │
  │                          ▼                                           │
  │             horizon H=15  →  serve first stride=7 frames             │
  │                          │   (one generated frame per 33.3 ms tick)  │
  └──────────────────────────┼───────────────────────────────────────────┘
        ▲                     │ GET /get_pose  (generated reference, 24×3)
        │ POST /push_state    ▼
        │ (tracked pose)    ┌──────────────────────────────────────────────┐
        └───────────────────┤ PHC / Isaac Gym tracker (HumanoidImMCPDemo)  │
                            │ each tick: push tracked → get ref → track    │
                            └──────────────────────────────────────────────┘
```

**Bootstrap (startup).** The server has no robot feedback yet, so it holds the default standing pose (`stats["default_state"]`); `GET /get_pose` returns stand. When PHC starts it tracks that stand pose and pushes its own tracked state every tick. After `K=2` tracked frames arrive (~2 ticks, ~67 ms), the first inference fires. There is no deadlock — stand is the bootstrap target that bridges "server waits for tracker" and "tracker needs a target".

**Steady state (per stride).**

1. One inference conditions on the **last 2 tracked frames**, generates `H=15`, and buffers the first `stride=7` frames (~233 ms of motion).
2. The 7 frames are served one per 33.3 ms tick. PHC tracks each generated reference in physics and pushes its tracked pose back via `/push_state`.
3. When the buffer empties, the server re-plans off the **latest 2 tracked frames**, re-anchoring the next horizon to the robot's real pose.

`GET /get_pose` always returns the flow model's generated reference (or the held stand pose while idle); the tracked pose only ever re-enters as conditioning for the next inference. Planning is strictly closed-loop — with no PHC pushing state, the server stays idle on stand.

**Running it (two terminals).** The server auto-detects the architecture from `stats.json`, so the same command runs X16 self or X16 cross — just point at the checkpoint dir.

```bash
# Terminal 1 — flow server (X16 cross)
python scripts/flow_server_x15.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_x16c_cross_h15/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_x16c_cross_h15/stats.json

# X16 self — swap the checkpoint dir:
python scripts/flow_server_x15.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_x16_history_physics30_h15_policy30/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_x16_history_physics30_h15_policy30/stats.json
```

Server keyboard controls: `1`/`2` = stand/locomotion, `W`/`A`/`S`/`D` = set heading (locomotion), `Space` = face forward + stand, `Esc` = quit. Commands can also be set programmatically: `curl 'http://localhost:8080/set_cmd?phase=loco&heading=0.0'`.

```bash
# Terminal 2 — PHC tracker (obs_v=7 enables the closed-loop push/get path)
python phc/run_hydra.py \
    learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1
```

Start the server first and wait for it to print `done.` (it warms up for ~1 s before opening the REST port), then launch PHC.

---

### Flow Model Integration

Scripts for running a flow model alongside PHC — generating poses from the flow model and tracking them with the PHC RL policy.

#### `scripts/flow_to_pkl.py` — Generate a PHC-trackable PKL from the flow model

Runs the no-EEF flow model on a commanded velocity trajectory, retargets the predicted joint positions to SMPL `pose_aa` via gradient-fit FK (see `flow_retarget.py`), and saves a PKL that PHC can track directly.

```bash
python scripts/flow_to_pkl.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json \
    --vel-x 0.5 --vel-y 0.0 --height 0.9 \
    --duration 4.0 \
    --out data/generated/walk_forward.pkl

# then track:
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp robot=smpl_humanoid \
    robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=data/generated/walk_forward.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1
```

#### `scripts/flow_server.py` — Real-time flow pose server

Runs the flow model as an HTTP server; PHC polls `GET /get_pose` every sim step. Keyboard controls (W/S/A/D/Q/E/Space) adjust commanded velocity and pelvis height in real time.

```bash
# Terminal 1 — start server
python scripts/flow_server.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json

# Terminal 2 — run PHC in demo mode
python phc/run_hydra.py \
    learning=im_mcp exp_name=phc_kp_mcp_iccv \
    env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
    robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
    env.z_activation=relu env.obs_v=7 \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
    env.num_envs=1 headless=False test=True epoch=-1
```

#### `scripts/flow_replay.py` — Joint-sphere replay (no policy)

Visualises raw flow model joint positions as spheres in Isaac Gym — bypasses IK and PHC entirely. Useful for sanity-checking flow output before retargeting.

```bash
python scripts/flow_replay.py \
    --flow-ckpt  ../generation/checkpoints/joint_flow_noeef_prevframe/best.pt \
    --flow-stats ../generation/checkpoints/joint_flow_noeef_prevframe/stats.json \
    --vel-x 0.4 --vel-y 0.0 --height 0.9 --duration 5.0
```

#### `scripts/flow_replay_smpl.py` — Articulated SMPL kinematic replay

Drives the full SMPL humanoid kinematically from retargeted flow output (no policy/physics). Articulated-body analogue of `flow_replay.py`.

```bash
# generate + replay
python scripts/flow_replay_smpl.py --vel-x 0.4 --vel-y 0.0 --height 0.9 --duration 5.0

# replay an existing pkl
python scripts/flow_replay_smpl.py --motion-file data/generated/walk_forward_04.pkl
```

#### `scripts/flow_retarget.py` — Gradient-fit retargeter

Not called directly — imported by `flow_to_pkl.py`, `flow_server.py`, and `flow_replay_smpl.py`. Fits `pose_aa` by gradient descent against target joint positions using PHC's differentiable FK (`Humanoid_Batch`), with a pose prior and temporal smoothing. Produces anatomically consistent, smooth rotations that PHC can track without flailing.

Two entry points: `fit_sequence(...)` for offline batch fitting, `StreamFitter` for per-frame warm-started real-time use.

#### `scripts/flow_compare_retarget_replay.py` — Side-by-side retarget comparison

Replays two SMPL humanoids in the same viewer: one retargeted with `fit_sequence` (offline, high quality) and one with `StreamFitter` (real-time style). Useful for validating that the streaming retargeter matches offline quality.

```bash
python scripts/flow_compare_retarget_replay.py \
    --vel-x 0.4 --height 0.9 --duration 50.0 --fps 30
```

#### `scripts/patch_hdf5_add_root.py` — Backfill `pose_aa_root` into 6D HDF5s

Adds `pose_aa_root` (T,3) to existing 6D taskspace HDF5 files that were extracted before root prediction was added. Reads the pelvis joint quaternion from the source PKL.

```bash
python scripts/patch_hdf5_add_root.py \
    data/amass/hdf5/accad_6d_taskspace.h5 \
    data/amass/hdf5/accad_6d_root_taskspace.h5 \
    --pkl data/amass/pkls/final_humanoid/accad_isaac_upright_slim_research_filtered.pkl

# omit --pkl to auto-read source_file from HDF5 attrs
python scripts/patch_hdf5_add_root.py \
    data/amass/hdf5/cmu_6d_taskspace.h5 \
    data/amass/hdf5/cmu_6d_root_taskspace.h5
```

---

# Table of Contents
* [News 🚩](#news-)
* [TODOs](#todos)
* [Introduction](#introduction)
    * [Docs](#docs)
    * [Current Results on Cleaned AMASS (11313 Sequences)](#current-results-on-cleaned-amass-11313-sequences)
    * [Dependencies](#dependencies)
* [Evaluation](#evaluation)
    * [Minimal Viable Evaluation](#minimal-viable-evaluation)
    * [Viewer Shortcuts](#viewer-shortcuts)
    * [Imitation](#imitation)
    * [In-the-wild Avatar Control](#in-the-wild-avatar-control)
    * [VR Controller Tracking](#vr-controller-tracking)
* [Training](#training)
    * [Data Processing AMASS](#data-processing-amass)
    * [Training PHC](#training-phc)
* [Research Workflow](#research-workflow)
* [Trouble Shooting](#trouble-shooting)
    * [Multiprocessing Issues](#multiprocessing-issues)
    * [Success Rate](#success-rate)
* [Citation](#citation)
* [References](#references)

## News 🚩

[August 21, 2025] Adding sample eval code in IsaacLab. PHC policy can directly be inferred in IsaacLab. `python scripts/eval_in_isaaclab.py`. 

[December 10, 2024] Release process for generating offline dataset (PHC_Act) for offline RL or behavior cloning (developed by [@kangnil](https://github.com/kangnil)). See [docs/offline_dataset.md](docs/offline_dataset.md) for more details. 

[December 9, 2024] Release retargeting [documentation](docs/retargeting.md) (for retargeting to your own humanoids using SMPL data). 

[October 8, 2024] Release support for Unitree H1 and G1 humanoid. 

[August 30, 2024] Release support for SMPLX humanoid. 

[Apirl 5, 2024] Upgrading to use [SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) to automatically create the SMPL humanoid. Please run `pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master` to SMPLSim. SMPLSim requires python 3.8; for python 3.7, please `pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@isaac37`.

[Feburary 28, 2024] Releasing Auto-PMCP procedure to train single primitives; this procedure can lead to very high-performant imiatators without PNN, though it wouldn't have the failure state recovery capability. See the `auto_pmcp_soft` flag. 

[Feburary 19, 2024] Releasing PHC+ model (100% success rate on AMASS) used in [PULSE](https://zhengyiluo.github.io/PULSE/).

[Feburary 17, 2024] Fixed a bug when overhauling the system to Hydra. Please pull the newest version :). 

[Feburary 1, 2024] Overhauling the config system to Hydra. 

[January 8, 2024] Support for running inference without SMPL model. 

[January 7, 2024] Release language-to-control demo (based on MDM). 

[December 19, 2023] Release VR controller tracking code. 

[December 14, 2023] Release webcam video-based control demo. 

[October 31, 2023] Remove dependency on mujoco 210 and update to the newest mujoco version (for creating xml robot; no more downloads and direct install with `pip install mujoco`!). Updated amass_occlusion_v3 to 11313 sequences for training (was 11315). Updated requirement.txt. 

[October 25, 2023] Training and Evaluation code released.


## TODOs

- [x] Add support for Unitree H1 & G1.

- [x] Add support for smplx/h (fingers!!!).

- [x] Release PHC+ model (100% success rate on AMASS) used in [PULSE](https://zhengyiluo.github.io/PULSE/). 

- [x] Release language-based demo code.  

- [x] Release vr controller tracking code.  

- [x] Release video-based demo code.  

- [x] Additional instruction on Isaac Gym SMPL robot. 

- [x] Release training code. 

- [x] Release evaluation code. 

## Introduction
We present a physics-based humanoid controller that achieves high-fidelity motion imitation and fault-tolerant behavior in the presence of noisy input (e.g. pose estimates from video or generated from language) and unexpected falls. Our controller scales up to learning ten thousand motion clips without using any external stabilizing forces and learns to naturally recover from fail-state. Given reference motion, our controller can perpetually control simulated avatars without requiring resets. At its core, we propose the progressive multiplicative control policy (PMCP), which dynamically allocates new network capacity to learn harder and harder motion sequences. PMCP allows efficient scaling for learning from large-scale motion databases and adding new tasks, such as fail-state recovery, without catastrophic forgetting. We demonstrate the effectiveness of our controller by using it to imitate noisy poses from video-based pose estimators and language-based motion generators in a live and real-time multi-person avatar use case.

> ❗️❗️❗️Notice that the current released models used a different coordinate system as SMPL (with negative z as gravity direction), and the humanoid is modifed in a way such that it is facing positive x direction (instead of the original SMPL facing). This is reflected in a "up_right_start" flag in the humanoid robot (`smpl_local_robot.py`) configuration. This is done to make the humanoid's heading to be eailerly defined and flipping left and right easier, but would require further modification for converting back to SMPL (which is provided in the code). In the future I am working towards removing this modification. 

> ❗️❗️❗️ Another notice is that while the MCP/Mixture of expert model is great for achieving high success rate, it is not absolutely necessary for the PHC to work. The PHC can work with a single primitive model and achieves high success rate; though it wouldn't have the failure state recovery capability.

### Docs 
- [Docs on SMPL_Robot](docs/smpl_robot_instruction.MD)
- [Docker Instructions](docs/docker_instruction.MD) (from [@kexul](https://github.com/kexul))
- [Webcam Demo](docs/video_to_control_demo.md) 
- [Language Demo](docs/language_to_control_demo.md) 
- [Retargeting to Your Own Humanoids](docs/retargeting.md)
- [Offline Dataset (PHC_Act)](docs/offline_dataset.md)

### Current Results on Cleaned AMASS (11313 Sequences)

All evaluation is done using the mean SMPL body pose and adjust the height, using the same evaluation protocal as in [UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl/tree/master). Noticed that different evaluation protocal will lead to different results, and Isaac gym itself can lead to (slightly) different results based on batch size/machine setup. 

| Models         | Succ | G-MPJPE | ACC |
|----------------|:----:|:------------:|:----:|
| PHC           | 98.9% |     37.5     | 3.3 |
| PHC-KP          | 98.7% |     40.7     | 3.5 |
| PHC+ [in Pulse](https://zhengyiluo.github.io/PULSE/)  | 100% |     26.6     | 2.7 |
| PHC-Prim (single primitive) | 99.9% |     25.9     | 2.3 |
| PHC-Fut (using future) | 100% |     25.3     | 2.5 |
| PHC-X-Prim (single primitive) | 99.9% |     24.7     | 3.6 |

### Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```


4. Use the following script to download trained models and sample data.

```
bash download_data.sh
```

this will download amass_isaac_standing_upright_slim.pkl, which is a standing still pose for testing. 

To evaluate with your own SMPL data, see the script `scripts/data_process/convert_data_smpl.py`. Pay speical attention to make sure the coordinate system is the same as the one used  in simulaiton (with negative z as gravity direction). 

Make sure you have the SMPL paramters properly setup by running the following scripts:
```
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

The SMPL model is used to adjust the height the humanoid robot to avoid penetnration with the ground during data loading. 


## Evaluation 


### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| r | reset episode |
| j | apply large force to the humanoid |
| l | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| m | cancel termination based on imitation |

... more shortcut can be found in `phc/env/tasks/base_task.py`

Notes on rendering: I am using pyvirtualdisplay to record the video such that you can see all humanoids at the same time (default function will only capture the first environment). You can disable it using the flag `no_virtual_display=True`. 

You can use the `render_o3d=True no_virtual_display=True` flag to render the SMPL mesh together with your Isaac Gym simulation in real time like this: 
<div float="center">
  <img src="assets/open3d.gif" />
</div>

to do the above visulaization, press m (to cancel termination based on imitation), and then press j (to apply a large force to the humanoid). 

### Imitation 

#### SMPL and SMPL-X

Pretrained checkpoint compatibility notes:

- When evaluating a released checkpoint, keep the checkpoint-specific overrides from the example command. If these flags are omitted, Hydra falls back to the default robot/env configs, which can noticeably hurt performance.
- `phc_x_pnn` expects `robot=smplx_humanoid env=env_im_x_pnn learning=im_pnn_big`.
- `phc_comp_kp_2` expects `env.obs_v=7 env.num_prim=3 robot.real_weight_porpotion_boxes=False learning.params.network.ending_act=False`.
- `phc_3` expects `env=env_im_getup_mcp learning=im_mcp_big env.num_prim=3 env.zero_out_far=False robot.real_weight_porpotion_boxes=False`.
- `phc_shape_mcp_iccv` expects `robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.z_activation=relu`.
- `phc_kp_mcp_iccv` expects `robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.obs_v=7`.
- `phc_shape_pnn_iccv` expects `robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False`.
- `phc_kp_pnn_iccv` expects `robot.freeze_hand=True robot.box_body=False env.obs_v=7`.
- In particular, `robot.freeze_hand` defaults to `False` in the robot configs and `env.z_activation` defaults to `silu` in `env_im_getup_mcp`, so those two settings should be overridden when loading `phc_shape_mcp_iccv` or `phc_kp_mcp_iccv`.

PHC-X-Prim (single primitive model)
```
python phc/run_hydra.py learning=im_pnn_big exp_name=phc_x_pnn  env=env_im_x_pnn robot=smplx_humanoid env.motion_file=sample_data/standing_x.pkl env.training_prim=0 epoch=-1 test=True  env.num_envs=1  headless=False
```

PHC+: keypoint model, can getup from the ground and walk back) | Best model for video/language model
```
python phc/run_hydra.py learning=im_mcp_big learning.params.network.ending_act=False exp_name=phc_comp_kp_2 env.obs_v=7 env=env_im_getup_mcp robot=smpl_humanoid robot.real_weight_porpotion_boxes=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth'] env.num_prim=3 env.num_envs=1  headless=False epoch=-1 test=True
```

PHC+: rotation + keypoint model, can getup from the ground but not walk back -- model used in PULSE due to time constraint
```
python phc/run_hydra.py learning=im_mcp_big  exp_name=phc_comp_3 env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1  headless=False epoch=-1 test=True
```

Evaluate full model:

```
## Shape + rotation + keypoint model

python phc/run_hydra.py learning=im_mcp exp_name=phc_shape_mcp_iccv test=True env=env_im_getup_mcp robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] env.num_envs=1  headless=False epoch=-1



## keypoint model
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv  test=True env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1

```

Evaluate on AMASS:

```
## rotation + keypoint model (100% - PHC+)
python phc/run_hydra.py learning=im_mcp_big  exp_name=phc_comp_3 env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1  headless=False im_eval=True


## Shape + rotation + keypoint model
python phc/run_hydra.py learning=im_mcp exp_name=phc_shape_mcp_iccv epoch=-1 test=True env=env_im_getup_mcp robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth'] env.num_envs=1  headless=False im_eval=True


## keypoint model
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv epoch=-1 test=True env=env_im_getup_mcp robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1024 env.obs_v=7  im_eval=True
```

Evaluate single primitive model:

```
## Shape + rotation + keypoint model
python phc/run_hydra.py learning=im_pnn exp_name=phc_shape_pnn_iccv epoch=-1 test=True env=env_im_pnn robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl  env.num_envs=1  headless=False


## keypoint model
python phc/run_hydra.py learning=im_pnn exp_name=phc_kp_pnn_iccv epoch=-1 test=True env=env_im_pnn env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl robot.freeze_hand=True robot.box_body=False env.num_envs=1 env.obs_v=7  headless=False
```

### In-the-wild Avatar Control

See [Webcam Demo](docs/video_to_control_demo.md)

The tldr is run:

```
python scripts/demo/video_to_pose_server.py
```

then

```
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True

```


See [Language-to-motion Demo](docs/language_to_control_demo.md)

```
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True

```

### VR Controller Tracking

```
python phc/run_hydra.py learning=im_big exp_name=phc_prim_vr env=env_vr robot=smpl_humanoid robot.box_body=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.num_envs=1 headless=False epoch=-1 test=True no_virtual_display=True
```



## Training



### Data Processing AMASS

We train on a subset of the [AMASS](https://amass.is.tue.mpg.de/) dataset.

For processing the AMASS, first, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:

`python scripts/data_process/convert_amass_data.py`

### Training PHC

#### H1 and G1

```
python phc/run_hydra.py project_name=Robot_IM robot=unitree_g1 env=env_im_g1_phc env.motion_file=sample_data/dance_sample_g1.pkl learning=im_pnn_big exp_name=unitree_g1_pnn sim=robot_sim control=robot_control learning.params.network.space.continuous.sigma_init.val=-1.7 
```

```
python phc/run_hydra.py  project_name=Robot_IM   robot=unitree_h1      env=env_im_h1_phc env.motion_file=sample_data/sample_dance_h1.pkl learning=im_pnn_big   exp_name=unitree_h1_pnn_realsim_092924 sim=robot_sim control=robot_control learning.params.network.space.continuous.sigma_init.val=-1.7 
```

After training, you can run eval by adding `num_threads=1 headless=False test=True epoch=-1` to the above scripts. 

#### Train single primitive

```
python phc/run_hydra.py learning=im_big exp_name=phc_prim env=env_im robot=smpl_humanoid env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl  
```


#### Train full PNN model 


Training PHC is not super automated yet, so it requires some (a lot of) manual steps, and invovles changing the config file a couple of times during training based on the training phase. The `phc_shape_pnn_train_iccv.yaml` config file provides a starting point for training primitives. 

First, we will train one primitive, and keep an eye on its performance (--has_eval) flag. In the config, the "training_prim" is the primitive that is being trained. This need to be updated accordingly. 

```
python phc/run_hydra.py learning=im_pnn_big env=env_im_pnn robot=smpl_humanoid env.motion_file=[motion_file] exp_name=[exp_name] 
```

After the performance plateaus, we will dump the most recent sequences that the primitives has failed on, and use them to train the next primitive. Here idx is the primitive that should be trained. 

```
python scripts/pmcp/forward_pmcp.py --exp [exp_name] --epoch [epoch] --idx {idx}
```
The above script will dump two files: one is the next hard sequences to learn, and anothere one is the checkpoint to resume with the copied primitive. 

To train the next primitive, run teh following script:
```
python phc/run_hydra.py learning=im_pnn_big env=env_im_pnn robot=smpl_humanoid env.motion_file=[motion_file] epoch=[include epoch+1 from previous step] env.fitting=True env.training_prim=1
```

Repeat this process until no hard sequences are left. Then, to train the fail-state recovery primitive on simple locomotion data. 


```
python phc/run_hydra.py learning=im_pnn_big exp_name=phc_shape_pnn_iccv env=env_im_pnn robot=smpl_humanoid env.motion_file=[motion_file] epoch=[include current epoch + 1 from the forward_pmcp step] env.fitting=True env.training_prim=[+1] env.zero_out_far=True env.zero_out_far_train=True env.getup_udpate_epoch={epoch}
```

After all primitives are trained, train the composer: 

```
python phc/run_hydra.py learning=im_mcp_big exp_name=[] env=env_im_getup_mcp robot=smpl_humanoid env.motion_file=[motion_file] env.models=['output/HumanoidIm/{exp_name}/Humanoid.pth']
```

When training the composer, you can repeat the process above (progressive mining hard sequences) to improve performance. 


You can also just train one model for imitation (no PNN):
```
python phc/run_hydra.py learning=im exp_name=phc_prim_iccv  env=env_im robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl
```

---

## Trouble Shooting


### Multiprocessing Issues
See [this issue](https://github.com/ZhengyiLuo/PerpetualHumanoidControl/issues/17) for some discusssions. 

For the data loading part, try use: 

at [this line](https://github.com/ZhengyiLuo/PerpetualHumanoidControl/blob/8e01930fbcaa3efb9fb8b4752f2cf52f41dfe260/phc/utils/motion_lib_base.py#L235), bascially, uncomment: 

```
mp.set_sharing_strategy('file_system')
```

which should fix the issue. Though using file_system has caused me problems before as well.


### Success Rate
The success rate is reported as "eval_success_rate" in the wandb logging, not the "success_rate", which is a episodic success rate used during training. 


## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            
```

Also consider citing these prior works that are used in this project:

```
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

## References
This repository is built on top of the following amazing repositories:
* Main code framework is from: [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Part of the SMPL_robot code is from: [UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)

Please follow the lisence of the above repositories for usage. 
