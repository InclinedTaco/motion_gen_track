> Historical research notes from the source repository. Start with [the unified setup](../docs/setup.md). The referenced `scripts/horizon_flow_to_pkl.py` was absent from the source snapshot and is not a supported command here.

# PHC Flow Fine-Tuning

This repo can fine-tune PHC against a frozen horizon flow model.  The training
setup starts each episode from the normal PHC standing reset, serves stand
references for the first few steps, then switches the flow command phase to
locomotion.  It does not warm-start the flow from a walking clip on reset.

## Online Flow Fine-Tuning Command

Use the in-process flow path for RL.  It runs one batched ODE solve for all
envs at each flow replanning step, keeps separate robot/prediction histories per
env, and avoids the single-stream Flask server bottleneck.  This is not 1024
separate flow servers or 1024 model copies; it is one frozen flow model queried
with a batch of per-env conditions.

For the current task-only flow fine-tuning run, prefer the bundled Hydra config
instead of the long override command:

```bash
cd <PROJECT_ROOT>/tracking
python phc/run_hydra.py --config-name flow_phc_x16f_task_only
```

This config resumes the trained MCP composer from
`output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth`, loads the frozen PNN primitive
bank from `output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth`, enables softmax over
the primitive weights, and uses the frozen x16f flow model as the online
reference generator.  It disables the problematic getup distance branch
(`env.zero_out_far=False`) and the power penalty (`env.power_reward=False`) so
the task reward is the online flow imitation reward.

The PPO reward is task-only:

```yaml
learning.params.config.task_reward_w: 1.0
learning.params.config.disc_reward_w: 0.0
learning.params.config.disc_coef: 0.0
```

`disc_reward_w=0.0` removes AMP discriminator reward from PPO returns, and
`disc_coef=0.0` prevents the discriminator loss from driving the update.  The
flow checkpoint remains frozen; fine-tuning updates the MCP composer policy,
critic, and optimizer state.

For a 10-env visible smoke test, reduce the PPO minibatches to match
`10 envs * horizon_length 32 = 320`:

```bash
cd <PROJECT_ROOT>/tracking
python phc/run_hydra.py --config-name flow_phc_x16f_task_only \
  env.num_envs=10 \
  learning.params.config.minibatch_size=320 \
  learning.params.config.amp_minibatch_size=320 \
  headless=False
```

`env.episode_length=300` is a 10 second episode because the simulator runs at
60 Hz with `controlFrequencyInv=2`, so PHC acts at 30 Hz.

The flow curriculum in this config uses the task's own `_flow_episode_idx`,
which starts at 0 for a new run even when the MCP policy checkpoint was saved at
a high PPO epoch.  The feedback source therefore ramps as intended: prediction
history until episode index 100, linearly toward robot feedback through 700, and
full robot-feedback conditioning after 700.

```bash
cd <PROJECT_ROOT>/tracking
python phc/run_hydra.py \
  learning=im_mcp \
  exp_name=flow_phc_walk_curriculum_x16f \
  project_name=PHC \
  no_log=False \
  env=env_im_getup_mcp \
  env.task=HumanoidImMCPDemo \
  env.obs_v=7 \
  env.episode_length=300 \
  env.num_envs=1024 \
  env.getup_schedule=False \
  +env.flow_ref_mode=inprocess \
  +env.flow_ckpt=../generation/checkpoints/joint_flow_x16f_phase_surviv_omomo_slowdrop_dropout_sched_h15_coldpad30_drop30/best.pt \
  +env.flow_stats=../generation/checkpoints/joint_flow_x16f_phase_surviv_omomo_slowdrop_dropout_sched_h15_coldpad30_drop30/stats.json \
  +env.flow_feedback_source=curriculum \
  +env.flow_reset_history_on_env_reset=True \
  +env.flow_enable_tracking_reset=True \
  +env.flow_tracking_reset_disable_collision=False \
  +env.flow_stand_steps=90 \
  +env.flow_stand_steps_min=45 \
  +env.flow_stand_steps_max=120 \
  +env.flow_hold_stand_ref=False \
  +env.flow_random_heading=True \
  +env.flow_heading_min=-3.14159265359 \
  +env.flow_heading_max=3.14159265359 \
  +env.flow_heading_change_steps_min=120 \
  +env.flow_heading_change_steps_max=240 \
  +env.flow_heading_turn_rate=0.01745329252 \
  +env.align_robot_start_to_flow_heading=True \
  +env.flow_curriculum_warmup_epochs=100 \
  +env.flow_curriculum_ramp_end_epochs=700 \
  +env.flow_stride=5 \
  +env.flow_ode_steps=10 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  robot=smpl_humanoid \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.z_activation=relu \
  learning.params.config.max_epochs=1000 \
  headless=True \
  test=False
```

For online-flow fine-tuning, each Isaac env owns an independent in-process flow
stream.  `flow_reset_history_on_env_reset=True` resets that stream from
`stats["default_state"]` whenever the PHC env resets.  The first part of each
episode samples stand-phase flow references, with stand duration randomized by
`flow_stand_steps_min/max`; after that env's stand window, the command switches
to locomotion.  `flow_random_heading=True` gives each env its own heading
command, changes targets only every `flow_heading_change_steps_min/max` frames,
and limits the turn rate with `flow_heading_turn_rate` so turns are gradual
rather than abrupt.  `flow_feedback_source=curriculum` starts training from
prediction history and gradually shifts to robot-state feedback between
`flow_curriculum_warmup_epochs` and `flow_curriculum_ramp_end_epochs`.

If 1024 envs does not fit with the flow model resident on the same GPU, lower
`env.num_envs` first.  Do not use `env.num_envs=1` for real PPO training; that
is only useful for smoke tests.

### Fine-Tuning Flag Reference

Core Hydra/RL flags:

- `learning=im_mcp`: trains the PHC imitation/MCP policy.
- `env=env_im_getup_mcp`: uses the getup-capable imitation environment config.
- `env.task=HumanoidImMCPDemo`: switches the task to the online-flow reference
  path.
- `env.obs_v=7`: uses body-position reference observations, which are what the
  online flow buffer provides.
- `env.num_envs=1024`: runs 1024 Isaac Gym envs.  Lower this first if GPU memory
  is tight.
- `env.episode_length=300`: caps each rollout at 300 simulator frames.
- `headless=True`: required for practical RL training throughput.
- `test=False`: enables training rather than viewer/eval behavior.
- `env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl`: keeps the
  underlying PHC/MotionLib machinery initialized, even though online flow refs
  are used for `obs_v=7`.
- `"env.models=[...Humanoid.pth]"`: initializes from the existing PHC checkpoint.
- `learning.params.config.max_epochs=1000`: PPO training length.

Flow model loading:

- `+env.flow_ref_mode=inprocess`: loads the flow model inside the Isaac task and
  batches flow queries across envs.
- `+env.flow_ckpt=.../best.pt`: frozen flow model checkpoint.
- `+env.flow_stats=.../stats.json`: normalization, default state, horizon length,
  past-frame count, and task schema for the flow model.
- `+env.flow_stride=5`: PHC consumes 5 generated frames before the env needs a new
  flow horizon.  Each env has its own cursor, so staggered resets do not force
  all envs to skip or restart.
- `+env.flow_ode_steps=10`: ODE solver steps for each flow sample.  Lowering this
  reduces compute at the cost of generation quality.

Per-env reset and history behavior:

- `+env.flow_reset_history_on_env_reset=True`: whenever an Isaac env resets, only
  that env's flow history and prediction history are reset to
  `stats["default_state"]`.
- `+env.flow_enable_tracking_reset=True`: enables early reset when the robot
  deviates too far from the online flow reference.
- `+env.flow_tracking_reset_disable_collision=False`: keeps online-flow tracking
  reset active even though the demo path disables collision-style debug resets.
- Reset conditions in this mode are episode timeout, optional online tracking
  failure, and replay-loop end when `flow_replay_file` is used.

Stand-to-locomotion schedule:

- `+env.flow_stand_steps=90`: default stand duration in frames.
- `+env.flow_stand_steps_min=45` and `+env.flow_stand_steps_max=120`: per-env
  random stand duration sampled on reset.
- `+env.flow_hold_stand_ref=False`: PHC tracks flow-generated stand references
  from frame 0.  Set this to `True` only when you want to hold the robot's current
  body pose instead of querying flow during stand.

Heading schedule:

- `+env.flow_random_heading=True`: samples an independent commanded walk heading
  for each env.
- `+env.flow_heading_min` / `+env.flow_heading_max`: heading sample range in
  radians.
- `+env.flow_heading_change_steps_min=120` and
  `+env.flow_heading_change_steps_max=240`: target heading changes only every
  4-8 seconds at 30 FPS, so the flow is not constantly turning.
- `+env.flow_heading_turn_rate=0.01745329252`: maximum heading change per frame
  in radians; this value is 1 degree/frame.
- `+env.align_robot_start_to_flow_heading=True`: rotates the PHC reset pose so
  the robot initially faces the same heading that the flow model is commanded to
  walk.

Feedback curriculum:

- `+env.flow_feedback_source=curriculum`: starts by conditioning flow on its own
  prediction history, then gradually switches to robot-state feedback.
- `+env.flow_curriculum_warmup_epochs=100`: before this many reset-equivalent
  epochs, the curriculum keeps using prediction history.
- `+env.flow_curriculum_ramp_end_epochs=700`: by this point, the curriculum uses
  robot feedback whenever prediction history is available.

## Flow Tracking Modes

These commands compare the same flow model across offline generation, normal
offline PHC tracking, online replay of an offline sequence, and live online
generation.  They are useful for isolating whether a tracking issue comes from
the generated motion, PHC's offline `MotionLib` path, the online reference
buffer, or live flow feedback.

Set the checkpoint directory once:

```bash
cd <PROJECT_ROOT>/tracking
CKPT_DIR=../generation/checkpoints/joint_flow_x16f_phase_surviv_omomo_slowdrop_dropout_sched_h15_coldpad30_drop30
```

### 1. Generate A 30s Offline Constant Walk

This runs the flow model autoregressively for 900 frames at the PHC convention
of 30 FPS.  It saves both raw decoded flow joints and PHC-loadable PKLs:

- `*_raw_joints.npz`: raw `(T, 24, 3)` flow joint positions.
- `*_rawtrack.pkl`: PHC PKL whose `obs_v=7` body-position refs come from the raw joints.
- `*_optimized.pkl`: PHC PKL produced by SMPL retarget/FK optimization.
- `*_raw_vs_optimized.gif`: raw vs optimized skeleton comparison.

```bash
python scripts/horizon_flow_to_pkl.py \
  --checkpoint $CKPT_DIR/best.pt \
  --stats $CKPT_DIR/stats.json \
  --motion constant_walk \
  --duration 30 \
  --stride 5 \
  --steps 10 \
  --seed 25 \
  --out data/generated/x16f_coldpad30_drop30_walk30_seed25_optimized.pkl \
  --raw-pkl-out data/generated/x16f_coldpad30_drop30_walk30_seed25_rawtrack.pkl \
  --raw-out data/generated/x16f_coldpad30_drop30_walk30_seed25_raw_joints.npz \
  --fit-joints-out data/generated/x16f_coldpad30_drop30_walk30_seed25_optimized_joints.npz \
  --compare-gif output/renderings/x16f_coldpad30_drop30_walk30_seed25_raw_vs_optimized.gif \
  --render-fps 12 \
  --max-render-frames 360 \
  --label x16f_coldpad30_drop30_walk30_seed25_optimized
```

### 2. Track Offline-Generated Motion Through Offline PHC

This is the normal offline tracking path:

```text
PHC MotionLib -> ref body positions/velocities -> obs_v=7 tracker
```

Raw body-position refs:

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp robot=smpl_humanoid \
  robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=data/generated/x16f_coldpad30_drop30_walk30_seed25_rawtrack.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 \
  no_log=True headless=False test=True epoch=-1
```

Optimized/retargeted refs:

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp robot=smpl_humanoid \
  robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=data/generated/x16f_coldpad30_drop30_walk30_seed25_optimized.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 \
  no_log=True headless=False test=True epoch=-1
```

Use this to answer: "Can PHC track this fixed generated sequence?"

### 3. Track Offline-Generated Motion Through The Online Pipeline

This does not sample the flow model live.  It replays the saved raw `.npz`
through `HumanoidImMCPDemo`, `_flow_ref_buffer`, online coordinate conversion,
and online velocity construction:

```text
offline raw joints .npz -> online flow replay buffer -> obs_v=7 tracker
```

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
  robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 env.getup_schedule=False \
  +env.flow_ref_mode=inprocess \
  +env.flow_ckpt=$CKPT_DIR/best.pt \
  +env.flow_stats=$CKPT_DIR/stats.json \
  +env.flow_replay_file=data/generated/x16f_coldpad30_drop30_walk30_seed25_raw_joints.npz \
  +env.flow_feedback_source=prediction \
  +env.flow_stand_steps=0 \
  +env.flow_stride=5 \
  +env.flow_ode_steps=10 \
  no_log=True headless=False test=True epoch=-1
```

Use this to answer: "Is the online reference buffer / online PHC task causing
the problem, even when the reference sequence is fixed?"

### 4. Live Online Generation Through The Online Pipeline

This samples the flow model live inside PHC.  No offline replay file is used:

```text
flow history -> live flow horizon generation -> online flow buffer -> obs_v=7 tracker
```

Pure autoregressive mode ignores robot feedback and rolls the model forward
from `stats["default_state"]`, which is the closest online equivalent to the
offline generator:

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
  robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 env.getup_schedule=False \
  +env.flow_ref_mode=inprocess \
  +env.flow_ckpt=$CKPT_DIR/best.pt \
  +env.flow_stats=$CKPT_DIR/stats.json \
  +env.flow_pure_ar=True \
  +env.flow_feedback_source=prediction \
  +env.flow_stand_steps=0 \
  +env.flow_seed=25 \
  +env.flow_stride=5 \
  +env.flow_ode_steps=10 \
  no_log=True headless=False test=True epoch=-1
```

To track flow references from the first frame, keep pure AR enabled but let the
first 90 frames use the flow model's stand phase before switching to locomotion.
`flow_hold_stand_ref=False` is the important difference from a robot-pose hold:
PHC tracks flow-generated stand references immediately instead of holding the
current robot pose.  `flow_command_heading` sets the walking direction; with
`align_robot_start_to_flow_heading=True`, PHC rotates its reset pose to face the
same heading before tracking starts.

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
  robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 env.getup_schedule=False \
  +env.flow_ref_mode=inprocess \
  +env.flow_ckpt=$CKPT_DIR/best.pt \
  +env.flow_stats=$CKPT_DIR/stats.json \
  +env.flow_pure_ar=True \
  +env.flow_feedback_source=prediction \
  +env.flow_command_heading=0.0 \
  +env.align_robot_start_to_flow_heading=True \
  +env.flow_stand_steps=90 \
  +env.flow_hold_stand_ref=False \
  +env.flow_seed=25 \
  +env.flow_stride=5 \
  +env.flow_ode_steps=10 \
  no_log=True headless=False test=True epoch=-1
```

Robot-feedback live generation conditions each new horizon on PHC's current
tracked body state instead:

```bash
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv \
  env=env_im_getup_mcp env.task=HumanoidImMCPDemo \
  robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False \
  env.z_activation=relu env.obs_v=7 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 env.episode_length=900 env.getup_schedule=False \
  +env.flow_ref_mode=inprocess \
  +env.flow_ckpt=$CKPT_DIR/best.pt \
  +env.flow_stats=$CKPT_DIR/stats.json \
  +env.flow_feedback_source=robot \
  +env.flow_stand_steps=0 \
  +env.flow_seed=25 \
  +env.flow_stride=5 \
  +env.flow_ode_steps=10 \
  no_log=True headless=False test=True epoch=-1
```

Use these to answer: "Is the issue caused by live flow sampling / feedback
history, rather than PHC tracking a fixed sequence?"

## Debug Server Command

The Flask server path is for visualization and single-env debugging.  It is not
the recommended RL fine-tuning path.

```bash
cd <PROJECT_ROOT>/tracking
python scripts/flow_server_x15.py \
  --flow-ckpt ../generation/checkpoints/joint_flow_x16f_phase_surviv_omomo_slowdrop_dropout_sched_h15_coldpad30_drop30/best.pt \
  --flow-stats ../generation/checkpoints/joint_flow_x16f_phase_surviv_omomo_slowdrop_dropout_sched_h15_coldpad30_drop30/stats.json \
  --feedback-source curriculum \
  --auto-walk-schedule \
  --stand-steps 30 \
  --curriculum-warmup-epochs 100 \
  --curriculum-ramp-end-epochs 700 \
  --curriculum-episode-length 300 \
  --initial-phase stand \
  --stride 5 \
  --ode-steps 10
```

`--feedback-source curriculum` starts with open-loop prediction feedback,
ramps toward robot-state feedback from epoch 100 to epoch 700, and uses full
robot-state feedback after epoch 700.  Robot-conditioned modes automatically
disable output alignment because the model output is already anchored to the
PHC-pushed robot state.

## Debug PHC Command

Use this only with the debug server above:

```bash
cd <PROJECT_ROOT>/tracking
python phc/run_hydra.py \
  learning=im_mcp \
  exp_name=flow_phc_walk_curriculum_x16f \
  project_name=PHC \
  no_log=False \
  env=env_im_getup_mcp \
  env.task=HumanoidImMCPDemo \
  env.obs_v=7 \
  env.episode_length=300 \
  env.num_envs=1 \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  robot=smpl_humanoid \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.z_activation=relu \
  learning.params.config.max_epochs=1000 \
  headless=True \
  test=False
```

W&B logging is enabled by `no_log=False`.  The run logs the normal PHC losses
and rewards plus flow tracking metrics:

- `wandb/flow/root_dist`
- `wandb/flow/mean_body_dist`
- `wandb/flow/max_body_dist`
- `wandb/flow/episode_idx`
- `wandb/flow/episode_step`
- `wandb/flow/is_loco`

## Notes

- Use `env.flow_ref_mode=inprocess` for RL fine-tuning with many envs.
- Keep `env.num_envs=1` only for the server-backed debug path.
- `env.obs_v=7` tracks reference body positions and velocities from the flow
  model.
- The frozen flow checkpoint is not updated by this command; only PHC is
  fine-tuned.
- Do not use a loco warm-start clip for the reset fine-tuning setup.  The
  desired behavior is stand at each PHC reset pose, then switch to loco in
  place.
