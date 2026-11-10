<div align="center">

# Flow Matching — X16

[![arXiv](assets/arXiv-2412.06264-red.svg)](https://arxiv.org/abs/2412.06264)
[![License: CC BY-NC 4.0](assets/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

Receding-horizon full-body pose generator. X16 takes `K` past state
frames plus a compact planner command (task phase, desired heading, active goal)
and predicts the next `H` future state frames in a receding-horizon loop. Output
is fed to PHC / Isaac Gym as a motion reference.

Two attention architectures are in active use, differing only in how the past
frames are consumed:

| Model | `--arch` | Past frames are… |
|---|---|---|
| **X16 self** | `history_transformer` | prepended as tokens and self-attended together with the horizon tokens |
| **X16 cross** | `cross_transformer` | held as a read-only memory the horizon tokens cross-attend into |

## Setup

Create the conda environment with all required dependencies, install the package
in editable mode, and set up the pre-commit hook:

```bash
conda env create -f environment.yml
conda activate flow_matching
pip install -e .
pre-commit install
```

## Training

Both models train from the same physics-rollout HDF5 files and differ only in
`--arch` (and `--outdir`). Run from the repo root.

### X16 self (history transformer)

```bash
python examples/joint_flow/train_horizon_x15.py \
    --hdf5 final_h5_files/physics_rollout/accad_straight_walk_physics.h5 \
           final_h5_files/physics_rollout/cmu_straight_walk_physics.h5 \
           final_h5_files/physics_rollout/omomo_train_physics.h5 \
    --oversample 19 19 1 \
    --outdir checkpoints/joint_flow_x16_history_physics30_h15_policy30 \
    --arch history_transformer \
    --hidden 512 --layers 8 --heads 8 \
    --past-frames 2 --horizon 15 \
    --mirror-sources 0 1 2 \
    --epochs 100 --batch 2048 --lr 3e-4 --amp
```

### X16 cross (cross-attention transformer)

```bash
python examples/joint_flow/train_horizon_x15.py \
    --hdf5 final_h5_files/physics_rollout/accad_straight_walk_physics.h5 \
           final_h5_files/physics_rollout/cmu_straight_walk_physics.h5 \
           final_h5_files/physics_rollout/omomo_train_physics.h5 \
    --oversample 19 19 1 \
    --outdir checkpoints/joint_flow_x16c_cross_h15 \
    --arch cross_transformer \
    --hidden 512 --layers 8 --heads 8 \
    --past-frames 2 --horizon 15 \
    --mirror-sources 0 1 2 \
    --epochs 100 --batch 2048 --lr 3e-4 --amp
```

Each run writes `best.pt` (weights + optimizer state) and `stats.json`
(normalization stats + the full training config). Both are required for
inference.

### Phase model with slow-locomotion dropping

For phase-conditioned models where `locomotion` should keep walking instead of
falling into a stop/stand attractor, drop slow locomotion windows instead of
relabeling them as stand and ramp past dropout after basic dynamics are learned:

```bash
python examples/joint_flow/train_horizon_x15.py \
    --hdf5 final_h5_files/physics_rollout/accad_cmu_straight_walk_survivors_strict_s2p0_h0p40_physics.h5 \
           final_h5_files/physics_rollout/omomo_train_physics.h5 \
    --oversample 39 1 \
    --outdir checkpoints/joint_flow_x16f_phase_survivor_omomo_slowdrop_dropout_sched_h15 \
    --arch full_cross_transformer \
    --condition-schema phase \
    --hidden 512 --layers 8 --heads 8 --frame-dim 76 --mlp-ratio 4.0 \
    --past-frames 2 --horizon 15 \
    --epochs 100 --batch 2048 --lr 3e-4 \
    --root-loss-weight 2.0 --heading-loss-weight 1.0 \
    --joint-loss-weight 0.25 --smooth-loss-weight 0.1 \
    --slow-loco-mode drop \
    --past-dropout 0.15 \
    --past-dropout-warmup-epochs 20 \
    --past-dropout-ramp-epochs 40 \
    --past-noise-std 0.05 --past-noise-ramp-epochs 50 \
    --mirror-sources 0 1 \
    --amp
```

## CLI flags

### Data

| Flag | Default | Description |
|---|---|---|
| `--hdf5` | required | One or more HDF5 files (space-separated). |
| `--oversample` | none | Per-file repeat factors (one int per `--hdf5` path). |
| `--outdir` | `checkpoints/joint_flow_x15_horizon` | Checkpoint output directory. |
| `--stats-in` | none | Reuse a precomputed `stats.json` (skips recomputation). |
| `--loss-log` | none | Path to write per-epoch loss CSV. |
| `--val-frac` | `0.05` | Validation fraction. |
| `--max-clips-per-source` | none | Cap clips loaded per HDF5 (quick smoke tests). |
| `--no-augment` | off | Disable yaw augmentation (stats/samples from raw clips). |
| `--mirror` | off | Double the dataset with left-right mirrored clips. |
| `--mirror-sources` | none | Mirror only these HDF5 source indices (0-based); overrides `--mirror`. |
| `--cold-start-pad` | `0` | Prepend N copies of frame 0 per clip for consistent rest→motion windows. |
| `--filter-loco-hand-speed` | `0.0` | Drop non-object locomotion samples whose target horizon overlaps sustained high pelvis-relative hand speed; 0 disables. |
| `--filter-loco-hand-height` | `0.0` | Drop non-object locomotion samples whose target horizon overlaps sustained high pelvis-relative hand height; 0 disables. |
| `--filter-loco-hand-window` | `1` | Number of sustained bad hand-motion frames required before a locomotion sample is dropped. |
| `--slow-loco-mode` | `relabel` | How to handle locomotion windows whose target-horizon root speed is below the stand threshold: `relabel` preserves existing stand-split behavior, `drop` removes those samples, and `keep` leaves them as locomotion. Use `drop` when training phase-conditioned locomotion that should not learn stop attractors. |
| `--no-precompute` | off | Skip pre-computing augmented windows into RAM (less memory, slower). |

### Architecture

| Flag | Default | Description |
|---|---|---|
| `--arch` | `concat` | `history_transformer` (X16 self), `cross_transformer` (X16 cross), `transformer` (X15 DiT), `concat`, or `film` (legacy). |
| `--hidden` | `1024` | Hidden size. |
| `--layers` | `5` | Transformer/residual depth. |
| `--heads` | `8` | Attention heads (transformer archs). |
| `--frame-dim` | `76` | Per-frame target dimension (must match state dim). |
| `--mlp-ratio` | `4.0` | Transformer feed-forward expansion ratio. |
| `--dropout` | `0.0` | Transformer attention/MLP dropout. |
| `--past-frames` | `2` | `K` — number of past state frames in the condition. |
| `--horizon` | `15` | `H` — number of future frames predicted per step. |

### Condition / observation rate

| Flag | Default | Description |
|---|---|---|
| `--motion-fps` | none | Generation/reference FPS; used with `--policy-hz` to derive the past-frame stride. |
| `--policy-hz` | none | Policy observation frequency. E.g. `--motion-fps 120 --policy-hz 30` spaces past frames by 4 motion frames. Must be passed together with `--motion-fps`. |
| `--use-target-speed` | off | Add GT mean root speed over the horizon as a conditioning scalar (locomotion frames only). |
| `--past-dropout` | `0.0` | Probability of zeroing the past-frames block per sample (PARC-style). Forces the model to follow task commands instead of overfitting to past frames; enables guidance blending and native cold starts at inference. |
| `--past-dropout-warmup-epochs` | `0` | Number of initial epochs with past dropout disabled. |
| `--past-dropout-ramp-epochs` | `0` | Epochs to linearly ramp past dropout from 0 to `--past-dropout` after warmup; 0 keeps constant dropout. |
| `--past-noise-std` | `0.0` | Gaussian noise std on past frames during training; simulates AR rollout / state-estimation error to reduce mean-reversion stalling. |
| `--past-noise-ramp-epochs` | `0` | Epochs to linearly ramp past-frame noise from 0 to `--past-noise-std`; 0 keeps constant noise. |

### Optimization

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `500` | Training epochs. |
| `--batch` | `8192` | Batch size. |
| `--grad-accum-steps` | `1` | Microbatches per optimizer step; effective batch = `--batch × --grad-accum-steps`. |
| `--lr` | `3e-4` | AdamW learning rate. |
| `--amp` | off | CUDA automatic mixed precision (reduces activation memory). |
| `--compile` | off | `torch.compile` the model (faster epochs; ~1 min one-time compile). |
| `--drop-last` | off | Drop the last incomplete batch each epoch. |
| `--num-workers` | `0` | DataLoader workers. |
| `--device` | auto | `cuda` or `cpu`. |

### Loss weights

| Flag | Default | Description |
|---|---|---|
| `--root-loss-weight` | `2.0` | Root XY/Z loss weight. |
| `--heading-loss-weight` | `1.0` | Heading (quat) loss weight. |
| `--joint-loss-weight` | `0.25` | Per-joint position loss weight. |
| `--smooth-loss-weight` | `0.1` | Frame-to-frame smoothness loss weight. |
| `--bone-loss-weight` | `0.0` | Bone-length consistency (penalizes predicted link lengths deviating from GT). |
| `--contact-loss-weight` | `0.0` | MDM-style foot-contact loss (penalizes foot velocity on GT-planted frames). |

### Rollout consistency (stride-matched self-conditioning)

| Flag | Default | Description |
|---|---|---|
| `--rollout-prob` | `0.0` | Stride-matched rollout consistency probability at full ramp (0 = off). |
| `--rollout-stride` | `5` | Predicted frames consumed before replanning in the rollout branch. Requires `K ≤ stride < H`. |
| `--rollout-ramp-epochs` | `500` | Epochs to linearly ramp `--rollout-prob` after warmup. |
| `--rollout-warmup-epochs` | `300` | Keep rollout consistency off for this many epochs. |
| `--rollout-gt-weight` | `0.25` | Weight for the stride-matched B-window GT loss. |
| `--overlap-loss-weight` | `0.5` | Weight for `A[stride:H]` vs `B[0:H-stride]` overlap consistency. |
| `--splice-vel-weight` | `0.1` | Weight for the first B frame matching the GT one-step splice. |

## State representation and condition

Each state frame is **76D**: `root_delta_xy (0:2)` + `root_z (2)` +
`quat_xyzw (3:7)` + `joint_rel (7:76)` (23 non-pelvis joints, root-relative
XYZ), expressed in an anchor frame (the current predicted frame during
autoregressive rollout).

**Condition per sample:** `K × 76D` past frames + `5D` task phase +
`2D` desired heading + `3D` active goal.

| Field | Dim | Notes |
|---|---|---|
| `task_phase` | 5 | One-hot `[stand, locomotion, pickup, carry, place]`. Non-object clips use locomotion or stand. |
| `desired_heading` | 2 | `(sin θ, cos θ)` of heading at horizon end, relative to anchor. Null `(0, 0)` during goal phases (steered by `active_goal`). |
| `active_goal` | 3 | Pickup goal (pickup phase) or place goal (carry/place phases), in anchor heading frame. Zeros for non-object clips. |

**Target:** `H × 76D` future state frames.

## Motion generation

All generation scripts auto-detect the architecture from `stats.json`, so the
same commands work for X16 self and X16 cross — just swap the checkpoint dir.

### Synthetic, command-driven generation

Drives the model with planner-style commands and renders a `.gif`/`.mp4`.

```bash
# Constant walking
python examples/joint_flow/visualization/animate_horizon_x15.py \
    --checkpoint checkpoints/joint_flow_x16c_cross_h15/best.pt \
    --stats      checkpoints/joint_flow_x16c_cross_h15/stats.json \
    --motion constant_walk --duration 5.0 \
    --out gif/x16c_walk.gif

# Walk → pick → carry → place (per-phase durations)
python examples/joint_flow/visualization/animate_horizon_x15.py \
    --checkpoint checkpoints/joint_flow_x16c_cross_h15/best.pt \
    --stats      checkpoints/joint_flow_x16c_cross_h15/stats.json \
    --motion walk_pcp \
    --walk-secs 2.5 --pickup-secs 2.0 --carry-secs 3.0 --place-secs 2.0 \
    --out gif/x16c_walk_pcp.gif
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `best.pt`. |
| `--stats` | required | Path to `stats.json`. |
| `--motion` | `constant_walk` | Preset: `stand`, `constant_walk`, `curved_walk`, `stop_start_walk`, `walk_pcp`, `pick_carry_place`. |
| `--duration` | `5.0` | Duration in seconds (presets without explicit per-phase durations). |
| `--turn-rate` | `30.0` | Commanded yaw rate (deg/s) for `curved_walk`. |
| `--walk-secs` | `2.5` | Locomotion duration (s) before pcp, `walk_pcp`. |
| `--pickup-secs` | `2.5` | Pickup duration (s), `walk_pcp`. |
| `--carry-secs` | `3.0` | Carry duration (s), `walk_pcp`. |
| `--place-secs` | `2.5` | Place duration (s), `walk_pcp`. |
| `--speed` | `1.3` | Desired root speed (m/s) for locomotion phases. |
| `--steps` | `20` | ODE solver steps per frame. |
| `--stride` | `5` | Frames kept per predicted horizon before re-planning. |
| `--fps` | from stats | Motion FPS (default: read from stats, fallback 30). |
| `--render-fps` | `30.0` | Output GIF FPS (frames subsampled from motion FPS). |
| `--past-guidance` | `0.0` | PARC-style blend toward the command-only branch (~0.65 typical); requires `--past-dropout` training. |
| `--warm-start-hdf5` | none | Seed past-frame history from K consecutive GT frames of a real clip. |
| `--warm-start-clip` | `0` | Clip index within `--warm-start-hdf5`. |
| `--warm-start-frame` | `0` | First GT frame of the K-frame warm-start window. |
| `--out` | `gif/x15_horizon.gif` | Output path (`.gif` or `.mp4`). |
| `--device` | auto | `cuda` or `cpu`. |

### Clip reconstruction (vs ground truth)

Autoregressive rollout on a real clip, rendered predicted-vs-GT side by side.

```bash
python examples/joint_flow/visualization/reconstruct_x15.py \
    --checkpoint checkpoints/joint_flow_x16c_cross_h15/best.pt \
    --stats      checkpoints/joint_flow_x16c_cross_h15/stats.json \
    --hdf5       final_h5_files/physics_rollout/omomo_train_physics.h5 \
    --clip-idx   0 --out gif/x16c_recon.gif
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `best.pt`. |
| `--stats` | required | Path to `stats.json`. |
| `--hdf5` | required | HDF5 file to load the clip from. |
| `--clip-idx` | `0` | Clip index in the HDF5. |
| `--max-frames` | `240` | Max frames to reconstruct. |
| `--teacher-forced` | off | Use GT past frames at every step instead of model predictions. |
| `--out` | `gif/x15_recon.gif` | Output path. |

### Side-by-side comparison

Stitch two rendered GIFs horizontally with labels (shorter clip holds its last
frame so both play full length).

```bash
python examples/joint_flow/visualization/stitch_side_by_side.py \
    --a gif/x16_walk_pcp.gif --b gif/x16c_walk_pcp.gif \
    --label-a "x16 self" --label-b "x16 cross" \
    --out gif/x16_vs_x16c.gif
```

## Past experiments

Earlier models, kept for reference. Legacy single-frame / prev-frame models used
`examples/joint_flow/train.py`; X15/X16 receding-horizon models use
`examples/joint_flow/train_horizon_x15.py`. Metrics are mean joint L2 where
measured; "—" means not benchmarked.

| Model | Checkpoint | `--arch` | L2 | Notes |
|---|---|---|---|---|
| Reference (single-frame) | `joint_flow_singleframe` | — | 1.69 cm | Baseline all earlier experiments were measured against. |
| Prev-frame conditioned | `joint_flow_prevframe` | — | 1.08 cm (ACCAD) | `--use-prev-frame`; former deployment model for RL tracking. |
| No-EEF | `joint_flow_noeef_prevframe` | — | — | `--no-eef`; planner-compatible (root vel + pelvis height only). |
| 6D rotation output | `joint_flow_6d` | — | — | `--use-6d`; IK-free decode via FK; active research direction. |
| Obj-conditioned (CMU+OMOMO 40×) | `joint_flow_obj_noeef_prevframe_accad_cmu_omomo40x` | — | OMOMO 1.77 cm | Former manipulation baseline; `--use-obj`. |
| Task-goal + phase | `joint_flow_pick_carry_place_startgoal_taskphase_noobj_noeef_prevframe_accad_cmu_omomo542x13_5050` | — | — | `--use-task-goals --use-task-phase`; no-obj carry experiment. |
| X15 DiT (all conditions via adaLN) | `joint_flow_x15_horizon_position_v7_concat_longer` | `transformer` | — | DiT-style horizon-token model; past frames + task command injected through adaLN/FiLM. Locomotion works; AR drift on OMOMO; cannot generate synthetic motions open-loop (overfits past frames, ignores task command). |
| X15 past-dropout + contact | `joint_flow_x15_v8_dropout_contact` | `concat` | — | `--past-dropout 0.15 --contact-loss-weight 100 --bone-loss-weight 1.0 --cold-start-pad 10 --past-noise-std 0.05`; PARC-style masking to force task-command following and fix AR drift. |
| **X16 self (current)** | `joint_flow_x16_history_physics30_h15_policy30` | `history_transformer` | — | Past frames attended as prepended tokens. |
| **X16 cross (current)** | `joint_flow_x16c_cross_h15` | `cross_transformer` | — | Past frames cross-attended as read-only memory. |

---

## About the `flow_matching` library

This repository is a fork of [`flow_matching`](https://arxiv.org/abs/2412.06264),
a PyTorch library for Flow Matching algorithms (continuous and discrete), part of
the [Flow Matching Guide and Codebase](https://arxiv.org/abs/2412.06264). The
humanoid motion research lives under `examples/joint_flow/`.

### Repository structure

```bash
.
├── flow_matching                  # Core library
│   ├── loss                       # Loss functions
│   ├── path                       # Path and schedulers
│   │   └── scheduler              # Schedulers and transformations
│   ├── solver                     # Solvers for continuous and discrete flows
│   └── utils
└── examples                       # Synthetic, image, and text examples
    ├── joint_flow                 # humanoid motion research (X15/X16)
    ├── image
    └── text
```

### FAQ

**I want to train a Flow Matching model, where can I find the training code?**
The library ships [training examples](examples): synthetic data for
[continuous](examples/2d_flow_matching.ipynb),
[discrete](examples/2d_discrete_flow_matching.ipynb), and
[Riemannian](examples/2d_riemannian_flow_matching_flat_torus.ipynb) Flow
Matching, full [image training](examples/image) (CIFAR10 and face-blurred
ImageNet), and a scalable discrete [text modeling](examples/text) example.

**Do you release pre-trained models?**
The upstream library does not; all models under [examples](examples) can be
trained from scratch with a single command.

**How to contribute to this codebase?**
Please follow the [contribution guide](CONTRIBUTING.md).

## License

The code in this repository is CC BY-NC licensed. See the [LICENSE](LICENSE) for details.

## Citation

```
@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code},
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264},
}
```
