# X15 Model: Conditioning and Output Specification

Version: `x15_horizon_position_v7`  
Source: `examples/joint_flow/dataset_horizon_x15.py`  
Architecture: `JointFlowConcatMLP` (`--arch concat`; see "Architecture" below — FiLM was found unable to learn this task)

---

## Overview

```
condition (161D) → Flow Model → H future state frames (H × 76D)
```

At every timestep t, an **anchor frame** is defined from the pelvis state at t-1. All spatial quantities — past frames, target frames, goals — are expressed relative to this anchor. During training a random yaw offset θ may be added to the anchor heading (`augment=True`; see the corrected note on θ below — it is a gauge rotation, not behavior synthesis, and the current recommended training uses `--no-augment`).

---

## Anchor Frame

Defined once per sample in `_build_sample`:

```
anchor_origin   = pelvis_world_xy[t-1]          (2D, metres)
anchor_z        = pelvis_world_z[t-1]            (scalar, metres)
anchor_heading  = yaw[t-1]  +  θ                (θ ~ U(0,2π) during training, 0 at inference)
anchor_pitch    = pitch[t-1]                     (no augmentation)
anchor_roll     = roll[t-1]                      (no augmentation)
```

`yaw`, `pitch`, `roll` come from `pose_aa_root` (axis-angle) decomposed as ZYX Euler angles via scipy.

**To convert a world-frame XY vector into the anchor frame:**
```
x_anchor =  cos(anchor_heading) * x_world + sin(anchor_heading) * y_world
y_anchor = -sin(anchor_heading) * x_world + cos(anchor_heading) * y_world
```
(`_rot_to_anchor` in code)

**To make an angle anchor-relative:**
```
angle_rel = arctan2(sin(angle - anchor_angle), cos(angle - anchor_angle))
```
(`_angle_rel` in code)

**Corrected note on random θ (2026-06-10):** the original rationale ("model learns to steer") was wrong. θ rotates the condition AND the targets together — a pure change of coordinates (gauge rotation). The relative geometry (turn amount, goal bearing vs body, heading vs future trajectory) is unchanged for every θ, so augmentation adds no new behavior and cannot teach turning (measured: 10-frame heading change in the corpus is ≤14° for ACCAD/CMU, ≤73° max for OMOMO, with or without θ). Deployment always operates at θ=0 (the anchor is its own frame), so augmentation spends capacity on gauges never used at inference. Current recommended training: `--no-augment`.

**At inference:** θ = 0. anchor_heading = actual yaw[t-1]. Planner provides desired_heading in this frame (locomotion phase only; see below).

---

## State Frame Layout (76D)

Used for both past frames (input) and future frames (output). Computed by `_state_frame`.

```
[ root_delta_xy (2) | root_z (1) | orientation_quat (4) | joint_rel_body (69) ]
  indices 0:2          index 2      indices 3:7             indices 7:76
```

### `root_delta_xy` — indices 0:2

**What:** Pelvis XY displacement from anchor origin.  
**Frame:** Anchor frame — displacement is rotated by anchor_heading.  
**Formula:**
```
root_delta = rot_to_anchor(pelvis_xy[frame] - anchor_xy, anchor_heading)
```
**Special case:** For the most recent past frame (frame = t-1 = anchor), this is always **[0, 0]** since the anchor origin IS that pelvis position.  
**Units:** Metres.

---

### `root_z` — index 2

**What:** Absolute world-space pelvis height.  
**Why:** The model must learn crouching depth for pickup/place phases. Without an explicit height signal the model has no way to know how low to go; the body-relative joints alone are ambiguous (crouching joints can occur at any height). Adding `root_z` lets the model predict the correct pelvis height trajectory directly.  
**Units:** Metres (world frame; anchor_z is NOT subtracted, so this is the absolute pelvis elevation).

---

### `orientation_quat` — indices 3:7

**What:** Full 3D orientation of the pelvis (yaw + pitch + roll) relative to anchor orientation.  
**Frame:** Anchor frame — all three angles are made relative to anchor_heading / anchor_pitch / anchor_roll before composing.  
**Formula:**
```python
heading_rel = angle_rel(yaw[frame],   anchor_heading)
pitch_rel   = angle_rel(pitch[frame], anchor_pitch)
roll_rel    = angle_rel(roll[frame],  anchor_roll)
q = R.from_euler("ZYX", [heading_rel, pitch_rel, roll_rel]).as_quat()  # [x, y, z, w]
if q[3] < 0: q = -q   # positive-w hemisphere convention
```
**Layout:** `[qx, qy, qz, qw]`  
**Why quaternion over sin/cos per angle:** 4D vs 6D, no gimbal lock. Positive-w convention prevents discontinuities from double-cover (q and -q represent the same rotation — adjacent frames stay on the same hemisphere for smooth human motion).  
**At default/padding state:** Identity quaternion `[0, 0, 0, 1]` (no rotation relative to anchor).

---

### `joint_rel_body` — indices 7:76

**What:** 3D positions of 23 non-pelvis joints, relative to pelvis, expressed in body frame.  
**Frame:** Body frame — pelvis subtracted (translation removed) AND XY rotated by anchor_heading (rotation removed). Z is unaffected by yaw.  
**Formula:**
```python
joint_rel = joint_positions[frame, 1:] - joint_positions[frame, 0:1]  # subtract pelvis (23×3)
joint_rel[:, :2] = rot_to_anchor(joint_rel[:, :2], anchor_heading)     # rotate XY to body frame
```
**Layout:** Joints 1–23 concatenated as `[j1x, j1y, j1z, j2x, j2y, j2z, ..., j23z]` = 69D.  
**Pelvis:** Always dropped (it is the origin, so always [0,0,0]).  
**Why body frame:** Subtracting pelvis position removes translation, rotating by -anchor_heading removes orientation. This means "right arm swings forward" looks identical regardless of which world direction the agent faces — the model learns pose patterns, not world-direction-specific patterns.  
**Units:** Metres.

---

## Condition Vector (161D)

```
[ past_frame[t-2] (76D) | past_frame[t-1] (76D) | task_phase (4D) | desired_heading (2D) | active_goal (3D) ]
  \_____________________________past (152D)______________________________/  \_______task (9D)________/
```

### Past frames — 152D (2 × 76D)

Both frames use the **same anchor** (computed from t-1).

**`past_frame[t-2]`** — where the agent was 2 frames ago:
- `root_delta_xy`: displacement from t-1 position in body frame → encodes direction and speed of recent motion
- `orientation_quat`: how the pelvis was oriented 2 frames ago relative to t-1 orientation → encodes orientation change rate
- `joint_rel_body`: full body pose 2 frames ago in body frame

**`past_frame[t-1]`** — current state (anchor IS t-1):
- `root_delta_xy`: **always [0, 0]** — anchor origin is this frame's pelvis position
- `orientation_quat`: **always identity [0,0,0,1]** — anchor orientation is this frame's orientation
- `joint_rel_body`: current full body pose in body frame — this is the most important signal

Together the two past frames encode:
- Current full body pose (from t-1)
- Recent motion direction and speed (from t-2 displacement)
- Rate of orientation change (from t-2 vs t-1 quaternion difference)
- Whether the agent was crouching, standing, mid-swing, etc.

**Padding:** Frames before the clip start (t < past_frames) use `default_state`: `[0,0, 0,0,0,1, median_locomotion_joints]`.

---

### `task_phase` — 4D one-hot

| Value | Phase | Data source | Meaning |
|---|---|---|---|
| `[1, 0, 0, 0]` | locomotion | ACCAD, CMU | straight walking, no object |
| `[0, 1, 0, 0]` | pickup | OMOMO | walking toward box + crouching + grasping |
| `[0, 0, 1, 0]` | carry | OMOMO | walking while holding object |
| `[0, 0, 0, 1]` | place | OMOMO | crouching and setting object down |

**How derived for OMOMO** (`_derive_task_phase` with `task_phase_dim=4`):
```
carry  = (obj_world_z > 0.95) AND stable AND (pelvis_z > 0.85)
place  = last 45% of clip AND obj low
pickup = everything else (approach + pre-grasp + grasp)
```
Every frame is assigned to exactly one phase — no zeros.

---

### `desired_heading` — 2D

**What:** The direction the agent should face at the **horizon-end frame** (t+H-1), in the anchor frame. Used for locomotion phase only; goal phases receive a null `[0, 0]` token.

**Phase gating:**
- `task_phase = locomotion`: `desired_heading = [sin(heading_rel), cos(heading_rel)]` where `heading_rel = angle_rel(yaw[t+H-1], anchor_heading)` (clipped to clip end).
- `task_phase = pickup / carry / place`: `desired_heading = [0, 0]` (null token — no heading signal).

**Why horizon-end rather than frame t?** Frame-t heading is nearly determined by the current and previous frames (≤14° change in H=10 frames for CMU/ACCAD), so it carries almost no new information. Horizon-end heading H frames ahead is genuinely predictive — it tells the model where to point the body at the end of the planned window, giving real steering control.

**Why null during goal phases?** During pickup/carry/place the planner cannot reliably compute a body heading independently of the motion (the model should decide how to orient during crouch/carry). Providing a GT-derived heading during training would leak future information; providing zeros makes the signal consistently uninformative so the model ignores it and steers via `active_goal` instead.

**At inference (locomotion phase):**
```python
desired_heading = rot_to_anchor(normalize(goal_xy - agent_xy), anchor_heading)
→ [sin(angle_to_goal), cos(angle_to_goal)]
```
- Locomotion: point toward pickup location
- Goal phases: `[0, 0]`

---

### `active_goal` — 3D

The target position for the current task phase. Meaning and frame depend on phase:

#### Locomotion — `[0, 0, 0]`
No goal. Model relies entirely on task_phase + desired_heading + past frames.

#### Pickup — anchor frame XYZ
**What:** World position of the pickup location (where the box sits on the ground).  
**Frame:**
- XY: `rot_to_anchor(pickup_world_xy - anchor_xy, anchor_heading)` → forward/sideways offset from agent in body frame
- Z: `pickup_world_z - anchor_z` → height of box surface relative to agent's pelvis height at t-1

**How derived:** Median world position of the object during the first 35% of the clip where `obj_world_z` is low (below 40th percentile) and the object is stable (step < 0.05m). Constant per clip, tiled across all frames.

**Interpretation:** "The box is X metres ahead of me, Y metres to my left, Z metres below my pelvis." Gets closer as the agent approaches.

#### Carry — anchor frame XYZ (place location)
**What:** World position of the **place goal**, expressed in the current anchor frame. Steers the carry walk toward the destination.  
**Frame:** Same as pickup/place — XY rotated to anchor frame, Z is anchor-relative.  
**Formula:**
```python
xy = rot_to_anchor(place_world_xy - anchor_xy, anchor_heading)
goal = [xy[0], xy[1], place_world_z - anchor_z]
```
**How derived:** Median world position of the object during the last 35% of the clip (same as place goal). Constant per clip, re-expressed in the current anchor frame each timestep.  
**Why place location, not hold position?** The old design provided the body-relative hold position (where the object sits in the hands), which gives no steering information — the model had no signal to know which direction to walk. Using the place location lets the model orient the carry walk toward the destination, consistent with how pickup uses the pickup location.

#### Place — anchor frame XYZ
Same layout as pickup goal but for the destination.  
**How derived:** Median world position of object during last 35% of clip where object is low and stable.

---

## Output: H Future State Frames (H × 76D)

The model predicts H consecutive state frames starting at t, all in the **same anchor frame as the condition** (anchor = pelvis at t-1).

Each output frame has the **identical 76D layout** as the past state frames above:
- `root_delta_xy (0:2)`: forward/sideways displacement of the predicted pelvis from anchor origin, in body frame
- `root_z (2)`: absolute predicted pelvis height (world Z)
- `orientation_quat (3:7)`: predicted pelvis orientation relative to anchor orientation
- `joint_rel_body (7:76)`: predicted 23-joint body pose in body frame

**Total target dimension:** H × 76 = 760D (for H=10).

**Aux loss slice indices** (in `horizon_aux_loss`):
- Root XYZ: `[:, :, 0:3]`
- Orientation: `[:, :, 3:7]`
- Joints: `[:, :, 7:]`

**Target frame clamping:** Frames beyond clip end repeat the last frame.

---

## Receding-Horizon Inference

At each inference step:
1. Planner computes anchor from current robot state (pelvis XY, Z, yaw at t-1)
2. Planner provides task_phase, desired_heading, active_goal (9 numbers total)
3. Past frames assembled from robot proprioceptive history
4. ODE integration → H future frames in anchor frame
5. First predicted frame used as reference for PHC tracker
6. Advance t by 1: new anchor = old t-1 → t, feed predicted frame back as new past_frame[t-1]

---

## Planner Interface at Inference (9 numbers)

| Signal | Dim | How to compute |
|---|---|---|
| `task_phase` | 4 | one-hot: [locomotion / pickup / carry / place] |
| `desired_heading` | 2 | locomotion: `[sin(φ), cos(φ)]` where `φ = angle_rel(goal_direction, yaw[t-1])`; goal phases: `[0, 0]` |
| `active_goal` | 3 | pickup: pickup XYZ in anchor frame; carry: place XYZ in anchor frame; place: place XYZ in anchor frame; loco: `[0,0,0]` |

Plus past 2 state frames from robot proprioception (joint positions + pelvis orientation from IMU/FK).

**Carry steering:** During carry the planner provides the **place location** (not a hold position), so the model steers the walk toward the destination. The model has learned from OMOMO carry frames that object-in-hand motion is correlated with walking toward the place site.

---

## Data Sources

| Dataset | Phase | Oversample | Notes |
|---|---|---|---|
| ACCAD straight walk | locomotion `[1,0,0,0]` | 23× | straight walk only, varied world directions |
| CMU straight walk | locomotion `[1,0,0,0]` | 23× | 56 clips, mean 394 frames (~13s) |
| OMOMO loco-manipulation | pickup/carry/place | 5× | 542 clips, 107k frames |

**OMOMO phase distribution:**
- pickup: 56.0% (approach + pre-grasp + grasp, all merged)
- carry: 25.9%
- place: 18.0%
- approach: 0% (removed — all non-carry non-place frames are pickup)

---

## Architecture

### `JointFlowConcatMLP` (current — `--arch concat`)

Condition concatenated into the trunk input; residual hidden layers:

```
t (scalar) → TimestepEmbedder(256) → t_emb (256D)
x_t (760D) + cond (161D) + t_emb (256D) → concat (1217D) → Linear → hidden (1024D) → SiLU
  residual block × 5: h = h + SiLU(Linear(h))
  → Linear → velocity (760D)
```

- **Parameters:** ~7.4M
- **Zero-init output projection** so initial velocity ≈ 0 (standard flow matching practice).

### Why not FiLM (`JointFlowMLP`)?

FiLM conditions via per-layer scale/shift applied to the x_t trunk. The condition is compressed through a 256D bottleneck before being projected to scale/shift, so the condition's effective bandwidth is ~256D. For x15 the condition is 161D and the target is 760D; the FiLM pathway has insufficient bandwidth to steer the trunk, which effectively learns an unconditional mean.

Evidence (overfit check on 4 OMOMO clips, 500 steps):
- FiLM: flow loss plateaus at **~0.97** (target variance = 1.0 → model learns nothing; copy-anchor baseline beats generated output at j0: 1.1 cm vs 18.7 cm)
- Concat: flow loss reaches **~0.15** (65× lower)
- UMAP manifold check: generated→GT NN distance 19.0 (FiLM) vs GT→GT baseline 2.2 — generated samples are not landing on the data manifold at all

Concat+residual resolves this by exposing the full condition to every layer via the input projection and keeping gradient paths short through residual connections.

---

## Experiments

### v4 → v5: Stats must be computed from augmented distribution

**Run:** v4 GPU augmented training (20 epochs).  
**Symptom:** Train loss flat at ~106k for all 15 epochs; no decrease.  
**Root cause:** `stats.json` was computed from the un-augmented dataset. For heading-relative dimensions the un-augmented std is near-zero (all ACCAD/CMU clips walk roughly the same direction). When θ-augmented samples were normalized by those near-zero stds, values exploded to ~100× magnitude, rendering the loss meaningless.  
**Fix (v5+):** Stats are computed from the augmented distribution (N_aug = N_clips × n_aug_samples random θ draws). Stds for all dimensions become well-conditioned.

### v5 → v6: No-augment recommended (θ is a gauge rotation)

**Finding:** Random θ rotates condition AND targets together — a pure change of coordinates. The relative geometry (turn amount, goal bearing vs body, heading vs future trajectory) is identical for every θ, so augmentation adds no new behavior and cannot teach the model to turn. Deployment runs at θ=0 (anchor is its own frame), so the model wastes capacity adapting to gauges that never appear at inference.  
**Corpus check:** 10-frame heading change ≤14° for ACCAD/CMU, ≤73° max for OMOMO, with or without θ. Turning is inherent to the data, not produced by augmentation.  
**Fix:** `--no-augment` flag added. GPU precomputes the full normalized cache once at startup; each epoch is a pure shuffle-gather on device (~15× faster than the CPU augmentation path).

### FiLM architecture failure → Concat architecture

**Run:** v7 FiLM model (2000 epochs, best_val=1.0089).  
**Symptom:** Reconstruction GIFs showed pure noise — model learned nothing.  
**Diagnosis:**  
1. UMAP manifold check: generated→GT NN distance median 19.0 (GT→GT baseline 2.2) — generated samples nowhere near the data manifold.  
2. t-bin loss profile: flat ~1.07 at all t (expected: higher at small t, decreasing toward t=1). No signal.  
3. One-shot x̂₁ MSE: 1.07 with target variance 1.0 — predicting a constant mean would score 1.0; the model is slightly worse.  
4. Overfit check (4 OMOMO clips): FiLM flow loss plateaued at ~0.97 vs concat's ~0.15.  
5. A/B conditioning test: providing a constant all-zeros condition produced near-identical loss to the real condition — FiLM has zero effective conditioning pull.  
**Root cause:** FiLM's 256D bottleneck cannot steer a 760D trunk for a 161D condition. The condition effectively disappears.  
**Fix:** `JointFlowConcatMLP` added; condition concatenated into input projection, bypassing the bottleneck. Overfit check confirms it learns.

### v7-concat full run

**Command:**
```bash
python examples/joint_flow/train_horizon_x15.py \
    --hdf5 \
      final_h5_files/accad_straight_walk_taskspace_with_root.h5 \
      final_h5_files/cmu_straight_walk_taskspace_with_root.h5 \
      final_h5_files/omomo_locomanipulation_crouch_plus_portable_new_train_split_taskspace.h5 \
    --oversample 23 23 5 \
    --outdir checkpoints/joint_flow_x15_horizon_position_v7_concat_longer \
    --hidden 1024 --layers 5 --lr 3e-4 --batch 32768 \
    --past-frames 2 --horizon 10 --no-augment --arch concat
```
**Result:** 2000 epochs, best_val=**0.47952** (flow=0.1215, aux=0.3592 at final epoch). Training stable throughout; no sign of collapse.

**Pending evaluation:** Teacher-forced and autoregressive reconstruction, UMAP manifold check, horizon-offset L2 profile vs baseline.

---

## Known Limitations

1. **Exposure bias:** Teacher-forced reconstruction works (CMU: 16.86 cm joint L2, drift 0.4 cm), but autoregressive rollout collapses (CMU: 159.68 cm L2, root drift 276 cm) for the v7-FiLM checkpoint. The v7-concat model has not yet been evaluated — this may improve substantially since the FiLM model learned nothing. Scheduled-sampling or DAGGER may be needed if collapse persists.
2. **desired_heading is body heading, not direction-to-goal.** During locomotion the two are approximately aligned (agent faces toward the box), but slight misalignment is possible. Future improvement: set `desired_heading = direction from pelvis to active_goal XY` to make the two signals consistent.
3. **No velocity conditioning.** Root velocity was removed (v5+) in favor of heading + goal signals. The model must infer speed from past frame displacements, which may limit speed range generalization beyond the training distribution.
