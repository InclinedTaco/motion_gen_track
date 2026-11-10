# Setup

## 1. Generation

Use Python 3.9 or newer with PyTorch supported by your machine. The imported
generator environment is `generation/environment.yml`; its dependency versions
are not a reproducible lockfile. For an existing compatible Python environment,
run from the repository root:

```bash
python -m pip install -r requirements-generation.txt
python scripts/motion.py doctor
```

Copy a **matching** `best.pt` and `stats.json` into
`generation/checkpoints/x16/`, or supply their absolute paths. Checkpoint and
normalization files must come from the same run. They are intentionally absent
from Git, and no public generator-weight download has been published here.
The preview launcher accepts phase-conditioned 76D X16 models, including
`history_transformer`, `cross_transformer`, and `full_cross_transformer`.

```bash
python scripts/motion.py generate \
  --checkpoint generation/checkpoints/x16/best.pt \
  --stats generation/checkpoints/x16/stats.json \
  --motion constant_walk --seconds 5 --device cpu --out outputs/walk.gif
```

Use `--device cuda` for GPU generation. `--steps` controls ODE integration;
`--stride` must not exceed the checkpoint's horizon. Generating a GIF does not
run physics or measure tracking accuracy.

## 2. PHC tracking

Tracking requires **Linux, an NVIDIA GPU, and legacy Isaac Gym Preview**, not
Isaac Sim / Isaac Lab. Use a compatible PHC runtime with Python 3.9+ for this
combined generator. The original PHC manual describes a Python 3.8 runtime;
that environment cannot simply install this generator, whose package requires
Python >=3.9. Full combined-environment installation has not yet been validated.

First install a compatible PyTorch and Isaac Gym build following their own
installation instructions, then activate that runtime and install:

```bash
python -m pip install -r requirements-tracking.txt
python scripts/motion.py doctor --tracking
```

The inherited PHC requirements include dependencies from Git branches. These
are retained from the source project and are not a fully pinned environment.
Avoid upgrading an existing working simulator environment in place.

Obtain SMPL v1.1 models from the SMPL project under its terms and place the
neutral, male, and female `.pkl` files in `tracking/data/smpl/` as described in
[the PHC manual](../tracking/UPSTREAM.md#dependencies). Licensed body models and
motion datasets are not redistributed here.

Download the upstream PHC sample data and pretrained models:

```bash
cd tracking
bash download_data.sh
cd ..
```

This script uses the original upstream Drive links; downloads are external
assets and availability is not guaranteed by this source release. For the
standard MCP tracker, the relevant paths are:

| Asset | Path |
| --- | --- |
| MCP composer | `tracking/output/HumanoidIm/phc_kp_mcp_iccv/Humanoid.pth` |
| PNN primitive bank | `tracking/output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth` |
| Standing reference | `tracking/sample_data/amass_isaac_standing_upright_slim.pkl` |
| Body parameters | `tracking/data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` |

```bash
python scripts/motion.py track \
  --checkpoint generation/checkpoints/x16/best.pt \
  --stats generation/checkpoints/x16/stats.json \
  --feedback robot
```

Use `--composer`, `--primitives`, and `--standing-motion` for external asset
locations. `--feedback prediction` runs the same live adapter using generated
history instead of tracked robot history. `--headless` disables the viewer.
No Flask server is needed: this path samples the flow model in-process.

## 3. Training and research utilities

Training is explicit and is separate from the quick start. The generator's
existing scripts run from `generation/`; PHC scripts run from `tracking/`.
See [generation research instructions](../generation/RESEARCH.md) and
[PHC flow research notes](../tracking/FLOW_RESEARCH.md). The legacy PHC
fine-tuning YAML includes an experiment-specific checkpoint path and absolute
resume epoch; configure those for your own assets before using it.

For offline FK retargeting, generate `tracking/final_humanoid.xml` with the
same robot flags as the tracker; do not silently substitute a generic XML.
The old README mentions `scripts/horizon_flow_to_pkl.py`, which was missing
from the source repository. That command is not offered in this quick start.
