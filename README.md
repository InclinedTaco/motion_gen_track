# Motion Generation & Tracking

**Command-conditioned human motion generation with physics-based tracking.**

**Shyam Charan designed and built the humanoid motion generator from scratch.**
Given recent body states, a task phase, a desired heading, and a goal, the
model predicts a sequence of full-body poses. PHC tracks those poses in Isaac
Gym. During live tracking, the simulated humanoid's measured state feeds back
into the generator's next prediction.

```mermaid
flowchart LR
    C[Task phase + heading + goal] --> G[Flow-matching generator]
    H[Recent state history] --> G
    G --> R[Future pose horizon]
    R --> P[PHC tracking policy]
    P --> S[Isaac Gym humanoid]
    S --> H
```

## What this project contains

- **Motion generation:** conditional flow matching, past-state conditioning,
  and self-attention / cross-attention horizon models.
- **Physics integration:** coordinate conversion, reference buffering, and
  batched in-process flow inference in the PHC environment.
- **Closed-loop feedback:** separate histories for each simulated humanoid,
  with robot-state or prediction feedback for controlled comparisons.
- **Research tooling:** motion-data processing, retargeting, generation
  visualization, and the original fine-tuning configuration.

The [humanoid generator](generation/examples/joint_flow/) uses Meta's Flow
Matching library for probability paths and ODE solvers. The tracking policy
comes from PHC. See [component credits](docs/attribution.md) and the
[source revisions](provenance.json).

## Get started

Start with **[setup and required assets](docs/setup.md)**. This is a source
release: pretrained weights, SMPL models, and motion datasets are obtained
separately. Generation and physics tracking have different runtime requirements.

From the repository root, after setup:

```bash
# Check the active Python environment without loading the simulator.
python scripts/motion.py doctor

# Preview a phase-conditioned X16 checkpoint (best.pt + its stats.json).
python scripts/motion.py generate \
  --checkpoint generation/checkpoints/x16/best.pt \
  --stats generation/checkpoints/x16/stats.json \
  --motion constant_walk --seconds 5 --out outputs/walk.gif

# Track live generated references with the PHC runtime and assets installed.
python scripts/motion.py track \
  --checkpoint generation/checkpoints/x16/best.pt \
  --stats generation/checkpoints/x16/stats.json \
  --feedback robot
```

`track` uses a frozen generator and PHC inference with one humanoid by default.
It does not start training. Add `--dry-run` to either command to inspect the
resolved launch command without loading assets; all paths are resolved before
changing working directory. The viewer runs until closed; `--episode-steps`
limits each episode, not the total viewer runtime.

## Repository guide

| Path | Contents |
| --- | --- |
| `generation/flow_matching/` | Upstream flow-matching library |
| `generation/examples/joint_flow/` | Humanoid datasets, models, training and visualization |
| `tracking/phc/` | PHC tasks, policies, and simulator configuration |
| `tracking/scripts/` | Flow-to-PHC adapters, retargeting and data tools |
| `scripts/motion.py` | Unified generation and live-tracking entry point |
| `tests/` | Portable integration and launch-contract checks |
| `docs/` | Setup, architecture, validation and attribution |

The original research instructions remain in
[generation/RESEARCH.md](generation/RESEARCH.md) and
[tracking/FLOW_RESEARCH.md](tracking/FLOW_RESEARCH.md). The upstream PHC manual
is [tracking/UPSTREAM.md](tracking/UPSTREAM.md). These include historical
experiments and asset references beyond the supported quick start.

## Validation and scope

See **[validation](docs/validation.md)** for checks actually run on this
consolidated version. Source checks and a generated skeleton preview do not
establish successful physics tracking. The documented entry point supports
76-value, phase-conditioned X16 humanoid checkpoints. X17 object-aware models
remain research code and are not interchangeable with that tracking adapter.

This project targets a simulated SMPL humanoid. It does not claim real-robot
deployment or a validated manipulation success rate. Upstream PHC paper
benchmarks are not results of this integration.

```bash
python -m unittest discover -s tests -v
python scripts/check_source.py
```

## License

The components retain their licenses: [generation](generation/LICENSE) uses
CC BY-NC 4.0; [tracking](tracking/LICENSE) uses BSD 3-Clause Clear. See
[LICENSE](LICENSE) and [attribution](docs/attribution.md). Third-party model
and dataset terms apply separately.
