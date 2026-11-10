# Attribution and project contribution

This repository consolidates Shyam Charan's generator and tracking integration
from `InclinedTaco/flow_matching` and `InclinedTaco/PHC`. Exact source revisions
are recorded in `provenance.json`; the source repositories retain their history.

The project-specific work includes humanoid conditioning and horizon-model
experiments under `generation/examples/joint_flow/`, the generator-to-PHC
adapters under `tracking/scripts/`, and online reference/feedback integration in
`tracking/phc/env/tasks/humanoid_im_mcp_demo.py`. These build on upstream code;
a source import is not a claim of authorship over every imported line.

- **Flow Matching**: Meta's library provides the probability paths, ODE solvers
  and utility framework. Upstream: https://github.com/facebookresearch/flow_matching.
  Its license is preserved in `generation/LICENSE` (CC BY-NC 4.0).
- **Perpetual Humanoid Control for Real-time Simulated Avatars**: upstream
  physics-based tracking, policy architecture and simulator infrastructure.
  Upstream: https://github.com/ZhengyiLuo/PHC. Its license is preserved in
  `tracking/LICENSE` (BSD 3-Clause Clear), with additional notices in imported files.
- **Isaac Gym, SMPL/SMPL-X, AMASS and OMOMO**: external simulator, body-model,
  and dataset dependencies retain their own terms. Their licensed model/data
  files are not included in this source consolidation.

The complete original PHC description and paper citations remain in
`tracking/UPSTREAM.md`. Its benchmark tables describe upstream PHC, not this
flow-conditioned integration.

## Résumé wording

A factual project description, without inventing performance claims:

> Integrated a command-conditioned flow-matching motion generator with a
> physics-based humanoid tracker, using receding-horizon inference and
> robot-state feedback to connect generated pose references to simulated control.

Add measured tracking, runtime, or manipulation results only after running a
reproducible evaluation with the corresponding released configuration.
