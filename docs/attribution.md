# Component credits

| Component | Credit |
| --- | --- |
| [Humanoid motion generator](../generation/examples/joint_flow/) — model architectures, conditioning, training, and inference | Shyam Charan |
| [Online reference and robot-state feedback adapter](../tracking/phc/env/tasks/humanoid_im_mcp_demo.py) | Shyam Charan |

The [source revisions](../provenance.json) identify the generator and tracking
repositories used in this release.

## Upstream components

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
