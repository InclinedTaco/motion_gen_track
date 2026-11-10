# Attribution and project contribution

Shyam Charan designed and implemented the humanoid motion generator from
scratch. Its model architectures, conditioning scheme, horizon training code,
and receding-horizon inference are in
[`generation/examples/joint_flow/`](../generation/examples/joint_flow/).

Shyam also connected the generator to PHC: the adapter sends generated pose
references to the tracking policy and feeds the simulated humanoid's state
back into the generator. The [source revisions](../provenance.json) identify
the original generator and tracker repositories.

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

## Résumé wording

> Built a flow-matching humanoid motion generator from scratch that predicts
> full-body pose sequences from task commands and recent body states. Connected
> it to PHC for physics-based tracking with robot-state feedback.
