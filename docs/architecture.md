# Generation and tracking architecture

## Generator

The active humanoid research code is under `generation/examples/joint_flow/`.
A flow-matching velocity model is integrated with an ODE solver to sample a
future state horizon. State history and planner-style commands condition the
model. The root launcher uses the existing horizon visualization script.

The supported phase model uses 76 values per state: root XY displacement,
root height, a four-value quaternion, and 23 root-relative joint positions.
Commands include task phase, desired heading, and an active goal. History
length, horizon, and normalization are read from the checkpoint's `stats.json`.
Do not infer these from a checkpoint directory name.

Self-attention and cross-attention architectures coexist. Object-aware X17
uses an expanded state and a different command schema; the root launcher
rejects it instead of passing it into the 76D tracking adapter.

## PHC integration

`tracking/phc/env/tasks/humanoid_im_mcp_demo.py` loads
`tracking/scripts/flow_server_x15.py` for in-process inference. Despite the
historical filename, no HTTP server is started by this path. It performs one
batched horizon query across environments, with separate reference cursors and
state histories per humanoid. Robot reset clears its corresponding history.

The adapter loads the generator from the sibling `generation/` directory.
`flow_server.py` supplies existing joint-order and coordinate conversion
helpers. PHC `obs_v=7` consumes reference body positions and velocities. The
MCP composer combines the pretrained primitive bank to control the humanoid.

The root launcher resolves user paths before changing working directory,
loads explicit composer and primitive checkpoints, sets `test=True`, and
turns off experiment logging. PHC's existing runner derives `train=False`
from that test flag. The upstream training code remains available separately.

## Scope of consolidation

This release imports source snapshots at the commits in `provenance.json`.
It retains the two Python trees to avoid collisions between their `scripts`,
`tests`, data paths and old dependencies. Shared entry points provide one
project workflow without forcing a simulator dependency into basic generation.

Original repository histories remain in the source repositories. Their local
uncommitted changes were not imported. Checkpoints, datasets, generated media,
and run logs are excluded from the new repository. Research descriptions and
upstream documentation remain available under clearly named reference files.
