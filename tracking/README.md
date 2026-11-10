# Physics-based tracking

PHC tracks references produced by the generator in `../generation/`.
Start with the [project quick start](../README.md) and [setup](../docs/setup.md).

- [Flow integration research notes](FLOW_RESEARCH.md)
- [Upstream PHC manual and attribution](UPSTREAM.md)
- `phc/env/tasks/humanoid_im_mcp_demo.py`: batched online reference integration
- `scripts/flow_server_x15.py`: horizon generation and feedback utilities

The unified `scripts/motion.py track` command runs inference. Legacy research
commands in this directory include training and require explicit configuration.
