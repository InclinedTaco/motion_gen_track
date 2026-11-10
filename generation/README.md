# Motion generation

The [joint-flow models](examples/joint_flow/) predict sequences of humanoid
poses from recent body states and task commands. This directory contains the
model architectures, motion data loaders, training scripts, and
visualization tools. The [Flow Matching library](flow_matching/) provides
probability paths and ODE solvers.

See the [quick start](../README.md#run) to generate a motion preview or the
[research guide](RESEARCH.md) to train a model. Run training commands from
this directory and supply checkpoints and HDF5 datasets separately.
