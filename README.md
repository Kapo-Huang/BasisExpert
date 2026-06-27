# VarExpert-INR

Unified INR framework for node and volume datasets.

All loaded node coordinates and targets, and all loaded volume targets, must
already be scaled into `[-1, 1]`. VarExpert-INR no longer performs runtime
normalization and will raise an error if loaded data falls outside that range.

## Commands

From the repository root:

```bash
python -m var_expert_inr.cli train --config configs/VarExpert/ionization.yaml
python -m var_expert_inr.cli predict --config configs/VarExpert/ionization.yaml
python -m var_expert_inr.cli evaluate --config configs/VarExpert/ionization.yaml
python -m var_expert_inr.mc_inr.cli train --config configs/MC-INR/ionization.yaml
python -m var_expert_inr.mc_inr.cli predict --config configs/MC-INR/ionization.yaml
python -m var_expert_inr.mc_inr.cli evaluate --config configs/MC-INR/ionization.yaml
python -m var_expert_inr.dc_inr.cli train --config configs/DC-INR/ionization.yaml --target GT
python -m var_expert_inr.dc_inr.cli predict --config configs/DC-INR/ionization.yaml --target GT
python -m var_expert_inr.dc_inr.cli evaluate --config configs/DC-INR/ionization.yaml --target GT
python -m var_expert_inr.apmgsrn.cli train --config configs/APMGSRN/ionization.yaml --target GT
python -m var_expert_inr.fv_srn.cli train --config configs/fV-SRN/ionization.yaml --target GT
python -m var_expert_inr.fv_srn.cli predict --config configs/fV-SRN/ionization.yaml --target GT
python -m var_expert_inr.fv_srn.cli evaluate --config configs/fV-SRN/ionization.yaml --target GT
python -m var_expert_inr.rmdsrn.cli train --config configs/RMDSRN/ionization.yaml --target GT
python -m var_expert_inr.rmdsrn.cli predict --config configs/RMDSRN/ionization.yaml --target GT
python -m var_expert_inr.rmdsrn.cli evaluate --config configs/RMDSRN/ionization.yaml --target GT
```

When running without installation, this repository ships a small package shim so
`python -m var_expert_inr.cli` works directly from the repo root.

Each train run writes outputs into `runs/<exp_id>/<timestamp>/`, including
`checkpoints/`, `configs/`, `logs/`, `metrics/`, and `predictions/`.
The resolved effective config is saved as `runs/<exp_id>/<timestamp>/configs/config.yaml`.
When `predict` or `evaluate` runs without an explicit checkpoint, it reuses the
latest timestamped run under the matching `exp_id`.
For `var_expert`, architecture fields that remain at default values are omitted
from the saved effective config and log output.

`mc_inr` is provided as a standalone subsystem under `var_expert_inr.mc_inr`.
It uses the same run directory layout and evaluation outputs as the unified
framework, but it does not participate in the main `var_expert_inr.cli` model
registry or training engine.

`dc_inr` is also provided as a standalone subsystem under
`var_expert_inr.dc_inr`. It performs block partition search, representative
selection, entropy-guided tiny INR training, and decompression for single-target
volume data, and it does not participate in the main `var_expert_inr.cli`
model registry.

`apmgsrn` is also provided as a standalone subsystem under
`var_expert_inr.apmgsrn`. It currently only supports single-target `ionization`
volume training by fitting one 3D APMGSRN model per timestep. Each training run
creates a fresh `runs/<exp_id>/<timestamp>/` directory containing `manifest.json`,
`configs/config.yaml`, aggregate outputs, and per-timestep artifacts under
`timesteps/`, and it does not participate in the main `var_expert_inr.cli`
model registry.

`fv_srn` is a standalone, pure-PyTorch reproduction of temporal fV-SRN.
It uses NeRF spatial Fourier features, a small SnakeAlt MLP, and learned
volumetric feature grids at configurable temporal keyframes. Intermediate
timesteps linearly interpolate their two neighboring grids. Training uses
world-space L1/L2 losses, density-guided initial sampling, and periodic
error-guided resampling. Runs save resumable FP32 checkpoints and compact
inference artifacts containing FP16 network weights and per-channel uint8
latent grids.

`rmdsrn` is a standalone temporal RMDSRN built on the same keyframe-interpolated
fV-SRN encoder. Five independent SnakeAlt decoders share the feature grid and
produce a reconstruction mean and unbiased ensemble variance. Training combines
per-member MSE with a detached-error KL variance regularizer whose weight grows
exponentially while the learning rate follows cosine annealing. Checkpoints are
resumable; FP32 inference artifacts contain only the shared encoder and decoder
parameters. Prediction writes separate `_mean.npy` and `_variance.npy` volumes,
and evaluation reports reconstruction quality, variance-error Pearson
correlation, and sampled top-1%/top-5% hit rates.

The checked-in fV-SRN Ionization config describes a full 100-timestep
`(T,Z,Y,X)=(100,248,248,600)` sequence. The sample target files currently in
this repository contain only two timesteps; they must be replaced with the
full sequence before that config can run. The loader deliberately reports this
shape mismatch instead of silently reinterpreting the data.
