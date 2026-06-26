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
python -m var_expert_inr.apmgsrn.cli train --config configs/APMGSRN/ionization.yaml --target GT
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

`apmgsrn` is also provided as a standalone subsystem under
`var_expert_inr.apmgsrn`. It currently only supports single-target `ionization`
volume training by fitting one 3D APMGSRN model per timestep. Its outputs are
written to `runs/apmgsrn/<exp_id>/` without timestamped subdirectories, and it
does not participate in the main `var_expert_inr.cli` model registry.
