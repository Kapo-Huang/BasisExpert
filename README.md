# VarExpert-INR

Unified INR framework for node and volume datasets.

All loaded node coordinates and targets, and all loaded volume targets, must
already be scaled into `[-1, 1]`. VarExpert-INR no longer performs runtime
normalization and will raise an error if loaded data falls outside that range.

## Commands

From the repository root:

```bash
python -m var_expert_inr.cli train --config configs/examples/node_light_basis_expert.yaml
python -m var_expert_inr.cli predict --config configs/examples/node_light_basis_expert.yaml
python -m var_expert_inr.cli evaluate --config configs/examples/node_light_basis_expert.yaml
```

When running without installation, this repository ships a small package shim so
`python -m var_expert_inr.cli` works directly from the repo root.

Each run writes the fully resolved effective config, including defaulted values,
into the run log under `runs/<exp_id>/logs/`.
