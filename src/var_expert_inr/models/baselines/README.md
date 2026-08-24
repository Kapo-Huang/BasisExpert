# Shared-engine baselines

[English](README.md) | [简体中文](README.zh-CN.md)

These models use the common config, dataset, training, checkpoint, prediction,
and evaluation stack. Their public config names are registered in
`var_expert_inr.models.registry` and can be invoked through
`python -m var_expert_inr.cli`.

## Registered models

| Config name | Implementation | Output mode |
| --- | --- | --- |
| `siren` | `siren.py` | Single target |
| `coordnet` | `coordnet.py` | Single target |
| `moe_inr` | `moe_inr.py` | Single target |
| `instant_ngp` | `instant_ngp.py` | Single target |
| `instant_vnr` | `instant_vnr.py` | Single target |
| `mvnet` | `mvnet.py` | Multiple scalar targets in one model |
| `stsr_inr` | `stsr_inr.py` | Multiple targets in one model |

`hash_grid.py` contains shared encoding components used by hash-grid baselines;
it is not a separate registered model.

## Shared lifecycle

All registered baselines receive dataset metadata from the common data factory.
The registry materializes defaults, validates model-specific keys and input or
output dimensions, and returns a common model adapter. Training then uses the
shared sampler, loss, logging, checkpoint, and prediction code.

```bash
python -m var_expert_inr.cli train \
  --config configs/main/SIREN/ionization__GT.yaml
```

Single-target models require the config to select exactly one target. Multi-
target models preserve their output-name order in the effective config and
checkpoint so prediction and evaluation can reconstruct the same mapping.

Experiment-ready settings belong in `configs/`, not in this directory. See the
[configuration guide](../../../../configs/README.md) for organization and the
[data guide](../../data/README.md) for input contracts.
