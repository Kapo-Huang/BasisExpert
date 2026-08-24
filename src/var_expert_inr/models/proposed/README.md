# Proposed models

[English](README.md) | [简体中文](README.zh-CN.md)

This package contains VarExpert and related multi-target INR implementations.
Code presence and package exports are distinct from registration in the public
experiment CLI.

## Availability

| Model | Python export | Registry config name | Unified CLI |
| --- | --- | --- | --- |
| VarExpert | `VarExpert` | `var_expert` | Yes |
| Shared-encoder INR | `SharedEncINR` | `shared_enc_inr` | Yes |
| Variable-agnostic MoE | `VariableAgnosticMoE` | Not registered | No |

VarExpert and SharedEncINR use the common dataset, training, checkpoint,
prediction, and evaluation stack:

```bash
python -m var_expert_inr.cli train \
  --config configs/main/VarExpert/ionization.yaml
```

`VariableAgnosticMoE` and `build_variable_agnostic_moe_from_config` are exported
for direct Python use and development. They are not accepted as a `model.name`
by `var_expert_inr.models.registry`; adding a class to this package alone does
not make it a runnable experiment model.

## Multi-target contract

Registered proposed models build one output head per dataset target and return
predictions keyed by target name. Target dimensions and order come from dataset
metadata and are preserved by the common effective-config and checkpoint
pipeline.

Reusable layers shared by the proposed models live in `components.py`.
Experiment architecture and training choices belong in `configs/`; see the
[configuration guide](../../../../configs/README.md).
