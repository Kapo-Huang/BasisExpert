# Self-contained methods

[English](README.md) | [简体中文](README.zh-CN.md)

Packages in this directory own method-specific parts of the lifecycle, such as
config parsing, data access, training, inference bundles, or prediction. They
therefore do not belong to the shared `var_expert_inr.models` registry even
when a package contains PyTorch model classes.

## Entrypoints and formal coverage

The dataset column below describes the current formal configs under
`configs/main/`, not every dataset a method could support after adding a new
config.

| Method | Module | Commands | Formal datasets | Lifecycle notes |
| --- | --- | --- | --- | --- |
| APMGSRN | `apmgsrn` | `train`, `evaluate` | Combustion, Ionization | Fits scalar temporal fields and writes one inference bundle for the run. |
| ECNR | `ecnr` | `train`, `predict`, `evaluate` | Combustion, Ionization | Owns packed multiscale training and compact inference checkpoints. |
| fV-SRN | `fv_srn` | `train`, `predict`, `evaluate` | Combustion, Ionization | Owns temporal feature-grid training and inference. |
| MC-INR | `mc_inr` | `train`, `predict`, `evaluate` | Combustion, Ionization, Katrina, RedSea | Uses its method-specific config, data, and checkpoint pipeline. |
| MINER | `miner` | `train`, `predict`, `evaluate` | Combustion, Ionization | Trains scalar fields by timestep and stores the completed temporal inference bundle. |
| NeuralExpert | `neural_expert` | `train`, `evaluate` | Combustion, Ionization, Katrina, RedSea | Supports volume and mesh lifecycles, including manager-pretrain configs. |
| RMDSRN | `rmdsrn` | `train`, `predict`, `evaluate` | Combustion, Ionization | Produces reconstruction means and ensemble variances. |

Invoke a method with:

```bash
python -m var_expert_inr.methods.<module>.cli <command> [arguments]
```

Examples:

```bash
python -m var_expert_inr.methods.apmgsrn.cli train \
  --config configs/main/APMGSRN/combustion_40NH3_1__Temperature.yaml

python -m var_expert_inr.methods.rmdsrn.cli predict \
  --config configs/main/RMDSRN/ionization__GT.yaml

python -m var_expert_inr.methods.neural_expert.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --metrics psnr,memory
```

Use `python -m var_expert_inr.methods.<module>.cli --help` and the command's
`--help` output for method-specific overrides.

## Unified CLI compatibility

ECNR and MINER are dispatched by `python -m var_expert_inr.cli` when their
model names are found in a config. This preserves the common experiment
commands while retaining their specialized runners:

```bash
python -m var_expert_inr.cli train \
  --config configs/main/MINER/ionization__GT.yaml
```

Other packages in this directory must use their method module for training and
prediction. Run-based evaluation is shared: every `evaluate --run ...`
entrypoint forwards to `var_expert_inr.evaluation` and writes the same report
layout.

## Checkpoints and runs

Formal training creates a fresh `runs/<exp_id>/<timestamp>/` directory and
saves an inference-oriented checkpoint or bundle below `checkpoints/`.
Method-owned manifests and intermediate metrics may differ, but completed runs
are adapted to the common run-based evaluator. Do not assume that optimizer or
training-progress state is present in a formal checkpoint.

Legacy config-based evaluation remains available only where the method CLI
declares it. Prefer `evaluate --run <run-dir>` for a self-contained report and
consistent target/timestep selection.

## Attribution

The native MINER subsystem includes its upstream license at
[`miner/LICENSE.MINER`](miner/LICENSE.MINER). This file does not define a
project-wide license for VarExpert-INR.
