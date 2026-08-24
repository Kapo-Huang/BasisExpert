# Experiment configurations

[English](README.md) | [简体中文](README.zh-CN.md)

This directory contains experiment YAML files. Configurations are grouped first
by the question an experiment answers and then by the published method name.
Run all commands from the repository root.

## Directory layout

| Directory | Purpose |
| --- | --- |
| `main/` | Primary comparisons at each method's default formal budget. |
| `rd_curve/` | Formal rate-distortion tiers: `Size041`, `Size082`, `Size163`, `Size326`, and `Size652`. |
| `exploration/` | Short schedule, optimizer, and feasibility probes. |
| `ablation/` | Controlled architecture, depth, and regularization studies. |
| `sensitivity/` | VarExpert expert-count and Top-K sweeps. |
| `variable_scaling/` | Formal VarExpert runs with different variable counts. |

Study directories use semantic names instead of chronology-based `vN` names.
Method names such as `fV-SRN` are preserved in paths to match paper and run
metadata.

## Formal selections

Selection files, not counts copied into documentation, define the active formal
matrix:

| Selection | File | Runner |
| --- | --- | --- |
| Main and RD curve | `scripts/main/all_configs.list` | `bash scripts/main/run_all.sh` |
| Main only | `scripts/main/configs.list` | `CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh` |
| RD curve only | `scripts/rd_curve/configs.list` | `bash scripts/rd_curve/run.sh` |

Each non-empty, non-comment line is one repository-relative YAML path. You may
copy a list and set `CONFIG_LIST_FILE` to run a custom subset without editing
the default lists.

## Unified config shape

Shared-engine configs use these top-level sections:

```yaml
experiment: descriptive_name
exp_id: stable-run-id
experiment_root: ${RUNS_ROOT}

data:
  kind: volume                 # volume or node
  dataset_name: ionization
  targets:
    GT: ${IONIZATION_ROOT}/target_GT.npy
  volume_shape: {X: 600, Y: 248, Z: 248, T: 100}

model:
  name: var_expert
  # Method-specific fields follow.

training:
  epochs: 600
  batch_size: 16000
  lr: 5.0e-5

evaluation:
  batch_size: 16000
```

`data.kind` is either `node` or `volume`. Node configs provide `coords_path`
and either `target_path` or `targets`; volume configs provide either
`target_path` or `targets` and may need `volume_shape` for flat arrays. A
single-target method can select one member of `targets` with `data.target`.
See the [data guide](../src/var_expert_inr/data/README.md) for shapes and range
requirements.

Standalone methods may own additional method-specific sections or schemas. Use
an existing config from that method's directory as the template and its
[method documentation](../src/var_expert_inr/methods/README.md) to choose the
correct CLI.

## Path placeholders

Generated configs use portable placeholders that are resolved when the config
is loaded:

| Placeholder | Default |
| --- | --- |
| `${REPO_ROOT}` | Repository root discovered from the config or package. |
| `${RUNS_ROOT}` | `<repo>/runs` for `original`; `/root/autodl-tmp/runs` for `autodl`. |
| `${DATASETS_ROOT}` | Adjacent `INR/Datasets` tree unless overridden. |
| `${COMBUSTION_ROOT}` | `<repo>/data/Volume/Combustion` or `<AUTODL_DATA_ROOT>/Combustion`. |
| `${IONIZATION_ROOT}` | `<repo>/data/Volume/Ionization` or `<AUTODL_DATA_ROOT>/Ionization`. |
| `${REDSEA_ROOT}` | `<repo>/data/Mesh/RedSea` or `<AUTODL_DATA_ROOT>/RedSea`. |
| `${KATRINA_ROOT}` | `<repo>/data/Mesh/Katrina` or `<AUTODL_DATA_ROOT>/Katrina`. |

The corresponding environment variable overrides each dataset root.
`AUTODL_DATA_ROOT` changes the common AutoDL base directory.

## Generating configurations

Generated files must be rebuilt with the matching entrypoint under
`scripts/<category>/`. The primary matrix is generated with:

```bash
python scripts/main/generate_configs.py
```

Other generators live beside their study runners, for example:

```bash
python scripts/ablation/generate_architecture.py
python scripts/exploration/generate_optimizer_tuning.py
python scripts/sensitivity/generate_var_expert_num.py
python scripts/sensitivity/generate_var_expert_topk.py
```

Do not create a new top-level `configs_*` directory. Add a semantic study
directory under the existing category and keep generated selection lists in
sync with the generated YAML files.
