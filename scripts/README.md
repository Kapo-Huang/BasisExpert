# Experiment automation

[English](README.md) | [简体中文](README.zh-CN.md)

The script tree mirrors `configs/` and contains config generators, formal and
study runners, shared Bash primitives, and dataset/result utilities. Run all
commands from the repository root.

## Directory layout

| Directory | Contents |
| --- | --- |
| `main/` | Primary config generator, complete formal runner, and selected dataset/model runners. |
| `rd_curve/` | RD-curve selection and wrapper runner. |
| `exploration/` | Short optimizer, ECNR, and RD feasibility studies. |
| `ablation/` | Architecture, depth, and regularization generators, runners, and summaries. |
| `sensitivity/` | VarExpert expert-count and Top-K generators, selections, and runners. |
| `variable_scaling/` | Variable-count selections and runners. |
| `tools/` | Dataset inspection/export and result-maintenance utilities. |
| `lib/` | Shared shell functions; not user-facing entrypoints. |

## Formal runners

```bash
# Main and RD-curve selections from scripts/main/all_configs.list
bash scripts/main/run_all.sh

# Main experiments only
CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh

# RD-curve experiments only
bash scripts/rd_curve/run.sh

# Any custom repository-relative selection list
CONFIG_LIST_FILE=scripts/my_configs.list bash scripts/main/run_all.sh
```

`run_all.sh` defines model grouping and execution order. Selection lists only
filter that matrix; paths not recognized by a predefined group are collected
into an additional group. Non-empty lines may use trailing `#` comments.

Specialized formal entrypoints include:

| Entrypoint | Selection |
| --- | --- |
| `run_neural_expert_non_ionization.sh` | SIREN plus NeuralExpert non-Ionization stages and evaluation. |
| `run_moe_non_ionization.sh` | MoE-INR non-Ionization experiments. |
| `run_selected_datasets.sh` | CoordNet Combustion, MVNet Katrina, and STSR-INR RedSea. |
| `run_combustion_fv_apmg_instantvnr.sh` | Combustion fV-SRN, APMGSRN, and InstantVNR. |
| `run_combustion_stsr_mvnet.sh` | Combustion STSR-INR and MVNet. |
| `run_combustion_miner_ecnr.sh` | Combustion MINER and ECNR. |

## Server profiles

Every user-facing Bash runner accepts `original` and `autodl`:

```bash
bash scripts/main/run_all.sh --env autodl
bash scripts/main/run_all.sh --env=autodl
bash scripts/main/run_all.sh env=autodl
SERVER_ENV=autodl bash scripts/main/run_all.sh
```

| Setting | `original` | `autodl` |
| --- | --- | --- |
| Python launch | `conda run -n ${CONDA_ENV} ${PYTHON_BIN}` | `${PYTHON_BIN}` directly |
| Default `CONDA_ENV` | `compression` | Not used by the launcher |
| Default `PYTHON_BIN` | `python` | `python` |
| Default `RUNS_ROOT` | `<repo>/runs` | `/root/autodl-tmp/runs` |
| Default dataset base | Repository `data/Mesh` and `data/Volume` | `/root/autodl-tmp` |

Use `AUTODL_DATA_ROOT` to replace the AutoDL dataset base, or
`COMBUSTION_ROOT`, `IONIZATION_ROOT`, `REDSEA_ROOT`, and `KATRINA_ROOT` to
override individual datasets.

## Batch controls and recovery

| Variable | Behavior |
| --- | --- |
| `BATCH_LOG_ROOT` | Reuse or choose the batch directory. |
| `MAX_PARALLEL_JOBS` | Bound concurrent jobs where the selected runner supports grouping. `0` means unbounded for `run_all.sh`. |
| `DRY_RUN=1` | Print commands without training or writing status rows. |
| `RUN_TOKEN` | Override the timestamp token used by a new batch. |
| `VAR_EXPERT_INR_NUM_THREADS` | Default thread count propagated to common numerical libraries. |

Each batch writes `status.tsv`, `failed.txt`, and per-attempt logs under
`batch_logs/<timestamp>/` unless `BATCH_LOG_ROOT` is set. The last status row
for a config path is authoritative. An `ok` config is skipped when a batch is
reused; failed, interrupted, missing, or invalid states start a fresh training
attempt. Resume identity is the config path, not a hash of its YAML contents.

## Study runners

Generators and runners live together by category. Representative workflows:

```bash
python scripts/ablation/generate_architecture.py
bash scripts/ablation/run_architecture.sh

python scripts/exploration/generate_optimizer_tuning.py
bash scripts/exploration/run_optimizer_tuning.sh

python scripts/sensitivity/generate_var_expert_num.py
bash scripts/sensitivity/run_var_expert_num.sh

python scripts/sensitivity/generate_var_expert_topk.py
bash scripts/sensitivity/run_var_expert_topk.sh

bash scripts/variable_scaling/run_v04.sh
```

Study-specific run and batch roots are defined by their entrypoints. Inspect
the corresponding generator and runner before reusing a `BATCH_LOG_ROOT` from
a different study.

## Dataset and result tools

| Tool | Purpose |
| --- | --- |
| `scripts/tools/combustion.py` | Inspect/render RealPDEBench combustion trajectories and export normalized volumes. |
| `scripts/tools/katrina_wet.py` | Inspect and export dynamic Katrina wet-point samples. |
| `scripts/tools/evaluate_neural_expert_config.py` | Evaluate a NeuralExpert config using its method lifecycle. |
| `scripts/tools/organize_runs_summary.py` | Build the local run-summary view and reports. |

Use `python <tool> --help` for the current subcommands and arguments. Dataset
paths should be supplied explicitly; examples must not depend on a developer's
absolute workstation path.
