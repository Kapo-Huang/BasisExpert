# VarExpert-INR

[English](README.md) | [简体中文](README.zh-CN.md)

VarExpert-INR is a PyTorch framework for training, predicting, and evaluating
implicit neural representations (INRs) on temporal node and structured-volume
fields. It provides a shared training engine for VarExpert and several
baselines, plus self-contained reproductions for methods with specialized
training or checkpoint lifecycles.

## Requirements and installation

- Python 3.10 or newer
- PyTorch and the dependencies declared in `pyproject.toml`
- A CUDA-capable environment is recommended for formal experiments

Install the project from the repository root:

```bash
python -m pip install -e .
```

Install the optional image, mesh, and perceptual-evaluation dependencies when
SSIM, LPIPS, or node rendering is needed:

```bash
python -m pip install -e ".[evaluation]"
```

Volume rendering additionally uses the sibling VolumeVis checkout:

```bash
python -m pip install -e "../Vis[lpips]"
```

## Data contract

Node targets and volume targets must already be scaled to `[-1, 1]`. Node
coordinates must also use that range unless the config supplies validated
`coordinate_stats_path` statistics for standardization. Structured-volume
coordinates are generated from array indices and normalized by the loader.

Formal datasets are not distributed by the Python package. Configure their
paths through the YAML files and environment variables described in the
[data guide](src/var_expert_inr/data/README.md) and
[configuration guide](configs/README.md).

## Quick start

Run commands from the repository root. A unified-engine training run needs only
an experiment config:

```bash
python -m var_expert_inr.cli train \
  --config configs/main/VarExpert/ionization.yaml
```

Prediction uses the latest timestamped run for the config's `exp_id` unless an
explicit checkpoint is supplied:

```bash
python -m var_expert_inr.cli predict \
  --config configs/main/VarExpert/ionization.yaml
```

Evaluate an existing run with selected metrics, targets, and inclusive
timestep ranges:

```bash
python -m var_expert_inr.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --metrics psnr,decode_time,memory \
  --targets GT \
  --timesteps 0,10:30,40:99:10
```

Methods with specialized lifecycles use their own module. For example:

```bash
python -m var_expert_inr.methods.apmgsrn.cli train \
  --config configs/main/APMGSRN/combustion_40NH3_1__Temperature.yaml
```

See the [method guide](src/var_expert_inr/methods/README.md) for the complete
entrypoint and capability matrix, and the
[evaluation guide](src/var_expert_inr/evaluation/README.md) for metric,
rendering, selection, and caching behavior.

## Formal experiment runners

The selection files are the source of truth for the current experiment matrix.
The default runner uses `scripts/main/all_configs.list`, which contains both
main and RD-curve experiments:

```bash
# Complete formal selection
bash scripts/main/run_all.sh

# Main experiments only
CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh

# RD-curve experiments only
bash scripts/rd_curve/run.sh
```

All Bash entrypoints default to the original `compression` Conda environment.
Select the AutoDL profile explicitly when needed:

```bash
bash scripts/main/run_all.sh --env autodl
SERVER_ENV=autodl bash scripts/rd_curve/run.sh
```

The [automation guide](scripts/README.md) documents server profiles, dataset
overrides, concurrency, dry runs, retry behavior, batch logs, and specialized
runners.

## Run layout

Training writes one timestamped run below the resolved `RUNS_ROOT`:

```text
runs/<exp_id>/<timestamp>/
├── checkpoints/   # inference checkpoint or bundle
├── configs/       # resolved effective config
├── logs/
├── metrics/
└── predictions/
```

Run-based evaluation writes self-contained reports below
`<run>/evaluations/<timestamp>/`. The original server profile defaults
`RUNS_ROOT` to the repository `runs/` directory; AutoDL defaults it to
`/root/autodl-tmp/runs`.

## Documentation map

| Area | Documentation |
| --- | --- |
| Experiment YAMLs and generation | [configs/README.md](configs/README.md) |
| Batch runners and utilities | [scripts/README.md](scripts/README.md) |
| Node and volume data contracts | [data/README.md](src/var_expert_inr/data/README.md) |
| Evaluation, metrics, and rendering | [evaluation/README.md](src/var_expert_inr/evaluation/README.md) |
| Self-contained method entrypoints | [methods/README.md](src/var_expert_inr/methods/README.md) |
| Shared-engine baselines | [baselines/README.md](src/var_expert_inr/models/baselines/README.md) |
| Proposed models | [proposed/README.md](src/var_expert_inr/models/proposed/README.md) |

## Operational notes

- `data/`, `runs/`, `runs_summary/`, and `batch_logs/` are local artifacts and
  are excluded from version control.
- Quality metrics require readable, shape-compatible ground truth. Performance
  metrics can evaluate checkpoint decoding without ground truth.
- Volume rendering requires VolumeVis. Node rendering requires a supported mesh
  or explicit vertices and cells; there is no point-cloud fallback.
- Generated configuration counts change as the matrix evolves. Consult the
  active `.list` files instead of relying on a number copied into prose.
