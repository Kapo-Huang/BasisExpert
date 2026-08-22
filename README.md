# VarExpert-INR

Unified INR framework for node and volume datasets.

All loaded node coordinates and targets, and all loaded volume targets, must
already be scaled into `[-1, 1]`. VarExpert-INR no longer performs runtime
normalization and will raise an error if loaded data falls outside that range.

## Commands

From the repository root:

```bash
python -m var_expert_inr.cli train --config configs/main/VarExpert/ionization.yaml
python -m var_expert_inr.cli train --config configs/main/SIREN/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/main/InstantNGP/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/main/InstantVNR/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/main/MVNet/ionization.yaml
python -m var_expert_inr.methods.mc_inr.cli train --config configs/main/MC-INR/ionization.yaml
python -m var_expert_inr.methods.neural_expert.cli train --config configs/main/NeuralExpert/ionization__GT__managerpretrain.yaml
python -m var_expert_inr.methods.neural_expert.cli train --config configs/main/NeuralExpert/ionization__GT.yaml
python -m var_expert_inr.methods.apmgsrn.cli train --config configs/main/APMGSRN/ionization__GT.yaml
python -m var_expert_inr.methods.fv_srn.cli train --config configs/main/fV-SRN/ionization__GT.yaml
python -m var_expert_inr.methods.rmdsrn.cli train --config configs/main/RMDSRN/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/main/ECNR/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/main/MINER/ionization__GT.yaml
```

MINER is integrated as a self-contained PyTorch subsystem. It trains one
scalar spatial field per timestep: Ionization uses the published 3D path with
four scales and `16^3` blocks, while the `128x128` Combustion fields use the
published 2D `32x32` blocks and automatically reduce to three compatible
scales. Training writes scale-complete and final timestep checkpoints under
the timestamped run. Resume a partial run with `--resume <run-or-scale-path>`;
run-based evaluation decodes only the requested timesteps. The external
reference checkout is not required at runtime.

## Evaluation

The run-based evaluator supports selected targets and inclusive timestep ranges
without materializing an entire temporal prediction:

```powershell
python -m var_expert_inr.cli evaluate `
  --run runs/siren-ionization-size082-GT/20260721_115139_637467 `
  --metrics psnr,ssim,lpips,decode_time,memory `
  --timesteps 0,10:30,40:99:10 `
  --targets GT
```

Available metrics are `psnr`, `ssim`, `lpips`, `decode_time`, and `memory`.
The default is `psnr`; the default target and timestep selections are `all`.
SSIM and LPIPS automatically render matching ground-truth and prediction
images. `--render` without `--metrics` renders predictions and does not require ground
truth. PSNR, SSIM, and LPIPS always require readable, shape-compatible ground
truth and fail before decoding when it is missing. Performance-only evaluation
(`decode_time` and/or `memory`) also works without ground truth.

Each new evaluation writes a self-contained report beneath
`<run>/evaluations/<timestamp>/`, including `manifest.json`, `metrics.json`,
`metrics.csv`, logs, and any requested PNGs. Decode timing excludes rendering,
metric calculation, and prediction-file writes. Memory reports process RSS and,
when applicable, CUDA allocated/reserved peaks.

Quality and render results are reused when their source, selection, render
profile, and GT fingerprints match; `--overwrite` forces a new evaluation.
Performance metrics always perform a fresh decode. The evaluator also resolves
legacy saved paths against the adjacent `INR/Datasets/<dataset_name>/` tree
before the repository's small sample-data tree, so existing runs can use the
full external Ionization arrays without editing their saved configs.

Volume rendering uses the adjacent VolumeVis project. Install the optional
evaluation dependencies and VolumeVis from the repository root:

```powershell
pip install -e ".[evaluation]"
pip install -e "..\Vis[lpips]"
```

Built-in dataset render profiles ship with `var_expert_inr.evaluation`; use
`--eval-config` for a custom profile. Ionization uses the packaged VolumeVis
presets. Katrina uses its `fort.14` mesh. Node rendering
requires a real VTU/VTK, ADCIRC `fort.14`, or explicit cell mesh; there is no
point-cloud fallback. A prediction-only node render must define a fixed `clim`
in its profile. Profiles may provide a VTU/VTK/`fort.14` path, or explicit
`vertices_path` and `cells_path` NumPy arrays (including timestep templates).

Standalone model entrypoints forward run-based evaluation to the same pipeline,
for example:

```powershell
python -m var_expert_inr.methods.apmgsrn.cli evaluate --run runs/<run> --timesteps 0:10
python -m var_expert_inr.methods.neural_expert.cli evaluate --run runs/<run> --metrics psnr,memory
```

The checked-in formal experiment matrix contains 354 configs. The established
matrix families have main Combustion (`40NH3_1`) experiments, while STSR-INR
adds a joint RedSea experiment. SIREN, CoordNet, and MoE-INR cover
all 13 exported fields, including the three-component `Velocity` target. Models
whose published implementation requires scalar outputs cover the other 12 fields;
MVNet jointly models those 12 scalar fields, while MC-INR and VarExpert jointly
model all 13 fields. The formal Ionization SIZE matrix contains `Size082`,
`Size163`, `Size326`, and `Size652` for VarExpert, CoordNet, MoE-INR, fV-SRN,
MINER, and STSR-INR, plus a VarExpert DWA loss-balancing main config.
All primary training stages except MVNet, NeuralExpert, and STSR-INR consume 14.4 billion physical samples. NeuralExpert uses 960 million sampled points (60,000 optimizer steps at 16,000 points per step).
InstantVNR accumulates four 16,000-sample batches into an approximately
paper-sized 64,000-sample optimizer update; other unified baselines retain their
configured update batches. MVNet uses its method-specific 300 epochs,
2,048-sample batches, and 1,500 random batches per epoch (921.6 million
samples). Model-size tiers use all parameters at two bytes per parameter
(theoretical FP16 size). The Ionization tier is a total five-variable budget:
single-target models receive one fifth of `0.82/1.63/3.26/6.52 MiB`, while
VarExpert and STSR-INR receive the full tier. MINER estimates its adaptive size
using two retained blocks per scale and timestep across all 100 timesteps.

Experiment assets are organized by purpose. Formal configs live in
`configs/main/` and `configs/rd_curve/`; exploratory, ablation, and sensitivity
studies live in their matching `configs/<category>/` directory. Method names
such as `fV-SRN` are preserved inside each category so they continue to match
the paper and run metadata.

The default selection file, `scripts/main/all_configs.list`, contains the full
formal matrix. Comment out or delete paths in that file to select a subset. The
remaining entries still follow the model grouping and main-before-RD-curve
order defined by the runner.

Run the selected configs in the `compression` conda environment:

```bash
bash scripts/main/run_all.sh
```

An alternate list can be supplied without modifying the default file:

```bash
CONFIG_LIST_FILE=scripts/my_configs.list bash scripts/main/run_all.sh
```

Two ready-made subsets are also provided:

```bash
# Main experiments only: 266 configs
bash scripts/main/run_all.sh

# RD-Curve experiments only: 88 Size-tier configs
bash scripts/rd_curve/run.sh
```

The script continues after individual failures and writes per-config logs plus
`status.tsv` and `failed.txt` under `batch_logs/<timestamp>/`. Reuse a batch by
passing its directory as `BATCH_LOG_ROOT`:

```bash
BATCH_LOG_ROOT=batch_logs/20260803_120000 bash scripts/main/run_all.sh
```

The final status recorded for each config path is authoritative. `ok` paths are
skipped; `running`, `failed`, missing, or invalid states are retrained from the
beginning without a checkpoint. Every retry gets an `attempt-N.log`, and a
`DRY_RUN=1` never writes status rows. Resume comparison is deliberately based
only on the config path: an old `ok` row still skips that path even if the YAML
contents have since changed.

The combined SIREN + NeuralExpert non-Ionization entrypoint runs the 13 SIREN
Combustion targets, then the 16 NeuralExpert manager pretrains and their 16
Bathymetry/Combustion main runs. It defaults to five parallel training jobs;
NeuralExpert full-dataset PSNR evaluations run serially afterward. SIREN keeps
its final deterministic 10% PSNR probe. Results are collected in
`experiment_psnr.tsv` beneath the batch log directory.

```bash
bash scripts/main/run_neural_expert_non_ionization.sh
DRY_RUN=1 bash scripts/main/run_neural_expert_non_ionization.sh
BATCH_LOG_ROOT=batch_logs/<existing-batch> bash scripts/main/run_neural_expert_non_ionization.sh
MAX_PARALLEL_JOBS=5 CONDA_ENV=compression bash scripts/main/run_neural_expert_non_ionization.sh
```

The CoordNet-Combustion + MVNet-Katrina + STSR-INR-RedSea entrypoint contains
15 configs: the 13 independent CoordNet Combustion targets, one joint MVNet
Katrina config, and one joint five-attribute STSR-INR RedSea config. Stages run
in that order, with at most five concurrent jobs by default. RedSea data is read
from the sibling `Datasets/Ocean/train` directory; set `DATASETS_ROOT` when the
datasets directory lives elsewhere.

```bash
bash scripts/main/run_selected_datasets.sh
DRY_RUN=1 bash scripts/main/run_selected_datasets.sh
BATCH_LOG_ROOT=batch_logs/<existing-batch> bash scripts/main/run_selected_datasets.sh
MAX_PARALLEL_JOBS=5 CONDA_ENV=compression bash scripts/main/run_selected_datasets.sh
DATASETS_ROOT=/path/to/Datasets bash scripts/main/run_selected_datasets.sh
```

Size-structure exploration is generated and run independently of the formal
matrix:

```bash
python scripts/ablation/generate_architecture.py
bash scripts/ablation/run_architecture.sh
```

This creates 126 Size163 configs under `configs/ablation/architecture/`. Their 50
epoch-equivalent budgets and fixed 1% PSNR probes are isolated under
`runs/exploration/<exp_id>/<timestamp>/`; batch logs go to
`batch_logs/exploration/<timestamp>/`. Each run writes
`metrics/exploration_psnr.tsv` at progress `5/50` through `50/50`. The batch
also writes `exploration_summary.tsv`, including the averaged trajectory,
final PSNR, NaN/Inf flag, scope count, and final training status. Resume works
the same way by setting `BATCH_LOG_ROOT` to an existing exploration batch.

When running without installation, this repository ships a small package shim so
`python -m var_expert_inr.cli` works directly from the repo root.

Each train run writes outputs into `runs/<exp_id>/<timestamp>/`, including
`checkpoints/`, `configs/`, `logs/`, `metrics/`, and `predictions/`.
InstantNGP is a pure PyTorch four-dimensional multiresolution hash grid. It
consumes the framework's existing `[-1, 1]` XYZT coordinates, accumulates 16
physical batches per optimizer update, and applies its learning-rate milestones
per optimizer step.
InstantVNR is a separate pure-PyTorch 4D extension of the
[official InstantVNR](https://github.com/VIDILabs/instantvnr) neural
representation. It uses the released eight-level, eight-feature HashGrid and
four-layer ReLU MLP defaults, L1 loss, Adam, and delayed piecewise exponential
decay while retaining this framework's XYZT and target scaling conventions. It
is intended for neural-field compression comparisons; the original native CUDA
renderer, macro-cell acceleration, out-of-core sampler, and interactive online
training system are outside this reproduction's scope. See the
[paper](https://arxiv.org/abs/2207.11620) for the complete rendering system.
MVNet is a shared multi-output residual SIREN that consumes the existing
`[-1, 1]` XYZT coordinates and predicts every scalar variable in the dataset in
one forward pass. Its output-column order is stored in each checkpoint and
reused by unified prediction and evaluation.
The resolved effective config is saved as `runs/<exp_id>/<timestamp>/configs/config.yaml`.

For unified-engine models, opt in to training-step peak memory measurement with:

```yaml
log:
  memory:
    enabled: true
    sample_interval_seconds: 0.01
```

The run writes `metrics/training_memory.json` with main-process RSS and PyTorch
peak CUDA allocated memory. Measurement covers data fetch, transfer, forward,
backward, and optimizer work, including pretraining and gradient-accumulation
microbatches. Validation, probes, checkpoints, and post-training prediction are
excluded. The feature is disabled by default because isolating CUDA step peaks
requires synchronization.

## RealPDEBench combustion tools

Run the standalone script with the `compression` conda environment. Inspect
real or numerical trajectories and optionally compute exact value statistics:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\tools\combustion.py inspect `
  --dataset-dir "E:\Research\Project\Scientific Compression\INR\Datasets\RealPDEBench\combustion\hf_dataset\real" `
  --scan-values
```

Render a full-resolution trajectory as a fixed-scale PNG sequence and MP4:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\tools\combustion.py render `
  --dataset-dir "E:\Research\Project\Scientific Compression\INR\Datasets\RealPDEBench\combustion\hf_dataset\real" `
  --sim-id 0NH3_1.h5 `
  --frames all `
  --scale global-minmax `
  --sampling-fps 4000 `
  --video-fps 30
```

Export all 15 numerical channels from `40NH3_1.h5` as 13 normalized structured-
volume targets (the three velocity components become one vector target):

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\tools\combustion.py export-volume `
  --dataset-dir "E:\Research\Project\Scientific Compression\INR\Datasets\RealPDEBench\combustion\hf_dataset\numerical" `
  --sim-id 40NH3_1.h5 `
  --output "data\Volume\Combustion"
```

The real dataset contains OH* chemiluminescence intensity rather than a
temperature field. Its Arrow rows store raw float32 `(T,H,W)` byte payloads;
the renderer preserves the native 128x128 camera orientation and derives the
displayed time from the configured sampling rate because real rows do not
contain coordinate or time arrays. Render outputs are written below
`runs/visualizations/combustion/` by default. Numerical targets are flattened
in C order from `(T,Y,X,C)`, so x changes fastest, followed by y and t. Their
normalization parameters and physical coordinate ranges are stored in the
generated `data/Volume/Combustion/manifest.json`.

## Katrina dynamic wet-point export

The Katrina source arrays under the adjacent Ocean dataset contain all 417,642
ADCIRC nodes for every timestep. Inspect the dynamic wet-point mask, defined by
finite `fort63` values other than the `-99999` dry sentinel, with:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\tools\katrina_wet.py inspect `
  --input-dir "E:\Research\Project\Scientific Compression\INR\Datasets\Ocean\train"
```

Export the normalized wet samples to the repository data tree with:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\tools\katrina_wet.py export `
  --input-dir "E:\Research\Project\Scientific Compression\INR\Datasets\Ocean\train" `
  --output "data\Mesh\Katrina_Wet"
```

The exporter normalizes X/Y/Z/T independently, normalizes scalar targets
independently, and uses one joint range for all three velocity components. It
also writes `wet_node_indices.npy`, `frame_offsets.npy`, and `manifest.json` so
the variable-size frames can be mapped back to the original fixed mesh.

When `predict` or `evaluate` runs without an explicit checkpoint, it reuses the
latest timestamped run under the matching `exp_id`.
For `var_expert`, architecture fields that remain at default values are omitted
from the saved effective config and log output.

`mc_inr` is provided as a standalone subsystem under `var_expert_inr.methods.mc_inr`.
It uses the same run directory layout and evaluation outputs as the unified
framework, but it does not participate in the main `var_expert_inr.cli` model
registry or training engine.

`apmgsrn` is also provided as a standalone subsystem under
`var_expert_inr.methods.apmgsrn`. It currently only supports single-target `ionization`
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

`ecnr` is an integrated three-scale reproduction for single-target temporal
volumes. It clusters normalized spatial blocks deterministically, trains packed
local SIREN MLPs from coarse content to fine residuals, applies block-guided
pruning and global codebook quantization, then runs a halo-correct tiled 3D CNN.
Its `.ecnr` artifact Huffman-encodes masks, assignments, and quantization
labels. The three primary scale stages consume the standard 14.4-billion-sample
experiment budget. Scale-boundary checkpoints can be resumed with
`train --config <config> --resume <scale_checkpoint.pth>`; compact inference
uses `predict/evaluate --config <config> --artifact <model.ecnr>`. Each run also
writes `metrics/training_cost.json`, including logical samples, packed-MLP
prediction counts, optimizer steps, phase timings, and peak CUDA memory.

The checked-in Ionization configs describe a full 100-timestep
`(T,Z,Y,X)=(100,248,248,600)` sequence. The sample target files currently in
this repository contain only two timesteps; they must be replaced with the
full sequence before that config can run. The loader deliberately reports this
shape mismatch instead of silently reinterpreting the data.
