# VarExpert-INR

Unified INR framework for node and volume datasets.

All loaded node coordinates and targets, and all loaded volume targets, must
already be scaled into `[-1, 1]`. VarExpert-INR no longer performs runtime
normalization and will raise an error if loaded data falls outside that range.

## Commands

From the repository root:

```bash
python -m var_expert_inr.cli train --config configs/VarExpert/ionization.yaml
python -m var_expert_inr.cli train --config configs/SIREN/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/CompactNGP/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/InstantNGP/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/MVNet/ionization.yaml
python -m var_expert_inr.cli train --config configs/FA-TR-INR/ionization__GT.yaml
python -m var_expert_inr.mc_inr.cli train --config configs/MC-INR/ionization.yaml
python -m var_expert_inr.neural_expert.cli train --config configs/NeuralExpert/ionization__GT__managerpretrain.yaml
python -m var_expert_inr.neural_expert.cli train --config configs/NeuralExpert/ionization__GT.yaml
python -m var_expert_inr.dc_inr.cli train --config configs/DC-INR/ionization__GT.yaml
python -m var_expert_inr.apmgsrn.cli train --config configs/APMGSRN/ionization__GT.yaml
python -m var_expert_inr.fv_srn.cli train --config configs/fV-SRN/ionization__GT.yaml
python -m var_expert_inr.rmdsrn.cli train --config configs/RMDSRN/ionization__GT.yaml
python -m var_expert_inr.cli train --config configs/ECNR/ionization__GT.yaml
```

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
python -m var_expert_inr.apmgsrn.cli evaluate --run runs/<run> --timesteps 0:10
python -m var_expert_inr.neural_expert.cli evaluate --run runs/<run> --metrics psnr,memory
```

The checked-in formal experiment matrix contains 356 configs. General models cover
Bathymetry, Katrina, and Ionization; volume-only models cover Ionization. Every
single-target model has one config per attribute, and Ionization additionally
has `Size082`, `Size163`, `Size326`, `Size652`, and `Size1304` variants plus a
VarExpert DWA loss-balancing config.
All primary training stages except MVNet consume 14.4 billion samples with an
effective batch size of 16,000. MVNet uses its method-specific 300 epochs,
2,048-sample batches, and 1,500 random batches per epoch (921.6 million
samples). Model-size tiers use all parameters at two bytes per parameter
(theoretical FP16 size). The Ionization tier is a total five-variable budget:
single-target models receive one fifth of `0.82/1.63/3.26/6.52/13.04 MiB`,
while VarExpert and MC-INR receive the full tier. APMGSRN counts all 100
timestep models toward that one-variable share; DC-INR receives the already
divided target through `compression.target_size_mib`.

Run the complete matrix sequentially in the `compression` conda environment:

```bash
bash scripts/run_all_configs.sh
```

The script continues after individual failures and writes per-config logs plus
`status.tsv` and `failed.txt` under `batch_logs/<timestamp>/`. Reuse a batch by
passing its directory as `BATCH_LOG_ROOT`:

```bash
BATCH_LOG_ROOT=batch_logs/20260803_120000 bash scripts/run_all_configs.sh
```

The final status recorded for each config path is authoritative. `ok` paths are
skipped; `running`, `failed`, missing, or invalid states are retrained from the
beginning without a checkpoint. Every retry gets an `attempt-N.log`, and a
`DRY_RUN=1` never writes status rows. Resume comparison is deliberately based
only on the config path: an old `ok` row still skips that path even if the YAML
contents have since changed.

Size-structure exploration is generated and run independently of the formal
matrix:

```bash
python scripts/generate_size_exploration_configs.py
bash scripts/run_size_exploration.sh
```

This creates 141 Size163 configs under `configs_exploration/`. Their 50
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
CompactNGP runs additionally write a baked FP16/2-bit representation under
`artifacts/`. Unified `predict` and `evaluate` accept it through `--artifact`;
checkpoint inference bakes the learned confidence tables once before querying.
InstantNGP is a pure PyTorch four-dimensional multiresolution hash grid. It
consumes the framework's existing `[-1, 1]` XYZT coordinates, accumulates 16
physical batches per optimizer update, and applies its learning-rate milestones
per optimizer step.
MVNet is a shared multi-output residual SIREN that consumes the existing
`[-1, 1]` XYZT coordinates and predicts every scalar variable in the dataset in
one forward pass. Its output-column order is stored in each checkpoint and
reused by unified prediction and evaluation.
FA-TR-INR runs use the unified checkpoint path without a separate inference
artifact. The model contracts five independent sine-MLP factors in the fixed
`x -> y -> f -> z -> t -> x` Tensor Ring order and directly consumes the
framework's existing `[-1, 1]` coordinates and targets.
The resolved effective config is saved as `runs/<exp_id>/<timestamp>/configs/config.yaml`.

## RealPDEBench combustion tools

Run the standalone script with the `compression` conda environment. Inspect
real or numerical trajectories and optionally compute exact value statistics:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\combustion.py inspect `
  --dataset-dir "E:\Research\Project\Scientific Compression\INR\Datasets\RealPDEBench\combustion\hf_dataset\real" `
  --scan-values
```

Render a full-resolution trajectory as a fixed-scale PNG sequence and MP4:

```powershell
D:\Anaconda3\envs\compression\python.exe scripts\combustion.py render `
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
D:\Anaconda3\envs\compression\python.exe scripts\combustion.py export-volume `
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
