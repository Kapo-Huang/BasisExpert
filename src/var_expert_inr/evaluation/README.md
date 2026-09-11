# Evaluation, metrics, and rendering

[English](README.md) | [简体中文](README.zh-CN.md)

The run-based evaluator provides one interface for shared-engine and
self-contained methods. It can decode checkpoints, read saved predictions,
select targets and timesteps, compute quality or performance metrics, and
render matching frames.

## Basic usage

```bash
python -m var_expert_inr.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --result-root EvalResult \
  --evaluation-id quality \
  --metrics psnr,ssim,lpips,error,decode_time,training_time,training_memory,inference_time,memory \
  --targets GT,H2 \
  --timesteps 0,10:30,40:99:10
```

The unified CLI accepts either `--run` or `--config`. With `--config`, it uses
the latest timestamped run for that config's `exp_id` unless an explicit
checkpoint identifies a run. Standalone method CLIs expose the same run-based
arguments through their `evaluate` command.

## Metrics and prerequisites

The default metric is `psnr`.

| Metric or action | Ground truth | Rendering | Main result |
| --- | --- | --- | --- |
| `psnr` | Required | No | MSE, MAE, and PSNR summaries. |
| `ssim` | Required | Required | SSIM on matched GT and prediction images. |
| `lpips` | Required | Required | LPIPS on matched GT and prediction images. |
| `error` | Required | Optional | Per-frame absolute-error and normalized error-percentage statistics; with `--render`, a fixed-scale error image. |
| `pearson_error` | Required | No | Absolute error in the sampled cross-variable Pearson matrix. |
| `mi_error` | Required | No | Absolute error in the sampled cross-variable histogram-MI matrix, in nats. |
| `decode_time` | Not required | No | Fresh decode timing, excluding rendering and metric work. |
| `training_time` | Not required | No | Runs a short training probe from the archived config and extrapolates the configured common sample budget. |
| `training_memory` | Not required | No | CPU RSS and CUDA allocated/reserved peaks across the complete bounded training probe. |
| `inference_time` | Not required | No | Uniformly decodes 10% of timesteps from the archived checkpoint and projects checkpoint-load plus full reconstruction time. |
| `memory` | Not required | No | Process RSS and, when available, CUDA allocated/reserved peaks. |
| `--render` | Optional | Required | Selected prediction frames; GT frames are added when available. |

PSNR, SSIM, LPIPS, and error analysis fail before decoding when ground truth is missing,
unreadable, or shape-incompatible. Checkpoint-based performance evaluation can
construct coordinates without targets; node evaluation still needs its
coordinate array.

Install optional dependencies with:

```bash
python -m pip install -e ".[evaluation]"
python -m pip install -e "../Vis[lpips]"  # Volume rendering only
```

## Selection syntax

- `--targets all` selects every configured target. Otherwise use a comma-
  separated list. `all` cannot be combined with explicit names.
- `--timesteps all` selects every timestep.
- A timestep token is `N` or an inclusive `start:end[:step]` range. Comma-
  separated combinations preserve first occurrence and reject out-of-range
  indices.

For example, `0,10:14,20:40:10` selects `0, 10, 11, 12, 13, 14, 20, 30, 40`.

## Evaluation source

`--source` accepts `auto`, `checkpoint`, or `prediction`.

- Explicit `--checkpoint` or `--prediction` paths take precedence.
- `auto` prefers the canonical/final `.pth` checkpoint under `checkpoints/`,
  then falls back to a saved `.npy` prediction under `predictions/`.
- `checkpoint` fails if no suitable checkpoint exists.
- Performance metrics requested from a saved prediction measure prediction-file
  access rather than model decoding and are labeled accordingly in the report.

The evaluator uses the effective config saved in the run. For legacy absolute
GT paths, it also checks portable dataset locations adjacent to the repository
and under the local `data/` tree.

## Rendering profiles

Built-in profiles cover Ionization, Combustion, RedSea, and Katrina. The
profile `renderer` is `volume`, `image2d`, or `mesh`; use
`--eval-config <profile.yaml>` to override a built-in view.

A volume profile declares `kind: volume`, layout, renderer options, and optional
target-to-preset mappings. Volume rendering requires the sibling VolumeVis
package. A node profile declares `kind: node`, point/cell association, camera
and color settings, and one of:

- `mesh_path` or `mesh_path_template` for VTK/VTU or ADCIRC `fort.14`; or
- both vertices and cells NumPy paths, optionally as timestep templates.

Combustion uses the `image2d` renderer and the default `viridis` color map.
The built-in RedSea mesh view selects the surface coordinate layer and maps
its values through the VTP `wet_mask_surface` array. Prepare its local,
git-ignored mesh once before rendering:

```powershell
New-Item -ItemType Directory -Force data/Mesh/RedSea/render
Copy-Item E:/Research/Project/Scientific Compression/INR/Datasets/RedSea_SciVisContest2020/0001/0001/paraview/surface_vtp/surface_0000.vtp `
  data/Mesh/RedSea/render/surface_0000.vtp
```

For another checkout, copy the same SciVis export to the destination path
shown above. The evaluator reports that expected path before decoding when the
asset is absent. Legacy runs named `bathymetry` use the RedSea profile.

There is no point-cloud fallback. Prediction-only rendering must provide a
fixed `clim` or target-specific `target_clims`; otherwise color limits are
derived from ground truth.

Error analysis operates directly in the normalized target space. Scalar
fields use `abs(prediction - ground_truth)` and vector fields use the
point-wise L2 norm. The reported percentage is `absolute_error / 2 * 100`,
using the fixed normalized GT range `[-1, 1]`. Each timestep reports exact
mean, maximum, p95, and p99 values; target summaries pool only mean and maximum.
Point-wise error arrays are not persisted.

Use `--error-vmin` and `--error-vmax`, or the matching `evaluation` YAML keys,
to set one comparison scale in percentage units. Defaults are `0` and `5`.
Error images clamp to this range and use white-to-red. Ionization error volumes
also map opacity from zero at no error to one at `error_vmax`.

## Reports and caching

Every evaluation recipe has a unique `evaluation_id`. Evaluation directories
contain only reports, state, logs, and relative artifact references:

```text
EvalResult/
├── artifacts/
│   ├── ground_truth/<dataset>/<target>/<render_key>/gt_tXXXX.png
│   ├── prediction/<dataset>/<model>/<source_key>/<target>/<render_key>/pred_tXXXX.png
│   ├── error/<dataset>/<model>/<comparison_key>/<target>/<render_key>/error_tXXXX.png
│   └── dependency_gt/<dataset>/<dependency_key>/
├── evaluations/<evaluation_id>/<Result-relative-path>/
│   ├── manifest.json
│   ├── metrics.json
│   ├── metrics.csv
│   ├── progress.json
│   └── logs/evaluate.log
├── batches/<evaluation_id>/
└── migration/
```

GT, prediction, and error images are written directly to the shared artifact
store. Their keys include content fingerprints, the normalized render profile,
external mesh/preset resources, and the renderer schema version. Paths stored
in reports are relative to `EvalResult`, so moving the whole tree preserves
references. Concurrent writers use per-artifact locks and atomic replacement.

`--overwrite` refreshes the current evaluation and model-derived artifacts; it
does not invalidate an unchanged GT artifact. `decode_time` and `memory` always
perform fresh measurements. `training_time`, `training_memory`, and `inference_time` are cached by
experiment and evaluation ID; `--overwrite` reruns their probes. GT
materialization is lazy: the first request renders a frame and all later
evaluations reference that same artifact without
copying it into their own directories.

Run a batch recipe and inspect or maintain the result tree with:

```bash
python scripts/evaluation/run_batch.py --config configs/evaluation/evaluation_result.yaml
python scripts/evaluation/run_batch.py --config configs/evaluation/evaluation_result_runtime.yaml
python scripts/evaluation/run_batch.py --config configs/evaluation/evaluation_training_memory.yaml
python scripts/evaluation/manage_eval_result.py status
python scripts/evaluation/manage_eval_result.py verify
python scripts/evaluation/manage_eval_result.py prune        # dry run
python scripts/evaluation/manage_eval_result.py prune --apply
```

Batch configuration uses only top-level `result_root` and `evaluation_id` for
layout selection. See `scripts/evaluation/examples/` for quality, figure,
error, and dependency invocations.

## Cross-variable dependency evaluation

`pearson_error` and `mi_error` are group-level metrics and run separately from
per-target quality/render metrics. Every timestep uses the same stable 20%
spatial-index sample for GT and reconstruction. Pearson is accumulated in
float64; MI uses GT quantile-bin edges cached and reused for reconstruction.
Only valid upper-triangle target pairs are averaged, first within a timestep
and then equally across selected timesteps.

Build the reusable dependency GT caches once, then use the same batch entry
point for archived groups:

```bash
python scripts/evaluation/build_dependency_gt_cache.py --dataset all
python scripts/evaluation/run_batch.py \
  --config configs/evaluation/evaluation_result_dependency.yaml
```

Joint runs are evaluated directly. A single-target run resolves the canonical
sibling runs under its parent directory; missing or duplicate targets are
errors. Outputs are isolated by `evaluation_id` under `EvalResult/evaluations`
and include
`metrics.json`, `metrics.csv`, `dependency_metrics.npz`, and a manifest with
all source/cache fingerprints.
