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
  --metrics psnr,ssim,lpips,decode_time,memory \
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
| `decode_time` | Not required | No | Fresh decode timing, excluding rendering and metric work. |
| `memory` | Not required | No | Process RSS and, when available, CUDA allocated/reserved peaks. |
| `--render` | Optional | Required | Selected prediction frames; GT frames are added when available. |

PSNR, SSIM, and LPIPS fail before decoding when ground truth is missing,
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

Packaged profiles currently cover Ionization volume rendering and Katrina node
rendering. Use `--eval-config <profile.yaml>` for another dataset or view.

A volume profile declares `kind: volume`, layout, renderer options, and optional
target-to-preset mappings. Volume rendering requires the sibling VolumeVis
package. A node profile declares `kind: node`, point/cell association, camera
and color settings, and one of:

- `mesh_path` or `mesh_path_template` for VTK/VTU or ADCIRC `fort.14`; or
- both vertices and cells NumPy paths, optionally as timestep templates.

There is no point-cloud fallback. Prediction-only rendering must provide a
fixed `clim` or target-specific `target_clims`; otherwise color limits are
derived from ground truth.

## Reports and caching

Each new evaluation writes:

```text
<run>/evaluations/<timestamp>/
├── manifest.json
├── metrics.json
├── metrics.csv
├── logs/evaluate.log
└── renders/<target>/...   # when rendering is requested
```

Quality and render results may be reused when the source, selections, render
profile, and ground-truth fingerprints match. `--overwrite` bypasses that
cache. `decode_time` and `memory` always perform fresh measurement and are not
served from the quality-result cache.
