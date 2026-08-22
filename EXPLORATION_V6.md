# Exploration v6: ECNR main-training calibration

Exploration v6 is an isolated 18-config calibration of the formal ECNR
Ionization training recipe on `GT`, `H2`, and `H_plus`. It does not modify the
formal ECNR configs or earlier exploration runs. Every run uses seed 42 and
executes the complete three-scale ECNR pipeline, including cumulative pruning,
quantization fine-tuning, boundary-CNN training, artifact creation, decoding,
and full-volume evaluation.

## Matrix

The official model, clustering, pruning sparsities, and quantization format are
fixed. Six training profiles are paired across all three targets:

- `official_control`: formal optimizer settings.
- `lr5e4`: primary learning rate `5e-4`.
- `lr2e3`: primary learning rate `2e-3`.
- `no_weight_decay`: primary weight decay disabled.
- `pruning_gamma09`: retain more learning rate after pruning events.
- `quant_lr5e5`: quantization fine-tuning learning rate `5e-5`.

The smoke budget uses 50 epochs per scale, 300 batches per epoch, pruning at
epochs 15/23/30/38, eight bounded quantization fine-tuning epochs, and ten CNN
epochs. This is 144,000,000 primary logical samples, or 1% of the formal ECNR
primary budget, while still exercising every production stage.

Generate the checked-in configs:

```bash
python scripts/generate_exploration_v6_configs.py
```

Run the batch in the `compression` environment:

```bash
bash scripts/run_exploration_v6.sh
```

ECNR processes full volumes and builds large temporary caches, so the default
is `MAX_PARALLEL_JOBS=1`. Override this only when host RAM and scratch storage
have been checked. Select another physical GPU with `DEVICE=cuda:N`.

Reuse `RUN_TOKEN` or set `BATCH_LOG_ROOT` to resume a batch; configs whose
latest status is `ok` are skipped.

## Outputs and selection

Each config evaluates its compact `.ecnr` artifact. The batch writes
`status.tsv`, `failed.txt`, `exploration_summary.tsv`, `profile_summary.tsv`,
and `needs_attention.txt` under `batch_logs/exploration_v6/<RUN_TOKEN>/`.
Full-volume predictions are temporary because `save_predictions=false`; they
are removed after metrics are written, while artifacts and metrics are kept.

A run needs attention when training or metrics are missing, metrics are
non-finite, artifact size or compression ratio is invalid, training cost is
missing, or its final PSNR is more than 1 dB below the paired official control.
A profile is eligible only when all three targets are clean. Eligible profiles
are ranked by median paired PSNR improvement, then compression ratio and
training time. `STRICT_VALIDATION=1` fails the batch when no eligible profile
exists.
