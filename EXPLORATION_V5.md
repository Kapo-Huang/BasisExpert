# Exploration v5: fV-SRN and InstantVNR stability

Exploration v5 is an isolated 42-config follow-up for Ionization `GT`, `H2`,
and `H_plus`. It does not modify formal configs or earlier exploration runs.
Every run uses seed 42, 50 epoch-equivalents, and fixed probes of at most
100,000 samples at progress 5, 10, ..., 50.

## Matrix

- fV-SRN: 24 configs. The formal Size163 and previous grid-heavy structures
  each run the Cartesian product of learning rates `1e-2` and `5e-3` with
  StepLR horizons 100 and 20; all use `gamma=0.5`.
- InstantVNR: 18 configs. The released HashGrid/MLP structure is reduced to
  170,972 parameters to match the 171,105-parameter fV-SRN Size163 structure:
  `log2_hashmap_size=11` and `hidden_features=105`, with levels, features per
  level, resolutions, decoder depth, and four-batch gradient accumulation kept
  fixed. Profiles cover the current optimizer, lower learning rates, faster
  decay, gradient clipping, and MSE versus L1.

Generate the checked-in configs:

```bash
python scripts/generate_exploration_v5_configs.py
```

Run them in the `compression` environment:

```bash
MAX_PARALLEL_JOBS=5 bash scripts/run_exploration_v5.sh
```

The default physical GPU is `cuda:0`. Select another GPU with the `DEVICE`
environment variable:

```bash
DEVICE=cuda:1 MAX_PARALLEL_JOBS=5 bash scripts/run_exploration_v5.sh
```

The runner validates the `cuda:N` form and maps the selected physical GPU
through `CUDA_VISIBLE_DEVICES`, so both fV-SRN and InstantVNR use the same card.

Set `DRY_RUN=1` to inspect all commands. Reuse `RUN_TOKEN` or set
`BATCH_LOG_ROOT` to resume a batch; configs already marked `ok` are skipped.

## Validation and outputs

The batch writes `status.tsv`, `failed.txt`, `exploration_summary.tsv`,
`profile_summary.tsv`, and `needs_attention.txt` under
`batch_logs/exploration_v5/<RUN_TOKEN>/`.

A run needs attention if a probe is missing or non-finite, final PSNR drops
more than 1 dB from its peak, or final PSNR improves less than 0.1 dB from the
first probe. fV-SRN also fails a per-target historical guard when it finishes
more than 1 dB below the previous Size163 grid-heavy result. A profile is
eligible only when all three targets are clean. Profiles are ranked within
each method by median paired improvement over its control, then by lower peak
drop and higher final PSNR.

With the default `STRICT_VALIDATION=1`, the batch fails only when a method has
no eligible profile. Individual rejected candidates remain visible in the
reports. Set `STRICT_VALIDATION=0` to produce reports without this final gate.
