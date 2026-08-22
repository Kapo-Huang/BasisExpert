# Exploration v4: RMDSRN schedules and CoordNet depth

Exploration v4 is an isolated 81-config follow-up to exploration v3. It does
not modify the formal configs or any v3 config/result. All runs use seed 42
and fixed 100,000-sample probes every five epoch-equivalents.

## Matrix

- CoordNet: 54 configs.
  - Equal-parameter `res2`, `res3`, `res5`, `res7`, and `res10` profiles for
    Size326, Size652, and Size1304 on GT, H_plus, and He (45 configs).
  - Size1304 learning-rate, clipping, and combined controls (9 configs).
- RMDSRN: 27 configs.
  - Corrected 900,000-step LR/lambda schedule horizons at `lambda_max=10`
    for all five Sizes on GT, H_plus, and PD (15 configs).
  - `lambda_max=1` and reconstruction-only `lambda_max=0` controls for
    Size082 and Size1304 (12 configs).

CoordNet depth profiles retain the formal learning rate and scheduler. Their
integer `init_features` values minimize parameter-count error relative to the
matching formal res10 model. RMDSRN trains for 75,000 steps but advances only
the first 75,000 steps of its formal 900,000-step schedules.

## Generate

```bash
python scripts/ablation/generate_depth_and_regularization.py
```

This recreates only `configs/ablation/depth_and_regularization/`.

## Run

```bash
conda activate compression
MAX_PARALLEL_JOBS=5 bash scripts/ablation/run_depth_and_regularization.sh
```

Five-way parallelism is the default. The runner uses a bounded queue within
each of four stages, supports `DRY_RUN=1`, and resumes from successful status
entries when the same `RUN_TOKEN` is reused.

Outputs are written under `batch_logs/exploration_v4/<RUN_TOKEN>/`:

- `status.tsv` and `failed.txt`
- `exploration_summary.tsv`
- `profile_summary.tsv`
- `needs_attention.txt`

Strict validation flags missing/non-finite probes, missing progress 50, a
final PSNR drop greater than 1 dB from the peak, or less than 0.1 dB gain from
the first probe. Set `STRICT_VALIDATION=0` to generate reports without making
attention items fail the batch.

RMDSRN probe details additionally contain member MSE, variance KL, lambda,
the weighted variance/member ratio, variance-error Pearson correlation, and
top-1%/top-5% error hit rates. Every v4 run retains one best-probe checkpoint
and a separate final checkpoint.

The profile report ranks eligible CoordNet profiles by clean-run count, then
median matched PSNR improvement over `res10_base_lr`, then shallower residual
depth. A profile with any run dropping more than 3 dB from its peak is marked
ineligible. RMDSRN profiles report reconstruction and uncertainty metrics but
are not promoted into the formal configuration automatically.
