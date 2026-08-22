# Exploration v3: all Size configs

Exploration v3 is a 50-epoch-equivalent smoke matrix for every formal Size
configuration listed in `scripts/run_rd_curve_configs.list`. It contains 210
configs across five sizes and nine model families. Model architecture, target,
partition, and compression settings are copied unchanged from the formal
configs; only training length, logging/checkpoint cadence, experiment IDs, and
output roots are changed.

## Generate configs

```bash
python scripts/exploration/generate_rd_curve_smoke.py
```

This rebuilds the isolated `configs/exploration/rd_curve_smoke/` directory. It does not
modify the formal `configs/` directory.

## Run

```bash
conda activate compression
MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_rd_curve_smoke.sh
```

The runner executes NeuralExpert manager-pretraining configs before the
matching reconstruction configs. Other families are grouped by family and
Size. Adjust `MAX_PARALLEL_JOBS` to fit available GPUs/resources.

To inspect all 210 commands without training:

```bash
DRY_RUN=1 MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_rd_curve_smoke.sh
```

To resume an interrupted batch, reuse the run token printed in the batch log
directory name:

```bash
RUN_TOKEN=YYYYMMDD_HHMMSS MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_rd_curve_smoke.sh
```

Successful entries already recorded in that token's `status.tsv` are skipped.

## Validation

Every config records fixed-sample PSNR at 5-epoch-equivalent intervals through
progress 50. After training, the runner writes:

- `batch_logs/exploration_v3/<RUN_TOKEN>/exploration_summary.tsv`
- `batch_logs/exploration_v3/<RUN_TOKEN>/needs_attention.txt`
- `batch_logs/exploration_v3/<RUN_TOKEN>/failed.txt`

Validation is strict by default. The batch exits nonzero when a run fails or is
incomplete, PSNR is non-finite, final PSNR drops more than 1 dB from its peak,
or final PSNR improves less than 0.1 dB from the first probe. Thresholds can be
changed without regenerating configs:

```bash
COLLAPSE_THRESHOLD_DB=0.5 MINIMUM_GAIN_DB=0.2 bash scripts/exploration/run_rd_curve_smoke.sh
```

Use `STRICT_VALIDATION=0` only when an attention report is desired without a
nonzero batch exit code.

## 50-epoch-equivalent mapping

| Family | v3 training length |
| --- | ---: |
| SIREN, CoordNet, MoE-INR, VarExpert | 50 epochs |
| MC-INR | 50 epochs, 5 meta-iterations, 50 fine-tune epochs |
| fV-SRN | 50 epochs |
| APMGSRN | 750 iterations |
| RMDSRN | 75,000 steps |
| NeuralExpert manager pretraining | 2,500 iterations |
| NeuralExpert reconstruction | 75,000 iterations |
