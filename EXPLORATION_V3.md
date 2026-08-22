# Exploration v3: RD-curve Size smoke

Exploration v3 is a 50-epoch-equivalent smoke matrix built from the formal
RD-curve selection in `scripts/rd_curve/configs.list`. Model architecture,
target, and compression settings are copied from the formal configs; only the
short training schedule, logging, experiment IDs, and output roots change.

## Generate configs

```bash
python scripts/exploration/generate_rd_curve_smoke.py
```

This rebuilds the isolated `configs/exploration/rd_curve_smoke/` directory. It does not
modify the formal `configs/` directory.

MINER can be regenerated independently without replacing any other family:

```bash
python scripts/exploration/generate_miner_rd_curve_smoke.py
```

This writes the four formal MINER sizes and five current Ionization targets to
`configs/exploration/rd_curve_smoke/MINER/`.

## Run

```bash
conda activate compression
MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_rd_curve_smoke.sh
```

To run only MINER, use the independent entrypoint. It keeps the same
`runs/exploration_v3` result root and writes batch logs under
`batch_logs/exploration_v3_miner`:

```bash
MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_miner_rd_curve_smoke.sh
```

Both runners accept `--env original` and `--env autodl`. Adjust
`MAX_PARALLEL_JOBS` to fit available GPUs and memory.

To resume an interrupted batch, reuse the run token printed in the batch log
directory name:

```bash
RUN_TOKEN=YYYYMMDD_HHMMSS MAX_PARALLEL_JOBS=5 bash scripts/exploration/run_rd_curve_smoke.sh
```

Successful entries already recorded in that token's `status.tsv` are skipped.

## Validation

Most families record fixed-sample PSNR at epoch-equivalent intervals through
progress 50. MINER instead records one native point after each completed
coarse-to-fine scale; its final point is the cropped full-resolution PSNR.
After training, the runner writes:

- `batch_logs/exploration_v3/<RUN_TOKEN>/exploration_summary.tsv`
- `batch_logs/exploration_v3/<RUN_TOKEN>/needs_attention.txt`
- `batch_logs/exploration_v3/<RUN_TOKEN>/failed.txt`

Validation is strict by default. General families apply completeness,
non-finite, gain, and collapse checks. The independent MINER runner requires
all effective scales and finite PSNR values but does not require cross-scale
monotonicity because the intermediate targets have different resolutions.
General-family thresholds can be changed without regenerating configs:

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
| MINER | 50 epochs per scale, timestep 0 |
