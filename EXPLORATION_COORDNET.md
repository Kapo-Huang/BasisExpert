# exploration_CoordNet: res10 low-learning-rate Size sweep

This experiment is an isolated follow-up to the CoordNet failures observed in
exploration v3/v4. It tests whether a lower learning rate stabilizes the formal
res10 architecture across every Size tier.

## Matrix

- Sizes: `Size082`, `Size163`, `Size326`, `Size652`, `Size1304`
- targets: `GT`, `H_plus`, `He`
- residual depth: `num_res=10`
- learning rates: `1e-5`, `5e-6`
- training length: 50 epochs
- total: 5 Sizes × 3 targets × 2 learning rates = 30 configs

The generator copies each matching formal CoordNet config, preserving its
Size-specific width, data definition, batch budget, and scheduler. It changes
only the isolated experiment identity/root, learning rate, 50-epoch training
and checkpoint cadence, and exploration probe settings. Every run uses seed 42
and a fixed 100,000-sample PSNR probe every five epochs. Both the best-probe and
final checkpoints are retained.

## Generate

```bash
python scripts/generate_exploration_coordnet_configs.py
```

Generated configs are written to:

```text
configs_exploration_CoordNet/CoordNet/<Size>/<lr-profile>/ionization__<target>.yaml
```

## Run

```bash
conda activate compression
MAX_PARALLEL_JOBS=5 bash scripts/run_exploration_coordnet.sh
```

The runner supports a dry run and resumable run tokens:

```bash
DRY_RUN=1 bash scripts/run_exploration_coordnet.sh
RUN_TOKEN=YYYYMMDD_HHMMSS bash scripts/run_exploration_coordnet.sh
```

Outputs are isolated under:

- runs: `runs/exploration_CoordNet/`
- batch logs: `batch_logs/exploration_CoordNet/<RUN_TOKEN>/`
- per-config summary: `exploration_summary.tsv`
- Size/LR summary and ranking: `profile_summary.tsv`
- validation report: `needs_attention.txt`

Strict validation flags failed/incomplete runs, non-finite PSNR, a final PSNR
drop greater than 1 dB from the peak, or less than 0.1 dB gain from the first
probe. Set `STRICT_VALIDATION=0` to write reports without making attention
items fail the batch. Thresholds can be overridden with
`COLLAPSE_THRESHOLD_DB` and `MINIMUM_GAIN_DB`.
