# Exploration v2

This follow-up matrix preserves the first exploration under `configs/ablation/architecture/`
and writes the new configs and runs to isolated locations:

- configs: `configs/sensitivity/routing_and_depth/`
- runs: `runs/exploration_v2/`
- batch logs: `batch_logs/exploration_v2/<run-token>/`

## Budget convention

Size labels denote the total FP16 parameter budget for all five Ionization
variables. A single-variable method receives one fifth of that number, while a
multi-variable method receives the full number.

| Label | Multi-variable total | Per single-variable model |
| --- | ---: | ---: |
| Size163 | 1.630 MiB | 0.326 MiB |
| Size326 | 3.260 MiB | 0.652 MiB |

The v2 matrix uses NeuralExpert at Size326 and MC-INR and VarExpert at the
Size163 multi-variable budget.

## Matrix

- MC-INR: the three original depth profiles, re-run after the target-layout fix.
- NeuralExpert: Size326 depth 1, 2, and 3 for all five variables, including
  one-to-one manager pretraining configs.
- VarExpert: experts8/top3 as a control; experts9 with top-1 through top-9;
  experts10 with top-1 through top-10. All use the same Size163 total budget.

Generate and run:

```bash
python scripts/sensitivity/generate_routing_and_depth.py
MAX_PARALLEL_JOBS=5 bash scripts/sensitivity/run_routing_and_depth.sh
```

The runner executes NeuralExpert managers first, then fixed MC-INR,
then the expert-count control, the top-k sweep, and finally NeuralExpert main
training. Successful configs are skipped when the same batch log directory is
resumed.
