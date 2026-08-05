# Exploration v2

This follow-up matrix preserves the first exploration under `configs_exploration/`
and writes the new configs and runs to isolated locations:

- configs: `configs_exploration_v2/`
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

The v2 matrix uses NeuralExpert at Size326, MC-INR and VarExpert at the
Size163 multi-variable budget, and DC-INR at the Size163 per-variable budget.

## Matrix

- MC-INR: the three original depth profiles, re-run after the target-layout fix.
- DC-INR: normalized block RMSE with DBSCAN eps 0.01, 0.05, and 0.10 for all
  five variables. `max_initial_neurons` remains only a non-binding safety cap.
- NeuralExpert: Size326 depth 1, 2, and 3 for all five variables, including
  one-to-one manager pretraining configs.
- VarExpert: experts8/top3 as a control; experts9 with top-1 through top-9;
  experts10 with top-1 through top-10. All use the same Size163 total budget.

Generate and run:

```bash
python scripts/generate_exploration_v2_configs.py
MAX_PARALLEL_JOBS=5 bash scripts/run_exploration_v2.sh
```

The runner executes NeuralExpert managers first, then fixed MC-INR/DC-INR,
then the expert-count control, the top-k sweep, and finally NeuralExpert main
training. Successful configs are skipped when the same batch log directory is
resumed.
