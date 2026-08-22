# Experiment configurations

Configurations are grouped by the question an experiment answers, then by the
published method name.

| Directory | Purpose |
| --- | --- |
| `main/` | Primary comparison experiments at the default model budget. |
| `rd_curve/` | Formal rate-distortion points (`Size082`, `Size163`, `Size326`, and `Size652`) for the six selected families. |
| `exploration/` | Short probes used to find viable schedules or configurations. |
| `ablation/` | Controlled architecture, depth, and regularization removals. |
| `sensitivity/` | Parameter sweeps such as learning rate, expert count, and Top-K. |

Study directories use semantic names instead of chronology-based `vN` names.
Generated files should be rebuilt with the matching entrypoint under
`scripts/<category>/`; do not introduce a new top-level `configs_*` directory.
