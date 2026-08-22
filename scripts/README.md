# Experiment automation

Script layout mirrors `configs/`:

| Directory | Contents |
| --- | --- |
| `main/` | Formal config generator, complete runner, and selected main runs. |
| `rd_curve/` | RD-curve selection and runner. |
| `exploration/` | RD smoke tests and optimizer probes. |
| `ablation/` | Architecture/depth/regularization generators and reports. |
| `sensitivity/` | Learning-rate and independent VarExpert expert-count/Top-K sweeps. |
| `tools/` | Dataset inspection and result-maintenance utilities. |
| `lib/` | Shared shell primitives; not a user-facing entrypoint. |

Run commands from the repository root. For example:

```bash
python scripts/main/generate_configs.py
bash scripts/main/run_all.sh
bash scripts/rd_curve/run.sh
python scripts/sensitivity/generate_var_expert_num.py
python scripts/sensitivity/generate_var_expert_topk.py
```

The VarExpert sensitivity generators intentionally write to separate config
and run roots. The expert-count study uses `shared_enc_inr` as the one-expert
control (without a gating network), then covers 2--8 VarExpert experts with
Top-K capped to two for the two-expert case. The Top-K study fixes seven
experts and covers Top-K 1--7.

Python script directories are packages so tests and other automation can import
their reusable functions without depending on filename-based loaders.
