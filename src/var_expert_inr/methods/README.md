# Self-contained methods

Each package in this directory owns a method-specific lifecycle (some or all of
config parsing, data access, training, inference checkpoints, and CLI). These
methods therefore do not belong in the shared `models/` registry even though
their packages contain model classes.

Use `python -m var_expert_inr.methods.<method>.cli ...` for their standalone
entrypoints. ECNR and MINER are additionally dispatched by the unified CLI for
backward-compatible experiment commands.
