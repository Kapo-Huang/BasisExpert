"""Self-contained reference methods with dedicated training pipelines.

Models that run through the shared VarExpert-INR engine live in
``var_expert_inr.models``.  Packages below this namespace own their config,
data, checkpoint, and runner lifecycle.
"""

__all__ = [
    "apmgsrn",
    "ecnr",
    "fv_srn",
    "mc_inr",
    "miner",
    "neural_expert",
    "rmdsrn",
]
