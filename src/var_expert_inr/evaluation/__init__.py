from .metrics import QualityAccumulator, evaluate_predictions, save_metrics
from .selection import (
    metrics_require_ground_truth,
    metrics_require_rendering,
    parse_metric_selection,
    parse_name_selection,
    parse_timestep_selection,
)
from .adapters import DecodeSession, RunAdapter, SUPPORTED_ADAPTERS


def evaluate_run(*args, **kwargs):
    # Import lazily so the training engine can import evaluation.metrics without
    # service.py immediately importing the partially initialized engine again.
    from .service import evaluate_run as _evaluate_run

    return _evaluate_run(*args, **kwargs)

__all__ = [
    "QualityAccumulator",
    "evaluate_predictions",
    "evaluate_run",
    "DecodeSession",
    "RunAdapter",
    "SUPPORTED_ADAPTERS",
    "metrics_require_ground_truth",
    "metrics_require_rendering",
    "parse_metric_selection",
    "parse_name_selection",
    "parse_timestep_selection",
    "save_metrics",
]
