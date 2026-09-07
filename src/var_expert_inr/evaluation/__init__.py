from .metrics import QualityAccumulator, evaluate_predictions, save_metrics
from .selection import (
    metrics_require_ground_truth,
    metrics_require_rendering,
    parse_metric_selection,
    parse_name_selection,
    parse_timestep_selection,
)
from .adapters import DecodeSession, RunAdapter, SUPPORTED_ADAPTERS
from .artifacts import ArtifactStore, resolve_artifact_reference


def evaluate_run(*args, **kwargs):
    # Import lazily so the training engine can import evaluation.metrics without
    # service.py immediately importing the partially initialized engine again.
    from .service import evaluate_run as _evaluate_run

    return _evaluate_run(*args, **kwargs)


def evaluate_dependency_run(*args, **kwargs):
    from .dependency_service import evaluate_dependency_run as _evaluate_dependency_run

    return _evaluate_dependency_run(*args, **kwargs)

__all__ = [
    "QualityAccumulator",
    "evaluate_predictions",
    "evaluate_run",
    "evaluate_dependency_run",
    "DecodeSession",
    "RunAdapter",
    "SUPPORTED_ADAPTERS",
    "ArtifactStore",
    "resolve_artifact_reference",
    "metrics_require_ground_truth",
    "metrics_require_rendering",
    "parse_metric_selection",
    "parse_name_selection",
    "parse_timestep_selection",
    "save_metrics",
]
