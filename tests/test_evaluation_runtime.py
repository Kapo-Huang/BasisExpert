from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import run_batch as batch_runner
from var_expert_inr.evaluation.runtime import (
    estimate_inference_time,
    estimate_training_time,
    uniform_fraction_timesteps,
)
from var_expert_inr.evaluation.selection import parse_metric_selection
from var_expert_inr.evaluation.service import EvaluationRequest


def test_runtime_metrics_are_selectable() -> None:
    assert parse_metric_selection("training_time,inference_time") == (
        "training_time",
        "inference_time",
    )


def test_uniform_fraction_selects_ceil_ten_percent() -> None:
    selected = uniform_fraction_timesteps(101, 0.1)

    assert len(selected) == 11
    assert selected[0] == 0
    assert selected[-1] == 100
    assert len(set(selected)) == len(selected)


def test_training_estimate_uses_common_total_budget() -> None:
    result = estimate_training_time(
        measured_samples=72_000_000,
        measured_seconds=36.0,
        total_samples=14_400_000_000,
    )

    assert result["samples_per_second"] == pytest.approx(2_000_000.0)
    assert result["estimated_training_seconds"] == pytest.approx(7200.0)


def test_inference_estimate_loads_checkpoint_once() -> None:
    result = estimate_inference_time(
        load_seconds=2.0,
        reconstruction_seconds=10.0,
        selected_values=100,
        total_values=1000,
    )

    assert result["estimated_full_reconstruction_seconds"] == pytest.approx(100.0)
    assert result["estimated_total_inference_seconds"] == pytest.approx(102.0)


def test_request_validates_runtime_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inference_fraction"):
        EvaluationRequest(run_dir=tmp_path, inference_fraction=0.0)


def test_runtime_cache_is_experiment_scoped(tmp_path: Path) -> None:
    run_dir = tmp_path / "Result" / "Main" / "Model" / "Dataset" / "Run"
    output_dir = tmp_path / "EvalResult" / "runtime"
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({
            "status": "complete",
            "performance": {"training_time": {"estimated_training_seconds": 1.0}},
        }),
        encoding="utf-8",
    )
    (output_dir / "metrics.csv").write_text("row_type\n", encoding="utf-8")

    state = batch_runner._existing_evaluation_state_at(
        run_dir,
        raw={},
        target="all",
        timesteps="all",
        requested_metrics=("training_time",),
        render=False,
        current_profile_fingerprint=None,
        result_root=tmp_path / "EvalResult",
        output_dir=output_dir,
        error_vmin=0.0,
        error_vmax=5.0,
    )

    assert state is not None
    assert state["reuse_reason"] == "experiment-runtime-cache"


def test_runtime_batch_summary_sums_model_dataset_groups(tmp_path: Path) -> None:
    records = []
    for index, seconds in enumerate((10.0, 15.0)):
        metrics_path = tmp_path / f"metrics_{index}.json"
        metrics_path.write_text(
            json.dumps({
                "performance": {
                    "training_time": {
                        "estimated_training_seconds": seconds,
                        "estimated_training_hours": seconds / 3600.0,
                        "samples_per_second": 1.0,
                    },
                    "inference_time": {
                        "estimated_total_inference_seconds": seconds / 2.0,
                        "estimated_total_inference_hours": seconds / 7200.0,
                        "values_per_second": 2.0,
                    },
                }
            }),
            encoding="utf-8",
        )
        records.append({
            "status": "success",
            "model": "Model",
            "dataset": "Dataset",
            "target": str(index),
            "run_dir": str(tmp_path / str(index)),
            "metrics_path": str(metrics_path),
        })

    batch_runner._write_summary(tmp_path, records, tmp_path / "config.yaml")
    summary = json.loads((tmp_path / "runtime_summary.json").read_text(encoding="utf-8"))

    assert summary["experiment_count"] == 2
    assert summary["groups"][0]["estimated_training_seconds"] == pytest.approx(25.0)
    assert summary["groups"][0]["estimated_total_inference_seconds"] == pytest.approx(12.5)


def test_batch_discovery_supports_both_config_layouts(tmp_path: Path) -> None:
    nested = tmp_path / "Result" / "Nested"
    root_config = tmp_path / "Result" / "RootConfig"
    (nested / "configs").mkdir(parents=True)
    (nested / "configs" / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (root_config).mkdir(parents=True)
    (root_config / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    for run_dir in (nested, root_config):
        (run_dir / "checkpoints").mkdir()
        (run_dir / "checkpoints" / "model.pth").write_bytes(b"checkpoint")

    discovered = batch_runner._discover_checkpoint_runs(tmp_path / "Result")

    assert discovered == sorted([nested.resolve(), root_config.resolve()])
