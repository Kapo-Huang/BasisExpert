from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from var_expert_inr.config.schema import EvaluationConfig
from var_expert_inr.evaluation.metrics import (
    ErrorAccumulator,
    QualityAccumulator,
    error_frame_statistics,
    error_percentage,
    pointwise_absolute_error,
    summarize_selected_quality,
    validate_error_bounds,
)
from var_expert_inr.evaluation.rendering import (
    _white_to_red_colormap,
    error_transfer_function,
    normalize_error_for_volume,
    render_error_image_frame,
)
from var_expert_inr.evaluation.selection import (
    metrics_require_ground_truth,
    parse_metric_selection,
)
from var_expert_inr.evaluation.service import evaluate_run


def test_scalar_error_statistics_use_fixed_normalized_gt_range() -> None:
    gt = np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32)
    pred = np.array([-1.0, -0.25, 0.5, 0.0], dtype=np.float32)
    absolute = pointwise_absolute_error(gt, pred)
    np.testing.assert_allclose(absolute, [0.0, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(error_percentage(absolute), [0.0, 12.5, 25.0, 50.0])

    stats = error_frame_statistics(absolute)
    assert stats["mean_absolute_error"] == pytest.approx(0.4375)
    assert stats["max_absolute_error"] == pytest.approx(1.0)
    assert stats["p95_absolute_error"] == pytest.approx(np.percentile(absolute, 95))
    assert stats["p99_absolute_error"] == pytest.approx(np.percentile(absolute, 99))
    assert stats["mean_error_percentage"] == pytest.approx(21.875)
    assert stats["max_error_percentage"] == pytest.approx(50.0)


def test_vector_error_is_pointwise_l2() -> None:
    gt = np.zeros((2, 3), dtype=np.float32)
    pred = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    absolute = pointwise_absolute_error(gt, pred)
    np.testing.assert_allclose(absolute, [5.0, 2.0])
    np.testing.assert_allclose(error_percentage(absolute), [250.0, 100.0])


def test_error_accumulator_pools_only_exact_mean_and_max() -> None:
    accumulator = ErrorAccumulator()
    accumulator.update(np.array([0.0, 0.5], dtype=np.float32))
    accumulator.update(np.array([1.0], dtype=np.float32))
    result = accumulator.as_dict()
    assert result == {
        "error_count": 3,
        "mean_absolute_error": 0.5,
        "max_absolute_error": 1.0,
        "mean_error_percentage": 25.0,
        "max_error_percentage": 50.0,
    }

    targets, aggregate = summarize_selected_quality(
        [],
        {"field": QualityAccumulator()},
        ("field",),
        ("error",),
        {"field": accumulator},
    )
    assert "p95_absolute_error" not in targets["field"]
    assert targets["field"]["mean_error_percentage"] == pytest.approx(25.0)
    assert aggregate["max_error_percentage"] == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-1.0, 5.0), (5.0, 5.0), (6.0, 5.0), (0.0, float("inf"))],
)
def test_invalid_error_bounds_are_rejected(lo: float, hi: float) -> None:
    with pytest.raises(ValueError):
        validate_error_bounds(lo, hi)
    with pytest.raises(ValueError):
        EvaluationConfig(error_vmin=lo, error_vmax=hi)


def test_error_metric_requires_ground_truth() -> None:
    assert parse_metric_selection("error") == ("error",)
    assert metrics_require_ground_truth(("error",))


def test_error_volume_mapping_clamps_to_fixed_scale() -> None:
    values = np.array([-1.0, 0.0, 2.5, 5.0, 8.0], dtype=np.float32)
    normalized = normalize_error_for_volume(
        values,
        error_vmin=0.0,
        error_vmax=5.0,
    )
    np.testing.assert_allclose(normalized, [-1.0, -1.0, 0.0, 1.0, 1.0])

    transfer = error_transfer_function()
    assert transfer["colorNodes"][0]["color"] == "#ffffff"
    assert transfer["colorNodes"][-1]["color"] == "#ff0000"
    assert transfer["opacityNodes"][0]["opacity"] == 0.0
    assert transfer["opacityNodes"][-1]["opacity"] == 1.0


def test_error_image_uses_white_red_and_fixed_clim(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "error.png"
    info = render_error_image_frame(
        np.array([[0.0, 2.5], [5.0, 10.0]], dtype=np.float32),
        output,
        profile={"kind": "volume", "renderer": "image2d", "window_size": [64, 64]},
        error_vmin=0.0,
        error_vmax=5.0,
    )
    assert output.is_file()
    assert info["clim"] == [0.0, 5.0]
    assert info["cmap"] == "white_to_red"
    cmap = _white_to_red_colormap()
    np.testing.assert_allclose(cmap(0.0)[:3], (1.0, 1.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(cmap(1.0)[:3], (1.0, 0.0, 0.0), atol=1e-7)


@pytest.mark.parametrize("model_name", ["siren", "mc_inr"])
def test_prediction_evaluation_writes_error_statistics_without_fields(
    tmp_path: Path,
    model_name: str,
) -> None:
    run_dir = tmp_path / "run"
    config_dir = run_dir / "configs"
    prediction_dir = run_dir / "predictions"
    config_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    ground_truth = np.array(
        [
            [[[-1.0, 0.0]]],
            [[[0.5, 1.0]]],
        ],
        dtype=np.float32,
    )
    prediction = np.array(
        [
            [[[-1.0, 0.2]]],
            [[[0.0, 0.0]]],
        ],
        dtype=np.float32,
    )
    gt_path = tmp_path / "target.npy"
    prediction_path = prediction_dir / "prediction.npy"
    np.save(gt_path, ground_truth)
    np.save(prediction_path, prediction)
    config = {
        "experiment": "error-integration",
        "exp_id": "error-integration",
        "experiment_root": str(tmp_path / "runs"),
        "data": {
            "kind": "volume",
            "dataset_name": "synthetic",
            "volume_shape": {"X": 2, "Y": 1, "Z": 1, "T": 2},
            "targets": {"field": str(gt_path)},
        },
        "model": {"name": model_name, "in_features": 4, "hidden_features": 4},
        "training": {"epochs": 1, "batch_size": 2},
        "evaluation": {"batch_size": 2},
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    result = evaluate_run(
        run_dir,
        metrics=("error",),
        timesteps="all",
        source="prediction",
        prediction=prediction_path,
        result_root=tmp_path / "evaluation",
        evaluation_id="error_test",
    )
    payload = result["metrics"]
    assert len(payload["per_timestep"]) == 2
    assert payload["per_timestep"][0]["mean_absolute_error"] == pytest.approx(0.1)
    assert payload["per_timestep"][1]["max_error_percentage"] == pytest.approx(50.0)
    assert payload["targets"]["field"]["mean_absolute_error"] == pytest.approx(0.425)
    assert payload["targets"]["field"]["max_error_percentage"] == pytest.approx(50.0)
    assert "p95_absolute_error" not in payload["targets"]["field"]
    assert not (Path(result["output_dir"]) / "errors").exists()
