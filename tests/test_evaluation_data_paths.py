from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.evaluation import run_batch as batch_runner
from var_expert_inr.evaluation.data_paths import (
    normalize_experiment_data_paths,
    normalize_raw_data_paths,
    resolve_evaluation_data_path,
)


def test_autodl_missing_file_reports_current_environment_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autodl_root = tmp_path / "autodl-tmp"
    monkeypatch.setenv("SERVER_ENV", "autodl")
    monkeypatch.setenv("AUTODL_DATA_ROOT", str(autodl_root))

    resolved = resolve_evaluation_data_path(
        "/mnt/legacy/data/Mesh/redsea/source_XYZT.npy",
        dataset_name="redsea",
        repo_root=tmp_path / "project",
    )

    assert resolved == autodl_root / "RedSea" / "source_XYZT.npy"


def test_raw_config_normalizes_every_supported_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autodl_root = tmp_path / "autodl-tmp"
    monkeypatch.setenv("SERVER_ENV", "autodl")
    monkeypatch.setenv("AUTODL_DATA_ROOT", str(autodl_root))
    raw = {
        "DATA": {
            "dataset_name": "bathymetry",
            "target_path": "/old/target.npy",
            "coords_path": "/old/coords.npy",
            "source_path": "/old/source.npy",
            "target_stats_path": "/old/target_stats.npz",
            "coordinate_stats_path": "/old/coordinate_stats.npz",
            "targets": {
                "SALT": "/old/target_SALT.npy",
                "TEMP": "/old/target_TEMP.npy",
            },
        }
    }

    normalized = normalize_raw_data_paths(
        raw,
        repo_root=tmp_path / "project",
        config_path=tmp_path / "archived" / "config.yaml",
    )

    expected_root = autodl_root / "RedSea"
    data = normalized["DATA"]
    assert data["target_path"] == str(expected_root / "target.npy")
    assert data["coords_path"] == str(expected_root / "coords.npy")
    assert data["source_path"] == str(expected_root / "source.npy")
    assert data["target_stats_path"] == str(expected_root / "target_stats.npz")
    assert data["coordinate_stats_path"] == str(expected_root / "coordinate_stats.npz")
    assert data["targets"] == {
        "SALT": str(expected_root / "target_SALT.npy"),
        "TEMP": str(expected_root / "target_TEMP.npy"),
    }
    assert raw["DATA"]["source_path"] == "/old/source.npy"


@dataclass(frozen=True)
class _DataConfig:
    dataset_name: str
    target_path: str | None
    targets: dict[str, str] | None
    coords_path: str | None
    coordinate_stats_path: str | None


@dataclass(frozen=True)
class _ExperimentConfig:
    data: _DataConfig


def test_schema_config_normalizes_coordinate_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_root = tmp_path / "custom-redsea"
    monkeypatch.setenv("SERVER_ENV", "autodl")
    monkeypatch.setenv("AUTODL_DATA_ROOT", str(tmp_path / "autodl-tmp"))
    monkeypatch.setenv("REDSEA_ROOT", str(custom_root))
    config = _ExperimentConfig(
        data=_DataConfig(
            dataset_name="redsea",
            target_path=None,
            targets={"SALT": "/old/target_SALT.npy"},
            coords_path="/old/source_XYZT.npy",
            coordinate_stats_path="/old/coordinate_stats.npz",
        )
    )

    normalized = normalize_experiment_data_paths(
        config,
        repo_root=tmp_path / "project",
    )

    assert normalized.data.coords_path == str(custom_root / "source_XYZT.npy")
    assert normalized.data.coordinate_stats_path == str(
        custom_root / "coordinate_stats.npz"
    )
    assert normalized.data.targets == {
        "SALT": str(custom_root / "target_SALT.npy")
    }


def test_worker_restores_explicit_server_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    observed: dict[str, str] = {}

    def fake_evaluate_run(*args, **kwargs):
        observed["server_env"] = os.environ["SERVER_ENV"]
        return {
            "output_dir": tmp_path / "output",
            "metrics_path": tmp_path / "metrics.json",
            "log_path": tmp_path / "evaluate.log",
        }

    monkeypatch.setenv("SERVER_ENV", "original")
    monkeypatch.setattr(batch_runner, "evaluate_run", fake_evaluate_run)
    request_path.write_text(
        json.dumps(
            {
                "run_dir": str(tmp_path / "run"),
                "server_env": "autodl",
                "metrics": ["training_time"],
                "timesteps": "all",
                "target": None,
                "source": "checkpoint",
                "render": False,
                "overwrite": False,
            }
        ),
        encoding="utf-8",
    )

    return_code = batch_runner._worker_main(request_path, result_path)

    assert return_code == 0
    assert observed["server_env"] == "autodl"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "success"

