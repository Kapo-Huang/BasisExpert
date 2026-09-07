from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from numpy.lib.format import open_memmap

from var_expert_inr.methods.ecnr.cache import CacheWorkspace
from var_expert_inr.methods.ecnr.cnn import BoundaryCNN, train_boundary_cnn
from var_expert_inr.methods.ecnr.pyramid import PyramidScale
from var_expert_inr.methods.ecnr import runner as ecnr_runner
from var_expert_inr.methods.ecnr.checkpoint_codec import load_inference_checkpoint
from var_expert_inr.methods.ecnr.runner import (
    _budgeted_batches,
    _clip_in_place,
    _framewise_add_in_place,
    _pyramid_scalar_budgets,
)


def test_budgeted_batches_are_deterministic_and_bounded() -> None:
    def sample(seed: int) -> np.ndarray:
        batches = list(
            _budgeted_batches(
                7,
                logical_samples=19,
                batch_size=4,
                rng=np.random.default_rng(seed),
            )
        )
        assert all(batch.size <= 4 for batch in batches)
        return np.concatenate(batches)

    first = sample(12)
    second = sample(12)
    assert first.tolist() == second.tolist()
    assert first.size == 19
    assert int(first.min()) >= 0
    assert int(first.max()) < 7


def test_pyramid_budget_is_exact_and_remainder_goes_to_scale_zero() -> None:
    pyramid = [
        PyramidScale(0, np.empty((4, 4, 4, 4), dtype=np.float32), np.arange(4)),
        PyramidScale(1, np.empty((2, 2, 2, 2), dtype=np.float32), np.arange(2)),
        PyramidScale(2, np.empty((1, 1, 1, 1), dtype=np.float32), np.arange(1)),
    ]
    budgets = _pyramid_scalar_budgets(pyramid, 101)
    assert sum(budgets.values()) == 101
    assert budgets[0] > budgets[1] > budgets[2]


def test_budgeted_cnn_respects_voxel_cap_and_read_only_targets() -> None:
    inputs = np.zeros((3, 3, 4, 5), dtype=np.float32)
    targets = np.ones_like(inputs)
    targets.setflags(write=False)
    model = BoundaryCNN(hidden_channels=2)
    with warnings.catch_warnings(record=True) as caught:
        result = train_boundary_cnn(
            model,
            inputs,
            targets,
            epochs=3,
            lr=1.0e-5,
            core_shape_zyx=(2, 2, 3),
            halo=1,
            device=torch.device("cpu"),
            seed=7,
            sampling_mode="budgeted_tiles",
            core_voxel_budget=100,
            log_every=1,
            progress_log_seconds=0,
        )
    assert result["sampling_mode"] == "budgeted_tiles"
    assert int(result["core_voxel_visits"]) <= 100
    assert int(result["optimizer_steps"]) == int(result["sampled_tiles"])
    assert int(result["halo_voxel_visits"]) >= int(result["core_voxel_visits"])
    assert not any("not writable" in str(item.message) for item in caught)


def test_full_volume_cnn_keeps_one_optimizer_step_per_frame() -> None:
    inputs = np.zeros((2, 2, 2, 2), dtype=np.float32)
    targets = np.ones_like(inputs)
    model = BoundaryCNN(hidden_channels=2)
    result = train_boundary_cnn(
        model,
        inputs,
        targets,
        epochs=2,
        lr=1.0e-5,
        core_shape_zyx=(2, 2, 2),
        halo=1,
        device=torch.device("cpu"),
        seed=3,
        sampling_mode="full_volume",
        log_every=0,
        progress_log_seconds=0,
    )
    assert int(result["optimizer_steps"]) == 4
    assert int(result["core_voxel_visits"]) == int(inputs.size * 2)


def test_cache_workspace_releases_files_and_removes_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    workspace = CacheWorkspace(cache_root)
    artifact = cache_root / "artifact.npy"
    values = open_memmap(artifact, mode="w+", dtype=np.float32, shape=(8,))
    values[:] = np.arange(8, dtype=np.float32)
    values.flush()
    workspace.register(artifact, values)
    workspace.release(artifact, arrays=(values,), label="test")
    assert not artifact.exists()
    assert workspace.metrics()["released_bytes"] > 0
    workspace.cleanup()
    assert not cache_root.exists()
    assert workspace.metrics()["final_bytes"] == 0


def test_in_place_composite_and_clip_match_reference(tmp_path: Path) -> None:
    path = tmp_path / "values.npy"
    values = open_memmap(path, mode="w+", dtype=np.float32, shape=(2, 2, 2, 2))
    values[:] = np.linspace(-2.0, 2.0, values.size, dtype=np.float32).reshape(values.shape)
    residual = np.full(values.shape, 0.25, dtype=np.float32)
    expected = np.clip(np.asarray(values).copy() + residual, -1.0, 1.0)
    _framewise_add_in_place(values, residual)
    _clip_in_place(values)
    np.testing.assert_allclose(values, expected)


def test_tiny_train_and_predict_leave_no_cache(tmp_path: Path, monkeypatch) -> None:
    target_path = tmp_path / "target.npy"
    target = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    np.save(target_path, target)
    config_path = tmp_path / "config.yaml"
    config = {
        "experiment": "tiny_ecnr",
        "exp_id": "tiny-ecnr",
        "experiment_root": str(tmp_path / "runs"),
        "data": {
            "kind": "volume",
            "dataset_name": "tiny",
            "split": "train",
            "target": "value",
            "target_path": str(target_path),
            "volume_shape": {"T": 2, "Z": 2, "Y": 2, "X": 2},
        },
        "model": {"block_shape_xyz": [2, 2, 2]},
        "training": {
            "epochs_per_scale": 0,
            "quantization_finetune_epochs": 0,
            "device": "cpu",
            "log_every": 0,
            "progress_log_seconds": 0,
        },
        "cnn": {"epochs": 0},
        "evaluation": {"batch_size": 16, "save_predictions": True},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def cheap_cnn_quantization(model, *, bits: int, seed: int):
        del seed
        parameters = {}
        for name, parameter in model.named_parameters():
            values = parameter.detach().cpu().numpy()
            parameters[name] = {
                "labels": np.zeros(values.shape, dtype=np.uint16),
                "codebook": np.array([float(values.mean())], dtype=np.float32),
            }
        return {"bits": int(bits), "parameters": parameters}

    monkeypatch.setattr(ecnr_runner, "_quantize_cnn", cheap_cnn_quantization)
    summary = ecnr_runner.run_train(config_path)
    checkpoint = Path(summary["checkpoint_path"])
    run_dir = checkpoint.parent.parent
    assert checkpoint.is_file()
    assert load_inference_checkpoint(checkpoint)["format"] == "ecnr_inference_v1"
    assert not (run_dir / "cache").exists()
    cost = yaml.safe_load((run_dir / "metrics" / "training_cost.json").read_text(encoding="utf-8"))
    assert cost["cache"]["final_bytes"] == 0

    prediction = ecnr_runner.run_predict(config_path, checkpoint=checkpoint)
    assert Path(prediction["prediction_path"]).is_file()
    assert not (run_dir / "cache").exists()


def test_train_exception_preserves_error_and_removes_cache(tmp_path: Path) -> None:
    config_path = tmp_path / "broken-config.yaml"
    runs_root = tmp_path / "runs"
    config = {
        "experiment": "broken_ecnr",
        "exp_id": "broken-ecnr",
        "experiment_root": str(runs_root),
        "data": {
            "kind": "volume",
            "dataset_name": "tiny",
            "split": "train",
            "target": "value",
            "target_path": str(tmp_path / "missing.npy"),
            "volume_shape": {"T": 2, "Z": 2, "Y": 2, "X": 2},
        },
        "training": {"device": "cpu"},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ecnr_runner.run_train(config_path)

    run_dirs = list((runs_root / "broken-ecnr").iterdir())
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "cache").exists()


def test_predict_load_exception_removes_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "broken-predict" / "existing-run"
    checkpoint = run_dir / "checkpoints" / "missing.pth"
    config_path = tmp_path / "predict-config.yaml"
    config = {
        "experiment": "broken_predict",
        "exp_id": "broken-predict",
        "experiment_root": str(tmp_path / "runs"),
        "data": {
            "kind": "volume",
            "dataset_name": "tiny",
            "split": "train",
            "target": "value",
            "target_path": str(tmp_path / "unused.npy"),
            "volume_shape": {"T": 2, "Z": 2, "Y": 2, "X": 2},
        },
        "training": {"device": "cpu"},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ecnr_runner.run_predict(config_path, checkpoint=checkpoint)

    assert not (run_dir / "cache").exists()
