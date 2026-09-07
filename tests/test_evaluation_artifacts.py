from __future__ import annotations

import shutil
import threading
import time
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from var_expert_inr.evaluation.artifacts import ArtifactStore
from var_expert_inr.evaluation.service import evaluate_run


PROFILE = {
    "kind": "volume",
    "renderer": "image2d",
    "cmap": "viridis",
    "clim": [-1.0, 1.0],
}


def test_identical_gt_content_has_path_independent_key(tmp_path: Path) -> None:
    first = tmp_path / "first.npy"
    second = tmp_path / "moved" / "second.npy"
    first.write_bytes(b"same-ground-truth")
    second.parent.mkdir()
    second.write_bytes(first.read_bytes())
    store = ArtifactStore(repo_root=tmp_path, result_root="EvalResult")

    first_spec = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=first, profile=PROFILE,
    )
    second_spec = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=second, profile=PROFILE,
    )

    assert first_spec.key == second_spec.key
    assert first_spec.path == second_spec.path


def test_shared_gt_is_materialized_once(tmp_path: Path) -> None:
    source = tmp_path / "gt.npy"
    source.write_bytes(b"ground-truth")
    store = ArtifactStore(repo_root=tmp_path, result_root="EvalResult")
    spec = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=source, profile=PROFILE,
    )
    calls = 0
    calls_lock = threading.Lock()

    def producer(path: Path):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        path.write_bytes(b"png")
        return {"renderer": "fake"}

    records = []
    threads = [threading.Thread(target=lambda: records.append(store.materialize(spec, producer))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(records) == 2
    assert sum(record.cache_hit for record in records) == 1
    assert spec.path.read_bytes() == b"png"
    assert not list(spec.path.parent.glob("*.lock"))


def test_profile_and_external_asset_changes_invalidate_key(tmp_path: Path) -> None:
    source = tmp_path / "gt.npy"
    mesh = tmp_path / "mesh.vtp"
    source.write_bytes(b"ground-truth")
    mesh.write_bytes(b"mesh-v1")
    store = ArtifactStore(repo_root=tmp_path, result_root="EvalResult")
    profile = {**PROFILE, "mesh_path": str(mesh)}
    first = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=source, profile=profile,
    )
    mesh.write_bytes(b"mesh-version-two")
    second = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=source, profile=profile,
    )
    third = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=source, profile={**profile, "cmap": "plasma"},
    )

    assert first.key != second.key
    assert second.key != third.key


def test_relative_artifact_reference_survives_result_root_move(tmp_path: Path) -> None:
    source = tmp_path / "gt.npy"
    source.write_bytes(b"ground-truth")
    store = ArtifactStore(repo_root=tmp_path, result_root="EvalResult")
    spec = store.ground_truth_spec(
        dataset="dataset", target="field", timestep=0,
        ground_truth_path=source, profile=PROFILE,
    )
    def producer(path: Path):
        path.write_bytes(b"png")
        return {}

    record = store.materialize(spec, producer)
    reference = store.relative(record.path)
    moved = tmp_path / "MovedEvalResult"
    shutil.move(store.root, moved)

    moved_store = ArtifactStore(repo_root=tmp_path, result_root=moved)
    assert moved_store.resolve(reference).read_bytes() == b"png"


def test_two_evaluations_reference_one_gt_without_local_copies(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    gt_path = tmp_path / "ground_truth.npy"
    np.save(gt_path, np.linspace(-1.0, 1.0, 4, dtype=np.float32).reshape(1, 1, 2, 2))
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "kind": "volume",
                "renderer": "image2d",
                "cmap": "viridis",
                "clim": [-1.0, 1.0],
                "window_size": [64, 64],
            }
        ),
        encoding="utf-8",
    )
    result_root = tmp_path / "EvalResult"
    gt_references = []
    for index in range(2):
        run_dir = tmp_path / "Result" / "Main" / f"Model{index}" / "Synthetic" / "Run"
        config_dir = run_dir / "configs"
        prediction_dir = run_dir / "predictions"
        config_dir.mkdir(parents=True)
        prediction_dir.mkdir(parents=True)
        prediction_path = prediction_dir / "prediction.npy"
        np.save(prediction_path, np.full((1, 1, 2, 2), index * 0.1, dtype=np.float32))
        config = {
            "experiment": f"artifact-{index}",
            "exp_id": f"artifact-{index}",
            "experiment_root": str(tmp_path / "runs"),
            "data": {
                "kind": "volume",
                "dataset_name": "synthetic",
                "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 1},
                "targets": {"field": str(gt_path)},
            },
            "model": {"name": "siren", "in_features": 4, "hidden_features": 4},
            "training": {"epochs": 1, "batch_size": 2},
            "evaluation": {"batch_size": 2},
        }
        (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result = evaluate_run(
            run_dir,
            metrics=("psnr",),
            timesteps="all",
            source="prediction",
            prediction=prediction_path,
            render=True,
            render_profile=profile_path,
            result_root=result_root,
            evaluation_id=f"evaluation_{index}",
            device="cpu",
        )
        row = result["metrics"]["per_timestep"][0]
        gt_references.append(row["gt_render_path"])
        assert row["gt_render_info"]["cache_hit"] is bool(index)
        assert not (Path(result["output_dir"]) / "renders").exists()
        persisted = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
        assert persisted["per_timestep"][0]["gt_render_path"].startswith("artifacts/")

    assert gt_references[0] == gt_references[1]
