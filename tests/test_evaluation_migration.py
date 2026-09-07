from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.evaluation.manage_eval_result import migrate, verify
from var_expert_inr.evaluation.rendering import load_render_profile, profile_fingerprint


def test_legacy_tree_migrates_paths_and_prunes_only_after_verification(tmp_path: Path) -> None:
    result_root = tmp_path / "EvalResult"
    run_dir = tmp_path / "Result" / "Main" / "Model" / "Combustion" / "Temperature"
    config_dir = run_dir / "configs"
    prediction_dir = run_dir / "predictions"
    config_dir.mkdir(parents=True)
    prediction_dir.mkdir(parents=True)
    gt_path = tmp_path / "gt.npy"
    prediction_path = prediction_dir / "prediction.npy"
    np.save(gt_path, np.zeros((1, 1, 2, 2), dtype=np.float32))
    np.save(prediction_path, np.ones((1, 1, 2, 2), dtype=np.float32))
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": "migration",
                "exp_id": "migration",
                "experiment_root": str(tmp_path / "runs"),
                "data": {
                    "kind": "volume",
                    "dataset_name": "combustion_40nh3_1",
                    "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 1},
                    "targets": {"Temperature": str(gt_path)},
                },
                "model": {"name": "siren", "in_features": 4, "hidden_features": 4},
                "training": {"epochs": 1, "batch_size": 2},
                "evaluation": {"batch_size": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    legacy_dir = result_root / "EvalResultMeric" / "Main" / "Model" / "Combustion" / "Temperature"
    render_dir = legacy_dir / "renders" / "Temperature"
    render_dir.mkdir(parents=True)
    gt_render = render_dir / "gt_t0000.png"
    pred_render = render_dir / "pred_t0000.png"
    gt_render.write_bytes(b"gt-png")
    pred_render.write_bytes(b"pred-png")
    profile = load_render_profile("combustion_40nh3_1", None, repo_root=Path(__file__).resolve().parents[1])
    manifest = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "model": "siren",
        "dataset_name": "combustion_40nh3_1",
        "source_path": str(prediction_path),
        "render_profile": {"fingerprint": profile_fingerprint(profile)},
        "error_analysis": {"enabled": False},
    }
    metrics = {
        "schema_version": 1,
        "status": "complete",
        "per_timestep": [
            {
                "target": "Temperature",
                "timestep": 0,
                "gt_render_path": "X:/old/gt_t0000.png",
                "pred_render_path": "X:/old/pred_t0000.png",
            }
        ],
    }
    (legacy_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (legacy_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (legacy_dir / "metrics.csv").write_text("target,timestep\nTemperature,0\n", encoding="utf-8")

    report = migrate(result_root, apply=True, prune=True)

    assert report["errors"] == []
    assert report["pruned"] is True
    assert not (result_root / "EvalResultMeric").exists()
    destination = result_root / "evaluations" / "legacy_metrics" / "Main" / "Model" / "Combustion" / "Temperature"
    migrated_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert migrated_manifest["result_root"] == "."
    migrated = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    row = migrated["per_timestep"][0]
    assert row["gt_render_path"].startswith("artifacts/ground_truth/")
    assert row["pred_render_path"].startswith("artifacts/prediction/")
    assert (result_root / row["gt_render_path"]).read_bytes() == b"gt-png"
    assert (result_root / row["pred_render_path"]).read_bytes() == b"pred-png"
    assert verify(result_root)["errors"] == []
