import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.methods.apmgsrn.cli import run_train


class APMGSRNTrainingTestCase(unittest.TestCase):
    TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _base_config(self, target_path: Path) -> dict:
        return {
            "experiment": "apmgsrn-smoke",
            "exp_id": "apmgsrn-smoke",
            "experiment_root": str(self.root / "runs"),
            "MODEL": {
                "model_name": "apmgsrn",
                "feature_grid_shape": [2, 2, 2],
                "n_features": 1,
                "n_grids": 2,
                "nodes_per_layer": 8,
                "n_layers": 1,
                "use_bias": False,
                "use_tcnn_if_available": False,
                "grid_initialization": "default",
            },
            "DATA": {
                "dataset_name": "ionization",
                "target": "GT",
                "targets": {
                    "GT": str(target_path),
                },
                "volume_shape": {"X": 2, "Y": 2, "Z": 2, "T": 3},
                "align_corners": True,
            },
            "TRAINING": {
                "iterations": 3,
                "points_per_iteration": 8,
                "prediction_points_per_batch": 4,
                "lr": 1.0e-2,
                "beta_1": 0.9,
                "beta_2": 0.99,
                "device": "cpu",
                "data_device": "cpu",
                "save_every": 0,
                "log_every": 1,
                "time_indices": "all",
                "seed": 3,
            },
        }

    def _assert_timestamp_run_dir(self, run_dir: Path, *, exp_id: str) -> None:
        self.assertEqual(run_dir.parent, self.root / "runs" / exp_id)
        self.assertRegex(run_dir.name, self.TIMESTAMP_PATTERN)

    def _assert_training_outputs_and_timestamp_behavior(self, target_array: np.ndarray, *, stem: str) -> None:
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        target_path = self.root / f"{stem}.npy"
        np.save(target_path, target_array)

        config_path = self._write_yaml(self.root / f"{stem}.yaml", self._base_config(target_path))
        result = run_train(config_path)

        run_dir = Path(result["run_dir"])
        manifest_path = Path(result["manifest_path"])
        prediction_path = Path(result["prediction_path"])
        metrics_path = Path(result["metrics_path"])
        self._assert_timestamp_run_dir(run_dir, exp_id="apmgsrn-smoke")
        self.assertEqual(manifest_path, run_dir / "manifest.json")
        self.assertEqual(prediction_path, run_dir / "predictions" / "apmgsrn-smoke.npy")
        self.assertEqual(metrics_path, run_dir / "metrics" / "aggregate.json")
        self.assertTrue((run_dir / "configs" / "config.yaml").exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(prediction_path.exists())
        self.assertTrue(metrics_path.exists())

        prediction = np.load(prediction_path)
        self.assertEqual(prediction.shape, volume.shape)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(sorted(manifest["timesteps"].keys()), ["t000", "t001", "t002"])
        for token, entry in manifest["timesteps"].items():
            timestep_dir = run_dir / "timesteps" / token
            self.assertTrue(Path(entry["checkpoint_path"]).exists(), token)
            self.assertTrue(Path(entry["prediction_path"]).exists(), token)
            self.assertTrue(Path(entry["metrics_path"]).exists(), token)
            self.assertEqual(Path(entry["checkpoint_path"]).parent, timestep_dir, token)
            self.assertEqual(Path(entry["prediction_path"]).parent, timestep_dir, token)
            self.assertEqual(Path(entry["metrics_path"]).parent, timestep_dir, token)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("GT", metrics["targets"])
        self.assertEqual(len(metrics["targets"]["GT"]["per_time"]), 3)

        rerun = run_train(config_path)
        rerun_dir = Path(rerun["run_dir"])
        self._assert_timestamp_run_dir(rerun_dir, exp_id="apmgsrn-smoke")
        self.assertNotEqual(rerun_dir, run_dir)
        run_dirs = sorted(path for path in (self.root / "runs" / "apmgsrn-smoke").iterdir() if path.is_dir())
        self.assertEqual(run_dirs, [run_dir, rerun_dir])
        self.assertEqual(rerun["completed_timesteps"], [0, 1, 2])
        self.assertEqual(rerun["skipped_timesteps"], [])

    def test_train_outputs_manifest_predictions_metrics_and_skip_behavior(self):
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        self._assert_training_outputs_and_timestamp_behavior(volume, stem="target_dense")

    def test_train_outputs_manifest_predictions_metrics_and_skip_behavior_for_flat_target(self):
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        self._assert_training_outputs_and_timestamp_behavior(volume.reshape(-1, 1), stem="target_flat")

    def test_same_exp_id_with_changed_config_creates_second_timestamped_run(self):
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        target_path = self.root / "target.npy"
        np.save(target_path, volume)

        config = self._base_config(target_path)
        config_path = self._write_yaml(self.root / "config.yaml", config)
        first = run_train(config_path)

        changed = self._base_config(target_path)
        changed["MODEL"]["n_grids"] = 3
        changed_path = self._write_yaml(self.root / "config_changed.yaml", changed)
        second = run_train(changed_path)

        first_run_dir = Path(first["run_dir"])
        second_run_dir = Path(second["run_dir"])
        self._assert_timestamp_run_dir(first_run_dir, exp_id="apmgsrn-smoke")
        self._assert_timestamp_run_dir(second_run_dir, exp_id="apmgsrn-smoke")
        self.assertNotEqual(first_run_dir, second_run_dir)
        self.assertEqual(second["completed_timesteps"], [0, 1, 2])
        self.assertEqual(second["skipped_timesteps"], [])

    def test_run_after_training_false_keeps_checkpoints_only(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 2, 2, 2)
        target_path = self.root / "target_checkpoint_only.npy"
        np.save(target_path, volume)

        config = self._base_config(target_path)
        config["exp_id"] = "apmgsrn-checkpoint-only"
        config["DATA"]["volume_shape"]["T"] = 1
        config["TRAINING"]["iterations"] = 1
        config["TRAINING"]["time_indices"] = [0]
        config["EVALUATION"] = {"run_after_training": False}
        config_path = self._write_yaml(self.root / "checkpoint_only.yaml", config)

        result = run_train(config_path)
        run_dir = Path(result["run_dir"])
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        self.assertNotIn("prediction_path", result)
        self.assertNotIn("metrics_path", result)
        self.assertTrue(Path(manifest["timesteps"]["t000"]["checkpoint_path"]).exists())
        self.assertIsNone(manifest["timesteps"]["t000"]["prediction_path"])
        self.assertIsNone(manifest["timesteps"]["t000"]["metrics_path"])
        self.assertEqual(list((run_dir / "predictions").glob("*.npy")), [])
        self.assertEqual(list((run_dir / "metrics").glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
