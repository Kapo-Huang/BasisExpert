import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_evaluate, run_predict, run_train


class TrainingPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _read_csv_rows(self, path: Path):
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_node_multitarget_train_predict_evaluate(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.5, 0.5, 0.0, 0.5],
                [0.2, 0.8, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        target_a = coords[:, :1] * 0.5
        target_b = np.concatenate([coords[:, 1:2], coords[:, 3:4]], axis=1)
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, target_a)
        np.save(b_path, target_b)

        config = {
            "experiment": "node-pipeline",
            "exp_id": "node-pipeline",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "light_basis_expert",
                "in_features": 4,
                "num_experts": 2,
                "base_dim": 2,
                "top_k": 1,
                "expert_num_layers": 2,
                "gate_num_layers": 2,
                "decoder_num_layers": 2,
                "head_num_layers": 2,
            },
            "training": {
                "epochs": 2,
                "batch_size": 3,
                "pred_batch_size": 3,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 1,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 3},
        }
        config_path = self._write_yaml(self.root / "node.yaml", config)
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        predict_result = run_predict(config_path)
        self.assertTrue(all(path.exists() for path in predict_result["prediction_paths"].values()))
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())

    def test_volume_single_target_train_predict_evaluate(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "volume-pipeline",
            "exp_id": "volume-pipeline",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target_path": str(volume_path),
            },
            "model": {
                "name": "siren",
                "in_features": 4,
                "hidden_features": 8,
                "hidden_layers": 1,
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "pred_batch_size": 4,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 2,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 4},
        }
        config_path = self._write_yaml(self.root / "volume.yaml", config)
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())

    def test_training_updates_global_model_size_catalog(self):
        experiment_root = self.root / "runs"
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "volume.npy"
        np.save(volume_path, volume)

        siren_config = {
            "experiment": "volume-pipeline",
            "exp_id": "volume-pipeline",
            "experiment_root": str(experiment_root),
            "data": {
                "kind": "volume",
                "target_path": str(volume_path),
            },
            "model": {
                "name": "siren",
                "in_features": 4,
                "hidden_features": 8,
                "hidden_layers": 1,
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "pred_batch_size": 4,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 2,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 4},
        }
        siren_config_path = self._write_yaml(self.root / "siren.yaml", siren_config)

        coords = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.5, 0.5, 0.0, 0.5],
                [0.2, 0.8, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        target_a = coords[:, :1] * 0.5
        target_b = np.concatenate([coords[:, 1:2], coords[:, 3:4]], axis=1)
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, target_a)
        np.save(b_path, target_b)
        light_config = {
            "experiment": "node-pipeline",
            "exp_id": "node-pipeline",
            "experiment_root": str(experiment_root),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "light_basis_expert",
                "in_features": 4,
                "num_experts": 2,
                "base_dim": 2,
                "top_k": 1,
                "expert_num_layers": 2,
                "gate_num_layers": 2,
                "decoder_num_layers": 2,
                "head_num_layers": 2,
            },
            "training": {
                "epochs": 2,
                "batch_size": 3,
                "pred_batch_size": 3,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 1,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 3},
        }
        light_config_path = self._write_yaml(self.root / "light.yaml", light_config)

        run_train(siren_config_path)
        catalog_path = experiment_root / "model_size_catalog.csv"
        self.assertTrue(catalog_path.exists())
        rows = self._read_csv_rows(catalog_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "siren")
        self.assertEqual(rows[0]["hidden_features"], "8")
        self.assertEqual(rows[0]["hidden_layers"], "1")

        run_train(siren_config_path)
        rows = self._read_csv_rows(catalog_path)
        self.assertEqual(len(rows), 1)

        run_train(light_config_path)
        rows = self._read_csv_rows(catalog_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["model_name"] for row in rows], ["light_basis_expert", "siren"])

        light_row = rows[0]
        self.assertEqual(light_row["base_dim"], "2")
        self.assertEqual(light_row["num_experts"], "2")
        self.assertTrue(light_row["model_config_hash"])
        self.assertGreater(int(light_row["param_count"]), 0)
        self.assertGreater(int(light_row["trainable_param_count"]), 0)
        self.assertGreater(int(light_row["fp16_size_bytes"]), 0)

        run_predict(siren_config_path)
        run_evaluate(siren_config_path)
        rows_after_non_train = self._read_csv_rows(catalog_path)
        self.assertEqual(rows_after_non_train, rows)


if __name__ == "__main__":
    unittest.main()
