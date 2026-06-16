import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_evaluate, run_predict, run_train
from var_expert_inr.training.engine import select_psnr_indices


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

    def _read_log_text(self, exp_id: str) -> str:
        logs_dir = self.root / "runs" / exp_id / "logs"
        log_path = next(logs_dir.glob("run_*.log"))
        return log_path.read_text(encoding="utf-8")

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

    def test_train_logs_psnr_and_timing(self):
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
            "experiment": "timing-node",
            "exp_id": "timing-node",
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
                "epochs": 1,
                "batch_size": 2,
                "pred_batch_size": 2,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 7,
                "val_split": 0.0,
                "log_every": 1,
                "log_psnr_every": 1,
                "psnr_sample_ratio": 0.5,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "log": {
                "effective_config": True,
                "model_stats": True,
                "epoch_summary": True,
                "startup_timing": True,
                "psnr": {
                    "enabled": True,
                    "per_target": True,
                },
                "timing": {
                    "enabled": True,
                    "epoch_breakdown": True,
                    "step_window": True,
                    "step_window_every_steps": 1,
                    "cuda_sync": False,
                },
            },
        }
        config_path = self._write_yaml(self.root / "timing_node.yaml", config)
        run_train(config_path)

        log_text = self._read_log_text("timing-node")
        self.assertIn("Config load:", log_text)
        self.assertIn("Run dir prepare:", log_text)
        self.assertIn("Dataset init:", log_text)
        self.assertIn("Model build:", log_text)
        self.assertIn("DataLoader build:", log_text)
        self.assertIn("Train total:", log_text)
        self.assertIn("Epoch 1/1 train=", log_text)
        self.assertIn("PSNR epoch 1/1: aggregate=", log_text)
        self.assertIn("a=", log_text)
        self.assertIn("b=", log_text)
        self.assertIn("Train timing window epoch 1/1 steps 1-1:", log_text)
        self.assertIn("Train epoch 1/1 timing(total):", log_text)

    def test_train_log_switches_can_disable_outputs(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "volume.npy"
        np.save(volume_path, volume)

        config = {
            "experiment": "quiet-volume",
            "exp_id": "quiet-volume",
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
                "epochs": 1,
                "batch_size": 4,
                "pred_batch_size": 4,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 3,
                "val_split": 0.0,
                "log_every": 1,
                "log_psnr_every": 1,
                "psnr_sample_ratio": 0.5,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "log": {
                "effective_config": True,
                "model_stats": True,
                "epoch_summary": False,
                "startup_timing": False,
                "psnr": {
                    "enabled": False,
                    "per_target": True,
                },
                "timing": {
                    "enabled": False,
                    "epoch_breakdown": True,
                    "step_window": True,
                    "step_window_every_steps": 1,
                    "cuda_sync": False,
                },
            },
        }
        config_path = self._write_yaml(self.root / "quiet_volume.yaml", config)
        run_train(config_path)

        log_text = self._read_log_text("quiet-volume")
        self.assertNotIn("Config load:", log_text)
        self.assertNotIn("DataLoader build:", log_text)
        self.assertNotIn("Train total:", log_text)
        self.assertNotIn("Epoch 1/1 train=", log_text)
        self.assertNotIn("PSNR epoch 1/1:", log_text)
        self.assertNotIn("timing window", log_text)
        self.assertNotIn("timing(total)", log_text)

    def test_select_psnr_indices_is_reproducible(self):
        first = select_psnr_indices(20, 0.25, 42)
        second = select_psnr_indices(20, 0.25, 42)
        third = select_psnr_indices(20, 0.25, 43)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))
        self.assertIsNone(select_psnr_indices(20, 1.0, 42))


if __name__ == "__main__":
    unittest.main()
