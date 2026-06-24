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

    def _latest_run_dir(self, exp_id: str, experiment_root: Path | None = None) -> Path:
        runs_root = experiment_root or (self.root / "runs")
        exp_dir = runs_root / exp_id
        candidates = sorted(path for path in exp_dir.iterdir() if path.is_dir())
        if not candidates:
            raise AssertionError(f"No run directories found under {exp_dir}")
        return candidates[-1]

    def _read_log_text(self, exp_id: str, experiment_root: Path | None = None) -> str:
        logs_dir = self._latest_run_dir(exp_id, experiment_root=experiment_root) / "logs"
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
                "name": "var_expert",
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
        run_dir = Path(train_result["checkpoint_path"]).parent.parent
        self.assertTrue((run_dir / "configs" / "config.yaml").exists())
        predict_result = run_predict(config_path)
        self.assertTrue(
            all(path.exists() and path.parent == run_dir / "predictions" for path in predict_result["prediction_paths"].values())
        )
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())
        self.assertEqual(Path(eval_result["metrics_path"]).parent, run_dir / "metrics")

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
        run_dir = Path(train_result["checkpoint_path"]).parent.parent
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())
        self.assertEqual(Path(eval_result["metrics_path"]).parent, run_dir / "metrics")

    def test_predict_and_evaluate_require_existing_timestamp_run(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "missing_volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "missing-run",
            "exp_id": "missing-run",
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
                "seed": 2,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 4},
        }
        config_path = self._write_yaml(self.root / "missing_run.yaml", config)

        with self.assertRaisesRegex(FileNotFoundError, "No timestamped run directory found"):
            run_predict(config_path)
        with self.assertRaisesRegex(FileNotFoundError, "No timestamped run directory found"):
            run_evaluate(config_path)

    def test_predict_and_evaluate_use_latest_timestamp_run(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "latest_volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "latest-run",
            "exp_id": "latest-run",
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
                "seed": 2,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 4},
        }
        config_path = self._write_yaml(self.root / "latest_run.yaml", config)

        first_train = run_train(config_path)
        first_run_dir = Path(first_train["checkpoint_path"]).parent.parent
        second_train = run_train(config_path)
        second_run_dir = Path(second_train["checkpoint_path"]).parent.parent

        self.assertNotEqual(first_run_dir, second_run_dir)
        run_dirs = sorted(path for path in (self.root / "runs" / "latest-run").iterdir() if path.is_dir())
        self.assertEqual(run_dirs, [first_run_dir, second_run_dir])

        predict_result = run_predict(config_path)
        self.assertTrue(
            all(path.parent == second_run_dir / "predictions" for path in predict_result["prediction_paths"].values())
        )
        eval_result = run_evaluate(config_path)
        self.assertEqual(Path(eval_result["metrics_path"]).parent, second_run_dir / "metrics")
        self.assertEqual(run_dirs, sorted(path for path in (self.root / "runs" / "latest-run").iterdir() if path.is_dir()))

    def test_volume_pretrain_uses_batches_per_epoch_budget(self):
        volume = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(3, 1, 1, 2)
        volume_path = self.root / "budget_volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "budgeted-pretrain",
            "exp_id": "budgeted-pretrain",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target_path": str(volume_path),
                "volume_shape": {"X": 2, "Y": 1, "Z": 1, "T": 3},
            },
            "model": {
                "name": "var_expert",
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
                "seed": 11,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
                "batches_per_epoch_budget": 1,
                "pretrain": {
                    "enabled": True,
                    "epochs": 1,
                    "lr": 1.0e-3,
                },
            },
            "log": {
                "timing": {
                    "enabled": False,
                },
            },
        }
        config_path = self._write_yaml(self.root / "budgeted_pretrain.yaml", config)
        run_train(config_path)

        log_text = self._read_log_text("budgeted-pretrain")
        self.assertIn("Pretrain start: epochs=1 batch_size=2", log_text)
        self.assertIn("batches_per_epoch_budget=1", log_text)
        self.assertIn("Pretrain DataLoader ready: batches_per_epoch=1", log_text)
        self.assertIn("budget_batches=1 batch_size=2", log_text)
        self.assertIn("Pretrain epoch 1/1 start: batches=1 batch_size=2", log_text)

    def test_volume_pretrain_falls_back_to_full_loader_without_budget(self):
        volume = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(3, 1, 1, 2)
        volume_path = self.root / "full_volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "full-pretrain",
            "exp_id": "full-pretrain",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target_path": str(volume_path),
                "volume_shape": {"X": 2, "Y": 1, "Z": 1, "T": 3},
            },
            "model": {
                "name": "var_expert",
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
                "seed": 12,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
                "batches_per_epoch_budget": 0,
                "pretrain": {
                    "enabled": True,
                    "epochs": 1,
                    "lr": 1.0e-3,
                },
            },
            "log": {
                "timing": {
                    "enabled": False,
                },
            },
        }
        config_path = self._write_yaml(self.root / "full_pretrain.yaml", config)
        run_train(config_path)

        log_text = self._read_log_text("full-pretrain")
        self.assertIn("Pretrain start: epochs=1 batch_size=2", log_text)
        self.assertIn("batches_per_epoch_budget=0", log_text)
        self.assertIn("Pretrain DataLoader ready: batches_per_epoch=3", log_text)
        self.assertIn("budget_batches=0 batch_size=2", log_text)
        self.assertIn("Pretrain epoch 1/1 start: batches=3 batch_size=2", log_text)

    def test_node_pretrain_requires_volume_dataset(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = coords[:, :1] * 0.5
        coords_path = self.root / "node_coords.npy"
        target_path = self.root / "node_target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        config = {
            "experiment": "node-pretrain-error",
            "exp_id": "node-pretrain-error",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "target_path": str(target_path),
            },
            "model": {
                "name": "var_expert",
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
                "seed": 13,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
                "pretrain": {
                    "enabled": True,
                    "epochs": 1,
                    "lr": 1.0e-3,
                },
            },
        }
        config_path = self._write_yaml(self.root / "node_pretrain_error.yaml", config)
        with self.assertRaisesRegex(ValueError, "volume dataset"):
            run_train(config_path)

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
        var_expert_config = {
            "experiment": "node-pipeline",
            "exp_id": "node-pipeline",
            "experiment_root": str(experiment_root),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "var_expert",
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
        var_expert_config_path = self._write_yaml(self.root / "var_expert.yaml", var_expert_config)

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

        run_train(var_expert_config_path)
        rows = self._read_csv_rows(catalog_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["model_name"] for row in rows], ["siren", "var_expert"])

        var_expert_row = rows[1]
        self.assertEqual(var_expert_row["base_dim"], "2")
        self.assertEqual(var_expert_row["num_experts"], "2")
        self.assertTrue(var_expert_row["model_config_hash"])
        self.assertGreater(int(var_expert_row["param_count"]), 0)
        self.assertGreater(int(var_expert_row["trainable_param_count"]), 0)
        self.assertGreater(int(var_expert_row["fp16_size_bytes"]), 0)

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
                "name": "var_expert",
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

    def test_var_expert_logs_utilization_and_ema_state(self):
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
            "experiment": "var-expert-logs",
            "exp_id": "var-expert-logs",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "var_expert",
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
                "batch_size": 3,
                "pred_batch_size": 3,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 5,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 0,
                "sampler": "uniform_random",
                "multiview_ema_loss": {
                    "enabled": True,
                    "warmup_steps": 0,
                },
            },
            "log": {
                "effective_config": True,
                "model_stats": True,
                "epoch_summary": True,
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
        config_path = self._write_yaml(self.root / "var_expert_logs.yaml", config)
        run_train(config_path)

        log_text = self._read_log_text("var-expert-logs")
        self.assertIn("Expert utilization rate:", log_text)
        self.assertIn("EMA balance state: step=", log_text)
        self.assertIn("effective_weights={", log_text)
        self.assertIn("EMA per-target loss (epoch avg):", log_text)
        self.assertIn("a=", log_text)
        self.assertIn("b=", log_text)

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


