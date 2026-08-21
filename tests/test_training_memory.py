import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.io import load_experiment_config
from var_expert_inr.config.schema import MemoryLogConfig
from var_expert_inr.utils.memory import TrainingMemoryTracker


class TrainingMemoryTestCase(unittest.TestCase):
    def _write_config(
        self,
        root: Path,
        *,
        exp_id: str,
        memory_enabled: bool,
        gradient_accumulation_steps: int = 1,
    ) -> Path:
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = root / f"{exp_id}.npy"
        np.save(volume_path, volume)
        payload = {
            "experiment": exp_id,
            "exp_id": exp_id,
            "experiment_root": str(root / "runs"),
            "data": {"kind": "volume", "target_path": str(volume_path)},
            "model": {
                "name": "siren",
                "in_features": 4,
                "hidden_features": 8,
                "hidden_layers": 1,
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
                "save_every": 0,
                "sampler": "uniform_random",
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "evaluation": {"save_predictions": False},
            "log": {
                "timing": {"enabled": False},
                "memory": {
                    "enabled": memory_enabled,
                    "sample_interval_seconds": 0.001,
                },
            },
        }
        config_path = root / f"{exp_id}.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return config_path

    def _latest_run(self, root: Path, exp_id: str) -> Path:
        return sorted((root / "runs" / exp_id).iterdir())[-1]

    def test_memory_config_defaults_and_validation(self):
        self.assertFalse(MemoryLogConfig().enabled)
        self.assertEqual(MemoryLogConfig().sample_interval_seconds, 0.01)
        with self.assertRaisesRegex(ValueError, "sample_interval_seconds must be positive"):
            MemoryLogConfig(sample_interval_seconds=0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root, exp_id="memory-config", memory_enabled=True)
            loaded = load_experiment_config(config_path)
            self.assertTrue(loaded.log.memory.enabled)
            self.assertEqual(loaded.log.memory.sample_interval_seconds, 0.001)
            self.assertTrue(loaded.to_dict()["log"]["memory"]["enabled"])

    def test_enabled_cpu_training_writes_peak_and_step_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(
                root,
                exp_id="memory-enabled",
                memory_enabled=True,
                gradient_accumulation_steps=2,
            )
            result = run_train(config_path)

            memory_path = Path(result["training_memory_path"])
            expected = Path(result["checkpoint_path"]).parent.parent / "metrics" / "training_memory.json"
            self.assertEqual(memory_path, expected)
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, result["training_memory"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["scope"], "optimization_steps")
            self.assertEqual(payload["device"], "cpu")
            self.assertEqual(payload["measured_data_steps"], 4)
            self.assertEqual(payload["measured_optimizer_steps"], 2)
            self.assertGreater(payload["cpu_rss_baseline_bytes"], 0)
            self.assertGreaterEqual(payload["cpu_rss_peak_bytes"], payload["cpu_rss_baseline_bytes"])
            self.assertGreaterEqual(payload["cpu_rss_peak_delta_bytes"], 0)
            self.assertIsNone(payload["cuda_peak_allocated_bytes"])
            self.assertIsNone(payload["error_type"])
            effective = yaml.safe_load(
                (memory_path.parent.parent / "configs" / "config.yaml").read_text(encoding="utf-8")
            )
            self.assertTrue(effective["log"]["memory"]["enabled"])
            self.assertEqual(effective["log"]["memory"]["sample_interval_seconds"], 0.001)

    def test_disabled_training_does_not_write_memory_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root, exp_id="memory-disabled", memory_enabled=False)
            result = run_train(config_path)
            self.assertNotIn("training_memory", result)
            self.assertNotIn("training_memory_path", result)
            memory_path = self._latest_run(root, "memory-disabled") / "metrics" / "training_memory.json"
            self.assertFalse(memory_path.exists())

    def test_failure_before_first_data_step_reports_null_peaks(self):
        tracker = TrainingMemoryTracker("cpu", sample_interval_seconds=0.001)
        payload = tracker.close(status="failed", error_type="RuntimeError")
        self.assertEqual(payload["measured_data_steps"], 0)
        self.assertEqual(payload["measured_optimizer_steps"], 0)
        self.assertIsNone(payload["cpu_rss_baseline_bytes"])
        self.assertIsNone(payload["cpu_rss_peak_bytes"])
        self.assertIsNone(payload["cpu_rss_peak_delta_bytes"])
        self.assertIsNone(payload["cuda_peak_allocated_bytes"])

    def test_failed_training_persists_partial_measurement_and_reraises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root, exp_id="memory-failed", memory_enabled=True)
            with patch(
                "var_expert_inr.training.engine.pointwise_loss",
                side_effect=RuntimeError("intentional training failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "intentional training failure"):
                    run_train(config_path)

            memory_path = self._latest_run(root, "memory-failed") / "metrics" / "training_memory.json"
            payload = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error_type"], "RuntimeError")
            self.assertEqual(payload["measured_data_steps"], 1)
            self.assertEqual(payload["measured_optimizer_steps"], 0)
            self.assertGreater(payload["cpu_rss_peak_bytes"], 0)

    def test_validation_checkpoint_and_prediction_run_outside_step_window(self):
        from var_expert_inr.training import engine

        class WindowTracker:
            instance = None

            def __init__(self, device, *, sample_interval_seconds):
                type(self).instance = self
                self.device = torch.device(device)
                self.sample_interval_seconds = sample_interval_seconds
                self.active = False
                self.data_steps = 0
                self.optimizer_steps = 0

            def start_data_step(self):
                self.active = True

            def confirm_data_step(self):
                self.data_steps += 1

            def cancel_data_step(self):
                self.active = False

            def record_optimizer_step(self):
                self.optimizer_steps += 1

            def finish_data_step(self):
                self.active = False

            def close(self, *, status, error_type=None):
                self.active = False
                return {
                    "schema_version": 1,
                    "status": status,
                    "scope": "optimization_steps",
                    "device": str(self.device),
                    "sample_interval_seconds": self.sample_interval_seconds,
                    "measured_data_steps": self.data_steps,
                    "measured_optimizer_steps": self.optimizer_steps,
                    "cpu_rss_baseline_bytes": 1,
                    "cpu_rss_peak_bytes": 1,
                    "cpu_rss_peak_delta_bytes": 0,
                    "cuda_peak_allocated_bytes": None,
                    "error_type": error_type,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root, exp_id="memory-boundaries", memory_enabled=True)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["training"]["val_split"] = 0.25
            config["training"]["save_every"] = 1
            config["evaluation"]["save_predictions"] = True
            config["exploration_probe"] = {
                "enabled": True,
                "total_epoch_equivalents": 1,
                "every_epoch_equivalents": 1,
                "sample_ratio": 1.0,
                "max_samples": 8,
                "seed": 7,
                "retain_best_checkpoint": False,
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            original_predict_batch = engine._predict_batch
            original_save_checkpoint = engine.save_checkpoint
            original_predict_dataset = engine.predict_dataset

            def checked_predict_batch(*args, **kwargs):
                if not torch.is_grad_enabled():
                    self.assertFalse(WindowTracker.instance.active)
                return original_predict_batch(*args, **kwargs)

            def checked_save_checkpoint(*args, **kwargs):
                self.assertFalse(WindowTracker.instance.active)
                return original_save_checkpoint(*args, **kwargs)

            def checked_predict_dataset(*args, **kwargs):
                self.assertFalse(WindowTracker.instance.active)
                return original_predict_dataset(*args, **kwargs)

            with (
                patch("var_expert_inr.training.engine.TrainingMemoryTracker", WindowTracker),
                patch("var_expert_inr.training.engine._predict_batch", new=checked_predict_batch),
                patch("var_expert_inr.training.engine.save_checkpoint", new=checked_save_checkpoint),
                patch("var_expert_inr.training.engine.predict_dataset", new=checked_predict_dataset),
            ):
                run_train(config_path)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_tracker_reports_known_allocation(self):
        tracker = TrainingMemoryTracker("cuda:0", sample_interval_seconds=0.001)
        tracker.start_data_step()
        tracker.confirm_data_step()
        allocation = torch.empty(1024 * 1024, dtype=torch.float32, device="cuda:0")
        tracker.record_optimizer_step()
        tracker.finish_data_step()
        payload = tracker.close(status="completed")
        self.assertGreaterEqual(payload["cuda_peak_allocated_bytes"], allocation.numel() * allocation.element_size())


if __name__ == "__main__":
    unittest.main()
