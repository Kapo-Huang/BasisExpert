import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.schema import ExplorationProbeConfig
from var_expert_inr.utils.exploration_probe import fixed_sample_indices, probe_due, probe_progress


class ExplorationProbeTestCase(unittest.TestCase):
    def test_fixed_sampling_and_progress_schedule(self):
        probe = ExplorationProbeConfig(
            enabled=True,
            total_epoch_equivalents=50,
            every_epoch_equivalents=5,
            sample_ratio=0.01,
            max_samples=100_000,
            seed=42,
        )
        first = fixed_sample_indices(20_000_000, probe)
        second = fixed_sample_indices(20_000_000, probe)
        salted = fixed_sample_indices(20_000_000, probe, salt=1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 100_000)
        self.assertFalse(np.array_equal(first, salted))
        due = [step for step in range(1, 51) if probe_due(step, 50, probe)]
        self.assertEqual(due, list(range(5, 51, 5)))
        self.assertEqual([probe_progress(step, 50, probe) for step in due], due)

    def test_unified_runner_writes_ten_probe_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coords = np.linspace(-1.0, 1.0, 400, dtype=np.float32).reshape(100, 4)
            targets = np.sin(coords[:, :1]).astype(np.float32)
            coords_path = root / "coords.npy"
            target_path = root / "target.npy"
            np.save(coords_path, coords)
            np.save(target_path, targets)
            config = {
                "experiment": "probe-test",
                "exp_id": "probe-test",
                "experiment_root": str(root / "runs"),
                "data": {"kind": "node", "coords_path": str(coords_path), "target_path": str(target_path)},
                "model": {"name": "siren", "hidden_features": 8, "hidden_layers": 1},
                "training": {
                    "epochs": 10,
                    "batch_size": 20,
                    "pred_batch_size": 32,
                    "num_workers": 0,
                    "lr": 1.0e-3,
                    "device": "cpu",
                    "seed": 7,
                    "val_split": 0.0,
                    "log_every": 1,
                    "log_psnr_every": 1,
                    "psnr_sample_ratio": 0.2,
                    "save_every": 0,
                    "sampler": "uniform_random",
                },
                "evaluation": {"save_predictions": False},
                "exploration_probe": {
                    "enabled": True,
                    "total_epoch_equivalents": 50,
                    "every_epoch_equivalents": 5,
                    "sample_ratio": 0.2,
                    "max_samples": 100_000,
                    "seed": 42,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            run_train(config_path)

            run_dir = max((root / "runs" / "probe-test").iterdir())
            metrics_path = run_dir / "metrics" / "exploration_psnr.tsv"
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 10)
            self.assertEqual([int(row["progress"]) for row in rows], list(range(5, 51, 5)))
            self.assertEqual({int(row["sample_count"]) for row in rows}, {20})
            self.assertEqual({row["scope"] for row in rows}, {"aggregate"})


if __name__ == "__main__":
    unittest.main()
