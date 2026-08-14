import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.schema import ExplorationProbeConfig
from var_expert_inr.utils.exploration_probe import (
    fixed_sample_indices,
    probe_due,
    probe_progress,
    probe_temporal_volume_ensemble_model,
)


class _ProbeVolume:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float32).reshape(1, 1, -1)
        self.spatial_shape = self.values.shape
        self.shape = {"T": 1}

    def frame(self, timestep: int) -> np.ndarray:
        if timestep != 0:
            raise IndexError(timestep)
        return self.values


class _FixedEnsemble(torch.nn.Module):
    def __init__(self, members: np.ndarray):
        super().__init__()
        self.register_buffer("members", torch.as_tensor(members, dtype=torch.float32))

    def forward_members(self, coords: torch.Tensor, timestep: int) -> torch.Tensor:
        del timestep
        indices = torch.round(coords[:, 0] * (self.members.shape[0] - 1)).long()
        return self.members[indices].unsqueeze(-1)


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
                    "grad_clip_norm": 1.0,
                },
                "evaluation": {"save_predictions": False},
                "exploration_probe": {
                    "enabled": True,
                    "total_epoch_equivalents": 50,
                    "every_epoch_equivalents": 5,
                    "sample_ratio": 0.2,
                    "max_samples": 100_000,
                    "seed": 42,
                    "retain_best_checkpoint": True,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            original_clip = torch.nn.utils.clip_grad_norm_
            with mock.patch(
                "torch.nn.utils.clip_grad_norm_", wraps=original_clip
            ) as clip:
                result = run_train(config_path)
            self.assertGreater(clip.call_count, 0)
            best_checkpoint = Path(result["best_probe_checkpoint_path"])
            final_checkpoint = Path(result["checkpoint_path"])
            self.assertTrue(best_checkpoint.exists())
            self.assertTrue(final_checkpoint.exists())
            self.assertNotEqual(best_checkpoint, final_checkpoint)
            self.assertIn(result["best_probe_progress"], range(5, 51, 5))

            run_dir = max((root / "runs" / "probe-test").iterdir())
            metrics_path = run_dir / "metrics" / "exploration_psnr.tsv"
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 10)
            self.assertEqual([int(row["progress"]) for row in rows], list(range(5, 51, 5)))
            self.assertEqual({int(row["sample_count"]) for row in rows}, {20})
            self.assertEqual({row["scope"] for row in rows}, {"aggregate"})

    def test_ensemble_probe_uncertainty_metrics_are_finite_and_reproducible(self):
        count = 100
        targets = np.zeros(count, dtype=np.float32)
        probe = ExplorationProbeConfig(
            enabled=True,
            sample_ratio=1.0,
            max_samples=count,
            seed=42,
        )
        amplitude = np.linspace(0.25, 2.0, count, dtype=np.float32)
        correlated = np.stack(
            [
                amplitude - amplitude / np.sqrt(2.0),
                amplitude + amplitude / np.sqrt(2.0),
            ],
            axis=1,
        )
        constant_variance = np.stack(
            [amplitude - 0.5, amplitude + 0.5], axis=1
        )
        random_members = np.random.default_rng(7).normal(size=(count, 5)).astype(np.float32)

        results = []
        for members in (correlated, constant_variance, random_members):
            result = probe_temporal_volume_ensemble_model(
                model=_FixedEnsemble(members),
                volume=_ProbeVolume(targets),
                device=torch.device("cpu"),
                batch_size=17,
                probe=probe,
                variance_weight=0.25,
                epsilon=1.0e-12,
                topk_fractions=[0.01, 0.05],
            )
            repeated = probe_temporal_volume_ensemble_model(
                model=_FixedEnsemble(members),
                volume=_ProbeVolume(targets),
                device=torch.device("cpu"),
                batch_size=31,
                probe=probe,
                variance_weight=0.25,
                epsilon=1.0e-12,
                topk_fractions=[0.01, 0.05],
            )
            self.assertEqual(result[1], count)
            self.assertAlmostEqual(result[0], repeated[0])
            self.assertEqual(result[2], repeated[2])
            for key, value in result[2].items():
                if key == "topk_hit_rate":
                    self.assertEqual(set(value), {"0.01", "0.05"})
                    self.assertTrue(all(math.isfinite(item) for item in value.values()))
                else:
                    self.assertTrue(math.isfinite(value), key)
            results.append(result)

        self.assertAlmostEqual(
            results[0][2]["variance_error_pearson"], 1.0, places=6
        )
        self.assertAlmostEqual(
            results[1][2]["variance_error_pearson"], 0.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
