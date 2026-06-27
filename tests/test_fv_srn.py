from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from var_expert_inr.fv_srn.config import load_config
from var_expert_inr.fv_srn.data import TemporalVolume, build_sample_pool
from var_expert_inr.fv_srn.model import SnakeAlt, TemporalFVSRN, nerf_fourier_matrix
from var_expert_inr.fv_srn.quantization import dequantize_grid, quantize_grids
from var_expert_inr.fv_srn.runner import run_evaluate, run_train


class FVSRNTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _config(self, data_path: Path, *, run_after_training: bool = True) -> Path:
        payload = {
            "exp_id": "fv-smoke-{target}",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target": "GT",
                "targets": {"GT": str(data_path), "PD": str(data_path)},
                "volume_shape": {"T": 2, "Z": 3, "Y": 3, "X": 4},
            },
            "model": {
                "name": "fv_srn", "grid_resolution": 2, "grid_channels": 2,
                "grid_init_std": 0.01, "keyframe_indices": [0, 1],
                "fourier_features": 3, "fourier_mode": "nerf",
                "hidden_features": 8, "hidden_layers": 2,
                "activation": "snake_alt", "activation_frequency": 1.0,
                "time_encoding": "none",
            },
            "training": {
                "epochs": 1, "samples_per_timestep": 32, "validation_fraction": 0.25,
                "batch_size": 32, "prediction_batch_size": 16, "lr": 0.01,
                "beta_1": 0.9, "beta_2": 0.999, "lr_step": 1, "lr_gamma": 0.5,
                "l1_weight": 1.0, "l2_weight": 0.0, "importance_floor": 0.2,
                "rebuild_every": 0, "rebuild_grid_size": 2,
                "rebuild_samples_per_cell": 1, "save_every": 0,
                "log_every": 1, "seed": 7, "device": "cpu",
            },
            "evaluation": {
                "batch_size": 16, "save_predictions": True,
                "run_after_training": run_after_training, "default_model": "compact",
            },
        }
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_model_encoding_interpolation_and_snake(self):
        matrix = nerf_fourier_matrix(4)
        self.assertEqual(tuple(matrix.shape), (4, 3))
        x = torch.tensor([[0.0, 0.5]])
        expected = (x + 1 - torch.cos(2 * x)) / 2
        torch.testing.assert_close(SnakeAlt(1)(x), expected)
        cfg = {
            "grid_resolution": 2, "grid_channels": 1, "grid_init_std": 0.01,
            "keyframe_indices": [0, 9, 18], "fourier_features": 3,
            "hidden_features": 4, "hidden_layers": 1, "activation_frequency": 1.0,
        }
        model = TemporalFVSRN(cfg)
        with torch.no_grad():
            model.feature_grids[0].fill_(0)
            model.feature_grids[1].fill_(9)
        features = model.grid_features(torch.rand(5, 3), 4.0)
        torch.testing.assert_close(features, torch.full_like(features, 4.0))
        self.assertEqual(model.keyframe_pair(18), (2, 2, 0.0))

    def test_data_shape_sampling_pool_and_quantization(self):
        dense = np.arange(72, dtype=np.float32).reshape(2, 3, 3, 4)
        dense = dense / dense.max() * 2 - 1
        path = self.root / "dense.npy"
        np.save(path, dense.reshape(-1, 1))
        volume = TemporalVolume(path, {"T": 2, "Z": 3, "Y": 3, "X": 4})
        corners = volume.sample(0, np.array([[0, 0, 0], [1, 1, 1]], np.float32))
        self.assertAlmostEqual(float(corners[0]), float(dense[0, 0, 0, 0]))
        self.assertAlmostEqual(float(corners[1]), float(dense[0, -1, -1, -1]))
        pool = build_sample_pool(
            volume, count_per_timestep=16, validation_fraction=0.25,
            floor=0.2, rng=np.random.default_rng(4),
        )
        self.assertEqual(len(pool.train_indices[0]), 12)
        grids = torch.stack([torch.zeros(2, 2, 2, 2), torch.ones(2, 2, 2, 2)])
        q, minimum, scale = quantize_grids(grids)
        restored = dequantize_grid(q[1], minimum[1], scale[1], device=torch.device("cpu"), dtype=torch.float32)
        torch.testing.assert_close(restored, grids[1])

    def test_config_rejects_missing_temporal_endpoint(self):
        data = np.zeros((2, 3, 3, 4), dtype=np.float32)
        path = self.root / "data.npy"
        np.save(path, data)
        config = self._config(path)
        payload = yaml.safe_load(config.read_text())
        payload["model"]["keyframe_indices"] = [0]
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "keyframe"):
            load_config(config)

    def test_train_compact_predict_evaluate(self):
        z, y, x = np.meshgrid(
            np.linspace(-1, 1, 3), np.linspace(-1, 1, 3),
            np.linspace(-1, 1, 4), indexing="ij",
        )
        data = np.stack([(x + y + z) / 3, (x - y + z) / 3]).astype(np.float32)
        data_path = self.root / "data.npy"
        np.save(data_path, data)
        config = self._config(data_path)
        result = run_train(config, target="GT")
        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertTrue(Path(result["artifact_path"]).exists())
        self.assertGreater(result["compact_cr"], 0)
        evaluated = run_evaluate(config, target="GT", artifact=result["artifact_path"])
        prediction = np.load(evaluated["prediction_path"], mmap_mode="r")
        self.assertEqual(prediction.shape, data.shape)
        self.assertIn("psnr", evaluated["metrics"]["aggregate"])


if __name__ == "__main__":
    unittest.main()
