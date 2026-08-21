import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.schema import ModelConfig, SchedulerConfig, VolumeShape
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model
from var_expert_inr.models.registry import materialize_model_config
from var_expert_inr.models.sota.hash_grid import (
    MultiresolutionHashEncoding,
    coherent_prime_hash,
)
from var_expert_inr.models.sota.instant_vnr import InstantVNR
from var_expert_inr.training.engine import build_training_scheduler


class InstantVNRHashGridTestCase(unittest.TestCase):
    @staticmethod
    def _encoding() -> MultiresolutionHashEncoding:
        return MultiresolutionHashEncoding(
            dimensions=4,
            n_levels=2,
            n_features_per_level=2,
            base_resolution=4,
            per_level_scale=2.0,
            log2_hashmap_size=8,
        )

    def test_four_dimensional_interpolation_has_sixteen_corners(self):
        encoding = self._encoding()
        coords = torch.rand(7, 4) * 2.0 - 1.0
        vertices, fractions, positions = encoding.grid_geometry(coords, 0)
        weights = encoding.interpolation_weights(fractions)
        self.assertEqual(tuple(vertices.shape), (7, 16, 4))
        self.assertEqual(tuple(weights.shape), (7, 16))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(7))
        torch.testing.assert_close(
            positions,
            0.5 * encoding.level_scales[0] * (coords + 1.0) + 0.5,
        )

    def test_resolutions_capacities_hash_and_boundaries_are_deterministic(self):
        encoding = self._encoding()
        self.assertEqual(encoding.level_resolutions.tolist(), [4, 8])
        self.assertEqual(encoding.level_entries.tolist(), [256, 256])
        self.assertEqual(tuple(encoding.corner_offsets.shape), (16, 4))
        vertices = torch.tensor(
            [[[0, 1, 2, 3], [17, 23, 42, 99]]], dtype=torch.long
        )
        first = coherent_prime_hash(vertices)
        second = coherent_prime_hash(vertices.clone())
        torch.testing.assert_close(first, second)
        boundary = torch.tensor(
            [[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]
        )
        output = encoding(boundary)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_encoding_and_decoder_receive_gradients(self):
        model = InstantVNR(
            n_levels=2,
            n_features_per_level=2,
            base_resolution=4,
            per_level_scale=2.0,
            log2_hashmap_size=8,
            hidden_features=8,
            hidden_layers=2,
        )
        prediction = model(torch.rand(6, 4) * 2.0 - 1.0)
        self.assertEqual(tuple(prediction.shape), (6, 1))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.encoding.feature_tables[0].grad)
        linear_layers = [m for m in model.decoder if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 3)
        self.assertTrue(all(layer.bias is None for layer in linear_layers))
        self.assertTrue(all(layer.weight.grad is not None for layer in linear_layers))


class InstantVNRIntegrationTestCase(unittest.TestCase):
    @staticmethod
    def _meta(*, kind="volume", input_dim=4, targets=("GT",), target_dim=1):
        return DatasetMeta(
            kind=kind,
            n_samples=16,
            input_dim=input_dim,
            target_names=tuple(targets),
            target_dims={name: target_dim for name in targets},
            volume_shape=(
                VolumeShape(X=2, Y=2, Z=2, T=2) if kind == "volume" else None
            ),
        )

    def test_registry_materializes_official_defaults_and_allows_overrides(self):
        materialized = materialize_model_config(
            ModelConfig(name="instant_vnr", params={}), self._meta()
        )
        self.assertEqual(
            materialized,
            {
                "name": "instant_vnr",
                "in_features": 4,
                "out_features": 1,
                "n_levels": 8,
                "n_features_per_level": 8,
                "base_resolution": 16,
                "per_level_scale": 2.0,
                "log2_hashmap_size": 19,
                "hidden_features": 64,
                "hidden_layers": 4,
            },
        )
        model = build_model(
            ModelConfig(
                name="instant_vnr",
                params={
                    "n_levels": 2,
                    "n_features_per_level": 2,
                    "base_resolution": 4,
                    "per_level_scale": 2.0,
                    "log2_hashmap_size": 8,
                    "hidden_features": 8,
                    "hidden_layers": 1,
                },
            ),
            self._meta(),
        )
        self.assertEqual(tuple(model(torch.rand(3, 4)).shape), (3, 1))

    def test_registry_rejects_unsupported_data_and_invalid_fields(self):
        cfg = ModelConfig(name="instant_vnr", params={})
        with self.assertRaisesRegex(ValueError, "only supports volume"):
            materialize_model_config(cfg, self._meta(kind="node"))
        with self.assertRaisesRegex(ValueError, "requires in_features=4"):
            materialize_model_config(cfg, self._meta(input_dim=3))
        with self.assertRaisesRegex(ValueError, "single-target"):
            materialize_model_config(cfg, self._meta(targets=("a", "b")))
        with self.assertRaisesRegex(ValueError, "scalar target"):
            materialize_model_config(cfg, self._meta(target_dim=3))
        with self.assertRaisesRegex(ValueError, "Unknown instant_vnr config keys"):
            materialize_model_config(
                ModelConfig(name="instant_vnr", params={"unknown": 1}),
                self._meta(),
            )
        with self.assertRaisesRegex(ValueError, "per_level_scale > 1"):
            materialize_model_config(
                ModelConfig(name="instant_vnr", params={"per_level_scale": 1.0}),
                self._meta(),
            )

    def test_delayed_piecewise_exponential_scheduler(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=5.0e-3)
        scheduler = build_training_scheduler(
            optimizer,
            SchedulerConfig(
                enabled=True,
                interval="optimizer_step",
                step_size=1000,
                decay_start=2000,
                gamma=0.99,
            ),
        )
        observed = {}
        requested = {1999, 2000, 2999, 3000}
        for update_index in range(3001):
            if update_index in requested:
                observed[update_index] = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(observed[1999], 5.0e-3, places=12)
        self.assertAlmostEqual(observed[2000], 5.0e-3 * 0.99, places=12)
        self.assertAlmostEqual(observed[2999], 5.0e-3 * 0.99, places=12)
        self.assertAlmostEqual(observed[3000], 5.0e-3 * 0.99**2, places=12)

    def test_small_four_dimensional_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(
                2, 2, 2, 2
            )
            volume_path = root / "volume.npy"
            np.save(volume_path, volume)
            config = {
                "experiment": "instant-vnr-smoke",
                "exp_id": "instant-vnr-smoke",
                "experiment_root": str(root / "runs"),
                "data": {"kind": "volume", "target_path": str(volume_path)},
                "model": {
                    "name": "instant_vnr",
                    "n_levels": 2,
                    "n_features_per_level": 2,
                    "base_resolution": 4,
                    "per_level_scale": 2.0,
                    "log2_hashmap_size": 8,
                    "hidden_features": 8,
                    "hidden_layers": 1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 4,
                    "pred_batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "batches_per_epoch_budget": 2,
                    "num_workers": 0,
                    "lr": 5.0e-3,
                    "loss_type": "l1",
                    "device": "cpu",
                    "val_split": 0.0,
                    "log_every": 1,
                    "save_every": 2,
                    "sampler": "budgeted_random",
                },
                "evaluation": {"batch_size": 4, "save_predictions": False},
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            result = run_train(config_path)
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            self.assertEqual(result["global_data_step"], 4)
            self.assertEqual(result["global_optimizer_step"], 2)


if __name__ == "__main__":
    unittest.main()
