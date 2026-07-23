import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_evaluate, run_predict, run_train
from var_expert_inr.models.sota.compact_ngp import (
    COHERENT_PRIMES,
    REVERSED_PRIMES,
    CompactNGP,
    hash_vertices,
    load_compact_ngp_artifact,
    pack_probe_offsets,
    save_compact_ngp_artifact,
    unpack_probe_offsets,
)


def _small_model() -> CompactNGP:
    return CompactNGP(
        num_levels=2,
        features_per_level=2,
        feature_table_size=8,
        index_table_size=8,
        num_probes=4,
        base_resolution=4,
        max_resolution=8,
        hidden_features=4,
        hidden_layers=2,
    )


class CompactNGPTestCase(unittest.TestCase):
    def test_hashes_match_uint32_scalar_reference(self):
        vertices = torch.tensor(
            [[0, 1, 2, 3], [2048, 4096, 65535, 17], [16, 16, 16, 16]],
            dtype=torch.long,
        )
        for factors in (COHERENT_PRIMES, REVERSED_PRIMES):
            expected = []
            for vertex in vertices.tolist():
                value = 0
                for coordinate, factor in zip(vertex, factors):
                    value ^= (coordinate * factor) & 0xFFFFFFFF
                expected.append(value & 0xFFFFFFFF)
            actual = hash_vertices(vertices, factors)
            torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.long))
        self.assertNotEqual(
            hash_vertices(vertices, COHERENT_PRIMES).tolist(),
            hash_vertices(vertices, REVERSED_PRIMES).tolist(),
        )

    def test_fused_grid_transform_and_unclipped_boundary(self):
        model = _small_model()
        coords = torch.tensor(
            [[-1.0, -0.5, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        for level in range(model.num_levels):
            vertices, fractions, grid_pos = model.grid_geometry(coords, level)
            scale = model.level_scales[level]
            explicit = scale * ((coords + 1.0) * 0.5) + 0.5
            torch.testing.assert_close(grid_pos, explicit)
            torch.testing.assert_close(
                fractions, grid_pos - torch.floor(grid_pos)
            )
            self.assertTrue(torch.all(vertices[:, 1:, :] >= 0))

        last = model.num_levels - 1
        vertices, _, _ = model.grid_geometry(coords[-1:], last)
        resolution = int(model.level_resolutions[last].item())
        self.assertGreater(int(vertices.max().item()), resolution - 1)

    def test_interpolation_weights_sum_to_one(self):
        model = _small_model()
        coords = torch.rand(7, 4) * 2.0 - 1.0
        _, fractions, _ = model.grid_geometry(coords, 0)
        weights = model.interpolation_weights(fractions)
        self.assertEqual(tuple(weights.shape), (7, 16))
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(7), atol=1e-6, rtol=1e-6
        )

    def test_straight_through_gradients_and_baked_cache(self):
        torch.manual_seed(7)
        model = _small_model()
        coords = torch.rand(5, 4) * 2.0 - 1.0
        output = model(coords)
        self.assertEqual(tuple(output.shape), (5, 1))
        output.sum().backward()
        self.assertGreater(
            sum(float(table.grad.abs().sum()) for table in model.feature_tables), 0.0
        )
        self.assertGreater(
            sum(float(table.grad.abs().sum()) for table in model.confidence_tables), 0.0
        )

        model.eval()
        first_cache = model._baked_indices
        expected = model(coords).detach()
        self.assertTrue(model._baked_valid)
        model.confidence_tables[0].data.fill_(float("nan"))
        actual = model(coords).detach()
        torch.testing.assert_close(actual, expected)
        self.assertIs(model._baked_indices, first_cache)
        model.train()
        self.assertFalse(model._baked_valid)
        self.assertIsNone(model._baked_indices)

    def test_two_bit_pack_and_artifact_roundtrip(self):
        offsets = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.uint8)
        packed = pack_probe_offsets(offsets)
        self.assertEqual(packed.numel(), 2)
        torch.testing.assert_close(unpack_probe_offsets(packed, 8), offsets)

        model = _small_model().eval()
        coords = torch.rand(6, 4) * 2.0 - 1.0
        reference = model(coords).detach()
        config = {
            "name": "compact_ngp",
            "in_features": 4,
            "out_features": 1,
            "num_levels": 2,
            "features_per_level": 2,
            "feature_table_size": 8,
            "index_table_size": 8,
            "num_probes": 4,
            "base_resolution": 4,
            "max_resolution": 8,
            "hidden_features": 4,
            "hidden_layers": 2,
        }

        class Shape:
            def to_dict(self):
                return {"X": 1, "Y": 1, "Z": 1, "T": 1}

        dataset = SimpleNamespace(
            meta=SimpleNamespace(volume_shape=Shape()),
            target_names=lambda: ("GT",),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compact.pt"
            result = save_compact_ngp_artifact(
                path,
                model=model,
                model_config=config,
                dataset=dataset,
                config_hash="hash",
            )
            inference, payload = load_compact_ngp_artifact(path)
            quantized = inference(coords)
            self.assertNotIn("confidence_tables", payload)
            self.assertEqual(payload["feature_tables"].dtype, torch.float16)
            self.assertEqual(payload["packed_indices"].dtype, torch.uint8)
            self.assertEqual(result["compact_payload_bytes"], 158)
            torch.testing.assert_close(quantized, reference, atol=5e-4, rtol=5e-3)

    def test_unified_training_checkpoint_and_artifact_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
            target_path = root / "volume.npy"
            np.save(target_path, volume)
            config = {
                "experiment": "compact-smoke",
                "exp_id": "compact-smoke",
                "experiment_root": str(root / "runs"),
                "data": {"kind": "volume", "target_path": str(target_path)},
                "model": {
                    "name": "compact_ngp",
                    "in_features": 4,
                    "num_levels": 2,
                    "features_per_level": 2,
                    "feature_table_size": 8,
                    "index_table_size": 8,
                    "num_probes": 4,
                    "base_resolution": 4,
                    "max_resolution": 8,
                    "hidden_features": 4,
                    "hidden_layers": 2,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "pred_batch_size": 4,
                    "num_workers": 0,
                    "lr": 1.0e-2,
                    "beta_1": 0.9,
                    "beta_2": 0.99,
                    "epsilon": 1.0e-15,
                    "weight_decay": 1.0e-6,
                    "device": "cpu",
                    "seed": 4,
                    "val_split": 0.0,
                    "log_every": 1,
                    "save_every": 0,
                    "sampler": "uniform_random",
                },
                "evaluation": {"batch_size": 4, "save_predictions": False},
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            trained = run_train(config_path)
            self.assertTrue(Path(trained["checkpoint_path"]).exists())
            self.assertTrue(Path(trained["artifact_path"]).exists())
            checkpoint_prediction = run_predict(
                config_path, checkpoint_path=trained["checkpoint_path"]
            )
            artifact_prediction = run_predict(
                config_path, artifact_path=trained["artifact_path"]
            )
            checkpoint_values = next(iter(checkpoint_prediction["predictions"].values()))
            artifact_values = next(iter(artifact_prediction["predictions"].values()))
            np.testing.assert_allclose(
                artifact_values, checkpoint_values, atol=5e-4, rtol=5e-3
            )
            evaluated = run_evaluate(
                config_path, artifact_path=trained["artifact_path"]
            )
            self.assertTrue(Path(evaluated["metrics_path"]).exists())


if __name__ == "__main__":
    unittest.main()
