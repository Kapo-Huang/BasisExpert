import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from var_expert_inr.config.schema import (
    ModelConfig,
    SchedulerConfig,
    VolumeShape,
)
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models.registry import materialize_model_config
from var_expert_inr.models.baselines.instant_ngp import (
    INSTANT_NGP_DECODER_L2_WEIGHT,
    InstantNGP,
    MultiresolutionHashEncoding4D,
    coherent_prime_hash,
)
from var_expert_inr.training.engine import (
    build_training_scheduler,
    training_budget,
)
from var_expert_inr.utils.checkpoint import (
    read_checkpoint_payload,
    save_checkpoint,
)


def _small_encoding() -> MultiresolutionHashEncoding4D:
    return MultiresolutionHashEncoding4D(
        n_levels=2,
        n_features_per_level=2,
        base_resolution=4,
        finest_resolution=8,
        log2_hashmap_size=8,
    )


class InstantNGPEncodingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = InstantNGP()

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_fixed_level_resolutions_capacities_and_alignment(self):
        encoding = self.model.encoding
        self.assertEqual(
            encoding.level_resolutions.tolist(),
            [
                16,
                21,
                26,
                34,
                43,
                54,
                69,
                87,
                111,
                141,
                180,
                229,
                291,
                371,
                472,
                600,
            ],
        )
        self.assertEqual(
            encoding.level_entries.tolist(),
            [65_536, 194_488, 456_976] + [524_288] * 13,
        )
        self.assertTrue(
            all(entries % 8 == 0 for entries in encoding.level_entries.tolist())
        )
        self.assertEqual(encoding.output_dim, 32)

    def test_coherent_prime_hash_matches_uint32_scalar_reference(self):
        vertices = torch.tensor(
            [
                [0, 1, 2, 3],
                [600, 248, 100, 65_535],
                [4_294_967_295, 17, 23, 42],
            ],
            dtype=torch.long,
        )
        primes = (1, 2_654_435_761, 805_459_861, 3_674_653_429)
        expected = []
        for vertex in vertices.tolist():
            value = 0
            for coordinate, prime in zip(vertex, primes):
                value ^= (coordinate * prime) & 0xFFFFFFFF
            expected.append(value & 0xFFFFFFFF)
        torch.testing.assert_close(
            coherent_prime_hash(vertices),
            torch.tensor(expected, dtype=torch.long),
        )

    def test_encoder_maps_without_mutating_minus_one_to_one_input(self):
        encoding = _small_encoding()
        coords = torch.tensor(
            [[-1.0, -0.5, 0.25, 1.0], [1.0, 0.0, -1.0, 0.5]],
            dtype=torch.float32,
        )
        original = coords.clone()
        for level in range(encoding.n_levels):
            _, fractions, positions = encoding.grid_geometry(coords, level)
            scale = encoding.level_scales[level]
            expected = 0.5 * scale * (coords + 1.0) + 0.5
            torch.testing.assert_close(positions, expected)
            torch.testing.assert_close(
                fractions,
                positions - torch.floor(positions),
            )
        encoding(coords)
        torch.testing.assert_close(coords, original)

    def test_dense_indices_are_x_first_and_boundaries_wrap(self):
        encoding = _small_encoding()
        vertices = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.long)
        indices = encoding.vertex_indices(vertices, level=0)
        raw_x_first = 1 + 4 * 2 + 4**2 * 3 + 4**3 * 4
        self.assertEqual(indices.item(), raw_x_first % 4**4)

    def test_four_dimensional_interpolation_uses_sixteen_corners(self):
        encoding = _small_encoding()
        coords = torch.rand(11, 4) * 2.0 - 1.0
        vertices, fractions, _ = encoding.grid_geometry(coords, level=0)
        weights = encoding.interpolation_weights(fractions)
        self.assertEqual(tuple(vertices.shape), (11, 16, 4))
        self.assertEqual(tuple(weights.shape), (11, 16))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(11),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_level_features_are_concatenated_in_level_order(self):
        encoding = _small_encoding()
        with torch.no_grad():
            for level, table in enumerate(encoding.feature_tables):
                table[:, 0].fill_(level + 0.25)
                table[:, 1].fill_(level + 0.75)
        output = encoding(torch.rand(5, 4) * 2.0 - 1.0)
        expected = torch.tensor([0.25, 0.75, 1.25, 1.75]).repeat(5, 1)
        torch.testing.assert_close(output, expected)

    def test_fixed_decoder_initialization_and_decoder_only_l2(self):
        model = self.model
        linear_layers = [
            module
            for module in model.decoder
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(32, 64), (64, 64), (64, 1)],
        )
        self.assertTrue(all(layer.bias is None for layer in linear_layers))
        for table in model.encoding.feature_tables:
            self.assertGreaterEqual(float(table.min()), -1.0e-4)
            self.assertLessEqual(float(table.max()), 1.0e-4)
        for layer in linear_layers:
            bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
            self.assertLessEqual(float(layer.weight.abs().max()), bound + 1.0e-6)

        expected_l2 = sum(
            parameter.float().square().sum()
            for parameter in model.decoder.parameters()
        )
        torch.testing.assert_close(
            model.decoder_l2_regularization(),
            expected_l2,
        )
        self.assertEqual(
            {
                id(parameter)
                for parameter in model.decoder.parameters()
            }.intersection(
                {id(table) for table in model.encoding.feature_tables}
            ),
            set(),
        )

    def test_forward_is_scalar_and_architecture_is_fixed(self):
        output = self.model(torch.rand(3, 4) * 2.0 - 1.0)
        self.assertEqual(tuple(output.shape), (3, 1))
        with self.assertRaisesRegex(ValueError, "fixed architecture"):
            InstantNGP(n_levels=15)


class InstantNGPIntegrationTestCase(unittest.TestCase):
    def _meta(self, *, targets=("GT",), x=600):
        return DatasetMeta(
            kind="volume",
            n_samples=1,
            input_dim=4,
            target_names=tuple(targets),
            target_dims={name: 1 for name in targets},
            volume_shape=VolumeShape(X=x, Y=248, Z=248, T=100),
        )

    def test_registry_accepts_only_fixed_single_attribute_ionization_shape(self):
        cfg = ModelConfig(
            name="instant_ngp",
            params={
                "in_features": 4,
                "n_levels": 16,
                "n_features_per_level": 2,
                "base_resolution": 16,
                "finest_resolution": 600,
                "log2_hashmap_size": 19,
                "hidden_features": 64,
                "hidden_layers": 2,
            },
        )
        materialized = materialize_model_config(cfg, self._meta())
        self.assertEqual(materialized["name"], "instant_ngp")
        self.assertEqual(materialized["out_features"], 1)
        with self.assertRaisesRegex(ValueError, "single-target"):
            materialize_model_config(cfg, self._meta(targets=("GT", "H2")))
        node_meta = DatasetMeta(
            kind="node",
            n_samples=1,
            input_dim=4,
            target_names=("GT",),
            target_dims={"GT": 1},
        )
        with self.assertRaisesRegex(ValueError, "only supports volume"):
            materialize_model_config(cfg, node_meta)
        with self.assertRaisesRegex(ValueError, "requires n_levels=16"):
            materialize_model_config(
                ModelConfig(name="instant_ngp", params={"n_levels": 15}),
                self._meta(),
            )

    def test_accumulated_gradients_match_full_effective_batch(self):
        torch.manual_seed(17)
        full = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1, bias=False),
        )
        accumulated = torch.nn.Sequential(
            torch.nn.Linear(4, 8, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1, bias=False),
        )
        accumulated.load_state_dict(full.state_dict())
        coords = torch.randn(32, 4)
        targets = torch.randn(32, 1)

        full_loss = torch.nn.functional.mse_loss(full(coords), targets)
        full_regularization = INSTANT_NGP_DECODER_L2_WEIGHT * sum(
            parameter.square().sum() for parameter in full.parameters()
        )
        (full_loss + full_regularization).backward()

        for micro_coords, micro_targets in zip(
            coords.split(8),
            targets.split(8),
        ):
            micro_loss = torch.nn.functional.mse_loss(
                accumulated(micro_coords),
                micro_targets,
            )
            (micro_loss / 4.0).backward()
        accumulated_regularization = INSTANT_NGP_DECODER_L2_WEIGHT * sum(
            parameter.square().sum()
            for parameter in accumulated.parameters()
        )
        accumulated_regularization.backward()

        for full_parameter, accumulated_parameter in zip(
            full.parameters(),
            accumulated.parameters(),
        ):
            torch.testing.assert_close(
                full_parameter.grad,
                accumulated_parameter.grad,
                atol=1.0e-7,
                rtol=1.0e-6,
            )

    def test_training_budget_and_cross_epoch_accumulation(self):
        budget = training_budget(
            epochs=600,
            data_steps_per_epoch=1_500,
            batch_size=16_000,
            gradient_accumulation_steps=16,
        )
        self.assertEqual(
            budget,
            {
                "data_steps": 900_000,
                "samples": 14_400_000_000,
                "optimizer_steps": 56_250,
                "remainder": 0,
            },
        )

        accumulation_count = 0
        optimizer_steps = 0
        for _ in range(1_500):
            accumulation_count += 1
            if accumulation_count == 16:
                optimizer_steps += 1
                accumulation_count = 0
        self.assertEqual(accumulation_count, 12)
        self.assertEqual(optimizer_steps, 93)
        for _ in range(4):
            accumulation_count += 1
            if accumulation_count == 16:
                optimizer_steps += 1
                accumulation_count = 0
        self.assertEqual(accumulation_count, 0)
        self.assertEqual(optimizer_steps, 94)

    def test_scheduler_uses_requested_optimizer_update_indices(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0e-2)
        scheduler = build_training_scheduler(
            optimizer,
            SchedulerConfig(
                enabled=True,
                interval="optimizer_step",
                milestones=(20_480, 30_720),
                gamma=0.33,
            ),
        )
        observed = {}
        requested = {20_479, 20_480, 30_719, 30_720}
        for update_index in range(30_721):
            if update_index in requested:
                observed[update_index] = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
        expected = {
            20_479: 1.0e-2,
            20_480: 3.3e-3,
            30_719: 3.3e-3,
            30_720: 1.089e-3,
        }
        for update_index, expected_lr in expected.items():
            self.assertAlmostEqual(
                observed[update_index],
                expected_lr,
                places=12,
            )

    def test_checkpoint_saves_counters_only_at_clean_boundary(self):
        model = torch.nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters())
        dataset = SimpleNamespace(target_names=lambda: ("GT",))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                dataset=dataset,
                epoch=600,
                config_hash="hash",
                global_data_step=900_000,
                global_optimizer_step=56_250,
                gradient_accumulation_count=0,
                path=checkpoint,
            )
            payload = read_checkpoint_payload(checkpoint)
            self.assertEqual(payload["epoch"], 600)
            self.assertEqual(payload["global_data_step"], 900_000)
            self.assertEqual(payload["global_optimizer_step"], 56_250)
            self.assertEqual(payload["gradient_accumulation_count"], 0)
            with self.assertRaisesRegex(
                ValueError,
                "completed optimizer step",
            ):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    dataset=dataset,
                    epoch=1,
                    config_hash="hash",
                    gradient_accumulation_count=12,
                    path=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
