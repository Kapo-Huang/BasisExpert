from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_evaluate, run_predict, run_train
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model, materialize_model_config
from var_expert_inr.models.sota.fa_tr_inr import (
    FactorMLP,
    FrequencyAwareTRINR,
    SineLinear,
)


class FrequencyAwareTRINRTestCase(unittest.TestCase):
    def test_factor_shapes_structure_and_initialization(self):
        model = FrequencyAwareTRINR()
        coords = torch.linspace(-1.0, 1.0, 3).reshape(-1, 1)
        expected_shapes = {
            "factor_x": (3, 22, 88),
            "factor_y": (3, 88, 3),
            "factor_f": (3, 3, 3),
            "factor_z": (3, 3, 5),
            "factor_t": (3, 5, 22),
        }
        for name, shape in expected_shapes.items():
            factor = getattr(model, name)
            self.assertIsInstance(factor, FactorMLP)
            self.assertEqual(tuple(factor(coords).shape), shape)
            self.assertEqual(len(factor.net), 4)
            self.assertIsInstance(factor.net[-1], torch.nn.Linear)

            first = factor.net[0]
            self.assertIsInstance(first, SineLinear)
            self.assertLessEqual(
                float(first.linear.weight.abs().max()),
                1.0 + 1.0e-7,
            )
            hidden_bound = math.sqrt(5.0 / 128.0) / 19.0
            for layer in factor.net[1:3]:
                self.assertIsInstance(layer, SineLinear)
                self.assertLessEqual(
                    float(layer.linear.weight.abs().max()),
                    hidden_bound + 1.0e-7,
                )

        parameter_ids = [
            {id(parameter) for parameter in getattr(model, name).parameters()}
            for name in expected_shapes
        ]
        for index, current in enumerate(parameter_ids):
            for other in parameter_ids[index + 1 :]:
                self.assertTrue(current.isdisjoint(other))

    def test_frequency_buffer_contraction_gradients_and_parameter_count(self):
        torch.manual_seed(3)
        model = FrequencyAwareTRINR()
        self.assertEqual(
            tuple(model.frequency_coordinates.reshape(-1).tolist()),
            (1.0, 2.0, 3.0),
        )
        self.assertFalse(model.frequency_coordinates.requires_grad)
        self.assertNotIn(
            "frequency_coordinates",
            dict(model.named_parameters()),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            467_502,
        )

        coords = torch.rand(2, 4) * 2.0 - 1.0
        components = model.frequency_components(coords)
        gx = model.factor_x(coords[:, 0:1])
        gy = model.factor_y(coords[:, 1:2])
        gf = model.factor_f(model.frequency_coordinates)
        gz = model.factor_z(coords[:, 2:3])
        gt = model.factor_t(coords[:, 3:4])
        reference = []
        for frequency in range(3):
            matrix = gx @ gy
            matrix = matrix @ gf[frequency]
            matrix = matrix @ gz
            matrix = matrix @ gt
            reference.append(matrix.diagonal(dim1=-2, dim2=-1).sum(-1))
        torch.testing.assert_close(
            components,
            torch.stack(reference, dim=-1),
        )
        self.assertEqual(tuple(model(coords).shape), (2, 1))
        model(coords).sum().backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.factor_f.parameters()
            ),
            0.0,
        )

    def test_registry_constraints_and_effective_config(self):
        volume_meta = DatasetMeta(
            kind="volume",
            n_samples=8,
            input_dim=4,
            target_names=("GT",),
            target_dims={"GT": 1},
        )
        config = ModelConfig(name="fa_tr_inr", params={})
        effective = materialize_model_config(config, volume_meta)
        self.assertEqual(
            effective["frequency_coordinates"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            effective["tensor_ring_ranks"],
            [22, 88, 3, 3, 5],
        )
        model = build_model(config, volume_meta)
        self.assertIsInstance(model.backbone, FrequencyAwareTRINR)

        node_meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("GT",),
            target_dims={"GT": 1},
        )
        with self.assertRaisesRegex(ValueError, "volume"):
            build_model(config, node_meta)
        multi_meta = DatasetMeta(
            kind="volume",
            n_samples=8,
            input_dim=4,
            target_names=("GT", "PD"),
            target_dims={"GT": 1, "PD": 1},
        )
        with self.assertRaisesRegex(ValueError, "single-target"):
            build_model(config, multi_meta)
        with self.assertRaisesRegex(ValueError, "Unknown fa_tr_inr"):
            build_model(
                ModelConfig(
                    name="fa_tr_inr",
                    params={"unknown": 1},
                ),
                volume_meta,
            )

    def test_unified_training_checkpoint_prediction_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            values = np.linspace(
                -1.0,
                1.0,
                8,
                dtype=np.float32,
            ).reshape(2, 1, 2, 2)
            target_path = root / "volume.npy"
            np.save(target_path, values)
            payload = {
                "experiment": "fa-tr-smoke",
                "exp_id": "fa-tr-smoke",
                "experiment_root": str(root / "runs"),
                "data": {
                    "kind": "volume",
                    "target_path": str(target_path),
                },
                "model": {"name": "fa_tr_inr"},
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "pred_batch_size": 4,
                    "num_workers": 0,
                    "lr": 1.0e-4,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1.0e-8,
                    "weight_decay": 0.0,
                    "device": "cpu",
                    "seed": 4,
                    "val_split": 0.0,
                    "log_every": 1,
                    "save_every": 0,
                    "sampler": "uniform_random",
                    "scheduler": {"enabled": False},
                },
                "evaluation": {
                    "batch_size": 4,
                    "save_predictions": False,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            trained = run_train(config_path)
            self.assertTrue(Path(trained["checkpoint_path"]).exists())
            self.assertNotIn("artifact_path", trained)
            predicted = run_predict(
                config_path,
                checkpoint_path=trained["checkpoint_path"],
            )
            prediction = next(iter(predicted["predictions"].values()))
            self.assertEqual(prediction.shape, (8, 1))
            evaluated = run_evaluate(
                config_path,
                checkpoint_path=trained["checkpoint_path"],
            )
            self.assertTrue(Path(evaluated["metrics_path"]).exists())


if __name__ == "__main__":
    unittest.main()
