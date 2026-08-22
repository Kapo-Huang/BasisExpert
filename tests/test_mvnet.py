import gc
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from var_expert_inr.cli import run_evaluate, run_predict
from var_expert_inr.config.io import load_experiment_config
from var_expert_inr.config.schema import (
    ModelConfig,
    SchedulerConfig,
    TrainingConfig,
)
from var_expert_inr.data import build_dataset
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model, materialize_model_config
from var_expert_inr.models.baselines.mvnet import (
    MVNet4D,
    ResidualSineBlock,
    SineLayer,
)
from var_expert_inr.training.engine import (
    build_training_scheduler,
    mvnet_joint_mse,
    split_multitarget_prediction,
    training_budget,
    validate_mvnet_training_config,
)
from var_expert_inr.utils.checkpoint import save_checkpoint


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=300,
        batch_size=2048,
        pred_batch_size=16000,
        gradient_accumulation_steps=1,
        num_workers=0,
        lr=1.0e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1.0e-8,
        weight_decay=0.0,
        val_split=0.0,
        early_stop_patience=0,
        loss_type="mse",
        sampler="budgeted_random",
        batches_per_epoch_budget=1500,
        scheduler=SchedulerConfig(
            enabled=True,
            interval="epoch",
            step_size=15,
            gamma=0.8,
        ),
    )


class MVNetModelTestCase(unittest.TestCase):
    def test_sine_layer_explicitly_multiplies_by_fixed_omega(self):
        layer = SineLayer(
            2,
            1,
            bias=True,
            is_first=True,
            omega_0=30.0,
        )
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[0.25, -0.5]]))
            layer.linear.bias.copy_(torch.tensor([0.1]))
        inputs = torch.tensor([[0.2, -0.4], [-0.1, 0.3]])
        expected = torch.sin(30.0 * layer.linear(inputs))
        torch.testing.assert_close(layer(inputs), expected)
        self.assertEqual(layer.omega_0, 30.0)
        self.assertFalse(
            any("omega" in name for name, _ in layer.named_parameters())
        )

    def test_residual_block_uses_half_sum_without_post_activation(self):
        block = ResidualSineBlock(features=120)
        block.layer1 = nn.Identity()
        block.layer2 = nn.Identity()
        inputs = torch.randn(7, 120)
        torch.testing.assert_close(block(inputs), inputs)

        class Zero(nn.Module):
            def forward(self, values):
                return torch.zeros_like(values)

        block.layer2 = Zero()
        torch.testing.assert_close(block(inputs), 0.5 * inputs)

    def test_architecture_has_twenty_one_sine_layers_and_no_forbidden_modules(self):
        model = MVNet4D(5)
        self.assertEqual(len(model.residual_blocks), 10)
        self.assertEqual(
            sum(isinstance(module, SineLayer) for module in model.modules()),
            21,
        )
        self.assertEqual(model.input_layer.linear.in_features, 4)
        self.assertEqual(model.input_layer.linear.out_features, 120)
        self.assertEqual(model.output_layer.in_features, 120)
        self.assertEqual(model.output_layer.out_features, 5)
        self.assertIsNotNone(model.output_layer.bias)
        forbidden = (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.Dropout,
            nn.Conv1d,
            nn.Embedding,
            nn.MultiheadAttention,
        )
        self.assertFalse(
            any(isinstance(module, forbidden) for module in model.modules())
        )

    def test_initialization_bounds_and_biases(self):
        torch.manual_seed(23)
        model = MVNet4D(5)
        input_bound = 1.0 / 4.0
        self.assertLessEqual(
            float(model.input_layer.linear.weight.abs().max()),
            input_bound,
        )
        hidden_bound = math.sqrt(6.0 / 120.0) / 30.0
        for block in model.residual_blocks:
            for layer in (block.layer1, block.layer2):
                self.assertLessEqual(
                    float(layer.linear.weight.abs().max()),
                    hidden_bound + 1.0e-8,
                )
                self.assertLessEqual(
                    float(layer.linear.bias.abs().max()),
                    1.0 / math.sqrt(120.0) + 1.0e-8,
                )
                self.assertGreater(
                    float(layer.linear.bias.abs().sum()),
                    0.0,
                )
        self.assertLessEqual(
            float(model.output_layer.weight.abs().max()),
            1.0 / 120.0 + 1.0e-8,
        )
        self.assertGreater(float(model.output_layer.bias.abs().sum()), 0.0)

    def test_forward_preserves_coordinate_values_and_is_unbounded_linear(self):
        model = MVNet4D(4)
        coords = torch.tensor(
            [
                [-1.0, -0.5, 0.0, 1.0],
                [1.0, 0.5, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        original = coords.clone()
        output = model(coords)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(coords, original)
        self.assertIsInstance(model.output_layer, nn.Linear)

    def test_parameter_count_matches_closed_form(self):
        for variables in (4, 5):
            model = MVNet4D(variables)
            actual = sum(parameter.numel() for parameter in model.parameters())
            expected = 291_000 + 121 * variables
            self.assertEqual(actual, expected)
            self.assertEqual(model.expected_parameter_count, expected)

    def test_fixed_architecture_rejects_changes(self):
        with self.assertRaisesRegex(ValueError, "fixed architecture"):
            MVNet4D(5, hidden_features=121)
        with self.assertRaisesRegex(ValueError, "at least two"):
            MVNet4D(1)


class MVNetIntegrationTestCase(unittest.TestCase):
    def _meta(
        self,
        *,
        input_dim=4,
        target_names=("a", "b", "c", "d"),
        target_dims=None,
    ):
        dimensions = target_dims or {
            name: 1 for name in target_names
        }
        return DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=input_dim,
            target_names=tuple(target_names),
            target_dims=dimensions,
        )

    def test_registry_derives_variable_count_and_rejects_invalid_metadata(self):
        cfg = ModelConfig(
            name="mvnet",
            params={
                "in_features": 4,
                "out_features": 4,
                "hidden_features": 120,
                "num_residual_blocks": 10,
                "omega_0": 30.0,
                "bias": True,
            },
        )
        materialized = materialize_model_config(cfg, self._meta())
        self.assertEqual(materialized["out_features"], 4)
        model = build_model(cfg, self._meta())
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            291_484,
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            materialize_model_config(
                ModelConfig(name="mvnet"),
                self._meta(target_names=("a",)),
            )
        with self.assertRaisesRegex(ValueError, "scalar target"):
            materialize_model_config(
                ModelConfig(name="mvnet"),
                self._meta(
                    target_names=("a", "b"),
                    target_dims={"a": 1, "b": 2},
                ),
            )
        with self.assertRaisesRegex(ValueError, "input_dim"):
            materialize_model_config(cfg, self._meta(input_dim=3))
        with self.assertRaisesRegex(ValueError, "hidden_features=120"):
            materialize_model_config(
                ModelConfig(
                    name="mvnet",
                    params={"hidden_features": 64},
                ),
                self._meta(),
            )

    def test_joint_mse_and_output_column_order(self):
        names = ("GT", "H2", "H_plus", "He", "PD")
        predictions = torch.randn(9, 5, requires_grad=True)
        targets = {
            name: torch.randn(9, 1)
            for name in names
        }
        expected_matrix = torch.cat(
            [targets[name] for name in names],
            dim=-1,
        )
        expected = torch.nn.functional.mse_loss(
            predictions,
            expected_matrix,
            reduction="mean",
        )
        actual = mvnet_joint_mse(predictions, targets, names)
        torch.testing.assert_close(actual, expected)

        split = split_multitarget_prediction(predictions, names)
        self.assertEqual(tuple(split), names)
        for index, name in enumerate(names):
            torch.testing.assert_close(
                split[name],
                predictions[:, index : index + 1],
            )

    def test_fixed_training_config_budget_and_scheduler_boundaries(self):
        cfg = _training_config()
        validate_mvnet_training_config(cfg)
        budget = training_budget(
            epochs=cfg.epochs,
            data_steps_per_epoch=cfg.batches_per_epoch_budget,
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        )
        self.assertEqual(
            budget,
            {
                "data_steps": 450_000,
                "samples": 921_600_000,
                "optimizer_steps": 450_000,
                "remainder": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "fixed reproduction"):
            validate_mvnet_training_config(
                replace(cfg, gradient_accumulation_steps=2)
            )

        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
        scheduler = build_training_scheduler(optimizer, cfg.scheduler)
        observed = {}
        for epoch in range(1, 32):
            observed[epoch] = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(observed[15], 1.0e-4, places=12)
        self.assertAlmostEqual(observed[16], 8.0e-5, places=12)
        self.assertAlmostEqual(observed[30], 8.0e-5, places=12)
        self.assertAlmostEqual(observed[31], 6.4e-5, places=12)

    def test_unified_checkpoint_predict_and_evaluate_preserve_all_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coords = np.linspace(
                -0.9,
                0.9,
                32,
                dtype=np.float32,
            ).reshape(8, 4)
            coords_path = root / "coords.npy"
            np.save(coords_path, coords)
            target_paths = {}
            for index, name in enumerate(("SALT", "TEMP", "U", "V")):
                values = np.tanh(
                    coords[:, :1] + np.float32(index * 0.1)
                ).astype(np.float32)
                path = root / f"{name}.npy"
                np.save(path, values)
                target_paths[name] = str(path)

            experiment_root = root / "runs"
            config_payload = {
                "experiment": "mvnet-test",
                "exp_id": "mvnet-test",
                "experiment_root": str(experiment_root),
                "data": {
                    "kind": "node",
                    "coords_path": str(coords_path),
                    "targets": target_paths,
                },
                "model": {
                    "name": "mvnet",
                    "in_features": 4,
                    "out_features": 4,
                    "hidden_features": 120,
                    "num_residual_blocks": 10,
                    "omega_0": 30.0,
                    "bias": True,
                },
                "training": {
                    "epochs": 300,
                    "batch_size": 2048,
                    "pred_batch_size": 8,
                    "gradient_accumulation_steps": 1,
                    "num_workers": 0,
                    "lr": 1.0e-4,
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1.0e-8,
                    "weight_decay": 0.0,
                    "val_split": 0.0,
                    "early_stop_patience": 0,
                    "loss_type": "mse",
                    "seed": 42,
                    "device": "cpu",
                    "sampler": "budgeted_random",
                    "batches_per_epoch_budget": 1500,
                    "scheduler": {
                        "enabled": True,
                        "interval": "epoch",
                        "step_size": 15,
                        "gamma": 0.8,
                    },
                    "pretrain": {"enabled": False},
                },
                "evaluation": {
                    "batch_size": 8,
                    "save_predictions": False,
                },
            }
            config_path = root / "mvnet.yaml"
            config_path.write_text(
                yaml.safe_dump(config_payload, sort_keys=False),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            dataset = build_dataset(
                config.data,
                model_name=config.model.name,
            )
            model = build_model(config.model, dataset.meta)
            run_dir = (
                experiment_root
                / config.exp_id
                / "20260727_120000_000000"
            )
            checkpoint_path = (
                run_dir / "checkpoints" / f"{config.exp_id}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=None,
                scheduler=None,
                dataset=dataset,
                epoch=300,
                config_hash="mvnet-test",
                path=checkpoint_path,
            )

            prediction_result = run_predict(
                config_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(
                tuple(prediction_result["predictions"]),
                dataset.target_names(),
            )
            self.assertTrue(
                all(
                    values.shape == (8, 1)
                    for values in prediction_result["predictions"].values()
                )
            )
            evaluation_result = run_evaluate(
                config_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertTrue(Path(evaluation_result["metrics_path"]).exists())
            self.assertEqual(
                set(evaluation_result["metrics"]["targets"]),
                set(dataset.target_names()),
            )
            del dataset, model, config
            gc.collect()


if __name__ == "__main__":
    unittest.main()
