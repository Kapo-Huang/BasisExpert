from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from var_expert_inr.rmdsrn.config import load_config
from var_expert_inr.rmdsrn.data import TemporalFrameSampler, TemporalVolume, sample_voxel_batch
from var_expert_inr.rmdsrn.losses import (
    exponential_variance_weight,
    rmdsrn_loss,
    variance_regularization_loss,
)
from var_expert_inr.rmdsrn.model import RMDSRN
from var_expert_inr.rmdsrn.runner import run_evaluate, run_train


class RMDSRNTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _config(self, data_path: Path, *, run_after_training: bool = True) -> Path:
        payload = {
            "experiment": "rmdsrn-smoke-{target}",
            "exp_id": "rmdsrn-smoke-{target}",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "dataset_name": "ionization",
                "target": "GT",
                "targets": {"GT": str(data_path), "PD": str(data_path)},
                "volume_shape": {"T": 2, "Z": 2, "Y": 2, "X": 3},
            },
            "model": {
                "name": "rmdsrn",
                "base_encoder": "temporal_fv_srn",
                "grid_resolution": 2,
                "grid_channels": 2,
                "grid_init_std": 0.01,
                "keyframe_indices": [0, 1],
                "fourier_features": 3,
                "fourier_mode": "nerf",
                "decoder_count": 3,
                "decoder_hidden_features": 8,
                "decoder_hidden_layers": 2,
                "activation": "snake_alt",
                "activation_frequency": 1.0,
            },
            "training": {
                "steps": 2,
                "batch_size": 16,
                "lr": 0.005,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "min_lr": 1.0e-7,
                "lambda_min": 0.0,
                "lambda_max": 1.0,
                "lambda_growth_rate": 5.0,
                "epsilon": 1.0e-12,
                "save_every": 1,
                "log_every": 1,
                "seed": 7,
                "device": "cpu",
            },
            "evaluation": {
                "batch_size": 8,
                "save_mean": True,
                "save_variance": True,
                "run_after_training": run_after_training,
                "default_model": "artifact",
                "uncertainty_sample_size": 20,
                "topk_fractions": [0.1, 0.25],
                "seed": 9,
            },
        }
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _model_config() -> dict:
        return {
            "grid_resolution": 2,
            "grid_channels": 1,
            "grid_init_std": 0.01,
            "keyframe_indices": [0, 1],
            "fourier_features": 3,
            "decoder_count": 3,
            "decoder_hidden_features": 4,
            "decoder_hidden_layers": 2,
            "activation_frequency": 1.0,
        }

    def test_config_defaults_override_and_validation(self):
        data_path = self.root / "data.npy"
        np.save(data_path, np.zeros((2, 2, 2, 3), dtype=np.float32))
        config_path = self._config(data_path)
        cfg = load_config(config_path, target_override="PD")
        self.assertEqual(cfg["data"]["target"], "PD")
        self.assertEqual(cfg["model"]["decoder_count"], 3)
        self.assertEqual(cfg["training"]["steps"], 2)
        self.assertNotIn("lr_schedule_steps", cfg["training"])
        self.assertNotIn("lambda_schedule_steps", cfg["training"])
        self.assertNotIn("grad_clip_norm", cfg["training"])

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["training"].update(
            {
                "lr_schedule_steps": 20,
                "lambda_schedule_steps": 20,
                "grad_clip_norm": 1.0,
            }
        )
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        explicit = load_config(config_path)
        self.assertEqual(explicit["training"]["lr_schedule_steps"], 20)
        self.assertEqual(explicit["training"]["lambda_schedule_steps"], 20)
        self.assertEqual(explicit["training"]["grad_clip_norm"], 1.0)

        payload["model"]["decoder_count"] = 1
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "decoder_count"):
            load_config(config_path)

        payload["model"]["decoder_count"] = 3
        payload["training"]["unknown"] = True
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown training"):
            load_config(config_path)

        payload["training"].pop("unknown")
        for field, value in (
            ("grad_clip_norm", -1.0),
            ("lr_schedule_steps", 1),
            ("lambda_schedule_steps", 1),
        ):
            invalid = yaml.safe_load(yaml.safe_dump(payload))
            invalid["training"][field] = value
            config_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, field):
                load_config(config_path)

    def test_model_interpolation_members_and_statistics(self):
        model = RMDSRN(self._model_config())
        with torch.no_grad():
            model.encoder.feature_grids[0].fill_(0.0)
            model.encoder.feature_grids[1].fill_(1.0)
        coords = torch.rand(5, 3)
        features = model.encoder.grid_features(coords, 0.5)
        torch.testing.assert_close(features, torch.full_like(features, 0.5))
        members = model.forward_members(coords, 0)
        self.assertEqual(tuple(members.shape), (5, 3, 1))
        mean, variance = model(coords, 0)
        torch.testing.assert_close(mean, members.mean(dim=1))
        torch.testing.assert_close(variance, members.var(dim=1, unbiased=True))
        self.assertIsNot(
            next(model.decoders[0].parameters()),
            next(model.decoders[1].parameters()),
        )

    def test_loss_detach_stability_and_weight_schedule(self):
        members = torch.tensor(
            [[[0.0], [1.0], [2.0]], [[1.0], [1.5], [3.0]]],
            requires_grad=True,
        )
        target = torch.tensor([[0.5], [2.0]])
        output = rmdsrn_loss(
            members,
            target,
            variance_weight=2.0,
            epsilon=1.0e-12,
        )
        expected_member = ((members - target.unsqueeze(1)) ** 2).mean()
        torch.testing.assert_close(output.member, expected_member)
        self.assertFalse(output.error_density.requires_grad)
        self.assertTrue(output.variance_density.requires_grad)
        output.total.backward()
        self.assertIsNotNone(members.grad)

        zero = torch.zeros((4, 1), requires_grad=True)
        zero_target = torch.zeros((4, 1))
        kl, error_density, variance_density = variance_regularization_loss(
            zero,
            zero.square(),
            zero_target,
        )
        self.assertTrue(torch.isfinite(kl))
        torch.testing.assert_close(error_density, torch.full((4,), 0.25))
        torch.testing.assert_close(variance_density, torch.full((4,), 0.25))

        weights = [
            exponential_variance_weight(
                step,
                10,
                minimum=0.0,
                maximum=10.0,
                growth_rate=500.0,
            )
            for step in range(1, 11)
        ]
        self.assertEqual(weights, sorted(weights))
        self.assertAlmostEqual(weights[-1], 10.0)
        self.assertLess(weights[0], 0.1)

    def test_voxel_sampling_and_temporal_sampler_restore(self):
        values = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 2, 2, 3)
        data_path = self.root / "data.npy"
        np.save(data_path, values)
        volume = TemporalVolume(data_path, {"T": 2, "Z": 2, "Y": 2, "X": 3})
        rng = np.random.default_rng(3)
        coords, targets = sample_voxel_batch(volume, timestep=1, count=12, rng=rng)
        self.assertEqual(coords.shape, (12, 3))
        self.assertEqual(targets.shape, (12, 1))
        self.assertTrue(np.all((coords >= 0.0) & (coords <= 1.0)))

        sampler = TemporalFrameSampler.create(2, np.random.default_rng(4))
        first = sampler.next()
        restored = TemporalFrameSampler.from_state_dict(sampler.state_dict())
        self.assertNotEqual(first, restored.next())
        self.assertEqual(sampler.next(), restored.order[restored.cursor - 1])

    def test_train_artifact_predict_evaluate_and_resume(self):
        z, y, x = np.meshgrid(
            np.linspace(-1, 1, 2),
            np.linspace(-1, 1, 2),
            np.linspace(-1, 1, 3),
            indexing="ij",
        )
        values = np.stack([(x + y + z) / 3.0, (x - y + z) / 3.0]).astype(np.float32)
        data_path = self.root / "data.npy"
        np.save(data_path, values)
        config_path = self._config(data_path, run_after_training=True)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["training"].update(
            {
                "lr_schedule_steps": 20,
                "lambda_schedule_steps": 20,
                "grad_clip_norm": 1.0,
            }
        )
        payload["exploration_probe"] = {
            "enabled": True,
            "total_epoch_equivalents": 50,
            "every_epoch_equivalents": 5,
            "sample_ratio": 1.0,
            "max_samples": 100,
            "seed": 42,
            "retain_best_checkpoint": True,
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        original_clip = torch.nn.utils.clip_grad_norm_
        with mock.patch(
            "torch.nn.utils.clip_grad_norm_", wraps=original_clip
        ) as clip:
            trained = run_train(config_path, target="GT")
        self.assertEqual(clip.call_count, 2)
        self.assertTrue(Path(trained["checkpoint_path"]).exists())
        self.assertTrue(Path(trained["best_probe_checkpoint_path"]).exists())
        self.assertNotEqual(
            Path(trained["checkpoint_path"]),
            Path(trained["best_probe_checkpoint_path"]),
        )
        self.assertTrue(Path(trained["artifact_path"]).exists())
        self.assertEqual(trained["lr_schedule_steps"], 20)
        self.assertEqual(trained["lambda_schedule_steps"], 20)
        self.assertAlmostEqual(
            trained["final_lambda"],
            exponential_variance_weight(
                2,
                20,
                minimum=0.0,
                maximum=1.0,
                growth_rate=5.0,
            ),
        )
        mean = np.load(trained["mean_prediction_path"], mmap_mode="r")
        variance = np.load(trained["variance_prediction_path"], mmap_mode="r")
        self.assertEqual(mean.shape, values.shape)
        self.assertEqual(variance.shape, values.shape)
        self.assertTrue(np.all(variance >= 0.0))
        aggregate = trained["metrics"]["aggregate"]
        self.assertIn("variance_error_pearson", aggregate)
        self.assertEqual(set(aggregate["topk_hit_rate"]), {"0.1", "0.25"})
        probe_metrics = Path(trained["checkpoint_path"]).parent.parent / "metrics" / "exploration_psnr.tsv"
        rows = probe_metrics.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)
        details = yaml.safe_load(rows[-1].split("\t")[-1])
        self.assertEqual(
            set(details),
            {
                "member_mse",
                "variance_kl",
                "variance_weight",
                "weighted_variance_loss",
                "weighted_variance_to_member_ratio",
                "variance_error_pearson",
                "topk_hit_rate",
            },
        )
        del mean
        del variance

        evaluated = run_evaluate(
            config_path,
            target="GT",
            checkpoint=trained["checkpoint_path"],
        )
        self.assertTrue(Path(evaluated["metrics_path"]).exists())

        intermediate = Path(trained["checkpoint_path"]).parent / "step_000001.pth"
        resumed = run_train(config_path, target="GT", resume=intermediate)
        self.assertTrue(Path(resumed["checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
