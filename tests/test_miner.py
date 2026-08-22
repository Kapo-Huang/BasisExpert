from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from var_expert_inr.evaluation.service import evaluate_run
from var_expert_inr.miner.blocks import (
    blockify,
    crop_padding,
    effective_scale_count,
    local_coordinate_grid,
    pad_to_scale_compatible,
    unblockify,
)
from var_expert_inr.miner.config import load_config
from var_expert_inr.miner.model import BlockSiren, propagate_state_to_finer_grid
from var_expert_inr.miner.runner import decode_checkpoint, run_train


class MinerCoreTestCase(unittest.TestCase):
    def test_block_roundtrip_and_local_coordinates_for_2d_and_3d(self):
        for shape, block_size in (((4, 6), 2), ((4, 6, 8), 2)):
            values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            blocks, grid_shape = blockify(values, block_size)
            restored = unblockify(blocks, grid_shape, block_size)
            torch.testing.assert_close(restored, values)
            coordinates = local_coordinate_grid(block_size, len(shape))
            self.assertEqual(tuple(coordinates.shape), (block_size ** len(shape), len(shape)))
            self.assertEqual(float(coordinates.min()), -1.0)
            self.assertEqual(float(coordinates.max()), 1.0)

    def test_scale_policy_and_reflect_padding(self):
        self.assertEqual(
            effective_scale_count((128, 128), block_size=32, requested_scales=4),
            3,
        )
        values = np.zeros((248, 248, 600), dtype=np.float32)
        padded, padding = pad_to_scale_compatible(values, block_size=16, scales=4)
        self.assertEqual(padded.shape, (256, 256, 640))
        self.assertEqual(padding, ((0, 8), (0, 8), (0, 40)))
        self.assertEqual(crop_padding(padded, values.shape).shape, values.shape)

    def test_siren_state_propagation_repeats_parent_channels(self):
        torch.manual_seed(3)
        model = BlockSiren(
            channels=4,
            in_features=2,
            hidden_features=3,
            hidden_layers=1,
            omega_0=30.0,
        )
        state = model.state_dict()
        fine = propagate_state_to_finer_grid(state, (2, 2))
        self.assertEqual(next(iter(fine.values())).shape[0], 16)
        divisor = 2.0
        torch.testing.assert_close(fine["layers.0.weight"][0], state["layers.0.weight"][0] / divisor)
        torch.testing.assert_close(fine["layers.0.weight"][5], state["layers.0.weight"][0] / divisor)


class MinerEndToEndTestCase(unittest.TestCase):
    def _payload(self, root: Path, values: np.ndarray, dimensions: int) -> Path:
        target_path = root / "target.npy"
        np.save(target_path, values)
        _, z_size, y_size, x_size = values.shape
        payload = {
            "experiment": "miner_smoke",
            "exp_id": f"miner-smoke-{dimensions}d",
            "experiment_root": str(root / "runs"),
            "data": {
                "kind": "volume",
                "dataset_name": "smoke",
                "target": "field",
                "target_path": str(target_path),
                "volume_shape": {"T": 1, "Z": z_size, "Y": y_size, "X": x_size},
            },
            "model": {
                "name": "miner",
                "scales": 2,
                "block_size": 2,
                "hidden_features": 4,
                "hidden_layers": 0,
                "omega_0": 10.0,
                "carry_start_scale": 2,
                "coarse_feature_multiplier": 2,
            },
            "training": {
                "epochs_per_scale": 1,
                "lr": 1.0e-3,
                "block_mse_threshold": 0.0,
                "scale_convergence_delta": 0.0,
                "global_mse_threshold": 0.0,
                "max_active_blocks_per_step": 2,
                "time_indices": "all",
                "seed": 7,
                "device": "cpu",
                "log_every": 1,
            },
            "evaluation": {"save_predictions": False, "run_after_training": False},
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return config_path

    def _run_case(self, dimensions: int) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            if dimensions == 2:
                y, x = np.meshgrid(
                    np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing="ij"
                )
                values = ((x + y) / 2.0)[None, None].astype(np.float32)
                expected_shape = (8, 8)
            else:
                z, y, x = np.meshgrid(
                    np.linspace(-1, 1, 4),
                    np.linspace(-1, 1, 4),
                    np.linspace(-1, 1, 4),
                    indexing="ij",
                )
                values = ((x + y + z) / 3.0)[None].astype(np.float32)
                expected_shape = (4, 4, 4)
            config_path = self._payload(root, values, dimensions)
            cfg = load_config(config_path)
            self.assertEqual(cfg["data"]["spatial_dimensions"], dimensions)
            result = run_train(config_path)
            run_dir = Path(result["run_dir"])
            checkpoint = run_dir / "timesteps" / "t0000" / "checkpoint.pth"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((checkpoint.parent / "scale_00_complete.pth").is_file())
            prediction = decode_checkpoint(checkpoint, device="cpu", batch_blocks=2)
            self.assertEqual(prediction.shape, expected_shape)
            self.assertTrue(np.isfinite(prediction).all())

            resumed = run_train(config_path, resume=run_dir)
            self.assertEqual(resumed["completed_timesteps"], [])
            self.assertEqual(resumed["skipped_timesteps"], [0])

            evaluated = evaluate_run(
                run_dir,
                metrics="psnr",
                timesteps="0",
                targets="field",
                device="cpu",
            )
            self.assertEqual(evaluated["metrics"]["status"], "complete")
            self.assertEqual(evaluated["metrics"]["per_timestep"][0]["timestep"], 0)

    def test_tiny_2d_train_checkpoint_resume_and_evaluate(self):
        self._run_case(2)

    def test_tiny_3d_train_checkpoint_resume_and_evaluate(self):
        self._run_case(3)


if __name__ == "__main__":
    unittest.main()
