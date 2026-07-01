import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.dc_inr.checkpoint import load_dc_checkpoint
from var_expert_inr.dc_inr.data import (
    BlockShape,
    DCTargetVolume,
    block_grid_shape_for_volume,
    block_id_to_grid_indices,
    sample_balanced_block_training_batch,
)
from var_expert_inr.dc_inr.runner import run_evaluate, run_predict, run_train
from var_expert_inr.dc_inr.search import (
    allocate_widths_for_entropies,
    cluster_representatives,
    compute_spatiotemporal_distance_matrix,
)


class DCINRTrainingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _write_volume(self, *, name: str = "target") -> Path:
        volume = np.array(
            [
                [[[ -1.0, -0.5], [0.0, 0.5]]],
                [[[ -0.8, -0.3], [0.2, 0.7]]],
            ],
            dtype=np.float32,
        )
        path = self.root / f"{name}.npy"
        np.save(path, volume)
        return path

    def _base_config(self, *, exp_id: str = "dc-smoke", use_targets: bool = False, target_name: str = "GT"):
        target_path = self._write_volume(name="shared")
        data = {
            "kind": "volume",
            "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 2},
        }
        if use_targets:
            data["target"] = target_name
            data["targets"] = {"GT": str(target_path), "ALT": str(target_path)}
        else:
            data["target_path"] = str(target_path)
        return {
            "experiment": exp_id,
            "exp_id": exp_id,
            "experiment_root": str(self.root / "runs"),
            "data": data,
            "model": {"name": "dc_inr"},
            "partition": {
                "candidate_block_shapes": [
                    {"sx": 2, "sy": 1, "sz": 1},
                    {"sx": 1, "sy": 2, "sz": 1},
                ],
                "distance_matrix_max_bytes": 1024 * 1024,
            },
            "compression": {
                "target_cr": 0.01,
                "max_initial_neurons": 8,
            },
            "training": {
                "epochs": 2,
                "lr": 1.0e-4,
                "points_per_timestep": 4,
                "prediction_batch_size": 8,
                "lr_milestones": [1],
                "lr_gamma": 0.5,
                "log_every": 0,
                "seed": 0,
                "device": "cpu",
            },
            "evaluation": {"batch_size": 8},
            "log": {
                "effective_config": False,
                "model_stats": False,
                "epoch_summary": False,
                "startup_timing": False,
                "psnr": {"enabled": True, "per_target": True},
                "timing": {
                    "enabled": False,
                    "epoch_breakdown": False,
                    "step_window": False,
                    "step_window_every_steps": 100,
                    "cuda_sync": False,
                },
            },
        }

    def test_block_partition_roundtrip_is_lossless(self):
        target_path = self._write_volume()
        volume = DCTargetVolume(
            target_path=target_path,
            target_name="GT",
            volume_shape=type("Shape", (), {"X": 2, "Y": 2, "Z": 1, "T": 2, "N": 8})(),
        )
        block_shape = BlockShape(sx=1, sy=2, sz=1)
        grid_shape = block_grid_shape_for_volume(volume.volume_shape, block_shape)
        blocks = volume.block_view(block_shape)
        reconstructed = np.zeros_like(volume.array_tzyx(), dtype=np.float32)
        for block_id in range(int(grid_shape.n_blocks)):
            bx, by, bz = block_id_to_grid_indices(block_id, grid_shape)
            x0 = bx * int(block_shape.sx)
            y0 = by * int(block_shape.sy)
            z0 = bz * int(block_shape.sz)
            reconstructed[:, z0 : z0 + block_shape.sz, y0 : y0 + block_shape.sy, x0 : x0 + block_shape.sx] = blocks[block_id]
        np.testing.assert_allclose(reconstructed, np.asarray(volume.array_tzyx(), dtype=np.float32))

    def test_balanced_sampler_returns_exact_batch_across_timesteps(self):
        block_values = np.arange(24, dtype=np.float32).reshape(3, 1, 2, 4)
        coords, targets = sample_balanced_block_training_batch(
            block_values=block_values,
            block_shape=BlockShape(sx=4, sy=2, sz=1),
            batch_size=17,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(coords.shape, (17, 4))
        self.assertEqual(targets.shape, (17, 1))
        time_values, counts = np.unique(coords[:, 3], return_counts=True)
        self.assertEqual(len(time_values), 3)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_dbscan_medoid_and_zero_entropy_widths_are_deterministic(self):
        features = np.array(
            [
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        distance = compute_spatiotemporal_distance_matrix(features, max_bytes=1024)
        representatives, assignments = cluster_representatives(distance, eps=1.0e-6, min_samples=1)
        np.testing.assert_array_equal(representatives, np.array([0, 2], dtype=np.int32))
        np.testing.assert_array_equal(assignments, np.array([0, 0, 1], dtype=np.int32))

        widths = allocate_widths_for_entropies(
            np.zeros((3,), dtype=np.float32),
            max_initial_neurons=9,
            min_initial_neurons=4,
        )
        np.testing.assert_array_equal(widths, np.array([4, 4, 4], dtype=np.int32))

    def test_train_predict_evaluate_and_checkpoint_reload(self):
        config_path = self._write_yaml(self.root / "dc_smoke.yaml", self._base_config())
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        self.assertTrue(Path(train_result["prediction_path"]).exists())
        self.assertTrue(Path(train_result["metrics_path"]).exists())

        checkpoint = load_dc_checkpoint(train_result["checkpoint_path"])
        self.assertLessEqual(
            int(np.asarray(checkpoint["representative_block_ids"]).size),
            int(np.asarray(checkpoint["block_to_representative"]).size),
        )

        prediction = np.load(train_result["prediction_path"])
        self.assertEqual(prediction.shape, (2, 1, 2, 2))
        self.assertIn("payload_cr", train_result["metrics"]["aggregate"])
        self.assertIn("checkpoint_bytes", train_result["metrics"]["aggregate"])

        predict_result = run_predict(config_path)
        self.assertTrue(Path(predict_result["prediction_path"]).exists())
        evaluate_result = run_evaluate(config_path)
        self.assertTrue(Path(evaluate_result["metrics_path"]).exists())
        self.assertIn("checkpoint_cr", evaluate_result["metrics"]["aggregate"])

    def test_checkpoint_target_mismatch_is_rejected(self):
        train_config = self._base_config(exp_id="dc-target-mismatch", use_targets=True, target_name="GT")
        train_path = self._write_yaml(self.root / "train.yaml", train_config)
        train_result = run_train(train_path)

        mismatch_config = self._base_config(exp_id="dc-target-mismatch", use_targets=True, target_name="ALT")
        mismatch_path = self._write_yaml(self.root / "mismatch.yaml", mismatch_config)
        with self.assertRaisesRegex(ValueError, "checkpoint target mismatch"):
            run_predict(mismatch_path, checkpoint_path=train_result["checkpoint_path"])


if __name__ == "__main__":
    unittest.main()
