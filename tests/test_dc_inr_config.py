import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.dc_inr.config import load_config
from var_expert_inr.dc_inr.runner import run_train


class DCINRConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.target_path = self.root / "target.npy"
        np.save(self.target_path, np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _base_config(self):
        return {
            "experiment": "dc-config",
            "exp_id": "dc-config",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target_path": str(self.target_path),
                "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 2},
            },
            "model": {"name": "dc_inr"},
            "partition": {
                "candidate_block_shapes": [
                    {"sx": 2, "sy": 1, "sz": 1},
                    {"sx": 1, "sy": 2, "sz": 1},
                ]
            },
            "compression": {
                "target_cr": 0.01,
                "max_initial_neurons": 8,
            },
            "training": {
                "epochs": 1,
                "lr": 1.0e-4,
                "points_per_timestep": 4,
                "prediction_batch_size": 8,
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

    def test_rejects_non_volume_kind(self):
        config = self._base_config()
        config["data"]["kind"] = "node"
        with self.assertRaisesRegex(ValueError, "only supports data.kind='volume'"):
            load_config(self._write_yaml(self.root / "bad_kind.yaml", config))

    def test_rejects_targets_without_selected_target(self):
        config = self._base_config()
        config["data"].pop("target_path")
        config["data"]["targets"] = {"GT": str(self.target_path)}
        with self.assertRaisesRegex(ValueError, "requires data.target"):
            load_config(self._write_yaml(self.root / "missing_target.yaml", config))

    def test_rejects_candidate_shapes_with_different_voxel_counts(self):
        config = self._base_config()
        config["partition"]["candidate_block_shapes"] = [
            {"sx": 2, "sy": 1, "sz": 1},
            {"sx": 2, "sy": 2, "sz": 1},
        ]
        with self.assertRaisesRegex(ValueError, "same voxel count"):
            load_config(self._write_yaml(self.root / "bad_candidates.yaml", config))

    def test_requires_one_size_constraint_and_max_initial_neurons(self):
        config = self._base_config()
        config["compression"].pop("target_cr")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_config(self._write_yaml(self.root / "missing_cr.yaml", config))

        config = self._base_config()
        config["compression"]["target_size_mib"] = 1.0
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_config(self._write_yaml(self.root / "two_constraints.yaml", config))

        config = self._base_config()
        config["compression"].pop("max_initial_neurons")
        with self.assertRaisesRegex(ValueError, "max_initial_neurons is required"):
            load_config(self._write_yaml(self.root / "missing_m.yaml", config))

    def test_run_train_rejects_indivisible_block_shape(self):
        config = self._base_config()
        config["partition"]["candidate_block_shapes"] = [{"sx": 3, "sy": 1, "sz": 1}]
        with self.assertRaisesRegex(ValueError, "must divide volume X"):
            run_train(self._write_yaml(self.root / "indivisible.yaml", config))


if __name__ == "__main__":
    unittest.main()
