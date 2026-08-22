import tempfile
import unittest
from pathlib import Path

import yaml

from var_expert_inr.methods.mc_inr.config import load_config


class MCINRConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _base_config(self):
        return {
            "experiment": "mc-node",
            "exp_id": "mc-node",
            "experiment_root": "runs",
            "data": {
                "kind": "node",
                "coords_path": "./coords.npy",
                "targets": {"a": "./a.npy", "b": "./b.npy"},
            },
            "model": {"name": "mc_inr"},
            "training": {
                "batch_size": 4,
                "pred_batch_size": 4,
                "device": "cpu",
                "initial_k": 2,
                "finetune_epochs": 2,
                "finetune_sampling_ratio": 1.0,
            },
        }

    def test_loads_new_defaults(self):
        loaded = load_config(self._write_yaml(self.root / "default.yaml", self._base_config()))
        self.assertEqual(loaded.training.meta_iterations, 2000)
        self.assertEqual(loaded.training.meta_inner_steps, 5)
        self.assertEqual(loaded.training.meta_inner_batch_size, 8192)
        self.assertAlmostEqual(loaded.training.meta_inner_lr, 1.0e-4)
        self.assertEqual(loaded.training.meta_batch_clusters, 4)
        self.assertEqual(loaded.training.meta_support_max_rows, 32768)
        self.assertAlmostEqual(loaded.training.meta_outer_lr, 5.0e-5)
        self.assertTrue(loaded.training.cluster_aware_batches)

    def test_meta_iterations_and_outer_lr_fallback(self):
        config = self._base_config()
        config["training"].update(
            {
                "epochs": 7,
                "lr": 1.0e-3,
            }
        )
        loaded = load_config(self._write_yaml(self.root / "legacy.yaml", config))
        self.assertEqual(loaded.training.epochs, 7)
        self.assertEqual(loaded.training.meta_iterations, 7)
        self.assertAlmostEqual(loaded.training.meta_outer_lr, 1.0e-3)

    def test_explicit_meta_fields_override_fallbacks(self):
        config = self._base_config()
        config["training"].update(
            {
                "epochs": 9,
                "meta_iterations": 3,
                "meta_inner_batch_size": 1024,
                "meta_support_max_rows": 4096,
                "lr": 1.0e-3,
                "meta_outer_lr": 2.0e-3,
            }
        )
        loaded = load_config(self._write_yaml(self.root / "explicit.yaml", config))
        self.assertEqual(loaded.training.epochs, 9)
        self.assertEqual(loaded.training.meta_iterations, 3)
        self.assertEqual(loaded.training.meta_inner_batch_size, 1024)
        self.assertEqual(loaded.training.meta_support_max_rows, 4096)
        self.assertAlmostEqual(loaded.training.meta_outer_lr, 2.0e-3)

    def test_rejects_non_positive_meta_inner_batch_size(self):
        config = self._base_config()
        config["training"]["meta_inner_batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "meta_inner_batch_size must be positive"):
            load_config(self._write_yaml(self.root / "bad_inner_batch.yaml", config))

    def test_rejects_non_positive_meta_support_max_rows(self):
        config = self._base_config()
        config["training"]["meta_support_max_rows"] = 0
        with self.assertRaisesRegex(ValueError, "meta_support_max_rows must be positive"):
            load_config(self._write_yaml(self.root / "bad_support_rows.yaml", config))

    def test_rejects_unknown_training_keys(self):
        config = self._base_config()
        config["training"]["unknown_flag"] = True
        with self.assertRaisesRegex(ValueError, "Unknown training keys"):
            load_config(self._write_yaml(self.root / "bad.yaml", config))

    def test_rejects_legacy_meta_support_ratio(self):
        config = self._base_config()
        config["training"]["meta_support_ratio"] = 0.3
        with self.assertRaisesRegex(ValueError, "Unknown training keys"):
            load_config(self._write_yaml(self.root / "legacy_support_ratio.yaml", config))

    def test_rejects_legacy_meta_sampling_ratio(self):
        config = self._base_config()
        config["training"]["meta_sampling_ratio"] = 0.3
        with self.assertRaisesRegex(ValueError, "Unknown training keys"):
            load_config(self._write_yaml(self.root / "legacy_sampling_ratio.yaml", config))

    def test_rejects_data_target_selector(self):
        config = self._base_config()
        config["data"]["target"] = "a"
        with self.assertRaisesRegex(ValueError, "does not support data.target"):
            load_config(self._write_yaml(self.root / "target.yaml", config))


if __name__ == "__main__":
    unittest.main()
