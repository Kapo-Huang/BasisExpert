import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.io import load_experiment_config


class ConfigLoadingTestCase(unittest.TestCase):
    def test_repo_car_configs_load(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_paths = [
            repo_root / "configs" / "BasisExpert" / "car.yaml",
            repo_root / "configs" / "MoE-INR" / "car.yaml",
            repo_root / "configs" / "CoordNet" / "car.yaml",
            repo_root / "configs" / "SIREN" / "car.yaml",
        ]

        loaded_configs = [load_experiment_config(path) for path in config_paths]

        self.assertEqual(loaded_configs[0].exp_id, "light_basis_expert-car")
        self.assertEqual(loaded_configs[0].data.dataset_name, "car")
        self.assertIsNone(loaded_configs[0].data.target)
        for loaded in loaded_configs[1:]:
            self.assertEqual(loaded.data.dataset_name, "car")
            self.assertEqual(loaded.data.target, "CoefPressure")
            self.assertIn("CoefPressure", loaded.data.targets)
            self.assertIn("-car-CoefPressure", loaded.exp_id)

    def test_load_config_rejects_unknown_top_level_key(self):
        config = {
            "exp_id": "unknown-top-level",
            "experiment_root": "runs",
            "unexpected": True,
            "data": {
                "kind": "volume",
                "target_path": "./target.npy",
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(config_path)

    def test_load_config_rejects_unknown_evaluation_key(self):
        config = {
            "exp_id": "unknown-eval",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target_path": "./target.npy",
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
            "evaluation": {
                "batch_size": 32,
                "unused_flag": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(config_path)

    def test_load_config_supports_target_selector_and_resolves_placeholders(self):
        config = {
            "experiment": "selector-{target}",
            "exp_id": "siren-{target}",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target": "b",
                "targets": {
                    "a": "./target_a.npy",
                    "b": "./target_b.npy",
                },
                "volume_shape": {
                    "X": 2,
                    "Y": 2,
                    "Z": 1,
                    "T": 2,
                },
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            loaded = load_experiment_config(config_path)
            self.assertEqual(loaded.experiment, "selector-b")
            self.assertEqual(loaded.exp_id, "siren-b")
            self.assertEqual(loaded.data.target, "b")
            self.assertEqual(sorted(loaded.data.targets), ["a", "b"])

    def test_load_config_rejects_target_selector_with_target_path(self):
        config = {
            "exp_id": "bad-selector",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target": "a",
                "target_path": "./target.npy",
                "targets": {
                    "a": "./target_a.npy",
                },
                "volume_shape": {
                    "X": 2,
                    "Y": 2,
                    "Z": 1,
                    "T": 2,
                },
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data.target cannot be combined with data.target_path"):
                load_experiment_config(config_path)

    def test_load_config_rejects_target_selector_with_target_dir(self):
        config = {
            "exp_id": "bad-selector",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target": "a",
                "target_dir": "./targets",
                "targets": {
                    "a": "./target_a.npy",
                },
                "volume_shape": {
                    "X": 2,
                    "Y": 2,
                    "Z": 1,
                    "T": 2,
                },
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data.target cannot be combined with data.target_dir"):
                load_experiment_config(config_path)

    def test_load_config_rejects_target_placeholder_without_target(self):
        config = {
            "experiment": "selector-{target}",
            "exp_id": "siren-{target}",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "targets": {
                    "a": "./target_a.npy",
                },
                "volume_shape": {
                    "X": 2,
                    "Y": 2,
                    "Z": 1,
                    "T": 2,
                },
            },
            "model": {
                "name": "siren",
            },
            "training": {
                "device": "cpu",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "uses '\\{target\\}' but data.target is not set"):
                load_experiment_config(config_path)

    def test_run_train_logs_effective_config_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
            volume_path = root / "volume.npy"
            np.save(volume_path, volume)

            config = {
                "experiment": "log-config",
                "exp_id": "log-config",
                "experiment_root": str(root / "runs"),
                "data": {
                    "kind": "volume",
                    "target_path": str(volume_path),
                },
                "model": {
                    "name": "siren",
                    "hidden_features": 8,
                    "hidden_layers": 1,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "lr": 1.0e-3,
                    "device": "cpu",
                    "seed": 0,
                    "val_split": 0.0,
                    "log_every": 1,
                    "sampler": "uniform_random",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            run_train(config_path)

            saved_config = yaml.safe_load((root / "runs" / "log-config" / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["model"]["name"], "siren")
            self.assertEqual(saved_config["model"]["out_features"], 1)
            self.assertEqual(saved_config["training"]["weight_decay"], 0.0)
            self.assertEqual(saved_config["evaluation"]["batch_size"], 16384)

            logs_dir = root / "runs" / "log-config" / "logs"
            log_path = next(logs_dir.glob("run_*.log"))
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Effective config:", log_text)
            self.assertIn("weight_decay: 0.0", log_text)
            self.assertIn("outermost_linear: true", log_text)
            self.assertIn("out_features: 1", log_text)

    def test_run_train_with_target_selector_uses_selected_target_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coords = np.array(
                [
                    [-1.0, -1.0, -1.0, -1.0],
                    [-0.5, 0.0, 0.5, -0.5],
                    [0.0, 0.5, -0.5, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            a = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
            b = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
            coords_path = root / "coords.npy"
            a_path = root / "target_a.npy"
            b_path = root / "target_b.npy"
            np.save(coords_path, coords)
            np.save(a_path, a)
            np.save(b_path, b)

            config = {
                "experiment": "selector-{target}",
                "exp_id": "selector-{target}",
                "experiment_root": str(root / "runs"),
                "data": {
                    "kind": "node",
                    "coords_path": str(coords_path),
                    "targets": {
                        "a": str(a_path),
                        "b": str(b_path),
                    },
                    "target": "b",
                },
                "model": {
                    "name": "siren",
                    "hidden_features": 8,
                    "hidden_layers": 1,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 2,
                    "num_workers": 0,
                    "lr": 1.0e-3,
                    "device": "cpu",
                    "seed": 0,
                    "val_split": 0.0,
                    "log_every": 1,
                    "sampler": "uniform_random",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            result = run_train(config_path)

            self.assertEqual(sorted(result["prediction_paths"]), ["b"])
            self.assertTrue(result["prediction_paths"]["b"].name.endswith("b.npy"))
            saved_config = yaml.safe_load((root / "runs" / "selector-b" / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["exp_id"], "selector-b")
            self.assertEqual(saved_config["experiment"], "selector-b")
            self.assertEqual(saved_config["data"]["target"], "b")
            self.assertEqual(sorted(saved_config["data"]["targets"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
