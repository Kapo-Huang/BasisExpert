import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.config.io import load_experiment_config


class ConfigLoadingTestCase(unittest.TestCase):
    def _latest_run_dir(self, experiment_root: Path, exp_id: str) -> Path:
        exp_dir = experiment_root / exp_id
        candidates = sorted(path for path in exp_dir.iterdir() if path.is_dir())
        if not candidates:
            raise AssertionError(f"No run directories found under {exp_dir}")
        return candidates[-1]

    def test_removed_datasets_have_no_configs(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_names = [str(path.relative_to(repo_root)).lower() for path in (repo_root / "configs").rglob("*.yaml")]
        self.assertFalse(any("car" in name or "linkage" in name for name in config_names))

    def test_repo_bathymetry_configs_load(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_paths = [
            repo_root / "configs" / "VarExpert" / "bathymetry.yaml",
            repo_root / "configs" / "MoE-INR" / "bathymetry__SALT.yaml",
            repo_root / "configs" / "CoordNet" / "bathymetry__SALT.yaml",
            repo_root / "configs" / "SIREN" / "bathymetry__SALT.yaml",
        ]

        loaded_configs = [load_experiment_config(path) for path in config_paths]

        self.assertEqual(loaded_configs[0].exp_id, "var-expert-bathymetry")
        self.assertEqual(loaded_configs[0].data.dataset_name, "bathymetry")
        self.assertIsNone(loaded_configs[0].data.target)
        for loaded in loaded_configs[1:]:
            self.assertEqual(loaded.data.dataset_name, "bathymetry")
            self.assertEqual(loaded.data.target, "SALT")
            self.assertIn("SALT", loaded.data.targets)
            self.assertIn("-bathymetry-SALT", loaded.exp_id)

    def test_combustion_config_uses_xyt_coordinates(self):
        repo_root = Path(__file__).resolve().parents[1]
        loaded = load_experiment_config(
            repo_root / "configs" / "VarExpert" / "combustion_40NH3_1.yaml"
        )
        self.assertEqual(loaded.data.coordinate_axes, ("x", "y", "t"))
        self.assertEqual(loaded.model.params["in_features"], 3)
        self.assertEqual(len(loaded.data.targets), 13)

    def test_repo_standard_configs_load_recursively(self):
        repo_root = Path(__file__).resolve().parents[1]
        configs_root = repo_root / "configs"
        self.assertFalse((configs_root / "examples").exists())

        config_paths = sorted(
            [
                *configs_root.joinpath("VarExpert").rglob("*.yaml"),
                *configs_root.joinpath("MoE-INR").rglob("*.yaml"),
                *configs_root.joinpath("CoordNet").rglob("*.yaml"),
                *configs_root.joinpath("SIREN").rglob("*.yaml"),
            ]
        )
        self.assertTrue(any(path.parts[-2] == "Size163" and path.name == "ionization__GT.yaml" for path in config_paths))
        self.assertFalse((configs_root / "VarExpert" / "ionization_e4_k3.yaml").exists())
        self.assertFalse(any(configs_root.joinpath("VarExpert").glob("exp_data_ionization_var_expert_*.yaml")))

        for path in config_paths:
            loaded = load_experiment_config(path)
            self.assertTrue(path.exists())
            self.assertTrue(str(loaded.source_config_path).endswith(".yaml"))

    def test_repo_root_placeholder_resolves_from_non_repo_cwd(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs" / "VarExpert" / "Size1304" / "ionization.yaml"
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                loaded = load_experiment_config(config_path)
            finally:
                os.chdir(previous_cwd)
        self.assertEqual(Path(loaded.experiment_root), repo_root / "runs")
        self.assertEqual(
            Path(loaded.training.pretrain.assignments_cache_path),
            repo_root / "data" / "cache" / "ionization_voxel_assignments_6.npy",
        )
        self.assertEqual(
            Path(loaded.data.targets["GT"]),
            repo_root / "data" / "Volume" / "Ionization" / "target_GT.npy",
        )

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

    def test_load_config_accepts_log_section(self):
        config = {
            "exp_id": "with-log",
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
            "log": {
                "effective_config": False,
                "model_stats": False,
                "epoch_summary": False,
                "startup_timing": False,
                "psnr": {
                    "enabled": False,
                    "per_target": False,
                },
                "timing": {
                    "enabled": False,
                    "epoch_breakdown": False,
                    "step_window": False,
                    "step_window_every_steps": 5,
                    "cuda_sync": False,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            loaded = load_experiment_config(config_path)
            self.assertFalse(loaded.log.effective_config)
            self.assertFalse(loaded.log.model_stats)
            self.assertFalse(loaded.log.epoch_summary)
            self.assertFalse(loaded.log.startup_timing)
            self.assertFalse(loaded.log.psnr.enabled)
            self.assertFalse(loaded.log.psnr.per_target)
            self.assertFalse(loaded.log.timing.enabled)
            self.assertEqual(loaded.log.timing.step_window_every_steps, 5)

    def test_load_config_accepts_multiview_dwa_loss_section(self):
        config = {
            "exp_id": "with-dwa",
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
                "multiview_dwa_loss": {
                    "enabled": True,
                    "temperature": 0.3,
                    "eps": 1.0e-10,
                    "warmup_epochs": 4,
                    "max_factor_max": 1.3,
                    "max_factor_min": 1.02,
                    "update_schedule": " COSINE ",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            loaded = load_experiment_config(config_path)
            self.assertTrue(loaded.training.multiview_dwa_loss.enabled)
            self.assertEqual(loaded.training.multiview_dwa_loss.temperature, 0.3)
            self.assertEqual(loaded.training.multiview_dwa_loss.eps, 1.0e-10)
            self.assertEqual(loaded.training.multiview_dwa_loss.warmup_epochs, 4)
            self.assertEqual(loaded.training.multiview_dwa_loss.max_factor_max, 1.3)
            self.assertEqual(loaded.training.multiview_dwa_loss.max_factor_min, 1.02)
            self.assertEqual(loaded.training.multiview_dwa_loss.update_schedule, "cosine")

    def test_load_config_rejects_invalid_multiview_dwa_loss_settings(self):
        base_config = {
            "exp_id": "bad-dwa",
            "experiment_root": "runs",
            "data": {"kind": "volume", "target_path": "./target.npy"},
            "model": {"name": "siren"},
            "training": {"device": "cpu"},
        }
        invalid_sections = (
            {"temperature": 0.0},
            {"eps": 0.0},
            {"warmup_epochs": -1},
            {"max_factor_min": 0.99},
            {"max_factor_max": 1.1, "max_factor_min": 1.2},
            {"update_schedule": "linear"},
            {"window_size": 5},
        )
        for section in invalid_sections:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as tmpdir:
                config = dict(base_config)
                config["training"] = {
                    "device": "cpu",
                    "multiview_dwa_loss": section,
                }
                config_path = Path(tmpdir) / "config.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_experiment_config(config_path)

    def test_load_config_rejects_unknown_log_key(self):
        config = {
            "exp_id": "bad-log",
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
            "log": {
                "unknown_flag": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_config(config_path)

    def test_load_config_rejects_removed_pretrain_keys(self):
        config = {
            "exp_id": "bad-pretrain",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target_path": "./target.npy",
            },
            "model": {
                "name": "var_expert",
                "num_experts": 2,
                "base_dim": 2,
                "top_k": 1,
            },
            "training": {
                "device": "cpu",
                "pretrain": {
                    "enabled": True,
                    "epochs": 1,
                    "batch_size": 32,
                    "assignments_method": "voxel_clustering",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown training.pretrain keys"):
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

    def test_load_config_rejects_removed_target_dir_key(self):
        config = {
            "exp_id": "bad-selector",
            "experiment_root": "runs",
            "data": {
                "kind": "volume",
                "target_dir": "./targets",
                "targets": {"a": "./target_a.npy"},
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
            with self.assertRaisesRegex(ValueError, "Unknown data keys: target_dir"):
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

            run_dir = self._latest_run_dir(root / "runs", "log-config")
            saved_config = yaml.safe_load((run_dir / "configs" / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["model"]["name"], "siren")
            self.assertEqual(saved_config["model"]["out_features"], 1)
            self.assertEqual(saved_config["training"]["weight_decay"], 0.0)
            self.assertEqual(saved_config["evaluation"]["batch_size"], 16384)
            self.assertTrue(saved_config["log"]["effective_config"])

            logs_dir = run_dir / "logs"
            log_path = next(logs_dir.glob("run_*.log"))
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("Effective config:", log_text)
            self.assertIn("weight_decay: 0.0", log_text)
            self.assertIn("outermost_linear: true", log_text)
            self.assertIn("out_features: 1", log_text)

    def test_run_train_saves_compact_var_expert_effective_config(self):
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
            b = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
            coords_path = root / "coords.npy"
            a_path = root / "target_a.npy"
            b_path = root / "target_b.npy"
            np.save(coords_path, coords)
            np.save(a_path, a)
            np.save(b_path, b)

            config = {
                "experiment": "compact-var-expert",
                "exp_id": "compact-var-expert",
                "experiment_root": str(root / "runs"),
                "data": {
                    "kind": "node",
                    "coords_path": str(coords_path),
                    "targets": {
                        "a": str(a_path),
                        "b": str(b_path),
                    },
                },
                "model": {
                    "name": "var_expert",
                    "num_experts": 2,
                    "base_dim": 2,
                    "top_k": 1,
                    "decoder_num_layers": 2,
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
                    "save_every": 0,
                    "sampler": "uniform_random",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            run_train(config_path)

            run_dir = self._latest_run_dir(root / "runs", "compact-var-expert")
            saved_model = yaml.safe_load((run_dir / "configs" / "config.yaml").read_text(encoding="utf-8"))["model"]
            self.assertEqual(saved_model["name"], "var_expert")
            self.assertEqual(saved_model["in_features"], 4)
            self.assertEqual(saved_model["num_experts"], 2)
            self.assertEqual(saved_model["base_dim"], 2)
            self.assertEqual(saved_model["top_k"], 1)
            self.assertEqual(saved_model["decoder_num_layers"], 2)
            self.assertNotIn("expert_num_frequencies", saved_model)
            self.assertNotIn("expert_num_layers", saved_model)
            self.assertNotIn("gate_num_layers", saved_model)
            self.assertNotIn("head_num_layers", saved_model)
            self.assertNotIn("expert_first_omega_0", saved_model)
            self.assertNotIn("head_hidden_omega_0", saved_model)

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
            run_dir = self._latest_run_dir(root / "runs", "selector-b")
            saved_config = yaml.safe_load((run_dir / "configs" / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["exp_id"], "selector-b")
            self.assertEqual(saved_config["experiment"], "selector-b")
            self.assertEqual(saved_config["data"]["target"], "b")
            self.assertEqual(sorted(saved_config["data"]["targets"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()


