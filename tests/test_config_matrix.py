import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from scripts import generate_config_matrix

from var_expert_inr.apmgsrn.model import APMGSRN
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.fv_srn.model import TemporalFVSRN
from var_expert_inr.mc_inr.data import TargetLayoutEntry
from var_expert_inr.mc_inr.model import MCINR
from var_expert_inr.models import build_model
from var_expert_inr.neural_expert.ionization.model_registry import build_model as build_neural_model
from var_expert_inr.rmdsrn.model import RMDSRN


DATASET_TARGETS = {
    "bathymetry": {"SALT", "TEMP", "U", "V"},
    "katrina": {"fort63", "fort64", "fort73", "speed", "v"},
    "ionization": {"GT", "H_plus", "H2", "He", "PD"},
}
COMBUSTION_DATASET = "combustion_40NH3_1"
COMBUSTION_TARGETS = set(generate_config_matrix.COMBUSTION_DATASET["targets"])
COMBUSTION_SCALAR_TARGETS = set(generate_config_matrix.COMBUSTION_SCALAR_TARGETS)
SIZES = {
    "Size082": 0.82,
    "Size163": 1.63,
    "Size326": 3.26,
    "Size652": 6.52,
    "Size1304": 13.04,
}
MOE_RERUN_LIST = "run_moe_non_ionization_main.list"
COMBINED_RUN_LIST = "run_neural_expert_non_ionization_main.list"
COORDNET_MVNET_STSR_RUN_LIST = "run_coordnet_combustion_mvnet_katrina_stsr_redsea.list"


class ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.config_root = cls.repo_root / "configs"
        cls.paths = sorted(cls.config_root.rglob("*.yaml"))

    def read_run_list(self, name: str) -> list[str]:
        list_path = self.repo_root / "scripts" / name
        selected = [
            line.split("#", 1)[0].strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
        ]
        return [line for line in selected if line]

    def test_matrix_contains_exactly_459_configs_and_no_removed_datasets(self):
        self.assertEqual(len(self.paths), 459)
        relative_names = [str(path.relative_to(self.config_root)).lower() for path in self.paths]
        self.assertFalse(any("car" in name or "linkage" in name for name in relative_names))

    def test_generator_preserves_combustion_and_generates_459_configs(self):
        committed_path = self.config_root / "VarExpert" / "combustion_40NH3_1.yaml"
        committed_payload = yaml.safe_load(committed_path.read_text(encoding="utf-8"))
        committed_moe = {
            path.relative_to(self.config_root / "MoE-INR").as_posix(): yaml.safe_load(
                path.read_text(encoding="utf-8")
            )
            for path in (self.config_root / "MoE-INR").rglob("*.yaml")
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs"
            with mock.patch.object(generate_config_matrix, "CONFIGS", generated_root):
                generate_config_matrix.main()
            generated_paths = sorted(generated_root.rglob("*.yaml"))
            generated_payload = yaml.safe_load(
                (generated_root / "VarExpert" / "combustion_40NH3_1.yaml").read_text(
                    encoding="utf-8"
                )
            )
            generated_moe = {
                path.relative_to(generated_root / "MoE-INR").as_posix(): yaml.safe_load(
                    path.read_text(encoding="utf-8")
                )
                for path in (generated_root / "MoE-INR").rglob("*.yaml")
            }

        self.assertEqual(len(generated_paths), 459)
        self.assertEqual(generated_payload, committed_payload)
        self.assertEqual(generated_moe, committed_moe)

    def test_default_run_list_contains_the_complete_formal_matrix(self):
        selected = self.read_run_list("run_all_configs.list")
        expected = {path.relative_to(self.repo_root).as_posix() for path in self.paths}

        self.assertEqual(len(selected), 459)
        self.assertEqual(set(selected), expected)

    def test_main_and_rd_curve_lists_partition_the_complete_matrix(self):
        all_configs = set(self.read_run_list("run_all_configs.list"))
        main_configs = self.read_run_list("run_main_configs.list")
        rd_curve_configs = self.read_run_list("run_rd_curve_configs.list")
        size_marker = "/Size"

        self.assertEqual(len(main_configs), 249)
        self.assertEqual(len(rd_curve_configs), 210)
        self.assertTrue(all(size_marker not in path for path in main_configs))
        self.assertTrue(all(size_marker in path for path in rd_curve_configs))
        self.assertTrue(set(main_configs).isdisjoint(rd_curve_configs))
        self.assertEqual(set(main_configs) | set(rd_curve_configs), all_configs)

    def test_moe_non_ionization_rerun_list_has_exact_main_scope(self):
        selected = self.read_run_list(MOE_RERUN_LIST)
        expected = {
            f"configs/MoE-INR/bathymetry__{target}.yaml"
            for target in DATASET_TARGETS["bathymetry"]
        }
        expected.update(
            f"configs/MoE-INR/katrina__{target}.yaml"
            for target in DATASET_TARGETS["katrina"]
        )
        expected.update(
            f"configs/MoE-INR/{COMBUSTION_DATASET}__{target}.yaml"
            for target in COMBUSTION_TARGETS
        )

        self.assertEqual(len(selected), 22)
        self.assertEqual(len(set(selected)), 22)
        self.assertEqual(set(selected), expected)
        self.assertTrue(all("/Size" not in path for path in selected))
        self.assertTrue(all("ionization" not in path.lower() for path in selected))

    def test_siren_neural_expert_combined_list_has_exact_scope_and_budgets(self):
        selected = self.read_run_list(COMBINED_RUN_LIST)
        siren_expected = {
            f"configs/SIREN/{COMBUSTION_DATASET}__{target}.yaml"
            for target in COMBUSTION_TARGETS
        }
        neural_targets = DATASET_TARGETS["bathymetry"] | COMBUSTION_SCALAR_TARGETS
        neural_main_expected = {
            f"configs/NeuralExpert/{dataset}__{target}.yaml"
            for dataset, targets in (
                ("bathymetry", DATASET_TARGETS["bathymetry"]),
                (COMBUSTION_DATASET, COMBUSTION_SCALAR_TARGETS),
            )
            for target in targets
        }
        neural_manager_expected = {
            path.removesuffix(".yaml") + "__managerpretrain.yaml"
            for path in neural_main_expected
        }
        expected = siren_expected | neural_main_expected | neural_manager_expected

        self.assertEqual(len(selected), 45)
        self.assertEqual(len(set(selected)), 45)
        self.assertEqual(set(selected), expected)
        self.assertTrue(all("/Size" not in path for path in selected))
        self.assertTrue(all("ionization" not in path.lower() for path in selected))
        self.assertTrue(all("katrina" not in path.lower() for path in selected))
        self.assertNotIn(
            f"configs/NeuralExpert/{COMBUSTION_DATASET}__Velocity.yaml",
            selected,
        )

        for relative in siren_expected:
            payload = yaml.safe_load((self.repo_root / relative).read_text(encoding="utf-8"))
            training = payload["training"]
            self.assertEqual(training["epochs"], 600, relative)
            self.assertEqual(training["batch_size"], 16000, relative)
            self.assertEqual(training["batches_per_epoch_budget"], 1500, relative)
            self.assertEqual(training["log_psnr_every"], 100, relative)
            self.assertEqual(training["psnr_sample_ratio"], 0.1, relative)

        for relative in neural_main_expected:
            main_path = self.repo_root / relative
            manager_path = main_path.with_name(f"{main_path.stem}__managerpretrain.yaml")
            main = yaml.safe_load(main_path.read_text(encoding="utf-8"))
            manager = yaml.safe_load(manager_path.read_text(encoding="utf-8"))
            self.assertEqual(main["TRAINING"]["n_points"], 16000, relative)
            self.assertEqual(main["TRAINING"]["num_epochs"], 60000, relative)
            self.assertEqual(manager["TRAINING"]["n_points"], 16000, manager_path)
            self.assertEqual(manager["TRAINING"]["num_epochs"], 30000, manager_path)
            self.assertEqual(main["MODEL"]["manager_pt_path"], manager["MODEL"]["manager_pt_path"])

    def test_coordnet_mvnet_stsr_list_and_redsea_config(self):
        selected = self.read_run_list(COORDNET_MVNET_STSR_RUN_LIST)
        coordnet = {
            f"configs/CoordNet/{COMBUSTION_DATASET}__{target}.yaml"
            for target in COMBUSTION_TARGETS
        }
        expected = coordnet | {
            "configs/MVNet/katrina.yaml",
            "configs/STSR-INR/redsea.yaml",
        }
        self.assertEqual(len(selected), 15)
        self.assertEqual(set(selected), expected)

        payload = yaml.safe_load(
            (self.config_root / "STSR-INR" / "redsea.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["data"]["dataset_name"], "redsea")
        self.assertEqual(
            list(payload["data"]["targets"]),
            ["fort63", "fort64", "fort73", "speed", "v"],
        )
        self.assertEqual(payload["model"]["name"], "stsr_inr")
        self.assertEqual(payload["training"]["epochs"], 60)
        self.assertEqual(payload["training"]["batch_size"], 8192)
        self.assertEqual(payload["training"]["sampler"], "uniform_random")

    def test_single_target_default_and_size_coverage(self):
        for family in ("SIREN", "CoordNet", "MoE-INR", "NeuralExpert"):
            for dataset, targets in DATASET_TARGETS.items():
                actual = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family).glob(f"{dataset}__*.yaml")
                    if "managerpretrain" not in path.stem
                }
                self.assertEqual(actual, targets, (family, dataset))
            combustion = {
                path.stem.split("__")[1]
                for path in (self.config_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
                if "managerpretrain" not in path.stem
            }
            expected_combustion = (
                COMBUSTION_SCALAR_TARGETS
                if family == "NeuralExpert"
                else COMBUSTION_TARGETS
            )
            self.assertEqual(combustion, expected_combustion, family)
            for size in SIZES:
                actual = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family / size).glob("ionization__*.yaml")
                    if "managerpretrain" not in path.stem
                }
                self.assertEqual(actual, DATASET_TARGETS["ionization"], (family, size))

        for family in ("APMGSRN", "fV-SRN", "RMDSRN"):
            actual = {path.stem.split("__")[1] for path in (self.config_root / family).glob("ionization__*.yaml")}
            self.assertEqual(actual, DATASET_TARGETS["ionization"])
            combustion = {
                path.stem.split("__")[1]
                for path in (self.config_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
            }
            self.assertEqual(combustion, COMBUSTION_SCALAR_TARGETS)
            for size in SIZES:
                sized = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family / size).glob("ionization__*.yaml")
                }
                self.assertEqual(sized, DATASET_TARGETS["ionization"])

        ecnr_targets = {
            path.stem.split("__")[1]
            for path in (self.config_root / "ECNR").glob("ionization__*.yaml")
        }
        self.assertEqual(ecnr_targets, DATASET_TARGETS["ionization"])
        ecnr_combustion = {
            path.stem.split("__")[1]
            for path in (self.config_root / "ECNR").glob(
                f"{COMBUSTION_DATASET}__*.yaml"
            )
        }
        self.assertEqual(ecnr_combustion, COMBUSTION_SCALAR_TARGETS)
        self.assertEqual(list((self.config_root / "ECNR").glob("Size*/*.yaml")), [])

        for family in ("InstantNGP", "InstantVNR"):
            instant_targets = {
                path.stem.split("__")[1]
                for path in (self.config_root / family).glob(
                    "ionization__*.yaml"
                )
            }
            self.assertEqual(instant_targets, DATASET_TARGETS["ionization"])
            instant_combustion = {
                path.stem.split("__")[1]
                for path in (self.config_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
            }
            self.assertEqual(instant_combustion, COMBUSTION_SCALAR_TARGETS)
            self.assertEqual(
                list((self.config_root / family).glob("bathymetry__*.yaml")),
                [],
            )
            self.assertEqual(
                list((self.config_root / family).glob("katrina__*.yaml")),
                [],
            )

    def test_neural_expert_has_matching_manager_pretrains(self):
        root = self.config_root / "NeuralExpert"
        main_configs = [path for path in root.rglob("*.yaml") if "managerpretrain" not in path.stem]
        self.assertEqual(len(main_configs), 51)
        for main in main_configs:
            manager = main.with_name(f"{main.stem}__managerpretrain.yaml")
            self.assertTrue(manager.exists(), main)
            main_payload = yaml.safe_load(main.read_text(encoding="utf-8"))
            manager_payload = yaml.safe_load(manager.read_text(encoding="utf-8"))
            self.assertEqual(main_payload["TRAINING"]["n_points"], 16000, main)
            self.assertEqual(main_payload["TRAINING"]["num_epochs"], 60000, main)
            self.assertEqual(manager_payload["TRAINING"]["n_points"], 16000, manager)
            self.assertEqual(manager_payload["TRAINING"]["num_epochs"], 30000, manager)

    def test_mvnet_has_one_multi_target_config_per_base_dataset(self):
        root = self.config_root / "MVNet"
        self.assertEqual(
            {path.stem for path in root.glob("*.yaml")},
            set(DATASET_TARGETS) | {COMBUSTION_DATASET},
        )
        expected_by_dataset = {
            **DATASET_TARGETS,
            COMBUSTION_DATASET: COMBUSTION_SCALAR_TARGETS,
        }
        for dataset, expected_targets in expected_by_dataset.items():
            payload = yaml.safe_load(
                (root / f"{dataset}.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(payload["data"]["targets"]),
                expected_targets,
            )
            self.assertNotIn("target", payload["data"])
            self.assertEqual(
                payload["model"]["out_features"],
                len(expected_targets),
            )
        self.assertEqual(list(root.glob("Size*/*.yaml")), [])

    def test_all_ionization_configs_use_100_timesteps(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = payload.get("data") or payload.get("DATA")
            if data.get("dataset_name") == "ionization":
                self.assertEqual(data["volume_shape"]["T"], 100, path)

    def test_all_moe_configs_declare_consistent_widths(self):
        root = self.config_root / "MoE-INR"
        main_paths = sorted(root.glob("*.yaml"))
        paths = sorted(root.rglob("*.yaml"))
        self.assertEqual(len(main_paths), 27)
        self.assertEqual(len(paths), 52)
        self.assertEqual(len(paths) - len(main_paths), 25)
        for path in paths:
            model = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
            self.assertEqual(model["encoder_feature_dim"], 8 * model["base_dim"], path)
            self.assertEqual(model["policy_hidden_dim"], model["base_dim"], path)

    def test_moe_main_widths_match_dataset_budgets(self):
        expected_widths = {
            "bathymetry": {"default": 18},
            "katrina": {"default": 16, "v": 15},
            COMBUSTION_DATASET: {"default": 23, "Velocity": 22},
        }
        total_mib = {dataset: 0.0 for dataset in expected_widths}

        for dataset, widths in expected_widths.items():
            for path in sorted((self.config_root / "MoE-INR").glob(f"{dataset}__*.yaml")):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                target = payload["data"]["target"]
                target_dim = 3 if target in {"v", "Velocity"} else 1
                expected_base_dim = widths.get(target, widths["default"])
                self.assertEqual(payload["model"]["base_dim"], expected_base_dim, path)
                meta = DatasetMeta(
                    kind=payload["data"]["kind"],
                    n_samples=1,
                    input_dim=payload["model"]["in_features"],
                    target_names=(target,),
                    target_dims={target: target_dim},
                    volume_shape=payload["data"].get("volume_shape"),
                )
                model_payload = payload["model"]
                model = build_model(
                    ModelConfig(
                        name=model_payload["name"],
                        params={key: value for key, value in model_payload.items() if key != "name"},
                    ),
                    meta,
                )
                total_mib[dataset] += sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)

        var_expert_totals = {}
        for dataset, target_dims in {
            "bathymetry": {name: 1 for name in DATASET_TARGETS["bathymetry"]},
            "katrina": {
                **{name: 1 for name in DATASET_TARGETS["katrina"]},
                "v": 3,
            },
        }.items():
            payload = yaml.safe_load(
                (self.config_root / "VarExpert" / f"{dataset}.yaml").read_text(encoding="utf-8")
            )
            model_payload = payload["model"]
            target_names = tuple(payload["data"]["targets"])
            model = build_model(
                ModelConfig(
                    name=model_payload["name"],
                    params={key: value for key, value in model_payload.items() if key != "name"},
                ),
                DatasetMeta(
                    kind="node",
                    n_samples=1,
                    input_dim=model_payload["in_features"],
                    target_names=target_names,
                    target_dims=target_dims,
                    volume_shape=None,
                ),
            )
            var_expert_totals[dataset] = sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)

        self.assertLessEqual(
            abs(total_mib["bathymetry"] - var_expert_totals["bathymetry"])
            / var_expert_totals["bathymetry"],
            0.05,
        )
        self.assertLessEqual(
            abs(total_mib["katrina"] - var_expert_totals["katrina"])
            / var_expert_totals["katrina"],
            0.05,
        )
        self.assertLessEqual(abs(total_mib[COMBUSTION_DATASET] - 1.11) / 1.11, 0.05)

    def test_all_combustion_configs_use_exported_shape(self):
        expected_shape = generate_config_matrix.COMBUSTION_DATASET["volume_shape"]
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = payload.get("data") or payload.get("DATA")
            if data.get("dataset_name", "").lower() == COMBUSTION_DATASET.lower():
                self.assertEqual(data["volume_shape"], expected_shape, path)

    def test_generated_paths_are_repo_root_relative(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for value_path, value in self._walk_strings(payload):
                if "../" in value or "..\\" in value:
                    self.fail(f"Generated config contains ambiguous relative path at {path}:{value_path}: {value}")
                if self._looks_like_generated_path(value_path, value):
                    self.assertTrue(
                        value.startswith("${REPO_ROOT}/")
                        or value.startswith("${DATASETS_ROOT}/"),
                        f"Expected a declared root token at {path}:{value_path}, got {value!r}",
                    )

    def test_generated_configs_disable_training_predictions_and_step_windows(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            family = path.relative_to(self.config_root).parts[0]
            if family in {"VarExpert", "MVNet", "SIREN", "CoordNet", "MoE-INR", "InstantNGP", "InstantVNR", "MC-INR", "fV-SRN", "ECNR", "STSR-INR"}:
                self.assertFalse(payload["evaluation"]["save_predictions"], path)
            if family in {"fV-SRN", "RMDSRN"}:
                self.assertFalse(payload["evaluation"]["run_after_training"], path)
            if family == "APMGSRN":
                self.assertFalse(payload["EVALUATION"]["run_after_training"], path)
            timing = (payload.get("log") or {}).get("timing")
            if timing is not None:
                self.assertFalse(timing["step_window"], path)

    def test_var_expert_balancing_profiles(self):
        for path in (self.config_root / "VarExpert").rglob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            ema = payload["training"]["multiview_ema_loss"]
            is_size_config = path.relative_to(self.config_root / "VarExpert").parts[0].startswith("Size")
            if is_size_config:
                self.assertEqual(ema["beta"], 0.99, path)
                self.assertEqual(ema["w_min"], 0.5, path)
                self.assertEqual(ema["w_max"], 2.0, path)
                self.assertEqual(ema["warmup_steps"], 75000, path)
                self.assertEqual(ema["alpha"], 1.0, path)
            else:
                self.assertEqual(ema["alpha"], 5.0, path)

    def test_exploration_v2_winners_drive_formal_size_profiles(self):
        expected_var = {
            "Size082": (8, 15, 4),
            "Size163": (9, 21, 4),
            "Size326": (9, 30, 4),
            "Size652": (9, 43, 4),
            "Size1304": (9, 61, 4),
        }
        for size in SIZES:
            var_model = yaml.safe_load(
                (self.config_root / "VarExpert" / size / "ionization.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual(
                (var_model["num_experts"], var_model["base_dim"], var_model["top_k"]),
                expected_var[size],
            )

            mc_model = yaml.safe_load(
                (self.config_root / "MC-INR" / size / "ionization.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual((mc_model["gfe_layers"], mc_model["lfe_layers"]), (3, 4))

            for target in DATASET_TARGETS["ionization"]:
                for suffix in ("", "__managerpretrain"):
                    neural_model = yaml.safe_load(
                        (self.config_root / "NeuralExpert" / size / f"ionization__{target}{suffix}.yaml").read_text(
                            encoding="utf-8"
                        )
                    )["MODEL"]
                    self.assertEqual(neural_model["decoder_n_hidden_layers"], 1)
                    self.assertEqual(neural_model["manager_n_hidden_layers"], 1)

    def test_apmgsrn_exploration_winner_drives_main_configs(self):
        expected_model = {
            "feature_grid_shape": [4, 4, 4],
            "n_features": 14,
            "n_grids": 1,
            "nodes_per_layer": 16,
            "n_layers": 3,
            "requires_padded_feats": True,
        }
        for path in (self.config_root / "APMGSRN").glob("*.yaml"):
            model = yaml.safe_load(path.read_text(encoding="utf-8"))["MODEL"]
            for key, value in expected_model.items():
                self.assertEqual(model[key], value, path)

    def test_instant_ngp_uses_fixed_model_optimizer_and_single_target(self):
        expected_targets = DATASET_TARGETS["ionization"] | COMBUSTION_SCALAR_TARGETS
        actual_targets = set()
        for path in (self.config_root / "InstantNGP").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            actual_targets.update(payload["data"]["targets"])
            self.assertEqual(len(payload["data"]["targets"]), 1, path)
            self.assertNotIn("target", payload["data"])

            model = payload["model"]
            self.assertEqual(model["name"], "instant_ngp")
            self.assertEqual(model["in_features"], 4)
            self.assertEqual(model["out_features"], 1)
            self.assertEqual(model["n_levels"], 16)
            self.assertEqual(model["n_features_per_level"], 2)
            self.assertEqual(model["base_resolution"], 16)
            self.assertEqual(model["finest_resolution"], 600)
            self.assertEqual(model["log2_hashmap_size"], 19)
            self.assertEqual(model["hidden_features"], 64)
            self.assertEqual(model["hidden_layers"], 2)

            training = payload["training"]
            self.assertEqual(training["epochs"], 600)
            self.assertEqual(training["batch_size"], 16_000)
            self.assertEqual(
                training["batches_per_epoch_budget"], 1_500
            )
            self.assertEqual(
                training["gradient_accumulation_steps"], 16
            )
            self.assertEqual(training["lr"], 1.0e-2)
            self.assertEqual(training["beta_1"], 0.9)
            self.assertEqual(training["beta_2"], 0.99)
            self.assertEqual(training["epsilon"], 1.0e-15)
            self.assertEqual(training["weight_decay"], 0.0)
            self.assertFalse(training["pretrain"]["enabled"])
            self.assertNotIn("max_steps", training)
            scheduler = training["scheduler"]
            self.assertTrue(scheduler["enabled"])
            self.assertEqual(scheduler["interval"], "optimizer_step")
            self.assertEqual(scheduler["milestones"], [20_480, 30_720])
            self.assertEqual(scheduler["gamma"], 0.33)
        self.assertEqual(actual_targets, expected_targets)

    def test_instant_vnr_uses_official_model_optimizer_and_unified_budget(self):
        expected_targets = DATASET_TARGETS["ionization"] | COMBUSTION_SCALAR_TARGETS
        actual_targets = set()
        for path in (self.config_root / "InstantVNR").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            actual_targets.update(payload["data"]["targets"])
            self.assertEqual(len(payload["data"]["targets"]), 1, path)
            self.assertEqual(
                payload["data"].get("coordinate_axes", ["x", "y", "z", "t"]),
                ["x", "y", "z", "t"],
                path,
            )

            self.assertEqual(
                payload["model"],
                {
                    "name": "instant_vnr",
                    "in_features": 4,
                    "out_features": 1,
                    "n_levels": 8,
                    "n_features_per_level": 8,
                    "base_resolution": 16,
                    "per_level_scale": 2.0,
                    "log2_hashmap_size": 19,
                    "hidden_features": 64,
                    "hidden_layers": 4,
                },
                path,
            )

            training = payload["training"]
            self.assertEqual(training["epochs"], 600)
            self.assertEqual(training["batch_size"], 16_000)
            self.assertEqual(training["batches_per_epoch_budget"], 1_500)
            self.assertEqual(training["gradient_accumulation_steps"], 4)
            self.assertEqual(training["loss_type"], "l1")
            self.assertEqual(training["lr"], 5.0e-3)
            self.assertEqual(training["beta_1"], 0.9)
            self.assertEqual(training["beta_2"], 0.999)
            self.assertEqual(training["epsilon"], 1.0e-15)
            self.assertEqual(training["weight_decay"], 1.0e-6)
            self.assertFalse(training["pretrain"]["enabled"])
            scheduler = training["scheduler"]
            self.assertTrue(scheduler["enabled"])
            self.assertEqual(scheduler["interval"], "optimizer_step")
            self.assertEqual(scheduler["decay_start"], 2_000)
            self.assertEqual(scheduler["step_size"], 1_000)
            self.assertEqual(scheduler["gamma"], 0.99)
        self.assertEqual(actual_targets, expected_targets)

    def test_mvnet_uses_fixed_model_optimizer_and_budget(self):
        expected_orders = {
            "bathymetry": ["SALT", "TEMP", "U", "V"],
            "katrina": ["fort63", "fort64", "fort73", "speed", "v"],
            "ionization": ["GT", "H2", "H_plus", "He", "PD"],
            COMBUSTION_DATASET: sorted(COMBUSTION_SCALAR_TARGETS),
        }
        for path in (self.config_root / "MVNet").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(payload["data"]["targets"]),
                expected_orders[path.stem],
            )
            model = payload["model"]
            self.assertEqual(model["name"], "mvnet")
            self.assertEqual(model["in_features"], 4)
            self.assertEqual(model["hidden_features"], 120)
            self.assertEqual(model["num_residual_blocks"], 10)
            self.assertEqual(model["omega_0"], 30.0)
            self.assertTrue(model["bias"])

            training = payload["training"]
            self.assertEqual(training["epochs"], 300)
            self.assertEqual(training["batch_size"], 2048)
            self.assertEqual(training["gradient_accumulation_steps"], 1)
            self.assertEqual(training["batches_per_epoch_budget"], 1500)
            self.assertEqual(training["sampler"], "budgeted_random")
            self.assertEqual(training["lr"], 1.0e-4)
            self.assertEqual(training["beta_1"], 0.9)
            self.assertEqual(training["beta_2"], 0.999)
            self.assertEqual(training["epsilon"], 1.0e-8)
            self.assertEqual(training["weight_decay"], 0.0)
            self.assertEqual(training["loss_type"], "mse")
            self.assertEqual(training["log_psnr_every"], 0)
            self.assertEqual(training["save_every"], 300)
            self.assertFalse(training["pretrain"]["enabled"])
            scheduler = training["scheduler"]
            self.assertTrue(scheduler["enabled"])
            self.assertEqual(scheduler["interval"], "epoch")
            self.assertEqual(scheduler["step_size"], 15)
            self.assertEqual(scheduler["gamma"], 0.8)

    def test_var_expert_ionization_dwa_config_uses_dwa_not_ema(self):
        payload = yaml.safe_load((self.config_root / "VarExpert" / "ionization_dwa.yaml").read_text(encoding="utf-8"))
        self.assertFalse(payload["training"]["multiview_ema_loss"]["enabled"])
        self.assertTrue(payload["training"]["multiview_dwa_loss"]["enabled"])
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["temperature"], 0.2)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["eps"], 1.0e-12)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["warmup_epochs"], 2)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["max_factor_max"], 1.25)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["max_factor_min"], 1.05)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["update_schedule"], "cosine")

    def test_primary_training_budget_is_exact(self):
        expected = 16_000 * 1_500 * 600
        for path in self.paths:
            if "managerpretrain" in path.stem:
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            family = path.relative_to(self.config_root).parts[0]
            if family == "MVNet":
                training = payload["training"]
                total = (
                    training["batch_size"]
                    * training["batches_per_epoch_budget"]
                    * training["epochs"]
                )
                self.assertEqual(total, 921_600_000, path)
                continue
            if family == "STSR-INR":
                training = payload["training"]
                self.assertEqual(training["epochs"], 60, path)
                self.assertEqual(training["batch_size"], 8192, path)
                self.assertNotIn("batches_per_epoch_budget", training)
                continue
            if family == "ECNR":
                training = payload["training"]
                total = (
                    3
                    * training["epochs_per_scale"]
                    * training["batch_size"]
                    * training["batches_per_epoch_budget"]
                )
                self.assertEqual(total, expected, path)
                self.assertEqual(training["primary_sample_budget"], expected, path)
                continue
            if family in {"VarExpert", "SIREN", "CoordNet", "MoE-INR", "InstantNGP", "InstantVNR"}:
                training = payload["training"]
                total = training["batch_size"] * training["batches_per_epoch_budget"] * training["epochs"]
            elif family == "NeuralExpert":
                total = payload["TRAINING"]["n_points"] * payload["TRAINING"]["num_epochs"]
                self.assertEqual(total, 960_000_000, path)
                continue
            elif family == "MC-INR":
                training = payload["training"]
                total = training["batch_size"] * training["batches_per_epoch_budget"] * training["finetune_epochs"]
            elif family == "APMGSRN":
                total = (
                    payload["DATA"]["volume_shape"]["T"]
                    * payload["TRAINING"]["iterations"]
                    * payload["TRAINING"]["points_per_iteration"]
                )
            elif family == "fV-SRN":
                training = payload["training"]
                total = (
                    training["epochs"]
                    * payload["data"]["volume_shape"]["T"]
                    * training["samples_per_timestep"]
                )
            elif family == "RMDSRN":
                total = payload["training"]["steps"] * payload["training"]["batch_size"]
            else:
                raise AssertionError(f"Unhandled family: {family}")
            if (
                family in {"APMGSRN", "fV-SRN"}
                and COMBUSTION_DATASET in path.name
            ):
                self.assertLess(abs(total - expected) / expected, 0.001, path)
            else:
                self.assertEqual(total, expected, path)

    def test_static_size_tiers_are_within_five_percent(self):
        for family in ("SIREN", "CoordNet", "MoE-INR", "VarExpert"):
            for size, target_mib in SIZES.items():
                filename = "ionization.yaml" if family == "VarExpert" else "ionization__GT.yaml"
                payload = yaml.safe_load((self.config_root / family / size / filename).read_text(encoding="utf-8"))
                model_payload = payload["model"]
                target_names = tuple(payload["data"]["targets"]) if family == "VarExpert" else ("GT",)
                meta = DatasetMeta(
                    kind="volume",
                    n_samples=1,
                    input_dim=4,
                    target_names=target_names,
                    target_dims={name: 1 for name in target_names},
                    volume_shape={"X": 1, "Y": 1, "Z": 1, "T": 1},
                )
                model = build_model(
                    ModelConfig(
                        name=model_payload["name"],
                        params={key: value for key, value in model_payload.items() if key != "name"},
                    ),
                    meta,
                )
                expected_mib = target_mib if family == "VarExpert" else target_mib / 5
                self._assert_size(model, expected_mib, family, size)

        for size, target_mib in SIZES.items():
            payload = yaml.safe_load(
                (self.config_root / "NeuralExpert" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            model, _ = build_neural_model(payload, payload["LOSS"])
            self._assert_size(model, target_mib / 5, "NeuralExpert", size)

            payload = yaml.safe_load(
                (self.config_root / "MC-INR" / size / "ionization.yaml").read_text(encoding="utf-8")
            )
            layout = tuple(
                TargetLayoutEntry(name, index, index + 1, 1)
                for index, name in enumerate(payload["data"]["targets"])
            )
            model = MCINR(
                centroids=np.zeros((12, 3), dtype=np.float32),
                target_layout=layout,
                in_features=4,
                hidden_features=payload["model"]["hidden_features"],
                gfe_layers=payload["model"]["gfe_layers"],
                lfe_layers=payload["model"]["lfe_layers"],
            )
            self._assert_size(model, target_mib, "MC-INR", size)

            payload = yaml.safe_load(
                (self.config_root / "APMGSRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            model = APMGSRN(payload["MODEL"], data_min=-1.0, data_max=1.0, use_tcnn=False)
            self._assert_size(model, target_mib / 5, "APMGSRN", size, multiplier=100)

            payload = yaml.safe_load(
                (self.config_root / "fV-SRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            self._assert_size(TemporalFVSRN(payload["model"]), target_mib / 5, "fV-SRN", size)

            payload = yaml.safe_load(
                (self.config_root / "RMDSRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            self._assert_size(RMDSRN(payload["model"]), target_mib / 5, "RMDSRN", size)

    def test_siren_and_coordnet_formal_size_axes(self):
        siren_expected = {
            "Size082": (3, 168),
            "Size163": (3, 237),
            "Size326": (3, 336),
            "Size652": (4, 412),
            "Size1304": (5, 522),
        }
        coord_expected = {
            "Size082": 15,
            "Size163": 22,
            "Size326": 31,
            "Size652": 43,
            "Size1304": 61,
        }
        for size in SIZES:
            siren = yaml.safe_load(
                (self.config_root / "SIREN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual((siren["hidden_layers"], siren["hidden_features"]), siren_expected[size])
            coordnet = yaml.safe_load(
                (self.config_root / "CoordNet" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual(coordnet["num_res"], 10)
            self.assertEqual(coordnet["init_features"], coord_expected[size])

    def _assert_size(self, model, target_mib, family, size, multiplier=1):
        actual_mib = sum(parameter.numel() for parameter in model.parameters()) * 2 * multiplier / (1024**2)
        self.assertLessEqual(abs(actual_mib - target_mib) / target_mib, 0.05, (family, size, actual_mib))

    def _walk_strings(self, value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from self._walk_strings(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from self._walk_strings(item, f"{prefix}[{index}]")
        elif isinstance(value, str):
            yield prefix, value

    def _looks_like_generated_path(self, value_path: str, value: str) -> bool:
        key = value_path.rsplit(".", 1)[-1]
        return (
            key.endswith("_path")
            or key == "experiment_root"
            or ".targets." in value_path
            or value.startswith("${REPO_ROOT}/")
        )


if __name__ == "__main__":
    unittest.main()
