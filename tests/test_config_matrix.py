import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from scripts.main import generate_configs as generate_config_matrix

from var_expert_inr.methods.apmgsrn.model import APMGSRN
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.methods.fv_srn.model import TemporalFVSRN
from var_expert_inr.methods.mc_inr.data import TargetLayoutEntry
from var_expert_inr.methods.mc_inr.model import MCINR
from var_expert_inr.models import build_model
from var_expert_inr.methods.neural_expert.ionization.model_registry import build_model as build_neural_model
from var_expert_inr.methods.rmdsrn.model import RMDSRN


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
}
MOE_RERUN_LIST = Path("main/moe_non_ionization.list")
COMBINED_RUN_LIST = Path("main/neural_expert_non_ionization.list")
COORDNET_MVNET_STSR_RUN_LIST = Path("main/selected_datasets.list")
COMBUSTION_FV_APMG_INSTANTVNR_LIST = Path(
    "main/combustion_fv_apmg_instantvnr.list"
)
COMBUSTION_STSR_MVNET_LIST = Path("main/combustion_stsr_mvnet.list")
COMBUSTION_MINER_ECNR_LIST = Path("main/combustion_miner_ecnr.list")


class ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.config_root = cls.repo_root / "configs"
        cls.main_root = cls.config_root / "main"
        cls.rd_curve_root = cls.config_root / "rd_curve"
        cls.paths = sorted(cls.main_root.rglob("*.yaml")) + sorted(
            cls.rd_curve_root.rglob("*.yaml")
        )

    def read_run_list(self, name: str | Path) -> list[str]:
        list_path = self.repo_root / "scripts" / name
        selected = [
            line.split("#", 1)[0].strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
        ]
        return [line for line in selected if line]

    def test_matrix_contains_exactly_355_configs_and_no_removed_datasets(self):
        self.assertEqual(len(self.paths), 355)
        relative_names = [str(path.relative_to(self.config_root)).lower() for path in self.paths]
        self.assertFalse(any("car" in name or "linkage" in name for name in relative_names))

    def test_generator_preserves_combustion_and_generates_355_configs(self):
        committed_path = self.main_root / "VarExpert" / "combustion_40NH3_1.yaml"
        committed_payload = yaml.safe_load(committed_path.read_text(encoding="utf-8"))
        committed_stsr = yaml.safe_load(
            (self.main_root / "STSR-INR" / "combustion_40NH3_1.yaml").read_text(
                encoding="utf-8"
            )
        )
        committed_moe = {
            path.relative_to(self.main_root / "MoE-INR").as_posix(): yaml.safe_load(
                path.read_text(encoding="utf-8")
            )
            for path in (self.main_root / "MoE-INR").rglob("*.yaml")
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs"
            generated_main = generated_root / "main"
            generated_rd_curve = generated_root / "rd_curve"
            with (
                mock.patch.object(generate_config_matrix, "CONFIGS_ROOT", generated_root),
                mock.patch.object(generate_config_matrix, "MAIN_CONFIGS", generated_main),
                mock.patch.object(generate_config_matrix, "RD_CURVE_CONFIGS", generated_rd_curve),
            ):
                generate_config_matrix.main()
            generated_paths = sorted(generated_root.rglob("*.yaml"))
            generated_payload = yaml.safe_load(
                (generated_main / "VarExpert" / "combustion_40NH3_1.yaml").read_text(
                    encoding="utf-8"
                )
            )
            generated_stsr = yaml.safe_load(
                (generated_main / "STSR-INR" / "combustion_40NH3_1.yaml").read_text(
                    encoding="utf-8"
                )
            )
            generated_moe = {
                path.relative_to(generated_main / "MoE-INR").as_posix(): yaml.safe_load(
                    path.read_text(encoding="utf-8")
                )
                for path in (generated_main / "MoE-INR").rglob("*.yaml")
            }

        self.assertEqual(len(generated_paths), 355)
        self.assertEqual(generated_payload, committed_payload)
        self.assertEqual(generated_stsr, committed_stsr)
        self.assertEqual(generated_moe, committed_moe)

    def test_default_run_list_contains_the_complete_formal_matrix(self):
        selected = self.read_run_list("main/all_configs.list")
        expected = {path.relative_to(self.repo_root).as_posix() for path in self.paths}

        self.assertEqual(len(selected), 355)
        self.assertEqual(set(selected), expected)

    def test_main_and_rd_curve_lists_partition_the_complete_matrix(self):
        all_configs = set(self.read_run_list("main/all_configs.list"))
        main_configs = self.read_run_list("main/configs.list")
        rd_curve_configs = self.read_run_list("rd_curve/configs.list")
        size_marker = "/Size"

        self.assertEqual(len(main_configs), 267)
        self.assertEqual(len(rd_curve_configs), 88)
        self.assertTrue(all(size_marker not in path for path in main_configs))
        self.assertTrue(all(size_marker in path for path in rd_curve_configs))
        self.assertTrue(set(main_configs).isdisjoint(rd_curve_configs))
        self.assertEqual(set(main_configs) | set(rd_curve_configs), all_configs)

    def test_moe_non_ionization_rerun_list_has_exact_main_scope(self):
        selected = self.read_run_list(MOE_RERUN_LIST)
        expected = {
            f"configs/main/MoE-INR/bathymetry__{target}.yaml"
            for target in DATASET_TARGETS["bathymetry"]
        }
        expected.update(
            f"configs/main/MoE-INR/katrina__{target}.yaml"
            for target in DATASET_TARGETS["katrina"]
        )
        expected.update(
            f"configs/main/MoE-INR/{COMBUSTION_DATASET}__{target}.yaml"
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
            f"configs/main/SIREN/{COMBUSTION_DATASET}__{target}.yaml"
            for target in COMBUSTION_TARGETS
        }
        neural_targets = DATASET_TARGETS["bathymetry"] | COMBUSTION_SCALAR_TARGETS
        neural_main_expected = {
            f"configs/main/NeuralExpert/{dataset}__{target}.yaml"
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
            f"configs/main/NeuralExpert/{COMBUSTION_DATASET}__Velocity.yaml",
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
            f"configs/main/CoordNet/{COMBUSTION_DATASET}__{target}.yaml"
            for target in COMBUSTION_TARGETS
        }
        expected = coordnet | {
            "configs/main/MVNet/katrina.yaml",
            "configs/main/STSR-INR/redsea.yaml",
        }
        self.assertEqual(len(selected), 15)
        self.assertEqual(set(selected), expected)

        payload = yaml.safe_load(
            (self.main_root / "STSR-INR" / "redsea.yaml").read_text(
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

    def test_combustion_group_lists_have_exact_scope_and_stage_order(self):
        scalar_filenames = [
            f"combustion_40NH3_1__{target}.yaml"
            for target in sorted(COMBUSTION_SCALAR_TARGETS)
        ]
        fv_apmg_instant = self.read_run_list(COMBUSTION_FV_APMG_INSTANTVNR_LIST)
        expected_fv_apmg_instant = [
            f"configs/main/{family}/{filename}"
            for family in ("fV-SRN", "APMGSRN", "InstantVNR")
            for filename in scalar_filenames
        ]
        self.assertEqual(fv_apmg_instant, expected_fv_apmg_instant)

        self.assertEqual(
            self.read_run_list(COMBUSTION_STSR_MVNET_LIST),
            [
                "configs/main/STSR-INR/combustion_40NH3_1.yaml",
                "configs/main/MVNet/combustion_40NH3_1.yaml",
            ],
        )

        miner_ecnr = self.read_run_list(COMBUSTION_MINER_ECNR_LIST)
        expected_miner_ecnr = [
            f"configs/main/{family}/{filename}"
            for family in ("MINER", "ECNR")
            for filename in scalar_filenames
        ]
        self.assertEqual(miner_ecnr, expected_miner_ecnr)

    def test_stsr_combustion_uses_all_targets_and_mvnet_training_budget(self):
        stsr = yaml.safe_load(
            (self.main_root / "STSR-INR" / "combustion_40NH3_1.yaml").read_text(
                encoding="utf-8"
            )
        )
        mvnet = yaml.safe_load(
            (self.main_root / "MVNet" / "combustion_40NH3_1.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(stsr["data"]["targets"]), COMBUSTION_TARGETS)
        self.assertNotIn("target", stsr["data"])
        self.assertNotIn("coordinate_axes", stsr["data"])
        self.assertEqual(
            stsr["model"],
            {
                "name": "stsr_inr",
                "in_features": 4,
                "init_features": 64,
                "num_res": 5,
                "omega_0": 5.0,
                "embedding_dims": 256,
                "outermost_linear": True,
                "use_global_latent": True,
            },
        )
        self.assertEqual(stsr["training"], mvnet["training"])

        target_dims = {target: 1 for target in COMBUSTION_TARGETS}
        target_dims["Velocity"] = 3
        meta = DatasetMeta(
            kind="volume",
            n_samples=1,
            input_dim=4,
            target_names=tuple(sorted(COMBUSTION_TARGETS)),
            target_dims=target_dims,
            volume_shape=None,
        )
        model_payload = dict(stsr["model"])
        model_name = model_payload.pop("name")
        model = build_model(ModelConfig(name=model_name, params=model_payload), meta)
        self.assertEqual(type(model.backbone).__name__, "STSRINR")
        self.assertEqual(model.backbone.target_dims["Velocity"], 3)

    def test_single_target_default_and_size_coverage(self):
        for family in ("SIREN", "CoordNet", "MoE-INR", "NeuralExpert"):
            for dataset, targets in DATASET_TARGETS.items():
                actual = {
                    path.stem.split("__")[1]
                    for path in (self.main_root / family).glob(f"{dataset}__*.yaml")
                    if "managerpretrain" not in path.stem
                }
                self.assertEqual(actual, targets, (family, dataset))
            combustion = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob(
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
            if family in {"CoordNet", "MoE-INR"}:
                for size in SIZES:
                    actual = {
                        path.stem.split("__")[1]
                        for path in (self.rd_curve_root / family / size).glob("ionization__*.yaml")
                    }
                    self.assertEqual(actual, DATASET_TARGETS["ionization"], (family, size))
            else:
                self.assertFalse((self.rd_curve_root / family).exists())

        for family in ("APMGSRN", "fV-SRN", "RMDSRN"):
            actual = {path.stem.split("__")[1] for path in (self.main_root / family).glob("ionization__*.yaml")}
            self.assertEqual(actual, DATASET_TARGETS["ionization"])
            combustion = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
            }
            self.assertEqual(combustion, COMBUSTION_SCALAR_TARGETS)
            if family == "fV-SRN":
                for size in SIZES:
                    sized = {
                        path.stem.split("__")[1]
                        for path in (self.rd_curve_root / family / size).glob("ionization__*.yaml")
                    }
                    self.assertEqual(sized, DATASET_TARGETS["ionization"])
            else:
                self.assertFalse((self.rd_curve_root / family).exists())

        for family in ("ECNR", "MINER"):
            ionization_targets = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob("ionization__*.yaml")
            }
            self.assertEqual(ionization_targets, DATASET_TARGETS["ionization"])
            combustion = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
            }
            self.assertEqual(combustion, COMBUSTION_SCALAR_TARGETS)
            if family == "MINER":
                for size in SIZES:
                    sized = {
                        path.stem.split("__")[1]
                        for path in (self.rd_curve_root / family / size).glob("ionization__*.yaml")
                    }
                    self.assertEqual(sized, DATASET_TARGETS["ionization"])
            else:
                self.assertFalse((self.rd_curve_root / family).exists())

        for family in ("InstantNGP", "InstantVNR"):
            instant_targets = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob(
                    "ionization__*.yaml"
                )
            }
            self.assertEqual(instant_targets, DATASET_TARGETS["ionization"])
            instant_combustion = {
                path.stem.split("__")[1]
                for path in (self.main_root / family).glob(
                    f"{COMBUSTION_DATASET}__*.yaml"
                )
            }
            self.assertEqual(instant_combustion, COMBUSTION_SCALAR_TARGETS)
            self.assertEqual(
                list((self.main_root / family).glob("bathymetry__*.yaml")),
                [],
            )
            self.assertEqual(
                list((self.main_root / family).glob("katrina__*.yaml")),
                [],
            )

    def test_neural_expert_has_matching_manager_pretrains(self):
        roots = (self.main_root / "NeuralExpert",)
        main_configs = [
            path
            for root in roots
            for path in root.rglob("*.yaml")
            if "managerpretrain" not in path.stem
        ]
        self.assertEqual(len(main_configs), 26)
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
        root = self.main_root / "MVNet"
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
        main_root = self.main_root / "MoE-INR"
        rd_curve_root = self.rd_curve_root / "MoE-INR"
        main_paths = sorted(main_root.glob("*.yaml"))
        paths = [*main_paths, *sorted(rd_curve_root.rglob("*.yaml"))]
        self.assertEqual(len(main_paths), 27)
        self.assertEqual(len(paths), 47)
        self.assertEqual(len(paths) - len(main_paths), 20)
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
            for path in sorted((self.main_root / "MoE-INR").glob(f"{dataset}__*.yaml")):
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
                (self.main_root / "VarExpert" / f"{dataset}.yaml").read_text(encoding="utf-8")
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
            family = path.relative_to(self.config_root).parts[1]
            if family in {"VarExpert", "MVNet", "SIREN", "CoordNet", "MoE-INR", "InstantNGP", "InstantVNR", "MC-INR", "fV-SRN", "ECNR", "MINER", "STSR-INR"}:
                self.assertFalse(payload["evaluation"]["save_predictions"], path)
            if family in {"fV-SRN", "RMDSRN"}:
                self.assertFalse(payload["evaluation"]["run_after_training"], path)
            if family == "APMGSRN":
                self.assertFalse(payload["EVALUATION"]["run_after_training"], path)
            timing = (payload.get("log") or {}).get("timing")
            if timing is not None:
                self.assertFalse(timing["step_window"], path)

    def test_var_expert_balancing_profiles(self):
        var_expert_paths = [
            *(self.main_root / "VarExpert").rglob("*.yaml"),
            *(self.rd_curve_root / "VarExpert").rglob("*.yaml"),
        ]
        for path in var_expert_paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            ema = payload["training"]["multiview_ema_loss"]
            is_size_config = path.is_relative_to(self.rd_curve_root)
            if is_size_config:
                self.assertEqual(ema["beta"], 0.99, path)
                self.assertEqual(ema["w_min"], 0.5, path)
                self.assertEqual(ema["w_max"], 2.0, path)
                self.assertEqual(ema["warmup_steps"], 75000, path)
                self.assertEqual(ema["alpha"], 1.0, path)
            else:
                self.assertEqual(ema["alpha"], 5.0, path)

    def test_selected_var_expert_profiles_drive_formal_size_configs(self):
        expected_var = {
            "Size082": (8, 15, 4),
            "Size163": (8, 22, 4),
            "Size326": (8, 31, 4),
            "Size652": (8, 45, 4),
        }
        for size in SIZES:
            var_model = yaml.safe_load(
                (self.rd_curve_root / "VarExpert" / size / "ionization.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual(
                (var_model["num_experts"], var_model["base_dim"], var_model["top_k"]),
                expected_var[size],
            )


    def test_apmgsrn_exploration_winner_drives_main_configs(self):
        expected_model = {
            "feature_grid_shape": [4, 4, 4],
            "n_features": 14,
            "n_grids": 1,
            "nodes_per_layer": 16,
            "n_layers": 3,
            "requires_padded_feats": True,
        }
        for path in (self.main_root / "APMGSRN").glob("*.yaml"):
            model = yaml.safe_load(path.read_text(encoding="utf-8"))["MODEL"]
            for key, value in expected_model.items():
                self.assertEqual(model[key], value, path)

    def test_instant_ngp_uses_fixed_model_optimizer_and_single_target(self):
        expected_targets = DATASET_TARGETS["ionization"] | COMBUSTION_SCALAR_TARGETS
        actual_targets = set()
        for path in (self.main_root / "InstantNGP").glob("*.yaml"):
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
        for path in (self.main_root / "InstantVNR").glob("*.yaml"):
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
            self.assertEqual(training["loss_type"], "mse")
            self.assertEqual(training["lr"], 1.0e-3)
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
        for path in (self.main_root / "MVNet").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(payload["data"]["targets"]),
                expected_orders[path.stem],
            )
            model = payload["model"]
            self.assertEqual(model["name"], "mvnet")
            self.assertEqual(model["in_features"], 4)
            self.assertEqual(model["hidden_features"], 206)
            self.assertEqual(model["num_residual_blocks"], 10)
            self.assertEqual(model["omega_0"], 30.0)
            self.assertTrue(model["bias"])

            training = payload["training"]
            self.assertEqual(training["epochs"], 300)
            self.assertEqual(training["batch_size"], 2048)
            self.assertEqual(training["gradient_accumulation_steps"], 1)
            self.assertEqual(training["batches_per_epoch_budget"], 1500)
            self.assertEqual(training["sampler"], "budgeted_random")
            self.assertEqual(training["lr"], 1.0e-5)
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
            self.assertEqual(scheduler["step_size"], 40)
            self.assertEqual(scheduler["gamma"], 0.92)

    def test_var_expert_ionization_dwa_config_uses_dwa_not_ema(self):
        payload = yaml.safe_load((self.main_root / "VarExpert" / "ionization_dwa.yaml").read_text(encoding="utf-8"))
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
            family = path.relative_to(self.config_root).parts[1]
            if family == "MVNet" or (
                family == "STSR-INR" and path.stem == COMBUSTION_DATASET
            ):
                training = payload["training"]
                total = (
                    training["batch_size"]
                    * training["batches_per_epoch_budget"]
                    * training["epochs"]
                )
                self.assertEqual(total, 921_600_000, path)
                continue
            if family == "STSR-INR" and path.is_relative_to(self.main_root):
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
            if family == "MINER":
                training = payload["training"]
                is_combustion = COMBUSTION_DATASET in path.name
                is_size_config = path.is_relative_to(self.rd_curve_root)
                self.assertEqual(training["epochs_per_scale"], 500 if is_combustion else 2000, path)
                expected_block_size = 40 if is_size_config else (32 if is_combustion else 16)
                self.assertEqual(payload["model"]["block_size"], expected_block_size, path)
                self.assertNotIn("primary_sample_budget", training)
                continue
            if family in {"VarExpert", "SIREN", "CoordNet", "MoE-INR", "InstantNGP", "InstantVNR", "STSR-INR"}:
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
        for family in ("CoordNet", "MoE-INR", "VarExpert", "STSR-INR"):
            for size, target_mib in SIZES.items():
                joint = family in {"VarExpert", "STSR-INR"}
                filename = "ionization.yaml" if joint else "ionization__GT.yaml"
                payload = yaml.safe_load((self.rd_curve_root / family / size / filename).read_text(encoding="utf-8"))
                model_payload = payload["model"]
                target_names = tuple(payload["data"]["targets"]) if joint else ("GT",)
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
                expected_mib = target_mib if joint else target_mib / 5
                self._assert_size(model, expected_mib, family, size)

        for size, target_mib in SIZES.items():
            payload = yaml.safe_load(
                (self.rd_curve_root / "fV-SRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            self._assert_size(TemporalFVSRN(payload["model"]), target_mib / 5, "fV-SRN", size)

    def test_miner_size_profiles_use_four_scale_two_block_estimate(self):
        expected_hidden = {"Size082": 3, "Size163": 4, "Size326": 6, "Size652": 9}
        expected_mib = {"Size082": 0.9403228759765625, "Size163": 1.5411376953125,
                        "Size326": 3.177642822265625, "Size652": 6.7195892333984375}
        for size, hidden in expected_hidden.items():
            payload = yaml.safe_load(
                (self.rd_curve_root / "MINER" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            model = payload["model"]
            self.assertEqual((model["scales"], model["block_size"], model["hidden_features"]), (4, 40, hidden))
            self.assertEqual((model["hidden_layers"], model["coarse_feature_multiplier"]), (2, 4))
            per_block = lambda width: 2 * width * width + 7 * width + 1
            params_per_timestep = 2 * per_block(4 * hidden) + 6 * per_block(hidden)
            total_mib = params_per_timestep * 100 * 5 * 2 / (1024**2)
            self.assertAlmostEqual(total_mib, expected_mib[size])

    def test_coordnet_and_stsr_formal_size_axes(self):
        coord_expected = {
            "Size082": 15,
            "Size163": 22,
            "Size326": 31,
            "Size652": 43,
        }
        stsr_expected = {"Size082": (20, 80), "Size163": (28, 112),
                         "Size326": (40, 160), "Size652": (56, 224)}
        for size in SIZES:
            coordnet = yaml.safe_load(
                (self.rd_curve_root / "CoordNet" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )["model"]
            self.assertEqual(coordnet["num_res"], 10)
            self.assertEqual(coordnet["init_features"], coord_expected[size])
            coord_training = yaml.safe_load(
                (self.rd_curve_root / "CoordNet" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )["training"]
            self.assertEqual(coord_training["lr"], 1.0e-5)
            stsr = yaml.safe_load(
                (self.rd_curve_root / "STSR-INR" / size / "ionization.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual((stsr["model"]["init_features"], stsr["model"]["embedding_dims"]), stsr_expected[size])
            self.assertEqual(stsr["model"]["num_res"], 10)
            self.assertEqual(stsr["training"]["lr"], 1.0e-5)

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
