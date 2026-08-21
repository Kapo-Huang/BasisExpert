import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.apmgsrn.model import APMGSRN
from var_expert_inr.apmgsrn.config import load_config as load_apmg_config
from var_expert_inr.config import load_experiment_config
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.fv_srn.model import TemporalFVSRN
from var_expert_inr.fv_srn.config import load_config as load_fv_config
from var_expert_inr.mc_inr.config import load_config as load_mc_config
from var_expert_inr.mc_inr.data import TargetLayoutEntry
from var_expert_inr.mc_inr.model import MCINR
from var_expert_inr.models import build_model
from var_expert_inr.neural_expert.ionization.model_registry import build_model as build_neural_model
from var_expert_inr.neural_expert.config import load_config as load_neural_config
from var_expert_inr.rmdsrn.config import load_config as load_rm_config
from var_expert_inr.rmdsrn.model import RMDSRN


TARGETS = {"GT", "H_plus", "H2", "He", "PD"}
SINGLE_TARGET_MIB = 1.63 / 5
MULTI_TARGET_MIB = 1.63


class ExplorationConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.root = cls.repo_root / "configs_exploration"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_count_root_and_probe(self):
        self.assertEqual(len(self.paths), 126)
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_root"], "${REPO_ROOT}/runs/exploration", path)
            self.assertEqual(
                payload["exploration_probe"],
                {
                    "enabled": True,
                    "total_epoch_equivalents": 50,
                    "every_epoch_equivalents": 5,
                    "sample_ratio": 0.01,
                    "max_samples": 100_000,
                    "seed": 42,
                },
                path,
            )

    def test_profile_and_target_coverage(self):
        expected_profiles = {
            "SIREN": {"depth2", "depth3", "depth5"},
            "CoordNet": {"res5", "res10", "res15"},
            "MoE-INR": {"experts4", "experts7", "experts10"},
            "VarExpert": {"experts4", "experts6", "experts8"},
            "MC-INR": {"depth3_4", "depth5_6", "depth7_8"},
            "NeuralExpert": {"depth1", "depth2", "depth3"},
            "APMGSRN": {"grid_heavy", "balanced", "decoder_heavy"},
            "fV-SRN": {"grid_heavy", "balanced", "decoder_heavy"},
            "RMDSRN": {"grid_heavy", "balanced", "decoder_heavy"},
        }
        for family, profiles in expected_profiles.items():
            actual = {path.name for path in (self.root / family / "Size163").iterdir() if path.is_dir()}
            self.assertEqual(actual, profiles, family)
            for profile in profiles:
                paths = list((self.root / family / "Size163" / profile).glob("*.yaml"))
                if family in {"VarExpert", "MC-INR"}:
                    self.assertEqual(len(paths), 1, (family, profile))
                elif family == "NeuralExpert":
                    self.assertEqual(len(paths), 10, (family, profile))
                else:
                    self.assertEqual({path.stem.split("__")[1] for path in paths}, TARGETS, (family, profile))

    def test_every_exploration_config_loads_with_its_runner(self):
        loaders = {
            "MC-INR": load_mc_config,
            "NeuralExpert": load_neural_config,
            "APMGSRN": load_apmg_config,
            "fV-SRN": load_fv_config,
            "RMDSRN": load_rm_config,
        }
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            loader = loaders.get(family, load_experiment_config)
            loaded = loader(path)
            probe = loaded.get("exploration_probe") if isinstance(loaded, dict) else loaded.exploration_probe
            self.assertIsNotNone(probe, path)
            enabled = probe["enabled"] if isinstance(probe, dict) else probe.enabled
            self.assertTrue(enabled, path)

    def test_neural_manager_paths_match_one_to_one(self):
        root = self.root / "NeuralExpert" / "Size163"
        mains = [path for path in root.rglob("*.yaml") if "managerpretrain" not in path.stem]
        self.assertEqual(len(mains), 15)
        manager_paths = set()
        for main_path in mains:
            manager_path = main_path.with_name(f"{main_path.stem}__managerpretrain.yaml")
            self.assertTrue(manager_path.exists(), main_path)
            main = yaml.safe_load(main_path.read_text(encoding="utf-8"))
            manager = yaml.safe_load(manager_path.read_text(encoding="utf-8"))
            self.assertEqual(main["TRAINING"]["n_points"], 16000, main_path)
            self.assertEqual(main["TRAINING"]["num_epochs"], 60000, main_path)
            self.assertEqual(main["TRAINING"]["log_every"], 6000, main_path)
            self.assertEqual(main["TRAINING"]["save_every"], 60000, main_path)
            self.assertEqual(manager["TRAINING"]["n_points"], 16000, manager_path)
            self.assertEqual(manager["TRAINING"]["num_epochs"], 2500, manager_path)
            self.assertEqual(manager["TRAINING"]["log_every"], 100, manager_path)
            self.assertEqual(manager["TRAINING"]["save_every"], 2500, manager_path)
            self.assertEqual(main["MODEL"]["manager_pt_path"], manager["MODEL"]["manager_pt_path"])
            manager_paths.add(main["MODEL"]["manager_pt_path"])
        self.assertEqual(len(manager_paths), 15)

    def test_profiles_have_distinct_structures_and_equal_fp16_budgets(self):
        families = ("SIREN", "CoordNet", "MoE-INR", "VarExpert", "MC-INR", "NeuralExpert", "APMGSRN", "fV-SRN", "RMDSRN")
        for family in families:
            profile_sizes = []
            profile_models = []
            for profile_dir in sorted((self.root / family / "Size163").iterdir()):
                config_path = next(path for path in profile_dir.glob("*.yaml") if "managerpretrain" not in path.stem)
                payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                model, signature, multiplier = self._build(family, payload)
                profile_sizes.append(sum(parameter.numel() for parameter in model.parameters()) * 2 * multiplier / (1024**2))
                profile_models.append(signature)
            target = MULTI_TARGET_MIB if family in {"VarExpert", "MC-INR"} else SINGLE_TARGET_MIB
            with self.subTest(family=family, sizes=profile_sizes):
                self.assertEqual(len(set(profile_models)), 3)
                for actual in profile_sizes:
                    self.assertLessEqual(abs(actual - target) / target, 0.05)

    def _build(self, family, payload):
        if family in {"SIREN", "CoordNet", "MoE-INR", "VarExpert"}:
            model_payload = payload["model"]
            targets = tuple(payload["data"]["targets"]) if family == "VarExpert" else (payload["data"]["target"],)
            meta = DatasetMeta(
                kind="volume",
                n_samples=1,
                input_dim=4,
                target_names=targets,
                target_dims={name: 1 for name in targets},
                volume_shape={"X": 1, "Y": 1, "Z": 1, "T": 1},
            )
            model = build_model(
                ModelConfig(name=model_payload["name"], params={key: value for key, value in model_payload.items() if key != "name"}),
                meta,
            )
            return model, repr(sorted(model_payload.items())), 1
        if family == "MC-INR":
            names = tuple(payload["data"]["targets"])
            layout = tuple(TargetLayoutEntry(name, index, index + 1, 1) for index, name in enumerate(names))
            model_payload = payload["model"]
            model = MCINR(
                centroids=np.zeros((12, 3), dtype=np.float32),
                target_layout=layout,
                in_features=4,
                hidden_features=model_payload["hidden_features"],
                gfe_layers=model_payload["gfe_layers"],
                lfe_layers=model_payload["lfe_layers"],
            )
            return model, (model_payload["gfe_layers"], model_payload["lfe_layers"], model_payload["hidden_features"]), 1
        if family == "NeuralExpert":
            model, _ = build_neural_model(payload, payload["LOSS"])
            model_payload = payload["MODEL"]
            return model, (model_payload["decoder_n_hidden_layers"], model_payload["manager_n_hidden_layers"], model_payload["decoder_hidden_dim"]), 1
        if family == "APMGSRN":
            model = APMGSRN(payload["MODEL"], data_min=-1.0, data_max=1.0, use_tcnn=False)
            signature = tuple(str(payload["MODEL"][key]) for key in ("feature_grid_shape", "n_grids", "n_features", "nodes_per_layer", "n_layers"))
            return model, signature, 100
        if family == "fV-SRN":
            model_payload = payload["model"]
            return TemporalFVSRN(model_payload), tuple(model_payload[key] for key in ("grid_resolution", "grid_channels", "hidden_features", "hidden_layers")), 1
        if family == "RMDSRN":
            model_payload = payload["model"]
            return RMDSRN(model_payload), tuple(model_payload[key] for key in ("grid_resolution", "grid_channels", "decoder_hidden_features", "decoder_hidden_layers")), 1
        raise AssertionError(f"Unsupported family {family}")


if __name__ == "__main__":
    unittest.main()
