import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.config import load_experiment_config
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.dc_inr.config import load_config as load_dc_config
from var_expert_inr.mc_inr.config import load_config as load_mc_config
from var_expert_inr.mc_inr.data import TargetLayoutEntry
from var_expert_inr.mc_inr.model import MCINR
from var_expert_inr.models import build_model
from var_expert_inr.neural_expert.config import load_config as load_neural_config
from var_expert_inr.neural_expert.ionization.model_registry import build_model as build_neural_model


TARGETS = {"GT", "H_plus", "H2", "He", "PD"}
SINGLE_SIZE326_MIB = 3.26 / 5
MULTI_SIZE163_MIB = 1.63


class ExplorationV2ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.root = cls.repo_root / "configs_exploration_v2"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_count_unique_ids_roots_and_probes(self):
        self.assertEqual(len(self.paths), 68)
        exp_ids = set()
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_root"], "${REPO_ROOT}/runs/exploration_v2", path)
            self.assertEqual(payload["exploration_probe"]["sample_ratio"], 0.01, path)
            self.assertEqual(payload["exploration_probe"]["max_samples"], 100_000, path)
            self.assertNotIn(payload["exp_id"], exp_ids, path)
            exp_ids.add(payload["exp_id"])

    def test_profile_coverage(self):
        mc_profiles = {path.name for path in (self.root / "MC-INR" / "Size163").iterdir()}
        self.assertEqual(mc_profiles, {"depth3_4", "depth5_6", "depth7_8"})

        dc_profiles = {path.name for path in (self.root / "DC-INR" / "Size163").iterdir()}
        self.assertEqual(dc_profiles, {"eps0p01", "eps0p05", "eps0p10"})
        for profile in dc_profiles:
            paths = list((self.root / "DC-INR" / "Size163" / profile).glob("*.yaml"))
            self.assertEqual({path.stem.split("__")[1] for path in paths}, TARGETS)

        neural_root = self.root / "NeuralExpert" / "Size326"
        self.assertEqual({path.name for path in neural_root.iterdir()}, {"depth1", "depth2", "depth3"})
        for profile in ("depth1", "depth2", "depth3"):
            self.assertEqual(len(list((neural_root / profile).glob("*.yaml"))), 10)

        var_profiles = {path.name for path in (self.root / "VarExpert" / "Size163").iterdir()}
        expected = {"experts8_top3"}
        expected.update(f"experts9_top{top_k}" for top_k in range(1, 10))
        expected.update(f"experts10_top{top_k}" for top_k in range(1, 11))
        self.assertEqual(var_profiles, expected)

    def test_every_config_loads_with_its_runner(self):
        loaders = {
            "MC-INR": load_mc_config,
            "DC-INR": load_dc_config,
            "NeuralExpert": load_neural_config,
        }
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            loaded = loaders.get(family, load_experiment_config)(path)
            probe = loaded.get("exploration_probe") if isinstance(loaded, dict) else loaded.exploration_probe
            enabled = probe["enabled"] if isinstance(probe, dict) else probe.enabled
            self.assertTrue(enabled, path)

    def test_single_and_multivariable_size_budgets_are_separate(self):
        target_names = ("GT", "H_plus", "H2", "He", "PD")
        meta = DatasetMeta(
            kind="volume",
            n_samples=1,
            input_dim=4,
            target_names=target_names,
            target_dims={name: 1 for name in target_names},
            volume_shape={"X": 1, "Y": 1, "Z": 1, "T": 1},
        )
        for config_path in (self.root / "VarExpert" / "Size163").rglob("*.yaml"):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model_payload = payload["model"]
            model = build_model(
                ModelConfig(
                    name=model_payload["name"],
                    params={key: value for key, value in model_payload.items() if key != "name"},
                ),
                meta,
            )
            actual = sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)
            self.assertLessEqual(abs(actual - MULTI_SIZE163_MIB) / MULTI_SIZE163_MIB, 0.05, config_path)

        neural_root = self.root / "NeuralExpert" / "Size326"
        for config_path in neural_root.rglob("*.yaml"):
            if "managerpretrain" in config_path.stem:
                continue
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model, _ = build_neural_model(payload, payload["LOSS"])
            actual = sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)
            self.assertLessEqual(abs(actual - SINGLE_SIZE326_MIB) / SINGLE_SIZE326_MIB, 0.05, config_path)

        for config_path in (self.root / "MC-INR" / "Size163").rglob("*.yaml"):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            layout = tuple(
                TargetLayoutEntry(name, index, index + 1, 1)
                for index, name in enumerate(payload["data"]["targets"])
            )
            model_cfg = payload["model"]
            model = MCINR(
                centroids=np.zeros((12, 3), dtype=np.float32),
                target_layout=layout,
                in_features=4,
                hidden_features=model_cfg["hidden_features"],
                gfe_layers=model_cfg["gfe_layers"],
                lfe_layers=model_cfg["lfe_layers"],
            )
            actual = sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)
            self.assertLessEqual(abs(actual - MULTI_SIZE163_MIB) / MULTI_SIZE163_MIB, 0.05, config_path)

        expected_dc_budget = MULTI_SIZE163_MIB / len(TARGETS)
        for config_path in (self.root / "DC-INR" / "Size163").rglob("*.yaml"):
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["compression"]["target_size_mib"], expected_dc_budget)

    def test_neural_manager_paths_match_main_configs(self):
        neural_root = self.root / "NeuralExpert" / "Size326"
        mains = [path for path in neural_root.rglob("*.yaml") if "managerpretrain" not in path.stem]
        self.assertEqual(len(mains), 15)
        for main_path in mains:
            manager_path = main_path.with_name(f"{main_path.stem}__managerpretrain.yaml")
            self.assertTrue(manager_path.exists(), manager_path)
            main = yaml.safe_load(main_path.read_text(encoding="utf-8"))
            manager = yaml.safe_load(manager_path.read_text(encoding="utf-8"))
            self.assertEqual(main["MODEL"]["manager_pt_path"], manager["MODEL"]["manager_pt_path"])


if __name__ == "__main__":
    unittest.main()
