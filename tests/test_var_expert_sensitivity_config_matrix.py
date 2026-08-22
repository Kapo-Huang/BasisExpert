import unittest
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from scripts.sensitivity.generate_var_expert_num import EXPERT_PROFILES
from scripts.sensitivity.generate_var_expert_topk import BASE_DIM, NUM_EXPERTS, TOP_K_VALUES
from var_expert_inr.config import load_experiment_config
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model
from var_expert_inr.models.proposed.shared_enc_inr import SharedEncINR


TARGETS = ("GT", "H_plus", "H2", "He", "PD")
SIZE163_MIB = 1.63
SIZE_TOLERANCE = 0.05


class VarExpertSensitivityConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.expert_root = cls.repo_root / "configs/sensitivity/var_expert_num"
        cls.topk_root = cls.repo_root / "configs/sensitivity/var_expert_topk"
        cls.expert_paths = sorted(cls.expert_root.rglob("*.yaml"))
        cls.topk_paths = sorted(cls.topk_root.rglob("*.yaml"))
        cls.meta = DatasetMeta(
            kind="volume",
            n_samples=1,
            input_dim=4,
            target_names=TARGETS,
            target_dims={name: 1 for name in TARGETS},
            volume_shape={"X": 1, "Y": 1, "Z": 1, "T": 1},
        )

    @staticmethod
    def _payload(path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @classmethod
    def _build_model(cls, path: Path):
        model_payload = cls._payload(path)["model"]
        return build_model(
            ModelConfig(
                name=model_payload["name"],
                params={key: value for key, value in model_payload.items() if key != "name"},
            ),
            cls.meta,
        )

    def test_exact_counts_profiles_and_unique_ids(self):
        self.assertEqual(len(self.expert_paths), 8)
        self.assertEqual(len(self.topk_paths), 7)

        expected_expert_profiles = {
            (
                "experts1_shared_enc"
                if experts == 1
                else f"experts{experts}_top{values['top_k']}"
            )
            for experts, values in EXPERT_PROFILES.items()
        }
        expected_topk_profiles = {f"experts{NUM_EXPERTS}_top{top_k}" for top_k in TOP_K_VALUES}
        self.assertEqual(
            {path.parent.name for path in self.expert_paths},
            expected_expert_profiles,
        )
        self.assertEqual(
            {path.parent.name for path in self.topk_paths},
            expected_topk_profiles,
        )

        exp_ids = [self._payload(path)["exp_id"] for path in self.expert_paths + self.topk_paths]
        self.assertEqual(len(exp_ids), len(set(exp_ids)))

    def test_every_config_loads_and_uses_its_study_root(self):
        for path in self.expert_paths:
            loaded = load_experiment_config(path)
            self.assertTrue(loaded.exploration_probe.enabled, path)
            payload = self._payload(path)
            self.assertEqual(
                payload["experiment_root"],
                "${REPO_ROOT}/runs/sensitivity/var_expert_num",
                path,
            )
        for path in self.topk_paths:
            loaded = load_experiment_config(path)
            self.assertTrue(loaded.exploration_probe.enabled, path)
            payload = self._payload(path)
            self.assertEqual(
                payload["experiment_root"],
                "${REPO_ROOT}/runs/sensitivity/var_expert_topk",
                path,
            )

    def test_expert_count_matrix_uses_shared_encoder_then_budget_matched_var_experts(self):
        actual = {}
        for path in self.expert_paths:
            model = self._payload(path)["model"]
            if model["name"] == "shared_enc_inr":
                actual[1] = {
                    "name": model["name"],
                    "base_dim": int(model["base_dim"]),
                }
                self.assertNotIn("num_experts", model, path)
                self.assertNotIn("top_k", model, path)
                pretrain = self._payload(path)["training"]["pretrain"]
                self.assertFalse(pretrain["enabled"], path)
                self.assertNotIn("assignments_cache_path", pretrain, path)
            else:
                actual[int(model["num_experts"])] = {
                    "name": model["name"],
                    "base_dim": int(model["base_dim"]),
                    "top_k": int(model["top_k"]),
                }
        self.assertEqual(actual, EXPERT_PROFILES)

    def test_topk_matrix_fixes_expert_count_and_width(self):
        actual_top_k = set()
        for path in self.topk_paths:
            model = self._payload(path)["model"]
            self.assertEqual(model["num_experts"], NUM_EXPERTS, path)
            self.assertEqual(model["base_dim"], BASE_DIM, path)
            actual_top_k.add(int(model["top_k"]))
        self.assertEqual(actual_top_k, set(TOP_K_VALUES))

    def test_each_study_changes_only_its_controlled_model_fields(self):
        def normalized(path: Path, *, expert_count_study: bool) -> dict:
            payload = deepcopy(self._payload(path))
            for key in ("experiment", "exp_id", "experiment_root"):
                payload.pop(key)
            if expert_count_study:
                payload["model"].pop("top_k")
                payload["model"].pop("num_experts")
                payload["model"].pop("base_dim")
                payload["training"]["pretrain"].pop("assignments_cache_path")
            else:
                payload["model"].pop("top_k")
            return payload

        expert_controls = {
            yaml.safe_dump(normalized(path, expert_count_study=True), sort_keys=True)
            for path in self.expert_paths
            if self._payload(path)["model"]["name"] == "var_expert"
        }
        topk_controls = {
            yaml.safe_dump(normalized(path, expert_count_study=False), sort_keys=True)
            for path in self.topk_paths
        }
        self.assertEqual(len(expert_controls), 1)
        self.assertEqual(len(topk_controls), 1)

    def test_all_models_stay_within_size163_budget_tolerance(self):
        for path in self.expert_paths + self.topk_paths:
            model = self._build_model(path)
            actual_mib = sum(parameter.numel() for parameter in model.parameters()) * 2 / (1024**2)
            relative_error = abs(actual_mib - SIZE163_MIB) / SIZE163_MIB
            self.assertLessEqual(relative_error, SIZE_TOLERANCE, path)

    def test_one_expert_control_uses_shared_encoder_without_gating(self):
        path = next(
            path
            for path in self.expert_paths
            if self._payload(path)["model"]["name"] == "shared_enc_inr"
        )
        model = self._build_model(path)
        outputs, aux = model(torch.rand(2, 4), return_aux=True)

        self.assertIsInstance(model.backbone, SharedEncINR)
        self.assertFalse(hasattr(model, "gating"))
        self.assertEqual(set(outputs), set(TARGETS))
        self.assertNotIn("masks", aux)
        self.assertNotIn("probs", aux)
        self.assertEqual(tuple(aux["H_views"].shape[:2]), (2, len(TARGETS)))


if __name__ == "__main__":
    unittest.main()
