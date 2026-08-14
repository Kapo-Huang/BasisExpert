from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import generate_exploration_v4_configs as generator
from scripts import summarize_exploration_v4 as summarizer
from var_expert_inr.config import load_experiment_config
from var_expert_inr.config.schema import ExplorationProbeConfig, TrainingConfig
from var_expert_inr.models.sota.coordnet import CoordNet
from var_expert_inr.rmdsrn.config import load_config as load_rmdsrn_config
from var_expert_inr.rmdsrn.losses import exponential_variance_weight


EXPECTED_WIDTHS = {
    "Size326": {2: 58, 3: 50, 5: 42, 7: 36, 10: 31},
    "Size652": {2: 80, 3: 70, 5: 58, 7: 50, 10: 43},
    "Size1304": {2: 113, 3: 99, 5: 82, 7: 71, 10: 61},
}


def _actual_coordnet_params(width: int, depth: int) -> int:
    model = CoordNet(in_features=4, out_features=1, init_features=width, num_res=depth)
    return sum(parameter.numel() for parameter in model.parameters())


class ExplorationV4ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.root = cls.repo_root / "configs_exploration_v4"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_matrix_coverage_and_unique_identity(self):
        self.assertEqual(len(self.paths), 81)
        counts = {"CoordNet": 0, "RMDSRN": 0}
        exp_ids: set[str] = set()
        destinations: set[Path] = set()
        coverage: set[tuple[str, str, str, str]] = set()
        for path in self.paths:
            family, size, profile = path.relative_to(self.root).parts[:3]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            target = payload["data"]["target"]
            counts[family] += 1
            self.assertEqual(payload["experiment_root"], "${REPO_ROOT}/runs/exploration_v4")
            self.assertEqual(payload["training"]["seed"], 42)
            self.assertEqual(payload["exploration_probe"], generator.PROBE)
            self.assertNotIn(payload["exp_id"], exp_ids)
            self.assertNotIn(path, destinations)
            exp_ids.add(payload["exp_id"])
            destinations.add(path)
            coverage.add((family, size, profile, target))
        self.assertEqual(counts, {"CoordNet": 54, "RMDSRN": 27})
        self.assertEqual(len(coverage), 81)

    def test_generator_rebuilds_only_an_isolated_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs_exploration_v4"
            with mock.patch.object(generator, "CONFIG_ROOT", generated_root):
                counts = generator.generate()
            self.assertEqual(counts, {"CoordNet": 54, "RMDSRN": 27})
            self.assertEqual(len(list(generated_root.rglob("*.yaml"))), 81)

    def test_coordnet_formula_matches_real_models_and_budget(self):
        self.assertEqual(generator.coordnet_widths(), EXPECTED_WIDTHS)
        for size, widths in EXPECTED_WIDTHS.items():
            formal_width = generator.COORD_FORMAL_WIDTHS[size]
            target_params = _actual_coordnet_params(formal_width, 10)
            for depth, width in widths.items():
                actual = _actual_coordnet_params(width, depth)
                self.assertEqual(actual, generator.coordnet_param_count(width, depth))
                self.assertLessEqual(abs(actual - target_params) / target_params, 0.021)

    def test_coordnet_depth_and_control_profiles_are_causal(self):
        for size, widths in EXPECTED_WIDTHS.items():
            formal = yaml.safe_load(
                (self.repo_root / "configs" / "CoordNet" / size / "ionization__GT.yaml").read_text(
                    encoding="utf-8"
                )
            )
            for depth, width in widths.items():
                for target in generator.COORD_TARGETS:
                    payload = yaml.safe_load(
                        (self.root / "CoordNet" / size / f"res{depth}_base_lr" / f"ionization__{target}.yaml").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(payload["model"]["num_res"], depth)
                    self.assertEqual(payload["model"]["init_features"], width)
                    self.assertEqual(payload["training"]["lr"], 5.0e-5)
                    self.assertEqual(payload["training"]["scheduler"], formal["training"]["scheduler"])
                    self.assertNotIn("grad_clip_norm", payload["training"])

        expected = {
            "res10_scaled_lr": (10, 61, 1.25e-5, 0.0),
            "res10_clip": (10, 61, 5.0e-5, 1.0),
            "res5_scaled_lr_clip": (5, 82, 1.25e-5, 1.0),
        }
        for profile, (depth, width, lr, clip) in expected.items():
            for target in generator.COORD_TARGETS:
                payload = yaml.safe_load(
                    (self.root / "CoordNet" / "Size1304" / profile / f"ionization__{target}.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["model"]["num_res"], depth)
                self.assertEqual(payload["model"]["init_features"], width)
                self.assertEqual(payload["training"]["lr"], lr)
                self.assertEqual(payload["training"].get("grad_clip_norm", 0.0), clip)

    def test_rmdsrn_profiles_and_schedule_values(self):
        counts: dict[str, int] = {}
        for path in sorted((self.root / "RMDSRN").rglob("*.yaml")):
            size, profile = path.relative_to(self.root / "RMDSRN").parts[:2]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            counts[profile] = counts.get(profile, 0) + 1
            training = payload["training"]
            self.assertEqual(training["steps"], 75_000)
            self.assertEqual(training["lr_schedule_steps"], 900_000)
            self.assertEqual(training["lambda_schedule_steps"], 900_000)
            self.assertTrue(payload["exploration_probe"]["retain_best_checkpoint"])
            if profile != "schedule900k_lambda10":
                self.assertIn(size, {"Size082", "Size1304"})
        self.assertEqual(
            counts,
            {"schedule900k_lambda0": 6, "schedule900k_lambda1": 6, "schedule900k_lambda10": 15},
        )

        final_lambda = exponential_variance_weight(
            75_000,
            900_000,
            minimum=0.0,
            maximum=10.0,
            growth_rate=500.0,
        )
        final_lr = 1.0e-7 + (0.005 - 1.0e-7) * (
            1.0 + math.cos(math.pi * 75_000 / 900_000)
        ) / 2.0
        self.assertAlmostEqual(final_lambda, 0.01359664, places=8)
        self.assertAlmostEqual(final_lr, 0.00491482, places=8)

    def test_all_configs_load_and_old_defaults_remain_implicit(self):
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            loaded = load_rmdsrn_config(path) if family == "RMDSRN" else load_experiment_config(path)
            probe = loaded["exploration_probe"] if family == "RMDSRN" else loaded.exploration_probe
            enabled = probe["enabled"] if isinstance(probe, dict) else probe.enabled
            self.assertTrue(enabled, path)

        old_path = self.repo_root / "configs_exploration_v3" / "RMDSRN" / "Size082" / "ionization__GT.yaml"
        old = load_rmdsrn_config(old_path)
        self.assertNotIn("lr_schedule_steps", old["training"])
        self.assertNotIn("lambda_schedule_steps", old["training"])
        self.assertNotIn("grad_clip_norm", old["training"])
        self.assertFalse(ExplorationProbeConfig().retain_best_checkpoint)
        self.assertEqual(TrainingConfig().grad_clip_norm, 0.0)

    def test_invalid_clipping_and_short_schedules_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "grad_clip_norm"):
            TrainingConfig(grad_clip_norm=-1.0)

        source = self.root / "RMDSRN" / "Size082" / "schedule900k_lambda10" / "ionization__GT.yaml"
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"
            for field, value in (("grad_clip_norm", -1.0), ("lr_schedule_steps", 74_999), ("lambda_schedule_steps", 1)):
                invalid = json.loads(json.dumps(payload))
                invalid["training"][field] = value
                path.write_text(yaml.safe_dump(invalid, sort_keys=False), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    load_rmdsrn_config(path)


class ExplorationV4SummaryTestCase(unittest.TestCase):
    def test_nonfinite_uncertainty_detail_is_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = Path(tmpdir) / "probe.tsv"
            metrics.write_text(
                "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails\n"
                '50\t50\tGT\t20\t100000\t1\t{"variance_kl":NaN}\n',
                encoding="utf-8",
            )
            trajectory, details, has_nonfinite = summarizer.read_probe(metrics)
            self.assertEqual(trajectory, {50: 20.0})
            self.assertIn(50, details)
            self.assertTrue(has_nonfinite)

    def test_failure_missing_metric_and_profile_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            config_root = repo / "configs_exploration_v4"
            run_root = repo / "runs" / "exploration_v4"
            status_path = repo / "batch" / "status.tsv"
            output = repo / "batch" / "summary.tsv"
            profile_output = repo / "batch" / "profiles.tsv"
            attention = repo / "batch" / "attention.txt"
            status_path.parent.mkdir(parents=True)
            status_rows = ["config\tstatus\texit_code\tlog"]

            for profile, final, status, middle in (
                ("res10_base_lr", 20.0, "ok", None),
                ("res5_base_lr", 21.5, "ok", None),
                ("res3_base_lr", None, "failed", None),
                ("res2_base_lr", 18.0, "ok", 22.0),
            ):
                config = config_root / "CoordNet" / "Size326" / profile / "ionization__GT.yaml"
                config.parent.mkdir(parents=True, exist_ok=True)
                exp_id = f"summary-{profile}"
                config.write_text(
                    yaml.safe_dump(
                        {
                            "exp_id": exp_id,
                            "data": {"target": "GT"},
                            "model": {"init_features": 4, "num_res": 1},
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                relative = config.relative_to(repo).as_posix()
                status_rows.append(f"{relative}\t{status}\t{0 if status == 'ok' else 1}\ttest.log")
                if final is not None:
                    metrics = run_root / exp_id / "run" / "metrics" / "exploration_psnr.tsv"
                    metrics.parent.mkdir(parents=True)
                    middle_row = "" if middle is None else f"25\t50\tGT\t{middle}\t100000\t1\t{{}}\n"
                    metrics.write_text(
                        "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails\n"
                        "5\t50\tGT\t19\t100000\t1\t{}\n"
                        f"{middle_row}"
                        f"50\t50\tGT\t{final}\t100000\t1\t{{}}\n",
                        encoding="utf-8",
                    )
            status_path.write_text("\n".join(status_rows) + "\n", encoding="utf-8")

            attention_count = summarizer.build_summary(
                config_root=config_root,
                status_path=status_path,
                output_path=output,
                profile_output_path=profile_output,
                attention_path=attention,
                repo_root=repo,
                run_root=run_root,
                collapse_threshold_db=1.0,
                minimum_gain_db=0.1,
            )
            self.assertEqual(attention_count, 2)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            failed = next(row for row in rows if row["profile"] == "res3_base_lr")
            self.assertIn("status=failed", failed["attention_reason"])
            self.assertIn("missing_progress_50", failed["attention_reason"])
            with profile_output.open("r", encoding="utf-8", newline="") as handle:
                profiles = list(csv.DictReader(handle, delimiter="\t"))
            res5 = next(row for row in profiles if row["profile"] == "res5_base_lr")
            self.assertEqual(float(res5["median_delta_vs_res10_db"]), 1.5)
            self.assertEqual(res5["candidate_eligible"], "true")
            self.assertEqual(res5["selection_rank"], "1")
            baseline = next(row for row in profiles if row["profile"] == "res10_base_lr")
            self.assertEqual(baseline["selection_rank"], "2")
            failed_profile = next(row for row in profiles if row["profile"] == "res3_base_lr")
            self.assertEqual(failed_profile["candidate_eligible"], "false")
            collapsed = next(row for row in profiles if row["profile"] == "res2_base_lr")
            self.assertEqual(collapsed["catastrophic_runs"], "1")
            self.assertEqual(collapsed["candidate_eligible"], "false")


if __name__ == "__main__":
    unittest.main()
