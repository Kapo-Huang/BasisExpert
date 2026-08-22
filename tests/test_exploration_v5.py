from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import generate_exploration_v5_configs as generator
from scripts import summarize_exploration_v5 as summarizer
from var_expert_inr.config import load_experiment_config
from var_expert_inr.fv_srn.config import load_config as load_fv_config
from var_expert_inr.fv_srn.model import TemporalFVSRN


class ExplorationV5ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.root = cls.repo_root / "configs_exploration_v5"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_matrix_coverage_and_unique_identity(self):
        self.assertEqual(len(self.paths), 42)
        counts = {"fV-SRN": 0, "InstantVNR": 0}
        coverage: set[tuple[str, str, str, str]] = set()
        exp_ids: set[str] = set()
        for path in self.paths:
            family, structure, profile = path.relative_to(self.root).parts[:3]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            target = payload["data"]["target"]
            counts[family] += 1
            coverage.add((family, structure, profile, target))
            self.assertEqual(
                payload["experiment_root"], "${REPO_ROOT}/runs/exploration_v5"
            )
            self.assertEqual(payload["training"]["seed"], 42)
            self.assertEqual(payload["exploration_probe"], generator.PROBE)
            self.assertNotIn(payload["exp_id"], exp_ids)
            exp_ids.add(payload["exp_id"])
        self.assertEqual(counts, {"fV-SRN": 24, "InstantVNR": 18})
        self.assertEqual(len(coverage), 42)

    def test_generator_rebuilds_only_isolated_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs_exploration_v5"
            sentinel = Path(tmpdir) / "formal-sentinel.yaml"
            sentinel.write_text("keep: true\n", encoding="utf-8")
            with mock.patch.object(generator, "CONFIG_ROOT", generated_root):
                counts = generator.generate()
            self.assertEqual(counts, {"fV-SRN": 24, "InstantVNR": 18})
            self.assertEqual(len(list(generated_root.rglob("*.yaml"))), 42)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep: true\n")

    def test_fv_structures_and_optimizer_factorial(self):
        expected_models = {
            "formal_size163": (6, 64, 32, 3),
            "grid_heavy": (7, 41, 16, 2),
        }
        expected_training = {
            name: (values["lr"], values["lr_step"], values["lr_gamma"])
            for name, values in generator.FV_PROFILES.items()
        }
        param_counts: dict[str, int] = {}
        for structure, model_values in expected_models.items():
            for profile, training_values in expected_training.items():
                for target in generator.TARGETS:
                    path = (
                        self.root
                        / "fV-SRN"
                        / structure
                        / profile
                        / f"ionization__{target}.yaml"
                    )
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    model = payload["model"]
                    self.assertEqual(
                        (
                            model["grid_resolution"],
                            model["grid_channels"],
                            model["hidden_features"],
                            model["hidden_layers"],
                        ),
                        model_values,
                    )
                    training = payload["training"]
                    self.assertEqual(
                        (training["lr"], training["lr_step"], training["lr_gamma"]),
                        training_values,
                    )
                    self.assertEqual(training["epochs"], 50)
                    self.assertFalse(payload["evaluation"]["run_after_training"])

            sample = next((self.root / "fV-SRN" / structure).rglob("*.yaml"))
            model_cfg = yaml.safe_load(sample.read_text(encoding="utf-8"))["model"]
            built = TemporalFVSRN(model_cfg)
            param_counts[structure] = sum(
                int(parameter.numel()) for parameter in built.parameters()
            )
        difference = abs(param_counts["formal_size163"] - param_counts["grid_heavy"])
        self.assertLess(difference / param_counts["formal_size163"], 0.01)

    def test_instant_profiles_keep_model_and_training_budget_fixed(self):
        models: list[dict] = []
        for profile, expected in generator.INSTANT_PROFILES.items():
            for target in generator.TARGETS:
                path = (
                    self.root
                    / "InstantVNR"
                    / generator.INSTANT_STRUCTURE
                    / profile
                    / f"ionization__{target}.yaml"
                )
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                models.append(payload["model"])
                training = payload["training"]
                self.assertEqual(training["epochs"], 50)
                self.assertEqual(training["batches_per_epoch_budget"], 1500)
                self.assertEqual(training["gradient_accumulation_steps"], 4)
                self.assertEqual(training["log_psnr_every"], 5)
                self.assertEqual(training["lr"], expected["lr"])
                self.assertEqual(training["loss_type"], expected["loss_type"])
                self.assertEqual(training["scheduler"]["interval"], "optimizer_step")
                self.assertEqual(
                    training["scheduler"]["gamma"], expected["scheduler_gamma"]
                )
                self.assertEqual(
                    training.get("grad_clip_norm", 0.0),
                    expected.get("grad_clip_norm", 0.0),
                )
        self.assertTrue(all(model == models[0] for model in models))

    def test_every_config_loads_with_its_method_loader(self):
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            loaded = load_fv_config(path) if family == "fV-SRN" else load_experiment_config(path)
            probe = loaded["exploration_probe"] if family == "fV-SRN" else loaded.exploration_probe
            enabled = probe["enabled"] if isinstance(probe, dict) else probe.enabled
            self.assertTrue(enabled, path)


class ExplorationV5SummaryTestCase(unittest.TestCase):
    @staticmethod
    def _write_config_and_metrics(
        *,
        repo: Path,
        family: str,
        structure: str,
        profile: str,
        target: str,
        values: dict[int, float],
        status_rows: list[str],
    ) -> None:
        config = (
            repo
            / "configs_exploration_v5"
            / family
            / structure
            / profile
            / f"ionization__{target}.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        exp_id = f"test-{family}-{structure}-{profile}-{target}".replace("_", "-")
        config.write_text(
            yaml.safe_dump(
                {"exp_id": exp_id, "data": {"target": target}}, sort_keys=False
            ),
            encoding="utf-8",
        )
        status_rows.append(
            f"{config.relative_to(repo).as_posix()}\tok\t0\ttest.log"
        )
        metrics = (
            repo
            / "runs"
            / "exploration_v5"
            / exp_id
            / "run"
            / "metrics"
            / "exploration_psnr.tsv"
        )
        metrics.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails"
        ]
        lines.extend(
            f"{progress}\t50\t{target}\t{value}\t100000\t1\t{{}}"
            for progress, value in sorted(values.items())
        )
        metrics.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _smooth(final: float) -> dict[int, float]:
        return {
            progress: final - 2.0 + 2.0 * index / 9.0
            for index, progress in enumerate(summarizer.EXPECTED_PROGRESS)
        }

    def test_summary_flags_anomalies_and_ranks_complete_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            status_path = repo / "batch" / "status.tsv"
            status_path.parent.mkdir(parents=True)
            status_rows = ["config\tstatus\texit_code\tlog"]

            fv_control = summarizer.FV_REFERENCE_FINAL_PSNR
            for target in summarizer.EXPECTED_TARGETS:
                self._write_config_and_metrics(
                    repo=repo,
                    family="fV-SRN",
                    structure="formal_size163",
                    profile="lr1e2_step100",
                    target=target,
                    values=self._smooth(fv_control[target]),
                    status_rows=status_rows,
                )
                self._write_config_and_metrics(
                    repo=repo,
                    family="fV-SRN",
                    structure="grid_heavy",
                    profile="lr5e3_step20",
                    target=target,
                    values=self._smooth(fv_control[target] + 0.5),
                    status_rows=status_rows,
                )

            instant_control = {"GT": 20.0, "H2": 30.0, "H_plus": 22.0}
            for target in summarizer.EXPECTED_TARGETS:
                self._write_config_and_metrics(
                    repo=repo,
                    family="InstantVNR",
                    structure="official_default",
                    profile="official_control",
                    target=target,
                    values=self._smooth(instant_control[target]),
                    status_rows=status_rows,
                )
                self._write_config_and_metrics(
                    repo=repo,
                    family="InstantVNR",
                    structure="official_default",
                    profile="lr1e3",
                    target=target,
                    values=self._smooth(instant_control[target] + 1.0),
                    status_rows=status_rows,
                )

            bad_fv = self._smooth(fv_control["GT"] - 1.5)
            bad_fv[45] = fv_control["GT"] + 1.0
            self._write_config_and_metrics(
                repo=repo,
                family="fV-SRN",
                structure="formal_size163",
                profile="bad",
                target="GT",
                values=bad_fv,
                status_rows=status_rows,
            )
            missing = self._smooth(31.0)
            missing.pop(25)
            self._write_config_and_metrics(
                repo=repo,
                family="InstantVNR",
                structure="official_default",
                profile="bad",
                target="H2",
                values=missing,
                status_rows=status_rows,
            )
            status_path.write_text("\n".join(status_rows) + "\n", encoding="utf-8")

            output = repo / "batch" / "summary.tsv"
            profiles = repo / "batch" / "profiles.tsv"
            attention = repo / "batch" / "attention.txt"
            result = summarizer.build_summary(
                config_root=repo / "configs_exploration_v5",
                status_path=status_path,
                output_path=output,
                profile_output_path=profiles,
                attention_path=attention,
                repo_root=repo,
                run_root=repo / "runs" / "exploration_v5",
                collapse_threshold_db=1.0,
                minimum_gain_db=0.1,
                fv_reference_tolerance_db=1.0,
            )
            self.assertEqual(result.attention_count, 2)
            self.assertEqual(result.missing_eligible_families, ())

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            bad_fv_row = next(
                row
                for row in rows
                if row["family"] == "fV-SRN" and row["profile"] == "bad"
            )
            self.assertIn("collapse=", bad_fv_row["attention_reason"])
            self.assertIn("historical_regression=", bad_fv_row["attention_reason"])
            missing_row = next(
                row
                for row in rows
                if row["family"] == "InstantVNR" and row["profile"] == "bad"
            )
            self.assertIn("missing_progress=25", missing_row["attention_reason"])

            with profiles.open("r", encoding="utf-8", newline="") as handle:
                profile_rows = list(csv.DictReader(handle, delimiter="\t"))
            fv_winner = next(
                row
                for row in profile_rows
                if row["family"] == "fV-SRN" and row["profile"] == "lr5e3_step20"
            )
            instant_winner = next(
                row
                for row in profile_rows
                if row["family"] == "InstantVNR" and row["profile"] == "lr1e3"
            )
            self.assertEqual(fv_winner["candidate_eligible"], "true")
            self.assertEqual(fv_winner["selection_rank"], "1")
            self.assertAlmostEqual(float(fv_winner["median_delta_vs_control_db"]), 0.5)
            self.assertEqual(instant_winner["candidate_eligible"], "true")
            self.assertEqual(instant_winner["selection_rank"], "1")
            self.assertAlmostEqual(float(instant_winner["median_delta_vs_control_db"]), 1.0)

    def test_nonfinite_values_are_removed_and_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.tsv"
            path.write_text(
                "progress\taggregate_psnr\n5\t20\n10\tnan\n15\t21\n",
                encoding="utf-8",
            )
            trajectory, has_nonfinite = summarizer.read_trajectory(path)
            self.assertEqual(trajectory, {5: 20.0, 15: 21.0})
            self.assertTrue(has_nonfinite)


if __name__ == "__main__":
    unittest.main()
