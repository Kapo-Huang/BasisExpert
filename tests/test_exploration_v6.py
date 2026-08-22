from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import generate_exploration_v6_configs as generator
from scripts import summarize_exploration_v6 as summarizer
from var_expert_inr.ecnr.config import load_config


class ExplorationV6ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "configs_exploration_v6"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_matrix_and_complete_ecnr_smoke_budget(self):
        self.assertEqual(len(self.paths), generator.EXPECTED_TOTAL)
        coverage: set[tuple[str, str]] = set()
        for path in self.paths:
            profile = path.relative_to(self.root).parts[2]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            target = payload["data"]["target"]
            coverage.add((profile, target))
            loaded = load_config(path)
            training = loaded["training"]
            self.assertEqual(training["epochs_per_scale"], 50)
            self.assertEqual(training["batches_per_epoch_budget"], 300)
            self.assertEqual(
                training["primary_sample_budget"],
                3
                * training["epochs_per_scale"]
                * training["batch_size"]
                * training["batches_per_epoch_budget"],
            )
            self.assertEqual(training["pruning_epochs"], [15, 23, 30, 38])
            self.assertTrue(loaded["evaluation"]["run_after_training"])
            self.assertFalse(loaded["evaluation"]["save_predictions"])
        self.assertEqual(
            coverage,
            {(profile, target) for profile in generator.PROFILES for target in generator.TARGETS},
        )


class ExplorationV6SummaryTestCase(unittest.TestCase):
    def test_profile_ranking_uses_paired_control_psnr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            config_root = repo / "configs_exploration_v6"
            run_root = repo / "runs" / "exploration_v6"
            status = repo / "status.tsv"
            status_rows = ["config\tstatus\texit_code\tlog"]
            for profile, improvement in (("official_control", 0.0), ("candidate", 0.5)):
                for index, target in enumerate(summarizer.EXPECTED_TARGETS):
                    exp_id = f"test-{profile}-{target}"
                    config = (
                        config_root
                        / "ECNR"
                        / "official_main"
                        / profile
                        / f"ionization__{target}.yaml"
                    )
                    config.parent.mkdir(parents=True, exist_ok=True)
                    config.write_text(
                        yaml.safe_dump(
                            {"exp_id": exp_id, "data": {"target": target}},
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    status_rows.append(
                        f"{config.relative_to(repo).as_posix()}\tok\t0\ttest.log"
                    )
                    metrics_dir = run_root / exp_id / "run" / "metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    (metrics_dir / f"{exp_id}.json").write_text(
                        json.dumps(
                            {
                                "aggregate": {
                                    "psnr": 30.0 + index + improvement,
                                    "mse": 0.01,
                                    "mae": 0.05,
                                    "model_bytes": 1_000,
                                    "cr": 100.0,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    (metrics_dir / "training_summary.json").write_text(
                        json.dumps({"artifact_path": str(metrics_dir.parent / "artifacts" / "model.ecnr")}),
                        encoding="utf-8",
                    )
                    (metrics_dir / "training_cost.json").write_text(
                        json.dumps({"total_seconds": 10.0, "peak_cuda_memory_bytes": 2_000}),
                        encoding="utf-8",
                    )
            status.write_text("\n".join(status_rows) + "\n", encoding="utf-8")

            output = repo / "exploration_summary.tsv"
            profiles = repo / "profile_summary.tsv"
            attention = repo / "needs_attention.txt"
            attention_count, no_eligible = summarizer.build_summary(
                config_root=config_root,
                status_path=status,
                output_path=output,
                profile_output_path=profiles,
                attention_path=attention,
                repo_root=repo,
                run_root=run_root,
                regression_tolerance_db=1.0,
            )
            self.assertEqual(attention_count, 0)
            self.assertFalse(no_eligible)
            with profiles.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["profile"], "candidate")
            self.assertEqual(rows[0]["selection_rank"], "1")
            self.assertAlmostEqual(float(rows[0]["median_delta_vs_control_db"]), 0.5)


if __name__ == "__main__":
    unittest.main()
