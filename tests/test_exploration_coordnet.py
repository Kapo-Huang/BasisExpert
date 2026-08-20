from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import generate_exploration_coordnet_configs as generator
from scripts import summarize_exploration_coordnet as summarizer
from var_expert_inr.config import load_experiment_config


class ExplorationCoordNetConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.root = cls.repo_root / "configs_exploration_CoordNet"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_matrix_coverage_and_unique_identity(self):
        self.assertEqual(len(self.paths), generator.EXPECTED_TOTAL)
        coverage: set[tuple[str, str, str]] = set()
        exp_ids: set[str] = set()
        for path in self.paths:
            family, size, profile = path.relative_to(self.root).parts[:3]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            target = payload["data"]["target"]
            self.assertEqual(family, "CoordNet")
            coverage.add((size, profile, target))
            self.assertNotIn(payload["exp_id"], exp_ids)
            exp_ids.add(payload["exp_id"])
        expected = {
            (size, profile, target)
            for size in generator.SIZES
            for profile in generator.LR_PROFILES
            for target in generator.TARGETS
        }
        self.assertEqual(coverage, expected)

    def test_generator_rebuilds_only_isolated_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs_exploration_CoordNet"
            with mock.patch.object(generator, "CONFIG_ROOT", generated_root):
                count = generator.generate()
            self.assertEqual(count, generator.EXPECTED_TOTAL)
            self.assertEqual(len(list(generated_root.rglob("*.yaml"))), 30)

    def test_payloads_preserve_formal_width_and_scheduler(self):
        for path in self.paths:
            _, size, profile, filename = path.relative_to(self.root).parts
            generated = yaml.safe_load(path.read_text(encoding="utf-8"))
            formal = yaml.safe_load(
                (generator.FORMAL_ROOT / size / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(generated["experiment_root"], generator.RUN_ROOT)
            self.assertEqual(generated["model"]["num_res"], 10)
            self.assertEqual(
                generated["model"]["init_features"], formal["model"]["init_features"]
            )
            self.assertEqual(
                generated["training"]["scheduler"], formal["training"]["scheduler"]
            )
            self.assertEqual(
                generated["training"]["lr"], generator.LR_PROFILES[profile]
            )
            self.assertEqual(generated["training"]["epochs"], 50)
            self.assertEqual(generated["training"]["log_psnr_every"], 5)
            self.assertEqual(generated["training"]["save_every"], 50)
            self.assertEqual(generated["training"]["seed"], 42)
            self.assertNotIn("grad_clip_norm", generated["training"])
            self.assertEqual(generated["exploration_probe"], generator.PROBE)

    def test_every_config_loads(self):
        for path in self.paths:
            loaded = load_experiment_config(path)
            self.assertEqual(loaded.model.name, "coordnet")
            self.assertEqual(loaded.training.epochs, 50)
            self.assertTrue(loaded.exploration_probe.enabled)
            self.assertTrue(loaded.exploration_probe.retain_best_checkpoint)


class ExplorationCoordNetSummaryTestCase(unittest.TestCase):
    def test_summary_flags_collapse_and_ranks_clean_learning_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            config_root = repo / "configs_exploration_CoordNet"
            run_root = repo / "runs" / "exploration_CoordNet"
            status = repo / "batch" / "status.tsv"
            summary = repo / "batch" / "exploration_summary.tsv"
            profiles = repo / "batch" / "profile_summary.tsv"
            attention = repo / "batch" / "needs_attention.txt"
            status.parent.mkdir(parents=True)
            status_rows = ["config\tstatus\texit_code\tlog"]

            for profile, learning_rate in generator.LR_PROFILES.items():
                for index, target in enumerate(generator.TARGETS):
                    config = (
                        config_root
                        / "CoordNet"
                        / "Size082"
                        / profile
                        / f"ionization__{target}.yaml"
                    )
                    config.parent.mkdir(parents=True, exist_ok=True)
                    exp_id = f"summary-{profile}-{target}"
                    config.write_text(
                        yaml.safe_dump(
                            {
                                "exp_id": exp_id,
                                "data": {"target": target},
                                "training": {"lr": learning_rate},
                            },
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    status_rows.append(
                        f"{config.relative_to(repo).as_posix()}\tok\t0\ttest.log"
                    )
                    metrics = (
                        run_root / exp_id / "run" / "metrics" / "exploration_psnr.tsv"
                    )
                    metrics.parent.mkdir(parents=True)
                    middle = 30.0 if profile == "lr1e-5" and index == 0 else 20.0
                    final = 20.0 + index + (1.0 if profile == "lr5e-6" else 0.0)
                    metrics.write_text(
                        "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails\n"
                        "5\t50\taggregate\t19\t100000\t1\t{}\n"
                        f"25\t50\taggregate\t{middle}\t100000\t1\t{{}}\n"
                        f"50\t50\taggregate\t{final}\t100000\t1\t{{}}\n",
                        encoding="utf-8",
                    )
            status.write_text("\n".join(status_rows) + "\n", encoding="utf-8")

            count = summarizer.build_summary(
                config_root=config_root,
                status_path=status,
                output_path=summary,
                profile_output_path=profiles,
                attention_path=attention,
                repo_root=repo,
                run_root=run_root,
                collapse_threshold_db=1.0,
                minimum_gain_db=0.1,
            )
            self.assertEqual(count, 1)
            with profiles.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            fast = next(row for row in rows if row["profile"] == "lr1e-5")
            slow = next(row for row in rows if row["profile"] == "lr5e-6")
            self.assertEqual(fast["clean_runs"], "2")
            self.assertEqual(fast["candidate_eligible"], "false")
            self.assertEqual(slow["clean_runs"], "3")
            self.assertEqual(slow["candidate_eligible"], "true")
            self.assertEqual(slow["selection_rank"], "1")


if __name__ == "__main__":
    unittest.main()
