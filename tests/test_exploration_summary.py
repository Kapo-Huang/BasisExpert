import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_size_exploration.py"
SPEC = importlib.util.spec_from_file_location("summarize_size_exploration", SCRIPT_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


class ExplorationSummaryTestCase(unittest.TestCase):
    def test_summary_uses_latest_status_and_reports_nonfinite_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "configs_exploration"
            config_path = config_root / "SIREN" / "Size163" / "depth3" / "ionization__GT.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "exp_id": "explore-siren-size163-depth3-GT",
                        "experiment_root": "${REPO_ROOT}/runs/exploration",
                        "data": {"target": "GT"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            relative = config_path.relative_to(root).as_posix()
            status_path = root / "batch_logs" / "status.tsv"
            status_path.parent.mkdir(parents=True)
            status_path.write_text(
                "config\tstatus\texit_code\tlog\n"
                f"{relative}\trunning\t\tfirst.log\n"
                f"{relative}\tfailed\t1\tfirst.log\n"
                f"{relative}\trunning\t\tsecond.log\n"
                f"{relative}\tok\t0\tsecond.log\n",
                encoding="utf-8",
            )
            metrics = root / "runs" / "exploration" / "explore-siren-size163-depth3-GT" / "20260803_120000" / "metrics" / "exploration_psnr.tsv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails\n"
                "5\t50\taggregate\t10\t100\t0.1\t{}\n"
                "10\t50\taggregate\tnan\t100\t0.1\t{}\n"
                "10\t50\tsecond-scope\t12\t100\t0.1\t{}\n",
                encoding="utf-8",
            )
            output = root / "batch_logs" / "exploration_summary.tsv"

            SUMMARY.build_summary(config_root, status_path, output, root)

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["training_status"], "ok")
            self.assertEqual(rows[0]["trajectory"], "5:10,10:12")
            self.assertEqual(rows[0]["final_psnr"], "12")
            self.assertEqual(rows[0]["has_nan_or_inf"], "true")
            self.assertEqual(rows[0]["scope_count"], "2")


if __name__ == "__main__":
    unittest.main()
