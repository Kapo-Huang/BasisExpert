import json
import tempfile
import unittest
from pathlib import Path

import torch

from var_expert_inr.evaluation.performance import DecodeMeasurement, combine_memory_samples
from var_expert_inr.evaluation.reporting import cache_key, path_fingerprint, write_json, write_metrics_csv


class EvaluationReportingPerformanceTestCase(unittest.TestCase):
    def test_report_serialization_handles_paths_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = write_json(root / "metrics.json", {"path": root, "psnr": float("inf")})
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["psnr"], "Infinity")
            csv_path = write_metrics_csv(root / "metrics.csv", [{"row_type": "performance", "seconds": 1.0}])
            self.assertIn("performance", csv_path.read_text(encoding="utf-8-sig"))

    def test_cache_fingerprint_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.bin"
            path.write_bytes(b"first")
            first = cache_key(path_fingerprint(path))
            path.write_bytes(b"second-version")
            second = cache_key(path_fingerprint(path))
            self.assertNotEqual(first, second)

    def test_cpu_memory_sampler_reports_consistent_peaks(self):
        with DecodeMeasurement(device=torch.device("cpu"), sample_interval_seconds=0.001) as measurement:
            allocation = bytearray(1024 * 1024)
            self.assertEqual(len(allocation), 1024 * 1024)
        combined = combine_memory_samples([measurement.as_dict()])
        self.assertGreaterEqual(combined["cpu_rss_peak_bytes"], combined["cpu_rss_baseline_bytes"])
        self.assertGreaterEqual(combined["cpu_rss_peak_delta_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
