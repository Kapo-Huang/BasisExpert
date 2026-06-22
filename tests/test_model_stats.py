import csv
import tempfile
import unittest
from pathlib import Path

import torch

from var_expert_inr.utils.model_stats import (
    build_model_catalog_row,
    collect_model_statistics,
    upsert_model_catalog,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.linear.bias.requires_grad = False
        self.register_parameter("scale", torch.nn.Parameter(torch.ones(4), requires_grad=False))


class ModelStatsTestCase(unittest.TestCase):
    def test_collect_model_statistics_matches_manual_counts(self):
        model = _TinyModel()

        stats = collect_model_statistics(model)

        self.assertEqual(stats["param_count"], 12)
        self.assertEqual(stats["trainable_param_count"], 6)
        self.assertEqual(stats["fp16_size_bytes"], 16)
        self.assertAlmostEqual(stats["fp16_size_mb"], 16 / (1024.0 * 1024.0))

    def test_upsert_model_catalog_deduplicates_and_sorts_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "runs" / "model_size_catalog.csv"

            inserted_siren = upsert_model_catalog(
                catalog_path,
                build_model_catalog_row(
                    model_name="siren",
                    model_params={"hidden_features": 8, "hidden_layers": 1},
                    stats={
                        "param_count": 10,
                        "trainable_param_count": 10,
                        "fp16_size_bytes": 20,
                        "fp16_size_mb": 20 / (1024.0 * 1024.0),
                    },
                ),
            )
            duplicate_siren = upsert_model_catalog(
                catalog_path,
                build_model_catalog_row(
                    model_name="siren",
                    model_params={"hidden_features": 8, "hidden_layers": 1},
                    stats={
                        "param_count": 10,
                        "trainable_param_count": 10,
                        "fp16_size_bytes": 20,
                        "fp16_size_mb": 20 / (1024.0 * 1024.0),
                    },
                ),
            )
            inserted_var_expert = upsert_model_catalog(
                catalog_path,
                build_model_catalog_row(
                    model_name="var_expert",
                    model_params={"base_dim": 2, "num_experts": 3},
                    stats={
                        "param_count": 42,
                        "trainable_param_count": 40,
                        "fp16_size_bytes": 64,
                        "fp16_size_mb": 64 / (1024.0 * 1024.0),
                    },
                ),
            )

            self.assertTrue(inserted_siren)
            self.assertFalse(duplicate_siren)
            self.assertTrue(inserted_var_expert)

            with catalog_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["model_name"] for row in rows], ["siren", "var_expert"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["hidden_features"], "8")
            self.assertEqual(rows[0]["hidden_layers"], "1")
            self.assertEqual(rows[1]["base_dim"], "2")
            self.assertEqual(rows[1]["num_experts"], "3")


if __name__ == "__main__":
    unittest.main()


