import unittest

from var_expert_inr.evaluation.selection import (
    metrics_require_ground_truth,
    metrics_require_rendering,
    parse_metric_selection,
    parse_name_selection,
    parse_timestep_selection,
)


class EvaluationSelectionTestCase(unittest.TestCase):
    def test_parses_inclusive_mixed_timestep_ranges(self):
        self.assertEqual(
            parse_timestep_selection("0,2:4,6:9:2", 10),
            (0, 2, 3, 4, 6, 8),
        )

    def test_all_selects_every_timestep(self):
        self.assertEqual(parse_timestep_selection("all", 3), (0, 1, 2))

    def test_rejects_invalid_and_out_of_range_timestep(self):
        with self.assertRaises(ValueError):
            parse_timestep_selection("4:2", 5)
        with self.assertRaises(IndexError):
            parse_timestep_selection("5", 5)

    def test_metric_requirements_are_conditional(self):
        self.assertEqual(parse_metric_selection(None), ("psnr",))
        self.assertTrue(metrics_require_ground_truth(("ssim",)))
        self.assertTrue(metrics_require_rendering(("lpips",)))
        self.assertFalse(metrics_require_ground_truth(("decode_time", "memory")))
        self.assertFalse(metrics_require_rendering(("psnr",)))

    def test_target_all_and_explicit_selection(self):
        self.assertIsNone(parse_name_selection("all"))
        self.assertEqual(parse_name_selection("GT,H2,GT"), ("GT", "H2"))


if __name__ == "__main__":
    unittest.main()
