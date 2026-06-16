import unittest

import numpy as np

from var_expert_inr.evaluation.metrics import PSNRAccumulator, psnr


class MetricsTestCase(unittest.TestCase):
    def test_psnr_accumulator_matches_direct_psnr(self):
        gt = np.array(
            [
                [-1.0, -0.5],
                [0.0, 0.5],
                [1.0, 0.25],
            ],
            dtype=np.float32,
        )
        pred = gt + np.array(
            [
                [0.1, -0.1],
                [0.05, 0.0],
                [-0.1, 0.1],
            ],
            dtype=np.float32,
        )
        accumulator = PSNRAccumulator()
        accumulator.update(gt[:2], pred[:2])
        accumulator.update(gt[2:], pred[2:])
        self.assertAlmostEqual(accumulator.compute(), psnr(gt, pred), places=10)


if __name__ == "__main__":
    unittest.main()
