import math
import unittest

import torch

from var_expert_inr.training.balancers import MultiAttrDWALoss


class MultiAttrDWALossTestCase(unittest.TestCase):
    def test_original_dwa_epoch_schedule(self):
        balancer = MultiAttrDWALoss(["a", "b"], temperature=2.0)

        total = balancer({"a": torch.tensor(2.0), "b": torch.tensor(4.0)})
        self.assertAlmostEqual(float(total.item()), 6.0)
        first = balancer.end_epoch(return_details=True)
        self.assertEqual(first["completed_epochs"], 1)
        self.assertEqual(first["next_weights"], {"a": 1.0, "b": 1.0})

        total = balancer({"a": torch.tensor(1.0), "b": torch.tensor(8.0)})
        self.assertAlmostEqual(float(total.item()), 9.0)
        second = balancer.end_epoch(return_details=True)

        ratios = torch.tensor([1.0 / 2.0, 8.0 / 4.0])
        expected_weights = 2.0 * torch.softmax(ratios / 2.0, dim=0)
        self.assertEqual(second["completed_epochs"], 2)
        self.assertAlmostEqual(sum(second["next_weights"].values()), 2.0, places=6)
        self.assertAlmostEqual(second["next_weights"]["a"], float(expected_weights[0].item()), places=6)
        self.assertAlmostEqual(second["next_weights"]["b"], float(expected_weights[1].item()), places=6)

        total = balancer({"a": torch.tensor(3.0), "b": torch.tensor(5.0)})
        expected_total = expected_weights[0] * 3.0 + expected_weights[1] * 5.0
        self.assertAlmostEqual(float(total.item()), float(expected_total.item()), places=6)

    def test_forward_details_and_update_stats_false(self):
        balancer = MultiAttrDWALoss(["a", "b"])
        total, losses_t, weights, details = balancer(
            {"a": torch.tensor(1.5), "b": torch.tensor(2.5), "extra": torch.tensor(9.0)},
            return_details=True,
            return_tensors=True,
            update_stats=False,
        )

        self.assertAlmostEqual(float(total.item()), 4.0)
        torch.testing.assert_close(losses_t, torch.tensor([1.5, 2.5]))
        torch.testing.assert_close(weights, torch.ones(2))
        self.assertEqual(details["method"], "dwa")
        self.assertEqual(details["completed_epochs"], 0)
        self.assertEqual(details["current_epoch_batch_count"], 0)
        self.assertEqual(int(balancer.epoch_batch_count.item()), 0)

    def test_state_dict_contains_all_dwa_buffers(self):
        balancer = MultiAttrDWALoss(["a", "b"])

        self.assertEqual(list(balancer.parameters()), [])
        self.assertEqual(
            set(balancer.state_dict()),
            {
                "current_weights",
                "epoch_loss_sum",
                "last_epoch_loss",
                "second_last_epoch_loss",
                "epoch_batch_count",
                "completed_epochs",
            },
        )

    def test_validates_inputs(self):
        with self.assertRaises(ValueError):
            MultiAttrDWALoss([])
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a", "a"])
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], temperature=0.0)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], eps=0.0)

        balancer = MultiAttrDWALoss(["a", "b"])
        with self.assertRaises(KeyError):
            balancer({"a": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor([1.0]), "b": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor(math.inf), "b": torch.tensor(1.0)})


if __name__ == "__main__":
    unittest.main()
