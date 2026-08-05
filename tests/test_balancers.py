import math
import unittest

import torch

from var_expert_inr.training.balancers import MultiAttrDWALoss


class MultiAttrDWALossTestCase(unittest.TestCase):
    @staticmethod
    def _submit_epoch(balancer, a_loss, b_loss):
        balancer({"a": torch.tensor(float(a_loss)), "b": torch.tensor(float(b_loss))})
        return balancer.end_epoch(return_details=True)

    def test_uses_adjacent_epoch_ratio_and_clips_update_in_log_space(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            temperature=0.2,
            total_epochs=10,
            warmup_epochs=2,
            max_factor_max=1.25,
            max_factor_min=1.25,
        )

        first = self._submit_epoch(balancer, 1.0, 1.0)
        self.assertFalse(first["previous_epoch_available"])
        self.assertFalse(first["dynamic_update_applied"])
        self.assertEqual(first["next_weights"], {"a": 1.0, "b": 1.0})

        second = self._submit_epoch(balancer, 10.0, 1.0)
        expected_ratio = torch.tensor([10.0, 1.0])
        expected_proposal = 2.0 * torch.softmax(expected_ratio / 0.2, dim=0)
        max_log_change = math.log(1.25)
        expected_log_change = torch.clamp(
            torch.log(expected_proposal.clamp_min(1.0e-12)),
            min=-max_log_change,
            max=max_log_change,
        )
        expected_weights = torch.exp(expected_log_change)
        expected_weights = expected_weights / expected_weights.mean()

        self.assertTrue(second["previous_epoch_available"])
        self.assertTrue(second["dynamic_update_applied"])
        self.assertEqual(second["previous_epoch_loss"], {"a": 1.0, "b": 1.0})
        self.assertEqual(second["loss_ratio"], {"a": 10.0, "b": 1.0})
        self.assertAlmostEqual(second["max_factor"], 1.25)
        self.assertAlmostEqual(second["max_log_change"], max_log_change)
        torch.testing.assert_close(
            torch.tensor(list(second["proposed_weights"].values())),
            expected_proposal,
        )
        torch.testing.assert_close(
            torch.tensor(list(second["applied_log_change"].values())),
            expected_log_change,
        )
        torch.testing.assert_close(balancer.current_weights, expected_weights)
        self.assertAlmostEqual(sum(second["next_weights"].values()), 2.0, places=6)

    def test_warmup_holds_uniform_weights_but_keeps_adjacent_epoch_signal(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            total_epochs=10,
            warmup_epochs=3,
        )

        self._submit_epoch(balancer, 1.0, 1.0)
        second = self._submit_epoch(balancer, 2.0, 1.0)
        self.assertFalse(second["dynamic_update_applied"])
        self.assertEqual(second["loss_ratio"], {"a": 2.0, "b": 1.0})
        self.assertEqual(second["next_weights"], {"a": 1.0, "b": 1.0})

        third = self._submit_epoch(balancer, 2.0, 2.0)
        self.assertTrue(third["dynamic_update_applied"])
        self.assertEqual(third["previous_epoch_loss"], {"a": 2.0, "b": 1.0})
        self.assertEqual(third["loss_ratio"], {"a": 1.0, "b": 2.0})

    def test_cosine_schedule_shrinks_to_minimum_factor(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            total_epochs=4,
            warmup_epochs=0,
            max_factor_max=1.25,
            max_factor_min=1.05,
        )

        self._submit_epoch(balancer, 1.0, 1.0)
        second = self._submit_epoch(balancer, 2.0, 1.0)
        third = self._submit_epoch(balancer, 4.0, 1.0)
        fourth = self._submit_epoch(balancer, 8.0, 1.0)

        self.assertAlmostEqual(second["max_factor"], 1.15)
        self.assertGreater(second["max_factor"], third["max_factor"])
        self.assertGreater(third["max_factor"], fourth["max_factor"])
        self.assertAlmostEqual(fourth["max_factor"], 1.05)

    def test_zero_previous_epoch_loss_uses_eps(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            total_epochs=10,
            warmup_epochs=0,
            eps=1.0e-12,
        )
        self._submit_epoch(balancer, 0.0, 1.0)
        details = self._submit_epoch(balancer, 1.0, 1.0)

        self.assertTrue(details["dynamic_update_applied"])
        self.assertTrue(math.isfinite(details["loss_ratio"]["a"]))
        self.assertTrue(all(math.isfinite(value) for value in details["next_weights"].values()))
        self.assertAlmostEqual(sum(details["next_weights"].values()), 2.0, places=6)

    def test_forward_details_and_update_stats_false(self):
        balancer = MultiAttrDWALoss(["a", "b"], total_epochs=30)
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
        self.assertEqual(details["temperature"], 0.2)
        self.assertEqual(details["update_schedule"], "cosine")
        self.assertEqual(details["completed_epochs"], 0)
        self.assertEqual(details["current_epoch_batch_count"], 0)
        self.assertEqual(int(balancer.epoch_batch_count.item()), 0)

    def test_state_dict_contains_all_dwa_buffers(self):
        balancer = MultiAttrDWALoss(["a", "b"], total_epochs=30)

        self.assertEqual(list(balancer.parameters()), [])
        self.assertEqual(
            set(balancer.state_dict()),
            {
                "current_weights",
                "epoch_loss_sum",
                "previous_epoch_loss",
                "epoch_batch_count",
                "completed_epochs",
            },
        )

    def test_validates_inputs(self):
        with self.assertRaises(ValueError):
            MultiAttrDWALoss([], total_epochs=30)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a", "a"], total_epochs=30)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], temperature=0.0, total_epochs=30)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], eps=0.0, total_epochs=30)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=0)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, warmup_epochs=-1)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, max_factor_min=0.99)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(
                ["a"],
                total_epochs=30,
                max_factor_min=1.2,
                max_factor_max=1.1,
            )
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, update_schedule="linear")

        balancer = MultiAttrDWALoss(["a", "b"], total_epochs=30)
        with self.assertRaises(KeyError):
            balancer({"a": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor([1.0]), "b": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor(math.inf), "b": torch.tensor(1.0)})


if __name__ == "__main__":
    unittest.main()
