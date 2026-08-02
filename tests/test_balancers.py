import math
import unittest

import torch

from var_expert_inr.training.balancers import MultiAttrDWALoss


class MultiAttrDWALossTestCase(unittest.TestCase):
    @staticmethod
    def _submit_epoch(balancer, a_loss, b_loss):
        balancer({"a": torch.tensor(float(a_loss)), "b": torch.tensor(float(b_loss))})
        return balancer.end_epoch(return_details=True)

    def test_overlapping_window_warmup_and_eta_schedule(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            temperature=2.0,
            total_epochs=30,
            window_size=5,
            warmup_epochs=20,
            eta_max=1.0,
            eta_min=0.1,
        )

        for epoch in range(1, 7):
            details = self._submit_epoch(balancer, epoch, epoch**2)

        self.assertTrue(details["window_ready"])
        self.assertFalse(details["warmup_complete"])
        self.assertFalse(details["dynamic_update_applied"])
        self.assertIsNone(details["eta"])
        self.assertEqual(details["next_weights"], {"a": 1.0, "b": 1.0})
        self.assertAlmostEqual(details["previous_window_loss"]["a"], 3.0)
        self.assertAlmostEqual(details["current_window_loss"]["a"], 4.0)
        self.assertAlmostEqual(details["previous_window_loss"]["b"], 11.0)
        self.assertAlmostEqual(details["current_window_loss"]["b"], 18.0)

        for epoch in range(7, 20):
            details = self._submit_epoch(balancer, epoch, epoch**2)
            self.assertEqual(details["next_weights"], {"a": 1.0, "b": 1.0})

        first_update = self._submit_epoch(balancer, 20, 20**2)
        previous_mean = torch.tensor([
            sum(range(15, 20)) / 5.0,
            sum(epoch**2 for epoch in range(15, 20)) / 5.0,
        ])
        current_mean = torch.tensor([
            sum(range(16, 21)) / 5.0,
            sum(epoch**2 for epoch in range(16, 21)) / 5.0,
        ])
        first_proposed = 2.0 * torch.softmax((current_mean / previous_mean) / 2.0, dim=0)
        self.assertTrue(first_update["warmup_complete"])
        self.assertTrue(first_update["dynamic_update_applied"])
        self.assertEqual(first_update["eta"], 1.0)
        torch.testing.assert_close(
            balancer.current_weights,
            first_proposed,
        )

        second_update = self._submit_epoch(balancer, 21, 21**2)
        previous_mean = current_mean
        current_mean = torch.tensor([
            sum(range(17, 22)) / 5.0,
            sum(epoch**2 for epoch in range(17, 22)) / 5.0,
        ])
        second_proposed = 2.0 * torch.softmax((current_mean / previous_mean) / 2.0, dim=0)
        expected_eta = 0.1 ** (1.0 / 10.0)
        expected_weights = (1.0 - expected_eta) * first_proposed + expected_eta * second_proposed
        self.assertAlmostEqual(second_update["eta"], expected_eta)
        torch.testing.assert_close(balancer.current_weights, expected_weights)

        for epoch in range(22, 31):
            details = self._submit_epoch(balancer, epoch, epoch**2)
        self.assertAlmostEqual(details["eta"], 0.1)
        self.assertAlmostEqual(sum(details["next_weights"].values()), 2.0, places=6)
        torch.testing.assert_close(
            balancer.epoch_loss_history,
            torch.tensor([[float(e), float(e**2)] for e in range(25, 31)]),
        )
        self.assertEqual(int(balancer.epoch_loss_history_count.item()), 6)

    def test_total_epochs_not_beyond_warmup_stays_uniform(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            total_epochs=6,
            window_size=5,
            warmup_epochs=20,
        )
        for epoch in range(1, 7):
            details = self._submit_epoch(balancer, epoch, epoch**2)
        self.assertTrue(details["window_ready"])
        self.assertFalse(details["dynamic_update_applied"])
        self.assertEqual(details["next_weights"], {"a": 1.0, "b": 1.0})

    def test_zero_previous_window_loss_uses_eps(self):
        balancer = MultiAttrDWALoss(
            ["a", "b"],
            total_epochs=10,
            window_size=5,
            warmup_epochs=0,
            eps=1.0e-12,
        )
        for _ in range(5):
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
                "epoch_loss_history",
                "epoch_loss_history_count",
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
            MultiAttrDWALoss(["a"], total_epochs=30, window_size=4)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, window_size=11)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, warmup_epochs=-1)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, eta_min=0.0)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, eta_min=0.6, eta_max=0.5)
        with self.assertRaises(ValueError):
            MultiAttrDWALoss(["a"], total_epochs=30, eta_max=1.1)

        balancer = MultiAttrDWALoss(["a", "b"], total_epochs=30)
        with self.assertRaises(KeyError):
            balancer({"a": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor([1.0]), "b": torch.tensor(1.0)})
        with self.assertRaises(ValueError):
            balancer({"a": torch.tensor(math.inf), "b": torch.tensor(1.0)})


if __name__ == "__main__":
    unittest.main()
