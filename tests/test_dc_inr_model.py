import unittest

import torch

from var_expert_inr.dc_inr.model import DCINRTiny, adjusted_even_width, dc_inr_parameter_count


class DCINRModelTestCase(unittest.TestCase):
    def test_width_is_rounded_to_even_and_forward_shape_is_scalar(self):
        model = DCINRTiny(5)
        self.assertEqual(model.width, 6)
        coords = torch.randn(7, 4)
        output = model(coords)
        self.assertEqual(tuple(output.shape), (7, 1))

    def test_parameter_formula_matches_actual_module(self):
        model = DCINRTiny(8)
        actual = sum(int(param.numel()) for param in model.parameters())
        self.assertEqual(actual, dc_inr_parameter_count(8))

    def test_adjusted_even_width_respects_minimum(self):
        self.assertEqual(adjusted_even_width(1), 4)
        self.assertEqual(adjusted_even_width(5), 6)
        self.assertEqual(adjusted_even_width(10), 10)


if __name__ == "__main__":
    unittest.main()
