import unittest
import torch
from research.vector_builders import LayerProfile, BUILDERS


class VectorTests(unittest.TestCase):
    def test_difference_retains_features_missing_in_other_pairs(self):
        a = torch.zeros(2, 3)
        b = torch.tensor([[2.0, 0.0, -1.0], [0.0, 4.0, -3.0]])
        result = BUILDERS["mean_difference"].build(
            LayerProfile(a, b, 10), torch.eye(3), 0.05
        )
        torch.testing.assert_close(result.feature_delta, torch.tensor([1.0, 2.0, -2.0]))
        self.assertAlmostEqual(float(result.direction.norm()), 0.5)
        torch.testing.assert_close(
            b, torch.tensor([[2.0, 0.0, -1.0], [0.0, 4.0, -3.0]])
        )

    def test_zero_and_cancellation_remain_zero(self):
        for b in (torch.zeros(2, 2), torch.tensor([[1.0, 2.0], [-1.0, -2.0]])):
            result = BUILDERS["mean_difference"].build(
                LayerProfile(torch.zeros(2, 2), b, 10), torch.eye(2), 0.05
            )
            self.assertEqual(float(result.direction.norm()), 0)

    def test_invalid_inputs_fail_without_producing_vector(self):
        builder = BUILDERS["mean_difference"]
        for a, b, decoder in (
            (torch.zeros(0, 2), torch.zeros(0, 2), torch.eye(2)),
            (torch.zeros(1, 2), torch.ones(2, 2), torch.eye(2)),
            (torch.zeros(1, 2), torch.full((1, 2), float("nan")), torch.eye(2)),
            (torch.zeros(1, 2), torch.ones(1, 2), torch.eye(3)),
        ):
            with self.assertRaises(ValueError):
                builder.build(LayerProfile(a, b, 1), decoder, 0.05)
