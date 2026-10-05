import unittest
import torch
from research.vector_builders import LayerProfile, VectorResult
from sae_dashboard.feature_explorer import analyze_profile, render_profile


class FeatureExplorerTests(unittest.TestCase):
    def setUp(self):
        # unchanged active, positive common, canceled common, outside, never active
        self.profile = LayerProfile(
            torch.tensor([[2.0, 0.0, 2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 1.0, 0.0, 3.0, 0.0], [2.0, 2.0, 2.0, 0.0, 0.0]]),
            1.0,
        )

    def test_counts_distinguish_activity_overlap_and_actual_removal(self):
        analysis = analyze_profile(self.profile)
        self.assertEqual(analysis.stats["active_a"], 2)
        self.assertEqual(analysis.stats["active_b"], 4)
        self.assertEqual(analysis.stats["active_either"], 4)
        self.assertEqual(analysis.stats["active_both"], 2)
        self.assertEqual(analysis.stats["common"], 2)
        self.assertEqual(analysis.stats["outside"], 1)
        self.assertEqual(analysis.stats["opposing_common"], 1)
        self.assertEqual(analysis.stats["canceled"], 1)
        self.assertEqual(analysis.stats["never_active"], 1)
        self.assertIsNone(analysis.stats["removed"])
        vector = VectorResult(analysis.mean.clone(), torch.zeros(3))
        baseline = analyze_profile(self.profile, vector)
        self.assertEqual(baseline.stats["removed"], 0)
        self.assertEqual(baseline.stats["final_nonzero"], 2)
        vector.feature_delta[3] = 0
        filtered = analyze_profile(self.profile, vector)
        self.assertEqual(filtered.stats["removed"], 1)
        self.assertEqual(filtered.stats["outside"], 1)

    def test_html_has_colors_exact_pair_values_and_lazy_neuronpedia(self):
        # Neuronpedia IDs are dictionary-specific; a five-feature toy dictionary
        # must not be presented as the 16k release.
        profile = LayerProfile(
            torch.nn.functional.pad(self.profile.a, (0, 16384 - 5)),
            torch.nn.functional.pad(self.profile.b, (0, 16384 - 5)),
            self.profile.reference_norm,
        )
        markup = render_profile(
            9, profile, pair_labels=["<script>alert(1)</script>", "second"]
        )
        self.assertIn("signed-value pos", markup)
        self.assertIn("signed-value neg", markup)
        self.assertIn(
            'data-src="https://www.neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-16k/2?',
            markup,
        )
        self.assertNotIn(" src=", markup)
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn(
            '<td>2</td><td>0</td><td><span class="signed-value neg">-2</span>', markup
        )
        self.assertIn("Not built", markup)
        self.assertNotIn("<iframe", render_profile(9, self.profile, neuronpedia=False))
        self.assertNotIn("<iframe", render_profile(9, self.profile))

    def test_empty_contrast_and_invalid_shapes(self):
        markup = render_profile(9, LayerProfile(torch.ones(2, 3), torch.ones(2, 3), 1))
        self.assertIn("No features in this category.", markup)
        with self.assertRaises(ValueError):
            analyze_profile(LayerProfile(torch.zeros(0, 3), torch.zeros(0, 3), 1))
        with self.assertRaises(ValueError):
            analyze_profile(self.profile, VectorResult(torch.zeros(2), torch.zeros(3)))
