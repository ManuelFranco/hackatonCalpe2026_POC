from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from sae_dashboard import causal_lab, model_runtime as runtime
from sae_dashboard.session import Session, Settings
from research.vector_builders import LayerProfile


class CausalWorkflowTests(unittest.TestCase):
    def test_profile_scale_is_capped_signed_and_does_not_change_profile(self):
        state = Session()
        state.capture_key = Settings().capture_key
        state.profile_id = "current"
        state.profile = {
            9: LayerProfile(
                torch.tensor([[8.0, 2.0], [8.0, 2.0]]),
                torch.tensor([[4.0, 6.0], [4.0, 6.0]]),
                10,
            )
        }
        before = state.profile[9].a.clone()
        with (
            patch.object(runtime, "ensure_models_loaded"),
            patch.dict(
                runtime.saes, {9: SimpleNamespace(W_dec=torch.eye(2))}, clear=True
            ),
        ):
            candidate = causal_lab.select_candidate(
                state, Settings(), "Common profile", 9, 0, 1
            )
            self.assertEqual(candidate["intervention_step"], -0.5)
            random = causal_lab.random_direction(candidate, 5)
            self.assertAlmostEqual(float(random.norm()), 1, places=5)
            torch.testing.assert_close(
                random, causal_lab.random_direction(candidate, 5)
            )
            with self.assertRaises(ValueError):
                causal_lab.select_candidate(state, Settings(), "Manual", 9, 2, 1)
        torch.testing.assert_close(state.profile[9].a, before)

    def test_responses_share_settings_and_never_modify_common_vectors(self):
        state = Session()
        settings = Settings(seed=21, temperature=0.7, max_new_tokens=91)
        candidate = {"layer": 9, "feature_id": 0, "intervention_step": 1}
        with (
            patch.object(causal_lab, "select_candidate", return_value=candidate),
            patch.object(causal_lab, "random_direction", return_value=torch.ones(2)),
            patch.object(
                causal_lab, "generate_feature_report", return_value={"answer": "answer"}
            ) as generate,
        ):
            result = causal_lab.responses(
                state,
                settings,
                "Manual",
                9,
                0,
                1,
                causal_lab.make_case("Q", None, ""),
                0,
                "Every decoding step",
            )
        self.assertEqual(len(result), 3)
        self.assertFalse(state.vectors)
        for call in generate.call_args_list:
            self.assertEqual(call.args[3:5], (0, 91))
            self.assertEqual(call.kwargs["temperature"], 0.7)
            self.assertEqual(call.kwargs["seed"], 21)
