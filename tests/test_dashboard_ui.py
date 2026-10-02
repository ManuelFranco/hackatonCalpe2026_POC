import unittest
from unittest.mock import patch
from gradio.helpers import special_args
from sae_dashboard.ui import build_demo
from sae_dashboard.session import Session
from sae_dashboard.manifests import DATA_ROOT


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = build_demo()

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_shared_parameters_and_eight_sections(self):
        components = self.demo.config["components"]
        for label in (
            "Seed",
            "Temperature",
            "Max new tokens",
            "Profile tokens",
            "Token aggregation",
        ):
            self.assertEqual(
                sum(c["props"].get("label") == label for c in components), 1
            )
        tabs = [
            c["props"]["label"]
            for c in components
            if c["type"] == "tabitem" and c["props"]["label"][0].isdigit()
        ]
        self.assertEqual(len(tabs), 8)
        self.assertTrue(tabs[0].startswith("0"))
        scope = next(
            c for c in components if c["props"].get("label") == "Profile tokens"
        )
        self.assertEqual(scope["props"]["value"], "all")
        self.assertEqual(self.demo.config["title"], "Hackathon 2026")
        upload = next(
            c for c in components if c["props"].get("label") == "Upload JSON manifests"
        )
        self.assertEqual(upload["props"]["file_types"], [".json"])

    def test_load_callback_clears_all_dependent_outputs(self):
        fn = next(fn for fn in self.demo.fns.values() if fn.name == "load")
        state = Session()
        state.vector_id = "old"
        result = fn.fn(state, str(DATA_ROOT / "cwe_120/manifest.json"))
        self.assertEqual(len(result), len(fn.outputs))
        self.assertIsNone(state.vector_id)
        self.assertIsNone(
            result[
                next(
                    i
                    for i, output in enumerate(fn.outputs)
                    if getattr(output, "label", None) == "Model package"
                )
            ]
        )
        self.assertTrue(all("build a common profile" in value for value in result[-4:]))
        self.assertEqual(len(state.manifests[0].pairs), 20)

    def test_saving_button_only_changes_its_own_session(self):
        fn = next(fn for fn in self.demo.fns.values() if fn.name == "toggle_saving")
        first, second = Session(), Session()
        fn.fn(first)
        self.assertTrue(first.save_enabled)
        self.assertFalse(second.save_enabled)
        fn.fn(first)
        self.assertFalse(first.save_enabled)

    def test_progress_injection_preserves_shared_capture_controls(self):
        fn = next(fn for fn in self.demo.fns.values() if fn.name == "capture")
        values, progress_index, _, _ = special_args(
            fn.fn, [Session(), 42, 0.3, 128, "all", "mean"]
        )
        self.assertIsNotNone(progress_index)
        with patch(
            "sae_dashboard.ui.workflow.build_profile", return_value=[]
        ) as capture:
            output = fn.fn(*values)
        self.assertEqual(len(output), len(fn.outputs))
        controls = capture.call_args.args[1]
        self.assertEqual(
            (controls.seed, controls.temperature, controls.max_new_tokens),
            (42, 0.3, 128),
        )

    def test_builtin_widgets_have_flat_english_overrides(self):
        from sae_dashboard.ui import english_widgets

        translations = english_widgets().translations_dict
        for locale in ("en", "es", "fr", "zh-CN"):
            self.assertEqual(
                translations[locale]["upload_text.drop_file"], "Drop File Here"
            )

    def test_strengths_default_to_zero_and_layers_have_explorers(self):
        components = self.demo.config["components"]
        for layer in (9, 17, 22, 29):
            slider = next(
                c
                for c in components
                if c["type"] == "slider" and c["props"].get("label") == f"Layer {layer}"
            )
            self.assertEqual(slider["props"]["value"], 0)
            explorer = next(
                c
                for c in components
                if c["type"] == "html"
                and c["props"].get("label") == f"Layer {layer} features"
            )
            self.assertIn("data-src", explorer["props"]["js_on_load"])

    def test_vectors_refresh_profile_diagnostics_and_invalidation_clears_them(self):
        import torch
        from research.vector_builders import LayerProfile, VectorResult

        state = Session()
        state.profile[9] = LayerProfile(torch.zeros(2, 2), torch.ones(2, 2), 1)

        def build(state, settings, method):
            state.vectors[9] = VectorResult(torch.tensor([0.0, 1.0]), torch.ones(2))
            return []

        fn = next(fn for fn in self.demo.fns.values() if fn.name == "vectors")
        with patch("sae_dashboard.ui.workflow.create_vectors", side_effect=build):
            result = fn.fn(state, "mean_difference", 0, 0, 256, "all", "mean")
        self.assertEqual(len(result), len(fn.outputs))
        self.assertIn("Features removed by vector method", result[-4])
        invalidate = next(
            fn for fn in self.demo.fns.values() if fn.name == "invalidate"
        )
        output = invalidate.fn(state)
        self.assertEqual(len(output), len(invalidate.outputs))
        self.assertFalse(state.profile)
        self.assertTrue(all("build a common profile" in value for value in output[-4:]))

    def test_preview_callback_shows_ground_truth_before_any_model_call(self):
        from research.benchmarks import BENCHMARKS

        adapter = BENCHMARKS["mmlu_pro"]
        item = {
            "question_id": "preview",
            "question": "What is 2 + 2?",
            "options": ["3", "4"],
            "answer": "B",
            "category": "math",
        }
        fn = next(fn for fn in self.demo.fns.values() if fn.name == "prepare_sample")
        with (
            patch.object(adapter, "load", return_value=[item]),
            patch("sae_dashboard.model_runtime.ensure_models_loaded") as model,
        ):
            result = fn.fn(
                Session(), "mmlu_pro", "test", 1, "", 0, 0, 256, "all", "mean"
            )
        self.assertEqual(len(result), len(fn.outputs))
        self.assertIn("What is 2 + 2?", result[2])
        self.assertIn("B. 4", result[3])
        self.assertIn("math", result[4])
        model.assert_not_called()

    def test_causal_lab_has_zero_dose_and_markdown_responses(self):
        components = self.demo.config["components"]
        dose = next(
            c
            for c in components
            if c["props"].get("label") == "Single-feature response dose"
        )
        self.assertEqual(dose["props"]["value"], 0)
        for label in ("Base · causal lab", "Single feature", "Random control"):
            panel = next(c for c in components if c["props"].get("label") == label)
            self.assertEqual(panel["type"], "markdown")
