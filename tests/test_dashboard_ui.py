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
        result = fn.fn(state, str(DATA_ROOT / "scripts/cwe_120/manifest.json"))
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

    def test_pope_selection_and_image_preview(self):
        from PIL import Image
        from research.benchmarks import BENCHMARKS

        state = Session()
        select = next(
            fn for fn in self.demo.fns.values() if fn.name == "change_benchmark"
        )
        result = select.fn(state, "pope")
        self.assertEqual(len(result), len(select.outputs))
        self.assertEqual(result[0]["value"], "random")
        self.assertEqual(result[0]["choices"], ["random", "popular", "adversarial"])
        self.assertFalse(result[1]["interactive"])
        self.assertTrue(select.fn(state, "mmmu")[1]["interactive"])
        image = Image.new("RGB", (4, 4), "blue")
        item = {
            "id": "preview",
            "question": "Is there a cat in the image?",
            "answer": "no",
            "image": image,
            "category": "random",
        }
        prepare = next(
            fn for fn in self.demo.fns.values() if fn.name == "prepare_sample"
        )
        with (
            patch.object(BENCHMARKS["pope"], "load", return_value=[item]),
            patch("sae_dashboard.model_runtime.ensure_models_loaded") as model,
        ):
            result = prepare.fn(
                state, "pope", "random", 1, "", 0, 0, 256, "all", "mean"
            )
        self.assertEqual(len(result), len(prepare.outputs))
        self.assertIn(item["question"], result[2])
        self.assertEqual(result[3], "### Correct answer / requirements\n\nno")
        self.assertTrue(result[5]["visible"])
        self.assertEqual(result[5]["value"], [(image, "Image 1")])
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

    def test_optional_manual_editor_uses_shared_input_and_invalidation_outputs(self):
        components = self.demo.config["components"]
        accordion = next(
            c
            for c in components
            if c["type"] == "accordion"
            and c["props"].get("label") == "Manual image–text pairs (optional)"
        )
        self.assertFalse(accordion["props"]["open"])
        state = Session()
        state.profile_id, state.vector_id = "old", "old"
        add = next(fn for fn in self.demo.fns.values() if fn.name == "add_manual_pair")
        output = add.fn(state, "Reference", None, "Target", None)
        self.assertEqual(len(output), len(add.outputs))
        self.assertIn("1 manual", output[0])
        self.assertIn("<td>Manual pairs</td>", output[1])
        self.assertIsNone(state.profile_id)
        self.assertIsNone(state.vector_id)
        select = next(
            fn for fn in self.demo.fns.values() if fn.name == "select_manual_pair"
        )
        selected = select.fn(state, "Pair 1")
        self.assertEqual(selected[:4], ("Reference", None, "Target", None))
        update = next(
            fn for fn in self.demo.fns.values() if fn.name == "update_manual_pair"
        )
        self.assertEqual(
            len(update.fn(state, "Pair 1", "Edited", None, "Target", None)),
            len(update.outputs),
        )
        remove = next(
            fn for fn in self.demo.fns.values() if fn.name == "remove_manual_pair"
        )
        self.assertEqual(len(remove.fn(state, "Pair 1")), len(remove.outputs))
        self.assertFalse(state.manifests)

    def test_input_summary_includes_all_sources_and_escapes_names(self):
        from sae_dashboard.manual_input_ui import input_summary

        summary = input_summary(
            [["<script>JSON</script>", 20, "abc"], ["Manual pairs", 1, "def"]]
        )
        self.assertIn("&lt;script&gt;JSON&lt;/script&gt;", summary)
        self.assertNotIn("<script>", summary)
        self.assertIn("<td>Manual pairs</td>", summary)
        self.assertEqual(summary.count("<tr>"), 3)
