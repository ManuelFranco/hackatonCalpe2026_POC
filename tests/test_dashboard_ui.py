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

    def test_shared_parameters_and_seven_sections(self):
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
        tabs = [c["props"]["label"] for c in components if c["type"] == "tabitem"]
        self.assertEqual(len(tabs), 7)
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
        result = fn.fn(state, str(DATA_ROOT / "cwe_c_pairs_dataset/manifest.json"))
        self.assertEqual(len(result), len(fn.outputs))
        self.assertIsNone(state.vector_id)
        self.assertIsNone(result[-1])
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
