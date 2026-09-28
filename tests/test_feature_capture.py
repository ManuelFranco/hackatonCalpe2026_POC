"""Regression checks for token selection, profile isolation and example labels."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import demo_gradio_hackaton as app
from sae_dashboard.disclosure_examples import DISCLOSURE_CASES


class FeatureCaptureTests(unittest.TestCase):
    def test_last_avoids_length_only_difference(self):
        # Same signal in a common prefix; the only change is a zero-valued token.
        a = torch.tensor([[8.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        b = torch.tensor([[8.0, 0.0], [0.0, 2.0]])
        with patch.object(app, "FEATURE_AGGREGATION", "mean"):
            all_a = app.aggregate_feature_acts(a, torch.zeros(3), "all")
            all_b = app.aggregate_feature_acts(b, torch.zeros(2), "all")
            self.assertNotEqual(float(all_a[0]), float(all_b[0]))
            torch.testing.assert_close(
                app.aggregate_feature_acts(a, torch.zeros(3), "last"),
                app.aggregate_feature_acts(b, torch.zeros(2), "last"),
            )

    def test_canceled_mean_is_zero_before_decoding(self):
        result = app.compute_common_feature_delta([
            torch.tensor([1.0, 2.0]), torch.tensor([-1.0 + 1e-7, 4.0]),
        ], epsilon=1e-6)
        self.assertTrue(bool(result["canceled_mask"][0]))
        self.assertEqual(float(result["masked_mean_delta"][0]), 0.0)
        self.assertEqual(float(result["masked_mean_delta"][1]), 3.0)

    def test_null_pair_veto_still_applies(self):
        result = app.compute_common_feature_delta([
            torch.tensor([2.0, -3.0]), torch.zeros(2),
        ], epsilon=1e-6)
        self.assertFalse(bool(result["common_mask"].any()))
        self.assertEqual(float(result["masked_mean_delta"].abs().sum()), 0.0)

    def test_condition_scores_and_norm_use_selected_position(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "LAYERS", [9]), \
             patch.object(app, "model", SimpleNamespace(config=SimpleNamespace())), \
             patch.dict(app.saes, {9: object()}, clear=True), \
             patch.dict(app.sae_releases_used, {9: app.SAE_RELEASE}, clear=True), \
             patch.object(app, "encode_sae_chunked", side_effect=lambda sae, x: x):
            root = Path(directory)
            result = app.save_condition(
                root, "A", "prompt", None, torch.tensor([10, 11]),
                {9: torch.tensor([[1000.0, 0.0], [0.0, 2.0]])}, token_scope="last",
            )
            torch.testing.assert_close(result[9]["feature_score"], torch.tensor([0.0, 2.0]))
            self.assertEqual(result[9]["residual_reference_norm"], 2.0)
            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual(metadata["num_scored_tokens"], 1)
            self.assertEqual(metadata["capture_scope"], app.CAPTURE_SCOPE)

    def test_profiles_keep_scope_and_reject_old_capture_format(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "LAYERS", [9]), \
             patch.dict(app.saes, {9: SimpleNamespace(W_dec=torch.eye(2))}, clear=True):
            scores = [
                {"A": {9: {"feature_score": torch.zeros(2), "residual_reference_norm": 10}},
                 "B": {9: {"feature_score": torch.tensor([v, v + 1]), "residual_reference_norm": 10}}}
                for v in (2.0, 4.0)
            ]
            profiles = {}
            for scope in ("all", "last"):
                path, *_ = app.save_profile(Path(directory) / scope, scores, token_scope=scope)
                profiles[scope] = path
            for scope in ("last", "all"):
                directions = app.load_steering_directions(profiles[scope], token_scope=scope)
                self.assertAlmostEqual(float(directions[9].norm()), 10 * app.STEERING_FRACTION_PER_UNIT)
            with self.assertRaisesRegex(ValueError, "feature_token_scope"):
                app.load_steering_directions(profiles["all"], token_scope="last")
            payload = torch.load(profiles["last"], weights_only=True)
            payload["format_version"] = 2
            torch.save(payload, profiles["last"])
            with self.assertRaisesRegex(ValueError, "Recalibrate"):
                app.load_steering_directions(profiles["last"], token_scope="last")

    def test_preset_matches_ui_outputs_and_clears_old_profile(self):
        demo = app.build_demo()
        try:
            dependency = next(d for d in demo.config["dependencies"]
                              if d["api_name"] == "load_disclosure_preset")
            values = app.load_disclosure_preset()
            self.assertEqual(len(values), len(dependency["outputs"]))
            self.assertIsNone(values[-2])
            self.assertEqual(values[0], min(3, app.MAX_PAIRS))
        finally:
            demo.close()


class DisclosureFixtureTests(unittest.TestCase):
    def test_reference_labels_match_compiled_programs(self):
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in DISCLOSURE_CASES.values():
                with self.subTest(case=case["id"]):
                    source = root / f"{case['id']}.c"
                    executable = root / case["id"]
                    source.write_text(case["source"])
                    subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                                    str(source), "-o", str(executable)], check=True,
                                   capture_output=True, timeout=20)
                    result = subprocess.run([str(executable)], cwd=root,
                                            env={"PATH": os.defpath}, check=True,
                                            capture_output=True, text=True, timeout=3)
                    label = "DISCLOSURE" if "DEMO_SECRET_123" in result.stdout else "NO_DISCLOSURE"
                    self.assertEqual(label, case["expected"])
                    self.assertEqual("YES" if "DEMO_SECRET_123" in result.stdout else "NO",
                                     case["expected_reply"])


if __name__ == "__main__":
    unittest.main()
