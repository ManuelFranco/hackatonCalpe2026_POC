"""Ensure the lab uses the selected calibration, without invented validation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
import torch

import demo_gradio_hackaton as app
from sae_dashboard.calibration_bridge import experiment_from_calibration, snapshot_image
from sae_dashboard.security_reports import SECURITY_REPORT_PROMPT


class CalibrationBridgeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(app, "LAYERS", [9]))
        self.enterContext(patch.dict(app.saes, {9: SimpleNamespace(W_dec=torch.eye(3))}, clear=True))
        self.enterContext(patch.object(app, "runtime_metadata", return_value={"model_commit": "test-commit"}))
        self.enterContext(patch.object(app, "ensure_models_loaded"))
        self.scores = []
        pairs = []
        for index, (a, b) in enumerate((([10., 5., 1.], [2., 5., 3.]),
                                       ([14., 5., 2.], [4., 5., 2.])), start=1):
            pair = {"pair_index": index}
            scores = {}
            for side, values in (("A", a), ("B", b)):
                directory = self.root / "pairs" / f"pair_{index:02d}" / f"condition_{side}"
                directory.mkdir(parents=True)
                score = torch.tensor(values)
                torch.save({"feature_score": score}, directory / "layer_9.pt")
                scores[side] = {9: {"feature_score": score, "residual_reference_norm": 1000}}
                pair[f"condition_{side}"] = {"prompt": f"Custom prompt {index}{side}",
                                               "image_filename": None, "image_path": None}
            self.scores.append(scores)
            pairs.append(pair)
        profile_path, *_ = app.save_profile(self.root / "profile", self.scores, token_scope="last")
        self.metadata = {"run_id": "my-calibration", "model_id": app.MODEL_ID, "num_pairs": 2,
                         "pairs": pairs, "runtime": app.runtime_metadata(), "aggregation": "mean",
                         "feature_token_scope": "last", "sae_ids": {str(k): v for k, v in app.SAE_IDS.items()}}
        (self.root / "experiment.json").write_text(json.dumps(self.metadata))
        self.session = {"run_id": "my-calibration", "run_dir": str(self.root), "profile_path": profile_path,
                        "num_pairs": 2, "feature_token_scope": "last"}

    def make_report(self, key="9:0", **kwargs):
        return experiment_from_calibration(app, self.session, key, calibration_id="my-calibration", **kwargs)

    def test_uses_real_saved_scores_and_preserves_query_without_claiming_validation(self):
        report = self.make_report(query_prompt="Is the answer YES or NO?")
        candidate = report["selected"]
        self.assertEqual((candidate["mean_A"], candidate["mean_B"], candidate["natural_delta"]), (12., 3., -9.))
        self.assertEqual(candidate["pair_deltas"], [-8., -10.])
        self.assertAlmostEqual(candidate["intervention_step"], -13.4, places=4)
        self.assertEqual(report["source_profile"]["sha256"], app.sha256_file(self.session["profile_path"]))
        self.assertEqual(report["default_case"], "query_from_section_2")
        self.assertEqual(report["cases"][-1]["prompt"], "Is the answer YES or NO?")
        self.assertTrue(all("expected" not in case for case in report["cases"]))
        self.assertEqual(report["validation"], [])
        self.assertFalse(report["validation_gate_passed"])
        self.assertEqual(json.loads(Path(report["artifact_path"]).read_text()), report)

    def test_boundary_and_prefix_propagate_from_calibration_to_lab(self):
        self.metadata["prefix_marker"] = "Action:"
        self.metadata["intervention_schedule"] = "First continuation step only"
        (self.root / "experiment.json").write_text(json.dumps(self.metadata))
        prefix = {"assistant_prefix": "Evidence: facts\nAction:", "assistant_prefix_ids": [11, 12],
                  "prefix_marker": "Action:"}
        with patch("sae_dashboard.calibration_bridge.generate_baseline_prefix", return_value=prefix) as generate:
            report = self.make_report(query_prompt="query")
        self.assertEqual(generate.call_args.args[-1], "Action:")
        self.assertEqual(report["cases"][-1]["assistant_prefix_ids"], [11, 12])
        self.assertEqual(report["intervention_schedule"], "First continuation step only")
        demo = app.build_demo()
        try:
            loader = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "load_causal_report")
            values = loader("saved", {"saved": report})
            self.assertEqual(values[-2:], (prefix["assistant_prefix"], "First continuation step only"))
        finally:
            demo.close()

    def test_rejects_excluded_features_stale_rows_and_different_model(self):
        for key in ("9:1", "9:2", "9:-1", "9:200", "29:0"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.make_report(key)
        with self.assertRaisesRegex(ValueError, "older calibration"):
            experiment_from_calibration(app, self.session, "9:0", calibration_id="previous-run")
        with patch.object(app, "runtime_metadata", return_value={"model_commit": "different"}), \
             self.assertRaisesRegex(ValueError, "revision changed"):
            self.make_report()

    def test_images_are_preserved_and_missing_old_images_are_rejected(self):
        original = self.root / "upload.png"
        original.write_bytes(b"synthetic-image-bytes")
        snapshot = snapshot_image(original, self.root / "pairs" / "pair_01" / "condition_A")
        self.metadata["pairs"][0]["condition_A"].update(image_filename="upload.png", image_path=snapshot)
        (self.root / "experiment.json").write_text(json.dumps(self.metadata))
        report = self.make_report(query_image=str(original), query_prompt="Inspect the image.")
        copied = report["cases"][-1]["image"]
        self.assertNotEqual(copied, str(original))
        self.assertEqual(Path(copied).read_bytes(), original.read_bytes())
        self.assertEqual(report["cases"][0]["image"], snapshot)
        self.metadata["pairs"][0]["condition_A"].pop("image_path")
        (self.root / "experiment.json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "images are unavailable"):
            self.make_report()

    def test_ui_transfer_registers_the_experiment_and_clears_old_measurements(self):
        demo = app.build_demo()
        try:
            dependency = next(d for d in demo.config["dependencies"]
                              if demo.fns[d["id"]].fn and demo.fns[d["id"]].fn.__name__ == "transfer_calibration_feature")
            callback = demo.fns[dependency["id"]].fn
            values = callback(self.session, None, "Custom query", {},
                              gr.EventData(None, {"layer": 9, "feature_id": 0, "calibration_id": "my-calibration"}))
            self.assertEqual(len(values), len(dependency["outputs"]))
            self.assertEqual(values[0]["mode"], "calibration")
            self.assertIn("No independent validation", values[4])
            self.assertIsNone(values[5])
            self.assertEqual(values[6], [])
            self.assertEqual(values[8], "")
            self.assertIsNone(values[9])
            self.assertEqual(values[12], "Custom query")
            name = values[-2]["value"]
            self.assertEqual(values[-1][name], values[0])
            loader = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "load_causal_report")
            self.assertEqual(loader(name, values[-1])[0]["selected"], values[0]["selected"])
        finally:
            demo.close()

    def test_image_report_preset_and_transfer_preserve_professional_task(self):
        report = self.make_report(query_prompt=SECURITY_REPORT_PROMPT)
        self.assertEqual((report["cases"][-1]["positive_label"], report["cases"][-1]["negative_label"]),
                         ("Attack", "Clean"))
        demo = app.build_demo()
        try:
            dependency = next(d for d in demo.config["dependencies"]
                              if demo.fns[d["id"]].fn and demo.fns[d["id"]].fn.__name__ == "load_image_report_preset")
            values = app.load_image_report_preset()
            self.assertEqual(len(values), len(dependency["outputs"]))
            controls = {c["id"]: c for c in demo.config["components"]}
            populated = {controls[i]["props"].get("label"): v for i, v in zip(dependency["outputs"], values)}
            self.assertEqual(populated["Query prompt"], SECURITY_REPORT_PROMPT)
            self.assertEqual(populated["Preview response max new tokens"], 320)
            for n in range(1, 4):
                for side in ("A", "B"):
                    self.assertEqual(populated[f"Prompt {side}{n} · optional if image is provided"], SECURITY_REPORT_PROMPT)
                    self.assertTrue(Path(populated[f"Image {side}{n} · optional"]).is_file())
            self.assertIsNone(values[-2])
        finally:
            demo.close()


if __name__ == "__main__":
    unittest.main()
