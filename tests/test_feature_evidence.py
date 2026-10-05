from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.feature_evidence import (
    classification,
    decision_probe,
    shortlist,
    summarize,
)
from research.vector_builders import LayerProfile
from sae_dashboard import (
    evidence_lab as lab,
    evidence_view as view,
    model_runtime as runtime,
)
from sae_dashboard.manifests import Condition, Manifest, Pair
from sae_dashboard.session import Session, Settings


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()
        self.fail = False

    def forward(self, hidden, **kwargs):
        self.seen = self.layer(hidden)
        if self.fail:
            raise RuntimeError("inference failed")
        return SimpleNamespace(logits=self.seen)


def toy_app():
    model = ToyModel()
    decoder = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    def prepare(image, prompt):
        if "BOUND" in prompt:
            values = [1.0, 1.02]
        elif "INTERPOLATED" in prompt:
            values = [0.2, 1.02]
        elif "evaluates to 4" in prompt or "len([10" in prompt:
            values = [2.0, 1.0]
        else:
            values = [1.0, 2.0]
        hidden = torch.tensor([[[8.0, 9.0], values]])
        return {"hidden": hidden}, torch.tensor([0, 1]), 2

    return SimpleNamespace(
        model=model,
        saes={9: SimpleNamespace(W_dec=decoder)},
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda x, **kw: [0 if x == "A" else 1])
        ),
        prepare_inputs=prepare,
        get_layer_module=lambda _: model.layer,
        extract_hidden=lambda x: x,
        replace_hidden=lambda old, new: new,
        encode_sae_chunked=lambda sae, x: torch.stack(
            [x[:, 0], x.sum(-1), x[:, 1]], -1
        ).relu(),
    )


def toy_session():
    state = Session()
    state.profile_id = "toy-calibration"
    state.capture_key = Settings().capture_key
    state.profile = {
        9: LayerProfile(
            torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]),
            torch.tensor([[0.2, 1.2, 1.0], [0.2, 1.2, 1.0]]),
            2.0,
        )
    }
    state.manifests = (
        Manifest(
            "Toy evidence",
            (
                Pair("one", Condition("BOUND one"), Condition("INTERPOLATED one")),
                Pair("two", Condition("BOUND two"), Condition("INTERPOLATED two")),
            ),
            "toy-data",
        ),
    )
    return state


@contextmanager
def toy_runtime():
    app = toy_app()
    with patch.multiple(
        runtime,
        **vars(app),
        ensure_models_loaded=lambda: None,
        MODEL_ID="toy-model",
        SAE_RELEASE="toy-sae",
        SAE_IDS={9: "toy-3"},
    ):
        yield app


def toy_study(percent=50):
    state = toy_session()
    prepared = lab.prepare_study(state, Settings(), lab.SOURCES[3], None, 9)
    with toy_runtime():
        result = lab.run_study(
            state,
            Settings(),
            prepared,
            ["0"],
            "",
            "Binds values",
            "Interpolates values",
            2,
            percent,
        )
    return state, prepared, result


class FeatureEvidenceTests(unittest.TestCase):
    def test_correlated_feature_moves_and_actual_reencoding_is_measured(self):
        app = toy_app()
        inputs, _, _ = app.prepare_inputs(None, "BOUND")
        original = inputs["hidden"].clone()
        result = decision_probe(app, inputs, [0, 1], 9, 0, 0.5)
        self.assertGreater(result["activation_after"], result["activation_before"])
        self.assertEqual(result["other_features_moved"], 1)
        self.assertEqual(result["neighbors"][0]["feature"], 1)
        self.assertTrue(result["budget_capped"])
        self.assertAlmostEqual(result["residual_change"], 0.02, places=6)
        self.assertAlmostEqual(result["target_energy_share"], 0.5, places=5)
        torch.testing.assert_close(app.model.seen[:, 0], original[:, 0])
        torch.testing.assert_close(inputs["hidden"], original)
        self.assertFalse(app.model.layer._forward_hooks)

    def test_random_control_matches_case_norm_and_zero_is_identity(self):
        app = toy_app()
        inputs, _, _ = app.prepare_inputs(None, "BOUND")
        target = decision_probe(app, inputs, [0, 1], 9, 0, -0.5)
        control = decision_probe(app, inputs, [0, 1], 9, 0, -0.5, 23)
        self.assertAlmostEqual(
            target["intended_delta_norm"], control["intended_delta_norm"], places=7
        )
        repeat = decision_probe(app, inputs, [0, 1], 9, 0, -0.5, 23)
        self.assertEqual(control, repeat)
        base = decision_probe(app, inputs, [0, 1], 9)
        zero = decision_probe(app, inputs, [0, 1], 9, 0, 0)
        self.assertEqual(zero["margin"], base["margin"])
        self.assertEqual(zero["other_features_moved"], 0)
        app.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            decision_probe(app, inputs, [0, 1], 9, 0, 0.5)
        self.assertFalse(app.model.layer._forward_hooks)

    def test_inactive_feature_does_not_manufacture_an_intervention(self):
        app = toy_app()
        inputs = {"hidden": torch.tensor([[[0.0, 2.0]]])}
        for seed in (None, 23):
            row = decision_probe(app, inputs, [0, 1], 9, 0, 0.5, seed)
            self.assertTrue(row["inactive"])
            self.assertEqual(row["residual_change"], 0)

    def test_signal_detection_distinguishes_response_bias_from_discrimination(self):
        truths = ["A"] * 4 + ["B"] * 4
        all_b = classification([{"expected": y, "prediction": "B"} for y in truths])
        all_a = classification([{"expected": y, "prediction": "A"} for y in truths])
        correct = classification([{"expected": y, "prediction": y} for y in truths])
        self.assertEqual(all_b["accuracy"], 0.5)
        self.assertAlmostEqual(all_b["dprime"], 0)
        self.assertAlmostEqual(all_a["dprime"], 0)
        self.assertLess(all_b["criterion"], 0)
        self.assertGreater(all_a["criterion"], 0)
        self.assertGreater(correct["dprime"], 2)
        tied = classification([{"expected": y, "prediction": "Tie"} for y in truths])
        self.assertEqual(tied["accuracy"], 0)
        self.assertIsNone(tied["dprime"])

    def test_unstable_contrast_is_not_ranked_above_stable_signal(self):
        profile = LayerProfile(
            torch.zeros(4, 2),
            torch.tensor([[1.0, 100.0], [1.0, -99.0], [1.0, 0.0], [1.0, 0.0]]),
            1,
        )
        self.assertEqual(shortlist(profile)[0]["feature"], 0)

    def test_end_to_end_counts_controls_replay_export_and_profile_preservation(self):
        state, prepared, result = toy_study()
        self.assertEqual(result["forwards"], 8 * 7 + 1)
        self.assertEqual(len(result["groups"]), 6)
        self.assertTrue(result["baseline_replay_passed"])
        best = lab.feature_groups(result)[0]
        self.assertEqual(best["mode"], "Raise")
        self.assertEqual(best["summary"]["accuracy"], 1)
        self.assertEqual(best["summary"]["control_regressions"], 0)
        self.assertEqual(result["base_summary"]["accuracy"], 0.5)
        self.assertEqual(state.profile_id, "toy-calibration")
        self.assertFalse(state.vectors)
        self.assertGreater(prepared["overlap"], 0)
        json.dumps(result, allow_nan=False)
        result["cases"][0]["prompt"] = "<script>alert('x')</script>"
        rendered = view.detail(result, best["key"], result["cases"][0]["case_id"])
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(lab, "ARTIFACT_ROOT", Path(directory)),
        ):
            paths = lab.export_report(state, result)
            self.assertTrue(all(Path(p).is_file() for p in paths))
            self.assertEqual(
                json.loads(Path(paths[1]).read_text())["profile_id"], "toy-calibration"
            )

    def test_zero_study_and_frozen_input_contract(self):
        state, prepared, result = toy_study(0)
        self.assertIn("Identity control", view.overview(result))
        self.assertIn("Causal effect not assessed", view.overview(result))
        result["activity_percent"] = 50
        self.assertIn("No effective intervention", view.overview(result))
        self.assertNotIn("BEST OBSERVED", view.overview(result))
        for group in result["groups"]:
            self.assertEqual(group["summary"]["correct_margin_change"], 0)
            self.assertEqual(group["summary"]["control_flips"], 0)
        with self.assertRaisesRegex(ValueError, "same unique cases"):
            summarize(result["baseline"], result["groups"][0]["rows"][:-1])
        state.profile_id = "new-profile"
        with self.assertRaisesRegex(ValueError, "profile changed"):
            lab.run_study(
                state,
                Settings(),
                prepared,
                [0],
                "",
                "A description",
                "B description",
                1,
                50,
            )

    def test_noop_groups_cannot_outrank_effective_interventions(self):
        from copy import deepcopy

        _, _, result = toy_study()
        best = lab.feature_groups(result)[0]
        noop = deepcopy(best)
        noop["feature"] = 999
        noop["summary"]["accuracy"] = 1.0
        noop["summary"]["correct_margin_change"] = 1e6
        for row in noop["rows"]:
            if row["group"] == "target":
                row["residual_change"] = 0.0
        result["groups"].append(noop)
        self.assertEqual(lab.effective_cases(noop), 0)
        self.assertNotEqual(lab.feature_groups(result)[0]["feature"], 999)

    def test_paired_plan_never_inserts_expected_labels_and_requires_definitions(self):
        prepared = {
            "manifests": (
                Manifest(
                    "private title",
                    (Pair("secret-id", Condition("first"), Condition("second")),),
                    "x",
                ),
            )
        }
        with self.assertRaises(ValueError):
            lab.make_plan(prepared, "", "", 1, 0)
        cases = lab.make_plan(prepared, "Description one", "Description two", 1, 0)
        self.assertEqual([c["expected"] for c in cases[:2]], ["A", "B"])
        self.assertNotIn("secret-id", cases[0]["prompt"])
        self.assertNotIn("Expected A", cases[0]["prompt"])
        self.assertEqual(len(cases), 6)


if __name__ == "__main__":
    unittest.main()
