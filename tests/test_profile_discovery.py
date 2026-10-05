from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.profile_discovery import (
    observe_candidates,
    rank_candidates,
    source_segments,
    source_token_spans,
)
from research.vector_builders import LayerProfile
from sae_dashboard import artifact_store, model_runtime as runtime, workflow
from sae_dashboard import profile_discovery_lab as lab, profile_discovery_view as view
from sae_dashboard.manifests import Condition, Manifest, Pair
from sae_dashboard.session import Session, Settings


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


@contextmanager
def input_runtime():
    captures, batches = [], []
    tokenizer = CharacterTokenizer()

    def render(messages, **kwargs):
        return (
            "<user>\n" + messages[0]["content"][-1]["text"].strip() + "\n<assistant>\n"
        )

    def prepare(image, text):
        rendered = render(runtime.make_messages(image, text))
        ids = torch.tensor(tokenizer(rendered)["input_ids"])
        return {"input_ids": ids[None], "text": text}, ids, len(ids)

    def capture(inputs):
        captures.append((inputs["text"], runtime.MODEL_LOCK._is_owned()))
        chars = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        residuals = torch.zeros(len(chars), 5)
        residuals[:, 0] = 2
        for i, char in enumerate(chars):
            residuals[i, 1] = 4 if char == "+" else 1 if char == "?" else 0
            residuals[i, 2] = 3 if char == "B" else 0
            residuals[i, 3] = 2 if char == "?" else 0
        residuals[-1, 4] = 2 if "+" in inputs["text"] else 1
        return {layer: residuals.clone() for layer in (9, 17, 22, 29)}

    def encode(sae, x):
        batches.append(len(x))
        return x.clone()

    with patch.multiple(
        runtime,
        LAYERS=[9, 17, 22, 29],
        prepare_inputs=prepare,
        capture_prompt_residuals=capture,
        encode_sae_chunked=encode,
        get_image_mask=lambda ids: torch.zeros_like(ids, dtype=torch.bool),
        ensure_models_loaded=lambda **kwargs: None,
        processor=SimpleNamespace(tokenizer=tokenizer, apply_chat_template=render),
        saes={layer: SimpleNamespace(W_dec=torch.eye(5)) for layer in (9, 17, 22, 29)},
        SAE_IDS={layer: "toy-5" for layer in (9, 17, 22, 29)},
    ):
        yield SimpleNamespace(captures=captures, batches=batches)


def input_session(settings=Settings()):
    state = Session()
    state.manifests = (
        Manifest(
            "Synthetic input test",
            (
                Pair(
                    "unicode",
                    Condition("  α=A\nquery ?  "),
                    Condition("  α=B\nquery +  "),
                ),
                Pair("second", Condition("β=A\nlookup ?"), Condition("β=B\nlookup +")),
            ),
            "toy-inputs",
        ),
    )
    workflow.build_profile(state, settings)
    return state


class ProfileDiscoveryTests(unittest.TestCase):
    def test_ranking_penalizes_outliers_keeps_both_signs_and_is_order_invariant(self):
        # One outlier must not outrank a shared, similarly sized difference.
        a = torch.tensor([[0.0, 0.0, 3.0, 2.0]] * 4)
        b = torch.tensor([[2.0, 0.0, 0.0, 2.0]] * 3 + [[2.0, 10.0, 0.0, 2.0]])
        original = a.clone(), b.clone()
        rows = rank_candidates(LayerProfile(a, b, 1))
        self.assertEqual([r["feature"] for r in rows], [2, 0, 1])
        self.assertEqual(rows[0]["same_sign"], 4)
        self.assertEqual(rows[-1]["coverage"], 0.25)
        self.assertLess(rows[-1]["score"], rows[1]["score"])
        reordered = rank_candidates(LayerProfile(a.flip(0), b.flip(0), 1))
        self.assertEqual([r["feature"] for r in reordered], [2, 0, 1])
        self.assertEqual([r["score"] for r in reordered], [r["score"] for r in rows])
        torch.testing.assert_close(a, original[0])
        torch.testing.assert_close(b, original[1])
        self.assertEqual(rank_candidates(LayerProfile(a, a, 1)), [])
        with self.assertRaises(ValueError):
            rank_candidates(LayerProfile(a, b * float("nan"), 1))

    def test_large_dictionary_observation_is_bounded_and_does_not_modify_residuals(
        self,
    ):
        batch_sizes = []
        residuals = torch.arange(80.0).reshape(40, 2)
        original = residuals.clone()

        def encode(sae, x):
            batch_sizes.append(len(x))
            z = torch.zeros(len(x), 262144)
            z[:, 262143] = x[:, 0]
            return z

        app = SimpleNamespace(encode_sae_chunked=encode)
        sae = SimpleNamespace(W_dec=SimpleNamespace(shape=(262144, 2)))
        values = observe_candidates(app, sae, residuals, [262143, 3])
        self.assertEqual(batch_sizes, [16, 16, 8])
        self.assertEqual(values.shape, (40, 2))
        torch.testing.assert_close(values[:, 0], original[:, 0])
        torch.testing.assert_close(residuals, original)
        with self.assertRaises(ValueError):
            observe_candidates(app, sae, residuals, [262144])

    def test_exact_offsets_keep_unicode_trimming_overlap_and_reject_mismatched_ids(
        self,
    ):
        tokenizer = CharacterTokenizer()
        text = "  α🙂\n  "
        rendered = "<user>α🙂<assistant>"
        ids = tokenizer(rendered)["input_ids"]
        spans = source_token_spans(tokenizer, rendered, ids, text)
        self.assertEqual([s for s in spans if s != [0, 0]], [[2, 3], [3, 4]])
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            source_token_spans(tokenizer, rendered, [1] * len(ids), text)
        with self.assertRaisesRegex(ValueError, "uniquely"):
            source_token_spans(tokenizer, "α🙂α🙂", [1], text)
        segments = source_segments(
            "α🙂!", [[0, 2], [1, 3]], [[1, 4], [3, 2]], [True, True]
        )
        self.assertEqual("".join(s["text"] for s in segments), "α🙂!")
        self.assertEqual(segments[1]["values"], [3, 4])
        self.assertEqual(segments[1]["positions"], [0, 1])
        scoped = source_segments(
            "α🙂!", [[0, 2], [1, 3]], [[1, 4], [3, 2]], [True, False]
        )
        self.assertEqual(scoped[1]["values"], [1, 4])
        self.assertEqual(scoped[1]["positions"], [0])
        self.assertFalse(scoped[-1]["selected"])

    def test_inspection_recaptures_only_one_pair_without_generation_or_vector_changes(
        self,
    ):
        with input_runtime() as toy, tempfile.TemporaryDirectory() as temp:
            state = input_session()
            original = state.profile[17].a.clone()
            profile_id = state.profile_id
            toy.captures.clear()
            toy.batches.clear()
            with (
                patch.object(runtime, "generate_answer") as generate,
                patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp) / "runs"),
            ):
                result = lab.inspect_pair(state, Settings(), 17, "0")
                generate.assert_not_called()
                self.assertFalse((Path(temp) / "runs").exists())
            self.assertEqual(len(toy.captures), 2)
            self.assertTrue(all(locked for _, locked in toy.captures))
            self.assertLessEqual(max(toy.batches), 16)
            self.assertEqual(state.profile_id, profile_id)
            self.assertFalse(state.vectors)
            torch.testing.assert_close(state.profile[17].a, original)
            self.assertEqual(
                result["features"], [r["feature"] for r in result["candidates"]]
            )
            self.assertTrue(all(c["spans"] for c in result["conditions"]))
            markup = view.render_observation(result)
            self.assertIn("All candidates", markup)
            self.assertIn("captured chat positions outside the source", markup)
            self.assertIn("Input positions", markup)
            self.assertIn("one activation scale across A and B", markup)

    def test_scope_and_aggregation_are_preserved_and_stale_profiles_fail(self):
        with input_runtime():
            for settings in (Settings(aggregation="max"), Settings(token_scope="last")):
                state = input_session(settings)
                result = lab.inspect_pair(state, settings, 17, "1")
                for condition in result["conditions"]:
                    self.assertEqual(
                        condition["capture"]["scope"], settings.token_scope
                    )
                    if settings.token_scope == "last":
                        self.assertEqual(sum(condition["selected"]), 1)
                if settings.token_scope == "last":
                    self.assertIn(
                        "1 captured chat positions", view.render_observation(result)
                    )
            state = input_session()
            with self.assertRaisesRegex(ValueError, "current token"):
                lab.inspect_pair(state, Settings(token_scope="last"), 17, "0")
            with self.assertRaisesRegex(ValueError, "A/B pair"):
                lab.inspect_pair(state, Settings(), 17, "-1")
            state.profile[17].b[0, :] += 1
            with self.assertRaisesRegex(ValueError, "Re-captured"):
                lab.inspect_pair(state, Settings(), 17, "0")
            state.invalidate_profile()
            with self.assertRaises(ValueError):
                lab.inspect_pair(state, Settings(), 17, "0")

    def test_failed_alignment_falls_back_and_markup_escapes_source_and_labels(self):
        with input_runtime():
            state = input_session()
            with patch.object(
                lab, "source_token_spans", side_effect=ValueError("mismatched IDs")
            ):
                result = lab.inspect_pair(state, Settings(), 17, "0")
            self.assertFalse(result["conditions"][0]["spans"])
            result["pair_label"] = "<script>bad()</script>"
            result["conditions"][0]["tokens"][0] = "<script>bad()</script>"
            markup = view.render_observation(result)
            self.assertNotIn("<script>", markup)
            self.assertIn("&lt;script&gt;", markup)
            self.assertIn("mismatched IDs", markup)
            ranked = view.render_candidates(
                result["candidates"], 17, ["<script>bad()</script>"]
            )
            self.assertNotIn("<script>", ranked)


class DiscoveryUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sae_dashboard.ui import build_demo

        cls.demo = build_demo()

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_profile_capture_populates_candidates_and_follow_passes_top_three_without_inference(
        self,
    ):
        capture = next(f for f in self.demo.fns.values() if f.name == "capture")
        follow = next(
            f for f in self.demo.fns.values() if f.name == "follow_candidates"
        )
        with input_runtime():
            state = input_session()
            with patch("sae_dashboard.ui.workflow.build_profile", return_value=[]):
                output = capture.fn(state, None, 0, 0, 256, "all", "mean")
            self.assertEqual(len(output), len(capture.outputs))
            ranked = next(
                i
                for i, c in enumerate(capture.outputs)
                if getattr(c, "label", None) == "Automatic feature candidates"
            )
            self.assertIn("no feature IDs needed", output[ranked])
            with patch.object(runtime, "generate_answer") as generate:
                output = follow.fn(state, 17, 0, 0, 256, "all", "mean")
                generate.assert_not_called()
            self.assertEqual(len(output), len(follow.outputs))
            self.assertEqual(output[0], 17)
            self.assertEqual(
                output[1],
                ", ".join(
                    str(r["feature"]) for r in rank_candidates(state.profile[17])[:3]
                ),
            )
            self.assertIsNone(output[2])  # Old generation trace is cleared.
            self.assertEqual(output[-1]["selected"], "extra")

    def test_invalidating_inputs_clears_discovery_and_all_callbacks_match_outputs(self):
        from gradio.helpers import special_args

        discover = next(
            f for f in self.demo.fns.values() if f.name == "discover_inputs"
        )
        invalidate = next(f for f in self.demo.fns.values() if f.name == "invalidate")
        with input_runtime():
            state = input_session()
            args, progress_index, _, _ = special_args(
                discover.fn, [state, 17, "0", 42, 0.3, 128, "all", "mean"]
            )
            self.assertIsNotNone(progress_index)
            output = discover.fn(*args)
            self.assertEqual(len(output), len(discover.outputs))
            self.assertEqual(output[0]["settings"]["seed"], 42)
            self.assertIn("All candidates", output[1])
            self.assertIn(discover.outputs[0], invalidate.outputs)
            output = invalidate.fn(state)
            self.assertEqual(len(output), len(invalidate.outputs))
            self.assertIsNone(output[invalidate.outputs.index(discover.outputs[0])])
            self.assertEqual(output[invalidate.outputs.index(discover.outputs[1])], "")


if __name__ == "__main__":
    unittest.main()
