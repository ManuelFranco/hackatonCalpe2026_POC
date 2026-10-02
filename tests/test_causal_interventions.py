from types import SimpleNamespace
import unittest
import torch
from research.causal_interventions import (
    generate_feature_report,
    probe,
    readout,
    prepare_case_inputs,
)


class CausalLabTests(unittest.TestCase):
    def test_report_steering_reaches_prefill_and_decoding_and_cleans_up(self):
        layer = torch.nn.Identity()
        seen = []
        fail = False

        def generate(inputs, length, max_tokens, temperature, seed):
            for hidden in (
                torch.tensor([[[8.0, 9.0], [1.0, 2.0]]]),
                torch.tensor([[[3.0, 4.0]]]),
            ):
                original = hidden.clone()
                seen.append(layer(hidden))
                torch.testing.assert_close(hidden, original)
            if fail:
                raise RuntimeError("generation failed")
            return "real answer"

        app = SimpleNamespace(
            prepare_inputs=lambda *args: ({}, None, 2),
            saes={0: SimpleNamespace(W_dec=torch.eye(2))},
            get_layer_module=lambda _: layer,
            extract_hidden=lambda output: output,
            replace_hidden=lambda old, new: new,
            generate_answer=generate,
        )
        candidate = {"layer": 0, "feature_id": 0, "intervention_step": -2}
        output = generate_feature_report(app, {"prompt": "report"}, candidate, 0.5)
        self.assertEqual(output["answer"], "real answer")
        self.assertEqual(output["forward_calls"], 2)
        torch.testing.assert_close(seen[0], torch.tensor([[[8.0, 9.0], [0.0, 2.0]]]))
        torch.testing.assert_close(seen[1], torch.tensor([[[2.0, 4.0]]]))
        seen.clear()
        generate_feature_report(app, {"prompt": "report"}, candidate, 0)
        torch.testing.assert_close(seen[1], torch.tensor([[[3.0, 4.0]]]))
        fail = True
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            generate_feature_report(app, {"prompt": "report"}, candidate, 1)
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_intervention_changes_only_final_position_and_hooks_are_removed(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
                self.fail = False

            def forward(self, hidden, **kwargs):
                self.seen = self.layer(hidden)
                if self.fail:
                    raise RuntimeError("deliberate failure")
                return SimpleNamespace(logits=self.seen)

        model = ToyModel()
        hidden = torch.tensor([[[8.0, 9.0], [1.0, 2.0]]])
        tokenizer = SimpleNamespace(
            encode=lambda s, **kw: [0] if s == "YES" else [1],
            decode=lambda ids, **kw: "YES" if ids[0] == 0 else "NO",
        )
        app = SimpleNamespace(
            processor=SimpleNamespace(tokenizer=tokenizer),
            model=model,
            LAYERS=[0],
            saes={0: SimpleNamespace(W_dec=torch.eye(2))},
            prepare_inputs=lambda *args: ({"hidden": hidden}, torch.tensor([0, 1]), 2),
            get_layer_module=lambda layer: model.layer,
            extract_hidden=lambda output: output,
            replace_hidden=lambda old, new: new,
            encode_sae_chunked=lambda sae, x: x.relu(),
        )
        case = {"prompt": "test", "positive_label": "YES", "negative_label": "NO"}
        base, _ = probe(app, case)
        zero, _ = probe(app, case, {"layer": 0, "feature_id": 0, "amount": 0})
        self.assertEqual(base["log_odds"], zero["log_odds"])
        changed, _ = probe(app, case, {"layer": 0, "feature_id": 0, "amount": 2})
        self.assertEqual(changed["log_odds"] - base["log_odds"], 2)
        torch.testing.assert_close(model.seen[:, 0], hidden[:, 0])
        torch.testing.assert_close(hidden, torch.tensor([[[8.0, 9.0], [1.0, 2.0]]]))
        self.assertEqual(len(model.layer._forward_hooks), 0)
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            probe(app, case, {"layer": 0, "feature_id": 0, "amount": 2})
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_conditional_probability_is_not_absolute_probability(self):
        result = readout(torch.tensor([-10.0, -11.0, 10.0]), [0, 1])
        self.assertAlmostEqual(result["log_odds"], 1)
        self.assertGreater(result["p_positive_given_labels"], 0.7)
        self.assertLess(result["label_probability_mass"], 1e-7)
        self.assertEqual(result["top_token_id"], 2)

    def test_first_step_schedule_and_shared_generation_settings(self):
        layer = torch.nn.Identity()
        seen, settings = [], []

        def generate(inputs, length, max_tokens, temperature, seed):
            settings.append((max_tokens, temperature, seed))
            for _ in range(2):
                seen.append(layer(torch.zeros(1, 1, 2)))
            return "# Markdown answer"

        app = SimpleNamespace(
            prepare_inputs=lambda *args: ({}, None, 2),
            saes={9: SimpleNamespace(W_dec=torch.eye(2))},
            get_layer_module=lambda _: layer,
            extract_hidden=lambda output: output,
            replace_hidden=lambda old, new: new,
            generate_answer=generate,
        )
        generate_feature_report(
            app,
            {"prompt": "Q"},
            {"layer": 9, "feature_id": 0, "intervention_step": 2},
            0.5,
            max_new_tokens=90,
            schedule="First continuation step only",
            temperature=0.6,
            seed=13,
        )
        self.assertEqual(settings, [(90, 0.6, 13)])
        torch.testing.assert_close(seen[0], torch.tensor([[[1.0, 0.0]]]))
        torch.testing.assert_close(seen[1], torch.zeros(1, 1, 2))
        self.assertFalse(layer._forward_hooks)

    def test_prefix_extends_masks_and_continuation_boundary(self):
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "token_type_ids": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.ones(1, 3),
        }
        app = SimpleNamespace(
            prepare_inputs=lambda *args: (inputs, inputs["input_ids"][0], 2),
            processor=SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda *args, **kw: [3, 4])
            ),
        )
        updated, ids, length = prepare_case_inputs(
            app, {"prompt": "Q", "assistant_prefix": "prefix"}
        )
        self.assertEqual(length, 4)
        self.assertEqual(ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(updated["attention_mask"].tolist(), [[1, 1, 1, 1]])
        self.assertEqual(updated["token_type_ids"].tolist(), [[1, 1, 0, 0]])
        self.assertEqual(inputs["input_ids"].shape[-1], 2)
        self.assertIs(updated["pixel_values"], inputs["pixel_values"])
