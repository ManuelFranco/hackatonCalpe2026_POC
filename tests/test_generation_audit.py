from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.code_review import review_python
from research.generation_trace import GenerationTrace, decoded_spans, feature_ids
from research.vector_builders import VectorResult
from sae_dashboard import evidence_lab, generation_lab as lab, generation_view as view
from sae_dashboard import model_runtime as runtime
from sae_dashboard.session import Session, Settings


@contextmanager
def toy_runtime():
    layers = {k: torch.nn.Identity() for k in (9, 17)}
    seen, batch_sizes = [], []
    pieces = [
        "def lookup(conn, uid):\n",
        "    return conn.execute('SELECT id FROM users WHERE id = ?', (uid,)).fetchone()",
        "<eos>",
    ]

    def generate(input_ids, max_new_tokens, **kwargs):
        generated = []
        for i in range(min(3, max_new_tokens)):
            x = torch.tensor(
                [[[-2.0, 1.0], [1.0, 2.0]]] if i == 0 else [[[1.0 + i, 2.0]]]
            )
            for k, module in layers.items():
                x = module(x)
                seen.append((i, k, x.clone()))
            # Exercise RNG preservation with observation hooks enabled.
            torch.rand(1)
            generated.append(i)
        return torch.cat([input_ids, torch.tensor([generated])], dim=1)

    def encode(sae, x):
        batch_sizes.append(len(x))
        return torch.stack([x[:, 0], x[:, 1], x.sum(-1)], -1).relu()

    model = SimpleNamespace(
        generate=generate, generation_config=SimpleNamespace(eos_token_id=2)
    )
    processor = SimpleNamespace(
        decode=lambda ids, skip_special_tokens, **kwargs: "".join(
            pieces[j] for j in ids if j != 2 or not skip_special_tokens
        ),
        tokenizer=SimpleNamespace(
            convert_ids_to_tokens=lambda ids: [pieces[j] for j in ids]
        ),
    )
    with patch.multiple(
        runtime,
        LAYERS=[9, 17],
        get_layer_module=layers.__getitem__,
        model=model,
        processor=processor,
        saes={k: SimpleNamespace(W_dec=torch.ones(3, 2)) for k in layers},
        encode_sae_chunked=encode,
        ensure_models_loaded=lambda **kw: None,
        prepare_inputs=lambda image, text: (
            {"input_ids": torch.tensor([[11, 12]])},
            torch.tensor([11, 12]),
            2,
        ),
        SAE_IDS={9: "toy-3", 17: "toy-3"},
        MODEL_ID="UI validation · synthetic model, not Gemma results",
        SAE_RELEASE="synthetic",
        STEER_LAST_TOKEN_ONLY=True,
    ):
        yield SimpleNamespace(layers=layers, seen=seen, batch_sizes=batch_sizes)


def toy_session():
    state = Session()
    state.profile, state.profile_id, state.capture_key = (
        {9: True},
        "profile",
        Settings().capture_key,
    )
    state.vectors = {
        9: VectorResult(torch.ones(3), torch.tensor([0.5, 0.0])),
        17: VectorResult(torch.ones(3), torch.tensor([0.0, 0.25])),
    }
    state.vector_id = "vectors"
    return state


def toy_result(mode=lab.MODES[1], **kwargs):
    with toy_runtime():
        return lab.run_generation(
            toy_session(),
            Settings(max_new_tokens=8),
            lab.DEFAULT_PROMPT,
            mode,
            17,
            kwargs.get("observed", "0, 2"),
            {9: 1.0, 17: 1.0},
        )


class GenerationTraceTests(unittest.TestCase):
    def test_text_offsets_clip_whitespace_and_special_tokens_and_refuse_rewrites(self):
        prefixes = [
            "",
            "  ",
            "  α\n",
            "  α\nSELECT",
            "  α\nSELECT ?\n  ",
            "  α\nSELECT ?\n  ",
        ]
        app = SimpleNamespace(
            processor=SimpleNamespace(decode=lambda ids, **kw: prefixes[len(ids)])
        )
        spans, status = decoded_spans(app, list(range(5)), "α\nSELECT ?")
        self.assertEqual(spans, [[0, 0], [0, 2], [2, 8], [8, 10], [10, 10]])
        self.assertTrue(status.startswith("Exact"))
        self.assertFalse(decoded_spans(app, list(range(5)), "Other code")[0])
        app.processor.decode = lambda ids, **kw: "�" if len(ids) == 1 else "🙂"
        self.assertFalse(decoded_spans(app, [0, 1], "🙂")[0])

    def test_observer_is_read_only_seeded_and_aligned_through_cached_generation(self):
        with toy_runtime() as toy:
            inputs, _, n = runtime.prepare_inputs(None, "prompt")
            plain = runtime.generate_answer(inputs, n, 8, 0.7, seed=42)
            rng = torch.random.get_rng_state().clone()
            trace = GenerationTrace(17)
            observed = runtime.generate_answer(inputs, n, 8, 0.7, seed=42, trace=trace)
            self.assertEqual(plain, observed)
            torch.testing.assert_close(rng, torch.random.get_rng_state())
            self.assertEqual(trace.forward_lengths, [2, 1, 1])
            self.assertEqual(trace.token_ids, [0, 1, 2])
            self.assertTrue(trace.ended_at_eos)
            self.assertTrue(
                all(torch.equal(a, b) for a, b in zip(trace.before, trace.after))
            )
            self.assertTrue(all(not m._forward_hooks for m in toy.layers.values()))

    def test_existing_last_all_and_first_step_schedules_are_preserved(self):
        for last in (True, False):
            for first_only in (True, False):
                with (
                    toy_runtime() as toy,
                    patch.object(runtime, "STEER_LAST_TOKEN_ONLY", last),
                ):
                    inputs, _, n = runtime.prepare_inputs(None, "prompt")
                    original = inputs["input_ids"].clone()
                    args = dict(
                        steering_directions={9: torch.tensor([2.0, 0.0])},
                        strengths={9: 1.0},
                        first_step_only=first_only,
                    )
                    plain = runtime.generate_answer(inputs, n, 8, 0, **args)
                    expected = [x for _, _, x in toy.seen]
                    toy.seen.clear()
                    trace = GenerationTrace(17)
                    observed = runtime.generate_answer(
                        inputs, n, 8, 0, **args, trace=trace
                    )
                    self.assertEqual(plain, observed)
                    for a, (_, _, b) in zip(expected, toy.seen):
                        torch.testing.assert_close(a, b)
                    self.assertEqual(float(toy.seen[0][2][0, 0, 0]), -2 if last else 0)
                    self.assertEqual(
                        float(toy.seen[2][2][0, 0, 0]), 2 if first_only else 4
                    )
                    torch.testing.assert_close(inputs["input_ids"], original)

    def test_capture_failure_and_alignment_failure_remove_hooks(self):
        with toy_runtime() as toy:
            inputs, _, n = runtime.prepare_inputs(None, "prompt")
            for method in ("capture", "finish"):
                trace = GenerationTrace(17)
                with patch.object(
                    trace, method, side_effect=ValueError("trace failed")
                ):
                    with self.assertRaisesRegex(ValueError, "trace failed"):
                        runtime.generate_answer(inputs, n, 8, 0, trace=trace)
                self.assertTrue(all(not m._forward_hooks for m in toy.layers.values()))
            trace = GenerationTrace(17)
            trace.capture(17, torch.ones(1, 2, 2), torch.ones(1, 2, 2))
            with self.assertRaisesRegex(ValueError, "counts differ"):
                trace.finish(torch.tensor([1, 2]), 2, 2)
            with self.assertRaisesRegex(ValueError, "cached"):
                trace.finish(torch.tensor([1]), 3, 2)

    def test_selection_and_encoding_are_bounded_for_large_dictionaries(self):
        with toy_runtime() as toy, patch("research.generation_trace.CHUNK_SIZE", 2):
            inputs, _, n = runtime.prepare_inputs(None, "prompt")
            trace = GenerationTrace(17)
            runtime.generate_answer(inputs, n, 8, 0, trace=trace)
            self.assertEqual(trace.select(runtime), [2, 0, 1])
            data = trace.measure(runtime, [0, 2], 8)
            self.assertEqual(data["before"], [[1, 3], [2, 4], [3, 5]])
            self.assertLessEqual(max(toy.batch_sizes), 2)
            self.assertEqual(data["termination"], "EOS")


class CodeReviewTests(unittest.TestCase):
    def test_multiline_sql_source_is_tracked_but_display_interpolation_is_not_flagged(
        self,
    ):
        r = review_python("""def lookup(conn, value):
    preview = f"SELECT {value}"
    query = (
        "SELECT id FROM users WHERE name = '"
        + value + "'"
    )
    return conn.execute(query).fetchone()
""")
        self.assertEqual(r["dynamic_queries"], 1)
        self.assertEqual(r["findings"][0]["query_lines"], [4, 5])
        self.assertEqual(r["findings"][0]["call_lines"], [7, 7])
        self.assertEqual(
            view.sql_line_classes(r), {4: "dynamic", 5: "dynamic", 7: "dynamic"}
        )
        self.assertNotIn(2, view.sql_line_classes(r))

    def test_only_adjacent_assignments_in_the_same_block_are_resolved(self):
        code = """def lookup(conn, uid):
    try:
        sql = "SELECT * FROM users WHERE id = ?"
        conn.execute(sql, (uid,))
        sql = f"SELECT {uid}"
        conn.execute(sql)
        conn.execute(sql)
        sql = "SELECT 1"
        if uid:
            conn.execute(sql)
    except Exception:
        conn.execute(sql)
"""
        r = review_python(code)
        self.assertEqual(
            [f["kind"] for f in r["findings"]],
            [
                "Literal query",
                "Dynamic query · review",
                "Unresolved query",
                "Unresolved query",
                "Unresolved query",
            ],
        )
        self.assertEqual(r["findings"][0]["assignment"], {"name": "sql", "line": 3})
        self.assertEqual(r["unresolved_queries"], 3)

    def test_direct_construction_is_reported_without_claiming_exploitability(self):
        text = """Intro\n```python
conn.execute("SELECT id FROM users WHERE id = ?", (uid,))
conn.execute(f"SELECT {column}")
conn.execute("SELECT " + column)
conn.execute("SELECT %s" % column)
conn.execute("SELECT {}".format(column))
conn.execute(query)
```"""
        review = review_python(text)
        self.assertEqual(review["syntax"], "Parsed")
        self.assertEqual(review["dynamic_queries"], 4)
        self.assertEqual(review["unresolved_queries"], 1)
        self.assertEqual(review["findings"][1]["line"], 4)
        self.assertFalse(review["executed"])

    def test_incomplete_non_python_and_unparsed_outputs_are_not_clean_bills(self):
        self.assertEqual(
            review_python("```javascript\nfoo();\n```")["syntax"], "Not assessed"
        )
        broken = review_python("```python\ndef foo(")
        self.assertEqual(broken["syntax"], "Not parsed")
        self.assertTrue(broken["blocks"][0]["unclosed_fence"])
        raw = review_python(
            'import os\nos.system("THIS MUST NEVER RUN")\n# conn.execute(f"{x}")'
        )
        self.assertEqual(raw["syntax"], "Parsed")
        self.assertFalse(raw["findings"])
        multi = review_python("~~~python\nx = 1\n~~~\n```python\nx = (\n```")
        self.assertEqual(len(multi["blocks"]), 2)
        self.assertEqual(multi["syntax"], "Not parsed")


class GenerationAuditTests(unittest.TestCase):
    def test_line_activity_handles_multiline_tokens_ignores_eos_and_refuses_bad_offsets(
        self,
    ):
        run = {
            "response": "first\nlast",
            "trace": {
                "text_spans": [[0, 5], [5, 7], [7, 10], [10, 10]],
                "after": [[1, 0], [4, 8], [2, 3], [999, 999]],
            },
        }
        self.assertEqual(
            view.line_activity(run, 2),
            [
                {"peaks": [4, 8], "tokens": "1–2"},
                {"peaks": [4, 8], "tokens": "2–3"},
            ],
        )
        run["trace"]["text_spans"][2] = [6, 11]
        self.assertIsNone(view.line_activity(run, 2))

    def test_code_shading_is_scoped_interactive_and_keeps_risk_separate_from_activity(
        self,
    ):
        result = toy_result()
        html = view.overview(result)
        self.assertIn('type="radio"', html)
        self.assertIn("Feature #0", html)
        self.assertIn("el-sql-literal", html)
        self.assertIn("not causal attribution", html)
        self.assertIn("predicted tokens", html)
        self.assertIn('href="#shade-', html)
        self.assertNotIn("<script", html)
        result["runs"][0]["trace"].pop("text_spans")
        self.assertIn("Feature shading unavailable", view.overview(result))
        result["runs"][1]["trace"].pop("text_spans")
        self.assertIn(" disabled", view.overview(result))

    def test_sparse_activity_opens_at_peak_without_changing_measurements(self):
        result = toy_result()
        trace = result["runs"][0]["trace"]
        trace["before"] = [[0, 0] for _ in range(100)]
        trace["after"] = [[0, 0] for _ in range(100)]
        trace["after"][70][0] = 10
        trace["token_ids"] = list(range(100))
        trace["tokens"] = ["x"] * 100
        self.assertEqual(view.activity_start(result), 55)
        html = view.timeline(result)
        self.assertIn("Tokens 55–86", html)
        self.assertIn("active 1/100", html)
        self.assertIn("peak token 71", html)
        self.assertIn("active 0/100 · inactive", html)
        self.assertEqual(trace["after"][70][0], 10)

    def test_base_needs_no_profile_and_current_mode_requires_existing_vectors(self):
        with toy_runtime():
            state = Session()
            result = lab.run_generation(
                state, Settings(max_new_tokens=2), "prompt", lab.MODES[0], 17, "", {}
            )
            self.assertEqual(len(result["runs"]), 1)
            self.assertEqual(result["runs"][0]["trace"]["termination"], "Token limit")
            self.assertEqual(result["features"], [2, 0, 1])
            self.assertIsNone(result["vector_id"])
            with self.assertRaisesRegex(ValueError, "common profile"):
                lab.run_generation(
                    state, Settings(), "prompt", lab.MODES[1], 17, "", {9: 1.0, 17: 0.0}
                )

    def test_local_before_after_is_distinct_from_upstream_steering_and_exports_escape(
        self,
    ):
        result = toy_result()
        base, steered = [r["trace"] for r in result["runs"]]
        self.assertEqual(base["before"][0], [1, 3])
        self.assertEqual(steered["before"][0], [1.5, 3.5])
        self.assertEqual(steered["after"][0], [1.5, 3.75])
        result["prompt"] = '<script>alert("x")</script>'
        result["runs"][0]["response"] = "<script>alert('x')</script>"
        self.assertNotIn("<script>", view.overview(result) + view.metadata(result))
        self.assertIn("&lt;script&gt;", view.overview(result))
        self.assertIn("token ID 0", view.timeline(result))
        json.dumps(result, allow_nan=False)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(evidence_lab, "ARTIFACT_ROOT", Path(directory)),
        ):
            paths = evidence_lab.export_report(Session(), result)
            self.assertIn("Generation audit", Path(paths[0]).read_text())
            self.assertEqual(json.loads(Path(paths[1]).read_text()), result)

    def test_invalid_selection_fails_before_generation(self):
        for text in ("1.5", "-1", "1,1", "1,2,3,4", "1,,2"):
            with self.assertRaises(ValueError):
                feature_ids(text)
        self.assertEqual(feature_ids("1, 2,0"), [1, 2, 0])
        with toy_runtime(), patch.object(runtime, "generate_answer") as generate:
            with self.assertRaisesRegex(ValueError, "feature IDs"):
                lab.run_generation(
                    Session(), Settings(), "prompt", lab.MODES[0], 17, "3", {}
                )
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
