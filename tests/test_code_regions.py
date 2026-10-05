from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from research.code_regions import select_regions, aligned_token_mask
from sae_dashboard import model_runtime as runtime, capture_view, workflow
from sae_dashboard.session import Session, Settings
from sae_dashboard.manifests import DATA_ROOT, load_manifest


def character_tokenizer(text, **kwargs):
    return {
        "input_ids": [ord(c) for c in text],
        "offset_mapping": [(i, i + 1) for i in range(len(text))],
    }


class CodeRegionTests(unittest.TestCase):
    def test_execution_region_excludes_display_only_interpolation(self):
        text = 'Context\n```python\ndef lookup(db, value):\n    query = "SELECT id FROM t WHERE x = ?"\n    preview = f"SELECT id FROM t WHERE x = \'{value}\'"\n    rows = db.execute(query, (value,)).fetchall()\n    return preview, rows\n```\nTrailing prose'
        snippets = {
            scope: "\n".join(text[a:b] for a, b in select_regions(text, scope))
            for scope in ("code", "sql_query", "sql_call", "sql_flow")
        }
        self.assertIn("preview =", snippets["code"])
        self.assertNotIn("Context", snippets["code"])
        self.assertNotIn("Trailing prose", snippets["code"])
        self.assertIn('query = "SELECT', snippets["sql_query"])
        self.assertNotIn("preview", snippets["sql_query"])
        self.assertNotIn("query =", snippets["sql_call"])
        self.assertIn("db.execute", snippets["sql_call"])
        self.assertIn("query =", snippets["sql_flow"])
        self.assertIn("db.execute", snippets["sql_flow"])

    def test_latest_local_reassignment_and_starred_arguments_are_followed(self):
        text = '```python\ndef lookup(db, value):\n    query = "SELECT 1"\n    query = f"SELECT {value}"\n    request = (query,)\n    return db.execute(*request)\n```'
        selected = "\n".join(text[a:b] for a, b in select_regions(text, "sql_query"))
        self.assertNotIn('"SELECT 1"', selected)
        self.assertIn('f"SELECT {value}"', selected)
        self.assertIn("request =", selected)

    def test_unicode_ast_offsets_match_exact_chat_tokens(self):
        text = "Context\n```python\ndef lookup(db, value):\n    café = f\"SELECT 'é', {value}\"\n    return db.execute(café)\n```"
        spans = select_regions(text, "sql_flow")
        self.assertTrue(any(text[a:b].startswith("café =") for a, b in spans))
        chat = "<bos>user\n" + text + "\nassistant\n"
        ids = torch.tensor([ord(c) for c in chat])
        mask = aligned_token_mask(character_tokenizer, chat, ids, text, spans)
        selected = "".join(c for c, keep in zip(chat, mask) if keep)
        self.assertIn("café", selected)
        self.assertNotIn("assistant", selected)
        with self.assertRaisesRegex(ValueError, "differ"):
            aligned_token_mask(character_tokenizer, chat, ids + 1, text, spans)
        with self.assertRaisesRegex(ValueError, "uniquely"):
            aligned_token_mask(character_tokenizer, text + text, [], text, spans)

    def test_multiline_calls_crlf_multiple_blocks_and_non_python_code(self):
        text = 'Preamble\r\n```python\r\ndb.execute(\r\n    "SELECT 1"\r\n)\r\n```\r\n```c\r\nint main(void) { return 0; }\r\n```'
        self.assertEqual(len(select_regions(text, "code")), 2)
        selected = "".join(text[a:b] for a, b in select_regions(text, "sql_call"))
        self.assertIn('"SELECT 1"', selected)
        self.assertNotIn("int main", selected)
        for text in ("Nothing to capture", "```python\nx = 1", "```python\nx =\n```"):
            with self.assertRaises(ValueError):
                select_regions(text, "sql_call")

    def test_trimmed_chat_preserves_source_offsets_and_excludes_template_tokens(self):
        prompt = '\r\n\t\r\ncafé = "SELECT 1"\r\ndb.execute(café)\r\n \t'
        prefix, suffix = "<bos><start_of_turn>user\n", "<end_of_turn>\nmodel\n"
        rendered = prefix + prompt.strip() + suffix
        ids = torch.tensor([ord(c) for c in rendered])
        for scope in ("code", "sql_query", "sql_call", "sql_flow"):
            with self.subTest(scope=scope):
                mask = aligned_token_mask(
                    character_tokenizer,
                    rendered,
                    ids,
                    prompt,
                    select_regions(prompt, scope),
                )
                self.assertFalse(any(mask[: len(prefix)]))
                self.assertFalse(any(mask[-len(suffix) :]))
                self.assertIn(
                    "café", "".join(c for c, keep in zip(rendered, mask) if keep)
                )
                if scope == "code":
                    self.assertEqual(
                        "".join(c for c, keep in zip(rendered, mask) if keep),
                        prompt.strip(),
                    )
        with self.assertRaisesRegex(ValueError, "differ"):
            aligned_token_mask(
                character_tokenizer, rendered, ids + 1, prompt, [(0, len(prompt))]
            )
        for rendered in (
            prompt.strip() * 2,
            prefix + prompt.strip().replace("SELECT", "ALTERED") + suffix,
        ):
            with (
                self.subTest(rendered=rendered),
                self.assertRaisesRegex(ValueError, "uniquely"),
            ):
                aligned_token_mask(
                    character_tokenizer, rendered, [], prompt, [(0, len(prompt))]
                )

    def test_repository_code_profiles_align_with_message_trimming(self):
        prefix, suffix = "<bos><start_of_turn>user\n", "<end_of_turn>\nmodel\n"
        fake = SimpleNamespace(
            tokenizer=character_tokenizer,
            apply_chat_template=lambda messages, **kwargs: (
                prefix + messages[0]["content"][0]["text"].strip() + suffix
            ),
        )
        with patch.object(runtime, "processor", fake):
            for family in ("sql_injection", "command_injection", "xss"):
                for path in (DATA_ROOT / family).glob("manifest*.json"):
                    for pair in load_manifest(path).pairs:
                        for condition in (pair.a, pair.b):
                            rendered = fake.apply_chat_template(
                                runtime.make_messages(None, condition.text)
                            )
                            ids = torch.tensor([ord(c) for c in rendered])
                            scopes = (
                                ("code", "sql_query", "sql_call", "sql_flow")
                                if family == "sql_injection"
                                else ("code",)
                            )
                            for scope in scopes:
                                with self.subTest(path=path, pair=pair.id, scope=scope):
                                    mask, detail = runtime.profile_token_selection(
                                        ids, None, condition.text, None, scope
                                    )
                                    self.assertGreater(detail["selected_tokens"], 0)
                                    self.assertFalse(bool(mask[: len(prefix)].any()))
                                    self.assertFalse(bool(mask[-len(suffix) :].any()))

    def test_new_scopes_are_opt_in_and_profile_aggregation_uses_selected_tokens(self):
        settings = Settings(token_scope="sql_call")
        settings.validate()
        self.assertNotEqual(settings.capture_key, Settings().capture_key)
        residual = torch.tensor([[1.0, 2.0], [5.0, 8.0], [100.0, 100.0]])
        with patch.object(runtime, "encode_sae_chunked", side_effect=lambda _, x: x):
            score, norm = workflow.capture_score(
                residual,
                torch.zeros(3, dtype=torch.bool),
                None,
                settings,
                torch.tensor([False, True, False]),
            )
        torch.testing.assert_close(score, residual[1])
        self.assertAlmostEqual(norm, float(residual[1].norm()))
        with self.assertRaisesRegex(ValueError, "text-only"):
            runtime.profile_token_selection([], [], "", object(), "code")

    def test_runtime_checks_alignment_and_preview_escapes_source(self):
        text = '```python\ndb.execute("SELECT <script>")\n```'
        chat = "<bos>" + text + "<assistant>"
        fake = SimpleNamespace(
            tokenizer=character_tokenizer, apply_chat_template=lambda *a, **kw: chat
        )
        with patch.object(runtime, "processor", fake):
            mask, detail = runtime.profile_token_selection(
                torch.tensor([ord(c) for c in chat]), None, text, None, "sql_call"
            )
        self.assertEqual(detail["selected_tokens"], int(mask.sum()))
        state = Session()
        state.capture_details = [{**detail, "text": text, "label": "case"}]
        html = capture_view.render(state)
        self.assertIn("<mark>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)

    def test_all_sql_pairs_have_nonempty_regions_in_every_scope(self):
        for path in (DATA_ROOT / "sql_injection").glob("manifest*.json"):
            manifest = load_manifest(path)
            for pair in manifest.pairs:
                for condition in (pair.a, pair.b):
                    for scope in ("code", "sql_query", "sql_call", "sql_flow"):
                        self.assertTrue(
                            select_regions(condition.text, scope),
                            (path, pair.id, scope),
                        )
