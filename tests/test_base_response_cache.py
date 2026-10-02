from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import torch
from sae_dashboard.base_response_cache import get_or_generate, tensor_fingerprint


class BaseCacheTests(unittest.TestCase):
    def test_exact_reuse_and_all_identity_fields_separate_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            request = {
                "prompt": "Q",
                "seed": 1,
                "temperature": 0.5,
                "max_new_tokens": 40,
                "model": "commit-a",
                "image": tensor_fingerprint(torch.zeros(1, 3)),
            }
            generate = Mock(return_value="Answer")
            self.assertEqual(
                get_or_generate(request, generate, directory), ("Answer", False)
            )
            self.assertEqual(
                get_or_generate(request, generate, directory), ("Answer", True)
            )
            self.assertEqual(generate.call_count, 1)
            changes = {
                "prompt": "Other Q",
                "seed": 2,
                "temperature": 0.6,
                "max_new_tokens": 41,
                "model": "commit-b",
                "image": tensor_fingerprint(torch.ones(1, 3)),
            }
            for field, value in changes.items():
                self.assertFalse(
                    get_or_generate(request | {field: value}, generate, directory)[1]
                )
            self.assertEqual(generate.call_count, 7)
            saved = [json.loads(p.read_text()) for p in Path(directory).glob("*.json")]
            self.assertEqual(len(saved), 7)
            self.assertTrue(all(row["answer"] == "Answer" for row in saved))

    def test_concurrent_sessions_generate_once_and_corrupt_json_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            generate = Mock(return_value="B")
            with ThreadPoolExecutor(4) as pool:
                results = list(
                    pool.map(
                        lambda _: get_or_generate({"prompt": "Q"}, generate, directory),
                        range(4),
                    )
                )
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(sum(hit for _, hit in results), 3)
            path = next(Path(directory).glob("*.json"))
            path.write_text("{corrupt")
            self.assertFalse(get_or_generate({"prompt": "Q"}, generate, directory)[1])
            self.assertEqual(json.loads(path.read_text())["answer"], "B")

    def test_generation_failure_never_creates_cached_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                get_or_generate(
                    {"prompt": "Q"}, Mock(side_effect=RuntimeError("failed")), directory
                )
            self.assertFalse(list(Path(directory).glob("*.json")))
