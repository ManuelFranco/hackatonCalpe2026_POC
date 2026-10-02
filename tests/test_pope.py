"""POPE scoring and multimodal integration without downloading model weights."""

import json
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from research.benchmarks import (
    BENCHMARKS,
    BenchmarkRequest,
    evaluate_base,
    evaluate_steered,
    prepare_benchmark,
)
from research.benchmarks.adapters import POPE


class POPETests(unittest.TestCase):
    def setUp(self):
        self.adapter = POPE()
        self.item = {
            "id": "0",
            "question_id": "1",
            "question": "Is there a cat in the image?",
            "answer": "yes",
            "image": Image.new("RGB", (4, 4), "blue"),
            "image_source": "fixture-cat",
            "category": "random",
        }

    def test_each_variant_loads_explicit_full_config(self):
        self.assertIsInstance(BENCHMARKS["pope"], POPE)
        self.assertEqual(self.adapter.default_split, "random")
        with patch("research.benchmarks.adapters.load_dataset") as load:
            for split in self.adapter.splits:
                self.adapter.load(split, "")
                load.assert_called_with(self.adapter.dataset_id, "Full", split)
            load.reset_mock()
            with self.assertRaisesRegex(ValueError, "subject filter"):
                self.adapter.load("random", "animals")
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "supported split"):
            prepare_benchmark(self.adapter, BenchmarkRequest("test"))

    def test_strict_yes_no_never_guesses_from_prose_or_empty_answers(self):
        for label in ("yes", "no"):
            item = dict(self.item, answer=label)
            for response in (label, label.upper(), f"  {label}.\n", f"{label}!"):
                with self.subTest(label=label, response=response):
                    self.assertTrue(self.adapter.score(item, response))
            wrong = "no" if label == "yes" else "yes"
            self.assertFalse(self.adapter.score(item, wrong))
            for response in (
                "",
                "unknown",
                "yesterday",
                "nobody",
                "yes or no",
                "yes, no",
                "No, there is a cat.",
                "Yes. Actually no.",
                "I cannot tell.",
                "The answer is yes.",
                "yes?",
                "not yes",
            ):
                with self.subTest(label=label, response=response):
                    details = self.adapter.details(item, response)
                    self.assertFalse(details["correct"])
                    self.assertFalse(details["valid_answer"])
                    self.assertIsNone(details["prediction"])

    def test_malformed_items_fail_before_inference(self):
        for changes in (
            {"answer": None},
            {"answer": "?"},
            {"answer": ""},
            {"image": None},
            {"question": " "},
        ):
            with (
                self.subTest(changes=changes),
                patch.object(
                    self.adapter, "load", return_value=[dict(self.item, **changes)]
                ),
            ):
                with self.assertRaisesRegex(ValueError, "POPE requires"):
                    prepare_benchmark(self.adapter, BenchmarkRequest("random", 1))

    def test_preview_and_paired_run_keep_images_and_exclude_base_failures(self):
        data = [
            dict(self.item, id=str(i), image=Image.new("RGB", (4, 4), color))
            for i, color in enumerate(("red", "blue", "green"))
        ]
        request = BenchmarkRequest("random", 3, seed=42)
        with patch.object(self.adapter, "load", return_value=data):
            sample = prepare_benchmark(self.adapter, request)
            repeated = prepare_benchmark(self.adapter, request)
        self.assertEqual(
            [item["index"] for item in sample.items],
            [item["index"] for item in repeated.items],
        )
        for item in sample.items:
            self.assertEqual(
                item["images"], [("Image 1", data[item["index"]]["image"])]
            )
            self.assertEqual(item["reference"], "yes")
            self.assertEqual(item["question_type"], "visual yes/no")
            self.assertEqual(item["image_source"], "fixture-cat")
            # Changing ground truth must not change the input seen by the model.
            self.assertEqual(
                item["prompt"], self.adapter.prompt(dict(item["raw"], answer="no"))
            )

        base = Mock(side_effect=[("YES.", True), ("yes", False), ("no", False)])
        rows = evaluate_base(sample, base)
        steered = Mock(side_effect=["yes", "no"])
        result = evaluate_steered(sample, rows, steered)
        self.assertEqual(steered.call_count, 2)
        for call, item in zip(base.call_args_list, sample.items):
            self.assertEqual(call.args, (item["images"], item["prompt"]))
        for call, item in zip(steered.call_args_list, sample.items[:2]):
            self.assertEqual(call.args, (item["images"], item["prompt"]))
        self.assertEqual(result["summary"]["base_accuracy_all_items"], 2 / 3)
        self.assertEqual(result["summary"]["base_cache_hits"], 1)
        self.assertEqual(result["summary"]["preserved"], 1)
        self.assertEqual(result["summary"]["regressed"], 1)
        self.assertEqual(result["summary"]["preservation_rate_on_base_correct"], 0.5)
        self.assertFalse(result["rows"][0]["changed"])
        self.assertTrue(result["rows"][1]["changed"])
        self.assertIsNone(result["rows"][2]["steered"])
        json.dumps(result)  # No image objects leak into exported metrics.

    def test_unparseable_base_answer_is_excluded_even_for_negative_label(self):
        with patch.object(
            self.adapter, "load", return_value=[dict(self.item, answer="no")]
        ):
            sample = prepare_benchmark(self.adapter, BenchmarkRequest("random", 1))
        rows = evaluate_base(sample, lambda *args: ("I cannot tell.", False))
        steered = Mock()
        result = evaluate_steered(sample, rows, steered)
        steered.assert_not_called()
        self.assertIsNone(result["summary"]["preservation_rate_on_base_correct"])
