import unittest
from unittest.mock import patch

from sae_dashboard.benchmarks import (
    ARCChallengeBenchmark,
    BenchmarkRunner,
    GSM8KBenchmark,
    MMLUProBenchmark,
)


class BenchmarkAdapterTests(unittest.TestCase):
    def test_mmlu_prompt_and_choice_scoring(self):
        item = {"question": "2 + 2?", "options": ["3", "4"], "answer": "B"}
        adapter = MMLUProBenchmark()
        self.assertIn("B. 4", adapter.format_prompt(item))
        self.assertTrue(adapter.score(item, "The answer is B."))

    def test_gsm8k_extracts_final_number(self):
        item = {"question": "What is 2+2?", "answer": "#### 4"}
        adapter = GSM8KBenchmark()
        self.assertTrue(adapter.score(item, "The calculation is complete. #### 4"))
        self.assertFalse(adapter.score(item, "The calculation is complete. #### 5"))

    def test_arc_prompt_uses_dataset_labels(self):
        item = {
            "question": "Which?",
            "choices": {"label": ["A", "B"], "text": ["one", "two"]},
            "answerKey": "A",
        }
        adapter = ARCChallengeBenchmark()
        self.assertIn("B. two", adapter.format_prompt(item))
        self.assertTrue(adapter.score(item, "A"))


class BenchmarkRunnerTests(unittest.TestCase):
    def test_runner_reports_paired_capability_changes(self):
        items = [
            {"question": "q1", "options": ["no", "yes"], "answer": "B", "category": "demo"},
            {"question": "q2", "options": ["no", "yes"], "answer": "B", "category": "demo"},
            {"question": "q3", "options": ["no", "yes"], "answer": "A", "category": "demo"},
        ]
        answers = iter(["B", "B", "B", "A", "B", "A"])

        def prepare_inputs(image, prompt, add_generation_prompt=True):
            return {}, None, 1

        def generate_answer(**kwargs):
            return next(answers)

        with patch("sae_dashboard.benchmarks.MMLUProBenchmark.load", return_value=items):
            result = BenchmarkRunner(prepare_inputs, generate_answer).run("mmlu_pro", max_items=None)

        summary = result.as_dict()
        self.assertEqual(summary["num_items"], 3)
        self.assertEqual(summary["base_correct"], 2)
        self.assertEqual(summary["steered_correct"], 2)
        self.assertEqual(summary["preserved"], 1)
        self.assertEqual(summary["regressed"], 1)
        self.assertEqual(summary["improved"], 1)
        self.assertEqual(summary["changed"], 2)
        self.assertEqual(summary["categories"]["demo"]["total"], 3)


if __name__ == "__main__":
    unittest.main()