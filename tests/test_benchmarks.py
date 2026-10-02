import unittest
from unittest.mock import patch

from sae_dashboard.benchmarks import (
    ARCChallengeBenchmark,
    BBHBenchmark,
    BoolQBenchmark,
    BenchmarkRunner,
    GSM8KBenchmark,
    MMLUProBenchmark,
    MMMUBenchmark,
    PIQABenchmark,
    POPEBenchmark,
    ScienceQABenchmark,
    TruthfulQABenchmark,
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

    def test_text_benchmark_adapters(self):
        bbh = BBHBenchmark()
        self.assertTrue(bbh.score({"input": "x", "target": "True"}, "True"))

        truthful = TruthfulQABenchmark()
        truthful_item = {
            "question": "Which is true?",
            "mc1_targets": {"choices": ["wrong", "right"], "labels": [0, 1]},
        }
        self.assertTrue(truthful.score(truthful_item, "B"))

        boolq = BoolQBenchmark()
        self.assertTrue(boolq.score({"answer": True}, "YES"))

        piqa = PIQABenchmark()
        self.assertTrue(piqa.score({"label": 1}, "B"))

    def test_multimodal_answer_adapters(self):
        mmmu = MMMUBenchmark()
        item = {"question": "Which?", "options": "['one', 'two']", "answer": "B", "image": object()}
        self.assertIn("B. two", mmmu.format_prompt(item))
        self.assertTrue(mmmu.score(item, "B"))
        self.assertIsNotNone(mmmu.image(item))

        scienceqa = ScienceQABenchmark()
        self.assertTrue(scienceqa.score({"choices": ["one", "two"], "answer": 1}, "B"))

        pope = POPEBenchmark()
        self.assertTrue(pope.score({"answer": "yes"}, "YES"))


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

    def test_runner_passes_the_same_materialized_image_to_both_conditions(self):
        class FakeImage:
            def save(self, path):
                path.write_bytes(b"fake image")

        items = [{"question": "q", "answer": "yes", "image": FakeImage()}]
        seen_images = []

        def prepare_inputs(image, prompt, add_generation_prompt=True):
            seen_images.append(image)
            self.assertTrue(image.endswith(".png"))
            self.assertTrue(__import__("pathlib").Path(image).exists())
            return {}, None, 1

        with patch("sae_dashboard.benchmarks.POPEBenchmark.load", return_value=items):
            result = BenchmarkRunner(prepare_inputs, lambda **kwargs: "yes").run("pope", max_items=None)

        self.assertEqual(result.total, 1)
        self.assertEqual(seen_images[0], seen_images[1])


if __name__ == "__main__":
    unittest.main()