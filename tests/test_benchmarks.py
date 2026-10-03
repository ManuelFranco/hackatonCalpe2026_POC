import unittest
from unittest.mock import patch
from research.benchmarks import BENCHMARKS, BenchmarkRequest, run_benchmark
from research.benchmarks.adapters import IFEval, MBPP, MMMU, MMLUPro, MMLUProStratifiedEasy


class BenchmarkTests(unittest.TestCase):
    def test_requested_registry_and_split_validation(self):
        self.assertEqual(
            set(BENCHMARKS),
            {
                "mmlu_pro",
                "mmlu_pro_stratified_easy",
                "mmmu",
                "pope",
                "ifeval",
                "mbpp",
            },
        )
        with self.assertRaises(ValueError):
            run_benchmark(
                IFEval(),
                BenchmarkRequest("test"),
                lambda *args: ("", False),
                lambda *args: "",
            )

    def test_choice_parsing_does_not_count_articles(self):
        adapter = MMLUPro()
        item = {"question": "Q", "options": ["x", "y"], "answer": "B"}
        self.assertTrue(adapter.score(item, "The answer is B."))
        self.assertIsNone(adapter.normalize(item, "a possible response"))
        self.assertFalse(adapter.score(item, "I cannot answer."))

    def test_paired_regressions_and_reproducible_sampling(self):
        adapter = MMLUPro()
        data = [
            {
                "question_id": i,
                "question": f"Q{i}",
                "options": ["x", "y"],
                "answer": "B",
            }
            for i in range(6)
        ]
        outputs = iter(["B", "B", "A"])
        steered = iter(["B", "A"])
        with patch.object(adapter, "load", return_value=data):
            first = run_benchmark(
                adapter,
                BenchmarkRequest("test", 3, seed=42),
                lambda *args: (next(outputs), False),
                lambda *args: next(steered),
            )
            second = run_benchmark(
                adapter,
                BenchmarkRequest("test", 3, seed=42),
                lambda *args: ("B", False),
                lambda *args: "B",
            )
        self.assertEqual(
            [r["id"] for r in first["rows"]], [r["id"] for r in second["rows"]]
        )
        self.assertEqual(first["summary"]["preserved"], 1)
        self.assertEqual(first["summary"]["regressed"], 1)
        self.assertEqual(first["summary"]["base_failed_excluded"], 1)
        self.assertIsNone(first["rows"][-1]["steered"])
        self.assertEqual(first["summary"]["preservation_rate_on_base_correct"], 0.5)

    def test_empty_dataset_and_invalid_count_fail(self):
        adapter = MMLUPro()
        with patch.object(adapter, "load", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No benchmark"):
                run_benchmark(
                    adapter,
                    BenchmarkRequest("test"),
                    lambda *args: ("", False),
                    lambda *args: "",
                )
            with self.assertRaisesRegex(ValueError, "between"):
                run_benchmark(
                    adapter,
                    BenchmarkRequest("test", 0),
                    lambda *args: ("", False),
                    lambda *args: "",
                )

    def test_mmmu_all_images_and_official_open_answers(self):
        adapter = MMMU()
        first, second = object(), object()
        item = {
            "question_type": "open",
            "question": "Q <image 1> <image 3>",
            "answer": "42",
            "image_1": first,
            "image_3": second,
        }
        self.assertEqual(
            adapter.images(item), [("Image 1", first), ("Image 3", second)]
        )
        self.assertTrue(adapter.score(item, "The answer is 42."))
        self.assertFalse(adapter.score(item, "The answer is 41."))
        self.assertNotIn("letter", adapter.prompt(item))

    def test_ifeval_official_strict_loose_and_null_padded_kwargs(self):
        adapter = IFEval()
        item = {
            "key": 1,
            "prompt": "Say hello.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["hello"], "num_words": None}],
        }
        self.assertEqual(adapter.prompt(item), item["prompt"])
        good = adapter.details(item, "hello")
        self.assertTrue(good["correct"])
        self.assertEqual(good["strict_instructions"], [True])
        self.assertFalse(adapter.details(item, "goodbye")["correct"])
        self.assertFalse(adapter.details(item, "")["loose_correct"])
        with (
            patch.object(adapter, "preflight"),
            patch.object(adapter, "load", return_value=[item]),
        ):
            result = run_benchmark(
                adapter,
                BenchmarkRequest("train"),
                lambda *args: ("hello", False),
                lambda *args: "goodbye",
            )
        self.assertEqual(result["summary"]["base_instruction_strict_accuracy"], 1)
        self.assertEqual(result["summary"]["steered_prompt_loose_accuracy"], 0)

    def test_mmlu_pro_stratified_easy_filters_lowest_difficulty(self):
        adapter = MMLUProStratifiedEasy()
        data = [
            {"difficulty": "-----"},
            {"difficulty": "----"},
        ]
        with patch("research.benchmarks.adapters.BenchmarkAdapter.load", return_value=data):
            filtered = adapter.load("train", "")
        self.assertEqual(len(filtered), 1)

    def test_mbpp_scores_by_executing_all_tests(self):
        adapter = MBPP()
        item = {
            "text": "Return the sum of two numbers.",
            "test_setup_code": "",
            "test_list": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        }
        self.assertIn("Return only executable", adapter.prompt(item))
        self.assertTrue(adapter.details(item, "```python\ndef add(a, b):\n    return a + b\n```")["correct"])
        self.assertFalse(adapter.details(item, "def add(a, b):\n    return a - b")["correct"])

    def test_mbpp_loads_jsonl_without_running_legacy_dataset_script(self):
        adapter = MBPP()
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                return_value="C:/cache/mbpp.jsonl",
            ) as download,
            patch("datasets.load_dataset", return_value=["row"]) as load,
        ):
            self.assertEqual(adapter.load("test", ""), ["row"])
        download.assert_called_once_with(
            repo_id="Muennighoff/mbpp",
            filename="data/mbpp.jsonl",
            repo_type="dataset",
        )
        load.assert_called_once_with(
            "json",
            data_files={"test": "C:/cache/mbpp.jsonl"},
            split="test",
        )
