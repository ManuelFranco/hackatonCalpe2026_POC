from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from research.benchmarks import prepare_benchmark, BenchmarkRequest
from research.benchmarks.adapters import MMLUPro, MMMU
from research.vector_builders import LayerProfile, VectorResult
from sae_dashboard import workflow, base_response_cache, model_runtime as runtime
from sae_dashboard.session import Session, Settings


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_preview_retains_prompt_images_metadata_and_reference_without_inference(
        self,
    ):
        adapter = MMMU()
        image = Image.new("RGB", (2, 2), "blue")
        item = {
            "id": "x",
            "question": "Which color <image 1>?",
            "options": ["red", "blue"],
            "answer": "B",
            "image_1": image,
            "subfield": "Art",
            "question_type": "multiple-choice",
        }
        with patch.object(adapter, "load", return_value=[item]):
            sample = prepare_benchmark(adapter, BenchmarkRequest("validation"))
        self.assertEqual(sample.items[0]["reference"], "B. blue")
        self.assertEqual(sample.items[0]["category"], "Art")
        self.assertEqual(sample.items[0]["images"], [("Image 1", image)])
        self.assertEqual(sample.items[0]["prompt"], adapter.prompt(item))
        item["answer"] = "?"
        with patch.object(adapter, "load", return_value=[item]):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                prepare_benchmark(adapter, BenchmarkRequest("test"))

    def test_repeated_runs_cache_only_base_and_skip_wrong_cases(self):
        state = Session()
        state.save_enabled = True
        state.profile = {9: LayerProfile(torch.zeros(1, 2), torch.ones(1, 2), 1)}
        state.capture_key = Settings().capture_key
        state.vectors = {9: VectorResult(torch.ones(2), torch.ones(2))}
        adapter = MMLUPro()
        dataset = [
            {
                "question_id": i,
                "question": f"Q{i}",
                "options": ["x", "y"],
                "answer": "B",
            }
            for i in range(2)
        ]
        fake_model = SimpleNamespace(
            config=SimpleNamespace(to_dict=lambda: {"commit": "123"}),
            generation_config=SimpleNamespace(to_dict=lambda: {"top_p": 0.95}),
        )
        calls = []

        def inputs(images, prompt):
            return {"input_ids": torch.tensor([[0 if "Q0" in prompt else 1]])}, None, 1

        def generate(**kwargs):
            steered = "steering_directions" in kwargs
            identifier = kwargs["inputs"]["input_ids"].item()
            calls.append((identifier, steered))
            return "B" if identifier == 0 else "A"

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(base_response_cache, "CACHE_ROOT", Path(temp)),
            patch.dict(workflow.BENCHMARKS, {"mmlu_pro": adapter}),
            patch.object(adapter, "load", return_value=dataset),
            patch.object(runtime, "LAYERS", [9]),
            patch.object(runtime, "model", fake_model),
            patch.object(
                runtime, "runtime_metadata", return_value={"model_commit": "123"}
            ),
            patch.object(runtime, "ensure_models_loaded"),
            patch.object(runtime, "prepare_inputs", side_effect=inputs),
            patch.object(runtime, "generate_answer", side_effect=generate),
            patch.object(workflow, "record_event") as record,
        ):
            workflow.prepare_evaluation(state, "mmlu_pro", "test", 2, "", Settings())
            workflow.benchmark_base(state, Settings())
            first = workflow.benchmark_steered(state, Settings(), {9: 0})
            workflow.benchmark_base(state, Settings())
            second = workflow.benchmark_steered(state, Settings(), {9: 1})
            self.assertEqual(second["summary"]["base_cache_hits"], 2)
            self.assertEqual(first["summary"]["steered_evaluated"], 1)
            self.assertEqual(calls.count((0, False)), 1)
            self.assertEqual(calls.count((1, False)), 1)
            self.assertEqual(calls.count((0, True)), 2)
            self.assertNotIn((1, True), calls)
            self.assertEqual(len(list(Path(temp).glob("*.json"))), 2)
            record.assert_not_called()
            with self.assertRaisesRegex(ValueError, "temperature"):
                workflow.benchmark_steered(state, Settings(temperature=0.3), {9: 1})

    def test_zero_eligible_cases_report_no_preservation_rate(self):
        from research.benchmarks import evaluate_base, evaluate_steered

        adapter = MMLUPro()
        with patch.object(
            adapter,
            "load",
            return_value=[{"question": "Q", "options": ["x", "y"], "answer": "B"}],
        ):
            sample = prepare_benchmark(adapter, BenchmarkRequest("test"))
        rows = evaluate_base(sample, lambda *args: ("A", False))

        def never(*args):
            raise AssertionError("No steered inference is allowed")

        result = evaluate_steered(sample, rows, never)
        self.assertIsNone(result["summary"]["preservation_rate_on_base_correct"])
        self.assertEqual(result["summary"]["steered_evaluated"], 0)
