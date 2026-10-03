from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import torch
from research.vector_builders import LayerProfile, VectorResult
from sae_dashboard import artifact_store, model_runtime as runtime, workflow
from sae_dashboard.manifests import Condition, Manifest, Pair
from sae_dashboard.session import Session, Settings


def ready_session(value=1):
    state = Session()
    state.capture_key = Settings().capture_key
    state.profile_id = "test-profile"
    state.vector_id = "test-vector"
    state.profile = {9: LayerProfile(torch.zeros(1, 2), torch.ones(1, 2), 10)}
    state.vectors = {
        9: VectorResult(
            torch.ones(2),
            torch.tensor([float(value), 0.0]),
            {"method": "mean_difference"},
        )
    }
    return state


class WorkflowTests(unittest.TestCase):
    def test_deepcopied_gradio_sessions_have_distinct_ids_locks_and_data(self):
        original = ready_session()
        clone = copy.deepcopy(original)
        self.assertNotEqual(original.id, clone.id)
        self.assertIsNot(original.lock, clone.lock)
        clone.vectors[9].direction[0] = 3
        self.assertEqual(float(original.vectors[9].direction[0]), 1)
        clone.invalidate_profile()
        self.assertTrue(original.profile)

    def test_experiments_write_only_when_enabled_and_never_overwrite(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp) / "runs"),
        ):
            first, second = Session(), Session()
            artifact_store.record_event(first, "probe", {"answer": "a"})
            self.assertFalse((Path(temp) / "runs").exists())
            first.save_enabled = second.save_enabled = True
            for state in (first, first, second):
                artifact_store.record_event(state, "probe", {"answer": state.id})
            files = list((Path(temp) / "runs").rglob("*.json"))
            self.assertEqual(len(files), 3)
            self.assertEqual(len(list((Path(temp) / "runs").iterdir())), 2)
            first.save_enabled = False
            artifact_store.record_event(first, "probe", {})
            self.assertEqual(len(list((Path(temp) / "runs").rglob("*.json"))), 3)

    def test_profile_vector_compare_pipeline_stays_in_memory(self):
        state = Session()
        state.manifests = (
            Manifest(
                "test",
                (
                    Pair("1", Condition("1"), Condition("3")),
                    Pair("2", Condition("2"), Condition("4")),
                ),
                "hash",
            ),
        )

        def prepare(image, prompt):
            return {"value": float(prompt)}, torch.tensor([1, 2]), 2

        def capture(inputs):
            return {9: torch.tensor([[inputs["value"], 0.0], [0.0, inputs["value"]]])}

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(
                patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp) / "runs")
            )
            stack.enter_context(patch.object(runtime, "LAYERS", [9]))
            stack.enter_context(patch.object(runtime, "ensure_models_loaded"))
            stack.enter_context(
                patch.object(runtime, "prepare_inputs", side_effect=prepare)
            )
            stack.enter_context(
                patch.object(runtime, "capture_prompt_residuals", side_effect=capture)
            )
            stack.enter_context(
                patch.object(
                    runtime, "get_image_mask", return_value=torch.tensor([False, False])
                )
            )
            stack.enter_context(
                patch.object(
                    runtime, "encode_sae_chunked", side_effect=lambda sae, x: x
                )
            )
            stack.enter_context(
                patch.dict(
                    runtime.saes, {9: SimpleNamespace(W_dec=torch.eye(2))}, clear=True
                )
            )
            workflow.build_profile(state, Settings())
            self.assertFalse(state.vectors)
            workflow.create_vectors(state, Settings(), "mean_difference")
            torch.testing.assert_close(state.vectors[9].feature_delta, torch.ones(2))
            stack.enter_context(
                patch.object(
                    runtime, "generate_answer", side_effect=["base", "steered"]
                )
            )
            self.assertEqual(
                workflow.compare(state, Settings(), {9: 1}, "5"), ("base", "steered")
            )
            self.assertFalse((Path(temp) / "runs").exists())
            with self.assertRaisesRegex(ValueError, "current token"):
                workflow.compare(state, Settings(token_scope="last"), {9: 1}, "5")
            workflow.build_profile(state, Settings(token_scope="last"))
            self.assertFalse(state.vectors)

    def test_parallel_sessions_do_not_mix_generations(self):
        states = [ready_session(2), ready_session(7)]
        calls = []

        def generate(**kwargs):
            token = (
                kwargs.get("steering_directions", {})
                .get(9, torch.tensor([0.0]))[0]
                .item()
            )
            calls.append((threading.current_thread().name, token))
            time.sleep(0.005)
            return str(token)

        with (
            patch.object(runtime, "LAYERS", [9]),
            patch.object(runtime, "ensure_models_loaded"),
            patch.object(runtime, "prepare_inputs", return_value=({}, None, 1)),
            patch.object(runtime, "generate_answer", side_effect=generate),
            ThreadPoolExecutor(2) as pool,
        ):
            futures = [
                pool.submit(workflow.compare, state, Settings(), {9: 1}, "prompt")
                for state in states
            ]
            answers = [future.result() for future in futures]
        self.assertEqual(answers, [("0.0", "2.0"), ("0.0", "7.0")])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(calls[2][0], calls[3][0])
        self.assertEqual(states[0].results[0]["steered"], "2.0")
        self.assertEqual(states[1].results[0]["steered"], "7.0")

    def test_export_roundtrip_without_sae_or_dashboard_dependency(self):
        state = ready_session(2)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp)),
            patch.object(runtime, "LAYERS", [9]),
        ):
            archive = workflow.export(state, Settings(seed=17), {9: -2}, False)
            unpack = Path(temp) / "unpacked"
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn("steered_model.py", bundle.namelist())
                for name in (
                    "serve_model.py",
                    "VSCODE.md",
                    "continue.example.yaml",
                    "requirements-server.txt",
                ):
                    self.assertIn(name, bundle.namelist())
                self.assertIn(
                    "-r requirements.txt",
                    bundle.read("requirements-server.txt").decode(),
                )
                self.assertFalse(
                    any(name.startswith("base_model") for name in bundle.namelist())
                )
                bundle.extractall(unpack)
            config = json.loads((unpack / "steering_config.json").read_text())
            self.assertEqual(config["strengths"], {"9": -2})
            spec = importlib.util.spec_from_file_location(
                "exported_model", unpack / "steered_model.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fake = SimpleNamespace(
                config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2))
            )
            fake.eval = lambda: fake
            with (
                patch.object(
                    module.Gemma3ForConditionalGeneration,
                    "from_pretrained",
                    return_value=fake,
                ),
                patch.object(
                    module.AutoProcessor, "from_pretrained", return_value=object()
                ),
            ):
                loaded = module.SteeredVLM(unpack)
            torch.testing.assert_close(loaded.directions["9"], torch.tensor([2.0, 0.0]))
            self.assertEqual(loaded.config["generation"]["seed"], 17)

    def test_full_export_saves_weights_only_on_explicit_export(self):
        state = ready_session()
        calls = []

        def save(path, **kwargs):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / f"{len(calls)}.json").write_text("{}")
            calls.append(kwargs)

        fake = SimpleNamespace(
            save_pretrained=save, config=None, device="cpu", dtype="float32"
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp)),
            patch.object(runtime, "model", fake),
            patch.object(runtime, "processor", fake),
            patch.object(runtime, "ensure_models_loaded"),
        ):
            archive = artifact_store.export_model(state, Settings(), {9: 1}, True)
            self.assertEqual(calls, [{"safe_serialization": True}, {}])
            with zipfile.ZipFile(archive) as bundle:
                self.assertTrue(
                    any(name.startswith("base_model/") for name in bundle.namelist())
                )

    def test_export_requirements_do_not_pin_platform_specific_cuda_suffixes(self):
        metadata = runtime.runtime_metadata()
        metadata["packages"]["torch"] = "2.6.0+cu124"
        metadata["packages"]["torchvision"] = "0.21.0+cu124"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp)),
            patch.object(runtime, "runtime_metadata", return_value=metadata),
        ):
            archive = artifact_store.export_model(
                ready_session(), Settings(), {9: 1}, False
            )
            with zipfile.ZipFile(archive) as bundle:
                requirements = bundle.read("requirements.txt").decode()
                self.assertIn("torch==2.6.0\n", requirements)
                self.assertNotIn("+cu", requirements)
                config = json.loads(bundle.read("steering_config.json"))
                self.assertEqual(config["runtime"]["packages"]["torch"], "2.6.0+cu124")
