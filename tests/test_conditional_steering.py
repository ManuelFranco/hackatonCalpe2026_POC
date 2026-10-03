from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from PIL import Image
from safetensors.torch import save_file
import torch
from research.conditional_steering import (
    GateTrace,
    LinearGate,
    apply_gated_delta,
    choose_threshold,
    classification_metrics,
    fit_gate,
    fit_linear_probe,
    gate_value,
    leakage_metrics,
    observe_gate,
    pool_visual_tokens,
    validate_gate_layer_order,
)
from research.vector_builders import LayerProfile, VectorResult
from sae_dashboard import (
    artifact_store,
    conditional_gate,
    gate_manifests,
    manifests,
    model_runtime as runtime,
    portable_model,
)
from sae_dashboard.session import Session, Settings

IMAGE = 7  # fake image token id


def make_gate(threshold=0.5, layer=1, mode="hard", temperature=1.0):
    return LinearGate(
        layer=layer,
        weight=torch.tensor([1.0, 0.0]),
        bias=0.0,
        threshold=threshold,
        center=torch.zeros(2),
        scale=1.0,
        target="military_vehicle",
        mode=mode,
        temperature=temperature,
    )


def prompt(image_value, text_value=100.0):
    """Two text tokens, two image tokens, two text tokens; channel 0 drives the gate."""
    ids = torch.tensor([[1, 2, IMAGE, IMAGE, 3, 4]])
    hidden = torch.zeros(1, 6, 2)
    hidden[0, :, 0] = text_value  # e.g. the literal word "car" in the question
    hidden[0, 2:4, 0] = image_value
    return {"input_ids": ids, "hidden": hidden}


class FakeLM:
    """Identity layers driven like cached generation: one prefill, then 1-token steps."""

    def __init__(self, layers=3, decode_steps=2, fail=False):
        self.layers = [torch.nn.Identity() for _ in range(layers)]
        self.decode_steps = decode_steps
        self.fail = fail
        self.config = SimpleNamespace(image_token_id=IMAGE)
        self.steps = []

    def run(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        self.steps.append(hidden.clone())
        return hidden

    def generate(self, input_ids, hidden, **kwargs):
        self.run(hidden.clone())
        if self.fail:
            raise RuntimeError("generation failed")
        for step in range(self.decode_steps):
            self.run(torch.full((1, 1, 2), float(step)))
        return torch.zeros(1, input_ids.shape[1] + self.decode_steps, dtype=torch.long)

    def __call__(self, input_ids, hidden, **kwargs):
        out = self.run(hidden.clone())
        return SimpleNamespace(logits=out[..., :1].repeat(1, 1, 3))


def runtime_patches(fake, layers=(1, 2), last_only=True):
    return [
        patch.object(runtime, "LAYERS", list(layers)),
        patch.object(runtime, "model", fake),
        patch.object(runtime, "processor", SimpleNamespace(decode=lambda *a, **k: "ok")),
        patch.object(runtime, "get_layer_module", side_effect=lambda i: fake.layers[i]),
        patch.object(runtime, "STEER_LAST_TOKEN_ONLY", last_only),
    ]


def run_generation(
    fake, inputs, strengths, gate=None, trace=None, directions=None, **kw
):
    from contextlib import ExitStack

    directions = directions or {
        1: torch.tensor([0.0, 1.0]),
        2: torch.tensor([0.0, 2.0]),
    }
    with ExitStack() as stack:
        for p in runtime_patches(fake, **kw):
            stack.enter_context(p)
        return runtime.generate_answer(
            inputs,
            6,
            4,
            0,
            directions,
            strengths,
            gate=gate,
            gate_trace=trace,
        )


class GateMathTests(unittest.TestCase):
    def test_hard_gate_opens_only_at_or_above_threshold(self):
        self.assertEqual(gate_value(2.0, 1.0), 1.0)
        self.assertEqual(gate_value(1.0, 1.0), 1.0)
        self.assertEqual(gate_value(0.999, 1.0), 0.0)
        self.assertEqual(gate_value(None, -100.0), 0.0)  # no image tokens

    def test_soft_gate_is_a_tempered_sigmoid(self):
        self.assertAlmostEqual(gate_value(1.0, 1.0, "soft", 0.5), 0.5)
        self.assertAlmostEqual(
            gate_value(2.0, 1.0, "soft", 0.5), 1 / (1 + math.exp(-2)), places=6
        )
        self.assertLess(gate_value(-5.0, 1.0, "soft", 0.5), 1e-5)
        with self.assertRaises(ValueError):
            gate_value(1.0, 0.0, "soft", 0.0)
        with self.assertRaises(ValueError):
            gate_value(1.0, 0.0, "maybe")

    def test_only_image_tokens_are_pooled(self):
        hidden = torch.tensor([[100.0, 1.0], [2.0, 3.0], [4.0, 5.0], [-50.0, 9.0]])
        mask = torch.tensor([False, True, True, False])
        torch.testing.assert_close(
            pool_visual_tokens(hidden, mask), torch.tensor([3.0, 4.0])
        )
        changed = hidden.clone()
        changed[[0, 3]] = 1e6  # text tokens cannot move the gate
        torch.testing.assert_close(
            pool_visual_tokens(changed[None], mask), torch.tensor([3.0, 4.0])
        )
        self.assertIsNone(pool_visual_tokens(hidden, torch.zeros(4, dtype=torch.bool)))
        with self.assertRaisesRegex(ValueError, "prefill"):
            pool_visual_tokens(hidden[:1], mask)

    def test_observe_gate_ignores_text_tokens(self):
        gate = make_gate()
        for image_value, expected in ((1.0, 1.0), (0.0, 0.0)):
            for text_value in (-100.0, 100.0):
                inputs = prompt(image_value, text_value)
                trace = observe_gate(
                    gate, inputs["hidden"], inputs["input_ids"][0] == IMAGE, GateTrace()
                )
                self.assertEqual(trace.value, expected)
                self.assertEqual(trace.image_tokens, 2)

    def test_gated_delta_adds_alpha_v_or_nothing(self):
        x = torch.zeros(1, 3, 2)
        v = torch.tensor([1.0, -1.0])
        out = apply_gated_delta(x, v, 2.0, 1.0, last_token_only=True)
        torch.testing.assert_close(out[0, -1], torch.tensor([2.0, -2.0]))
        torch.testing.assert_close(out[0, :-1], torch.zeros(2, 2))
        out = apply_gated_delta(x, v, 2.0, 1.0, last_token_only=False)
        torch.testing.assert_close(out, torch.tensor([2.0, -2.0]).expand(1, 3, 2))
        self.assertIsNone(apply_gated_delta(x, v, 2.0, 0.0, True))
        torch.testing.assert_close(
            apply_gated_delta(x, v, 2.0, 0.25, True)[0, -1], torch.tensor([0.5, -0.5])
        )

    def test_layer_order_rejects_steering_before_the_gate(self):
        validate_gate_layer_order(9, {9: 1.0, 17: 2.0, 3: 0.0})
        with self.assertRaisesRegex(ValueError, r"\[3\]"):
            validate_gate_layer_order(9, {3: 1.0, 17: 2.0})

    def test_gate_validation_rejects_bad_artifacts(self):
        good = make_gate()
        for changes in (
            {"weight": torch.tensor([float("nan"), 0.0])},
            {"center": torch.zeros(3)},
            {"threshold": float("inf")},
            {"scale": 0.0},
            {"mode": "fuzzy"},
            {"temperature": -1.0},
            {"layer": -1},
            {"target": " "},
        ):
            fields = {
                name: getattr(good, name)
                for name in (
                    "layer weight bias threshold center scale target mode temperature"
                ).split()
            }
            with self.assertRaises(ValueError, msg=str(changes)):
                LinearGate(**{**fields, **changes})
        with self.assertRaisesRegex(ValueError, "width"):
            good.validate(hidden_size=3)

    def test_gate_artifact_roundtrip(self):
        gate = make_gate(threshold=0.25, mode="soft", temperature=0.5)
        loaded = LinearGate.from_artifacts(gate.tensors(), gate.config())
        torch.testing.assert_close(loaded.weight, gate.weight)
        self.assertEqual(
            (loaded.layer, loaded.threshold, loaded.mode, loaded.temperature),
            (1, 0.25, "soft", 0.5),
        )
        with self.assertRaises(ValueError):
            LinearGate.from_artifacts({"weight": gate.weight}, gate.config())


class CalibrationTests(unittest.TestCase):
    def data(self, seed):
        g = torch.Generator().manual_seed(seed)
        pos = torch.randn(12, 8, generator=g) + torch.tensor([3.0] + [0.0] * 7)
        neg = torch.randn(12, 8, generator=g) - torch.tensor([3.0] + [0.0] * 7)
        return torch.cat([pos, neg]), torch.tensor([1.0] * 12 + [0.0] * 12)

    def test_probe_is_deterministic_and_separates_classes(self):
        x, y = self.data(0)
        first = fit_linear_probe(x, y, seed=3)
        second = fit_linear_probe(x, y, seed=3)
        torch.testing.assert_close(first["weight"], second["weight"])
        self.assertEqual(first["bias"], second["bias"])
        scores = first["train_scores"]
        self.assertTrue(bool((scores[:12] > scores[12:].max()).all()))

    def test_fit_gate_uses_validation_threshold_with_zero_false_positives(self):
        x, y = self.data(0)
        vx, vy = self.data(1)
        gate = fit_gate(x, y, vx, vy, layer=9, target="military_vehicle")
        val = gate.metadata["validation_metrics"]
        self.assertEqual(val["fp"], 0)
        self.assertEqual(val["tpr"], 1.0)
        self.assertEqual(gate.layer, 9)
        scores = torch.tensor([gate.score(row) for row in vx])
        self.assertEqual(classification_metrics(scores, vy, gate.threshold)["fp"], 0)

    def test_threshold_respects_false_positive_budget(self):
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.75, 0.1])
        labels = torch.tensor([1, 1, 1, 1, 0, 0])
        threshold, metrics = choose_threshold(scores, labels, 0.0)
        self.assertEqual(metrics["fp"], 0)
        self.assertGreater(threshold, 0.75)
        self.assertAlmostEqual(metrics["tpr"], 0.5)
        threshold, metrics = choose_threshold(scores, labels, 0.5)
        self.assertEqual((metrics["fp"], metrics["tpr"]), (1, 1.0))
        with self.assertRaises(ValueError):
            choose_threshold(scores, torch.ones(6), 0.0)

    def test_leakage_metrics(self):
        result = leakage_metrics([2.0, 4.0, -0.5, 0.5, 0.0], [1, 1, 0, 0, 0], eps=0.0)
        self.assertEqual(result["E_target"], 3.0)
        self.assertAlmostEqual(result["E_leak"], 1 / 3)
        self.assertAlmostEqual(result["selectivity"], 9.0)
        self.assertEqual(result["max_abs_leak"], 0.5)


class RuntimeGateTests(unittest.TestCase):
    def test_open_gate_matches_legacy_and_closed_gate_is_zero(self):
        strengths = {1: 1.0, 2: 0.5}
        legacy = FakeLM()
        run_generation(legacy, prompt(1.0), strengths)
        unsteered = FakeLM()
        run_generation(unsteered, prompt(1.0), {1: 0.0, 2: 0.0})

        opened, trace = FakeLM(), GateTrace()
        run_generation(opened, prompt(1.0), strengths, make_gate(), trace)
        self.assertEqual((trace.value, trace.score), (1.0, 1.0))
        for a, b in zip(legacy.steps, opened.steps):
            torch.testing.assert_close(a, b)
        # Prefill and both cached decoding steps are steered with the stored gate.
        self.assertEqual(len(opened.steps), 3)
        for step, base in zip(opened.steps, unsteered.steps):
            torch.testing.assert_close(step[0, -1] - base[0, -1], torch.tensor([0.0, 2.0]))

        closed, trace = FakeLM(), GateTrace()
        run_generation(closed, prompt(0.0), strengths, make_gate(), trace)
        self.assertEqual(trace.value, 0.0)
        reference = FakeLM()
        run_generation(reference, prompt(0.0), {1: 0.0, 2: 0.0})
        for a, b in zip(closed.steps, reference.steps):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_gate_reads_unmodified_states_at_a_shared_steering_layer(self):
        fake, trace = FakeLM(), GateTrace()
        # All-position steering at the gate layer moves the gated channel by +5; if
        # the gate read its own intervention it would score 5 and open.
        gate = make_gate(threshold=0.5, layer=1)
        directions = {1: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 0.0])}
        with patch.object(
            LinearGate, "score", autospec=True, side_effect=LinearGate.score
        ) as score:
            run_generation(
                fake,
                prompt(0.0),
                {1: 5.0, 2: 0.0},
                gate,
                trace,
                directions=directions,
                last_only=False,
            )
        self.assertEqual((trace.score, trace.value), (0.0, 0.0))
        self.assertEqual(score.call_count, 1)  # prefill only, reused while decoding
        torch.testing.assert_close(fake.steps[0], prompt(0.0)["hidden"])

    def test_gate_layer_outside_steering_layers_and_text_only_prompt(self):
        fake, trace = FakeLM(), GateTrace()
        inputs = prompt(1.0)
        inputs["input_ids"] = torch.tensor([[1, 2, 3, 4, 5, 6]])  # no image
        run_generation(fake, inputs, {1: 1.0, 2: 1.0}, make_gate(layer=0), trace)
        self.assertEqual((trace.score, trace.value, trace.image_tokens), (None, 0.0, 0))
        self.assertTrue(all(not layer._forward_hooks for layer in fake.layers))

    def test_invalid_configuration_installs_no_hooks(self):
        fake = FakeLM()
        with self.assertRaisesRegex(ValueError, "before the gate"):
            run_generation(fake, prompt(1.0), {1: 1.0}, make_gate(layer=2), GateTrace())
        with self.assertRaisesRegex(ValueError, "GateTrace"):
            run_generation(fake, prompt(1.0), {1: 1.0}, make_gate(), None)
        used = GateTrace(score=1.0, value=1.0)
        with self.assertRaisesRegex(ValueError, "fresh"):
            run_generation(fake, prompt(1.0), {1: 1.0}, make_gate(), used)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            run_generation(fake, prompt(1.0), {1: 0.0}, make_gate(layer=5), GateTrace())
        self.assertTrue(all(not layer._forward_hooks for layer in fake.layers))

    def test_hooks_are_removed_after_failure(self):
        fake = FakeLM(fail=True)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_generation(fake, prompt(1.0), {1: 1.0}, make_gate(), GateTrace())
        self.assertTrue(all(not layer._forward_hooks for layer in fake.layers))

    def test_next_token_logits_uses_the_gate(self):
        from contextlib import ExitStack

        directions = {1: torch.tensor([3.0, 0.0]), 2: torch.tensor([0.0, 0.0])}
        results = {}
        for name, image_value in (("open", 1.0), ("closed", 0.0)):
            fake, trace = FakeLM(), GateTrace()
            with ExitStack() as stack:
                for p in runtime_patches(fake):
                    stack.enter_context(p)
                base = runtime.next_token_logits(prompt(image_value))
                steered = runtime.next_token_logits(
                    prompt(image_value), directions, {1: 1.0}, make_gate(), trace
                )
            results[name] = float(steered[0] - base[0])
            self.assertTrue(all(not layer._forward_hooks for layer in fake.layers))
        self.assertEqual(results, {"open": 3.0, "closed": 0.0})

    def test_concurrent_requests_keep_separate_gate_traces(self):
        fakes = {True: FakeLM(), False: FakeLM()}
        traces = {}
        barrier = threading.Barrier(2)

        def work(is_vehicle):
            barrier.wait()
            trace = GateTrace()
            with runtime.MODEL_LOCK:
                run_generation(
                    fakes[is_vehicle],
                    prompt(1.0 if is_vehicle else 0.0),
                    {1: 1.0},
                    make_gate(),
                    trace,
                )
            traces[is_vehicle] = trace

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(work, (True, False)))
        self.assertEqual((traces[True].value, traces[False].value), (1.0, 0.0))
        self.assertIsNot(traces[True], traces[False])


def ready_session(gate=None, enabled=False):
    state = Session()
    state.capture_key = Settings().capture_key
    state.profile_id = "profile"
    state.vector_id = "vector"
    state.profile = {9: LayerProfile(torch.zeros(1, 2), torch.ones(1, 2), 10)}
    state.vectors = {
        9: VectorResult(torch.ones(2), torch.tensor([2.0, 0.0]), {}),
        17: VectorResult(torch.ones(2), torch.tensor([0.0, 1.0]), {}),
    }
    state.conditional_gate = gate
    state.conditional_gate_id = "gate-id" if gate else None
    state.gate_enabled = enabled
    return state


class SessionAndWorkflowTests(unittest.TestCase):
    def test_profile_invalidation_keeps_the_gate_and_gate_invalidation_keeps_vectors(self):
        state = ready_session(make_gate(layer=9), enabled=True)
        state.invalidate_profile()
        self.assertFalse(state.vectors)
        self.assertIsNotNone(state.conditional_gate)
        self.assertTrue(state.gate_enabled)
        state = ready_session(make_gate(layer=9), enabled=True)
        state.invalidate_gate()
        self.assertIsNone(state.conditional_gate)
        self.assertFalse(state.gate_enabled)
        self.assertTrue(state.vectors)

    def test_deepcopied_sessions_do_not_share_gate_state(self):
        original = ready_session(make_gate(layer=9), enabled=True)
        clone = copy.deepcopy(original)
        clone.invalidate_gate()
        self.assertIsNotNone(original.conditional_gate)
        self.assertTrue(original.gate_enabled)

    def test_enabling_requires_a_built_gate(self):
        state = ready_session()
        with self.assertRaisesRegex(ValueError, "Build"):
            conditional_gate.set_gate_enabled(state, True)
        self.assertFalse(state.gate_enabled)
        state.conditional_gate = make_gate(layer=9)
        self.assertIn("enabled", conditional_gate.set_gate_enabled(state, True))

    def test_compare_gated_passes_gate_and_reports_trace(self):
        state = ready_session(make_gate(layer=9), enabled=True)
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            if "gate_trace" in kwargs:
                kwargs["gate_trace"].score = 0.0
                kwargs["gate_trace"].value = 0.0
                return "steered"
            return "base"

        with (
            patch.object(runtime, "ensure_models_loaded"),
            patch.object(runtime, "prepare_inputs", return_value=({}, None, 1)),
            patch.object(runtime, "generate_answer", side_effect=generate),
            patch("sae_dashboard.workflow.require_profile"),
        ):
            base, steered, summary = conditional_gate.compare_gated(
                state, Settings(), {9: 1.0, 17: 0.0}, "Describe it."
            )
        self.assertEqual((base, steered), ("base", "steered"))
        self.assertIs(calls[1]["gate"], state.conditional_gate)
        self.assertNotIn("gate", calls[0])
        self.assertIn("not applied", summary)
        self.assertEqual(state.results[-1]["gate"]["value"], 0.0)
        self.assertEqual(state.results[-1]["conditional_gate_id"], "gate-id")
        with (
            patch("sae_dashboard.workflow.require_profile"),
            self.assertRaisesRegex(ValueError, "before the gate"),
        ):
            conditional_gate.compare_gated(
                ready_session(make_gate(layer=17)), Settings(), {9: 1.0}, "x"
            )

    def test_leakage_summary_reports_zero_change_for_closed_gates(self):
        gate = make_gate()
        manifest = SimpleNamespace(name="eval", split="test")

        def record(label, hard, score, value, global_shift, gated_shift):
            return {
                "label": label,
                "hard_negative": hard,
                "gate": {"score": score, "value": value},
                "global_damage_shift": global_shift,
                "gated_damage_shift": gated_shift,
                "gated_delta_damaged_q": gated_shift,
                "gated_delta_intact_q": 0.0,
            }

        records = [
            record(1, False, 2.0, 1.0, 3.0, 3.0),
            record(0, False, -1.0, 0.0, 2.0, 0.0),
            record(0, True, 0.6, 1.0, 1.0, 1.0),
            record(0, False, None, 0.0, 1.0, 0.0),
        ]
        summary = conditional_gate.summarize_leakage(gate, manifest, records)
        self.assertEqual(summary["closed_gate_max_abs_delta"], 0.0)
        self.assertEqual(summary["gated"]["E_target"], 3.0)
        self.assertAlmostEqual(summary["gated"]["E_leak"], 1 / 3)
        self.assertAlmostEqual(summary["global"]["E_leak"], 4 / 3)
        self.assertEqual(summary["hard_negative_fpr"], 1.0)
        self.assertEqual(summary["gate_classification"]["fp"], 1)
        self.assertIn("E_leak", conditional_gate.leakage_markdown(gate, summary))


class GateManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        for name, color in (("a.png", "red"), ("b.png", "blue"), ("c.png", "green")):
            Image.new("RGB", (4, 4), color).save(self.root / name)
        self.patches = [
            patch.object(manifests, "DATA_ROOT", self.root),
            patch.object(gate_manifests, "DATA_ROOT", self.root),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.temp.cleanup()

    def write(self, name, **overrides):
        raw = {
            "version": 1,
            "kind": "binary_gate",
            "name": name,
            "target": "military_vehicle",
            "split": "train",
            "examples": [
                {"id": "v1", "label": "target", "category": "tank", "image": "a.png"},
                {
                    "id": "n1",
                    "label": "non_target",
                    "category": "truck",
                    "hard_negative": True,
                    "image": "b.png",
                },
            ],
            **overrides,
        }
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(raw))
        return path

    def test_loads_labels_and_lists_only_real_gate_manifests(self):
        manifest = gate_manifests.load_gate_manifest(self.write("train"))
        self.assertEqual(manifest.labels, [1, 0])
        self.assertEqual(manifest.summary()["hard_negatives"], 1)
        self.assertEqual(manifest.prompt, gate_manifests.DEFAULT_GATE_PROMPT)
        self.write("template", template=True)
        listed = [p.name for p in gate_manifests.repository_gate_manifests()]
        self.assertEqual(listed, ["train.json"])

    def test_rejects_malformed_gate_manifests(self):
        for overrides, message in (
            ({"template": True}, "template"),
            ({"target": ""}, "target"),
            ({"split": "dev"}, "split"),
            ({"kind": "pairs"}, "binary_gate"),
            ({"examples": [{"id": "x", "label": "maybe", "image": "a.png"}]}, "label"),
            ({"examples": [{"id": "x", "label": 1}]}, "image"),
            (
                {"examples": [{"label": 1, "hard_negative": True, "image": "a.png"}]},
                "hard negative",
            ),
            ({"examples": [{"label": 1, "image": "../escape.png"}]}, "data root"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                gate_manifests.load_gate_manifest(self.write("bad", **overrides))

    def test_split_disjointness_and_reserved_images(self):
        train = gate_manifests.load_gate_manifest(self.write("train"))
        validation = gate_manifests.load_gate_manifest(
            self.write("validation", split="validation")
        )
        with self.assertRaisesRegex(ValueError, "both"):
            gate_manifests.check_disjoint(train, validation)
        (self.root / "damage.test.manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "name": "damage test",
                    "split": "test",
                    "pairs": [{"id": "1", "A": {"image": "a.png"}, "B": {"image": "c.png"}}],
                }
            )
        )
        reserved = gate_manifests.reserved_evaluation_hashes()
        self.assertEqual(len(reserved), 2)
        with self.assertRaisesRegex(ValueError, "reserved"):
            gate_manifests.check_not_reserved(train, reserved)

    def test_loading_gate_manifests_keeps_the_damage_profile(self):
        self.write("train")
        val = self.write(
            "validation",
            split="validation",
            examples=[
                {"id": "v2", "label": 1, "image": "c.png"},
                {"id": "n2", "label": 0, "image": "c.png"},
            ],
        )
        state = ready_session(make_gate(layer=9), enabled=True)
        conditional_gate.set_gate_manifests(state, str(self.root / "train.json"), str(val))
        self.assertTrue(state.vectors)
        self.assertEqual(state.profile_id, "profile")
        self.assertIsNone(state.conditional_gate)  # new data invalidates the old gate
        self.assertFalse(state.gate_enabled)


class ExportTests(unittest.TestCase):
    def load_module(self, unpack):
        spec = importlib.util.spec_from_file_location(
            "exported_gate_model", unpack / "steered_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load(self, module, unpack, layers=30):
        fake = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2)),
            model=SimpleNamespace(
                language_model=SimpleNamespace(layers=[None] * layers)
            ),
        )
        fake.eval = lambda: fake
        with (
            patch.object(
                module.Gemma3ForConditionalGeneration, "from_pretrained", return_value=fake
            ),
            patch.object(module.AutoProcessor, "from_pretrained", return_value=object()),
        ):
            return module.SteeredVLM(unpack)

    def export(self, temp, state, strengths):
        with (
            patch.object(artifact_store, "ARTIFACT_ROOT", Path(temp)),
            patch.object(runtime, "LAYERS", [9, 17]),
        ):
            archive = artifact_store.export_model(state, Settings(seed=5), strengths, False)
        unpack = Path(temp) / "unpacked"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(unpack)
        return unpack

    def test_gated_export_roundtrip(self):
        gate = make_gate(threshold=0.75, layer=9, mode="soft", temperature=0.5)
        with tempfile.TemporaryDirectory() as temp:
            unpack = self.export(
                temp, ready_session(gate, enabled=True), {9: 1.5, 17: -2.0}
            )
            config = json.loads((unpack / "steering_config.json").read_text())
            self.assertEqual(config["format_version"], 2)
            self.assertEqual(config["conditional_gate"]["gate_id"], "gate-id")
            loaded = self.load(self.load_module(unpack), unpack)
        torch.testing.assert_close(loaded.directions["9"], torch.tensor([2.0, 0.0]))
        self.assertEqual(loaded.config["strengths"], {"9": 1.5, "17": -2.0})
        torch.testing.assert_close(loaded.gate["weight"], gate.weight)
        torch.testing.assert_close(loaded.gate["center"], gate.center)
        self.assertEqual(
            (
                loaded.gate["layer"],
                loaded.gate["bias"],
                loaded.gate["threshold"],
                loaded.gate["mode"],
                loaded.gate["temperature"],
            ),
            (9, 0.0, 0.75, "soft", 0.5),
        )

    def test_ungated_export_stays_version_one(self):
        with tempfile.TemporaryDirectory() as temp:
            # A built but disabled gate must not change the legacy package.
            unpack = self.export(
                temp, ready_session(make_gate(layer=9)), {9: 1.0, 17: 0.0}
            )
            config = json.loads((unpack / "steering_config.json").read_text())
            self.assertEqual(config["format_version"], 1)
            self.assertNotIn("conditional_gate", config)
            self.assertFalse((unpack / "conditional_gate.safetensors").exists())
            self.assertIsNone(self.load(self.load_module(unpack), unpack).gate)

    def test_export_rejects_enabled_missing_gate_and_bad_layer_order(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "not built"):
                self.export(temp, ready_session(None, enabled=True), {9: 1.0})
            with self.assertRaisesRegex(ValueError, "before the gate"):
                self.export(
                    temp, ready_session(make_gate(layer=17), enabled=True), {9: 1.0}
                )

    def test_loader_rejects_invalid_gate_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            unpack = self.export(
                temp, ready_session(make_gate(layer=9), enabled=True), {9: 1.0, 17: 0.0}
            )
            module = self.load_module(unpack)
            config_path = unpack / "steering_config.json"
            original = json.loads(config_path.read_text())
            tensors_path = unpack / "conditional_gate.safetensors"
            good_tensors = make_gate().tensors()

            def expect(message, config=None, tensors=None, layers=30):
                config_path.write_text(json.dumps(config or original))
                save_file(tensors or good_tensors, str(tensors_path))
                with self.assertRaisesRegex(ValueError, message):
                    self.load(module, unpack, layers)

            gate = original["conditional_gate"]
            expect("shape", tensors={"weight": torch.zeros(3), "center": torch.zeros(3)})
            expect(
                "NaN",
                tensors={"weight": torch.tensor([float("nan"), 0.0]), "center": torch.zeros(2)},
            )
            expect("exactly", tensors={"weight": torch.zeros(2)})
            expect("does not exist", layers=5)
            expect("mode", config={**original, "conditional_gate": {**gate, "mode": "x"}})
            expect(
                "threshold",
                config={**original, "conditional_gate": {**gate, "threshold": "high"}},
            )
            expect("layer", config={**original, "conditional_gate": {**gate, "layer": -1}})
            expect(
                "precedes",
                config={**original, "conditional_gate": {**gate, "layer": 17}},
            )
            expect(
                "require",
                config={k: v for k, v in original.items() if k != "conditional_gate"},
            )
            expect("version", config={**original, "format_version": 3})
            expect(
                "finite",
                config={**original, "strengths": {"9": float("inf"), "17": 0.0}},
            )

    def test_portable_gate_math_matches_research_gate(self):
        g = torch.Generator().manual_seed(0)
        gate = LinearGate(
            layer=0,
            weight=torch.randn(4, generator=g),
            bias=0.3,
            threshold=0.1,
            center=torch.randn(4, generator=g),
            scale=1.7,
            target="military_vehicle",
            mode="soft",
            temperature=0.4,
        )
        portable = portable_model.validate_gate(gate.config(), gate.tensors(), 4)
        hidden = torch.randn(1, 5, 4, generator=g)
        mask = torch.tensor([False, True, True, True, False])
        expected = observe_gate(gate, hidden, mask, GateTrace())
        actual = portable_model.gate_from_hidden(portable, hidden, mask)
        self.assertAlmostEqual(actual["score"], expected.score, places=5)
        self.assertAlmostEqual(actual["value"], expected.value, places=5)

    def test_portable_hooks_gate_and_cleanup(self):
        layers = [torch.nn.Identity(), torch.nn.Identity()]
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=layers))
        )
        gate = portable_model.validate_gate(
            make_gate(layer=0).config(), make_gate().tensors(), 2
        )
        directions = {"1": torch.tensor([0.0, 1.0])}
        for image_value, expected in ((1.0, 1.0), (0.0, 0.0)):
            inputs = prompt(image_value)
            trace = {}
            with portable_model.steering_hooks(
                model,
                directions,
                {"1": 2.0},
                True,
                gate=gate,
                image_mask=inputs["input_ids"][0] == IMAGE,
                trace=trace,
            ):
                hidden = layers[1](layers[0](inputs["hidden"].clone()))
                step = layers[1](layers[0](torch.zeros(1, 1, 2)))
            self.assertEqual(trace["value"], expected)
            self.assertEqual(float(hidden[0, -1, 1]), 2.0 * expected)
            self.assertEqual(float(step[0, -1, 1]), 2.0 * expected)
            self.assertTrue(all(not layer._forward_hooks for layer in layers))
        with self.assertRaisesRegex(ValueError, "precede"):
            with portable_model.steering_hooks(
                model,
                {"0": torch.zeros(2)},
                {"0": 1.0},
                True,
                gate={**gate, "layer": 1},
                image_mask=torch.zeros(6, dtype=torch.bool),
                trace={},
            ):
                pass
        self.assertTrue(all(not layer._forward_hooks for layer in layers))


class GateUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sae_dashboard.ui import build_demo

        cls.demo = build_demo()

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_gate_controls_exist_and_compare_dispatches_on_session_state(self):
        labels = {c["props"].get("label") for c in self.demo.config["components"]}
        for label in (
            "Gate training manifest",
            "Gate validation manifest",
            "Gate layer",
            "Gate mode",
            "Enable vehicle gate (conditional steering)",
            "Evaluation manifest",
        ):
            self.assertIn(label, labels)
        fn = next(f for f in self.demo.fns.values() if f.name == "run_comparison")
        values = [0, 0.0, 16, "all", "mean", 1.0, 0.0, 0.0, 0.0]
        with (
            patch(
                "sae_dashboard.ui.workflow.compare", return_value=("b", "s")
            ) as legacy,
            patch(
                "sae_dashboard.ui.gating.compare_gated",
                return_value=("b", "g", "trace"),
            ) as gated,
        ):
            state = ready_session(make_gate(layer=9))
            result = fn.fn(state, "prompt", None, *values)
            self.assertEqual(len(result), len(fn.outputs))
            self.assertIn("Gate disabled", result[2])
            state.gate_enabled = True
            self.assertEqual(fn.fn(state, "prompt", None, *values), ("b", "g", "trace"))
        self.assertEqual((legacy.call_count, gated.call_count), (1, 1))

    def test_toggle_without_gate_reports_and_unchecks(self):
        fn = next(f for f in self.demo.fns.values() if f.name == "toggle_gate")
        state = Session()
        status, checkbox = fn.fn(state, True)
        self.assertIn("Build the vehicle gate", status)
        self.assertFalse(checkbox)
        self.assertFalse(state.gate_enabled)


if __name__ == "__main__":
    unittest.main()
