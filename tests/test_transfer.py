from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

from research.transfer import decision_readout, summarize_cell
from research.code_regions import code_blocks
from research.vector_builders import LayerProfile, VectorResult
from sae_dashboard import (
    transfer_lab as lab,
    transfer_view as view,
    model_runtime as runtime,
    artifact_store,
)
from sae_dashboard.manifests import DATA_ROOT, Manifest, Pair, Condition
from sae_dashboard.session import Session, Settings
from test_feature_evidence import toy_app


def session():
    state = Session()
    state.capture_key, state.profile_id, state.vector_id = (
        Settings().capture_key,
        "profile",
        "vector",
    )
    state.profile = {9: LayerProfile(torch.zeros(1, 3), torch.ones(1, 3), 2)}
    state.vectors = {9: VectorResult(torch.ones(3), torch.tensor([1.0, 0.0]))}
    state.manifests = (
        Manifest(
            "Training",
            (
                Pair(
                    "train",
                    Condition("BOUND training"),
                    Condition("INTERPOLATED training"),
                ),
            ),
            "training-hash",
        ),
    )
    return state


def write_manifest(directory):
    path = Path(directory) / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "name": "Target <script>",
                "split": "validation",
                "labels": {"A": "Binds values", "B": "Interpolates values"},
                "pairs": [
                    {
                        "id": f"case-{i}",
                        "A": {"text": f"BOUND {i}"},
                        "B": {"text": f"INTERPOLATED {i}"},
                    }
                    for i in range(3)
                ],
            }
        )
    )
    return str(path)


class TransferTests(unittest.TestCase):
    def test_readout_matches_last_and_all_hooks_and_cleans_up_on_failure(self):
        app = toy_app()
        inputs, _, _ = app.prepare_inputs(None, "BOUND")
        before = inputs["hidden"].clone()
        for last in (True, False):
            measured = decision_readout(
                app, inputs, [0, 1], {9: torch.tensor([1.0, 0.0])}, {9: 0.25}, last
            )
            self.assertTrue(measured["input_changed"])
            self.assertAlmostEqual(
                float(app.model.seen[0, -1, 0]), float(before[0, -1, 0]) + 0.25
            )
            self.assertAlmostEqual(
                float(app.model.seen[0, 0, 0]),
                float(before[0, 0, 0]) + (0 if last else 0.25),
            )
            self.assertFalse(app.model.layer._forward_hooks)
        app.model.fail = True
        with self.assertRaises(RuntimeError):
            decision_readout(app, inputs, [0, 1], {9: torch.ones(2)}, {9: 1})
        self.assertFalse(app.model.layer._forward_hooks)
        torch.testing.assert_close(inputs["hidden"], before)

    def test_summary_includes_base_failures_false_alarms_and_ties(self):
        base = [
            {"case_id": "a", "expected": "A", "prediction": "B"},
            {"case_id": "b", "expected": "B", "prediction": "B"},
        ]
        treated = [
            {
                "case_id": "a",
                "expected": "A",
                "prediction": "A",
                "input_changed": True,
                "label_mass": 0.9,
            },
            {
                "case_id": "b",
                "expected": "B",
                "prediction": "Tie",
                "input_changed": True,
                "label_mass": 0.1,
            },
        ]
        score = summarize_cell(base, treated)
        self.assertEqual(
            (
                score["corrections"],
                score["regressions"],
                score["n"],
                score["low_label_mass"],
            ),
            (1, 1, 2, 1),
        )
        self.assertEqual(score["delta_accuracy"], 0)
        self.assertEqual(score["ties_b"], 1)

    def test_frozen_vectors_survive_input_changes_and_do_not_alias_live_tensors(self):
        state = session()
        with patch.multiple(runtime, LAYERS=[9], SAE_IDS={9: "toy"}):
            identifier = lab.freeze_current(state, Settings(), {9: 0.25}, "SQL")
            state.vectors[9].direction[0] = 99
            state.invalidate_profile()
            snapshot = state.transfer_sources[identifier]
            self.assertEqual(snapshot.directions[9][0], 1)
            self.assertEqual(snapshot.strengths[9], 0.25)
            self.assertFalse(state.profile)
            self.assertIsNone(state.transfer_plan)
            self.assertFalse(snapshot.directions[9].requires_grad)

    def test_matrix_reuses_each_base_once_and_pairs_remain_balanced(self):
        state, app = session(), toy_app()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    artifact_store, "ARTIFACT_ROOT", Path(directory) / "automatic"
                )
            )
            stack.enter_context(
                patch.object(lab, "ARTIFACT_ROOT", Path(directory) / "explicit")
            )
            stack.enter_context(
                patch.multiple(
                    runtime,
                    **vars(app),
                    LAYERS=[9],
                    SAE_IDS={9: "toy"},
                    ensure_models_loaded=lambda **kw: None,
                    runtime_metadata=lambda: {"model_commit": "toy-revision"},
                )
            )
            first = lab.freeze_current(state, Settings(), {9: 0.25}, "SQL")
            second = lab.freeze_current(state, Settings(), {9: -0.25}, "Other")
            path = write_manifest(directory)
            plan = lab.prepare_transfer(state, [first, second], [path], 2, 42)
            keys = [c["case_id"] for c in plan["targets"][0]["cases"]]
            self.assertEqual(
                [c["expected"] for c in plan["targets"][0]["cases"]],
                ["A", "B", "A", "B"],
            )
            self.assertEqual(
                keys,
                [
                    c["case_id"]
                    for c in lab.prepare_transfer(
                        state, [first, second], [path], 2, 42
                    )["targets"][0]["cases"]
                ],
            )
            with patch.object(
                runtime.model, "forward", wraps=runtime.model.forward
            ) as forward:
                result = lab.run_transfer(state)
            self.assertEqual(forward.call_count, 12)
            self.assertEqual(result["forwards"], 12)
            self.assertEqual(len(result["cells"]), 2)
            self.assertTrue(all(c["summary"]["n"] == 4 for c in result["cells"]))
            self.assertFalse((Path(directory) / "automatic").exists())
            exported = lab.export_report(state)
            report = Path(exported[0]).read_text()
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            self.assertEqual(
                json.loads(Path(exported[1]).read_text())["sources"][0]["strengths"],
                {"9": 0.25},
            )
            self.assertIn("False alarms", view.matrix(result))
            runtime.model.fail = True
            with self.assertRaises(RuntimeError):
                lab.run_transfer(state)
            self.assertFalse(runtime.model.layer._forward_hooks)

    def test_overlap_is_marked_and_missing_labels_fail_before_model_loading(self):
        state = session()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(runtime, LAYERS=[9], SAE_IDS={9: "toy"}),
        ):
            source = lab.freeze_current(state, Settings(), {9: 1}, "SQL")
            path = Path(write_manifest(directory))
            raw = json.loads(path.read_text())
            raw["pairs"][0]["A"]["text"] = "BOUND training"
            path.write_text(json.dumps(raw))
            plan = lab.prepare_transfer(state, [source], [str(path)], 3, 0)
            self.assertEqual(plan["targets"][0]["overlap"][source], 1)
            for labels in ({"A": 1, "B": "Raw"}, [], {"A": "Same", "B": "Same"}):
                raw["labels"] = labels
                path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "labels"):
                    lab.prepare_transfer(state, [source], [str(path)], 3, 0)
            del raw["labels"]
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "labels.A"):
                lab.prepare_transfer(state, [source], [str(path)], 3, 0)

    def test_new_dataset_labels_match_fixture_behavior_without_running_processes(self):
        for family in ("command_injection", "xss"):
            for path in (DATA_ROOT / family).glob("manifest*.json"):
                raw = json.loads(path.read_text())
                for pair in raw["pairs"]:
                    for side in ("A", "B"):
                        code = code_blocks(pair[side]["text"])[0].text
                        compile(code, str(path), "exec")
                        namespace = {}
                        if family == "xss":
                            exec(code, namespace)
                            args = ["<b>probe</b>"] * len(
                                inspect.signature(namespace["render"]).parameters
                            )
                            value = namespace["render"](*args)
                            rendered = (
                                value["html_response"]
                                if isinstance(value, dict)
                                else value
                            )
                            self.assertEqual(
                                "<b>probe</b>" in rendered,
                                side == "B",
                                (path, pair["id"], side),
                            )
                        else:
                            with (
                                patch("subprocess.run") as run,
                                patch("subprocess.check_output") as output,
                                patch("subprocess.Popen") as process,
                            ):
                                process.return_value = MagicMock()
                                exec(code, namespace)
                                args = ["alpha beta"] * len(
                                    inspect.signature(namespace["lookup"]).parameters
                                )
                                namespace["lookup"](*args)
                                invoked = next(
                                    mock
                                    for mock in (run, output, process)
                                    if mock.called
                                )
                                command = invoked.call_args.args[0]
                                if isinstance(command, list) and command[:2] == [
                                    "/bin/sh",
                                    "-c",
                                ]:
                                    command = command[2]
                                argv = (
                                    shlex.split(command)
                                    if isinstance(command, str)
                                    else command
                                )
                                self.assertEqual(
                                    argv.count("alpha beta") == len(args),
                                    side == "A",
                                    (path, pair["id"], side),
                                )
