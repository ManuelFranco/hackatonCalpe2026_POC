from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from sae_dashboard import model_runtime as runtime
from sae_dashboard.session import Settings
from sae_dashboard.workflow import capture_score
from sae_dashboard.portable_model import steering_hooks


class RuntimeTests(unittest.TestCase):
    def test_mps_generation_uses_eager_attention(self):
        for requested in ("mps", "auto"):
            with (
                patch.object(runtime, "MODEL_DEVICE", requested),
                patch.object(runtime, "MODEL_DTYPE", "auto"),
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=True),
            ):
                options = runtime.model_load_options()
            self.assertEqual(options["device_map"], {"": "mps"})
            self.assertEqual(options["attn_implementation"], "eager")
        for requested in ("cpu", "cuda", "auto"):
            with (
                patch.object(runtime, "MODEL_DEVICE", requested),
                patch("torch.cuda.is_available", return_value=True),
            ):
                self.assertNotIn("attn_implementation", runtime.model_load_options())

    def test_streamed_scores_match_direct_aggregation(self):
        residual = torch.tensor([[2.0, 0.0], [0.0, 6.0], [4.0, 3.0]])
        for scope, expected_rows in [
            ("all", residual),
            ("last", residual[-1:]),
            ("non_image", residual[[0, 2]]),
        ]:
            for aggregation in ("mean", "max"):
                with (
                    patch.object(runtime, "SAE_CHUNK_TOKENS", 1),
                    patch.object(
                        runtime, "encode_sae_chunked", side_effect=lambda s, x: x
                    ),
                ):
                    score, norm = capture_score(
                        residual,
                        torch.tensor([False, True, False]),
                        None,
                        Settings(token_scope=scope, aggregation=aggregation),
                    )
                expected = (
                    expected_rows.mean(0)
                    if aggregation == "mean"
                    else expected_rows.amax(0)
                )
                torch.testing.assert_close(score, expected)
                self.assertAlmostEqual(norm, float(expected_rows.norm(dim=-1).mean()))

    def test_generation_hooks_are_removed_after_failure(self):
        layer = torch.nn.Identity()
        seen = []

        def fail(**kwargs):
            seen.append(layer(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])))
            raise RuntimeError("generation failed")

        with (
            patch.object(runtime, "LAYERS", [9]),
            patch.object(runtime, "get_layer_module", return_value=layer),
            patch.object(runtime, "model", SimpleNamespace(generate=fail)),
            patch.object(runtime, "STEER_LAST_TOKEN_ONLY", True),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                runtime.generate_answer(
                    {}, 2, 16, 0, {9: torch.tensor([2.0, 0.0])}, {9: 1}
                )
        torch.testing.assert_close(seen[0], torch.tensor([[[1.0, 2.0], [5.0, 4.0]]]))
        self.assertFalse(layer._forward_hooks)

    def test_export_hooks_match_last_and_all_position_semantics(self):
        layer = torch.nn.Identity()
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
        )
        for last in (True, False):
            with self.assertRaises(RuntimeError):
                with steering_hooks(
                    model, {"0": torch.tensor([2.0, 0.0])}, {"0": -1}, last
                ):
                    output = layer(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
                    expected = (
                        [[[1.0, 2.0], [1.0, 4.0]]]
                        if last
                        else [[[-1.0, 2.0], [1.0, 4.0]]]
                    )
                    torch.testing.assert_close(output, torch.tensor(expected))
                    raise RuntimeError("test")
            self.assertFalse(layer._forward_hooks)

    def test_base_probe_does_not_load_saes(self):
        with (
            patch.object(runtime, "model", object()),
            patch.object(runtime, "processor", object()),
            patch.dict(runtime.saes, {}, clear=True),
            patch.object(runtime, "validate_sae_registry", side_effect=AssertionError),
        ):
            runtime.ensure_models_loaded(with_saes=False)

    def test_mmmu_multiple_images_keep_labels(self):
        first, second = object(), object()
        messages = runtime.make_messages(
            [("Image 1", first), ("Image 3", second)], "Question <image 3>"
        )
        content = messages[0]["content"]
        self.assertEqual(content[0]["text"], "Image 1:")
        self.assertIs(content[1]["image"], first)
        self.assertEqual(content[2]["text"], "Image 3:")
        self.assertIs(content[3]["image"], second)
