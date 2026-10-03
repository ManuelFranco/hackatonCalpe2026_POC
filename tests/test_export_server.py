import base64
import io
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image
import torch

from sae_dashboard.portable_model import SteeredVLM
from sae_dashboard.portable_server import create_app


class ExportServerTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "text": "A steered answer.",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }
        self.vlm = Mock()
        self.vlm.chat.return_value = self.result
        self.client = TestClient(create_app(self.vlm))
        self.payload = {
            "model": "gemma-steered",
            "messages": [
                {"role": "system", "content": "Help with code."},
                {"role": "user", "content": "Remember foo."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "What was the name?"},
            ],
        }

    def test_models_history_generation_parameters_and_usage(self):
        self.assertEqual(
            self.client.get("/v1/models").json()["data"][0]["id"], "gemma-steered"
        )
        response = self.client.post(
            "/v1/chat/completions",
            json={
                **self.payload,
                "max_tokens": 256,
                "temperature": 0,
                "top_p": 0.9,
                "seed": 7,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(
            result["choices"][0]["message"]["content"], self.result["text"]
        )
        self.assertEqual(result["usage"], self.result["usage"])
        args, kwargs = self.vlm.chat.call_args
        self.assertEqual(
            [m["role"] for m in args[0]], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(args[0][1]["content"][0]["text"], "Remember foo.")
        self.assertEqual(
            kwargs,
            {
                "max_context_tokens": 8192,
                "max_new_tokens": 256,
                "temperature": 0,
                "top_p": 0.9,
                "seed": 7,
                "stop": [],
            },
        )

    def test_sse_stream_with_final_usage_and_done(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                **self.payload,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        events = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(
            chunks[1]["choices"][0]["delta"]["content"], self.result["text"]
        )
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[3]["usage"], self.result["usage"])

    def test_image_attachment_reaches_loader_as_pixels(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        payload = {
            "model": "gemma-steered",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color?"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        }
        self.assertEqual(
            self.client.post("/v1/chat/completions", json=payload).status_code, 200
        )
        image = self.vlm.chat.call_args.args[0][0]["content"][1]["image"]
        self.assertEqual(image.size, (4, 3))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_invalid_requests_do_not_run_the_model(self):
        invalid = [
            {"model": "unknown"},
            {"tools": [{"type": "function"}]},
            {"messages": []},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "assistant", "content": "x"}]},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///tmp/photo.png"},
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/photo.png"},
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,not-base64"
                                },
                            }
                        ],
                    }
                ]
            },
            {"max_tokens": 0},
            {"temperature": -1},
            {"stop": [""]},
            {"max_tokens": 10, "max_completion_tokens": 10},
            {"frequency_penalty": 1},
            {"steering_strength": 0},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                response = self.client.post(
                    "/v1/chat/completions", json={**self.payload, **changes}
                )
                self.assertIn(response.status_code, (400, 404, 422))
        self.vlm.chat.assert_not_called()

    def test_context_error_is_reported_before_stream_starts(self):
        self.vlm.chat.side_effect = ValueError("Conversation exceeds context limit.")
        response = self.client.post(
            "/v1/chat/completions", json={**self.payload, "stream": True}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("context limit", response.json()["detail"])

    def test_optional_api_key_protects_both_routes(self):
        client = TestClient(create_app(self.vlm, api_key="test-secret"))
        self.assertEqual(client.get("/v1/models").status_code, 401)
        self.assertEqual(
            client.post("/v1/chat/completions", json=self.payload).status_code, 401
        )
        self.vlm.chat.assert_not_called()
        headers = {"Authorization": "Bearer test-secret"}
        self.assertEqual(client.get("/v1/models", headers=headers).status_code, 200)
        self.assertEqual(
            client.post(
                "/v1/chat/completions", headers=headers, json=self.payload
            ).status_code,
            200,
        )


class PortableChatTests(unittest.TestCase):
    def setUp(self):
        class Inputs(dict):
            def to(self, device):
                return self

        self.layer = torch.nn.Identity()
        self.vlm = SteeredVLM.__new__(SteeredVLM)
        self.vlm.config = {
            "generation": {"seed": 17, "temperature": 0, "max_new_tokens": 3},
            "strengths": {"0": -2},
            "steer_last_token_only": True,
        }
        self.vlm.lock = threading.Lock()
        self.vlm.directions = {"0": torch.tensor([1.0, 2.0])}
        self.vlm.processor = Mock()
        self.vlm.processor.apply_chat_template.return_value = Inputs(
            input_ids=torch.tensor([[1, 2]])
        )
        self.vlm.processor.decode.return_value = "hello"
        self.vlm.model = SimpleNamespace(
            device="cpu",
            generation_config=SimpleNamespace(eos_token_id=[9]),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[self.layer])),
            generate=Mock(side_effect=self.generate),
        )
        self.messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def generate(self, **kwargs):
        hidden = self.layer(torch.zeros(1, 2, 2))
        torch.testing.assert_close(hidden[0, 0], torch.zeros(2))
        torch.testing.assert_close(hidden[0, 1], torch.tensor([-2.0, -4.0]))
        return torch.tensor([[1, 2, 7, 9]])

    def test_chat_uses_steering_and_cleans_hooks_and_preserves_generate_api(self):
        for _ in range(2):
            result = self.vlm.chat(self.messages, max_context_tokens=10)
            self.assertEqual(result["text"], "hello")
            self.assertEqual(
                result["usage"],
                {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            )
            self.assertEqual(result["finish_reason"], "stop")
            self.assertFalse(self.layer._forward_hooks)
        self.assertEqual(
            self.vlm.processor.apply_chat_template.call_args.args[0], self.messages
        )
        self.assertEqual(self.vlm.generate("hi"), "hello")

    def test_exception_removes_hooks_and_releases_lock(self):
        self.vlm.model.generate.side_effect = RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            self.vlm.chat(self.messages)
        self.assertFalse(self.layer._forward_hooks)
        self.assertFalse(self.vlm.lock.locked())

    def test_context_limit_prevents_generation(self):
        with self.assertRaisesRegex(ValueError, "context limit"):
            self.vlm.chat(self.messages, max_context_tokens=4)
        self.vlm.model.generate.assert_not_called()
        self.assertFalse(self.layer._forward_hooks)

    def test_stop_and_length_reporting(self):
        self.vlm.model.generate.side_effect = lambda **kwargs: torch.tensor(
            [[1, 2, 7, 8, 8]]
        )
        self.assertEqual(self.vlm.chat(self.messages)["finish_reason"], "length")
        self.vlm.processor.decode.return_value = "hello END trailing"
        result = self.vlm.chat(self.messages, stop=["END"])
        self.assertEqual(result["text"], "hello ")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(
            self.vlm.model.generate.call_args.kwargs["stop_strings"], ["END"]
        )
