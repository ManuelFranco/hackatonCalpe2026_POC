"""Standalone loader copied into each export; requires torch, transformers, safetensors."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed


@contextmanager
def steering_hooks(model, directions, strengths, last_token_only=True):
    handles = []
    try:
        for layer, direction in directions.items():
            alpha = float(strengths[str(layer)])

            def hook(module, inputs, output, direction=direction, alpha=alpha):
                if alpha == 0:
                    return None
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                delta = direction.to(device=hidden.device, dtype=hidden.dtype) * alpha
                steered = hidden.clone()
                if last_token_only:
                    steered[:, -1, :] += delta
                else:
                    steered += delta.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (steered, *output[1:])
                if isinstance(output, list):
                    return [steered, *output[1:]]
                return steered

            handles.append(
                model.model.language_model.layers[int(layer)].register_forward_hook(
                    hook
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


class SteeredVLM:
    def __init__(self, directory, device_map="auto", torch_dtype="auto"):
        root = Path(directory)
        self.config = json.loads((root / "steering_config.json").read_text())
        if self.config.get("format_version") != 1:
            raise ValueError("Unsupported steering export version.")
        local_model = root / "base_model"
        source = str(local_model) if local_model.exists() else self.config["model_id"]
        revision = None if local_model.exists() else self.config.get("model_revision")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            source, revision=revision, device_map=device_map, torch_dtype=torch_dtype
        ).eval()
        self.processor = AutoProcessor.from_pretrained(source, revision=revision)
        self.directions = load_file(str(root / "steering_vectors.safetensors"))
        width = self.model.config.text_config.hidden_size
        if set(self.directions) != set(self.config["strengths"]):
            raise ValueError("Exported vector layers and strengths do not match.")
        for value in self.directions.values():
            if value.shape != (width,) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid exported direction.")
        self.lock = threading.Lock()

    def generate(self, prompt="", image=None, **overrides):
        content = []
        if image is not None:
            content.append(
                {
                    "type": "image",
                    **(
                        {"path": str(image)}
                        if isinstance(image, (str, Path))
                        else {"image": image}
                    ),
                }
            )
        if prompt:
            content.append({"type": "text", "text": prompt})
        if not content:
            raise ValueError("Provide text, an image, or both.")
        return self.chat([{"role": "user", "content": content}], **overrides)["text"]

    def chat(self, messages, max_context_tokens=None, **overrides):
        """Generate from the full Transformers chat history with exported steering."""
        settings = {**self.config["generation"], **overrides}
        temperature = float(settings["temperature"])
        with self.lock:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_pan_and_scan=False,
            ).to(self.model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            max_new_tokens = int(settings["max_new_tokens"])
            if (
                max_context_tokens
                and prompt_tokens + max_new_tokens > max_context_tokens
            ):
                raise ValueError(
                    "Conversation and requested output exceed the server context limit. "
                    "Start a new chat or reduce the output token limit."
                )
            set_seed(int(settings["seed"]))
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "use_cache": True,
            }
            if temperature > 0:
                kwargs.update(
                    temperature=temperature, top_p=settings.get("top_p", 0.95)
                )
            stop = settings.get("stop") or []
            if stop:
                kwargs.update(stop_strings=stop, tokenizer=self.processor.tokenizer)
            with (
                steering_hooks(
                    self.model,
                    self.directions,
                    self.config["strengths"],
                    self.config["steer_last_token_only"],
                ),
                torch.inference_mode(),
            ):
                output = self.model.generate(**inputs, **kwargs)
            generated = output[0, prompt_tokens:]
            text = self.processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            finish = "length" if len(generated) >= max_new_tokens else "stop"
            if len(generated) and generated[-1].item() in eos:
                finish = "stop"
            positions = [text.index(s) for s in stop if s in text]
            if positions:
                text = text[: min(positions)]
                finish = "stop"
            return {
                "text": text,
                "finish_reason": finish,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(generated),
                    "total_tokens": prompt_tokens + len(generated),
                },
            }
