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
        settings = {**self.config["generation"], **overrides}
        temperature = float(settings["temperature"])
        with self.lock:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                do_pan_and_scan=False,
            ).to(self.model.device)
            set_seed(int(settings["seed"]))
            kwargs = {
                "max_new_tokens": int(settings["max_new_tokens"]),
                "do_sample": temperature > 0,
                "use_cache": True,
            }
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=0.95)
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
            return self.processor.decode(
                output[0, inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
