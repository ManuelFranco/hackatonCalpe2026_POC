"""Standalone loader copied into each export; requires torch, transformers, safetensors.

format_version 1: global steering, h' = h + alpha * v.
format_version 2: conditional steering, h' = h + alpha * g * v, where g comes from a
linear probe over the mean image-token state at the gate layer during prefill.
The gate math mirrors research/conditional_steering.py; this file must stay
self-contained, so it cannot import the research package.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading
import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed

SUPPORTED_FORMATS = (1, 2)
GATE_MODES = ("hard", "soft")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Gate {name} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"Gate {name} must be finite.")
    return float(value)


def validate_gate(config, tensors, width, num_layers=None):
    """Return a normalized gate dict or raise ValueError for malformed artifacts."""
    if not isinstance(config, dict):
        raise ValueError("conditional_gate must be an object.")
    if config.get("format_version") != 1:
        raise ValueError("Unsupported conditional gate format.")
    if set(tensors) != {"weight", "center"}:
        raise ValueError("The gate tensors must be exactly weight and center.")
    for name in ("weight", "center"):
        value = tensors[name]
        if value.shape != (width,):
            raise ValueError(f"Gate {name} must have shape ({width},).")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Gate {name} contains NaN or infinite values.")
    layer = config.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("Gate layer must be a nonnegative integer.")
    if num_layers is not None and layer >= num_layers:
        raise ValueError(f"Gate layer {layer} does not exist in this model.")
    if config.get("hidden_size", width) != width:
        raise ValueError("Gate hidden_size does not match the model.")
    mode = config.get("mode", "hard")
    if mode not in GATE_MODES:
        raise ValueError(f"Gate mode must be one of {GATE_MODES}.")
    scale = _finite_number(config.get("scale"), "scale")
    temperature = _finite_number(config.get("temperature", 1.0), "temperature")
    if scale <= 0 or temperature <= 0:
        raise ValueError("Gate scale and temperature must be positive.")
    return {
        "layer": layer,
        "weight": tensors["weight"].float(),
        "center": tensors["center"].float(),
        "bias": _finite_number(config.get("bias"), "bias"),
        "threshold": _finite_number(config.get("threshold"), "threshold"),
        "scale": scale,
        "mode": mode,
        "temperature": temperature,
        "target": str(config.get("target", "")),
    }


def gate_from_hidden(gate, hidden, image_mask):
    """Score and gate value from image-token states only; no image -> gate closed."""
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError("Conditional gating supports batch size 1 only.")
        hidden = hidden[0]
    mask = image_mask.reshape(-1).bool()
    if mask.numel() != hidden.shape[0]:
        raise ValueError("The gate must be computed from the full prefill.")
    count = int(mask.sum())
    if not count:
        return {"score": None, "value": 0.0, "image_tokens": 0}
    pooled = hidden[mask.to(hidden.device)].float().mean(dim=0).detach().cpu()
    normalized = (pooled - gate["center"]) / gate["scale"]
    score = float(normalized @ gate["weight"]) + gate["bias"]
    if not math.isfinite(score):
        raise ValueError("The gate score is not finite.")
    if gate["mode"] == "hard":
        value = 1.0 if score >= gate["threshold"] else 0.0
    else:
        value = float(
            torch.sigmoid(torch.tensor((score - gate["threshold"]) / gate["temperature"]))
        )
    return {"score": score, "value": value, "image_tokens": count}


@contextmanager
def steering_hooks(
    model,
    directions,
    strengths,
    last_token_only=True,
    gate=None,
    image_mask=None,
    trace=None,
):
    """Install steering hooks for one request. `trace` is a fresh dict per request."""
    handles = []
    layers = model.model.language_model.layers
    if gate is not None:
        if image_mask is None or trace is None:
            raise ValueError("Gated steering needs the prompt image mask and a trace.")
        early = [
            int(layer)
            for layer in directions
            if float(strengths[str(layer)]) != 0 and int(layer) < gate["layer"]
        ]
        if early:
            raise ValueError(f"Steering layers {early} precede the gate layer.")
    try:

        def steer(output, direction, alpha):
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

        if gate is None:
            for layer, direction in directions.items():
                alpha = float(strengths[str(layer)])

                def hook(module, inputs, output, direction=direction, alpha=alpha):
                    if alpha == 0:
                        return None
                    return steer(output, direction, alpha)

                handles.append(layers[int(layer)].register_forward_hook(hook))
        else:
            hooked = sorted({int(layer) for layer in directions} | {gate["layer"]})
            by_layer = {int(layer): direction for layer, direction in directions.items()}
            for layer in hooked:
                state = {"calls": 0}

                def hook(module, inputs, output, layer=layer, state=state):
                    state["calls"] += 1
                    if layer == gate["layer"] and state["calls"] == 1:
                        hidden = output[0] if isinstance(output, (tuple, list)) else output
                        trace.update(gate_from_hidden(gate, hidden, image_mask))
                    if layer not in by_layer:
                        return None
                    alpha = float(strengths[str(layer)])
                    if alpha == 0:
                        return None
                    if "value" not in trace:
                        raise RuntimeError("Steering ran before the gate was computed.")
                    effective = alpha * trace["value"]
                    if effective == 0:
                        return None
                    return steer(output, by_layer[layer], effective)

                handles.append(layers[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


class SteeredVLM:
    def __init__(self, directory, device_map="auto", torch_dtype="auto"):
        root = Path(directory)
        self.config = json.loads((root / "steering_config.json").read_text())
        version = self.config.get("format_version")
        if version not in SUPPORTED_FORMATS:
            raise ValueError("Unsupported steering export version.")
        if version == 1 and "conditional_gate" in self.config:
            raise ValueError("Version 1 exports cannot contain a conditional gate.")
        if version == 2 and "conditional_gate" not in self.config:
            raise ValueError("Version 2 exports require a conditional_gate section.")
        local_model = root / "base_model"
        source = str(local_model) if local_model.exists() else self.config["model_id"]
        revision = None if local_model.exists() else self.config.get("model_revision")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            source, revision=revision, device_map=device_map, torch_dtype=torch_dtype
        ).eval()
        self.processor = AutoProcessor.from_pretrained(source, revision=revision)
        self.directions = load_file(str(root / "steering_vectors.safetensors"))
        width = self.model.config.text_config.hidden_size
        strengths = self.config.get("strengths")
        if not isinstance(strengths, dict) or set(self.directions) != set(strengths):
            raise ValueError("Exported vector layers and strengths do not match.")
        for value in strengths.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Exported strengths must be numbers.")
            if not math.isfinite(value):
                raise ValueError("Exported strengths must be finite.")
        for value in self.directions.values():
            if value.shape != (width,) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid exported direction.")
        self.gate = None
        if version == 2:
            gate_config = self.config["conditional_gate"]
            file = gate_config.get("tensors", "conditional_gate.safetensors")
            if Path(file).name != file:
                raise ValueError("Gate tensors must be a file in the export directory.")
            layers = getattr(
                getattr(getattr(self.model, "model", None), "language_model", None),
                "layers",
                None,
            )
            self.gate = validate_gate(
                gate_config,
                load_file(str(root / file)),
                width,
                None if layers is None else len(layers),
            )
            for layer, value in strengths.items():
                if float(value) != 0 and int(layer) < self.gate["layer"]:
                    raise ValueError("A steering layer precedes the gate layer.")
        self.lock = threading.Lock()

    def generate(self, prompt="", image=None, return_gate=False, **overrides):
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
            trace = {}
            image_mask = None
            if self.gate is not None:
                token_id = getattr(self.model.config, "image_token_id", None)
                ids = inputs["input_ids"][0].detach().cpu()
                image_mask = (
                    ids.eq(int(token_id))
                    if token_id is not None
                    else torch.zeros_like(ids, dtype=torch.bool)
                )
            with (
                steering_hooks(
                    self.model,
                    self.directions,
                    self.config["strengths"],
                    self.config["steer_last_token_only"],
                    gate=self.gate,
                    image_mask=image_mask,
                    trace=trace,
                ),
                torch.inference_mode(),
            ):
                output = self.model.generate(**inputs, **kwargs)
            text = self.processor.decode(
                output[0, inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        return (text, dict(trace)) if return_gate else text
