"""Gemma inference and SAE capture. Call model operations under MODEL_LOCK."""

from __future__ import annotations
import importlib.metadata
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from sae_lens import SAE
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed

MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
MODEL_REVISION = os.getenv("GEMMA_MODEL_REVISION")
MODEL_DEVICE = os.getenv("GEMMA_DEVICE", "auto").strip().lower()
MODEL_DTYPE = os.getenv("GEMMA_DTYPE", "auto").strip().lower()
LAYERS = [9, 17, 22, 29]
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2-4b-it-res")
SAE_IDS = {i: f"layer_{i}_width_16k_l0_medium" for i in LAYERS}
FEATURE_AGGREGATION = os.getenv("FEATURE_AGGREGATION", "mean").strip().lower()
FEATURE_TOKEN_SCOPE = os.getenv("FEATURE_TOKEN_SCOPE", "all").strip().lower()
SAE_CHUNK_TOKENS = int(os.getenv("SAE_CHUNK_TOKENS", "128"))
STEERING_FRACTION_PER_UNIT = float(os.getenv("STEERING_FRACTION_PER_UNIT", "0.05"))
STEER_LAST_TOKEN_ONLY = os.getenv("STEER_LAST_TOKEN_ONLY", "1") != "0"
CAPTURE_SCOPE = "prompt_with_assistant_prefix_no_generated_tokens"
MODEL_LOCK = threading.RLock()
model: Any = None
processor: Any = None
saes: Dict[int, SAE] = {}
sae_releases_used: Dict[int, str] = {}


def unwrap_sae(loaded: Any) -> SAE:
    return loaded[0] if isinstance(loaded, tuple) else loaded


def validate_configuration() -> None:
    if MODEL_DEVICE not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("GEMMA_DEVICE must be auto, mps, cuda or cpu.")
    if MODEL_DTYPE not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("GEMMA_DTYPE must be auto, bfloat16, float16 or float32.")
    if FEATURE_AGGREGATION not in {"mean", "max"}:
        raise ValueError("FEATURE_AGGREGATION must be mean or max.")
    if FEATURE_TOKEN_SCOPE not in {"all", "non_image", "last"}:
        raise ValueError("FEATURE_TOKEN_SCOPE must be all, non_image or last.")
    if SAE_CHUNK_TOKENS < 1:
        raise ValueError("SAE_CHUNK_TOKENS must be positive.")
    if not math.isfinite(STEERING_FRACTION_PER_UNIT) or STEERING_FRACTION_PER_UNIT <= 0:
        raise ValueError("STEERING_FRACTION_PER_UNIT must be finite and positive.")


def validate_sae_registry() -> None:
    """Validate local SAE-Lens metadata without downloading any weights."""
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    registry = get_pretrained_saes_directory()
    if SAE_RELEASE not in registry:
        raise ValueError(f"Unknown SAE_RELEASE in this SAE-Lens version: {SAE_RELEASE}")
    release = registry[SAE_RELEASE]
    if release.model != MODEL_ID:
        raise ValueError(
            f"SAE release is for {release.model}, but GEMMA_MODEL_ID is {MODEL_ID}."
        )
    missing = [sae_id for sae_id in SAE_IDS.values() if sae_id not in release.saes_map]
    if missing:
        raise ValueError(f"SAE IDs missing from {SAE_RELEASE}: {missing}")


def model_load_options() -> Dict[str, Any]:
    device = MODEL_DEVICE
    if device == "auto":
        if torch.cuda.is_available():
            device = "auto"  # Accelerate may distribute across NVIDIA devices.
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Check macOS/PyTorch or use GEMMA_DEVICE=cpu."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and the PyTorch build."
        )
    dtype = "auto" if MODEL_DTYPE == "auto" else getattr(torch, MODEL_DTYPE)
    options = {
        "device_map": "auto" if device == "auto" else {"": device},
        "torch_dtype": dtype,
    }
    if device == "mps":
        # PyTorch 2.6 MPS SDPA crashes during Gemma's grouped-query cached decoding.
        options["attn_implementation"] = "eager"
    return options


def ensure_models_loaded(with_saes: bool = True) -> None:
    """Must be called while holding MODEL_LOCK (hooks share one model)."""
    global model, processor
    if (
        model is not None
        and processor is not None
        and (not with_saes or len(saes) == len(LAYERS))
    ):
        return
    validate_configuration()
    if with_saes:
        validate_sae_registry()
    print(f"Loading {MODEL_ID}...")
    if model is None:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, **model_load_options()
        ).eval()
    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if not with_saes:
        return
    hidden_size = model.config.text_config.hidden_size
    for layer_idx in LAYERS:
        if layer_idx in saes:
            continue
        print(f"Loading SAE {SAE_IDS[layer_idx]} from {SAE_RELEASE}...")
        sae = (
            unwrap_sae(
                SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_IDS[layer_idx])
            )
            .cpu()
            .eval()
        )
        if not hasattr(sae, "W_dec") or sae.W_dec.shape[1] != hidden_size:
            raise ValueError(
                f"Layer {layer_idx}: SAE decoder does not match model hidden size."
            )
        if sae.cfg.metadata.hook_name != f"blocks.{layer_idx}.hook_resid_post":
            raise ValueError(f"Layer {layer_idx}: SAE hook does not match resid_post.")
        if sae.cfg.normalize_activations != "none":
            raise ValueError(
                "This steering implementation requires an unnormalized SAE decoder."
            )
        saes[layer_idx] = sae
        sae_releases_used[layer_idx] = SAE_RELEASE
    print(f"Model and SAEs ready. Primary device: {model.device}")


def runtime_metadata() -> Dict[str, Any]:
    packages = (
        "torch",
        "torchvision",
        "transformers",
        "sae-lens",
        "gradio",
        "accelerate",
        "datasets",
        "safetensors",
        "pillow",
        "sentencepiece",
    )
    return {
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "model_revision_requested": MODEL_REVISION,
        "model_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
        "device": str(model.device) if model is not None else None,
        "dtype": str(model.dtype) if model is not None else None,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_requested": MODEL_DEVICE,
        "dtype_requested": MODEL_DTYPE,
    }


def tensor_stats(name: str, x: torch.Tensor) -> Dict[str, Any]:
    xf = x.detach().float().cpu()
    finite = torch.isfinite(xf)
    finite_count = int(finite.sum())
    total = int(xf.numel())
    nan_count = int(torch.isnan(xf).sum())
    posinf_count = int(torch.isposinf(xf).sum())
    neginf_count = int(torch.isneginf(xf).sum())
    if finite_count:
        vals = xf[finite]
        min_v = float(vals.min())
        max_v = float(vals.max())
        mean_v = float(vals.mean())
    else:
        min_v = max_v = mean_v = float("nan")
    return {
        "name": name,
        "shape": list(xf.shape),
        "dtype": str(x.dtype),
        "finite": finite_count,
        "total": total,
        "nan": nan_count,
        "+inf": posinf_count,
        "-inf": neginf_count,
        "min_finite": min_v,
        "max_finite": max_v,
        "mean_finite": mean_v,
    }


def assert_finite(name: str, x: torch.Tensor) -> None:
    if torch.isfinite(x).all():
        return
    stats = tensor_stats(name, x)
    raise RuntimeError(
        f"Non-finite tensor detected in {name}: "
        f"shape={stats['shape']} dtype={stats['dtype']} "
        f"nan={stats['nan']} +inf={stats['+inf']} -inf={stats['-inf']} "
        f"finite={stats['finite']}/{stats['total']} "
        f"min_finite={stats['min_finite']} max_finite={stats['max_finite']}"
    )


def validate_generation_settings(
    max_new_tokens: int, temperature: float, seed: int
) -> None:
    if not 1 <= int(max_new_tokens) <= 1024:
        raise ValueError("Max new tokens must be between 1 and 1024.")
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    if (
        not math.isfinite(float(seed))
        or int(seed) != seed
        or not 0 <= int(seed) < 2**32
    ):
        raise ValueError("Seed must be an integer between 0 and 4294967295.")


def get_layer_module(layer_idx: int):
    return model.model.language_model.layers[layer_idx]


def extract_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def replace_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    if isinstance(output, list):
        out = list(output)
        out[0] = new_hidden
        return out
    return new_hidden


def make_messages(image_path: Any, prompt: str) -> List[Dict[str, Any]]:
    # PIL images stay in memory, including all images used by MMMU.
    images = (
        image_path
        if isinstance(image_path, (list, tuple))
        else ([image_path] if image_path is not None else [])
    )
    content = []
    for image in images:
        if isinstance(image, tuple):
            label, image = image
            content.append({"type": "text", "text": label + ":"})
        if isinstance(image, (str, Path)):
            content.append({"type": "image", "path": str(Path(image).resolve())})
        else:
            content.append({"type": "image", "image": image})
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_inputs(
    image_path: Optional[str],
    prompt: str,
    add_generation_prompt: bool = True,
) -> Tuple[Any, torch.Tensor, int]:
    messages = make_messages(image_path, prompt)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=False,
    )
    input_ids_cpu = inputs["input_ids"][0].detach().cpu().clone()
    input_len = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(model.device)
    return inputs, input_ids_cpu, input_len


def get_image_mask(input_ids_cpu: torch.Tensor) -> torch.Tensor:
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        return torch.zeros_like(input_ids_cpu, dtype=torch.bool)
    return input_ids_cpu.eq(int(image_token_id))


def capture_prompt_residuals(
    inputs: Any,
    steering_directions: Optional[Dict[int, torch.Tensor]] = None,
    strengths: Optional[Dict[int, float]] = None,
) -> Dict[int, torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in LAYERS:
        module = get_layer_module(layer_idx)

        def make_hook(idx: int):
            cache: Dict[Tuple[str, str], torch.Tensor] = {}

            def hook_fn(module, module_inputs, output):
                x = extract_hidden(output)
                x_out = x
                if steering_directions is not None and strengths is not None:
                    alpha = float(strengths.get(idx, 0.0))
                    if alpha != 0.0:
                        key = (str(x.device), str(x.dtype))
                        if key not in cache:
                            cache[key] = steering_directions[idx].to(
                                device=x.device,
                                dtype=x.dtype,
                            )
                        delta = cache[key] * alpha
                        if STEER_LAST_TOKEN_ONLY:
                            x_out = x.clone()
                            x_out[:, -1, :] = x_out[:, -1, :] + delta
                        else:
                            x_out = x + delta.view(1, 1, -1)
                captured[idx] = x_out[0].detach().float().cpu()
                if x_out is not x:
                    return replace_hidden(output, x_out)
                return None

            return hook_fn

        handles.append(module.register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.inference_mode():
            model(
                **inputs,
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = [idx for idx in LAYERS if idx not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture layers: {missing}")
    return captured


def generate_answer(
    inputs: Any,
    input_len: int,
    max_new_tokens: int,
    temperature: float,
    steering_directions: Optional[Dict[int, torch.Tensor]] = None,
    strengths: Optional[Dict[int, float]] = None,
    seed: int = 0,
    first_step_only: bool = False,
) -> str:
    validate_generation_settings(max_new_tokens, temperature, seed)
    handles = []
    if steering_directions is not None and strengths is not None:
        for layer_idx in LAYERS:
            module = get_layer_module(layer_idx)

            def make_hook(idx: int):
                calls = 0
                cache: Dict[Tuple[str, str], torch.Tensor] = {}

                def hook_fn(module, module_inputs, output):
                    nonlocal calls
                    calls += 1
                    if first_step_only and calls > 1:
                        return None
                    alpha = float(strengths.get(idx, 0.0))
                    if alpha == 0.0:
                        return None
                    x = extract_hidden(output)
                    key = (str(x.device), str(x.dtype))
                    if key not in cache:
                        cache[key] = steering_directions[idx].to(
                            device=x.device,
                            dtype=x.dtype,
                        )
                    delta = alpha * cache[key]
                    if STEER_LAST_TOKEN_ONLY:
                        x_out = x.clone()
                        x_out[:, -1, :] = x_out[:, -1, :] + delta
                    else:
                        x_out = x + delta.view(1, 1, -1)
                    return replace_hidden(output, x_out)

                return hook_fn

            handles.append(module.register_forward_hook(make_hook(layer_idx)))
    try:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
        }
        if float(temperature) > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(temperature),
                    "top_p": 0.95,
                }
            )
        else:
            gen_kwargs["do_sample"] = False
        # Each paired call starts from the same RNG state, even at temperature > 0.
        # All model callbacks hold MODEL_LOCK, so requests cannot interleave here.
        set_seed(int(seed))
        with torch.inference_mode():
            generated = model.generate(**inputs, **gen_kwargs)
        new_tokens = generated[0, input_len:]
        visible = processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if visible:
            return visible
        raw = processor.decode(
            new_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        ids_preview = new_tokens[:16].detach().cpu().tolist()
        return (
            "[No visible text: the model terminated immediately. "
            f"token_ids={ids_preview}; raw_decode={raw!r}]"
        )
    finally:
        for handle in handles:
            handle.remove()


def encode_sae_chunked(sae: SAE, residuals: torch.Tensor) -> torch.Tensor:
    sae_param = next(sae.parameters())
    chunks: List[torch.Tensor] = []
    assert_finite("residuals_before_sae", residuals)
    for start in range(0, residuals.shape[0], SAE_CHUNK_TOKENS):
        x = residuals[start : start + SAE_CHUNK_TOKENS].to(
            device=sae_param.device,
            dtype=sae_param.dtype,
        )
        assert_finite(f"sae_input_chunk_{start}", x)
        with torch.inference_mode():
            z = sae.encode(x)
        z_cpu = z.detach().float().cpu()
        assert_finite(f"sae_output_chunk_{start}", z_cpu)
        chunks.append(z_cpu)
    out = torch.cat(chunks, dim=0)
    assert_finite("sae_activations", out)
    return out


def feature_token_mask(
    image_mask: torch.Tensor, token_scope: str = FEATURE_TOKEN_SCOPE
) -> torch.Tensor:
    if image_mask.numel() == 0:
        raise ValueError("Cannot aggregate an empty prompt.")
    if token_scope == "last":
        mask = torch.zeros_like(image_mask, dtype=torch.bool)
        mask[-1] = True
        return mask
    if token_scope == "non_image":
        return ~image_mask.bool()
    if token_scope == "all":
        return torch.ones_like(image_mask, dtype=torch.bool)
    raise ValueError(f"Invalid feature token scope: {token_scope}")


def aggregate_feature_acts(
    feature_acts: torch.Tensor,
    image_mask: torch.Tensor,
    token_scope: str = FEATURE_TOKEN_SCOPE,
) -> torch.Tensor:
    mask = feature_token_mask(image_mask, token_scope)
    if mask.numel() != feature_acts.shape[0] or not mask.any():
        raise ValueError("Feature token selection is empty or has the wrong length.")
    x = feature_acts[mask].float()
    if FEATURE_AGGREGATION == "max":
        return x.amax(dim=0)
    if FEATURE_AGGREGATION == "mean":
        return x.mean(dim=0)
    raise ValueError(f"Invalid FEATURE_AGGREGATION: {FEATURE_AGGREGATION}")
