#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 4B-IT + Gemma Scope 2 SAEs
Multi-pair common-feature B - A steering with a Gradio web interface.
Overview
========
This application builds a reusable steering profile from multiple input
condition pairs (A_i, B_i):
    Condition A = (Prompt A, optional Image A)
    Condition B = (Prompt B, optional Image B)
Each condition may contain:
- text only,
- image + text,
- image only,
- or even the same image with different prompts.
For every configured transformer layer and every pair, the app:
1. captures the post-layer residual stream (`resid_post`),
2. encodes it with the matching Gemma Scope 2 sparse autoencoder (SAE),
3. aggregates token-level SAE activations into one feature vector,
4. computes the contrastive feature direction:
       feature_delta = score(B) - score(A)
5. intersects the nonzero-delta feature masks across ALL pairs, then averages
   their signed deltas with non-common features set to zero.
6. projects that feature-space difference back into residual space:
       raw_direction = masked_mean_delta @ W_dec
7. rescales the residual direction to a stable magnitude relative to the
   residual norm observed during calibration.
The resulting B - A profile can then be applied to ANY later query, with or
without an image:
       h' = h + alpha * scaled_direction
A positive alpha adds the B - A direction; a negative alpha subtracts it.
The behavioral effect must be measured. Alpha = 0 disables steering.
What the app saves
==================
- full SAE activations for Condition A and Condition B,
- residual stream tensors (optional, enabled by default),
- feature scores and B - A feature deltas,
- raw and scaled residual steering directions,
- experiment metadata and numerical diagnostics,
- steered-query activations and metadata,
- ZIP bundles for convenient export.
Quick start
===========
1. Install dependencies:
   uv sync --locked
2. Authenticate with Hugging Face if required:
   uv run hf auth login
3. Run:
   uv run python demo_gradio_common_features.py
4. Open the local Gradio URL shown in the terminal.
Optional environment variables
==============================
GEMMA_MODEL_ID                default: google/gemma-3-4b-it
GEMMA_SAE_RUNS_DIR            default: gemma_sae_common_feature_runs
GRADIO_SERVER_NAME            default: 0.0.0.0
GRADIO_SERVER_PORT            default: 7860
TOP_DIFFS_TO_SHOW             default: 20
MAX_CONTRASTIVE_PAIRS          default: 8 (between 2 and 20)
FEATURE_PRESENCE_EPS          default: 1e-6 (strict abs(delta) > epsilon)
SAE_CHUNK_TOKENS              default: 128
FEATURE_AGGREGATION           default: mean
FEATURE_TOKEN_SCOPE           default: all (also: non_image, last)
STEERING_FRACTION_PER_UNIT    default: 0.05
STEER_LAST_TOKEN_ONLY         default: 1
SAVE_RESIDUALS                default: 1
GEMMA_MODEL_REVISION          optional Hugging Face commit/tag
GEMMA_DEVICE                  default: auto (also: mps, cuda, cpu)
GEMMA_DTYPE                   default: auto (also: bfloat16, float16, float32)
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import torch
from sae_lens import SAE
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, set_seed

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
MODEL_REVISION = os.getenv("GEMMA_MODEL_REVISION")
MODEL_DEVICE = os.getenv("GEMMA_DEVICE", "auto").strip().lower()
MODEL_DTYPE = os.getenv("GEMMA_DTYPE", "auto").strip().lower()
LAYERS = [9, 17, 29]
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2-4b-it-res")
SAE_IDS = {
    9: "layer_9_width_16k_l0_medium",
    17: "layer_17_width_16k_l0_medium",
    29: "layer_29_width_16k_l0_medium",
}
FEATURE_AGGREGATION = os.getenv("FEATURE_AGGREGATION", "mean").strip().lower()
FEATURE_TOKEN_SCOPE = os.getenv("FEATURE_TOKEN_SCOPE", "all").strip().lower()
STEERING_FRACTION_PER_UNIT = float(os.getenv("STEERING_FRACTION_PER_UNIT", "0.05"))
STEER_LAST_TOKEN_ONLY = os.getenv("STEER_LAST_TOKEN_ONLY", "1") != "0"
TOP_DIFFS_TO_SHOW = int(os.getenv("TOP_DIFFS_TO_SHOW", "20"))
MAX_PAIRS = int(os.getenv("MAX_CONTRASTIVE_PAIRS", "8"))
MIN_PAIRS = 2
FEATURE_PRESENCE_EPS = float(os.getenv("FEATURE_PRESENCE_EPS", "1e-6"))
SAE_CHUNK_TOKENS = int(os.getenv("SAE_CHUNK_TOKENS", "128"))
SAVE_RESIDUALS = os.getenv("SAVE_RESIDUALS", "1") != "0"
RUNS_DIR = Path(os.getenv("GEMMA_SAE_RUNS_DIR", "gemma_sae_common_feature_runs"))
MODEL_LOCK = threading.Lock()
# ============================================================
# MODEL LOADING (deferred until the first request or --preload)
# ============================================================
model: Any = None
processor: Any = None
saes: Dict[int, SAE] = {}
sae_releases_used: Dict[int, str] = {}


# ============================================================
# SAE LOADING
# ============================================================
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
    if SAE_CHUNK_TOKENS < 1 or TOP_DIFFS_TO_SHOW < 1:
        raise ValueError("SAE_CHUNK_TOKENS and TOP_DIFFS_TO_SHOW must be positive.")
    if MAX_PAIRS < MIN_PAIRS or MAX_PAIRS > 20:
        raise ValueError("MAX_CONTRASTIVE_PAIRS must be between 2 and 20.")
    if not math.isfinite(FEATURE_PRESENCE_EPS) or FEATURE_PRESENCE_EPS < 0:
        raise ValueError("FEATURE_PRESENCE_EPS must be finite and non-negative.")
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
    return {
        "device_map": "auto" if device == "auto" else {"": device},
        "torch_dtype": dtype,
    }


def ensure_models_loaded() -> None:
    """Must be called while holding MODEL_LOCK (hooks share one model)."""
    global model, processor
    if model is not None and processor is not None and len(saes) == len(LAYERS):
        return
    validate_configuration()
    validate_sae_registry()
    print(f"Loading {MODEL_ID}...")
    if model is None:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, **model_load_options()
        ).eval()
    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
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


# ============================================================
# GENERAL UTILITIES
# ============================================================
def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def validate_condition_inputs(
    condition_name: str,
    image_path: Optional[str],
    prompt: str,
) -> str:
    prompt = (prompt or "").strip()
    if image_path and not Path(image_path).exists():
        raise gr.Error(
            f"The image for condition {condition_name} is no longer available in the Gradio temporary directory."
        )
    if not prompt and not image_path:
        raise gr.Error(
            f"Condition {condition_name} must contain text, an image, or both."
        )
    return prompt


def validate_generation_settings(
    max_new_tokens: int, temperature: float, seed: int
) -> None:
    if not 1 <= int(max_new_tokens) <= 1024:
        raise gr.Error("Max new tokens must be between 1 and 1024.")
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
        raise gr.Error("Temperature must be between 0 and 2.")
    if (
        not math.isfinite(float(seed))
        or int(seed) != seed
        or not 0 <= int(seed) < 2**32
    ):
        raise gr.Error("Seed must be an integer between 0 and 4294967295.")


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


def make_messages(image_path: Optional[str], prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if image_path:
        content.append(
            {
                "type": "image",
                "path": str(Path(image_path).resolve()),
            }
        )
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


# ============================================================
# RESID_POST CAPTURE / GENERATION
# ============================================================
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
) -> str:
    validate_generation_settings(max_new_tokens, temperature, seed)
    handles = []
    if steering_directions is not None and strengths is not None:
        for layer_idx in LAYERS:
            module = get_layer_module(layer_idx)

            def make_hook(idx: int):
                cache: Dict[Tuple[str, str], torch.Tensor] = {}

                def hook_fn(module, module_inputs, output):
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


# ============================================================
# SAE ENCODING + AGGREGATION
# ============================================================
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


def feature_token_mask(image_mask: torch.Tensor) -> torch.Tensor:
    if image_mask.numel() == 0:
        raise ValueError("Cannot aggregate an empty prompt.")
    if FEATURE_TOKEN_SCOPE == "last":
        mask = torch.zeros_like(image_mask, dtype=torch.bool)
        mask[-1] = True
        return mask
    if FEATURE_TOKEN_SCOPE == "non_image":
        return ~image_mask.bool()
    if FEATURE_TOKEN_SCOPE == "all":
        return torch.ones_like(image_mask, dtype=torch.bool)
    raise ValueError(f"Invalid FEATURE_TOKEN_SCOPE: {FEATURE_TOKEN_SCOPE}")


def aggregate_feature_acts(
    feature_acts: torch.Tensor, image_mask: torch.Tensor
) -> torch.Tensor:
    mask = feature_token_mask(image_mask)
    if mask.numel() != feature_acts.shape[0] or not mask.any():
        raise ValueError("Feature token selection is empty or has the wrong length.")
    x = feature_acts[mask].float()
    if FEATURE_AGGREGATION == "max":
        return x.amax(dim=0)
    if FEATURE_AGGREGATION == "mean":
        return x.mean(dim=0)
    raise ValueError(f"Invalid FEATURE_AGGREGATION: {FEATURE_AGGREGATION}")


def delta_to_residual_direction(
    layer_idx: int, feature_delta: torch.Tensor
) -> torch.Tensor:
    sae = saes[layer_idx]
    W_dec = sae.W_dec.detach().float().cpu()
    if feature_delta.numel() != W_dec.shape[0]:
        raise ValueError(
            f"SAE delta for layer {layer_idx}: {feature_delta.numel()} features, "
            f"but W_dec has {W_dec.shape[0]}."
        )
    return torch.matmul(feature_delta.float().cpu(), W_dec)


# ============================================================
# CONDITION / PROFILE STORAGE
# ============================================================
def top_signed_differences(
    layer_idx: int,
    delta: torch.Tensor,
    positive_label: str,
    negative_label: str,
) -> List[List[Any]]:
    k = min(TOP_DIFFS_TO_SHOW, int(delta.numel()))
    abs_values, indices = torch.topk(delta.abs(), k=k)
    rows: List[List[Any]] = []
    for rank, (feature_id, abs_value) in enumerate(
        zip(indices.tolist(), abs_values.tolist()),
        start=1,
    ):
        value = float(delta[int(feature_id)])
        rows.append(
            [
                layer_idx,
                rank,
                int(feature_id),
                value,
                float(abs_value),
                (
                    "NON-FINITE"
                    if not torch.isfinite(torch.tensor(value))
                    else (
                        positive_label
                        if value > 0
                        else negative_label if value < 0 else "equal"
                    )
                ),
            ]
        )
    return rows


def save_condition(
    condition_dir: Path,
    condition_name: str,
    prompt: str,
    image_path: Optional[str],
    input_ids_cpu: torch.Tensor,
    residuals: Dict[int, torch.Tensor],
) -> Dict[int, Dict[str, Any]]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    image_mask = get_image_mask(input_ids_cpu)
    result: Dict[int, Dict[str, Any]] = {}
    for layer_idx in LAYERS:
        resid = residuals[layer_idx]
        assert_finite(f"layer_{layer_idx}_resid_post_{condition_name}", resid)
        feature_acts = encode_sae_chunked(saes[layer_idx], resid)
        feature_score = aggregate_feature_acts(feature_acts, image_mask)
        assert_finite(
            f"layer_{layer_idx}_feature_score_{condition_name}", feature_score
        )
        payload: Dict[str, Any] = {
            "condition": condition_name,
            "capture_scope": "prompt_only_no_generation_no_assistant_prefix",
            "model_id": MODEL_ID,
            "layer": layer_idx,
            "sae_release": sae_releases_used[layer_idx],
            "sae_id": SAE_IDS[layer_idx],
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": FEATURE_TOKEN_SCOPE,
            "input_ids": input_ids_cpu,
            "image_token_mask": image_mask,
            "sae_activations": feature_acts,
            "feature_score": feature_score.float(),
        }
        if SAVE_RESIDUALS:
            payload["resid_post"] = resid
        torch.save(payload, condition_dir / f"layer_{layer_idx}.pt")
        residual_reference_norm = float(
            torch.linalg.vector_norm(
                resid[feature_token_mask(image_mask)].float(), dim=-1
            ).mean()
        )
        result[layer_idx] = {
            "feature_score": feature_score.cpu(),
            "residual_reference_norm": residual_reference_norm,
            "residual_stats": tensor_stats(
                f"layer_{layer_idx}_resid_post_{condition_name}", resid
            ),
            "feature_stats": tensor_stats(
                f"layer_{layer_idx}_sae_activations_{condition_name}", feature_acts
            ),
            "feature_score_stats": tensor_stats(
                f"layer_{layer_idx}_feature_score_{condition_name}", feature_score
            ),
        }
        del feature_acts
    metadata = {
        "condition": condition_name,
        "capture_scope": "prompt_only_no_generation_no_assistant_prefix",
        "model_id": MODEL_ID,
        "prompt": prompt,
        "image_filename": Path(image_path).name if image_path else None,
        "image_sha256": sha256_file(image_path),
        "num_input_tokens": int(input_ids_cpu.numel()),
        "num_image_tokens": int(image_mask.sum()),
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "num_scored_tokens": int(feature_token_mask(image_mask).sum()),
        "layers": LAYERS,
        "created_unix": time.time(),
    }
    with open(condition_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return result


def compute_common_feature_delta(
    deltas: List[torch.Tensor], epsilon: float
) -> Dict[str, torch.Tensor]:
    """An eligible feature has |B_i-A_i| > epsilon in EVERY pair.
    Presence is independent of sign. Opposing signs are reported; signed means
    may cancel. Only the masked mean (not any rejected feature) is decoded.
    """
    if len(deltas) < MIN_PAIRS:
        raise ValueError(f"At least {MIN_PAIRS} pair deltas are required.")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative.")
    shapes = {tuple(d.shape) for d in deltas}
    if len(shapes) != 1 or deltas[0].ndim != 1:
        raise ValueError("Each pair delta must be a 1-D tensor of identical length.")
    stack = torch.stack([d.detach().float().cpu() for d in deltas], dim=0)
    assert_finite("stacked_pair_deltas", stack)
    presence = stack.abs() > epsilon
    count = presence.sum(dim=0)
    common = presence.all(dim=0)
    mean = stack.mean(dim=0)
    masked_mean = torch.where(common, mean, torch.zeros_like(mean))
    positive = stack.gt(epsilon)
    negative = stack.lt(-epsilon)
    same_sign = common & (positive.all(dim=0) | negative.all(dim=0))
    opposing_sign = common & positive.any(dim=0) & negative.any(dim=0)
    canceled = common & masked_mean.abs().le(epsilon)
    effective = common & ~canceled
    return {
        "stack": stack,
        "presence": presence,
        "presence_count": count,
        "common_mask": common,
        "mean_delta": mean,
        "masked_mean_delta": masked_mean,
        "same_sign_mask": same_sign,
        "opposing_sign_mask": opposing_sign,
        "canceled_mask": canceled,
        "effective_mask": effective,
    }


def selected_feature_rows(
    layer_idx: int, result: Dict[str, torch.Tensor], included: bool
) -> List[List[Any]]:
    # A preview; the exported CSV retains EVERY feature and EVERY pair delta.
    counts = result["presence_count"]
    mask = (
        result["common_mask"] if included else (counts.gt(0) & ~result["common_mask"])
    )
    ids = mask.nonzero(as_tuple=True)[0].tolist()
    stack = result["stack"]
    mean = result["mean_delta"]
    if included:
        ids.sort(key=lambda j: abs(float(result["masked_mean_delta"][j])), reverse=True)
    else:
        ids.sort(key=lambda j: float(stack[:, j].abs().amax()), reverse=True)
    rows: List[List[Any]] = []
    for j in ids[:TOP_DIFFS_TO_SHOW]:
        pair_vals = [float(v) for v in stack[:, j].tolist()]
        rows.append(
            [
                layer_idx,
                j,
                int(counts[j]),
                len(pair_vals),
                round(float(mean[j]), 7),
                round(float(result["masked_mean_delta"][j]), 7),
                (
                    "sí"
                    if bool(result["same_sign_mask"][j])
                    else (
                        "signos opuestos"
                        if bool(result["opposing_sign_mask"][j])
                        else "no aplica"
                    )
                ),
                (
                    "cancelación"
                    if bool(result["canceled_mask"][j])
                    else ("incluida" if included else "EXCLUIDA")
                ),
                ", ".join(f"{v:+.5g}" for v in pair_vals),
            ]
        )
    return rows


def save_profile(
    profile_dir: Path,
    pairs_scores: List[Dict[str, Dict[int, Dict[str, Any]]]],
) -> Tuple[
    str,
    List[List[Any]],
    List[List[Any]],
    List[List[Any]],
    List[List[Any]],
    Dict[int, torch.Tensor],
    Dict[int, Dict[str, Any]],
]:
    """Save individual deltas and an intersection-only aggregated direction."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    n = len(pairs_scores)
    if n < MIN_PAIRS:
        raise ValueError(f"At least {MIN_PAIRS} pairs are required.")
    layer_rows, included_rows, excluded_rows, pair_rows = [], [], [], []
    directions: Dict[int, torch.Tensor] = {}
    layer_summary: Dict[int, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "layers": LAYERS,
        "num_pairs": n,
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "presence_epsilon": FEATURE_PRESENCE_EPS,
        "eligibility": "abs(score(B_i)-score(A_i)) > epsilon for EVERY pair, per layer",
        "sign_policy": "sign is not a presence criterion; report opposite signs; signed mean can cancel",
        "masked_delta": "where(common_mask, mean(pair_deltas), 0)",
        "raw_residual_formula": "raw_direction = masked_delta @ W_dec",
        "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
        "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
        "capture_scope": "prompt/image forward only; no generated response tokens",
        "layers_info": {},
    }
    report_path = profile_dir / "features_all_layers.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "feature_id",
                "present_pairs",
                "total_pairs",
                "common",
                "same_sign",
                "opposing_sign",
                "canceled",
                "effective",
                "mean_signed_delta",
                "masked_mean_delta",
                *[f"delta_pair_{i+1:02d}" for i in range(n)],
            ]
        )
        for layer_idx in LAYERS:
            pair_deltas: List[torch.Tensor] = []
            reference_norms: List[float] = []
            for pair_i, scores in enumerate(pairs_scores, start=1):
                a = scores["A"][layer_idx]["feature_score"].float().cpu()
                b = scores["B"][layer_idx]["feature_score"].float().cpu()
                delta = b - a
                assert_finite(f"layer_{layer_idx}_pair_{pair_i}_delta", delta)
                pair_deltas.append(delta)
                reference_norms.extend(
                    [
                        float(scores["A"][layer_idx]["residual_reference_norm"]),
                        float(scores["B"][layer_idx]["residual_reference_norm"]),
                    ]
                )
                top_rows = top_signed_differences(
                    layer_idx, delta, "mayor en B", "mayor en A"
                )
                pair_rows.extend([[pair_i, *row] for row in top_rows])
            result = compute_common_feature_delta(pair_deltas, FEATURE_PRESENCE_EPS)
            included_rows.extend(selected_feature_rows(layer_idx, result, True))
            excluded_rows.extend(selected_feature_rows(layer_idx, result, False))
            common = result["common_mask"]
            counts = result["presence_count"]
            masked = result["masked_mean_delta"]
            raw = delta_to_residual_direction(layer_idx, masked)
            assert_finite(f"layer_{layer_idx}_masked_residual_direction", raw)
            raw_norm = float(torch.linalg.vector_norm(raw))
            reference_norm = sum(reference_norms) / len(reference_norms)
            target_norm = reference_norm * STEERING_FRACTION_PER_UNIT
            direction = (
                raw * (target_norm / raw_norm)
                if raw_norm > 1e-12
                else torch.zeros_like(raw)
            )
            assert_finite(f"layer_{layer_idx}_scaled_direction", direction)
            directions[layer_idx] = direction.cpu()
            for j in range(masked.numel()):
                writer.writerow(
                    [
                        layer_idx,
                        j,
                        int(counts[j]),
                        n,
                        int(common[j]),
                        int(result["same_sign_mask"][j]),
                        int(result["opposing_sign_mask"][j]),
                        int(result["canceled_mask"][j]),
                        int(result["effective_mask"][j]),
                        float(result["mean_delta"][j]),
                        float(masked[j]),
                        *[float(v) for v in result["stack"][:, j].tolist()],
                    ]
                )
            any_presence = counts.gt(0)
            info = {
                "total_features": masked.numel(),
                "num_pairs": n,
                "present_in_at_least_one": int(any_presence.sum()),
                "common_features": int(common.sum()),
                "excluded_not_common": int((any_presence & ~common).sum()),
                "never_present": int((~any_presence).sum()),
                "common_same_sign": int(result["same_sign_mask"].sum()),
                "common_opposing_sign": int(result["opposing_sign_mask"].sum()),
                "common_canceled_mean": int(result["canceled_mask"].sum()),
                "effective_nonzero_features": int(result["effective_mask"].sum()),
                "pair_present_counts": [int(v) for v in result["presence"].sum(dim=1)],
                "masked_delta_l2": float(torch.linalg.vector_norm(masked)),
                "raw_residual_l2": raw_norm,
                "reference_residual_norm": reference_norm,
                "scaled_direction_l2": float(torch.linalg.vector_norm(direction)),
            }
            layer_summary[layer_idx] = info
            layer_rows.append(
                [
                    layer_idx,
                    n,
                    info["total_features"],
                    info["present_in_at_least_one"],
                    info["common_features"],
                    info["excluded_not_common"],
                    info["common_opposing_sign"],
                    info["common_canceled_mean"],
                    info["effective_nonzero_features"],
                    round(info["scaled_direction_l2"], 5),
                ]
            )
            summary["layers_info"][str(layer_idx)] = info
            torch.save(
                {
                    "layer": layer_idx,
                    "pair_deltas_B_minus_A": result["stack"],
                    "presence_per_pair": result["presence"],
                    "presence_count": counts,
                    "common_feature_mask": common,
                    "same_sign_mask": result["same_sign_mask"],
                    "opposing_sign_mask": result["opposing_sign_mask"],
                    "canceled_mask": result["canceled_mask"],
                    "effective_mask": result["effective_mask"],
                    "mean_feature_delta_unfiltered": result["mean_delta"],
                    "feature_delta_common_only": masked,
                    "feature_presence_epsilon": FEATURE_PRESENCE_EPS,
                    "raw_residual_direction": raw.float(),
                    "reference_residual_norm": reference_norm,
                    "scaled_steering_direction": direction.float(),
                    "W_dec_shape": tuple(saes[layer_idx].W_dec.shape),
                },
                profile_dir / f"layer_{layer_idx}_common_profile.pt",
            )
    (profile_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    profile_path = profile_dir / "steering_profile.pt"
    torch.save(
        {
            "format_version": 2,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layers": LAYERS,
            "num_pairs": n,
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": FEATURE_TOKEN_SCOPE,
            "sae_release": SAE_RELEASE,
            "sae_ids": SAE_IDS,
            "presence_epsilon": FEATURE_PRESENCE_EPS,
            "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
            "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
            "directions": {str(i): directions[i].float() for i in LAYERS},
        },
        profile_path,
    )
    return (
        str(profile_path.resolve()),
        layer_rows,
        included_rows,
        excluded_rows,
        pair_rows,
        directions,
        layer_summary,
    )


def zip_directory(run_dir: Path, output_name: str) -> str:
    zip_path = run_dir / output_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            if path == zip_path:
                continue
            zf.write(path, arcname=str(path.relative_to(run_dir)))
    return str(zip_path.resolve())


# ============================================================
# PROFILE LOADING FOR LATER QUERIES
# ============================================================
def load_steering_directions(profile_path: str) -> Dict[int, torch.Tensor]:
    payload = torch.load(profile_path, map_location="cpu", weights_only=True)
    if payload.get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"The profile belongs to {payload.get('model_id')}, but the current model is {MODEL_ID}."
        )
    directions = {
        int(layer_idx): tensor.float().cpu()
        for layer_idx, tensor in payload["directions"].items()
    }
    missing = [idx for idx in LAYERS if idx not in directions]
    if missing:
        raise RuntimeError(f"The steering profile is missing layers: {missing}")
    for layer_idx in LAYERS:
        direction = directions[layer_idx]
        expected_width = saes[layer_idx].W_dec.shape[1]
        if direction.ndim != 1 or direction.numel() != expected_width:
            raise ValueError(
                f"The steering direction for layer {layer_idx} has the wrong shape."
            )
        assert_finite(f"profile_layer_{layer_idx}", direction)
    if payload.get("format_version") != 2:
        raise ValueError(
            "This application requires a multi-pair profile (format_version=2)."
        )
    expected = {
        "model_revision": MODEL_REVISION,
        "sae_release": SAE_RELEASE,
        "sae_ids": SAE_IDS,
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
        "presence_epsilon": FEATURE_PRESENCE_EPS,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Profile configuration mismatch: {key}. Recalibrate.")
    return directions


# ============================================================
# CALLBACK 1: A/B CALIBRATION
# ============================================================
def calibrate_contrastive_profile_pairs(*args: Any):
    """Gradio inputs: count, 4*MAX_PAIRS conditions, generation controls."""
    pair_count = int(args[0])
    if not MIN_PAIRS <= pair_count <= MAX_PAIRS:
        raise gr.Error(f"Número de pares inválido ({MIN_PAIRS}-{MAX_PAIRS}).")
    pair_fields = args[1 : 1 + 4 * MAX_PAIRS]
    max_new_tokens, temperature, seed, make_previews = args[1 + 4 * MAX_PAIRS :]
    validate_generation_settings(max_new_tokens, temperature, seed)
    conditions = []
    for i in range(pair_count):
        pa, ia, pb, ib = pair_fields[4 * i : 4 * i + 4]
        pa = validate_condition_inputs(f"A{i+1}", ia, pa)
        pb = validate_condition_inputs(f"B{i+1}", ib, pb)
        conditions.append(
            {"A": {"prompt": pa, "image": ia}, "B": {"prompt": pb, "image": ib}}
        )
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    profile_dir = run_dir / "common_feature_profile_B_minus_A"
    run_dir.mkdir(parents=True, exist_ok=True)
    answers = ["" for _ in range(2 * MAX_PAIRS)]
    pairs_scores = []
    metadata_pairs: List[Dict[str, Any]] = []
    with MODEL_LOCK:
        ensure_models_loaded()
        for idx, pair in enumerate(conditions, start=1):
            pair_dir = run_dir / "pairs" / f"pair_{idx:02d}"
            pair_scores: Dict[str, Dict[int, Dict[str, Any]]] = {}
            pair_metadata: Dict[str, Any] = {"pair_index": idx}
            for label in ("A", "B"):
                prompt = pair[label]["prompt"]
                image_path = pair[label]["image"]
                # IMPORTANT: no assistant generation prefix or output tokens are
                # included in these calibration activations.
                inputs, ids_cpu, _ = prepare_inputs(
                    image_path, prompt, add_generation_prompt=False
                )
                residuals = capture_prompt_residuals(inputs)
                pair_scores[label] = save_condition(
                    pair_dir / f"condition_{label}",
                    label,
                    prompt,
                    image_path,
                    ids_cpu,
                    residuals,
                )
                del inputs, residuals
                answer = "(Vista previa desactivada: las activaciones son solo del prompt/imagen.)"
                if make_previews:
                    generation_inputs, _, generation_length = prepare_inputs(
                        image_path, prompt, add_generation_prompt=True
                    )
                    answer = generate_answer(
                        generation_inputs,
                        generation_length,
                        int(max_new_tokens),
                        float(temperature),
                        seed=int(seed),
                    )
                    del generation_inputs
                answers[2 * (idx - 1) + (0 if label == "A" else 1)] = answer
                pair_metadata[f"condition_{label}"] = {
                    "prompt": prompt,
                    "image_filename": Path(image_path).name if image_path else None,
                    "image_sha256": sha256_file(image_path),
                    "optional_preview_response": answer if make_previews else None,
                    "capture_scope": "prompt/image only; no generated answer tokens",
                }
            pairs_scores.append(pair_scores)
            metadata_pairs.append(pair_metadata)
        (
            profile_path,
            layer_rows,
            included_rows,
            excluded_rows,
            pair_rows,
            directions,
            layer_info,
        ) = save_profile(profile_dir, pairs_scores)
    experiment_metadata = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "layers": LAYERS,
        "num_pairs": pair_count,
        "pairs": metadata_pairs,
        "sae_ids": {str(k): v for k, v in SAE_IDS.items()},
        "sae_releases_used": {str(k): v for k, v in sae_releases_used.items()},
        "runtime": runtime_metadata(),
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "presence_epsilon": FEATURE_PRESENCE_EPS,
        "capture_scope": "prompt_only_no_assistant_prefix; no generation during capture",
        "optional_preview_generation": {
            "enabled": bool(make_previews),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        },
        "formula": "intersection(|delta_i|>epsilon) * mean_i(delta_i) @ W_dec",
        "created_unix": time.time(),
    }
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bundle = zip_directory(run_dir, "calibration_bundle.zip")
    state = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "profile_path": profile_path,
        "num_pairs": pair_count,
    }
    report = [
        f"**Perfil común creado:** `{run_id}` · **{pair_count} pares** · "
        f"criterio `|Bᵢ−Aᵢ| > {FEATURE_PRESENCE_EPS:g}` en TODOS los pares.  ",
        "Solo las features comunes pasan al decoder; las demás se anulan antes de aplicar `W_dec`.  ",
        "Las diferencias se han medido leyendo exclusivamente cada prompt/imagen, **sin respuestas generadas**.  ",
    ]
    for layer_idx in LAYERS:
        item = layer_info[layer_idx]
        report.append(
            f"- Capa **{layer_idx}**: **{item['common_features']}** comunes; "
            f"**{item['excluded_not_common']}** exclusivas de algunos pares (descartadas); "
            f"{item['common_opposing_sign']} comunes con signos opuestos; "
            f"{item['common_canceled_mean']} con media cancelada; "
            f"**{item['effective_nonzero_features']}** con señal final; "
            f"‖dirección α=1‖₂ = `{item['scaled_direction_l2']:.4f}`."
        )
    report.append(
        "La tabla es una vista previa; el ZIP incluye un CSV con **todas** las features y sus deltas por par."
    )
    return (
        *answers,
        layer_rows,
        included_rows,
        excluded_rows,
        pair_rows,
        bundle,
        state,
        "\n".join(report),
    )


def save_query_steered_activations(
    run_dir: Path,
    query_name: str,
    prompt: str,
    image_path: Optional[str],
    input_ids_cpu: torch.Tensor,
    residuals: Dict[int, torch.Tensor],
    strengths: Dict[int, float],
    generation_metadata: Dict[str, Any],
) -> str:
    query_dir = run_dir / "queries" / query_name
    query_dir.mkdir(parents=True, exist_ok=True)
    image_mask = get_image_mask(input_ids_cpu)
    for layer_idx in LAYERS:
        feature_acts = encode_sae_chunked(saes[layer_idx], residuals[layer_idx])
        score = aggregate_feature_acts(feature_acts, image_mask)
        payload: Dict[str, Any] = {
            "model_id": MODEL_ID,
            "layer": layer_idx,
            "prompt": prompt,
            "image_filename": Path(image_path).name if image_path else None,
            "input_ids": input_ids_cpu,
            "image_token_mask": image_mask,
            "sae_activations": feature_acts,
            "feature_score": score.float(),
            "steering_strength": float(strengths[layer_idx]),
            "feature_token_scope": FEATURE_TOKEN_SCOPE,
            "capture_scope": "prompt_only_after_steering",
        }
        if SAVE_RESIDUALS:
            payload["resid_post"] = residuals[layer_idx]
        torch.save(payload, query_dir / f"layer_{layer_idx}.pt")
        del feature_acts
    metadata = {
        "prompt": prompt,
        "image_filename": Path(image_path).name if image_path else None,
        "image_sha256": sha256_file(image_path),
        "strengths": {str(k): float(v) for k, v in strengths.items()},
        "layers": LAYERS,
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "capture_scope": "prompt_only_after_steering; separate forward pass; no generated answer tokens",
        "runtime": runtime_metadata(),
        "generation": generation_metadata,
        "created_unix": time.time(),
    }
    with open(query_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    zip_path = query_dir.parent / f"{query_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(query_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(query_dir.parent)))
    return str(zip_path.resolve())


# ============================================================
# CALLBACK 2: ARBITRARY QUERY TO THE STEERED MODEL
# ============================================================
def ask_steered_model(
    query_image: Optional[str],
    query_prompt: str,
    max_new_tokens: int,
    temperature: float,
    strength_9: float,
    strength_17: float,
    strength_29: float,
    session_state: Optional[Dict[str, Any]],
    seed: int = 0,
):
    validate_generation_settings(max_new_tokens, temperature, seed)
    query_prompt = validate_condition_inputs("query", query_image, query_prompt)
    if not session_state:
        raise gr.Error("Crea primero el perfil común en la sección 1.")
    profile_path = session_state.get("profile_path")
    if not profile_path or not Path(profile_path).exists():
        raise gr.Error("No se encuentra el perfil común. Vuelve a calibrarlo.")
    strengths = {
        9: float(strength_9),
        17: float(strength_17),
        29: float(strength_29),
    }
    if any(not math.isfinite(v) or abs(v) > 10 for v in strengths.values()):
        raise gr.Error("Steering strengths must be finite and between -10 and 10.")
    run_dir = Path(session_state["run_dir"])
    query_name = time.strftime("query_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    with MODEL_LOCK:
        ensure_models_loaded()
        directions = load_steering_directions(profile_path)
        base_inputs, input_ids_cpu, input_len = prepare_inputs(
            image_path=query_image,
            prompt=query_prompt,
        )
        steered_inputs, _, steered_input_len = prepare_inputs(
            image_path=query_image,
            prompt=query_prompt,
        )
        capture_inputs, _, _ = prepare_inputs(
            image_path=query_image,
            prompt=query_prompt,
        )
        base_answer = generate_answer(
            inputs=base_inputs,
            input_len=input_len,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            seed=int(seed),
        )
        steered_answer = generate_answer(
            inputs=steered_inputs,
            input_len=steered_input_len,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            steering_directions=directions,
            strengths=strengths,
            seed=int(seed),
        )
        steered_residuals = capture_prompt_residuals(
            inputs=capture_inputs,
            steering_directions=directions,
            strengths=strengths,
        )
        query_zip = save_query_steered_activations(
            run_dir=run_dir,
            query_name=query_name,
            prompt=query_prompt,
            image_path=query_image,
            input_ids_cpu=input_ids_cpu,
            residuals=steered_residuals,
            strengths=strengths,
            generation_metadata={
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "seed": int(seed),
                "profile_sha256": sha256_file(profile_path),
                "base_answer": base_answer,
                "steered_answer": steered_answer,
            },
        )
    status = (
        f"**Query `{query_name}`** · profile `{session_state['run_id']}`  \n"
        f"α9=`{strengths[9]:+.2f}` · α17=`{strengths[17]:+.2f}` · α29=`{strengths[29]:+.2f}`.  \n"
        f"BASE and STEERED use the same seed (`{int(seed)}`). "
        "Esta consulta reutiliza el perfil de features comunes sin recalibrar."
    )
    return base_answer, steered_answer, query_zip, status


# ============================================================
# GRADIO UI
# ============================================================
APP_CSS = """
:root {
    --app-radius: 18px;
}
.gradio-container {
    max-width: 1440px !important;
}
#hero {
    padding: 24px 28px;
    border-radius: 22px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, rgba(90,80,255,0.10), rgba(0,180,160,0.08));
    border: 1px solid rgba(127,127,127,0.18);
}
#hero h1 {
    margin-bottom: 8px;
}
.section-card {
    border: 1px solid rgba(127,127,127,0.18);
    border-radius: var(--app-radius);
    padding: 16px;
}
.condition-a {
    border-top: 4px solid rgba(90, 110, 255, 0.75);
}
.condition-b {
    border-top: 4px solid rgba(0, 170, 145, 0.75);
}
.result-card {
    border-top: 4px solid rgba(180, 120, 0, 0.72);
}
.small-note {
    font-size: 0.93rem;
    opacity: 0.86;
}
.formula-box {
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(127,127,127,0.18);
    background: rgba(127,127,127,0.045);
}
"""
TUTORIAL_MD = r"""
### Guía rápida
**1. Añade pares A/B.** Se muestran dos por defecto; puedes añadir hasta el máximo
configurado. Cada A y B admite texto, imagen o los dos. Cada par debe estar completo.
**2. Pulsa «Build common-feature profile».** Se ejecuta un forward pass para cada
prompt/imagen, sin generar tokens de respuesta. En cada capa se calculan las
activaciones SAE y un vector `delta_i = score(B_i) − score(A_i)` por par.
**3. Intersección estricta:** una feature se considera presente en un par cuando
`abs(delta_i) > FEATURE_PRESENCE_EPS`. Solo las features presentes en **todos**
los pares pasan al perfil. Se promedian sus deltas **con signo**. Si hay signos
opuestos, la interfaz los señala; si el promedio se cancela, no aporta steering.
**4. Consulta las tablas.** Se muestran recuentos por capa, features comunes,
features excluidas y diferencias por par. El ZIP conserva todas las features en
CSV y tensores `.pt` sin truncar.
**5. Consulta el modelo con otra imagen/prompt**, compara BASE con STEERED y
ajusta los sliders por capa. `0` = sin steering. El signo de alpha interviene
la activación, no garantiza un comportamiento determinado.
Las respuestas de calibración, si activas su vista previa, **nunca** se usan
para medir las activaciones ni para construir el perfil.
"""


def change_pair_count(count: int, change: int):
    count = max(MIN_PAIRS, min(MAX_PAIRS, int(count) + change))
    return (
        count,
        *[gr.update(visible=i < count) for i in range(MAX_PAIRS)],
        f"**Pares activos: {count}/{MAX_PAIRS}.** Los pares ocultos no se calibran.",
    )


def build_demo() -> gr.Blocks:
    """Construct the interface without downloading model weights."""
    with gr.Blocks(
        title="Gemma 3 · Common-Feature SAE Steering",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """# Gemma 3 · Common-Feature SAE Steering
Crea un perfil B − A reutilizable a partir de **varios pares multimodales**.
Únicamente se steerean las features presentes en **todos** los vectores B−A.""",
            elem_id="hero",
        )
        with gr.Accordion("How to use this app", open=True):
            gr.Markdown(TUTORIAL_MD)
        with gr.Accordion("Method and steering formula", open=False):
            gr.Markdown(f"""
<div class="formula-box">
**Capas:** `{LAYERS}` · **Agregación:** `{FEATURE_AGGREGATION}` ·
**Tokens:** `{FEATURE_TOKEN_SCOPE}` · **Epsilon:** `{FEATURE_PRESENCE_EPS:g}`  
**Fraction/alpha:** `{STEERING_FRACTION_PER_UNIT}` ·
**Steer last token only:** `{STEER_LAST_TOKEN_ONLY}`
Para cada par `i` y cada capa `l`:
`d_i = score_SAE(B_i) − score_SAE(A_i)`  
`common[j] = AND_i(abs(d_i[j]) > epsilon)`  
`d_common = where(common, mean_i(d_i), 0)`  
`v_raw = d_common @ W_dec`  
`v = normalize(v_raw) × reference_residual_norm × STEERING_FRACTION_PER_UNIT`  
`h_last' = h_last + alpha × v`
El criterio de presencia no exige igualdad de signos: las inversiones de signo
se muestran por separado. Las activaciones de calibración se extraen **solo
sobre la lectura del prompt/imagen**, sin prefijo de generación del asistente.
</div>
""")
        session_state = gr.State(value=None)
        pair_count = gr.State(value=MIN_PAIRS)
        gr.Markdown("## 1. Build the common B − A steering profile")
        gr.Markdown(
            "Define al menos dos pares independientes. Cada condición debe contener texto, imagen o ambos.",
            elem_classes=["small-note"],
        )
        with gr.Row():
            add_pair = gr.Button("＋ Add A/B pair", variant="secondary")
            remove_pair = gr.Button("− Remove last pair", variant="secondary")
        pair_count_label = gr.Markdown(
            f"**Pares activos: {MIN_PAIRS}/{MAX_PAIRS}.** Los pares ocultos no se calibran."
        )
        pair_groups = []
        flat_pair_inputs = []
        flat_pair_answers = []
        for pair_i in range(MAX_PAIRS):
            with gr.Group(visible=pair_i < MIN_PAIRS) as pair_group:
                gr.Markdown(f"### Par {pair_i+1} · B{pair_i+1} − A{pair_i+1}")
                with gr.Row(equal_height=True):
                    with gr.Column(
                        scale=1, elem_classes=["section-card", "condition-a"]
                    ):
                        gr.Markdown(f"#### Condition A{pair_i+1} · reference/source")
                        image_a = gr.Image(
                            label=f"Image A{pair_i+1} · optional",
                            type="filepath",
                            height=240,
                        )
                        prompt_a = gr.Textbox(
                            label=f"Prompt A{pair_i+1} · optional if image is provided",
                            placeholder="Describe this scene in a neutral way.",
                            lines=4,
                        )
                        answer_a = gr.Textbox(
                            label=f"Gemma response for A{pair_i+1} (optional preview)",
                            lines=5,
                            interactive=False,
                        )
                    with gr.Column(
                        scale=1, elem_classes=["section-card", "condition-b"]
                    ):
                        gr.Markdown(f"#### Condition B{pair_i+1} · target/comparison")
                        image_b = gr.Image(
                            label=f"Image B{pair_i+1} · optional",
                            type="filepath",
                            height=240,
                        )
                        prompt_b = gr.Textbox(
                            label=f"Prompt B{pair_i+1} · optional if image is provided",
                            placeholder="Describe this scene focusing on emotion and atmosphere.",
                            lines=4,
                        )
                        answer_b = gr.Textbox(
                            label=f"Gemma response for B{pair_i+1} (optional preview)",
                            lines=5,
                            interactive=False,
                        )
            pair_groups.append(pair_group)
            flat_pair_inputs.extend([prompt_a, image_a, prompt_b, image_b])
            flat_pair_answers.extend([answer_a, answer_b])
        add_pair.click(
            fn=lambda n: change_pair_count(n, +1),
            inputs=[pair_count],
            outputs=[pair_count, *pair_groups, pair_count_label],
        )
        remove_pair.click(
            fn=lambda n: change_pair_count(n, -1),
            inputs=[pair_count],
            outputs=[pair_count, *pair_groups, pair_count_label],
        )
        calibration_previews = gr.Checkbox(
            label="Generate optional calibration response previews (not used for activation capture)",
            value=False,
        )
        with gr.Row():
            calibration_max_new_tokens = gr.Slider(
                1,
                1024,
                value=192,
                step=1,
                label="Preview response max new tokens",
            )
            calibration_temperature = gr.Slider(
                0.0,
                2.0,
                value=0.0,
                step=0.05,
                label="Preview temperature · 0 = greedy",
            )
        calibration_seed = gr.Number(
            value=0, precision=0, label="Preview seed · shared by all A and B"
        )
        calibrate_btn = gr.Button(
            "Build common-feature profile", variant="primary", size="lg"
        )
        calibration_status = gr.Markdown()
        calibration_bundle = gr.File(
            label="Download full calibration bundle · all pairs + intersection + profile"
        )
        gr.Markdown("### Feature intersection by layer")
        gr.Markdown(
            "Las features presentes en algunos pares, pero no en todos, se descartan incluso si sus deltas son grandes.",
            elem_classes=["small-note"],
        )
        layer_table = gr.Dataframe(
            headers=[
                "layer",
                "pairs",
                "total_features",
                "present_any",
                "present_all",
                "excluded_not_common",
                "common_opposite_signs",
                "common_canceled",
                "effective_features",
                "scaled_direction_L2",
            ],
            datatype=["number"] * 10,
            interactive=False,
            wrap=True,
        )
        gr.Markdown(f"### Common features (top {TOP_DIFFS_TO_SHOW} per layer)")
        cols = [
            "layer",
            "feature_id",
            "present_pairs",
            "total_pairs",
            "mean_delta",
            "delta_used_for_steering",
            "signs",
            "status",
            "individual_pair_deltas",
        ]
        types = [
            "number",
            "number",
            "number",
            "number",
            "number",
            "number",
            "str",
            "str",
            "str",
        ]
        included_table = gr.Dataframe(
            headers=cols, datatype=types, interactive=False, wrap=True
        )
        gr.Markdown(f"### Discarded features (top {TOP_DIFFS_TO_SHOW} per layer)")
        excluded_table = gr.Dataframe(
            headers=cols, datatype=types, interactive=False, wrap=True
        )
        with gr.Accordion("Largest individual B−A differences by pair", open=False):
            pair_table = gr.Dataframe(
                headers=[
                    "pair",
                    "layer",
                    "rank",
                    "feature_id",
                    "signed_delta",
                    "abs_delta",
                    "interpretation",
                ],
                datatype=["number"] * 6 + ["str"],
                interactive=False,
                wrap=True,
            )
        gr.Markdown("## 2. Query the calibrated steered model")
        gr.Markdown(
            "Consulta con un prompt y/o imagen nuevo. Se reutiliza el perfil filtrado sin recalibrar.",
            elem_classes=["small-note"],
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["section-card"]):
                query_image = gr.Image(
                    label="Query image · optional", type="filepath", height=300
                )
                query_prompt = gr.Textbox(
                    label="Query prompt", placeholder="Ask Gemma anything...", lines=5
                )
                with gr.Row():
                    query_max_new_tokens = gr.Slider(
                        1, 1024, value=192, step=1, label="Max new tokens"
                    )
                    query_temperature = gr.Slider(
                        0, 2, value=0, step=0.05, label="Temperature · 0 = greedy"
                    )
                query_seed = gr.Number(
                    value=0,
                    precision=0,
                    label="Query seed · shared by BASE and STEERED",
                )
                gr.Markdown("#### Steering strength by layer")
                gr.Markdown(
                    "`0` = off · positive = toward mean B−A · negative = toward A−B",
                    elem_classes=["small-note"],
                )
                strength_9 = gr.Slider(
                    -10, 10, value=0, step=0.25, label="Layer 9 · alpha"
                )
                strength_17 = gr.Slider(
                    -10, 10, value=0, step=0.25, label="Layer 17 · alpha"
                )
                strength_29 = gr.Slider(
                    -10, 10, value=0, step=0.25, label="Layer 29 · alpha"
                )
                ask_btn = gr.Button("Run BASE + STEERED", variant="primary", size="lg")
                query_status = gr.Markdown()
                query_archive = gr.File(
                    label="Download steered PROMPT activations + response metadata"
                )
            with gr.Column(scale=1, elem_classes=["section-card", "result-card"]):
                query_base_answer = gr.Textbox(
                    label="BASE response", lines=13, interactive=False
                )
                query_steered_answer = gr.Textbox(
                    label="STEERED response", lines=13, interactive=False
                )
        with gr.Accordion("Saved artifacts and experiment structure", open=False):
            gr.Markdown("""
```text
run_id/
├── pairs/
│   ├── pair_01/
│   │   ├── condition_A/{layer_9.pt,layer_17.pt,layer_29.pt,metadata.json}
│   │   └── condition_B/{layer_9.pt,layer_17.pt,layer_29.pt,metadata.json}
│   ├── pair_02/...
│   └── ...
├── common_feature_profile_B_minus_A/
│   ├── layer_9_common_profile.pt
│   ├── layer_17_common_profile.pt
│   ├── layer_29_common_profile.pt
│   ├── features_all_layers.csv     # every feature + every pair delta
│   ├── steering_profile.pt
│   └── profile_summary.json
├── queries/
└── experiment.json
```
Los `.pt` individuales retienen activaciones SAE token por token y scores FP32.
El perfil almacena todos los deltas, la máscara de intersección y la dirección filtrada.
Las activaciones de consultas se capturan en un **forward separado sobre el prompt**, nunca sobre la respuesta.
""")
        calibrate_btn.click(
            fn=calibrate_contrastive_profile_pairs,
            inputs=[
                pair_count,
                *flat_pair_inputs,
                calibration_max_new_tokens,
                calibration_temperature,
                calibration_seed,
                calibration_previews,
            ],
            outputs=[
                *flat_pair_answers,
                layer_table,
                included_table,
                excluded_table,
                pair_table,
                calibration_bundle,
                session_state,
                calibration_status,
            ],
        )
        ask_btn.click(
            fn=ask_steered_model,
            inputs=[
                query_image,
                query_prompt,
                query_max_new_tokens,
                query_temperature,
                strength_9,
                strength_17,
                strength_29,
                session_state,
                query_seed,
            ],
            outputs=[
                query_base_answer,
                query_steered_answer,
                query_archive,
                query_status,
            ],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Overview")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate imports, SAE registry and UI without model downloads",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load Gemma and the SAEs before serving the UI",
    )
    args = parser.parse_args()
    validate_configuration()
    validate_sae_registry()
    demo = build_demo()
    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "model_weights_loaded": False,
                    "sae_release": SAE_RELEASE,
                    "runtime": runtime_metadata(),
                },
                indent=2,
            )
        )
        demo.close()
        return
    if args.preload:
        with MODEL_LOCK:
            ensure_models_loaded()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        css=APP_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
