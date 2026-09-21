#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma 3 4B-IT + Gemma Scope 2 SAEs
General contrastive B - A steering with a Gradio web interface.

Overview
========
This application builds a reusable steering profile from two arbitrary input
conditions:

    Condition A = (Prompt A, optional Image A)
    Condition B = (Prompt B, optional Image B)

Each condition may contain:
- text only,
- image + text,
- image only,
- or even the same image with different prompts.

For every configured transformer layer, the app:
1. captures the post-layer residual stream (`resid_post`),
2. encodes it with the matching Gemma Scope 2 sparse autoencoder (SAE),
3. aggregates token-level SAE activations into one feature vector,
4. computes the contrastive feature direction:

       feature_delta = score(B) - score(A)

5. projects that feature-space difference back into residual space:

       raw_direction = feature_delta @ W_dec

6. rescales the residual direction to a stable magnitude relative to the
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

   uv run python demo_gradio_hackaton.py

4. Open the local Gradio URL shown in the terminal.

Optional environment variables
==============================
GEMMA_MODEL_ID                default: google/gemma-3-4b-it
GEMMA_SAE_RUNS_DIR            default: gemma_sae_contrastive_runs_ab
GRADIO_SERVER_NAME            default: 0.0.0.0
GRADIO_SERVER_PORT            default: 7860
TOP_DIFFS_TO_SHOW             default: 20
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
SAE_CHUNK_TOKENS = int(os.getenv("SAE_CHUNK_TOKENS", "128"))
SAVE_RESIDUALS = os.getenv("SAVE_RESIDUALS", "1") != "0"

RUNS_DIR = Path(
    os.getenv("GEMMA_SAE_RUNS_DIR", "gemma_sae_contrastive_runs_ab")
)
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
        raise RuntimeError("MPS is unavailable. Check macOS/PyTorch or use GEMMA_DEVICE=cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and the PyTorch build.")
    dtype = "auto" if MODEL_DTYPE == "auto" else getattr(torch, MODEL_DTYPE)
    return {"device_map": "auto" if device == "auto" else {"": device}, "torch_dtype": dtype}


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
        sae = unwrap_sae(
            SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_IDS[layer_idx])
        ).cpu().eval()
        if not hasattr(sae, "W_dec") or sae.W_dec.shape[1] != hidden_size:
            raise ValueError(f"Layer {layer_idx}: SAE decoder does not match model hidden size.")
        if sae.cfg.metadata.hook_name != f"blocks.{layer_idx}.hook_resid_post":
            raise ValueError(f"Layer {layer_idx}: SAE hook does not match resid_post.")
        if sae.cfg.normalize_activations != "none":
            raise ValueError("This steering implementation requires an unnormalized SAE decoder.")
        saes[layer_idx] = sae
        sae_releases_used[layer_idx] = SAE_RELEASE
    print(f"Model and SAEs ready. Primary device: {model.device}")


def runtime_metadata() -> Dict[str, Any]:
    packages = ("torch", "torchvision", "transformers", "sae-lens", "gradio", "accelerate")
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


def validate_generation_settings(max_new_tokens: int, temperature: float, seed: int) -> None:
    if not 1 <= int(max_new_tokens) <= 1024:
        raise gr.Error("Max new tokens must be between 1 and 1024.")
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) <= 2:
        raise gr.Error("Temperature must be between 0 and 2.")
    if not math.isfinite(float(seed)) or int(seed) != seed or not 0 <= int(seed) < 2**32:
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
) -> Tuple[Any, torch.Tensor, int]:
    messages = make_messages(image_path, prompt)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
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



def delta_to_residual_direction(layer_idx: int, feature_delta: torch.Tensor) -> torch.Tensor:
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
                    "NON-FINITE" if not torch.isfinite(torch.tensor(value))
                    else positive_label if value > 0
                    else negative_label if value < 0
                    else "equal"
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
        assert_finite(f"layer_{layer_idx}_feature_score_{condition_name}", feature_score)

        payload: Dict[str, Any] = {
            "condition": condition_name,
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
            torch.linalg.vector_norm(resid[feature_token_mask(image_mask)].float(), dim=-1).mean()
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



def save_profile(
    profile_dir: Path,
    a_scores: Dict[int, Dict[str, Any]],
    b_scores: Dict[int, Dict[str, Any]],
) -> Tuple[str, List[List[Any]], Dict[int, torch.Tensor]]:
    profile_dir.mkdir(parents=True, exist_ok=True)

    table_rows: List[List[Any]] = []
    steering_directions: Dict[int, torch.Tensor] = {}
    summary: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "layers": LAYERS,
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "formula": "feature_delta = score(B) - score(A)",
        "raw_residual_formula": "raw_direction = feature_delta @ W_dec",
        "steering_scaling": (
            "scaled_direction = normalize(raw_direction) * "
            "reference_residual_norm * STEERING_FRACTION_PER_UNIT"
        ),
        "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
        "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
        "layers_info": {},
    }

    for layer_idx in LAYERS:
        a = a_scores[layer_idx]["feature_score"].float()
        b = b_scores[layer_idx]["feature_score"].float()
        assert_finite(f"layer_{layer_idx}_A_feature_score", a)
        assert_finite(f"layer_{layer_idx}_B_feature_score", b)
        delta = b - a
        assert_finite(f"layer_{layer_idx}_feature_delta_B_minus_A", delta)

        raw_direction = delta_to_residual_direction(layer_idx, delta).float()
        assert_finite(f"layer_{layer_idx}_raw_residual_direction", raw_direction)
        raw_norm = float(torch.linalg.vector_norm(raw_direction))

        a_resid_norm = float(a_scores[layer_idx]["residual_reference_norm"])
        b_resid_norm = float(b_scores[layer_idx]["residual_reference_norm"])
        reference_resid_norm = 0.5 * (a_resid_norm + b_resid_norm)

        target_norm_per_alpha = reference_resid_norm * STEERING_FRACTION_PER_UNIT
        if raw_norm > 1e-12:
            direction = raw_direction * (target_norm_per_alpha / raw_norm)
        else:
            direction = torch.zeros_like(raw_direction)
        assert_finite(f"layer_{layer_idx}_scaled_direction", direction)

        steering_directions[layer_idx] = direction.cpu()
        table_rows.extend(
            top_signed_differences(
                layer_idx=layer_idx,
                delta=delta,
                positive_label="higher in B",
                negative_label="higher in A",
            )
        )

        nonzero = int(delta.ne(0).sum())
        positive = int(delta.gt(0).sum())
        negative = int(delta.lt(0).sum())

        torch.save(
            {
                "layer": layer_idx,
                "aggregation": FEATURE_AGGREGATION,
                "A_feature_score": a.float(),
                "B_feature_score": b.float(),
                "feature_delta_B_minus_A": delta.float(),
                "raw_residual_direction": raw_direction.float(),
                "raw_residual_direction_l2": raw_norm,
                "reference_residual_norm": reference_resid_norm,
                "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
                "scaled_steering_direction": direction.float(),
                "scaled_direction_l2": float(torch.linalg.vector_norm(direction)),
                "W_dec_shape": tuple(saes[layer_idx].W_dec.shape),
            },
            profile_dir / f"layer_{layer_idx}_contrastive_profile.pt",
        )

        summary["layers_info"][str(layer_idx)] = {
            "feature_delta_l2": float(torch.linalg.vector_norm(delta)),
            "raw_residual_direction_l2": raw_norm,
            "reference_residual_norm": reference_resid_norm,
            "scaled_direction_l2_per_alpha_1": float(torch.linalg.vector_norm(direction)),
            "alpha_10_fraction_of_reference_norm": 10.0 * STEERING_FRACTION_PER_UNIT,
            "nonzero_features": nonzero,
            "positive_features": positive,
            "negative_features": negative,
            "A_feature_score_stats": a_scores[layer_idx].get("feature_score_stats"),
            "B_feature_score_stats": b_scores[layer_idx].get("feature_score_stats"),
            "delta_stats": tensor_stats(f"layer_{layer_idx}_feature_delta_B_minus_A", delta),
        }

    with open(profile_dir / "profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    profile_path = profile_dir / "steering_profile.pt"
    torch.save(
        {
            "format_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layers": LAYERS,
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": FEATURE_TOKEN_SCOPE,
            "sae_release": SAE_RELEASE,
            "sae_ids": SAE_IDS,
            "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
            "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
            "directions": {
                str(layer_idx): steering_directions[layer_idx].float()
                for layer_idx in LAYERS
            },
        },
        profile_path,
    )

    return str(profile_path.resolve()), table_rows, steering_directions



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
            raise ValueError(f"The steering direction for layer {layer_idx} has the wrong shape.")
        assert_finite(f"profile_layer_{layer_idx}", direction)
    if payload.get("format_version") == 1:
        expected = {
            "model_revision": MODEL_REVISION,
            "sae_release": SAE_RELEASE,
            "sae_ids": SAE_IDS,
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": FEATURE_TOKEN_SCOPE,
            "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Profile configuration mismatch: {key}. Recalibrate.")

    return directions


# ============================================================
# CALLBACK 1: A/B CALIBRATION
# ============================================================


def calibrate_contrastive_profile_ab(
    prompt_a: str,
    image_a: Optional[str],
    prompt_b: str,
    image_b: Optional[str],
    max_new_tokens: int,
    temperature: float,
    seed: int = 0,
):
    validate_generation_settings(max_new_tokens, temperature, seed)
    prompt_a = validate_condition_inputs("A", image_a, prompt_a)
    prompt_b = validate_condition_inputs("B", image_b, prompt_b)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    cond_a_dir = run_dir / "condition_A"
    cond_b_dir = run_dir / "condition_B"
    profile_dir = run_dir / "contrastive_profile_B_minus_A"
    run_dir.mkdir(parents=True, exist_ok=True)

    with MODEL_LOCK:
        ensure_models_loaded()
        # ---------------------- Condition A ----------------------
        a_inputs, a_ids_cpu, a_input_len = prepare_inputs(
            image_path=image_a,
            prompt=prompt_a,
        )
        a_residuals = capture_prompt_residuals(a_inputs)
        a_scores = save_condition(
            condition_dir=cond_a_dir,
            condition_name="A",
            prompt=prompt_a,
            image_path=image_a,
            input_ids_cpu=a_ids_cpu,
            residuals=a_residuals,
        )
        a_answer = generate_answer(
            inputs=a_inputs,
            input_len=a_input_len,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            seed=int(seed),
        )
        del a_inputs, a_residuals

        # ---------------------- Condition B ----------------------
        b_inputs, b_ids_cpu, b_input_len = prepare_inputs(
            image_path=image_b,
            prompt=prompt_b,
        )
        b_residuals = capture_prompt_residuals(b_inputs)
        b_scores = save_condition(
            condition_dir=cond_b_dir,
            condition_name="B",
            prompt=prompt_b,
            image_path=image_b,
            input_ids_cpu=b_ids_cpu,
            residuals=b_residuals,
        )
        b_answer = generate_answer(
            inputs=b_inputs,
            input_len=b_input_len,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            seed=int(seed),
        )
        del b_inputs, b_residuals

        # ---------------------- B - A profile ----------------------
        profile_path, delta_rows, directions = save_profile(
            profile_dir=profile_dir,
            a_scores=a_scores,
            b_scores=b_scores,
        )

    experiment_metadata = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "layers": LAYERS,
        "sae_ids": {str(k): v for k, v in SAE_IDS.items()},
        "sae_releases_used": {str(k): v for k, v in sae_releases_used.items()},
        "runtime": runtime_metadata(),
        "generation": {"max_new_tokens": int(max_new_tokens), "temperature": float(temperature), "seed": int(seed)},
        "condition_A": {
            "prompt": prompt_a,
            "image": Path(image_a).name if image_a else None,
            "image_sha256": sha256_file(image_a),
            "answer": a_answer,
        },
        "condition_B": {
            "prompt": prompt_b,
            "image": Path(image_b).name if image_b else None,
            "image_sha256": sha256_file(image_b),
            "answer": b_answer,
        },
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": FEATURE_TOKEN_SCOPE,
        "capture_scope": "prompt_only; post-layer residuals; no generated answer tokens",
        "steering_formula": (
            "h_last' = h_last + alpha * scaled_direction; "
            "scaled_direction = normalize((score(B)-score(A)) @ W_dec) "
            "* reference_residual_norm * STEERING_FRACTION_PER_UNIT"
        ),
        "created_unix": time.time(),
    }
    with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2, ensure_ascii=False)

    bundle_zip = zip_directory(run_dir, "calibration_bundle.zip")

    state = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "profile_path": profile_path,
    }

    norm_lines = []
    for layer_idx in LAYERS:
        norm_lines.append(
            f"layer `{layer_idx}`: ||scaled direction (α=1)||₂ = "
            f"`{float(torch.linalg.vector_norm(directions[layer_idx])):.4f}`"
        )

    status = (
        f"**Contrastive profile created:** `{run_id}`  \n"
        f"Contrast: `score(B) - score(A)`; aggregation `{FEATURE_AGGREGATION}`; tokens `{FEATURE_TOKEN_SCOPE}`.  \n"
        + " · ".join(norm_lines)
        + "  \nYou can now query any new text/image below without recalibrating."
    )

    return (
        a_answer,
        b_answer,
        delta_rows,
        bundle_zip,
        state,
        status,
    )


# ============================================================
# SAVE ACTIVATIONS FOR A STEERED QUERY
# ============================================================


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
        raise gr.Error("Create a profile first in '1) Build the B - A steering profile'.")

    profile_path = session_state.get("profile_path")
    if not profile_path or not Path(profile_path).exists():
        raise gr.Error("The steering profile could not be found. Please recalibrate the experiment.")

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
        "This query reuses the saved B−A profile without recalibration."
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
### Quick tutorial

**1. Define Condition A and Condition B.**  
Each condition can contain **text only**, **image only**, or **image + text**. The two
conditions are fully independent, and they may even use the **same image**.

**2. Click “Build B − A profile”.**  
For layers **9, 17, and 29**, the app captures `resid_post`, encodes it with the
corresponding SAE, aggregates feature activations, and computes:

`feature_delta = score(B) - score(A)`

The feature delta is decoded back to residual space with `W_dec` and rescaled to a
stable steering magnitude.

**3. Inspect the feature-difference table.**  
Positive values are features that are stronger in **B**. Negative values are features
that are stronger in **A**.

**4. Query the steered model with anything.**  
After calibration, the query section is independent from A and B. You can enter a new
text prompt, upload a different image, use text only, or use image + text.

**5. Adjust the per-layer steering sliders.**
- `0` = no steering
- positive values = add the **B − A** activation direction
- negative values = subtract the **B − A** activation direction

Start around `±1` and increase gradually. Large values can strongly distort generation.
The sign specifies an activation intervention, not a guaranteed behavioral effect.
This experiment does not calculate a percentage of image/text influence or remove a modality.
"""


def build_demo() -> gr.Blocks:
    """Build the interface without loading model weights."""
    with gr.Blocks(
        title="Gemma 3 · Contrastive SAE Steering",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """
    # Gemma 3 · Contrastive SAE Steering
    Build a reusable **B − A sparse-feature steering profile** from two arbitrary multimodal conditions, then apply it to completely new Gemma 3 queries.
            """,
            elem_id="hero",
        )

        with gr.Accordion("How to use this app", open=True):
            gr.Markdown(TUTORIAL_MD)

        with gr.Accordion("Method and steering formula", open=False):
            gr.Markdown(
                f"""
    <div class="formula-box">

    **Configured layers:** `{LAYERS}`  
    **Feature aggregation:** `{FEATURE_AGGREGATION}`  
    **Tokens used for feature scoring:** `{FEATURE_TOKEN_SCOPE}`  
    **Steering fraction per alpha unit:** `{STEERING_FRACTION_PER_UNIT}`  
    **Steer last token only:** `{STEER_LAST_TOKEN_ONLY}`

    For each layer:

    `d = score_SAE(B) - score_SAE(A)`  
    `v_raw = d @ W_dec`  
    `v = normalize(v_raw) × reference_residual_norm × STEERING_FRACTION_PER_UNIT`  
    `h_last' = h_last + alpha × v`

    The raw, unscaled feature delta and residual direction are saved to disk for analysis.
    `all` includes image and template tokens; `non_image` excludes image placeholders;
    `last` uses the last prompt position. None is a causal attribution percentage.

    </div>
                """
            )

        session_state = gr.State(value=None)

        # --------------------------------------------------------
        # STEP 1: A/B CALIBRATION
        # --------------------------------------------------------
        gr.Markdown("## 1. Build the B − A steering profile")
        gr.Markdown(
            "Define two independent conditions. Each one must contain at least text, an image, or both.",
            elem_classes=["small-note"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["section-card", "condition-a"]):
                gr.Markdown("### Condition A · reference / source")
                image_a = gr.Image(
                    label="Image A · optional",
                    type="filepath",
                    height=300,
                )
                prompt_a = gr.Textbox(
                    label="Prompt A · optional when an image is provided",
                    placeholder="Example: Describe this scene in a neutral way.",
                    lines=5,
                )
                answer_a = gr.Textbox(
                    label="Gemma response for Condition A",
                    lines=8,
                    interactive=False,
                )

            with gr.Column(scale=1, elem_classes=["section-card", "condition-b"]):
                gr.Markdown("### Condition B · target / comparison")
                image_b = gr.Image(
                    label="Image B · optional",
                    type="filepath",
                    height=300,
                )
                prompt_b = gr.Textbox(
                    label="Prompt B · optional when an image is provided",
                    placeholder="Example: Describe this scene focusing on emotion and atmosphere.",
                    lines=5,
                )
                answer_b = gr.Textbox(
                    label="Gemma response for Condition B",
                    lines=8,
                    interactive=False,
                )

        with gr.Row():
            calibration_max_new_tokens = gr.Slider(
                minimum=1,
                maximum=1024,
                value=192,
                step=1,
                label="Calibration response max new tokens",
            )
            calibration_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.0,
                step=0.05,
                label="Calibration temperature · 0 = greedy",
            )
        calibration_seed = gr.Number(value=0, precision=0, label="Calibration seed · shared by A and B")

        calibrate_btn = gr.Button(
            "Build B − A profile",
            variant="primary",
            size="lg",
        )
        calibration_status = gr.Markdown()

        with gr.Row():
            calibration_bundle = gr.File(
                label="Download calibration bundle · A + B + B−A profile",
            )

        gr.Markdown("### Largest SAE feature differences")
        gr.Markdown(
            "`delta = score(B) - score(A)`. Positive means stronger in B; negative means stronger in A.",
            elem_classes=["small-note"],
        )

        delta_table = gr.Dataframe(
            headers=[
                "layer",
                "rank",
                "feature_id",
                "signed_delta",
                "absolute_delta",
                "interpretation",
            ],
            datatype=["number", "number", "number", "number", "number", "str"],
            interactive=False,
            wrap=True,
        )

        # --------------------------------------------------------
        # STEP 2: ARBITRARY STEERED QUERY
        # --------------------------------------------------------
        gr.Markdown("## 2. Query the calibrated steered model")
        gr.Markdown(
            "This input is independent from Conditions A and B. The saved B−A profile is reused without recalibration.",
            elem_classes=["small-note"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["section-card"]):
                query_image = gr.Image(
                    label="Query image · optional",
                    type="filepath",
                    height=300,
                )
                query_prompt = gr.Textbox(
                    label="Query prompt",
                    placeholder="Ask Gemma anything...",
                    lines=5,
                )

                with gr.Row():
                    query_max_new_tokens = gr.Slider(
                        minimum=1,
                        maximum=1024,
                        value=192,
                        step=1,
                        label="Max new tokens",
                    )
                    query_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.05,
                        label="Temperature · 0 = greedy",
                    )
                query_seed = gr.Number(value=0, precision=0, label="Query seed · shared by BASE and STEERED")

                gr.Markdown("#### Steering strength by layer")
                gr.Markdown(
                    "`0` = off · positive = toward B−A · negative = toward A−B",
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

                ask_btn = gr.Button(
                    "Run BASE + STEERED",
                    variant="primary",
                    size="lg",
                )
                query_status = gr.Markdown()
                query_archive = gr.File(
                    label="Download steered prompt activations + response metadata",
                )

            with gr.Column(scale=1, elem_classes=["section-card", "result-card"]):
                query_base_answer = gr.Textbox(
                    label="BASE response",
                    lines=13,
                    interactive=False,
                )
                query_steered_answer = gr.Textbox(
                    label="STEERED response",
                    lines=13,
                    interactive=False,
                )

        with gr.Accordion("Saved artifacts and experiment structure", open=False):
            gr.Markdown(
                """
    Each calibration run contains roughly:

    ```text
    run_id/
    ├── condition_A/
    │   ├── layer_9.pt
    │   ├── layer_17.pt
    │   ├── layer_29.pt
    │   └── metadata.json
    ├── condition_B/
    │   ├── layer_9.pt
    │   ├── layer_17.pt
    │   ├── layer_29.pt
    │   └── metadata.json
    ├── contrastive_profile_B_minus_A/
    │   ├── layer_9_contrastive_profile.pt
    │   ├── layer_17_contrastive_profile.pt
    │   ├── layer_29_contrastive_profile.pt
    │   ├── steering_profile.pt
    │   └── profile_summary.json
    ├── queries/
    └── experiment.json
    ```

    The `.pt` files retain FP32 SAE scores/deltas and residual directions so the experiment can be inspected outside the web UI.
    Query activations come from a separate forward pass over the prompt, after steering.
    They do not record the generated answer tokens. Responses, seeds and generation settings are saved in JSON.
                """
            )

        calibrate_btn.click(
            fn=calibrate_contrastive_profile_ab,
            inputs=[
                prompt_a,
                image_a,
                prompt_b,
                image_b,
                calibration_max_new_tokens,
                calibration_temperature,
                calibration_seed,
            ],
            outputs=[
                answer_a,
                answer_b,
                delta_table,
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
    parser.add_argument("--check", action="store_true", help="Validate imports, SAE registry and UI without model downloads")
    parser.add_argument("--preload", action="store_true", help="Load Gemma and the SAEs before serving the UI")
    args = parser.parse_args()
    validate_configuration()
    validate_sae_registry()
    demo = build_demo()
    if args.check:
        print(json.dumps({"status": "ok", "model_weights_loaded": False, "sae_release": SAE_RELEASE, "runtime": runtime_metadata()}, indent=2))
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
