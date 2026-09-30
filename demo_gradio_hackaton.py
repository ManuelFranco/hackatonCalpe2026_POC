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
For each layer 9, 17, 22, and 29 and every pair, the app:
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
- per-layer feature tables with expandable Neuronpedia embeds.
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
GEMMA_SAE_RUNS_DIR            default: gemma_sae_common_feature_runs
GRADIO_SERVER_NAME            default: 0.0.0.0
GRADIO_SERVER_PORT            default: 7860
TOP_DIFFS_TO_SHOW             default: 20
MAX_CONTRASTIVE_PAIRS          default: 8 (between 2 and 20)
FEATURE_PRESENCE_EPS          default: 1e-6 (strict abs(delta) > epsilon)
SAE_CHUNK_TOKENS              default: 128
FEATURE_AGGREGATION           default: mean
FEATURE_TOKEN_SCOPE           default: last (also: all, non_image; UI selectable)
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
import html
import importlib.metadata
import json
import math
import os
import sys
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
from sae_dashboard.disclosure_examples import DISCLOSURE_CASES
from sae_dashboard.causal_ui import build_causal_lab
from sae_dashboard.causal_lab import generate_baseline_prefix, prepare_case_inputs
from sae_dashboard.security_reports import ROOT as FIXTURE_ROOT, security_report_cases
from sae_dashboard.calibration_bridge import snapshot_image
from sae_dashboard.cyber_integrity import (FIXTURES as INTEGRITY_FIXTURES, BASE_PROMPT, INTEGRITY_PROMPT,
                             integrity_cases, score_known_fixture, score_markdown)
from sae_dashboard.benchmarks import BENCHMARKS, BenchmarkRunner, save_benchmark_run

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
MODEL_REVISION = os.getenv("GEMMA_MODEL_REVISION")
MODEL_DEVICE = os.getenv("GEMMA_DEVICE", "auto").strip().lower()
MODEL_DTYPE = os.getenv("GEMMA_DTYPE", "auto").strip().lower()
LAYERS = [9, 17, 22, 29]
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2-4b-it-res")
SAE_IDS = {
    9: "layer_9_width_16k_l0_medium",
    17: "layer_17_width_16k_l0_medium",
    22: "layer_22_width_16k_l0_medium",
    29: "layer_29_width_16k_l0_medium",
}
# Source IDs verified against the matching Gemma Scope 2 Neuronpedia listings.
# Keep these aligned with SAE_IDS (16k resid_post, l0_medium).
NEURONPEDIA_MODEL = "gemma-3-4b-it"
NEURONPEDIA_SOURCES = {
    9: "9-gemmascope-2-res-16k",
    17: "17-gemmascope-2-res-16k",
    22: "22-gemmascope-2-res-16k",
    29: "29-gemmascope-2-res-16k",
}
NEURONPEDIA_EMBED_QUERY = (
    "embed=true&embedexplanation=true&embedplots=true&embedsteer=true"
    "&embedactivations=false&embedlink=true&embedtest=true"
)
FEATURE_AGGREGATION = os.getenv("FEATURE_AGGREGATION", "mean").strip().lower()
FEATURE_TOKEN_SCOPE = os.getenv("FEATURE_TOKEN_SCOPE", "last").strip().lower()
CAPTURE_SCOPE = "prompt_with_assistant_prefix_no_generated_tokens"
BOUNDARY_CAPTURE_SCOPE = "prompt_with_base_generated_prefix_before_decision"
STEERING_FRACTION_PER_UNIT = float(os.getenv("STEERING_FRACTION_PER_UNIT", "0.05"))
STEER_LAST_TOKEN_ONLY = os.getenv("STEER_LAST_TOKEN_ONLY", "1") != "0"
TOP_DIFFS_TO_SHOW = int(os.getenv("TOP_DIFFS_TO_SHOW", "20"))
MAX_PAIRS = int(os.getenv("MAX_CONTRASTIVE_PAIRS", "8"))
MIN_PAIRS = 1
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
    if set(NEURONPEDIA_SOURCES) != set(LAYERS) or set(SAE_IDS) != set(LAYERS):
        raise ValueError("Each layer needs a matching SAE and Neuronpedia source.")
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
    feature_acts: torch.Tensor, image_mask: torch.Tensor,
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
    token_scope: str = FEATURE_TOKEN_SCOPE,
    prefix_data: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    prefix_data = prefix_data or {}
    scope = BOUNDARY_CAPTURE_SCOPE if prefix_data.get("prefix_marker") else CAPTURE_SCOPE
    condition_dir.mkdir(parents=True, exist_ok=True)
    image_mask = get_image_mask(input_ids_cpu)
    result: Dict[int, Dict[str, Any]] = {}
    for layer_idx in LAYERS:
        resid = residuals[layer_idx]
        assert_finite(f"layer_{layer_idx}_resid_post_{condition_name}", resid)
        feature_acts = encode_sae_chunked(saes[layer_idx], resid)
        feature_score = aggregate_feature_acts(feature_acts, image_mask, token_scope)
        assert_finite(
            f"layer_{layer_idx}_feature_score_{condition_name}", feature_score
        )
        payload: Dict[str, Any] = {
            "condition": condition_name,
            "capture_scope": scope,
            **prefix_data,
            "model_id": MODEL_ID,
            "layer": layer_idx,
            "sae_release": sae_releases_used[layer_idx],
            "sae_id": SAE_IDS[layer_idx],
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": token_scope,
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
                resid[feature_token_mask(image_mask, token_scope)].float(), dim=-1
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
        "capture_scope": scope,
            **prefix_data,
        "model_id": MODEL_ID,
        "prompt": prompt,
        "image_filename": Path(image_path).name if image_path else None,
        "image_sha256": sha256_file(image_path),
        "num_input_tokens": int(input_ids_cpu.numel()),
        "num_image_tokens": int(image_mask.sum()),
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": token_scope,
        "num_scored_tokens": int(feature_token_mask(image_mask, token_scope).sum()),
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
    # Do not normalize an epsilon-sized cancellation into a full-strength vector.
    masked_mean = torch.where(effective, masked_mean, torch.zeros_like(masked_mean))
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
        stable = sorted(ids, key=lambda j: (bool(result["same_sign_mask"][j]),
                        abs(float(mean[j])) / max(float(stack[:, j].square().mean().sqrt()), 1e-8)), reverse=True)
        ids = list(dict.fromkeys(ids[:max(1, TOP_DIFFS_TO_SHOW // 2)] + stable))
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
                    "yes"
                    if bool(result["same_sign_mask"][j])
                    else (
                        "opposite signs"
                        if bool(result["opposing_sign_mask"][j])
                        else "not applicable"
                    )
                ),
                (
                    "canceled"
                    if bool(result["canceled_mask"][j])
                    else ("included" if included else "EXCLUDED")
                ),
                ", ".join(f"{v:+.5g}" for v in pair_vals),
            ]
        )
    return rows


def save_profile(
    profile_dir: Path,
    pairs_scores: List[Dict[str, Dict[int, Dict[str, Any]]]],
    token_scope: str = FEATURE_TOKEN_SCOPE,
    prefix_marker: str = "",
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
        "feature_token_scope": token_scope,
        "presence_epsilon": FEATURE_PRESENCE_EPS,
        "eligibility": "abs(score(B_i)-score(A_i)) > epsilon for EVERY pair, per layer",
        "sign_policy": "sign is not a presence criterion; report opposite signs; signed mean can cancel",
        "masked_delta": "where(common_mask & abs(mean(pair_deltas)) > epsilon, mean(pair_deltas), 0)",
        "raw_residual_formula": "raw_direction = masked_delta @ W_dec",
        "steering_fraction_per_unit": STEERING_FRACTION_PER_UNIT,
        "steer_last_token_only": STEER_LAST_TOKEN_ONLY,
        "capture_scope": BOUNDARY_CAPTURE_SCOPE if prefix_marker else CAPTURE_SCOPE,
        "prefix_marker": prefix_marker,
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
            "format_version": 3,
            "capture_scope": BOUNDARY_CAPTURE_SCOPE if prefix_marker else CAPTURE_SCOPE,
        "prefix_marker": prefix_marker,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layers": LAYERS,
            "num_pairs": n,
            "aggregation": FEATURE_AGGREGATION,
            "feature_token_scope": token_scope,
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
def load_steering_directions(
    profile_path: str, token_scope: str = FEATURE_TOKEN_SCOPE
) -> Dict[int, torch.Tensor]:
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
    if payload.get("format_version") != 3:
        raise ValueError(
            "Recalibrate: format 3 aligns capture with the assistant prefix."
        )
    expected = {
        "model_revision": MODEL_REVISION,
        "sae_release": SAE_RELEASE,
        "sae_ids": SAE_IDS,
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": token_scope,
        "capture_scope": BOUNDARY_CAPTURE_SCOPE if payload.get("prefix_marker") else CAPTURE_SCOPE,
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
        raise gr.Error(f"Invalid number of pairs ({MIN_PAIRS}-{MAX_PAIRS}).")
    pair_fields = args[1 : 1 + 4 * MAX_PAIRS]
    controls = args[1 + 4 * MAX_PAIRS :]
    max_new_tokens, temperature, seed, make_previews, token_scope = controls[:5]
    prefix_marker = str(controls[5]).strip() if len(controls) > 5 else ""
    if token_scope not in {"last", "all", "non_image"}:
        raise gr.Error("Choose last, all or non_image for feature capture.")
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
                prefix_data = generate_baseline_prefix(sys.modules[__name__], image_path, prompt, prefix_marker)
                case = {"image": image_path, "prompt": prompt, **prefix_data}
                inputs, ids_cpu, _ = prepare_case_inputs(sys.modules[__name__], case)
                residuals = capture_prompt_residuals(inputs)
                pair_scores[label] = save_condition(
                    pair_dir / f"condition_{label}",
                    label,
                    prompt,
                    image_path,
                    ids_cpu,
                    residuals,
                    token_scope=token_scope, prefix_data=prefix_data,
                )
                del inputs, residuals
                answer = "(Preview disabled: activations are captured from the prompt/image only.)"
                if make_previews:
                    generation_inputs, _, generation_length = prepare_case_inputs(sys.modules[__name__], case)
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
                    **prefix_data,
                    "prompt": prompt,
                    "image_filename": Path(image_path).name if image_path else None,
                    "image_path": snapshot_image(image_path, pair_dir / f"condition_{label}"),
                    "image_sha256": sha256_file(image_path),
                    "optional_preview_response": answer if make_previews else None,
                    "capture_scope": BOUNDARY_CAPTURE_SCOPE if prefix_marker else CAPTURE_SCOPE,
                    "num_input_tokens": int(ids_cpu.numel()),
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
        ) = save_profile(profile_dir, pairs_scores, token_scope=token_scope, prefix_marker=prefix_marker)
    experiment_metadata = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "layers": LAYERS,
        "num_pairs": pair_count,
        "pairs": metadata_pairs,
        "prefix_marker": prefix_marker,
        "intervention_schedule": "First continuation step only" if prefix_marker else "Every decoding step",
        "sae_ids": {str(k): v for k, v in SAE_IDS.items()},
        "sae_releases_used": {str(k): v for k, v in sae_releases_used.items()},
        "runtime": runtime_metadata(),
        "aggregation": FEATURE_AGGREGATION,
        "feature_token_scope": token_scope,
        "presence_epsilon": FEATURE_PRESENCE_EPS,
        "capture_scope": BOUNDARY_CAPTURE_SCOPE if prefix_marker else CAPTURE_SCOPE,
        "optional_preview_generation": {
            "enabled": bool(make_previews),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        },
        "formula": "where(intersection(|delta_i|>epsilon) & (|mean_i(delta_i)|>epsilon), mean_i(delta_i), 0) @ W_dec",
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
        "prefix_marker": prefix_marker,
        "num_pairs": pair_count,
        "feature_token_scope": token_scope,
    }
    report = [
        f"**Common profile created:** `{run_id}` · **{pair_count} pair(s)** · "
        f"criterion `|Bᵢ−Aᵢ| > {FEATURE_PRESENCE_EPS:g}` in EVERY pair.",
        "",
        "Only common features reach the decoder; all others are zeroed before applying `W_dec`.",
        "",
        (f"**Decision boundary:** after BASE generates `{prefix_marker}`. Each exact prefix is saved; the decision after it is excluded from calibration."
         if prefix_marker else "Activations captured before the first answer token; no generated response enters calibration."),
        "",
        f"**Tokens: `{token_scope}`.** Captured with the assistant prefix at the same position as the query.",
        "",
    ]
    for layer_idx in LAYERS:
        item = layer_info[layer_idx]
        report.append(
            f"- Layer **{layer_idx}**: **{item['common_features']}** common; "
            f"**{item['excluded_not_common']}** exclusive to some pairs (discarded); "
            f"{item['common_opposing_sign']} common with opposite signs; "
            f"{item['common_canceled_mean']} canceled by averaging; "
            f"**{item['effective_nonzero_features']}** with a final signal; "
            f"‖direction α=1‖₂ = `{item['scaled_direction_l2']:.4f}`."
        )
    report.append("")
    if token_scope != "last" and FEATURE_AGGREGATION == "mean":
        report.append(
            "**Token averaging:** an A/B length difference can change the mean "
            "of otherwise identical activations. Compare with `last` before interpreting features.\n"
        )
    report.append(
        "The table is a preview; the ZIP includes a CSV with **all** features and their deltas for each pair."
    )
    # Four independently rendered results per layer: overview, common,
    # excluded, and top individual pair deltas. Each feature can expand its
    # matching Neuronpedia iframe directly in the corresponding table.
    html_results: List[str] = []
    for layer_idx in LAYERS:
        html_results.extend([
            render_layer_overview(layer_idx, layer_info[layer_idx]),
            render_feature_table(layer_idx, included_rows, included=True, calibration_id=run_id),
            render_feature_table(layer_idx, excluded_rows, included=False),
            render_individual_pair_table(layer_idx, pair_rows),
        ])
    return (*answers, *html_results, bundle, state, "\n".join(report))


def save_query_steered_activations(
    run_dir: Path,
    query_name: str,
    prompt: str,
    image_path: Optional[str],
    input_ids_cpu: torch.Tensor,
    residuals: Dict[int, torch.Tensor],
    strengths: Dict[int, float],
    generation_metadata: Dict[str, Any],
    token_scope: str = FEATURE_TOKEN_SCOPE,
) -> str:
    query_dir = run_dir / "queries" / query_name
    query_dir.mkdir(parents=True, exist_ok=True)
    image_mask = get_image_mask(input_ids_cpu)
    for layer_idx in LAYERS:
        feature_acts = encode_sae_chunked(saes[layer_idx], residuals[layer_idx])
        score = aggregate_feature_acts(feature_acts, image_mask, token_scope)
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
            "feature_token_scope": token_scope,
            "capture_scope": BOUNDARY_CAPTURE_SCOPE if generation_metadata.get("prefix_marker") else CAPTURE_SCOPE,
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
        "feature_token_scope": token_scope,
        "capture_scope": BOUNDARY_CAPTURE_SCOPE if generation_metadata.get("prefix_marker") else CAPTURE_SCOPE,
        "capture_pass": "separate forward over the prompt and optional saved BASE prefix; continuation decision excluded",
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
    strength_22: float,
    strength_29: float,
    session_state: Optional[Dict[str, Any]],
    seed: int = 0,
):
    validate_generation_settings(max_new_tokens, temperature, seed)
    query_prompt = validate_condition_inputs("query", query_image, query_prompt)
    if not session_state:
        raise gr.Error("Build the common profile in section 1 first.")
    profile_path = session_state.get("profile_path")
    if not profile_path or not Path(profile_path).exists():
        raise gr.Error("The common profile cannot be found. Recalibrate it.")
    token_scope = session_state.get("feature_token_scope")
    if token_scope not in {"last", "all", "non_image"}:
        raise gr.Error("The profile needs to be recalibrated with this version.")
    strengths = {
        9: float(strength_9),
        17: float(strength_17),
        22: float(strength_22),
        29: float(strength_29),
    }
    if any(not math.isfinite(v) or abs(v) > 10 for v in strengths.values()):
        raise gr.Error("Steering strengths must be finite and between -10 and 10.")
    run_dir = Path(session_state["run_dir"])
    query_name = time.strftime("query_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    with MODEL_LOCK:
        ensure_models_loaded()
        directions = load_steering_directions(profile_path, token_scope=token_scope)
        prefix_marker = session_state.get("prefix_marker", "")
        prefix_data = generate_baseline_prefix(sys.modules[__name__], query_image, query_prompt, prefix_marker)
        case = {"image": query_image, "prompt": query_prompt, **prefix_data}
        base_inputs, input_ids_cpu, input_len = prepare_case_inputs(sys.modules[__name__], case)
        steered_inputs, _, steered_input_len = prepare_case_inputs(sys.modules[__name__], case)
        capture_inputs, _, _ = prepare_case_inputs(sys.modules[__name__], case)
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
            first_step_only=bool(prefix_marker),
        )
        steered_residuals = capture_prompt_residuals(
            inputs=capture_inputs,
            steering_directions=directions,
            strengths=strengths,
        )
        integrity_result = score_known_fixture(query_image, query_prompt, base_answer, steered_answer)
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
                **prefix_data,
                "intervention_schedule": "First continuation step only" if prefix_marker else "Every decoding step",
                "profile_sha256": sha256_file(profile_path),
                "base_answer": base_answer,
                "steered_answer": steered_answer,
                "integrity_evaluation": integrity_result,
            },
            token_scope=token_scope,
        )
    status = (
        f"**Query:** `{query_name}`\n\n"
        f"**Profile:** `{session_state['run_id']}`\n\n"
        f"**Feature token scope:** `{token_scope}`\n\n"
        f"α9=`{strengths[9]:+.2f}` · α17=`{strengths[17]:+.2f}` · "
        f"α22=`{strengths[22]:+.2f}` · α29=`{strengths[29]:+.2f}`.\n\n"
        f"BASE and STEERED use the same seed (`{int(seed)}`). "
        "This query reuses the common-feature profile without recalibration."
    )
    if prefix_marker:
        status += (f"\n\n**Decision boundary:** `{prefix_marker}`. BASE generates the prefix once; both conditions replay it. "
                   "The common profile is applied only on the first continuation step. Prefix preservation is by construction.")
    status += score_markdown(integrity_result)
    return base_answer, steered_answer, query_zip, status


def run_capability_benchmark(
    benchmark_name: str,
    split: str,
    max_items: int,
    category: str,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    strength_9: float,
    strength_17: float,
    strength_22: float,
    strength_29: float,
    session_state: Optional[Dict[str, Any]],
):
    """Run a paired capability benchmark using the current calibrated profile."""
    validate_generation_settings(max_new_tokens, temperature, seed)
    if benchmark_name not in BENCHMARKS:
        raise gr.Error("Choose a supported benchmark.")
    if not session_state:
        raise gr.Error(
            "No steering profile is loaded. First click 'Build common-feature profile' "
            "in section 1, then run this benchmark."
        )
    profile_path = session_state.get("profile_path")
    token_scope = session_state.get("feature_token_scope")
    if not profile_path or not Path(profile_path).exists():
        raise gr.Error("The calibrated profile cannot be found.")
    if token_scope not in {"last", "all", "non_image"}:
        raise gr.Error("The profile needs to be recalibrated with this version.")
    strengths = {9: float(strength_9), 17: float(strength_17), 22: float(strength_22), 29: float(strength_29)}
    if any(not math.isfinite(value) or abs(value) > 10 for value in strengths.values()):
        raise gr.Error("Steering strengths must be finite and between -10 and 10.")
    category_filter = category if category and category != "All" else None
    with MODEL_LOCK:
        ensure_models_loaded()
        directions = load_steering_directions(profile_path, token_scope=token_scope)
        result = BenchmarkRunner(prepare_inputs, generate_answer).run(
            benchmark=benchmark_name,
            split=split,
            max_items=int(max_items),
            category=category_filter,
            directions=directions,
            strengths=strengths,
            temperature=float(temperature),
            seed=int(seed),
            max_new_tokens=int(max_new_tokens),
        )
        result_path = save_benchmark_run(
            result,
            Path(session_state["run_dir"]),
            {
                "profile_path": str(Path(profile_path).resolve()),
                "profile_sha256": sha256_file(profile_path),
                "token_scope": token_scope,
                "strengths": {str(layer): value for layer, value in strengths.items()},
                "temperature": float(temperature),
                "seed": int(seed),
                "max_new_tokens": int(max_new_tokens),
            },
        )
    summary_data = result.as_dict()
    summary = [[label, summary_data[key]] for label, key in [
        ("Items", "num_items"), ("BASE accuracy", "base_accuracy"),
        ("STEERED accuracy", "steered_accuracy"), ("Accuracy delta", "accuracy_delta"),
        ("Preserved", "preserved"), ("Regressed", "regressed"),
        ("Improved", "improved"), ("Changed answers", "changed"),
        ("Preservation rate", "preservation_rate"), ("Regression rate", "regression_rate"),
        ("Change rate", "change_rate"),
    ]]
    details = [
        [row.get("question_id"), row.get("category", row.get("task", "all")), row["gold"],
         row["base"], row["steered"], row["base_correct"], row["steered_correct"], row["changed"]]
        for row in result.rows
    ]
    status = (
        f"### {BENCHMARKS[benchmark_name].name} completed\n\n"
        f"- Items: **{result.total}**\n"
        f"- BASE accuracy: **{summary_data['base_accuracy']:.3f}**\n"
        f"- STEERED accuracy: **{summary_data['steered_accuracy']:.3f}**\n"
        f"- Accuracy delta: **{summary_data['accuracy_delta']:+.3f}**\n\n"
        f"Saved results: `{result_path}`"
    )
    return status, summary, details, result_path


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
.gradio-container .prose {
    line-height: 1.65;
    overflow-wrap: anywhere;
}
.gradio-container .prose p {
    margin: 0 0 0.85em;
}
.gradio-container .prose ul,
.gradio-container .prose ol {
    padding-inline-start: 1.5em;
    margin: 0.65em 0 1em;
}
.gradio-container .prose li + li {
    margin-top: 0.55em;
}
.gradio-container .prose pre {
    overflow-x: auto;
    white-space: pre;
    line-height: 1.55;
    margin: 0.85em 0;
}
.gradio-container .prose :not(pre) > code {
    white-space: break-spaces;
}
.block.response-markdown {
    padding: 14px 16px;
    border: 1px solid rgba(127,127,127,0.18);
    border-radius: 12px;
    background: rgba(127,127,127,0.035);
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
.block.formula-box {
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid rgba(127,127,127,0.18);
    background: rgba(127,127,127,0.045);
}

/* Layer-by-layer feature tables: same soft theme and section-card styling. */
.layer-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 12px;
    margin: 8px 0 16px;
}
.layer-stat {
    background: rgba(127,127,127,.045);
    border: 1px solid rgba(127,127,127,.18);
    border-radius: 13px;
    padding: 14px 16px;
}
.layer-stat .stat-value { display:block; font-size:1.55rem; font-weight:750; font-variant-numeric:tabular-nums; }
.layer-stat .stat-label { display:block; font-size:.81rem; opacity:.75; line-height:1.35; }
.layer-info { margin:4px 0 18px; font-size:.94rem; opacity:.84; }
.layer-table-caption { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:16px 0 9px; }
.layer-table-caption h4 { font-weight:700; font-size:1.04rem; margin:0; }
.layer-table-caption small { opacity:.7; }
.feature-table { width:100%; overflow-x:auto; border:1px solid rgba(127,127,127,.18); border-radius:14px; }
.feature-grid {
    display:grid;
    grid-template-columns: 90px 94px 110px 110px 110px 122px minmax(190px,1fr) 100px;
    align-items:center;
    gap:12px;
    min-width:1050px;
    padding:10px 14px;
    font-size:.89rem;
}
.feature-grid--pair { grid-template-columns:65px 110px 145px 110px minmax(160px,1fr) 100px; min-width:740px; }
.feature-grid-head { font-size:.75rem; text-transform:uppercase; letter-spacing:.055em; font-weight:750;
    background:rgba(127,127,127,.085); opacity:.78; }
.feature-row { border-top:1px solid rgba(127,127,127,.12); }
.feature-row:nth-child(even) > summary { background:rgba(127,127,127,.025); }
.feature-row > summary { cursor:pointer; list-style:none; }
.feature-row > summary::-webkit-details-marker { display:none; }
.feature-row > summary:hover { background:rgba(90,80,255,.06); }
.feature-row[open] > summary { background:rgba(90,80,255,.075); }
.feature-id { display:inline-block; min-width:54px; text-align:center; padding:5px 9px; border-radius:8px;
    font-weight:780; font-variant-numeric:tabular-nums; }
.feature-id.pos { color:#087c4b; background:rgba(16,185,129,.14); }
.feature-id.neg { color:#c23546; background:rgba(244,63,94,.12); }
.feature-id.zero { color:inherit; background:rgba(127,127,127,.12); }
.signed-value {font-weight:720; font-variant-numeric:tabular-nums; white-space:nowrap;}
.signed-value.pos {color:#098153;}
.signed-value.neg {color:#d13b4d;}
.signed-value.zero {opacity:.6;}
.feature-presence {font-variant-numeric:tabular-nums; padding:4px 8px; border-radius:8px;
    background:rgba(90,80,255,.11); display:inline-block; font-weight:650; white-space:nowrap;}
.feature-presence.full { color:#087c4b; background:rgba(16,185,129,.12); }
.feature-chip { padding:4px 8px; border-radius:99px; display:inline-block; font-size:.79rem;
    background:rgba(127,127,127,.1); }
.feature-chip.ambiguous {background:rgba(245,158,11,.15);color:#986308;}
.feature-pairs {font-size:.82rem; white-space:normal; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; opacity:.86;}
.np-trigger {justify-self:end; display:inline-flex; align-items:center; gap:4px; font-size:.77rem; font-weight:740;
    border:1px solid rgba(90,80,255,.28); background:rgba(90,80,255,.075);
    padding:6px 10px; border-radius:8px; color:inherit; white-space:nowrap; }
.np-trigger:before { content:'+'; font-size:1rem; line-height:0; padding-right:3px; }
.feature-row[open] .np-trigger:before { content:'−'; }
.np-panel {padding:14px 18px 19px; border-top:1px solid rgba(127,127,127,.12);
    background:rgba(127,127,127,.025);}
.np-panel-title {margin-bottom:11px; font-weight:650; font-size:.88rem;}
.np-iframe {display:block; height:300px; width:540px; max-width:100%; border:1px solid rgba(127,127,127,.20);
    border-radius:10px; background:white;}
.np-external {display:inline-block; margin-top:9px; font-size:.82rem; text-decoration:underline;}
.feature-empty {padding:25px 18px; opacity:.75; font-size:.91rem;}
@media (prefers-color-scheme:dark) {
 .feature-id.pos,.signed-value.pos,.feature-presence.full {color:#5ce0aa;}
 .feature-id.neg,.signed-value.neg {color:#ff8590;}
 .feature-chip.ambiguous {color:#ffc875;}
}
"""
TUTORIAL_MD = r"""
### Quick guide

1. **Add A/B pairs.** One pair is shown by default; you can add up to the configured maximum.
    Each A and B accepts text, an image, or both. Each active pair must be complete.

2. **Choose the token scope and click “Build common-feature profile”.** `last` reads features
    at the final input position, after reading all content and the assistant prefix. It does not
    truncate the prompt. `all` averages all text positions and can confuse length changes with
    activation changes. A forward pass runs for each prompt/image (and the BASE prefix if you
    choose a decision boundary marker). Each layer produces SAE activations and a vector
    `delta_i = score(B_i) − score(A_i)` for each pair.

3. **Strict intersection:** a feature is present in a pair when
    `abs(delta_i) > FEATURE_PRESENCE_EPS`. Only features present in **every** pair
    enter the profile. Their **signed** deltas are averaged. Opposite signs are flagged;
    a canceled average contributes no steering.

4. **Inspect the per-layer tables.** Features with a positive mean B−A are green and
    negative ones are red. Each row has EXPAND and a Neuronpedia iframe. Counts, common
    features, excluded features, and per-pair differences are shown. The ZIP keeps every
    feature in the CSV and untruncated `.pt` tensors.

5. **Query the model with another image/prompt**, compare BASE with STEERED, and adjust
    the per-layer sliders. `0` means no steering. Alpha's sign changes the activation,
    but does not guarantee a particular behavior.

Full previews are not used to build the profile. If you choose a **decision boundary marker**,
a BASE prefix is generated and saved for each condition: its tokens are included in the capture
input, but the decision after the marker is excluded. The query and section 3 preserve that
intervention point.
"""



# ============================================================
# PRESENTATION: PER-LAYER TABLES + ON-DEMAND NEURONPEDIA EXPAND
# ============================================================
def neuronpedia_url(layer_idx: int, feature_id: int) -> str:
    """Build a trusted URL from configured SAE IDs and integer feature indices."""
    if layer_idx not in NEURONPEDIA_SOURCES:
        raise ValueError(f"Unconfigured Neuronpedia layer: {layer_idx}")
    feature_id = int(feature_id)
    if feature_id < 0:
        raise ValueError("Feature ID must be non-negative")
    return (
        f"https://www.neuronpedia.org/{NEURONPEDIA_MODEL}/"
        f"{NEURONPEDIA_SOURCES[layer_idx]}/{feature_id}?{NEURONPEDIA_EMBED_QUERY}"
    )


def _signed_class(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else "zero"


def _signed_html(value: float) -> str:
    return f'<span class="signed-value {_signed_class(value)}">{value:+.6g}</span>'


def _np_expanded_panel(layer_idx: int, feature_id: int) -> str:
    """Only constant text and validated integer feature IDs enter this HTML."""
    url = html.escape(neuronpedia_url(layer_idx, feature_id), quote=True)
    title = html.escape(f"Neuronpedia · layer {layer_idx}, feature {feature_id}", quote=True)
    return (
        '<div class="np-panel">'
        f'<div class="np-panel-title">Layer {layer_idx} · Feature #{feature_id} · Neuronpedia</div>'
        f'<iframe class="np-iframe" src="{url}" title="{title}" loading="lazy" '
        'referrerpolicy="strict-origin-when-cross-origin" '
        'style="height:300px;width:540px;max-width:100%;" '
        'allow="clipboard-write"></iframe>'
        f'<a class="np-external" href="{url}" target="_blank" rel="noopener noreferrer">'
        'Abrir en Neuronpedia ↗</a>'
        '</div>'
    )


def _feature_header(title: str, description: str) -> str:
    return (
        '<div class="layer-table-caption">'
        f'<h4>{html.escape(title)}</h4><small>{html.escape(description)}</small>'
        '</div>'
    )


def render_empty_layer_overview(layer_idx: int) -> str:
    return (
        '<div class="feature-empty">'
        f'Layer {layer_idx}: calibrate at least one A/B pair to see features, '
        'signs, and Neuronpedia links.'
        '</div>'
    )


def render_layer_overview(layer_idx: int, info: Dict[str, Any]) -> str:
    cards = (
        ("A/B pairs", info["num_pairs"]),
        ("Features present in any pair", info["present_in_at_least_one"]),
        ("Common in every pair", info["common_features"]),
        ("Excluded by intersection", info["excluded_not_common"]),
        ("Common with opposite signs", info["common_opposing_sign"]),
        ("Canceled after averaging", info["common_canceled_mean"]),
        ("With a final signal", info["effective_nonzero_features"]),
    )
    html_cards = ''.join(
        '<div class="layer-stat">'
        f'<span class="stat-value">{int(number):,}</span>'
        f'<span class="stat-label">{html.escape(label)}</span></div>'
        for label, number in cards
    )
    return (
        f'<div class="layer-stat-grid">{html_cards}</div>'
        '<div class="layer-info">'
        f'Layer {layer_idx} · SAE 16k · {int(info["total_features"]):,} total features · '
        f'‖scaled direction, α=1‖₂ = <strong>{info["scaled_direction_l2"]:.5g}</strong>. '
        'Presence is |Bᵢ−Aᵢ| &gt; ε in <strong>every</strong> pair. '
        'Green = positive mean B−A; red = negative mean. '
        'Presence does not require matching signs.</div>'
    )


def render_feature_table(
    layer_idx: int, rows: List[List[Any]], included: bool, calibration_id: Optional[str] = None,
) -> str:
    layer_rows = [r for r in rows if int(r[0]) == layer_idx]
    title = "Common features · included" if included else "Non-common features · excluded"
    description = (
        "Preview includes large mean differences and consistent paired contrasts; canceled means do not contribute."
        if included else
        "Ordered by peak |B−A|; never included in the final direction."
    )
    heading = _feature_header(title, description)
    head = (
        '<div class="feature-grid feature-grid-head">'
        '<span>Feature</span><span>Presence</span><span>Mean B−A</span>'
        '<span>Δ steering</span><span>Signs</span><span>Status</span>'
        '<span>Pair Δ values</span><span>Neuronpedia</span></div>'
    )
    if not layer_rows:
        return heading + '<div class="feature-table"><div class="feature-empty">No features in this category.</div></div>'
    parts = [heading, '<div class="feature-table">', head]
    for row in layer_rows:
        _, fid, count, total, avg, steering, signs, status, pair_deltas = row
        fid, count, total = int(fid), int(count), int(total)
        mean = float(avg)
        value = float(steering)
        feature_class = _signed_class(mean)
        presence_class = "full" if count == total else ""
        status_class = "ambiguous" if "opposite" in str(signs) or "canceled" in str(status) else ""
        pair_label = html.escape(str(pair_deltas))
        transfer = (
            '<div class="feature-transfer">'
            f'<button type="button" data-causal-feature="{fid}" data-layer="{layer_idx}" '
            f'data-calibration="{html.escape(calibration_id, quote=True)}" '
            f'aria-label="Test layer {layer_idx} feature {fid} in block 3">'
            'Test this feature in block 3 →</button>'
            '<span>Use this calibration and the current query from section 2.</span></div>'
            if included and calibration_id and status != "canceled" else ""
        )
        parts.extend([
            '<details class="feature-row">',
            '<summary class="feature-grid">',
            f'<span><span class="feature-id {feature_class}">#{fid}</span></span>',
            f'<span><span class="feature-presence {presence_class}">{count}/{total}</span></span>',
            f'<span>{_signed_html(mean)}</span>',
            f'<span>{_signed_html(value)}</span>',
            f'<span><span class="feature-chip {status_class}">{html.escape(str(signs))}</span></span>',
            f'<span><span class="feature-chip {status_class}">{html.escape(str(status))}</span></span>',
            f'<span class="feature-pairs">{pair_label}</span>',
            '<span class="np-trigger">EXPAND</span>',
            '</summary>',
            transfer,
            _np_expanded_panel(layer_idx, fid),
            '</details>',
        ])
    parts.append('</div>')
    return ''.join(parts)


def render_individual_pair_table(layer_idx: int, rows: List[List[Any]]) -> str:
    layer_rows = [r for r in rows if int(r[1]) == layer_idx]
    heading = _feature_header(
        "Largest differences for each pair",
        "Individual B−A values; this does not imply that the feature is included in the intersection.",
    )
    head = (
        '<div class="feature-grid feature-grid--pair feature-grid-head">'
        '<span>Pair</span><span>Feature</span><span>Δ B−A</span>'
        '<span>|Δ|</span><span>Interpretation</span><span>Neuronpedia</span></div>'
    )
    if not layer_rows:
        return heading + '<div class="feature-table"><div class="feature-empty">Sin diferencias individuales.</div></div>'
    parts = [heading, '<div class="feature-table">', head]
    for pair_i, _, rank, fid, delta, abs_delta, interpretation in layer_rows:
        fid, pair_i = int(fid), int(pair_i)
        delta = float(delta)
        parts.extend([
            '<details class="feature-row">',
            '<summary class="feature-grid feature-grid--pair">',
            f'<span><span class="feature-chip">#{pair_i} · top {int(rank)}</span></span>',
            f'<span><span class="feature-id {_signed_class(delta)}">#{fid}</span></span>',
            f'<span>{_signed_html(delta)}</span>',
            f'<span class="signed-value">{float(abs_delta):.6g}</span>',
            f'<span class="feature-pairs">{html.escape(str(interpretation))}</span>',
            '<span class="np-trigger">EXPAND</span>',
            '</summary>',
            _np_expanded_panel(layer_idx, fid),
            '</details>',
        ])
    parts.append('</div>')
    return ''.join(parts)


def change_pair_count(count: int, change: int):
    count = max(MIN_PAIRS, min(MAX_PAIRS, int(count) + change))
    return (
        count,
        *[gr.update(visible=i < count) for i in range(MAX_PAIRS)],
        f"**Active pairs: {count}/{MAX_PAIRS}.** Hidden pairs are not calibrated.",
    )


def response_markdown(title: str, min_height: int) -> gr.Markdown:
    """Render generated answers as sanitized Markdown, preserving line breaks."""
    gr.Markdown(f"#### {title}")
    return gr.Markdown(
        value="",
        line_breaks=True,
        sanitize_html=True,
        buttons=["copy"],
        min_height=min_height,
        max_height=600,
        elem_classes=["response-markdown"],
    )


def load_disclosure_preset():
    """Replace inputs and invalidate the old profile without running the model."""
    count = min(3, MAX_PAIRS)
    fields = []
    for i in range(MAX_PAIRS):
        if i < count:
            fields.extend([
                DISCLOSURE_CASES[f"train{i+1:02d}_A"]["prompt"], None,
                DISCLOSURE_CASES[f"train{i+1:02d}_B"]["prompt"], None,
            ])
        else:
            fields.extend(["", None, "", None])
    empty_layers = []
    for layer in LAYERS:
        empty_layers.extend([render_empty_layer_overview(layer), "", "", ""])
    return (
        *change_pair_count(count, 0), *fields,
        "last", True, 96, 0.0, 0,
        *["" for _ in range(2 * MAX_PAIRS)], *empty_layers,
        *load_disclosure_query("valid04_A"), "valid04_A",
        96, 0.0, 0, 0.0, 0.0, 0.0, 0.0,
        None, None, "Short English examples loaded. Build a new profile before querying.",
    )


def load_disclosure_query(case_id: str):
    if case_id not in DISCLOSURE_CASES:
        raise gr.Error("Unknown disclosure test case.")
    return None, DISCLOSURE_CASES[case_id]["prompt"], "", "", None, ""


def load_image_report_query(case_id: str):
    case = next((c for c in security_report_cases() if c["id"] == case_id), None)
    if case is None:
        raise gr.Error("Unknown image report case.")
    return str(FIXTURE_ROOT / case["image"]), case["prompt"], "", "", None, ""


def load_image_report_preset():
    """Three clean/attack screenshot pairs; identical English report task."""
    count = min(3, MAX_PAIRS)
    cases = {c["id"]: c for c in security_report_cases()}
    fields = []
    for i in range(MAX_PAIRS):
        for side in ("A", "B"):
            case = cases[f"web_train{i+1:02d}_{side}_report"] if i < count else None
            fields.extend([case["prompt"], str(FIXTURE_ROOT / case["image"])] if case else ["", None])
    empty_layers = []
    for layer in LAYERS:
        empty_layers.extend([render_empty_layer_overview(layer), "", "", ""])
    return (
        *change_pair_count(count, 0), *fields,
        "last", True, 320, 0.0, 0,
        *["" for _ in range(2 * MAX_PAIRS)], *empty_layers,
        *load_image_report_query("web_valid04_A_report"), "valid04_A",
        320, 0.0, 0, 0.0, 0.0, 0.0, 0.0,
        None, None, "English image report examples loaded: A = Clean, B = Attack. Build a new profile before querying.",
    )



def load_integrity_query(case_id: str, task_instruction: str = "Source-constrained"):
    if task_instruction not in {"Source-constrained", "Ordinary baseline"}:
        raise gr.Error("Unknown task instruction.")
    case = next((c for c in integrity_cases() if c["id"] == case_id), None)
    if case is None:
        raise gr.Error("Unknown incident intake fixture.")
    prompt = INTEGRITY_PROMPT if task_instruction == "Source-constrained" else BASE_PROMPT
    return str(INTEGRITY_FIXTURES / case["image"]), prompt, "", "", None, ""


def load_integrity_preset():
    cases = {c["id"]: c for c in integrity_cases()}
    count = min(3, MAX_PAIRS)
    fields = []
    for i in range(MAX_PAIRS):
        for side in ("A", "B"):
            case = cases[f"train{i+1:02d}_{side}"] if i < count else None
            fields.extend([INTEGRITY_PROMPT, str(INTEGRITY_FIXTURES / case["image"])] if case else ["", None])
    empty_layers = []
    for layer in LAYERS:
        empty_layers.extend([render_empty_layer_overview(layer), "", "", ""])
    return (
        *change_pair_count(count, 0), *fields,
        "last", True, 128, 0.0, 0,
        *["" for _ in range(2 * MAX_PAIRS)], *empty_layers,
        *load_integrity_query("valid04_A"), "valid04_A",
        128, 0.0, 0, 0.0, 0.0, 0.0, 0.0,
        None, None, "Incident intake loaded: A = ordinary comment, B = injection attempt. Both must preserve the same record fields. Build a new profile.",
    )


def build_demo() -> gr.Blocks:
    """Construct the interface without downloading model weights."""
    with gr.Blocks(
        title="Gemma 3 · Common-Feature SAE Steering",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            """# Gemma 3 · Common-Feature SAE Steering

Create a reusable B − A profile from **multiple multimodal pairs**.

The common profile uses features present in **all** B−A vectors.

To measure an individual feature, open the [Single-feature causal lab](#causal-lab):
decision curves, ablation, inverse questions, and random controls.""",
            elem_id="hero",
        )
        with gr.Accordion("How to use this app", open=True):
            gr.Markdown(TUTORIAL_MD)
        with gr.Accordion("Method and steering formula", open=False):
            gr.Markdown(f"""
### Profile configuration

- **Layers:** `{LAYERS}`
- **Aggregation:** `{FEATURE_AGGREGATION}`
- **Default token scope:** `{FEATURE_TOKEN_SCOPE}`; selected during calibration.
- **Epsilon:** `{FEATURE_PRESENCE_EPS:g}`
- **Fraction/alpha:** `{STEERING_FRACTION_PER_UNIT}`
- **Steer last token only:** `{STEER_LAST_TOKEN_ONLY}`

### Direction calculation and steering application

For each pair `i` and layer `l`:

```text
d_i = score_SAE(B_i) − score_SAE(A_i)
common[j] = AND_i(abs(d_i[j]) > epsilon)
d_common = where(common, mean_i(d_i), 0)
d_common[abs(d_common) <= epsilon] = 0
v_raw = d_common @ W_dec
v = normalize(v_raw) × reference_residual_norm × STEERING_FRACTION_PER_UNIT
h_last' = h_last + alpha × v
```

The presence criterion does not require matching signs: sign reversals are shown separately.

Activations are extracted **before generating the response**, including the assistant
prefix to align calibration and query. `last` measures a position that has read all
content; `all` and `non_image` aggregate multiple positions.
""", elem_classes=["formula-box"])
        session_state = gr.State(value=None)
        pair_count = gr.State(value=MIN_PAIRS)
        gr.Markdown("## 1. Build the common B − A steering profile")
        gr.Markdown(
            "Define at least one independent A/B pair. Each condition must contain text, an image, or both.",
            elem_classes=["small-note"],
        )
        with gr.Row():
            add_pair = gr.Button("＋ Add A/B pair", variant="secondary")
            remove_pair = gr.Button("− Remove last pair", variant="secondary")
            disclosure_preset = gr.Button("Load short disclosure examples (English)")
            image_report_preset = gr.Button("Load image security report examples (English)")
            integrity_preset = gr.Button("Load cyber incident intake (English)")
        gr.Markdown(
            "The disclosure examples ask whether stdout contains a fictional secret. "
            "Expected: **A = NO**, **B = YES**. Start with zero steering; "
            "use the separate validation and test cases to check transfer.",
            elem_classes=["small-note"],
        )
        pair_count_label = gr.Markdown(
            f"**Active pairs: {MIN_PAIRS}/{MAX_PAIRS}.** Hidden pairs are not calibrated."
        )
        pair_groups = []
        flat_pair_inputs = []
        flat_pair_answers = []
        for pair_i in range(MAX_PAIRS):
            with gr.Group(visible=pair_i < MIN_PAIRS) as pair_group:
                gr.Markdown(f"### Pair {pair_i+1} · B{pair_i+1} − A{pair_i+1}")
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
                        answer_a = response_markdown(
                            f"Gemma response for A{pair_i+1} (optional preview)",
                            min_height=120,
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
                        answer_b = response_markdown(
                            f"Gemma response for B{pair_i+1} (optional preview)",
                            min_height=120,
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
        calibration_prefix_marker = gr.Textbox(label="Decision boundary marker · optional",
            info="Blank: capture before the first answer token. Otherwise BASE generates a prefix through this marker "
                 "(for example Action:). Blocks 1–3 reuse that boundary and intervene on the first continuation step.")
        calibration_token_scope = gr.Dropdown(
            choices=[("Last input token (recommended)", "last"),
                     ("All input tokens", "all"),
                     ("Non-image input tokens", "non_image")],
            value=FEATURE_TOKEN_SCOPE,
            label="Feature token scope",
            info="Last reads the full input and measures its final position before the response. "
                 "Changes apply to the next calibration; queries keep their profile's setting.",
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
            "Independent tables are shown for each layer. A feature ID is green when its mean Δ B−A "
            "is positive and red when it is negative. Click EXPAND to inspect the feature in Neuronpedia. "
            "In an included row, click “Test this feature in section 3” to measure it individually "
            "using this calibration and the query from section 2.",
            elem_classes=["small-note"],
        )
        layer_html_outputs: List[gr.HTML] = []
        common_feature_tables = []
        with gr.Tabs():
            for layer_idx in LAYERS:
                with gr.Tab(f"Layer {layer_idx}"):
                    gr.Markdown(f"#### Layer {layer_idx} · Gemma Scope 2 · resid_post · 16k")
                    overview_html = gr.HTML(value=render_empty_layer_overview(layer_idx), show_label=False)
                    common_html = gr.HTML(value="", show_label=False, js_on_load="""
element.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-causal-feature]');
    if (!button || !element.contains(button)) return;
    event.preventDefault();
    event.stopPropagation();
    trigger('click', {layer: Number(button.dataset.layer),
                      feature_id: Number(button.dataset.causalFeature),
                      calibration_id: button.dataset.calibration});
});
""", css_template="""
.feature-transfer { display:flex; flex-wrap:wrap; align-items:center; gap:12px; padding:16px; }
.feature-transfer button { cursor:pointer; background:#0e7490; color:white; border:0;
    border-radius:8px; padding:10px 16px; font-weight:600; }
.feature-transfer button:focus-visible { outline:3px solid #8b5cf6; outline-offset:3px; }
.feature-transfer span { font-size:13px; color:var(--body-text-color-subdued); }
""")
                    common_feature_tables.append(common_html)
                    discarded_html = gr.HTML(value="", show_label=False)
                    with gr.Accordion("Largest individual B−A differences by pair", open=False):
                        individual_html = gr.HTML(value="", show_label=False)
                    layer_html_outputs.extend([
                        overview_html, common_html, discarded_html, individual_html,
                    ])
        gr.Markdown("## 2. Query the calibrated steered model")
        gr.Markdown(
            "Query with a new prompt and/or image. The filtered profile is reused without recalibration.",
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
                with gr.Accordion("Short disclosure validation and test cases", open=False):
                    disclosure_case = gr.Dropdown(
                        choices=[(f"{key} · {case['title']}", key)
                                 for key, case in DISCLOSURE_CASES.items()
                                 if case["split"] != "train"],
                        value="valid04_A", label="English C test case",
                    )
                    disclosure_query = gr.Button("Load test case")
                with gr.Accordion("Image security report cases (English)", open=False):
                    gr.Markdown("Reports identify the audience, quote evidence, explain the mechanism and give a verdict and action. "
                                "Clean means no visible AI-directed injection, not a security audit of the entire website.")
                    image_report_case = gr.Dropdown(
                        choices=[(f"{c['id']} · {c['split']}", c["id"])
                                 for c in security_report_cases() if c["split"] != "train"],
                        value="web_valid04_A_report", label="Image report case")
                    image_report_query = gr.Button("Load image report case")
                with gr.Accordion("Cyber incident intake cases (English)", open=False):
                    gr.Markdown("Extract the full incident title and the authoritative Owner row while ignoring external comments. "
                                "Query results check exact fields, output format and the attacker objective separately. "
                                "Positive B−A steering is not assumed to be defensive. Start at zero.")
                    integrity_case = gr.Dropdown(
                        choices=[(f"{c['id']} · {c['split']} · {c['attack'] or 'ordinary / control'}", c["id"])
                                 for c in integrity_cases()],
                        value="valid04_A", label="Incident intake case")
                    integrity_instruction = gr.Dropdown(
                        choices=["Source-constrained", "Ordinary baseline"], value="Source-constrained",
                        label="Task instruction · compare at zero steering")
                    integrity_query = gr.Button("Load incident intake case")
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
                strength_22 = gr.Slider(
                    -10, 10, value=0, step=0.25, label="Layer 22 · alpha"
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
                query_base_answer = response_markdown(
                    "BASE response", min_height=260
                )
                query_steered_answer = response_markdown(
                    "STEERED response", min_height=260
                )
        gr.Markdown("## 3. Capability / Coherence Benchmark")
        gr.Markdown(
            "Run the same items with BASE and STEERED across multiple Hugging Face datasets. "
            "Results measure accuracy, preservation, regressions, improvements, and changed answers.",
            elem_classes=["small-note"],
        )
        with gr.Row():
            benchmark_name = gr.Dropdown(
                choices=[(adapter.name, key) for key, adapter in BENCHMARKS.items()],
                value="mmlu_pro", label="Benchmark",
            )
            benchmark_split = gr.Dropdown(
                choices=["test", "validation", "train"], value="test", label="Split"
            )
            benchmark_items = gr.Slider(
                1, 500, value=100, step=1, label="Items"
            )
        with gr.Row():
            benchmark_category = gr.Textbox(
                value="All", label="Category or task filter", placeholder="All"
            )
            benchmark_temperature = gr.Slider(
                0, 2, value=0, step=0.05, label="Temperature · 0 = greedy"
            )
            benchmark_max_tokens = gr.Slider(
                1, 128, value=32, step=1, label="Max new tokens"
            )
            benchmark_seed = gr.Number(value=0, precision=0, label="Seed")
        benchmark_run_btn = gr.Button("Run capability benchmark", variant="primary")
        benchmark_status = gr.Markdown(
            "Build a common-feature profile in section 1 before running a benchmark. "
            "The benchmark compares BASE and STEERED using that profile."
        )
        benchmark_summary = gr.Dataframe(
            headers=["Metric", "Value"], datatype=["str", "number"], interactive=False,
        )
        benchmark_details = gr.Dataframe(
            headers=["Question ID", "Category / task", "Gold", "BASE", "STEERED",
                     "BASE correct", "STEERED correct", "Changed"], interactive=False,
        )
        benchmark_archive = gr.File(label="Download benchmark results JSON")
        with gr.Accordion("Saved artifacts and experiment structure", open=False):
            gr.Markdown("""
```text
run_id/
├── pairs/
│   ├── pair_01/
│   │   ├── condition_A/{layer_9.pt,layer_17.pt,layer_22.pt,layer_29.pt,metadata.json}
│   │   └── condition_B/{layer_9.pt,layer_17.pt,layer_22.pt,layer_29.pt,metadata.json}
│   ├── pair_02/...
│   └── ...
├── common_feature_profile_B_minus_A/
│   ├── layer_9_common_profile.pt
│   ├── layer_17_common_profile.pt
│   ├── layer_22_common_profile.pt
│   ├── layer_29_common_profile.pt
│   ├── features_all_layers.csv     # every feature + every pair delta
│   ├── steering_profile.pt
│   └── profile_summary.json
├── queries/
├── benchmarks/
│   └── {benchmark}_{timestamp}/results.json
└── experiment.json
```

Los `.pt` individuales retienen activaciones SAE token por token y scores FP32.

The profile stores all deltas, the intersection mask, and the filtered direction.

Query activations are captured in a **separate forward pass over the prompt**, never over the response.

Benchmark results preserve the exact profile, strengths, seed, and generation settings used for each paired run.
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
                calibration_token_scope,
                calibration_prefix_marker,
            ],
            outputs=[
                *flat_pair_answers,
                *layer_html_outputs,
                calibration_bundle,
                session_state,
                calibration_status,
            ],
        )
        preset_outputs = [
                pair_count, *pair_groups, pair_count_label, *flat_pair_inputs,
                calibration_token_scope, calibration_previews,
                calibration_max_new_tokens, calibration_temperature, calibration_seed,
                *flat_pair_answers, *layer_html_outputs,
                query_image, query_prompt, query_base_answer, query_steered_answer,
                query_archive, query_status, disclosure_case,
                query_max_new_tokens, query_temperature, query_seed,
                strength_9, strength_17, strength_22, strength_29,
                calibration_bundle, session_state, calibration_status,
            ]
        disclosure_preset.click(fn=load_disclosure_preset, outputs=preset_outputs)
        image_report_preset.click(fn=load_image_report_preset, outputs=preset_outputs)
        integrity_preset.click(fn=load_integrity_preset, outputs=preset_outputs)
        for preset in (disclosure_preset, image_report_preset, integrity_preset):
            preset.click(lambda: "", outputs=[calibration_prefix_marker], api_name=False, queue=False)
        integrity_query.click(fn=load_integrity_query, inputs=[integrity_case, integrity_instruction],
                              outputs=[query_image, query_prompt, query_base_answer,
                                       query_steered_answer, query_archive, query_status])
        image_report_query.click(fn=load_image_report_query, inputs=[image_report_case],
                                 outputs=[query_image, query_prompt, query_base_answer,
                                          query_steered_answer, query_archive, query_status])
        disclosure_query.click(
            fn=load_disclosure_query, inputs=[disclosure_case],
            outputs=[query_image, query_prompt, query_base_answer,
                     query_steered_answer, query_archive, query_status],
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
                strength_22,
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
        benchmark_run_btn.click(
            fn=run_capability_benchmark,
            inputs=[
                benchmark_name, benchmark_split, benchmark_items, benchmark_category,
                benchmark_temperature, benchmark_max_tokens, benchmark_seed,
                strength_9, strength_17, strength_22, strength_29, session_state,
            ],
            outputs=[benchmark_status, benchmark_summary, benchmark_details, benchmark_archive],
        )
        build_causal_lab(sys.modules[__name__], session_state, query_image, query_prompt, common_feature_tables)
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
