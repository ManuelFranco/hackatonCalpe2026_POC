"""Session orchestration. Research algorithms are injected through registries."""

from dataclasses import asdict
import math
import uuid
import torch
from research.vector_builders import BUILDERS, LayerProfile
from research.benchmarks import (
    BENCHMARKS,
    BenchmarkRequest,
    prepare_benchmark,
    evaluate_base,
    evaluate_steered,
    summarize,
)
from . import base_response_cache
from . import model_runtime as runtime
from .artifact_store import export_model, record_event
from .manifests import load_manifests
from .session import Session, Settings


def set_manifests(session: Session, paths: list[str]):
    manifests = load_manifests(paths)
    with session.lock:
        session.manifests = manifests
        session.invalidate_profile()
    return [[m.name, len(m.pairs), m.fingerprint[:12]] for m in manifests]


def capture_score(residual, image_mask, sae, settings):
    selected = residual[runtime.feature_token_mask(image_mask, settings.token_scope)]
    if not selected.shape[0]:
        raise ValueError("The token selection contains no tokens.")
    score = None
    for start in range(0, len(selected), runtime.SAE_CHUNK_TOKENS):
        acts = runtime.encode_sae_chunked(
            sae, selected[start : start + runtime.SAE_CHUNK_TOKENS]
        )
        reduced = acts.sum(0) if settings.aggregation == "mean" else acts.amax(0)
        score = (
            reduced
            if score is None
            else (
                score + reduced
                if settings.aggregation == "mean"
                else torch.maximum(score, reduced)
            )
        )
    if settings.aggregation == "mean":
        score /= len(selected)
    norm = float(selected.float().norm(dim=-1).mean())
    return score, norm


def build_profile(session: Session, settings: Settings, progress=None):
    settings.validate()
    with session.lock:
        if not session.manifests:
            raise ValueError("Load at least one manifest in section 1.")
        scores = {layer: {"A": [], "B": [], "norms": []} for layer in runtime.LAYERS}
        pairs = [pair for m in session.manifests for pair in m.pairs]
        for index, pair in enumerate(pairs):
            # Release the shared model between pairs so other sessions can run.
            with runtime.MODEL_LOCK:
                runtime.ensure_models_loaded()
                for label, condition in (("A", pair.a), ("B", pair.b)):
                    image = condition.load_image()
                    inputs, ids, _ = runtime.prepare_inputs(image, condition.text)
                    residuals = runtime.capture_prompt_residuals(inputs)
                    mask = runtime.get_image_mask(ids)
                    for layer in runtime.LAYERS:
                        score, norm = capture_score(
                            residuals[layer], mask, runtime.saes[layer], settings
                        )
                        scores[layer][label].append(score)
                        scores[layer]["norms"].append(norm)
                    del inputs, residuals
            if progress:
                progress(
                    (index + 1) / len(pairs),
                    desc=f"Capturing pair {index + 1}/{len(pairs)}",
                )
        profile = {
            layer: LayerProfile(
                torch.stack(s["A"]),
                torch.stack(s["B"]),
                sum(s["norms"]) / len(s["norms"]),
            )
            for layer, s in scores.items()
        }
        session.invalidate_profile()
        session.profile, session.capture_key = profile, settings.capture_key
        session.profile_id = uuid.uuid4().hex
        record_event(
            session,
            "profile",
            {
                "settings": asdict(settings),
                "model_id": runtime.MODEL_ID,
                "runtime": runtime.runtime_metadata(),
                "manifests": [asdict(m) for m in session.manifests],
                "layers": {
                    str(layer): {
                        "pairs": len(p.a),
                        "features": p.a.shape[1],
                        "reference_norm": p.reference_norm,
                    }
                    for layer, p in profile.items()
                },
            },
        )
        return [
            [layer, len(p.a), p.a.shape[1], round(p.reference_norm, 4)]
            for layer, p in profile.items()
        ]


def require_profile(session, settings, vectors=False):
    settings.validate()
    if not session.profile or session.capture_key != settings.capture_key:
        raise ValueError(
            "Build the common profile with the current token scope and aggregation first."
        )
    if vectors and not session.vectors:
        raise ValueError("Create steering vectors in section 3 first.")


def create_vectors(session: Session, settings: Settings, method: str):
    with session.lock:
        require_profile(session, settings)
        if method not in BUILDERS:
            raise ValueError("Select a registered vector method.")
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded()
            vectors = {
                layer: BUILDERS[method].build(
                    profile,
                    runtime.saes[layer].W_dec,
                    runtime.STEERING_FRACTION_PER_UNIT,
                )
                for layer, profile in session.profile.items()
            }
        # Enforce the contract for every future research strategy as well.
        for layer, vector in vectors.items():
            if vector.direction.shape != (
                runtime.saes[layer].W_dec.shape[1],
            ) or not bool(torch.isfinite(vector.direction).all()):
                raise ValueError(
                    f"Vector method returned an invalid direction for layer {layer}."
                )
            if vector.feature_delta.shape != session.profile[layer].a.shape[
                1:
            ] or not bool(torch.isfinite(vector.feature_delta).all()):
                raise ValueError(
                    f"Vector method returned an invalid feature delta for layer {layer}."
                )
        session.vectors, session.vector_id = vectors, uuid.uuid4().hex
        record_event(
            session,
            "vectors",
            {
                "method": method,
                "layers": {str(k): v.metadata for k, v in vectors.items()},
            },
        )
        return [
            [
                layer,
                int(torch.count_nonzero(v.feature_delta)),
                round(float(v.direction.norm()), 5),
            ]
            for layer, v in vectors.items()
        ]


def validate_strengths(strengths):
    if set(strengths) != set(runtime.LAYERS) or any(
        not math.isfinite(v) or not -10 <= v <= 10 for v in strengths.values()
    ):
        raise ValueError("Each layer strength must be finite and between -10 and 10.")


def generate_pair(session, settings, strengths, image, prompt):
    with runtime.MODEL_LOCK:
        runtime.ensure_models_loaded(with_saes=False)
        inputs, _, length = runtime.prepare_inputs(image, prompt)
        kwargs = {
            "inputs": inputs,
            "input_len": length,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "seed": settings.seed,
        }
        base = runtime.generate_answer(**kwargs)
        steered = runtime.generate_answer(
            **kwargs,
            steering_directions={k: v.direction for k, v in session.vectors.items()},
            strengths=strengths,
        )
    return base, steered


def probe(session, settings, prompt, image=None):
    settings.validate()
    if not (prompt or "").strip() and image is None:
        raise ValueError("Enter a prompt or an image.")
    with session.lock, runtime.MODEL_LOCK:
        runtime.ensure_models_loaded(with_saes=False)
        inputs, _, length = runtime.prepare_inputs(image, prompt)
        answer = runtime.generate_answer(
            inputs,
            length,
            settings.max_new_tokens,
            settings.temperature,
            seed=settings.seed,
        )
        record_event(
            session,
            "gemma_test",
            {
                "prompt": prompt,
                "has_image": image is not None,
                "answer": answer,
                "settings": asdict(settings),
            },
        )
    return answer


def compare(session, settings, strengths, prompt, image=None):
    with session.lock:
        require_profile(session, settings, vectors=True)
        validate_strengths(strengths)
        if not (prompt or "").strip() and image is None:
            raise ValueError("Enter a prompt or an image.")
        base, steered = generate_pair(session, settings, strengths, image, prompt)
        record_event(
            session,
            "comparison",
            {
                "prompt": prompt,
                "has_image": image is not None,
                "base": base,
                "steered": steered,
                "strengths": strengths,
                "settings": asdict(settings),
            },
        )
        return base, steered


def generation_key(settings):
    return (settings.seed, settings.temperature, settings.max_new_tokens)


def prepare_evaluation(session, name, split, count, category, settings):
    settings.validate()
    if name not in BENCHMARKS:
        raise ValueError("Select a supported benchmark.")
    if int(count) != count:
        raise ValueError("Item count must be an integer.")
    with session.lock:
        sample = prepare_benchmark(
            BENCHMARKS[name],
            BenchmarkRequest(split, int(count), category, settings.seed),
        )
        session.benchmark_sample = sample
        session.benchmark_base = []
        session.benchmark_settings = None
        session.benchmark_result = None
        return sample


def require_sample(session, settings):
    settings.validate()
    sample = session.benchmark_sample
    if sample is None or sample.request.seed != settings.seed:
        raise ValueError(
            "Prepare and review the benchmark sample with the current seed first."
        )
    return sample


def benchmark_generate(session, settings, strengths, images, prompt, *, base):
    with runtime.MODEL_LOCK:
        runtime.ensure_models_loaded(with_saes=False)
        inputs, _, length = runtime.prepare_inputs(images, prompt)
        kwargs = dict(
            inputs=inputs,
            input_len=length,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            seed=settings.seed,
        )
        if not base:
            return runtime.generate_answer(
                **kwargs,
                steering_directions={
                    k: v.direction for k, v in session.vectors.items()
                },
                strengths=strengths,
            )
        # Fingerprint the actual processed tokens/image tensors and generation defaults.
        # Resolved model revision, precision and software versions prevent stale reuse.
        request = {
            "cache_version": 1,
            "prompt_protocol": "gemma-chat-template-v1",
            "model_id": runtime.MODEL_ID,
            "runtime": runtime.runtime_metadata(),
            "model_config": runtime.model.config.to_dict(),
            "generation_defaults": runtime.model.generation_config.to_dict(),
            "seed": settings.seed,
            "temperature": settings.temperature,
            "max_new_tokens": settings.max_new_tokens,
            "prompt": prompt,
            "inputs": {
                k: base_response_cache.tensor_fingerprint(v) for k, v in inputs.items()
            },
        }
        # Normalize JSON tuples before comparing the request with its saved form.
        import json

        request = json.loads(base_response_cache.canonical(request))
        return base_response_cache.get_or_generate(
            request, lambda: runtime.generate_answer(**kwargs)
        )


def benchmark_base(session, settings, progress=None):
    with session.lock:
        sample = require_sample(session, settings)
        rows = evaluate_base(
            sample,
            lambda images, prompt: benchmark_generate(
                session, settings, {}, images, prompt, base=True
            ),
            progress,
        )
        session.benchmark_base = rows
        session.benchmark_settings = generation_key(settings)
        session.benchmark_result = summarize(sample, rows)
        return session.benchmark_result


def benchmark_steered(session, settings, strengths, progress=None):
    with session.lock:
        sample = require_sample(session, settings)
        if not session.benchmark_base or session.benchmark_settings != generation_key(
            settings
        ):
            raise ValueError(
                "Evaluate the base model with the current seed, temperature and token limit first."
            )
        require_profile(session, settings, vectors=True)
        validate_strengths(strengths)
        result = evaluate_steered(
            sample,
            session.benchmark_base,
            lambda images, prompt: benchmark_generate(
                session, settings, strengths, images, prompt, base=False
            ),
            progress,
        )
        # Steered benchmark responses stay in session memory even when saving is enabled.
        session.benchmark_result = result
        return result


def export(session, settings, strengths, include_weights):
    with session.lock:
        require_profile(session, settings, vectors=True)
        validate_strengths(strengths)
        return export_model(session, settings, strengths, include_weights)
