"""Session-scoped causal experiments; shared runtime operations are serialized."""

import html
import math
import torch
from research.causal_interventions import (
    probe,
    generate_feature_report,
    prepare_case_inputs,
)
from . import model_runtime as runtime
from .artifact_store import record_event
from .workflow import require_profile


def profile_candidates(session, settings):
    with session.lock:
        require_profile(session, settings)
        candidates = []
        for layer, profile in session.profile.items():
            mean = (profile.b - profile.a).mean(0)
            for feature in torch.argsort(mean.abs(), descending=True, stable=True)[
                :20
            ].tolist():
                if mean[feature] != 0:
                    candidates.append(
                        (
                            f"Layer {layer} · #{feature} · mean B − A {float(mean[feature]):+.4g}",
                            f"{layer}:{feature}",
                        )
                    )
        if not candidates:
            raise ValueError(
                "No nonzero mean feature contrasts in the current profile. Use manual selection."
            )
        return candidates


def select_candidate(session, settings, source, layer, feature, coefficient):
    settings.validate()
    if layer not in runtime.LAYERS or int(feature) != feature or feature < 0:
        raise ValueError(
            "Select a configured layer and a nonnegative integer feature ID."
        )
    feature = int(feature)
    if source == "Common profile":
        require_profile(session, settings)
    runtime.ensure_models_loaded()
    decoder = runtime.saes[layer].W_dec
    if feature >= decoder.shape[0]:
        raise ValueError(f"Feature ID must be less than {decoder.shape[0]}.")
    row = decoder[feature].detach().float().cpu()
    if not bool(torch.isfinite(row).all()) or row.norm() == 0:
        raise ValueError("The selected decoder direction is zero or non-finite.")
    candidate = {
        "layer": layer,
        "feature_id": feature,
        "source": source,
        "model_id": runtime.MODEL_ID,
        "sae_release": runtime.SAE_RELEASE,
        "sae_id": runtime.SAE_IDS[layer],
        "dictionary_size": int(decoder.shape[0]),
    }
    if source == "Common profile":
        require_profile(session, settings)
        profile = session.profile[layer]
        a, b = profile.a[:, feature], profile.b[:, feature]
        delta = float((b - a).mean())
        if delta == 0:
            raise ValueError(
                "This feature has no mean A/B contrast. Choose another or use manual selection."
            )
        p95 = float(torch.quantile(torch.cat([a, b]).abs(), 0.95))
        magnitude = min(p95, 0.05 * profile.reference_norm / float(row.norm()))
        coefficient = math.copysign(magnitude, delta)
        candidate.update(
            profile_id=session.profile_id,
            mean_a=float(a.mean()),
            mean_b=float(b.mean()),
            mean_delta=delta,
            pair_count=len(a),
            same_direction_pairs=int(((b - a) * delta > 0).sum()),
        )
    elif source != "Manual":
        raise ValueError("Choose Common profile or Manual.")
    if not math.isfinite(coefficient) or abs(coefficient) > 10000:
        raise ValueError(
            "Native coefficient must be finite and between -10000 and 10000."
        )
    return candidate | {"intervention_step": coefficient}


def make_case(prompt, image, prefix, positive="YES", negative="NO", spaced=False):
    if not (prompt or "").strip() and image is None:
        raise ValueError("Enter a prompt or an image.")
    return {
        "prompt": prompt or "",
        "image": image,
        "assistant_prefix": prefix or "",
        "positive_label": (" " if spaced else "") + positive.strip(),
        "negative_label": (" " if spaced else "") + negative.strip(),
    }


def random_direction(candidate, seed):
    decoder = (
        runtime.saes[candidate["layer"]]
        .W_dec[candidate["feature_id"]]
        .detach()
        .float()
        .cpu()
    )
    direction = torch.randn(
        decoder.shape, generator=torch.Generator().manual_seed(seed)
    )
    return direction * decoder.norm() / direction.norm().clamp_min(1e-12)


def token_map(case, candidate):
    inputs, ids, _ = prepare_case_inputs(runtime, case)
    residuals = runtime.capture_prompt_residuals(inputs)[candidate["layer"]]
    # Chunked per-feature extraction avoids retaining tokens × all features.
    values = torch.cat(
        [
            runtime.encode_sae_chunked(runtime.saes[candidate["layer"]], chunk)[
                :, candidate["feature_id"]
            ].clone()
            for chunk in residuals.split(runtime.SAE_CHUNK_TOKENS)
        ]
    )
    mask = runtime.get_image_mask(ids)
    maximum = max(float(values.abs().max()), 1e-8)
    spans = []
    for index, token in enumerate(ids.tolist()):
        if bool(mask[index]):
            continue
        value = float(values[index])
        token_text = runtime.processor.tokenizer.decode(
            [token], skip_special_tokens=False
        )
        spans.append(
            f'<span title="Position {index}; activation {value:.6g}" style="background:rgba(22,163,174,{0.08 + 0.72 * abs(value) / maximum:.3f});padding:2px;border-radius:3px">{html.escape(token_text)}</span>'
        )
    return (
        f"<p>{int(mask.sum())} visual-token positions omitted. Darker means greater activation; hover for values.</p>"
        '<div style="white-space:pre-wrap;overflow-wrap:anywhere;line-height:2;font-family:monospace">'
        + "".join(spans)
        + "</div>"
    )


def sweep(
    session, settings, source, layer, feature, coefficient, case, maximum, progress=None
):
    if maximum not in (1, 2, 3, 4):
        raise ValueError("Maximum sweep dose must be an integer from 1 to 4.")
    with session.lock, runtime.MODEL_LOCK:
        candidate = select_candidate(
            session, settings, source, layer, feature, coefficient
        )
        random = random_direction(candidate, settings.seed)
        base, _ = probe(runtime, case, {**candidate, "amount": 0})
        doses = sorted(
            set([-float(maximum), -1.0, -0.5, 0.0, 0.5, 1.0, float(maximum)])
        )
        measurements = []
        for label, direction in (("SAE feature", None), ("Random control", random)):
            for dose in doses:
                measured, _ = probe(
                    runtime,
                    case,
                    {**candidate, "amount": dose * candidate["intervention_step"]},
                    direction=direction,
                )
                measurements.append({"intervention": label, "dose": dose, **measured})
                if progress:
                    progress(
                        len(measurements) / (2 * len(doses)),
                        desc=f"{label}: dose {dose:+g}",
                    )
        ablation, _ = probe(
            runtime, case, {**candidate, "amount": -base["activation_before"]}
        )
        heatmap = token_map(case, candidate)
        zero_ok = all(
            m["log_odds"] == base["log_odds"] for m in measurements if m["dose"] == 0
        )
        result = {
            "candidate": candidate,
            "base": base,
            "measurements": measurements,
            "ablation": ablation,
            "zero_control_passed": zero_ok,
        }
        record_event(
            session,
            "causal_sweep",
            {
                **result,
                "case": {k: v for k, v in case.items() if k != "image"},
                "has_image": case["image"] is not None,
                "seed": settings.seed,
            },
        )
        return result, heatmap


def responses(
    session, settings, source, layer, feature, coefficient, case, dose, schedule
):
    with session.lock, runtime.MODEL_LOCK:
        candidate = select_candidate(
            session, settings, source, layer, feature, coefficient
        )
        random = random_direction(candidate, settings.seed)
        results = [
            generate_feature_report(
                runtime,
                case,
                candidate,
                amount,
                settings.max_new_tokens,
                direction=direction,
                schedule=schedule,
                temperature=settings.temperature,
                seed=settings.seed,
            )
            for amount, direction in ((0, None), (dose, None), (dose, random))
        ]
        record_event(
            session,
            "causal_responses",
            {
                "candidate": candidate,
                "dose": dose,
                "schedule": schedule,
                "seed": settings.seed,
                "temperature": settings.temperature,
                "max_new_tokens": settings.max_new_tokens,
                "case": {k: v for k, v in case.items() if k != "image"},
                "has_image": case["image"] is not None,
                "responses": results,
            },
        )
        return results
