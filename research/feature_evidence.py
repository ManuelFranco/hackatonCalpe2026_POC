"""Bounded, single-position causal measurements and paired decision statistics.

This module never installs persistent steering or chooses a desired wrong answer.
Callers own the model lock. A/B probabilities are conditional decision readouts.
"""

import math
from statistics import NormalDist

import torch


def shortlist(profile, limit=12):
    """Calibration stability is a candidate heuristic, not causal evidence."""
    delta = profile.b.detach().float().cpu() - profile.a.detach().float().cpu()
    if delta.ndim != 2 or not delta.shape[0] or not torch.isfinite(delta).all():
        raise ValueError("A finite paired profile is required.")
    mean = delta.mean(0)
    rms = delta.square().mean(0).sqrt()
    presence = (delta.abs() > 1e-6).float().mean(0)
    stability = mean.abs() / rms.clamp_min(1e-8) * presence
    ids = torch.argsort(stability, descending=True, stable=True)
    return [
        {
            "feature": int(j),
            "stability": float(stability[j]),
            "contrast": float(mean[j]),
            "same_sign": int(((delta[:, j] * mean[j]) > 0).sum()),
            "pairs": len(delta),
        }
        for j in ids[:limit]
        if rms[j] > 1e-6 and mean[j].abs() > 1e-6
    ]


def activation_effect(before, after, feature):
    """Local re-encoding changes; this is not behavioral isolation."""
    change = after - before
    threshold = torch.maximum(before.abs(), after.abs()) * 0.001 + 1e-6
    moved = change.abs() > threshold
    others = moved.clone()
    others[feature] = False
    energy = float(change.square().sum())
    top = change.abs().clone()
    top[feature] = 0
    ids = torch.argsort(top, descending=True, stable=True)[:6]
    return {
        "activation_before": float(before[feature]),
        "activation_after": float(after[feature]),
        "other_features_moved": int(others.sum()),
        "target_energy_share": float(change[feature].square()) / energy
        if energy > 1e-16
        else None,
        "neighbors": [
            {
                "feature": int(j),
                "before": float(before[j]),
                "after": float(after[j]),
                "delta": float(change[j]),
            }
            for j in ids
            if others[j]
        ],
    }


def decision_probe(
    app,
    inputs,
    label_tokens,
    layer,
    feature=None,
    change=0.0,
    random_seed=None,
    budget=0.02,
):
    """One full forward, one decoder direction, final input position only.

    Rescale the current encoded activity by +/- change, capped at an intended
    residual norm budget. Random controls match that case's intended delta norm.
    Re-encode the actual, dtype-rounded residual rather than assuming clamping.
    """
    if not math.isfinite(change) or not -1 <= change <= 1:
        raise ValueError("Activity change must be between -1 and 1.")
    if not math.isfinite(budget) or not 0 < budget <= 0.05:
        raise ValueError("Residual budget must be positive and at most 5%.")
    sae = app.saes[layer]
    if feature is not None and not 0 <= feature < sae.W_dec.shape[0]:
        raise ValueError("Feature ID is outside the loaded SAE dictionary.")
    diagnostics = {}

    def hook(module, args, output):
        if diagnostics:
            raise ValueError("Expected exactly one selected-layer forward.")
        hidden = app.extract_hidden(output)
        if hidden.shape[0] != 1:
            raise ValueError("Decision experiments require one case per forward.")
        before = hidden[0, -1].detach().float().cpu().clone()
        z_before = app.encode_sae_chunked(sae, before[None])[0]
        if not torch.isfinite(z_before).all():
            raise ValueError("Non-finite SAE activations.")
        diagnostics["residual_change"] = 0.0
        if feature is None:
            return None
        decoder = sae.W_dec[feature].detach().float().cpu()
        delta = float(z_before[feature]) * change * decoder
        intended_norm = float(delta.norm())
        cap = budget * float(before.norm())
        capped = intended_norm > cap
        if capped:
            delta = delta * (cap / max(intended_norm, 1e-12))
        if random_seed is not None:
            direction = torch.randn(
                decoder.shape, generator=torch.Generator().manual_seed(random_seed)
            )
            delta = direction / direction.norm().clamp_min(1e-12) * delta.norm()
            if change < 0:
                delta = -delta
        changed = hidden.clone()
        changed[:, -1, :] += delta.to(hidden.device, hidden.dtype)
        after = changed[0, -1].detach().float().cpu()
        z_after = app.encode_sae_chunked(sae, after[None])[0]
        if not torch.isfinite(z_after).all():
            raise ValueError("Non-finite re-encoded activations.")
        diagnostics.update(activation_effect(z_before, z_after, feature))
        diagnostics.update(
            residual_change=float(
                (after - before).norm() / before.norm().clamp_min(1e-12)
            ),
            intended_delta_norm=float(delta.norm()),
            actual_delta_norm=float((after - before).norm()),
            budget_capped=capped,
            inactive=bool(z_before[feature] == 0),
        )
        return app.replace_hidden(output, changed) if delta.count_nonzero() else None

    handle = app.get_layer_module(layer).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = app.model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1].detach().float().cpu()
        if not diagnostics or not torch.isfinite(logits).all():
            raise ValueError("Missing capture or non-finite decision logits.")
        a, b = (float(logits[token]) for token in label_tokens)
        margin = b - a
        selected = logits[list(label_tokens)]
        return {
            **diagnostics,
            "margin": margin,
            "p_b": float(selected.softmax(0)[1]),
            "label_mass": float((selected.logsumexp(0) - logits.logsumexp(0)).exp()),
            "prediction": "B" if margin > 0 else "A" if margin < 0 else "Tie",
        }
    finally:
        handle.remove()


def classification(rows):
    """B is the positive condition. Ties are counted separately and as errors."""
    if not rows:
        raise ValueError("Cannot score an empty study.")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "ties_a": 0, "ties_b": 0}
    for row in rows:
        truth, prediction = row["expected"], row["prediction"]
        if truth not in ("A", "B") or prediction not in ("A", "B", "Tie"):
            raise ValueError("Decision rows require A/B truth and A/B/Tie predictions.")
        if prediction == "Tie":
            counts["ties_" + truth.lower()] += 1
        else:
            key = {
                ("A", "A"): "tn",
                ("A", "B"): "fp",
                ("B", "A"): "fn",
                ("B", "B"): "tp",
            }[(truth, prediction)]
            counts[key] += 1
    n_b = counts["tp"] + counts["fn"] + counts["ties_b"]
    n_a = counts["tn"] + counts["fp"] + counts["ties_a"]
    hit = counts["tp"] / n_b if n_b else None
    false_alarm = counts["fp"] / n_a if n_a else None
    # Loglinear correction avoids infinite inverse-normal values on small samples.
    dprime = criterion = None
    if n_a and n_b and not counts["ties_a"] + counts["ties_b"]:
        z_hit = NormalDist().inv_cdf((counts["tp"] + 0.5) / (n_b + 1))
        z_false = NormalDist().inv_cdf((counts["fp"] + 0.5) / (n_a + 1))
        dprime, criterion = z_hit - z_false, -(z_hit + z_false) / 2
    return {
        **counts,
        "n": len(rows),
        "n_a": n_a,
        "n_b": n_b,
        "accuracy": (counts["tp"] + counts["tn"]) / len(rows),
        "hit_rate": hit,
        "false_alarm_rate": false_alarm,
        "dprime": dprime,
        "criterion": criterion,
    }


def summarize(base, treated):
    baseline = {row["case_id"]: row for row in base}
    if len(baseline) != len(base) or {r["case_id"] for r in treated} != set(baseline):
        raise ValueError(
            "Baseline and intervention must contain the same unique cases."
        )
    if len(treated) != len(base):
        raise ValueError("Duplicate intervention cases.")
    targets = [r for r in treated if r["group"] == "target"]
    controls = [r for r in treated if r["group"] == "control"]
    for row in treated:
        original = baseline[row["case_id"]]
        if (row["expected"], row["group"]) != (original["expected"], original["group"]):
            raise ValueError("Intervention labels must match the frozen baseline.")
    preserved = [
        r for r in controls if baseline[r["case_id"]]["prediction"] == r["expected"]
    ]
    margins = [
        (1 if r["expected"] == "B" else -1)
        * (r["margin"] - baseline[r["case_id"]]["margin"])
        for r in targets
    ]
    return {
        **classification(targets),
        "correct_margin_change": sum(margins) / len(margins),
        "control_total": len(controls),
        "control_eligible": len(preserved),
        "control_regressions": sum(r["prediction"] != r["expected"] for r in preserved),
        "control_flips": sum(
            r["prediction"] != baseline[r["case_id"]]["prediction"] for r in controls
        ),
        "inactive_cases": sum(r.get("inactive", False) for r in targets),
        "mean_other_features": sum(r.get("other_features_moved", 0) for r in targets)
        / len(targets),
        "max_residual_change": max(r["residual_change"] for r in treated),
        "mean_label_mass": sum(r["label_mass"] for r in targets) / len(targets),
    }
