"""Single-feature interventions and explicit next-token decision measurements.

The caller owns MODEL_LOCK. No model is loaded by importing this module.
"""

from pathlib import Path
import json
import math
import time
import uuid

import torch


def label_ids(app, positive, negative):
    ids = [app.processor.tokenizer.encode(s, add_special_tokens=False)
           for s in (positive, negative)]
    if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
        raise ValueError("Choose two distinct labels that each encode as one token (e.g. YES / NO).")
    return [x[0] for x in ids]


def readout(logits, ids):
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits in the causal readout.")
    selected = logits[ids]
    margin = float(selected[0] - selected[1])
    return {
        "log_odds": margin,
        "p_positive_given_labels": float(selected.softmax(-1)[0]),
        "label_probability_mass": float((selected.logsumexp(0) - logits.logsumexp(0)).exp()),
        "top_token_id": int(logits.argmax()),
    }


def prepare_case_inputs(app, case):
    """Append a shared assistant continuation; retain image tensors and prompt length."""
    inputs, ids, length = app.prepare_inputs(case.get("image"), case["prompt"])
    prefix = case.get("assistant_prefix", "")
    if not prefix:
        return inputs, ids, length
    encoded = case.get("assistant_prefix_ids")
    if encoded is None:
        encoded = app.processor.tokenizer.encode(prefix, add_special_tokens=False)
    extra = torch.tensor([encoded], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    inputs = dict(inputs)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], extra), -1)
    for key, fill in (("attention_mask", 1), ("token_type_ids", 0)):
        if key in inputs:
            old = inputs[key]
            inputs[key] = torch.cat((old, torch.full_like(extra, fill, dtype=old.dtype)), -1)
    return inputs, inputs["input_ids"][0].detach().cpu(), length


def generate_baseline_prefix(app, image, prompt, marker, max_new_tokens=160):
    """Generate once, then retain exact assistant tokens through a user-selected marker.

    The marker must end at a token boundary; the following decision never enters
    the replay prefix. No SAE intervention is active during this generation.
    Caller owns MODEL_LOCK. Works with text and image inputs.
    """
    if not marker:
        return {"assistant_prefix": "", "assistant_prefix_ids": []}
    inputs, _, length = app.prepare_inputs(image, prompt)
    with torch.inference_mode():
        sequence = app.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    ids = sequence[0, length:].detach().cpu().tolist()
    tok = app.processor.tokenizer
    for end in range(1, len(ids) + 1):
        text = tok.decode(ids[:end], skip_special_tokens=True)
        if marker in text:
            if not text.endswith(marker):
                raise ValueError("The marker ends inside a token. Choose a marker ending at a token boundary.")
            return {"assistant_prefix": text, "assistant_prefix_ids": ids[:end],
                    "baseline_full_answer": tok.decode(ids, skip_special_tokens=True), "prefix_marker": marker}
    raise ValueError(f"The BASE answer did not contain the marker {marker!r} within {max_new_tokens} tokens.")


def probe(app, case, intervention=None, capture=False, direction=None):
    """Measure one forward; preserve SAE reconstruction error during intervention.

    intervention = {layer, feature_id, amount}; amount is in native SAE activation
    units and adds amount * W_dec[j] at the final input position. A zero amount
    does not modify hidden states. No residual-norm normalization is applied.
    """
    ids = label_ids(app, case["positive_label"], case["negative_label"])
    inputs, _, length = prepare_case_inputs(app, case)
    captured, diagnostics, handles, norms = {}, {}, [], {}
    layers = app.LAYERS if capture else ([int(intervention["layer"])] if intervention else [])
    for layer in layers:
        def hook(module, args, output, layer=layer):
            x = app.extract_hidden(output)
            before = x[0, -1].detach().float().cpu()
            z = None
            if capture:
                z = app.encode_sae_chunked(app.saes[layer], before[None])[0]
                captured[layer] = z
                norms[layer] = float(before.norm())
            if intervention and layer == int(intervention["layer"]):
                feature = int(intervention["feature_id"])
                if z is None:
                    z = app.encode_sae_chunked(app.saes[layer], before[None])[0]
                amount = float(intervention["amount"])
                decoder = app.saes[layer].W_dec[feature].detach().float().cpu() if direction is None else direction
                delta = amount * decoder
                after = before if amount == 0 else (x[0, -1] + delta.to(x.device, x.dtype)).float().cpu()
                diagnostics.update({"activation_before": float(z[feature]),
                                    "requested_activation_change": amount,
                                    "residual_delta_norm": float((after - before).norm()),
                                    "relative_residual_change": float((after - before).norm() / before.norm().clamp_min(1e-12))})
                if amount:
                    changed = x.clone()
                    changed[:, -1, :] += delta.to(x.device, x.dtype)
                    return app.replace_hidden(output, changed)
            return None
        handles.append(app.get_layer_module(layer).register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = app.model(**inputs, use_cache=False, logits_to_keep=1)
        measurement = readout(output.logits[0, -1], ids)
        measurement.update({"input_tokens": length, "assistant_prefix": case.get("assistant_prefix", ""), **diagnostics})
        if capture:
            measurement["residual_norms"] = norms
        measurement["top_token"] = app.processor.tokenizer.decode([measurement["top_token_id"]])
        measurement["choice"] = case["positive_label"] if measurement["log_odds"] > 0 else case["negative_label"]
        if "expected" in case:
            measurement["correct"] = measurement["choice"] == case["expected"]
        return measurement, captured
    finally:
        for handle in handles:
            handle.remove()


def generate_feature_report(app, case, candidate, dose, max_new_tokens=256, direction=None,
                            schedule="Every decoding step"):
    """Generate freely with one additive decoder direction at each final position.

    Includes prefill and cached decoding, just like section 2's default schedule.
    No label is forced and no SAE reconstruction replaces the residual stream.
    The caller owns MODEL_LOCK. Always remove the hook, including on failure.
    """
    dose = float(dose)
    if not math.isfinite(dose) or abs(dose) > 4:
        raise ValueError("Report dose must be finite and between -4 and 4.")
    if schedule not in {"Every decoding step", "First continuation step only"}:
        raise ValueError("Unknown intervention schedule.")
    inputs, _, length = prepare_case_inputs(app, case)
    layer, feature = int(candidate["layer"]), int(candidate["feature_id"])
    decoder = (app.saes[layer].W_dec[feature].detach().float().cpu()
               if direction is None else direction)
    amount = dose * float(candidate["intervention_step"])
    delta = amount * decoder
    diagnostics = {"input_tokens": length, "dose": dose, "amount": amount,
                   "forward_calls": 0, "intervention_count": 0, "schedule": schedule,
                   "assistant_prefix": case.get("assistant_prefix", "")}

    def hook(module, args, output):
        x = app.extract_hidden(output)
        diagnostics["forward_calls"] += 1
        if amount == 0 or (schedule == "First continuation step only" and diagnostics["forward_calls"] > 1):
            return None
        diagnostics["intervention_count"] += 1
        changed = x.clone()
        changed[:, -1, :] += delta.to(x.device, x.dtype)
        if diagnostics["forward_calls"] == 1:
            before, after = x[0, -1].float(), changed[0, -1].float()
            diagnostics["prefill_relative_residual_change"] = float(
                (after - before).norm() / before.norm().clamp_min(1e-12))
        return app.replace_hidden(output, changed)

    handle = app.get_layer_module(layer).register_forward_hook(hook)
    try:
        answer = app.generate_answer(inputs, length, int(max_new_tokens), 0.0, seed=0)
    finally:
        handle.remove()
    return {"answer": answer, **diagnostics}


def rank_candidates(records, top_per_layer=6):
    """Rank training-only B-A contrasts; require >=75% consistent nonzero pairs."""
    by_pair = {}
    for row in records:
        by_pair.setdefault(row["case"]["pair"], {})[row["case"]["side"]] = row
    if not by_pair or any(set(p) != {"A", "B"} for p in by_pair.values()):
        raise ValueError("Every training pair needs one A and one B.")
    pairs = list(by_pair.values())
    candidates = []
    for layer in pairs[0]["A"]["features"]:
        a = torch.stack([p["A"]["features"][layer] for p in pairs])
        b = torch.stack([p["B"]["features"][layer] for p in pairs])
        d = b - a
        mean = d.mean(0)
        sign = mean.sign()
        agreement = (d * sign > 1e-6).float().mean(0)
        # Standardized paired contrast rewards consistency and discounts outliers.
        score = mean.abs() / d.square().mean(0).sqrt().clamp_min(1e-6)
        eligible = (agreement >= 0.75) & (mean.abs() > 1e-6)
        stable = torch.argsort(torch.where(eligible, score, -torch.ones_like(score)), descending=True)
        large = torch.argsort(torch.where(eligible, mean.abs(), -torch.ones_like(score)), descending=True)
        # Include both stable and large contrasts; tiny consistent deltas alone
        # can sit below bfloat16 precision and need not influence the decision.
        ranked = list(dict.fromkeys(stable[:(top_per_layer + 1) // 2].tolist()
                                   + large.tolist()))[:top_per_layer]
        for j in ranked:
            if not eligible[j]:
                continue
            candidates.append({"layer": int(layer), "feature_id": j,
                               "train_agreement": float(agreement[j]), "train_score": float(score[j]),
                               "mean_A": float(a[:, j].mean()), "mean_B": float(b[:, j].mean()),
                               "natural_delta": float(mean[j]), "pair_deltas": d[:, j].tolist(),
                               "activation_p95": float(torch.quantile(torch.cat([a[:, j], b[:, j]]), .95))})
    return candidates


def compare_candidate(app, candidate, cases, baselines):
    effects = []
    for case in cases:
        observations = {}
        for dose in (-1, 1):
            m, _ = probe(app, case, {**candidate, "amount": dose * candidate["intervention_step"]})
            observations[str(dose)] = m
        base = baselines[case["id"]]
        effects.append({"case_id": case["id"], "base": base, "minus": observations["-1"],
                        "plus": observations["1"],
                        "symmetric_effect": (observations["1"]["log_odds"] - observations["-1"]["log_odds"]) / 2})
    values = [r["symmetric_effect"] for r in effects]
    return {"candidate": candidate, "mean_effect": sum(values) / len(values),
            "positive_fraction": sum(v > 0 for v in values) / len(values), "cases": effects}


def scale_candidate(candidate, reference_norm, decoder_norm):
    magnitude = min(candidate["activation_p95"], .05 * reference_norm / max(decoder_norm, 1e-12))
    return {**candidate, "intervention_step": math.copysign(magnitude, candidate["natural_delta"]),
            "reference_residual_norm": reference_norm, "decoder_norm": decoder_norm}


def run_discovery(app, cases, output, progress=None, top_per_layer=6, mode="counterbalanced"):
    """Select on validation only; evaluate the selected feature on held-out cases."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"counterbalanced", "decision_control"}:
        raise ValueError("Unknown discovery mode.")
    report = {"format_version": 2, "mode": mode, "created_unix": time.time(), "runtime": app.runtime_metadata(),
              "model_id": app.MODEL_ID, "sae_release": app.SAE_RELEASE,
              "sae_ids": {str(k): v for k, v in app.SAE_IDS.items()},
              "protocol": "Training-only SAE contrasts; validation-only feature selection; reserved tests with both YES/NO polarities. Decision-control mode selects on direct questions only. Readout is conditional on two labels, not free generation. Unit dose is the training 95th percentile activation, capped at 5% of the training residual norm.",
              "cases": cases, "baseline": {}, "candidates": [], "validation": []}
    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")
    def update(message):
        if progress:
            progress(message)
        save()
    training = []
    for i, case in enumerate(cases):
        # Held-out cases are not measured until selection has finished below.
        if case["split"] == "test":
            continue
        baseline, features = probe(app, case, capture=case["split"] == "train")
        report["baseline"][case["id"]] = baseline
        if features and (mode == "counterbalanced" or case["positive_label"] == "YES"):
            training.append({"case": case, "features": features, "norms": baseline["residual_norms"]})
        update(f"Captured {case['id']}")
    report["candidates"] = rank_candidates(training, top_per_layer)
    for candidate in report["candidates"]:
        layer = candidate["layer"]
        reference_norm = sum(row["norms"][layer] for row in training) / len(training)
        decoder_norm = float(app.saes[layer].W_dec[candidate["feature_id"]].float().norm())
        candidate.update(scale_candidate(candidate, reference_norm, decoder_norm))
    valid = [c for c in cases if c["split"] == "valid"
             and (mode == "counterbalanced" or c["positive_label"] == "YES")]
    for c in report["candidates"]:
        result = compare_candidate(app, c, valid, report["baseline"])
        report["validation"].append(result)
        update(f"Validated layer {c['layer']} feature {c['feature_id']}: effect {result['mean_effect']:+.3f}, consistency {result['positive_fraction']:.0%}")
    if not report["validation"]:
        raise ValueError("No consistent training candidates. Add more varied A/B examples.")
    selected = max(report["validation"], key=lambda r: (r["positive_fraction"], r["mean_effect"]))
    report["selected"] = selected["candidate"]
    report["validation_gate_passed"] = selected["positive_fraction"] >= .75 and selected["mean_effect"] >= .1
    update("Feature frozen; starting held-out evaluation")
    tests = [c for c in cases if c["split"] == "test"]
    for case in tests:
        report["baseline"][case["id"]], _ = probe(app, case)
    report["held_out"] = compare_candidate(app, report["selected"], tests, report["baseline"])
    # A seeded random residual direction with exactly the selected decoder norm.
    selected = report["selected"]
    decoder = app.saes[selected["layer"]].W_dec[selected["feature_id"]].detach().float().cpu()
    gen = torch.Generator().manual_seed(20260928)
    random_direction = torch.randn(decoder.shape, generator=gen)
    random_direction *= decoder.norm() / random_direction.norm()
    controls = []
    for case in tests:
        plus, _ = probe(app, case, {**selected, "amount": selected["intervention_step"]}, direction=random_direction)
        minus, _ = probe(app, case, {**selected, "amount": -selected["intervention_step"]}, direction=random_direction)
        controls.append({"case_id": case["id"], "plus": plus, "minus": minus,
                         "symmetric_effect": (plus["log_odds"] - minus["log_odds"]) / 2})
    report["random_direction_control"] = controls
    first = tests[0]
    zero, _ = probe(app, first, {**selected, "amount": 0})
    report["zero_control"] = {"case_id": first["id"], "identical_log_odds": zero["log_odds"] == report["baseline"][first["id"]]["log_odds"]}
    update("Discovery and held-out controls complete")
    return report


def new_report_path(app):
    return app.RUNS_DIR / (time.strftime("causal_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]) / "report.json"
