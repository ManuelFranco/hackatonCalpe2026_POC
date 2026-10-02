"""Single-feature interventions and explicit next-token decision measurements.

The caller owns MODEL_LOCK. No model is loaded by importing this module.
"""

import math

import torch


def label_ids(app, positive, negative):
    ids = [
        app.processor.tokenizer.encode(s, add_special_tokens=False)
        for s in (positive, negative)
    ]
    if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
        raise ValueError(
            "Choose two distinct labels that each encode as one token (e.g. YES / NO)."
        )
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
        "label_probability_mass": float(
            (selected.logsumexp(0) - logits.logsumexp(0)).exp()
        ),
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
    extra = torch.tensor(
        [encoded], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype
    )
    inputs = dict(inputs)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], extra), -1)
    for key, fill in (("attention_mask", 1), ("token_type_ids", 0)):
        if key in inputs:
            old = inputs[key]
            inputs[key] = torch.cat(
                (old, torch.full_like(extra, fill, dtype=old.dtype)), -1
            )
    return inputs, inputs["input_ids"][0].detach().cpu(), inputs["input_ids"].shape[-1]


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
        sequence = app.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    ids = sequence[0, length:].detach().cpu().tolist()
    tok = app.processor.tokenizer
    for end in range(1, len(ids) + 1):
        text = tok.decode(ids[:end], skip_special_tokens=True)
        if marker in text:
            if not text.endswith(marker):
                raise ValueError(
                    "The marker ends inside a token. Choose a marker ending at a token boundary."
                )
            return {
                "assistant_prefix": text,
                "assistant_prefix_ids": ids[:end],
                "baseline_full_answer": tok.decode(ids, skip_special_tokens=True),
                "prefix_marker": marker,
            }
    raise ValueError(
        f"The BASE answer did not contain the marker {marker!r} within {max_new_tokens} tokens."
    )


def probe(app, case, intervention=None, capture=False, direction=None):
    """Measure one forward; preserve SAE reconstruction error during intervention.

    intervention = {layer, feature_id, amount}; amount is in native SAE activation
    units and adds amount * W_dec[j] at the final input position. A zero amount
    does not modify hidden states. No residual-norm normalization is applied.
    """
    if intervention and not math.isfinite(float(intervention["amount"])):
        raise ValueError("Intervention amount must be finite.")
    ids = label_ids(app, case["positive_label"], case["negative_label"])
    inputs, _, length = prepare_case_inputs(app, case)
    captured, diagnostics, handles, norms = {}, {}, [], {}
    layers = (
        app.LAYERS
        if capture
        else ([int(intervention["layer"])] if intervention else [])
    )
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
                decoder = (
                    app.saes[layer].W_dec[feature].detach().float().cpu()
                    if direction is None
                    else direction
                )
                delta = amount * decoder
                after = (
                    before
                    if amount == 0
                    else (x[0, -1] + delta.to(x.device, x.dtype)).float().cpu()
                )
                diagnostics.update(
                    {
                        "activation_before": float(z[feature]),
                        "requested_activation_change": amount,
                        "residual_delta_norm": float((after - before).norm()),
                        "relative_residual_change": float(
                            (after - before).norm() / before.norm().clamp_min(1e-12)
                        ),
                    }
                )
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
        measurement.update(
            {
                "input_tokens": length,
                "assistant_prefix": case.get("assistant_prefix", ""),
                **diagnostics,
            }
        )
        if capture:
            measurement["residual_norms"] = norms
        measurement["top_token"] = app.processor.tokenizer.decode(
            [measurement["top_token_id"]]
        )
        measurement["choice"] = (
            case["positive_label"]
            if measurement["log_odds"] > 0
            else case["negative_label"]
        )
        if "expected" in case:
            measurement["correct"] = measurement["choice"] == case["expected"]
        return measurement, captured
    finally:
        for handle in handles:
            handle.remove()


def generate_feature_report(
    app,
    case,
    candidate,
    dose,
    max_new_tokens=256,
    direction=None,
    schedule="Every decoding step",
    temperature=0.0,
    seed=0,
):
    """Generate freely with one additive decoder direction at each final position.

    Includes prefill and cached decoding, at the final position of each model forward.
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
    decoder = (
        app.saes[layer].W_dec[feature].detach().float().cpu()
        if direction is None
        else direction
    )
    amount = dose * float(candidate["intervention_step"])
    delta = amount * decoder
    diagnostics = {
        "input_tokens": length,
        "dose": dose,
        "amount": amount,
        "forward_calls": 0,
        "intervention_count": 0,
        "schedule": schedule,
        "assistant_prefix": case.get("assistant_prefix", ""),
    }

    def hook(module, args, output):
        x = app.extract_hidden(output)
        diagnostics["forward_calls"] += 1
        if amount == 0 or (
            schedule == "First continuation step only"
            and diagnostics["forward_calls"] > 1
        ):
            return None
        diagnostics["intervention_count"] += 1
        changed = x.clone()
        changed[:, -1, :] += delta.to(x.device, x.dtype)
        if diagnostics["forward_calls"] == 1:
            before, after = x[0, -1].float(), changed[0, -1].float()
            diagnostics["prefill_relative_residual_change"] = float(
                (after - before).norm() / before.norm().clamp_min(1e-12)
            )
        return app.replace_hidden(output, changed)

    handle = app.get_layer_module(layer).register_forward_hook(hook)
    try:
        answer = app.generate_answer(
            inputs, length, int(max_new_tokens), temperature, seed=seed
        )
    finally:
        handle.remove()
    return {"answer": answer, **diagnostics}
