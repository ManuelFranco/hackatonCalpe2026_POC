"""Label-free feature measurements and controlled input views.

No model or dashboard state is owned here. The caller serializes model access.
"""

import re
import torch
from .causal_interventions import prepare_case_inputs

VIEWS = ("Full input", "Code + preamble", "Text without code")
FENCES = re.compile(r"^```[^\n]*\n.*?^```[ \t]*$", re.MULTILINE | re.DOTALL)


def input_view(text, view):
    if view == "Full input":
        return text
    blocks = list(FENCES.finditer(text))
    if view not in VIEWS or not blocks:
        raise ValueError(
            "Code/text controls require fenced code blocks. Use Full input for other cases."
        )
    if view == "Code + preamble":
        # Preserve declared trust assumptions; omit inter-block and trailing reviews.
        return (
            text[: blocks[0].start()].strip()
            + "\n\n"
            + "\n\n".join(m.group() for m in blocks)
        )
    return FENCES.sub("", text).strip()


def measure_feature(app, case, candidate, token_scope):
    inputs, ids, _ = prepare_case_inputs(app, case)
    captured = []

    def capture(module, args, output):
        captured.append(app.extract_hidden(output)[0].detach().float().cpu())

    handle = app.get_layer_module(candidate["layer"]).register_forward_hook(capture)
    try:
        with torch.inference_mode():
            app.model(**inputs, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected one residual capture for the selected layer.")
    values = torch.cat(
        [
            app.encode_sae_chunked(app.saes[candidate["layer"]], chunk)[
                :, candidate["feature_id"]
            ].clone()
            for chunk in captured[0].split(app.SAE_CHUNK_TOKENS)
        ]
    )
    mask = app.feature_token_mask(app.get_image_mask(ids), token_scope)
    selected = values[mask]
    if not selected.numel() or not bool(torch.isfinite(selected).all()):
        raise ValueError("Empty or non-finite feature measurements.")
    return {
        "mean": float(selected.mean()),
        "max": float(selected.max()),
        "active_fraction": float((selected > 0).float().mean()),
        "selected_tokens": len(selected),
    }


def summarize_ratings(rows, keys):
    """Manual, complete-case paired means; missing scores never become zero."""
    import math

    if len(rows) != len(keys):
        raise ValueError("Keep one rating row per generated response.")
    rated = {}
    for row, (case, condition) in zip(rows, keys):
        if list(row[:2]) != [case, condition]:
            raise ValueError("Do not edit case or condition identifiers.")
        values = row[2:5]
        if len(values) != 3 or any(v is None or str(v).strip() == "" for v in values):
            continue
        scores = [float(v) for v in values]
        if any(math.isnan(v) for v in scores):
            continue
        if any(not math.isfinite(v) or v not in (0, 1, 2) for v in scores):
            raise ValueError(
                "Each score must be 0, 1 or 2; leave unfinished rows blank."
            )
        rated[(case, condition)] = sum(scores) / 3
    summary = []
    for condition in dict.fromkeys(condition for _, condition in keys):
        available = [
            (case, score) for (case, name), score in rated.items() if name == condition
        ]
        paired = [
            score - rated[(case, "Base")]
            for case, score in available
            if (case, "Base") in rated
        ]
        summary.append(
            {
                "condition": condition,
                "rated_cases": len(available),
                "mean_quality": sum(s for _, s in available) / len(available)
                if available
                else None,
                "paired_cases": len(paired),
                "mean_delta_vs_base": sum(paired) / len(paired) if paired else None,
            }
        )
    return summary
