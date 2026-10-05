"""Automatic input-feature candidates and bounded, read-only token observations."""

import math

import torch


LIMIT = 12
CHUNK_SIZE = 16
EPSILON = 1e-6


def rank_candidates(profile, limit=LIMIT, polarity="Both signs"):
    """Rank stable paired contrasts; scores are heuristics, never confidence."""
    if not isinstance(limit, int) or not 1 <= limit <= 32:
        raise ValueError("Choose between 1 and 32 discovery candidates.")
    if polarity not in {"Both signs", "B higher", "A higher"}:
        raise ValueError("Choose B higher, A higher or Both signs.")
    a, b = profile.a.detach().float().cpu(), profile.b.detach().float().cpu()
    if (
        a.ndim != 2
        or a.shape != b.shape
        or not all(a.shape)
        or not bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    ):
        raise ValueError("Discovery requires matching, finite paired profiles.")
    delta = b - a
    mean = delta.mean(0)
    rms = delta.square().mean(0).sqrt()
    coverage = (delta.abs() > EPSILON).float().mean(0)
    stability = mean.abs() / rms.clamp_min(EPSILON)
    score = mean.abs() * stability * coverage
    if polarity != "Both signs":
        score = score * (mean > 0 if polarity == "B higher" else mean < 0)
    order = torch.argsort(score, descending=True, stable=True)
    rows = []
    for j in order[:limit].tolist():
        if mean[j].abs() <= EPSILON or score[j] <= EPSILON:
            continue
        same = (delta[:, j] * mean[j] > 0) & (delta[:, j].abs() > EPSILON)
        rows.append(
            {
                "feature": j,
                "mean_a": float(a[:, j].mean()),
                "mean_b": float(b[:, j].mean()),
                "contrast": float(mean[j]),
                "score": float(score[j]),
                "stability": float(stability[j]),
                "coverage": float(coverage[j]),
                "same_sign": int(same.sum()),
                "pairs": len(a),
                "pair_deltas": delta[:, j].tolist(),
            }
        )
    return rows


def observe_candidates(app, sae, residuals, features):
    """Encode in small batches; retain tokens × candidates, not tokens × width."""
    width = sae.W_dec.shape[0]
    if (
        not features
        or len(features) > 32
        or len(set(features)) != len(features)
        or any(not isinstance(j, int) or not 0 <= j < width for j in features)
    ):
        raise ValueError(
            "Candidate IDs must be distinct and inside the SAE dictionary."
        )
    if residuals.ndim != 2 or not residuals.shape[0]:
        raise ValueError("Token observation requires nonempty residuals.")
    chunks = []
    for start in range(0, len(residuals), CHUNK_SIZE):
        encoded = app.encode_sae_chunked(sae, residuals[start : start + CHUNK_SIZE])
        if encoded.shape != (min(CHUNK_SIZE, len(residuals) - start), width):
            raise ValueError("Candidate observations have the wrong shape.")
        chosen = encoded[:, features].detach().float().cpu().clone()
        del encoded
        if not bool(torch.isfinite(chosen).all()):
            raise ValueError("Candidate activations must be finite.")
        chunks.append(chosen)
    return torch.cat(chunks)


def source_token_spans(tokenizer, rendered, input_ids, prompt):
    """Verified input-token offsets in the original source, including trimmed text."""
    source, source_start = prompt, 0
    if not source or rendered.count(source) != 1:
        source = prompt.strip()
        source_start = len(prompt) - len(prompt.lstrip())
    if not source or rendered.count(source) != 1:
        raise ValueError("Source text cannot be uniquely located in the chat template.")
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
    except (TypeError, ValueError, NotImplementedError):
        raise ValueError(
            "Source highlighting requires verified tokenizer offsets."
        ) from None
    expected = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    if list(encoded["input_ids"]) != expected:
        raise ValueError("Offset token IDs differ from the captured model input.")
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(expected):
        raise ValueError("Tokenizer offsets have the wrong length.")
    offset, size = rendered.index(source), len(source)
    spans = []
    for start, end in offsets:
        if not 0 <= start <= end <= len(rendered):
            raise ValueError("Tokenizer offsets fall outside the rendered input.")
        left, right = max(start, offset), min(end, offset + size)
        spans.append(
            [left - offset + source_start, right - offset + source_start]
            if left < right
            else [0, 0]
        )
    return spans


def source_segments(text, spans, values, selected):
    """Use the maximum for overlapping token spans without inventing alignment."""
    if len(spans) != len(values) or len(spans) != len(selected):
        raise ValueError("Source spans and observations must have matching lengths.")
    events = {0: [], len(text): []}
    for i, (start, end) in enumerate(spans):
        if not 0 <= start <= end <= len(text):
            raise ValueError("Source spans fall outside the original text.")
        if not all(math.isfinite(float(v)) for v in values[i]):
            raise ValueError("Source observations must be finite.")
        if start != end:
            events.setdefault(start, []).append((i, True))
            events.setdefault(end, []).append((i, False))
    positions, active, result = sorted(events), set(), []
    for start, end in zip(positions, positions[1:]):
        for i, entering in events[start]:
            if entering:
                active.add(i)
            else:
                active.discard(i)
        indices = sorted(active)
        measured = [i for i in indices if selected[i]]
        result.append(
            {
                "text": text[start:end],
                "positions": measured or indices,
                "selected": bool(measured),
                "values": [
                    max(float(values[i][j]) for i in measured)
                    for j in range(len(values[0]))
                ]
                if measured
                else [],
            }
        )
    return result
