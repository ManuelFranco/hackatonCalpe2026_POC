"""Read-only, next-token-aligned SAE observations. No generation policy lives here."""

import re

import torch


CHUNK_SIZE = 16  # Bound temporary dense activations, including 262k dictionaries.


def feature_ids(text):
    if not (text or "").strip():
        return []
    parts = [part.strip() for part in text.split(",")]
    if not all(re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError("Use comma-separated integer feature IDs.")
    ids = list(map(int, parts))
    if len(ids) > 3 or len(set(ids)) != len(ids):
        raise ValueError("Select up to 3 distinct feature IDs.")
    return ids


def decoded_spans(app, ids, response):
    """Align only append-only decoding; never infer offsets from token spellings."""

    def decode(tokens):
        return app.processor.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    full = decode(ids)
    if full.strip() != response:
        return [], "Unavailable: decoded text differs from the response"
    left = len(full) - len(full.lstrip())
    right = left + len(response)
    spans, previous = [], ""
    for end in range(1, len(ids) + 1):
        prefix = decode(ids[:end])
        if not full.startswith(prefix) or not prefix.startswith(previous):
            return [], "Unavailable: tokenizer rewrites decoded prefixes"
        spans.append(
            [
                max(left, min(len(previous), right)) - left,
                max(left, min(len(prefix), right)) - left,
            ]
        )
        previous = prefix
    return spans, "Exact incremental decode; offsets refer to the visible response"


class GenerationTrace:
    def __init__(self, layer):
        self.layers = (layer,)
        self.before, self.after, self.forward_lengths = [], [], []
        self.token_ids = None

    def capture(self, layer, before, after):
        if layer not in self.layers or before.ndim != 3 or before.shape[0] != 1:
            raise ValueError(
                "Generation traces require one sequence and one observed layer."
            )
        # The final state predicts the NEXT token, including at prefill.
        self.before.append(before[0, -1].detach().cpu().clone())
        self.after.append(after[0, -1].detach().cpu().clone())
        self.forward_lengths.append(int(before.shape[1]))

    def finish(self, tokens, input_len, eos_ids):
        self.token_ids = tokens.detach().cpu().tolist()
        if not self.token_ids or len(self.token_ids) != len(self.before):
            raise ValueError(
                "Token/forward counts differ; cannot align this generation trace."
            )
        if self.forward_lengths != [input_len] + [1] * (len(self.token_ids) - 1):
            raise ValueError(
                "Trace requires standard cached, single-sequence generation."
            )
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        self.ended_at_eos = self.token_ids[-1] in eos_ids

    def _chunks(self, app, states):
        sae = app.saes[self.layers[0]]
        for start in range(0, len(states), CHUNK_SIZE):
            yield app.encode_sae_chunked(
                sae, torch.stack(states[start : start + CHUNK_SIZE])
            )

    def select(self, app):
        """Top baseline activation ranges, then peak activity for ties; not causality."""
        low = high = None
        for z in self._chunks(app, self.before):
            lo, hi = z.amin(0), z.amax(0)
            low = lo if low is None else torch.minimum(low, lo)
            high = hi if high is None else torch.maximum(high, hi)
        by_peak = torch.argsort(high, descending=True, stable=True)
        order = by_peak[
            torch.argsort((high - low)[by_peak], descending=True, stable=True)
        ]
        return [int(j) for j in order[:3] if high[j] > 0]

    def measure(self, app, ids, max_new_tokens, response=None):
        before = [z[:, ids].tolist() for z in self._chunks(app, self.before)]
        before = [row for chunk in before for row in chunk]
        if all(torch.equal(a, b) for a, b in zip(self.before, self.after)):
            after = before
        else:
            after = [z[:, ids].tolist() for z in self._chunks(app, self.after)]
            after = [row for chunk in after for row in chunk]
        text = (
            response
            if response is not None
            else app.processor.decode(
                self.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        )
        spans, alignment = decoded_spans(app, self.token_ids, text)
        return {
            "token_ids": self.token_ids,
            "tokens": app.processor.tokenizer.convert_ids_to_tokens(self.token_ids),
            "before": before,
            "after": after,
            "text_spans": spans,
            "text_alignment": alignment,
            "forward_lengths": self.forward_lengths,
            "termination": "EOS"
            if self.ended_at_eos
            else (
                "Token limit" if len(self.token_ids) >= max_new_tokens else "Other stop"
            ),
        }
