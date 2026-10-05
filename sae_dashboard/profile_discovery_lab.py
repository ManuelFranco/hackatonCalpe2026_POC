"""Discover from the common profile, then recapture just one unchanged A/B input."""

from dataclasses import asdict

import torch

from research.profile_discovery import (
    observe_candidates,
    rank_candidates,
    source_token_spans,
)
from . import model_runtime as runtime
from .artifact_store import record_event
from .workflow import require_profile


def pairs(session):
    return [
        (f"{manifest.name} / {pair.id}", pair)
        for manifest in session.manifests
        for pair in manifest.pairs
    ]


def candidates(session, settings, layer):
    require_profile(session, settings)
    if layer not in runtime.LAYERS or layer not in session.profile:
        raise ValueError("Choose a layer from the current common profile.")
    return rank_candidates(session.profile[layer])


def inspect_pair(session, settings, layer, pair_index, progress=None):
    with session.lock:
        rows = candidates(session, settings, layer)
        if not rows:
            raise ValueError("No nonzero, stable contrasts in this layer.")
        available = pairs(session)
        if not str(pair_index).isdigit() or not 0 <= int(pair_index) < len(available):
            raise ValueError("Choose an A/B pair from the current profile.")
        index = int(pair_index)
        profile = session.profile[layer]
        if len(available) != len(profile.a):
            raise ValueError("The loaded pairs no longer match this common profile.")
        label, pair = available[index]
        features = [row["feature"] for row in rows]
        conditions = []
        for name, condition in (("A", pair.a), ("B", pair.b)):
            # No generation and no steering. Release the shared model between inputs.
            with runtime.MODEL_LOCK:
                runtime.ensure_models_loaded(sae_layers=[layer])
                sae = runtime.saes[layer]
                if sae.W_dec.shape[0] != profile.a.shape[1]:
                    raise ValueError(
                        "The loaded SAE differs from the profile dictionary."
                    )
                image = condition.load_image()
                inputs, ids, _ = runtime.prepare_inputs(image, condition.text)
                image_mask = runtime.get_image_mask(ids)
                selected, detail = runtime.profile_token_selection(
                    ids, image_mask, condition.text, image, settings.token_scope
                )
                if progress:
                    progress(len(conditions) / 2, desc=f"Observing input {name}")
                residuals = runtime.capture_prompt_residuals(inputs)
                activity = observe_candidates(runtime, sae, residuals[layer], features)
                if len(activity) != len(ids) or selected.shape != ids.shape:
                    raise ValueError("Input token and activation counts differ.")
                measured = activity[selected]
                aggregate = (
                    measured.mean(0)
                    if settings.aggregation == "mean"
                    else measured.amax(0)
                )
                expected = (profile.a if name == "A" else profile.b)[index, features]
                if not torch.allclose(
                    aggregate, expected.float().cpu(), rtol=0.002, atol=1e-5
                ):
                    raise ValueError(
                        "Re-captured activations differ from this profile. Rebuild it."
                    )
                spans, alignment = (
                    [],
                    "Visual inputs use a token strip; image positions are omitted.",
                )
                if image is None and condition.text:
                    try:
                        rendered = runtime.processor.apply_chat_template(
                            runtime.make_messages(None, condition.text),
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        spans = source_token_spans(
                            runtime.processor.tokenizer, rendered, ids, condition.text
                        )
                        alignment = (
                            "Verified input-token offsets in the original source."
                        )
                    except ValueError as error:
                        alignment = f"Token strip: {error}"
                conditions.append(
                    {
                        "name": name,
                        "text": condition.text,
                        "tokens": runtime.processor.tokenizer.convert_ids_to_tokens(
                            ids.tolist()
                        ),
                        "token_ids": ids.tolist(),
                        "values": activity.tolist(),
                        "selected": selected.tolist(),
                        "image_mask": image_mask.tolist(),
                        "spans": spans,
                        "alignment": alignment,
                        "capture": detail,
                    }
                )
                del inputs, residuals, activity
        result = {
            "profile_id": session.profile_id,
            "layer": layer,
            "pair_index": index,
            "pair_label": label,
            "candidates": rows,
            "features": features,
            "settings": asdict(settings),
            "conditions": conditions,
            "model_id": runtime.MODEL_ID,
            "sae_id": runtime.SAE_IDS[layer],
        }
        record_event(session, "profile_discovery", result)
        if progress:
            progress(1, desc="Candidate locations ready")
        return result
