"""Manual A/B inputs use the same capture contract as JSON manifests."""

import hashlib
import io
import json
from PIL import Image
from .manifests import Condition, Manifest, Pair, MAX_PAIRS, MAX_MANIFEST_BYTES

MANUAL_NAME = "Manual pairs"


def manual_condition(text, image):
    text = text or ""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("Condition text must be a string smaller than 5 MB.")
    if image is not None and not isinstance(image, Image.Image):
        raise ValueError("Upload an image using the image field.")
    if not text.strip() and image is None:
        raise ValueError("Each A/B condition needs text, an image, or both.")
    if image is None:
        return Condition(text)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    data = buffer.getvalue()
    return Condition(
        text, image_sha256=hashlib.sha256(data).hexdigest(), image_bytes=data
    )


def combine_inputs(manifests, pairs):
    """Validate before committing changes, so invalid edits leave the session intact."""
    manifests = tuple(manifests)
    if sum(len(m.pairs) for m in manifests) + len(pairs) > MAX_PAIRS:
        raise ValueError(
            f"Use at most {MAX_PAIRS} pairs across manifests and manual input."
        )
    if not pairs:
        return manifests
    if any(m.name == MANUAL_NAME for m in manifests):
        raise ValueError(
            'The name "Manual pairs" is reserved while using manual input. Rename that JSON manifest.'
        )
    digest = hashlib.sha256(
        json.dumps([p.metadata() for p in pairs], sort_keys=True).encode()
    ).hexdigest()
    return (*manifests, Manifest(MANUAL_NAME, tuple(pairs), digest))


def change_pair(
    session, action, selected=None, a_text="", a_image=None, b_text="", b_image=None
):
    if action not in {"add", "update", "remove"}:
        raise ValueError("Unknown manual input action.")
    # Snapshot uploads before taking the session lock. Never retain mutable PIL inputs.
    a, b = (
        (manual_condition(a_text, a_image), manual_condition(b_text, b_image))
        if action != "remove"
        else (None, None)
    )
    with session.lock:
        pairs = list(session.manual_pairs)
        if action == "add":
            pair_id = f"Pair {session.manual_next_id}"
            pairs.append(Pair(pair_id, a, b))
        else:
            index = next((i for i, p in enumerate(pairs) if p.id == selected), None)
            if index is None:
                raise ValueError("Select an existing manual pair first.")
            if action == "update":
                pairs[index] = Pair(selected, a, b)
            else:
                pairs.pop(index)
        loaded = tuple(
            m
            for m in session.manifests
            if not (session.manual_pairs and m.name == MANUAL_NAME)
        )
        combined = combine_inputs(loaded, pairs)
        session.manifests, session.manual_pairs = combined, tuple(pairs)
        if action == "add":
            session.manual_next_id += 1
        session.invalidate_profile()
        return [[m.name, len(m.pairs), m.fingerprint[:12]] for m in combined]


def read_pair(session, selected):
    with session.lock:
        pair = next((p for p in session.manual_pairs if p.id == selected), None)
        if pair is None:
            raise ValueError("Select an existing manual pair first.")
        return pair.a.text, pair.a.load_image(), pair.b.text, pair.b.load_image()
