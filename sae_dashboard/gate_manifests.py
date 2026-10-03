"""Labelled image sets for calibrating a conditional-steering gate.

Kept separate from A/B steering manifests: loading a gate manifest never touches
the session's damage profile or vectors. Asset resolution reuses manifests.py, so
paths stay confined to GEMMA_DATA_ROOT and images are fingerprinted.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .manifests import DATA_ROOT, MAX_MANIFEST_BYTES, Condition, _condition

GATE_KIND = "binary_gate"
GATE_SPLITS = ("train", "validation", "test")
MAX_GATE_EXAMPLES = 2000
DEFAULT_GATE_PROMPT = "Describe the image."
LABELS = {"target": 1, "positive": 1, 1: 1, "non_target": 0, "negative": 0, 0: 0}


@dataclass(frozen=True)
class GateExample:
    id: str
    label: int  # 1 = target category present, 0 = absent
    category: str
    hard_negative: bool
    condition: Condition

    def metadata(self):
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "hard_negative": self.hard_negative,
            **self.condition.metadata(),
        }


@dataclass(frozen=True)
class GateManifest:
    name: str
    target: str
    split: str
    prompt: str
    examples: tuple[GateExample, ...]
    fingerprint: str

    @property
    def labels(self) -> list[int]:
        return [e.label for e in self.examples]

    def summary(self):
        positives = sum(self.labels)
        hard = sum(e.hard_negative for e in self.examples)
        return {
            "name": self.name,
            "target": self.target,
            "split": self.split,
            "examples": len(self.examples),
            "target_examples": positives,
            "non_target_examples": len(self.examples) - positives,
            "hard_negatives": hard,
            "fingerprint": self.fingerprint,
        }


def is_gate_manifest(raw) -> bool:
    return isinstance(raw, dict) and raw.get("kind") == GATE_KIND


def repository_gate_manifests() -> list[Path]:
    """Loadable gate manifests under the data root (templates are excluded)."""
    found = []
    for path in sorted(DATA_ROOT.rglob("*.json")):
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if is_gate_manifest(raw) and not raw.get("template"):
            found.append(path)
    return found


def load_gate_manifest(path: str | Path) -> GateManifest:
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Gate manifest must be smaller than 5 MB.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not is_gate_manifest(raw):
        raise ValueError(f'Expected a gate manifest with "kind": "{GATE_KIND}".')
    if raw.get("template"):
        raise ValueError(
            "This file is a format template. Copy it, add real images and remove "
            '"template": true.'
        )
    if raw.get("version") != 1:
        raise ValueError("Expected a version 1 gate manifest.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Gate manifest name is required.")
    target = raw.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Gate manifests must name their target category explicitly.")
    split = raw.get("split")
    if split not in GATE_SPLITS:
        raise ValueError(f"Gate manifest split must be one of {GATE_SPLITS}.")
    prompt = raw.get("prompt", DEFAULT_GATE_PROMPT)
    if not isinstance(prompt, str):
        raise ValueError("Gate prompt must be a string.")
    asset_root = raw.get("asset_root")
    if asset_root is not None and not isinstance(asset_root, str):
        raise ValueError("asset_root must be a string relative to the data root.")
    base = (
        (DATA_ROOT / asset_root).resolve()
        if asset_root is not None
        else (
            path.resolve().parent
            if path.resolve().is_relative_to(DATA_ROOT)
            else DATA_ROOT
        )
    )
    if not base.is_relative_to(DATA_ROOT):
        raise ValueError("asset_root must stay inside the configured data root.")
    entries = raw.get("examples")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_GATE_EXAMPLES:
        raise ValueError(f"A gate manifest requires 1–{MAX_GATE_EXAMPLES} examples.")
    examples = []
    for index, item in enumerate(entries, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Gate example {index} must be an object.")
        label = item.get("label")
        if isinstance(label, bool) or label not in LABELS:
            raise ValueError(
                f"Gate example {index}: label must be target/positive/1 or "
                "non_target/negative/0."
            )
        if not item.get("image"):
            raise ValueError(
                f"Gate example {index}: an image is required; the gate reads only "
                "visual tokens."
            )
        category = item.get("category", "")
        if not isinstance(category, str):
            raise ValueError(f"Gate example {index}: category must be a string.")
        hard = item.get("hard_negative", False)
        if not isinstance(hard, bool):
            raise ValueError(f"Gate example {index}: hard_negative must be boolean.")
        if hard and LABELS[label] == 1:
            raise ValueError(f"Gate example {index}: a target cannot be a hard negative.")
        condition = _condition({"text": prompt, "image": item["image"]}, base)
        examples.append(
            GateExample(
                str(item.get("id", index)), LABELS[label], category, hard, condition
            )
        )
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("Example IDs must be unique within a gate manifest.")
    digest = hashlib.sha256(
        json.dumps(
            {
                "target": target,
                "split": split,
                "prompt": prompt,
                "examples": [e.metadata() for e in examples],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return GateManifest(
        name.strip(), target.strip(), split, prompt, tuple(examples), digest
    )


def check_disjoint(*manifests: GateManifest) -> None:
    """Reject image reuse across splits (same file hash in train and validation)."""
    seen: dict[str, str] = {}
    for manifest in manifests:
        for example in manifest.examples:
            digest = example.condition.image_sha256
            if digest in seen and seen[digest] != manifest.split:
                raise ValueError(
                    f"Image of example {example.id} appears in both the "
                    f"{seen[digest]} and {manifest.split} gate splits."
                )
            seen.setdefault(digest, manifest.split)


def reserved_evaluation_hashes() -> dict[str, str]:
    """Image hashes from repository A/B manifests marked as validation or test splits.

    Gate calibration must not silently consume images held out for the damage study.
    """
    from .manifests import load_manifest

    reserved: dict[str, str] = {}
    for path in sorted(DATA_ROOT.rglob("*manifest*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict) or raw.get("split") not in ("validation", "test"):
            continue
        try:
            manifest = load_manifest(path)
        except (OSError, ValueError, KeyError):
            continue
        label = f"{path.relative_to(DATA_ROOT)} ({raw['split']})"
        for pair in manifest.pairs:
            for condition in (pair.a, pair.b):
                if condition.image_sha256:
                    reserved.setdefault(condition.image_sha256, label)
    return reserved


def check_not_reserved(manifest: GateManifest, reserved: dict[str, str]) -> None:
    for example in manifest.examples:
        source = reserved.get(example.condition.image_sha256)
        if source:
            raise ValueError(
                f"Gate {manifest.split} example {example.id} reuses an image reserved "
                f"for evaluation in {source}."
            )
