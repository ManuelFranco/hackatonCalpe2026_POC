"""Calibration inputs: confined JSON assets and session-owned manual images."""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
from PIL import Image

DATA_ROOT = Path(
    os.getenv("GEMMA_DATA_ROOT", Path(__file__).resolve().parents[1] / "data")
).resolve()
MAX_MANIFEST_BYTES = 5_000_000
MAX_PAIRS = 500


@dataclass(frozen=True)
class Condition:
    text: str
    image: str | None = None
    image_sha256: str | None = None
    image_bytes: bytes | None = field(default=None, repr=False)

    def metadata(self):
        result = {
            "text": self.text,
            "image": self.image,
            "image_sha256": self.image_sha256,
        }
        if self.image_bytes is not None:
            result["image_source"] = "manual upload (session memory)"
        return result

    def load_image(self):
        if self.image_bytes is not None:
            with Image.open(io.BytesIO(self.image_bytes)) as image:
                return image.convert("RGB")
        if self.image is None:
            return None
        path = Path(self.image)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.image_sha256:
            raise ValueError("A manifest image changed on disk. Reload the manifest.")
        with Image.open(path) as image:
            return image.convert("RGB")


@dataclass(frozen=True)
class Pair:
    id: str
    a: Condition
    b: Condition

    def metadata(self):
        return {"id": self.id, "a": self.a.metadata(), "b": self.b.metadata()}


@dataclass(frozen=True)
class Manifest:
    name: str
    pairs: tuple[Pair, ...]
    fingerprint: str

    def metadata(self):
        return {
            "name": self.name,
            "pairs": [p.metadata() for p in self.pairs],
            "fingerprint": self.fingerprint,
        }


def _asset(value: str, base: Path) -> Path:
    if not isinstance(value, str):
        raise ValueError("Asset paths must be strings.")
    path = (base / value).resolve()
    if not path.is_relative_to(DATA_ROOT) or not path.is_file():
        raise ValueError(
            f"Asset must be a file inside the configured data root: {value}"
        )
    return path


def _condition(raw: dict, base: Path) -> Condition:
    if not isinstance(raw, dict):
        raise ValueError("Each condition must be an object with text and/or image.")
    text = raw.get("text") or ""
    if not isinstance(text, str):
        raise ValueError("Condition text must be a string.")
    if raw.get("text_file"):
        if text.strip():
            raise ValueError("Use text or text_file, not both.")
        file = _asset(raw["text_file"], base)
        if file.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Text assets must be smaller than 5 MB.")
        text = file.read_text(encoding="utf-8")
    image = _asset(raw["image"], base) if raw.get("image") else None
    if not text.strip() and image is None:
        raise ValueError("Each A/B condition needs text, an image, or both.")
    if image:
        with Image.open(image) as im:
            im.verify()
    return Condition(
        text,
        str(image) if image else None,
        hashlib.sha256(image.read_bytes()).hexdigest() if image else None,
    )


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest must be smaller than 5 MB.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Legacy CWE manifests are kept readable for collaborators' existing datasets.
    if isinstance(raw, list):
        raw = {
            "version": 1,
            "name": "CWE code pairs",
            "asset_root": "cwe_c_pairs_dataset",
            "pairs": [
                {
                    "id": item["cwe_id"],
                    "A": {"text_file": item["normal_file"]},
                    "B": {"text_file": item["vulnerable_file"]},
                }
                for item in raw
            ],
        }
    # Per-CWE manifests from the current v2 dataset use safe/vulnerable file mappings.
    if (
        isinstance(raw, dict)
        and "cwe_id" in raw
        and "pairs" in raw
        and "version" not in raw
    ):
        cwe = raw["cwe_id"]
        if (
            not isinstance(cwe, str)
            or not cwe.startswith("CWE-")
            or not cwe[4:].isdigit()
        ):
            raise ValueError("Invalid CWE identifier.")
        asset_root = raw.get("asset_root")
        if asset_root is None:
            if path.resolve().is_relative_to(DATA_ROOT):
                asset_root = str(path.resolve().parent.relative_to(DATA_ROOT))
            else:
                folder = "code_lines" if raw.get("line_only") else "scripts"
                candidate = f"{folder}/cwe_{cwe[4:]}"
                asset_root = (
                    candidate if (DATA_ROOT / candidate).is_dir() else f"cwe_{cwe[4:]}"
                )
        suffix = " · code lines" if raw.get("line_only") else ""
        raw = {
            "version": 1,
            "name": f"{cwe} · {raw.get('title', 'Code safety')}{suffix}",
            "asset_root": asset_root,
            "pairs": [
                {
                    "id": str(item["pair"]),
                    "A": {"text_file": item["safe_file"]},
                    "B": {"text_file": item["vulnerable_file"]},
                }
                for item in raw["pairs"]
            ],
        }
    if isinstance(raw, dict) and "cwes" in raw:
        raise ValueError(
            "This JSON is a dataset index. Choose a per-CWE manifest instead."
        )
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Expected a version 1 manifest object.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Manifest name is required.")
    # Explicit asset_root is relative to DATA_ROOT; otherwise repository manifests
    # use their own directory and uploaded JSONs use DATA_ROOT.
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
    entries = raw.get("pairs")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_PAIRS:
        raise ValueError(f"A manifest requires 1–{MAX_PAIRS} A/B pairs.")
    pairs = []
    for index, item in enumerate(entries, 1):
        if not isinstance(item, dict) or "A" not in item or "B" not in item:
            raise ValueError(f"Pair {index} requires A and B conditions.")
        pair_id = str(item.get("id", index))
        pairs.append(
            Pair(pair_id, _condition(item["A"], base), _condition(item["B"], base))
        )
    if len({p.id for p in pairs}) != len(pairs):
        raise ValueError("Pair IDs must be unique within a manifest.")
    digest = hashlib.sha256(
        json.dumps([p.metadata() for p in pairs], sort_keys=True).encode()
    ).hexdigest()
    return Manifest(name.strip(), tuple(pairs), digest)


def load_manifests(paths: list[str]) -> tuple[Manifest, ...]:
    if not paths:
        raise ValueError("Select or upload at least one JSON manifest.")
    result = tuple(load_manifest(path) for path in paths)
    if len({m.name for m in result}) != len(result):
        raise ValueError("Use a unique name for every manifest.")
    if sum(len(m.pairs) for m in result) > MAX_PAIRS:
        raise ValueError(f"Load at most {MAX_PAIRS} pairs in one session.")
    return result
