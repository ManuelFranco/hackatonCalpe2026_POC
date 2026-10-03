"""Portable benchmark-base artifacts shared by the offline runner and dashboard."""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from research.benchmarks import BENCHMARKS, BenchmarkRequest
from research.benchmarks.runner import PreparedBenchmark


FORMAT_VERSION = 1


def _encode(value: Any) -> Any:
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return {
            "__type__": "image",
            "format": "PNG",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported benchmark artifact value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "image":
        try:
            image = Image.open(io.BytesIO(base64.b64decode(value["data"])))
            return image.convert("RGB")
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError("Benchmark artifact contains an invalid image.") from exc
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def save_base_artifact(path, sample, rows, result, metadata=None):
    """Save the complete prepared sample and base evaluation as one JSON file."""
    payload = {
        "format": "gemma-sae-benchmark-base",
        "format_version": FORMAT_VERSION,
        "created_unix": time.time(),
        "metadata": _encode(metadata or {}),
        "sample": {
            "benchmark_key": next(
                key for key, adapter in BENCHMARKS.items() if adapter is sample.adapter
            ),
            "benchmark": sample.adapter.name,
            "dataset_id": sample.adapter.dataset_id,
            "dataset_fingerprint": sample.dataset_fingerprint,
            "request": _encode(
                {
                    "split": sample.request.split,
                    "max_items": sample.request.max_items,
                    "category": sample.request.category,
                    "seed": sample.request.seed,
                }
            ),
            "items": [
                {
                    "index": item["index"],
                    "raw": _encode(item["raw"]),
                    "id": item["id"],
                    "category": item["category"],
                    "question_type": item["question_type"],
                    "difficulty": item["difficulty"],
                    "prompt": item["prompt"],
                    "reference": item["reference"],
                    "images": [
                        {"label": label, "image": _encode(image)}
                        for label, image in item["images"]
                    ],
                }
                for item in sample.items
            ],
        },
        "generation": _encode(metadata or {}),
        "rows": _encode(rows),
        "result": _encode(result),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_base_artifact(path):
    """Load an offline base artifact into the objects expected by the dashboard."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark artifact: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "gemma-sae-benchmark-base"
        or payload.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("Unsupported benchmark artifact format.")
    try:
        sample_data = payload["sample"]
        adapter = BENCHMARKS[sample_data["benchmark_key"]]
        request_data = _decode(sample_data["request"])
        request = BenchmarkRequest(**request_data)
        items = []
        for saved in sample_data["items"]:
            raw = _decode(saved["raw"])
            item = {
                "index": saved["index"],
                "raw": raw,
                "id": saved["id"],
                "category": saved["category"],
                "question_type": saved["question_type"],
                "difficulty": saved["difficulty"],
                "prompt": saved["prompt"],
                "reference": saved["reference"],
                "images": [
                    (image["label"], _decode(image["image"]))
                    for image in saved["images"]
                ],
            }
            items.append(item)
        sample = PreparedBenchmark(
            adapter,
            request,
            items,
            sample_data.get("dataset_fingerprint"),
        )
        rows = _decode(payload["rows"])
        result = _decode(payload["result"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid benchmark artifact contents: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError("Benchmark artifact rows do not match its sample.")
    if not isinstance(result, dict) or not isinstance(result.get("summary"), dict):
        raise ValueError("Benchmark artifact has no valid evaluation summary.")
    return sample, rows, result, _decode(payload.get("metadata", {}))
