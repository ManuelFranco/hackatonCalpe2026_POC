"""Paired benchmark engine: inject a generator; never import the web application."""

from dataclasses import dataclass
import random
from typing import Callable
from .adapters import BenchmarkAdapter


@dataclass(frozen=True)
class BenchmarkRequest:
    split: str
    max_items: int = 20
    category: str = ""
    seed: int = 0


def run_benchmark(
    adapter: BenchmarkAdapter,
    request: BenchmarkRequest,
    generate_pair: Callable,
    progress: Callable | None = None,
) -> dict:
    if request.split not in adapter.splits:
        raise ValueError(
            f"Choose a supported split for {adapter.name}: {adapter.splits}"
        )
    if not 1 <= request.max_items <= 10000:
        raise ValueError("Choose between 1 and 10000 benchmark items.")
    adapter.preflight()
    dataset = adapter.load(request.split, request.category.strip())
    if not len(dataset):
        raise ValueError("No benchmark items match this subject and split.")
    indices = list(range(len(dataset)))
    random.Random(request.seed).shuffle(indices)
    indices = indices[: request.max_items]
    rows = []
    for offset, index in enumerate(indices):
        item = dataset[index]
        prompt, images = adapter.prompt(item), adapter.images(item)
        base, steered = generate_pair(images, prompt)
        base_score, steered_score = (
            adapter.details(item, base),
            adapter.details(item, steered),
        )
        rows.append(
            {
                "index": index,
                **adapter.metadata(item),
                "prompt": prompt,
                "base": base,
                "steered": steered,
                "base_score": base_score,
                "steered_score": steered_score,
                "changed": adapter.normalize(item, base)
                != adapter.normalize(item, steered),
            }
        )
        if progress:
            progress(
                (offset + 1) / len(indices),
                desc=f"{adapter.name}: {offset + 1}/{len(indices)}",
            )
    base_correct = sum(r["base_score"]["correct"] for r in rows)
    steered_correct = sum(r["steered_score"]["correct"] for r in rows)
    preserved = sum(
        r["base_score"]["correct"] and r["steered_score"]["correct"] for r in rows
    )
    summary = {
        "num_items": len(rows),
        "base_accuracy": base_correct / len(rows),
        "steered_accuracy": steered_correct / len(rows),
        "accuracy_delta": (steered_correct - base_correct) / len(rows),
        "preserved": preserved,
        "regressed": base_correct - preserved,
        "improved": steered_correct - preserved,
        "changed": sum(r["changed"] for r in rows),
        "preservation_rate": preserved / base_correct if base_correct else None,
    }
    if adapter.name == "IFEval":
        for condition in ("base", "steered"):
            scores = [r[f"{condition}_score"] for r in rows]
            summary[f"{condition}_prompt_loose_accuracy"] = sum(
                s["loose_correct"] for s in scores
            ) / len(scores)
            for mode in ("strict", "loose"):
                checks = [v for s in scores for v in s[f"{mode}_instructions"]]
                summary[f"{condition}_instruction_{mode}_accuracy"] = sum(checks) / len(
                    checks
                )
    return {
        "benchmark": adapter.name,
        "dataset_id": adapter.dataset_id,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "split": request.split,
        "category": request.category,
        "sample_seed": request.seed,
        "metric": adapter.metric,
        "protocol": "paired zero-shot dashboard evaluation",
        "summary": summary,
        "rows": rows,
    }
