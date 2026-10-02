"""Previewable samples and base-gated evaluation, independent of the web app."""

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


@dataclass
class PreparedBenchmark:
    adapter: BenchmarkAdapter
    request: BenchmarkRequest
    items: list[dict]
    dataset_fingerprint: str | None


def prepare_benchmark(adapter, request):
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
    items = []
    for index in indices[: request.max_items]:
        item = dataset[index]
        # Validate answer availability before any inference (MMMU test labels are hidden).
        reference = adapter.reference(item)
        items.append(
            {
                "index": index,
                "raw": item,
                **adapter.metadata(item),
                "prompt": adapter.prompt(item),
                "images": adapter.images(item),
                "reference": reference,
            }
        )
    return PreparedBenchmark(
        adapter, request, items, getattr(dataset, "_fingerprint", None)
    )


def evaluate_base(prepared, generate: Callable, progress=None):
    rows = []
    for index, item in enumerate(prepared.items):
        answer, cache_hit = generate(item["images"], item["prompt"])
        rows.append(
            {k: v for k, v in item.items() if k not in {"raw", "images"}}
            | {
                "base": answer,
                "base_cached": cache_hit,
                "base_score": prepared.adapter.details(item["raw"], answer),
                "steered": None,
                "steered_score": None,
                "changed": None,
            }
        )
        if progress:
            progress(
                (index + 1) / len(prepared.items),
                desc=f"Base: {index + 1}/{len(prepared.items)}",
            )
    return rows


def evaluate_steered(prepared, base_rows, generate: Callable, progress=None):
    if len(base_rows) != len(prepared.items):
        raise ValueError("Evaluate the base model on this sample first.")
    rows = []
    for index, (item, base) in enumerate(zip(prepared.items, base_rows)):
        row = dict(base, steered=None, steered_score=None, changed=None)
        if base["base_score"]["correct"]:
            answer = generate(item["images"], item["prompt"])
            row.update(
                steered=answer,
                steered_score=prepared.adapter.details(item["raw"], answer),
                changed=prepared.adapter.normalize(item["raw"], base["base"])
                != prepared.adapter.normalize(item["raw"], answer),
            )
        rows.append(row)
        if progress:
            progress(
                (index + 1) / len(prepared.items),
                desc=f"Steered: {index + 1}/{len(prepared.items)} (base failures skipped)",
            )
    return summarize(prepared, rows)


def summarize(prepared, rows):
    eligible = [r for r in rows if r["base_score"]["correct"]]
    evaluated = [r for r in eligible if r["steered_score"] is not None]
    preserved = sum(r["steered_score"]["correct"] for r in evaluated)
    summary = {
        "num_items": len(rows),
        "base_correct": len(eligible),
        "base_failed_excluded": len(rows) - len(eligible),
        "base_accuracy_all_items": len(eligible) / len(rows),
        "base_cache_hits": sum(r["base_cached"] for r in rows),
        "steered_evaluated": len(evaluated),
        "preserved": preserved,
        "regressed": len(evaluated) - preserved,
        "preservation_rate_on_base_correct": preserved / len(evaluated)
        if evaluated
        else None,
    }
    if prepared.adapter.name == "IFEval":
        for condition, subset in (("base", rows), ("steered", evaluated)):
            scores = [r[f"{condition}_score"] for r in subset]
            summary[f"{condition}_prompt_loose_accuracy"] = (
                sum(s["loose_correct"] for s in scores) / len(scores)
                if scores
                else None
            )
            for mode in ("strict", "loose"):
                checks = [v for s in scores for v in s[f"{mode}_instructions"]]
                summary[f"{condition}_instruction_{mode}_accuracy"] = (
                    sum(checks) / len(checks) if checks else None
                )
    return {
        "benchmark": prepared.adapter.name,
        "dataset_id": prepared.adapter.dataset_id,
        "dataset_fingerprint": prepared.dataset_fingerprint,
        "split": prepared.request.split,
        "category": prepared.request.category,
        "sample_seed": prepared.request.seed,
        "metric": prepared.adapter.metric,
        "protocol": "zero-shot; steered evaluated only on base-correct items",
        "summary": summary,
        "rows": rows,
    }


def run_benchmark(adapter, request, generate_base, generate_steered, progress=None):
    prepared = prepare_benchmark(adapter, request)
    base = evaluate_base(prepared, generate_base, progress)
    return evaluate_steered(prepared, base, generate_steered, progress)
