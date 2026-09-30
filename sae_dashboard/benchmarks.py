"""Generic paired BASE versus STEERED capability benchmarks."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


CHOICES = "ABCDEFGHIJ"


def _load_dataset(path: str, config: Optional[str], split: str):
    from datasets import load_dataset

    if config:
        return load_dataset(path, config, split=split)
    return load_dataset(path, split=split)


def _last_choice(text: str, choices: str = CHOICES) -> Optional[str]:
    matches = re.findall(rf"\b([{re.escape(choices)}])\b", (text or "").upper())
    return matches[-1] if matches else None


def _last_number(text: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text or "")
    return matches[-1].replace(",", "") if matches else None


class BenchmarkAdapter:
    """Dataset-specific loading, prompting, scoring, and metadata."""

    name: str
    key: str

    def load(self, split: str):
        raise NotImplementedError

    def format_prompt(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        raise NotImplementedError

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return (answer or "").strip().upper() or None

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class MMLUProBenchmark(BenchmarkAdapter):
    name = "MMLU-Pro"
    key = "mmlu_pro"

    def load(self, split: str):
        return _load_dataset("TIGER-Lab/MMLU-Pro", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        lines = [
            "Answer the following multiple-choice question.",
            "Return ONLY the letter of the correct answer.",
            "",
            f"Question: {item['question']}",
            "",
            "Options:",
        ]
        lines.extend(f"{letter}. {option}" for letter, option in zip(CHOICES, item["options"]))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, CHOICES[: len(item["options"])])

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == str(item["answer"]).strip().upper()

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("question_id"), "category": item.get("category")}


class GSM8KBenchmark(BenchmarkAdapter):
    name = "GSM8K"
    key = "gsm8k"

    def load(self, split: str):
        return _load_dataset("openai/gsm8k", "main", split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            "Solve the following math word problem. Show your reasoning, then "
            "give the final numeric answer after '####'.\n\n"
            f"Problem: {item['question']}\n\nAnswer:"
        )

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_number(answer)

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        expected = _last_number(str(item["answer"]))
        return self.normalize_answer(item, answer) == expected

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("id")}


class ARCChallengeBenchmark(BenchmarkAdapter):
    name = "ARC-Challenge"
    key = "arc_challenge"

    def load(self, split: str):
        return _load_dataset("allenai/ai2_arc", "ARC-Challenge", split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        question = item["question"]
        choices = item["choices"]
        lines = [
            "Answer the following multiple-choice science question.",
            "Return ONLY the letter of the correct answer.",
            "",
            f"Question: {question}",
            "",
            "Options:",
        ]
        lines.extend(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, "ABCDE")

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == str(item["answerKey"]).strip().upper()

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("id")}


BENCHMARKS: Dict[str, BenchmarkAdapter] = {
    adapter.key: adapter
    for adapter in (MMLUProBenchmark(), GSM8KBenchmark(), ARCChallengeBenchmark())
}


@dataclass
class BenchmarkRun:
    benchmark: str
    split: str
    total: int
    base_correct: int
    steered_correct: int
    preserved: int
    regressed: int
    improved: int
    changed: int
    rows: List[Dict[str, Any]]
    categories: Dict[str, Dict[str, int]]

    def as_dict(self) -> Dict[str, Any]:
        total = self.total
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "num_items": total,
            "base_correct": self.base_correct,
            "steered_correct": self.steered_correct,
            "base_accuracy": self.base_correct / total if total else 0.0,
            "steered_accuracy": self.steered_correct / total if total else 0.0,
            "accuracy_delta": (self.steered_correct - self.base_correct) / total if total else 0.0,
            "preserved": self.preserved,
            "regressed": self.regressed,
            "improved": self.improved,
            "changed": self.changed,
            "preservation_rate": self.preserved / self.base_correct if self.base_correct else None,
            "regression_rate": self.regressed / self.base_correct if self.base_correct else None,
            "change_rate": self.changed / total if total else 0.0,
            "categories": self.categories,
            "rows": self.rows,
        }


class BenchmarkRunner:
    """Run any registered benchmark through the app's generation functions."""

    def __init__(
        self,
        prepare_inputs: Callable[..., Any],
        generate_answer: Callable[..., str],
    ) -> None:
        self.prepare_inputs = prepare_inputs
        self.generate_answer = generate_answer

    def run(
        self,
        benchmark: str,
        split: str = "test",
        max_items: Optional[int] = 100,
        category: Optional[str] = None,
        directions: Optional[Dict[int, Any]] = None,
        strengths: Optional[Dict[int, float]] = None,
        temperature: float = 0.0,
        seed: int = 0,
        max_new_tokens: int = 32,
    ) -> BenchmarkRun:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        adapter = BENCHMARKS[benchmark]
        dataset = adapter.load(split)
        if category:
            dataset = [item for item in dataset if item.get("category") == category]
        if max_items is not None:
            dataset = dataset.select(range(min(int(max_items), len(dataset)))) if hasattr(dataset, "select") else list(dataset)[: int(max_items)]

        rows: List[Dict[str, Any]] = []
        categories: Dict[str, Dict[str, int]] = {}
        for item in dataset:
            prompt = adapter.format_prompt(item)
            base_inputs, _, base_input_len = self.prepare_inputs(None, prompt, add_generation_prompt=True)
            steered_inputs, _, steered_input_len = self.prepare_inputs(None, prompt, add_generation_prompt=True)
            base_answer = self.generate_answer(
                inputs=base_inputs, input_len=base_input_len, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=seed,
            )
            steered_answer = self.generate_answer(
                inputs=steered_inputs, input_len=steered_input_len, max_new_tokens=max_new_tokens,
                temperature=temperature, steering_directions=directions, strengths=strengths,
                seed=seed,
            )
            base_normalized = adapter.normalize_answer(item, base_answer)
            steered_normalized = adapter.normalize_answer(item, steered_answer)
            base_ok = adapter.score(item, base_answer)
            steered_ok = adapter.score(item, steered_answer)
            group = str(item.get("category", item.get("task", "all")))
            stats = categories.setdefault(group, {"total": 0, "base_correct": 0, "steered_correct": 0})
            stats["total"] += 1
            stats["base_correct"] += int(base_ok)
            stats["steered_correct"] += int(steered_ok)
            rows.append({
                **adapter.metadata(item), "gold": str(item.get("answer", item.get("answerKey", ""))),
                "base": base_normalized, "steered": steered_normalized,
                "base_correct": base_ok, "steered_correct": steered_ok,
                "changed": base_normalized != steered_normalized,
            })

        base_correct = sum(int(row["base_correct"]) for row in rows)
        steered_correct = sum(int(row["steered_correct"]) for row in rows)
        preserved = sum(int(row["base_correct"] and row["steered_correct"]) for row in rows)
        regressed = sum(int(row["base_correct"] and not row["steered_correct"]) for row in rows)
        improved = sum(int(not row["base_correct"] and row["steered_correct"]) for row in rows)
        changed = sum(int(row["changed"]) for row in rows)
        return BenchmarkRun(
            adapter.name, split, len(rows), base_correct, steered_correct,
            preserved, regressed, improved, changed, rows, categories,
        )


def save_benchmark_run(result: BenchmarkRun, root: Path, metadata: Dict[str, Any]) -> str:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = root / "benchmarks" / f"{result.benchmark.lower().replace('-', '_')}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, **result.as_dict(), "run_id": run_id}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str((run_dir / "results.json").resolve())