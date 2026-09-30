"""Generic paired BASE versus STEERED capability benchmarks."""

from __future__ import annotations

import ast
import json
import re
import tempfile
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


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .\n\t")


def _answer_from_text(text: str) -> Optional[str]:
    return _normalize_text(text) or None


def _options(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(option) for option in parsed]
    return [str(option) for option in value]


def _image_from_item(item: Dict[str, Any]) -> Any:
    for key in ("image", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"):
        if item.get(key) is not None:
            return item[key]
    return None


def _materialize_image(image: Any, directory: str, index: int) -> Optional[str]:
    if image is None:
        return None
    if isinstance(image, (str, Path)):
        return str(image)
    path = Path(directory) / f"benchmark_image_{index:05d}.png"
    if isinstance(image, bytes):
        path.write_bytes(image)
    elif hasattr(image, "save"):
        image.save(path)
    else:
        raise TypeError(f"Unsupported benchmark image type: {type(image)!r}")
    return str(path)


class BenchmarkAdapter:
    """Dataset-specific loading, prompting, scoring, and metadata."""

    name: str
    key: str
    supported_splits: tuple[str, ...] = ("test",)

    uses_category_as_config = False

    def load(self, split: str, category: Optional[str] = None):
        raise NotImplementedError

    def format_prompt(self, item: Dict[str, Any]) -> str:
        raise NotImplementedError

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        raise NotImplementedError

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return (answer or "").strip().upper() or None

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def image(self, item: Dict[str, Any]) -> Any:
        return None


class MMLUProBenchmark(BenchmarkAdapter):
    name = "MMLU-Pro"
    key = "mmlu_pro"
    supported_splits = ("validation", "test")

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("TIGER-Lab/MMLU-Pro", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        options = _options(item["options"])
        lines = [
            "Answer the following multiple-choice question.",
            "Return ONLY the letter of the correct answer.",
            "",
            f"Question: {item['question']}",
            "",
            "Options:",
        ]
        lines.extend(f"{letter}. {option}" for letter, option in zip(CHOICES, options))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, CHOICES[: len(_options(item["options"]))])

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == str(item["answer"]).strip().upper()

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("question_id"), "category": item.get("category")}


class GSM8KBenchmark(BenchmarkAdapter):
    name = "GSM8K"
    key = "gsm8k"
    supported_splits = ("train", "test")

    def load(self, split: str, category: Optional[str] = None):
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
    supported_splits = ("train", "validation", "test")

    def load(self, split: str, category: Optional[str] = None):
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


class BBHBenchmark(BenchmarkAdapter):
    name = "BBH"
    key = "bbh"
    uses_category_as_config = True

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("lukaemon/bbh", category or "boolean_expressions", split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            "Solve the following BIG-Bench Hard task. Return only the final answer.\n\n"
            f"Input: {item['input']}\n\nAnswer:"
        )

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _answer_from_text(answer)

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == _normalize_text(item["target"])

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("id")}


class TruthfulQABenchmark(BenchmarkAdapter):
    name = "TruthfulQA"
    key = "truthfulqa"
    supported_splits = ("validation",)

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("truthfulqa/truthful_qa", "multiple_choice", split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        targets = item["mc1_targets"]
        lines = [
            "Answer the following question truthfully.",
            "Return ONLY the letter of the best answer.",
            "",
            f"Question: {item['question']}",
            "",
            "Options:",
        ]
        lines.extend(f"{letter}. {choice}" for letter, choice in zip(CHOICES, targets["choices"]))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, CHOICES[: len(item["mc1_targets"]["choices"])])

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        choice = self.normalize_answer(item, answer)
        if choice is None:
            return False
        index = CHOICES.index(choice)
        return bool(item["mc1_targets"]["labels"][index])

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"category": item.get("category"), "type": item.get("type")}


class BoolQBenchmark(BenchmarkAdapter):
    name = "BoolQ"
    key = "boolq"
    supported_splits = ("train", "validation")

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("google/boolq", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            "Answer the question using the passage. Return ONLY YES or NO.\n\n"
            f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n\nAnswer:"
        )

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        match = re.search(r"\b(YES|NO)\b", (answer or "").upper())
        return match.group(1) if match else None

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        expected = "YES" if item["answer"] else "NO"
        return self.normalize_answer(item, answer) == expected


class PIQABenchmark(BenchmarkAdapter):
    name = "PIQA"
    key = "piqa"
    supported_splits = ("train", "validation")

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("baber/piqa", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            "Choose the more physically plausible solution. Return ONLY the letter.\n\n"
            f"Goal: {item['goal']}\n"
            f"A. {item['sol1']}\n"
            f"B. {item['sol2']}\n\nAnswer:"
        )

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, "AB")

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == ("A" if int(item["label"]) == 0 else "B")


class MMMUBenchmark(BenchmarkAdapter):
    name = "MMMU"
    key = "mmmu"
    uses_category_as_config = True
    supported_splits = ("dev", "validation", "test")

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("MMMU/MMMU", category or "Accounting", split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        options = _options(item["options"])
        lines = [
            "Answer the following multimodal multiple-choice question.",
            "Return ONLY the letter of the correct answer.",
            "",
            f"Question: {item['question']}",
            "",
            "Options:",
        ]
        lines.extend(f"{letter}. {option}" for letter, option in zip(CHOICES, options))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, CHOICES[: len(_options(item["options"]))])

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == str(item["answer"]).strip().upper()

    def image(self, item: Dict[str, Any]) -> Any:
        return _image_from_item(item)

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("id"), "subject": item.get("subject")}


class ScienceQABenchmark(BenchmarkAdapter):
    name = "ScienceQA"
    key = "scienceqa"
    supported_splits = ("train", "validation", "test")

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("derek-thomas/ScienceQA", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        lines = [
            "Answer the following multimodal science question.",
            "Return ONLY the letter of the correct answer.",
            "",
            f"Question: {item['question']}",
            "",
            "Options:",
        ]
        lines.extend(f"{letter}. {option}" for letter, option in zip(CHOICES, item["choices"]))
        lines.extend(["", "Answer:"])
        return "\n".join(lines)

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        return _last_choice(answer, CHOICES[: len(item["choices"])])

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == CHOICES[int(item["answer"])]

    def image(self, item: Dict[str, Any]) -> Any:
        return _image_from_item(item)

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("id"), "subject": item.get("subject"), "task": item.get("task")}


class POPEBenchmark(BenchmarkAdapter):
    name = "POPE"
    key = "pope"
    supported_splits = ("test",)

    def load(self, split: str, category: Optional[str] = None):
        return _load_dataset("lmms-lab-encoder/POPE", None, split)

    def format_prompt(self, item: Dict[str, Any]) -> str:
        return (
            "Answer the visual question using only the image. Return ONLY YES or NO.\n\n"
            f"Question: {item['question']}\n\nAnswer:"
        )

    def normalize_answer(self, item: Dict[str, Any], answer: str) -> Optional[str]:
        match = re.search(r"\b(YES|NO)\b", (answer or "").upper())
        return match.group(1) if match else None

    def score(self, item: Dict[str, Any], answer: str) -> bool:
        return self.normalize_answer(item, answer) == str(item["answer"]).strip().upper()

    def image(self, item: Dict[str, Any]) -> Any:
        return _image_from_item(item)

    def metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"question_id": item.get("question_id", item.get("id")), "category": item.get("category")}


BENCHMARKS: Dict[str, BenchmarkAdapter] = {
    adapter.key: adapter
    for adapter in (
        MMLUProBenchmark(), GSM8KBenchmark(), ARCChallengeBenchmark(), BBHBenchmark(),
        TruthfulQABenchmark(), MMMUBenchmark(), ScienceQABenchmark(), POPEBenchmark(),
        BoolQBenchmark(), PIQABenchmark(),
    )
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
        if split not in adapter.supported_splits:
            supported = ", ".join(adapter.supported_splits)
            raise ValueError(f"{adapter.name} does not provide split '{split}'. Use: {supported}.")
        dataset = adapter.load(split, category)
        if category and not adapter.uses_category_as_config:
            dataset = [item for item in dataset if item.get("category") == category]
        if max_items is not None:
            dataset = dataset.select(range(min(int(max_items), len(dataset)))) if hasattr(dataset, "select") else list(dataset)[: int(max_items)]

        rows: List[Dict[str, Any]] = []
        categories: Dict[str, Dict[str, int]] = {}
        with tempfile.TemporaryDirectory(prefix="benchmark_") as image_directory:
            for index, item in enumerate(dataset):
                prompt = adapter.format_prompt(item)
                image_path = _materialize_image(adapter.image(item), image_directory, index)
                base_inputs, _, base_input_len = self.prepare_inputs(image_path, prompt, add_generation_prompt=True)
                steered_inputs, _, steered_input_len = self.prepare_inputs(image_path, prompt, add_generation_prompt=True)
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
                group = str(item.get("category", item.get("task", category or "all")))
                stats = categories.setdefault(group, {"total": 0, "base_correct": 0, "steered_correct": 0})
                stats["total"] += 1
                stats["base_correct"] += int(base_ok)
                stats["steered_correct"] += int(steered_ok)
                rows.append({
                    **adapter.metadata(item), "gold": str(item.get("answer", item.get("answerKey", item.get("label", "")))),
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