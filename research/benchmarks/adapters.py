"""Dataset loading and scoring; no UI, model globals, or experiment storage."""

from __future__ import annotations
import ast
import re
from typing import Any

CHOICES = "ABCDEFGHIJ"


def load_dataset(path, config, split):
    from datasets import load_dataset as hf_load

    return hf_load(path, config, split=split) if config else hf_load(path, split=split)


def options(item):
    value = item["options"]
    return ast.literal_eval(value) if isinstance(value, str) else value


def parse_choice(answer, count):
    # Never promote the article "a" or an unrelated letter in prose to an answer.
    choices = CHOICES[:count]
    exact = re.fullmatch(rf"\s*\(?([{choices}])\)?[.\s]*", answer)
    if exact:
        return exact.group(1)
    explicit = re.findall(rf"(?:answer\s*(?:is|:)\s*|\()([{choices}])\b", answer, re.I)
    return explicit[-1].upper() if explicit else None


class BenchmarkAdapter:
    name: str
    dataset_id: str
    splits: tuple[str, ...]
    default_split: str
    metric = "accuracy"
    supports_category = True

    def load(self, split, category):
        return load_dataset(self.dataset_id, None, split)

    def preflight(self):
        pass

    def images(self, item):
        return []

    def normalize(self, item, response):
        return response.strip()

    def details(self, item, response):
        return {"correct": self.score(item, response)}

    def metadata(self, item):
        return {
            "id": item.get("id", item.get("question_id", item.get("key"))),
            "category": item.get("category", item.get("subfield", "")),
            "question_type": item.get(
                "question_type",
                "multiple-choice" if "options" in item else "instruction-following",
            ),
            "difficulty": item.get("difficulty", item.get("topic_difficulty", "")),
        }


class MMLUPro(BenchmarkAdapter):
    name = "MMLU-Pro"
    dataset_id = "TIGER-Lab/MMLU-Pro"
    splits = ("validation", "test")
    default_split = "test"

    def load(self, split, category):
        data = super().load(split, category)
        return (
            data.filter(lambda row: row["category"] == category) if category else data
        )

    def prompt(self, item):
        choices = options(item)
        if not 2 <= len(choices) <= len(CHOICES):
            raise ValueError("Expected 2–10 multiple-choice options.")
        lines = ["Return only the letter of the correct answer.", item["question"]]
        lines.extend(f"{key}. {value}" for key, value in zip(CHOICES, choices))
        return "\n\n".join(lines)

    def normalize(self, item, response):
        return parse_choice(response, len(options(item)))

    def reference(self, item):
        answer = item.get("answer")
        choices = options(item)
        if not isinstance(answer, str) or answer not in CHOICES[: len(choices)]:
            raise ValueError(
                "This split has unavailable answer labels; select a labeled split."
            )
        return f"{answer}. {choices[CHOICES.index(answer)]}"

    def score(self, item, response):
        answer = item.get("answer")
        if not isinstance(answer, str) or answer not in CHOICES[: len(options(item))]:
            raise ValueError(
                "The selected dataset contains an unavailable answer label."
            )
        return self.normalize(item, response) == answer


class MMMU(MMLUPro):
    name = "MMMU"
    dataset_id = "MMMU/MMMU"
    splits = ("dev", "validation", "test")
    default_split = "validation"

    def load(self, split, category):
        # A blank category means all subjects, not an implicit Accounting subset.
        from datasets import concatenate_datasets, get_dataset_config_names

        subjects = [category] if category else get_dataset_config_names(self.dataset_id)
        return concatenate_datasets(
            [load_dataset(self.dataset_id, subject, split) for subject in subjects]
        )

    def prompt(self, item):
        if item.get("question_type") == "open":
            return "Give a concise answer to the question.\n\n" + item["question"]
        return super().prompt(item)

    def reference(self, item):
        if item.get("question_type") == "open":
            answer = item.get("answer")
            if not answer or answer == "?":
                raise ValueError(
                    "This MMMU split has no answer labels; select dev or validation."
                )
            return str(answer)
        return super().reference(item)

    def images(self, item):
        images = []
        for i in range(1, 8):
            value = item.get(f"image_{i}")
            if value is not None:
                images.append((f"Image {i}", value))
        return images

    def normalize(self, item, response):
        if item.get("question_type") == "open":
            from .vendor.mmmu_eval import parse_open_response

            return sorted(parse_open_response(response), key=str)
        return super().normalize(item, response)

    def score(self, item, response):
        if item.get("question_type") == "open":
            from .vendor.mmmu_eval import eval_open

            if not item.get("answer") or item["answer"] == "?":
                raise ValueError("The selected MMMU split has missing answer labels.")
            return eval_open(item["answer"], self.normalize(item, response))
        return super().score(item, response)


class POPE(BenchmarkAdapter):
    """Simple object-presence VQA with an explicit, strict yes/no protocol."""

    name = "POPE"
    dataset_id = "lmms-lab-encoder/POPE"
    splits = ("random", "popular", "adversarial")
    default_split = "random"
    metric = "strict yes/no accuracy"
    supports_category = False

    def load(self, split, category):
        if category:
            raise ValueError("POPE has no subject filter; choose a split instead.")
        # Full exposes the three variants separately; default/test mixes them.
        return load_dataset(self.dataset_id, "Full", split)

    def prompt(self, item):
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("POPE requires a non-empty question.")
        return (
            "Look at the image and answer the question. "
            "Return only yes or no.\n\n" + question.strip()
        )

    def images(self, item):
        image = item.get("image")
        if image is None:
            raise ValueError("POPE requires an image for every question.")
        return [("Image 1", image)]

    def reference(self, item):
        answer = item.get("answer")
        if not isinstance(answer, str) or answer.strip().lower() not in {"yes", "no"}:
            raise ValueError("POPE requires a labeled yes/no answer.")
        return answer.strip().lower()

    def normalize(self, item, response):
        # Do not infer yes from absence of 'no', or accept contradictory prose.
        match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", response, re.I)
        return match.group(1).lower() if match else None

    def score(self, item, response):
        return self.normalize(item, response) == self.reference(item)

    def details(self, item, response):
        prediction = self.normalize(item, response)
        return {
            "correct": prediction == self.reference(item),
            "prediction": prediction,
            "valid_answer": prediction is not None,
        }

    def metadata(self, item):
        return {
            **super().metadata(item),
            "question_type": "visual yes/no",
            "image_source": item.get("image_source", ""),
        }


class IFEval(BenchmarkAdapter):
    name = "IFEval"
    dataset_id = "google/IFEval"
    splits = ("train",)
    default_split = "train"
    metric = "prompt-level strict accuracy"
    supports_category = False

    def preflight(self):
        import nltk
        from pathlib import Path

        nltk.data.path.append(
            str(Path(__file__).resolve().parents[2] / ".cache" / "nltk")
        )
        from langdetect import DetectorFactory
        from .vendor.ifeval import evaluation_lib  # noqa: F401

        DetectorFactory.seed = 0
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError as exc:
            raise ValueError(
                "IFEval needs tokenizer data. Run: make benchmark-setup"
            ) from exc

    def load(self, split, category):
        if category:
            raise ValueError("IFEval has no subject filter; clear the subject field.")
        return super().load(split, category)

    def prompt(self, item):
        # Adding our own answer-format instruction would change this benchmark.
        return item["prompt"]

    def reference(self, item):
        from .vendor.ifeval.instructions_registry import INSTRUCTION_DICT

        requirements = []
        for identifier, kwargs in zip(item["instruction_id_list"], item["kwargs"]):
            instruction = INSTRUCTION_DICT[identifier](identifier)
            description = instruction.build_description(
                **{k: v for k, v in kwargs.items() if v is not None}
            )
            if "prompt" in (instruction.get_instruction_args() or {}):
                description = instruction.build_description(prompt=item["prompt"])
            requirements.append(f"- {description}")
        return (
            "No single reference answer. All of these requirements must pass (strict):\n\n"
            + "\n".join(requirements)
        )

    def details(self, item: dict[str, Any], response: str):
        from .vendor.ifeval import evaluation_lib as official

        ids, kwargs = item["instruction_id_list"], item["kwargs"]
        if not ids or len(ids) != len(kwargs):
            raise ValueError("Malformed IFEval instruction list.")
        # HF Arrow fills unused keyword columns with nulls.
        inp = official.InputExample(
            key=item.get("key", 0),
            prompt=item["prompt"],
            instruction_id_list=ids,
            kwargs=[{k: v for k, v in row.items() if v is not None} for row in kwargs],
        )
        mapping = {inp.prompt: response}
        strict = official.test_instruction_following_strict(inp, mapping)
        loose = official.test_instruction_following_loose(inp, mapping)
        return {
            "correct": strict.follow_all_instructions,
            "strict_instructions": strict.follow_instruction_list,
            "loose_correct": loose.follow_all_instructions,
            "loose_instructions": loose.follow_instruction_list,
        }

    def score(self, item, response):
        return self.details(item, response)["correct"]
