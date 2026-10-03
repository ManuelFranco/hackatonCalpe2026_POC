"""Run base-model benchmarks through a local Ollama server.

The resulting JSON contains the complete prepared sample, prompts, references,
images, responses, scores and Ollama metadata. It can be loaded in the dashboard
to evaluate only the steered model.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Allow ``python scripts/run_local_benchmark.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.benchmarks import (
    BENCHMARKS,
    BenchmarkRequest,
    evaluate_base,
    prepare_benchmark,
    summarize,
)
from sae_dashboard.benchmark_artifact import save_base_artifact


def ollama_chat(base_url, model, prompt, images, temperature, seed, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    encoded_images = []
    for _, image in images:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    if encoded_images:
        messages[0]["images"] = encoded_images
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_predict": max_tokens,
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=3600) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    answer = payload.get("message", {}).get("content")
    if not isinstance(answer, str):
        raise RuntimeError("Ollama returned no textual message.")
    return answer, {
        "duration_seconds": time.perf_counter() - started,
        "model": payload.get("model", model),
        "done_reason": payload.get("done_reason"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "total_duration": payload.get("total_duration"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu_pro")
    parser.add_argument("--split", default=None)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--category", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="benchmark_base.json")
    args = parser.parse_args()

    adapter = BENCHMARKS[args.benchmark]
    split = args.split or adapter.default_split
    request = BenchmarkRequest(split, args.items, args.category, args.seed)
    sample = prepare_benchmark(adapter, request)
    telemetry = []

    def generate(images, prompt):
        answer, details = ollama_chat(
            args.ollama_url,
            args.model,
            prompt,
            images,
            args.temperature,
            args.seed,
            args.max_new_tokens,
        )
        telemetry.append(details)
        return answer, False

    rows = evaluate_base(sample, generate)
    for row, details in zip(rows, telemetry):
        row["base_generation"] = details
    result = summarize(sample, rows)
    save_base_artifact(
        args.output,
        sample,
        rows,
        result,
        metadata={
            "runner": "ollama",
            "ollama_url": args.ollama_url,
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(
        f"Saved {len(rows)} base evaluations to {args.output} "
        f"({result['summary']['base_correct']}/{result['summary']['num_items']} correct)."
    )


if __name__ == "__main__":
    main()
