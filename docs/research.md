# Research contracts

## Vector construction

Own `research/vector_builders/`. It imports PyTorch and Python only; it does not import
Gradio, the model runtime, session state, or storage.

`LayerProfile` supplies matching CPU `[pairs, features]` tensors `a` and `b`, plus the
average selected-token residual norm. A `VectorBuilder.build(profile, decoder, fraction)`
returns a `VectorResult` with a feature delta, a residual direction and JSON-serializable
metadata. Treat input tensors as read-only. Return finite CPU tensors and do not write files.

Register a strategy in `BUILDERS` to expose it in the method dropdown after restart.
No filtering, weighting or robust estimators are implemented: `MeanDifference` is the
baseline `mean(B - A) @ W_dec`, normalized to `reference_norm * fraction`. Zero residual
directions stay zero. The workflow validates custom strategy output shapes and finiteness.

```bash
uv run python -m unittest discover -s tests -p 'test_vector_builders.py' -v
```

## Benchmarks

Own `research/benchmarks/`. `BenchmarkAdapter` supplies loading, prompt construction,
images, normalization, scoring and metadata. Register adapters in `BENCHMARKS`.
`prepare_benchmark(adapter, request)` freezes a reproducible sample with exact prompts,
labeled images, metadata and ground-truth references before inference. `evaluate_base`
accepts an injected generator returning `(answer, cache_hit)`; `evaluate_steered` accepts
a generator returning text and calls it **only for base-correct rows**. The combined
`run_benchmark(adapter, request, generate_base, generate_steered, progress)` is also available.
It returns a JSON-compatible result. There are no dashboard imports or output files.

The workflow supplies shared generation parameters, profile vectors and strengths.
The runner selects a seeded sample, reports its row indices and dataset fingerprint,
and measures full-sample base accuracy plus preservation/regressions on the base-correct subset.
A zero-sized eligible subset yields a null preservation rate. Steered accuracy over the
full sample and improvements on base failures are deliberately not reported.

- MMLU-Pro uses its official Hugging Face dataset with a zero-shot letter-only prompt.
- MMMU loads all subjects when the subject field is blank. It supports all image columns
  with their original numeric labels, multiple-choice and open questions. Open-answer
  parsing/scoring uses pinned upstream code. Multiple-choice scoring accepts an explicit
  letter, never randomly guesses an answer when parsing fails.
- IFEval preserves the original prompt and uses pinned Google Research strict and loose
  verifiers. It reports both prompt-level and instruction-level accuracy. Null-padded
  Hugging Face kwargs are stripped before passing them to the official verifier.
- POPE uses the [LMMS-Lab image-inclusive distribution](https://huggingface.co/datasets/lmms-lab-encoder/POPE)
  of [Polling-based Object Probing Evaluation](https://github.com/RUCAIBox/POPE)
  (Li et al., EMNLP 2023). Its `Full` configuration provides `random`, `popular` and
  `adversarial` splits, each with 3,000 one-image object-presence questions. Start
  with `random` for a simple visual preservation check; the other variants select
  popular or co-occurring absent objects. No subject filter is supported.
  The prompt requests only `yes` or `no`. Our strict scorer accepts either case,
  surrounding whitespace and a single final `.` or `!`; explanations, ambiguous
  and empty answers are incorrect. This deliberately differs from the original
  POPE text heuristic and is not an official POPE leaderboard score. Preview shows
  the image and ground truth; row metadata retains the source image identifier.
  The existing seeded sampler is used without class balancing, so small samples
  need not contain equal numbers of positive and negative questions. Base accuracy
  and preservation use the same denominators as the other adapters. This measures
  object grounding, not every aspect of coherence. Images download on first use
  through Hugging Face Datasets and are kept in its cache, not in this repository.
  Initial preparation may download the full configuration (about 255 MB), even
  when requesting only a few `random` items. Subsequent preparations reuse it.

These are paired dashboard protocols, not reproductions of published leaderboard runs.
The shared token budget affects long-form IFEval tasks. Empty selections and invalid
splits fail before inference. Scorer exceptions fail the run instead of counting as wrong.
Dataset revisions can evolve; record the returned fingerprint when comparing runs.
Upstream evaluator versions and licenses are in [vendor/SOURCES.md](../research/benchmarks/vendor/SOURCES.md).

```bash
make benchmark-setup
uv run python -m unittest discover -s tests -p 'test_benchmarks.py' -v
```

## Integration boundaries

- `sae_dashboard/model_runtime.py`: model lifecycle, input preparation, capture and hooks.
- `sae_dashboard/workflow.py`: session operations, research interfaces and model locking.
- `sae_dashboard/manifests.py`: validated JSON input and asset resolution.
- `sae_dashboard/session.py`: private session state and shared parameter snapshot.
- `sae_dashboard/artifact_store.py`: opt-in results and explicit exports.
- `sae_dashboard/portable_model.py`: self-contained loader copied into exports.
- `sae_dashboard/ui.py`: English controls, section layout and event wiring.

Keep model calls under `MODEL_LOCK` and session mutations under the session lock.
Acquire the session lock before the model lock. Never put per-user values into runtime
globals. New vector methods or benchmark adapters should not need UI edits.

## Single-feature causal lab

`research/causal_interventions.py` contains additive decoder interventions and next-token
readouts, injected with a model runtime. Callers own the model lock. Hooks are removed on
success and failure. Zero dose leaves hidden states untouched; first-step and every-step
schedules are available for generation. Shared prefixes extend masks and the continuation
boundary, so displayed output contains newly generated tokens only.

`sae_dashboard/causal_lab.py` handles current-profile/manual selection, coefficient scaling,
seeded equal-norm random controls, ablation, token maps and optional recording.
`causal_ui.py` renders the Extra tab. No common-profile vector is mutated by the lab.

`sae_dashboard/base_response_cache.py` owns atomic JSON base-only caching. Benchmark
research adapters must not implement persistence or call the model directly.
