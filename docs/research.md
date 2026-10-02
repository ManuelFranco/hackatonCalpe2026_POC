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
`run_benchmark(adapter, request, generate_pair, progress)` calls an injected generator
that accepts labeled in-memory images and a prompt and returns `(base, steered)`.
It returns a JSON-compatible result. There are no dashboard imports or output files.

The workflow supplies shared generation parameters, profile vectors and strengths.
The runner selects a seeded sample, reports its row indices and dataset fingerprint,
and measures paired accuracy, regression, improvement and preservation.

- MMLU-Pro uses its official Hugging Face dataset with a zero-shot letter-only prompt.
- MMMU loads all subjects when the subject field is blank. It supports all image columns
  with their original numeric labels, multiple-choice and open questions. Open-answer
  parsing/scoring uses pinned upstream code. Multiple-choice scoring accepts an explicit
  letter, never randomly guesses an answer when parsing fails.
- IFEval preserves the original prompt and uses pinned Google Research strict and loose
  verifiers. It reports both prompt-level and instruction-level accuracy. Null-padded
  Hugging Face kwargs are stripped before passing them to the official verifier.

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
