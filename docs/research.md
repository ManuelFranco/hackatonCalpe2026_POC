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

## Feature research protocol (Extra)

Extra now starts with label-free measurements and freely generated responses.
`research/feature_studies.py` owns input transformations, chunked single-layer activation
measurement and complete-case manual rating summaries. `sae_dashboard/feature_study.py`
owns session orchestration and optional recording. Next-token label readouts remain
available in the research API, but are no longer the primary dashboard workflow.

### Reproduce the layer 22 / feature 11749 investigation

The reference screenshot uses **Gemma 3 4B IT, residual layer 22, Gemma Scope 2,
262k dictionary**, not 16k. Feature IDs are dictionary-specific. Start this server with
`SAE_WIDTH=262k` (the default remains `16k`) and the same SAE release/model as the
reference. Startup validates the configured SAE IDs against the installed SAE registry.
Extra's reference button sets the expected size to **262144**; inference refuses a
mismatch. Inspect the feature to see the actual model/release/SAE ID, profile statistics
and matching Neuronpedia embed. The common-profile embeds also use the profile width.

1. Load `data/sql_injection/manifest.json` in section 1 and build the profile with
   **all / mean**. In Extra select the reference, keep **Common profile**, and inspect it.
2. Load current section 1 inputs into the research workspace. Select all three
   activation views and 12 pairs. Inspect mean paired B − A and sign consistency.
   The current code-flow-v2 dataset has no reference reviews: **Full input** and
   **Code + preamble** retain the same content. **Text without code** leaves the
   identical neutral context on A/B, a negative control. On older datasets it can
   expose contrasts in reference explanations. Non-code cases should use **Full input**. Unsupported
   transformations stop with an explanation rather than silently skipping examples.
3. Upload `data/sql_injection/manifest.validation.json` into Extra and click load. This
   keeps the training profile and its dose calibration intact. The JSON still resolves
   assets under the server's configured data root, using the normal manifest loader.
4. Select a validation pair and **Code + preamble**. Leave the shared instruction blank:
   these validation files already request a neutral three-sentence analysis of how
   supplied values reach the database call and affect the query, with changes only if needed. Expand the exact input preview;
   no reference answer should enter these response prompts. For other datasets, an
   optional shared instruction is added only for generation, not activation measurements.
5. Run dose **0**, ablation off, random directions **1**. Both A and B receive identical
   generation settings across base / positive / negative / random conditions. The
   zero-control indicator compares exact answer strings, excluding optional ablation.
6. Try **0.5**, then **1**, with **3 random directions** and optionally dynamic ablation.
   This is 12 generations per pair, or 14 with ablation. Rate mechanism, consequence and
   recommendation 0–2 in the editable rubric. Mean quality averages these three scores;
   mean change uses only cases with both the condition and base rated. Blank rows are
   excluded, not zeros. Conditions with unequal rated counts are not directly comparable.
   Record false vulnerability claims separately, especially on protected A inputs.
7. Repeat on every validation pair and multiple generation seeds. Freeze the dose,
   instruction and rubric before using `manifest.test.json`. Rating summaries are per run;
   enable experiment saving to retain linked run IDs and combine runs in research analysis.

**Expected evidence, not a promised outcome:** a code-sensitive feature should retain
useful A/B separation without the reference explanation, and positive intervention should
improve grounded analysis without inventing vulnerabilities in A. A contrast restricted to
reference reviews, or changes resembling random controls, weakens this interpretation.
The external feature name and a 12/12 calibration sign match are hypotheses, not validation.
That screenshot used the older code-plus-review dataset. Rebuild the profile and
re-rank candidates on code-flow-v2; feature 11749 may become weaker or disappear.

The three view means include the chat template and shared token scope. They have different
contexts and lengths and are not an additive attribution decomposition. Input measurements
always use mean; selecting max aggregation at the top affects profile calibration only.
Feature ablation subtracts the current encoded contribution at each selected last position;
it neither guarantees zero re-encoded activation nor removes all prompt-position evidence.
Random directions match the additive decoder-vector norm, not the dynamically varying
ablation magnitude. Diagnostics report the largest *observed* relative residual change;
the 5% calibration cap is not a per-step bound. All hooks use try/finally cleanup.

No results are written by default. The common experiment-saving toggle controls activation
studies, generated-response records and manual ratings. Response records include exact
prompts, image pixel fingerprints, SAE identity, shared settings, control seeds and a run ID
that links ratings. The feature study never updates the common-profile vectors.
