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

### Automatic profile discovery

`research/profile_discovery.py` ranks input features before vector construction.
For paired differences `d = B - A`, the score is
`abs(mean(d)) * abs(mean(d)) / RMS(d) * coverage(d)`, with a `1e-6` denominator
floor and contrast-presence threshold. A candidate must have nonzero mean and
score above that threshold. Ties use feature ID. This is a ranking heuristic in
native activation units; it is not calibrated confidence, a statistical test or
causal evidence. Coverage and same-direction counts use all calibration pairs.
No discovery ranking changes the profile or vector builder.

`sae_dashboard/profile_discovery_lab.py` recaptures just one selected A/B pair,
without generation or steering, under the existing session/model locks. Only the
selected layer's SAE is encoded, in batches of at most 16 positions; retained
activations are `[tokens, candidates]`. Their profile-scope aggregation must match
the original profile. Source offsets require retokenization of the complete chat
with identical token IDs. Unavailable alignment falls back to an explicit token
strip. Overlapping source offsets use the maximum activity among contributing
tokens. Image positions are counted but never presented as pixel attribution.
The mixed map compares per-feature normalized activities, with one scale across
A/B; hover values preserve raw magnitudes. Input/source changes clear discovery
results. Explicitly following the top three updates Extra's layer/IDs and clears
its previous result, without changing steering strengths or generating a response.

### Generated SQL intervention experiments

**Test candidates on generated SQL** is an optional Common profile panel. It
filters candidate polarity before the limit (default B higher, at most six), then
generates SQLite lookup-function continuations from the same exact assistant
prefix. Labels and evaluation probes are never appended to the model input.
The target is a behavioral result in a synthetic fixture, not an A/B label token.

Search covers individual decoder directions, the three possible pairs of the best
three singles, and first-token/continuous alternatives for the best initial
intervention. Directions sum equal unit decoder contributions and are normalized
as a whole. The budget defaults to 2% of the current final-position residual norm,
with a 5% maximum; actual changes after dtype rounding are recorded and reduced
when necessary to remain within budget. Residuals are never SAE reconstructions.
Only the chosen layer is observed, using the existing next-token trace and bounded
SAE encoding. Other SAE features can also change under a decoder intervention.

Selection minimizes known and unknown functional regressions, then maximizes
comparable cases, new goal hits, total target hits and functional/assessed cases,
with ties favoring fewer features and a shorter window. A choice without a complete
positive search gain is explicitly exploratory. All
selection uses search tasks only. Reserved continuations compare the frozen
direction/schedule with Base, its reverse and two seeded random directions at the
same budget. A separate full-generation check removes the function prefix and
applies the frozen direction continuously, including its two random controls;
this is a different schedule, not proof of automatic query-region detection.
A repeated baseline checks token-ID replay. Confirmation requires new reserved
goal hits, no functional regressions, more reserved hits than either random
control, a passing baseline replay, and complete baseline/candidate/random-control
assessments on the reserved tasks. New and lost hits count only task pairs where
both outputs were assessed. It is an observation on this small sample,
not a statistical significance claim.

`research/sql_generation_checks.py` parses generated Python and interprets a small
whitelist: one lookup function, local assignments/unpacking/returns, conditionals,
short-circuit booleans, equality/identity comparisons, row indices, basic
conversions, scalar strings, f-strings, simple formatting,
cursor/execute/fetchone/close, bound parameters and bounded error handling. It does
not use exec/eval or import generated code. SQL runs in a fresh two-row memory
fixture with a read-only authorizer and an instruction budget. Three ordinary
lookups must pass before the SQL probe can count toward the goal. Syntax errors,
unsupported operations and failed ordinary lookups cannot establish the goal.
Unsupported functionality is unknown (JSON null), distinct from a demonstrated
failure or runtime error. Unsupported operations and interpreter limits cannot be
swallowed by generated exception handlers. One blocked
probe does not establish general security. Pair gains at equal total norm do not
establish feature synergy. Reserved tasks are excluded from this experiment's
search; independence from arbitrary user-provided profile inputs is not proven.

Session job IDs invalidate an in-flight experiment when its profile or parameters
change; Stop / clear stops at a generation boundary. Model locks are released
between generations. Failed/cancelled runs do not publish a completed report.
Existing vectors remain independent. Opt-in experiment saving and explicit
HTML/JSON export follow the existing artifact policy. JSON includes prompts,
input/output token IDs, settings, traces, checks, actual per-step intervention
norms, the frozen unit direction and random seeds.
**Re-evaluate displayed code** updates behavioral checks and summaries without
loading a model or generating new tokens. It retains the originally frozen
candidate and records the source report ID and evaluator version; it never uses
reserved results to select a different candidate.

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

## Causal evidence lab (Extra)

Extra contains the full-response observation audit. The former A/B diagnostic UI
has been removed; its research APIs are retained below for existing callers.
Code-region capture and frozen-vector transfer are documented in the
[step-by-step guide](code_regions_transfer.md). The transfer matrix is in section 5,
separate from the existing base-gated coherence protocol.

### Full generation

Choose **Base only** (no profile required) or **Base + current steering** (section 3
vectors and shared layer strengths). Select one observed layer and optionally up to
three comma-separated feature IDs. Blank IDs select the largest `max(z_j) - min(z_j)`
over the baseline generation; ties use peak activity then feature ID. The same IDs
are observed in both runs. This ranking measures variation, not causal importance.
The map starts near the largest observed activation and reports active-token counts
and peak positions. Sparse features can be inactive throughout the initial window.

Generation uses the existing `generate_answer` path and shared seed, temperature and
token limit. The observer copies the last residual at each forward, before and after
the local vector addition. The first state predicts the first output token; subsequent
states predict subsequent tokens. EOS is retained in the trace. Counts and cached
forward lengths are checked before displaying any alignment. This is **every step**,
not just the final input position once. `STEER_LAST_TOKEN_ONLY=0` still affects all
prefill positions; cached decoding has one position. No new steering policy is added.

SAEs are encoded after generation in batches of at most 16 positions. Only selected
activation columns are retained; automatic selection uses a streaming range pass.
Only the observed SAE is loaded if missing, including with `SAE_WIDTH=262k`. Before/after
isolates the local addition; upstream steering is already present in “before”. After
outputs diverge, matching token indices across runs is not a matched-context causal test.

The static review parses Python blocks and classifies direct `execute`, `executemany`
and `executescript` arguments as literal, dynamically constructed or unresolved.
Version 3 also resolves an immediately preceding, single-name assignment in the same
statement block, recording its source line. Aliases, earlier assignments, cross-branch
values and receiver types remain unresolved. It does not trace input taint, execute
code or test functionality. Zero findings does not establish security. Token-limit
stops and unclosed code fences are flagged as potentially incomplete.

The code shading selector offers SQL review (red dynamic construction, amber unresolved,
teal literal SQL) or an observed feature. SQL review marks the query expression and its
call, including multiline expressions; interpolated display-only strings are not marked.
Click a finding to jump to its code line. Feature shading uses the maximum measured
post-intervention activity among tokens contributing to each line, with one scale per
feature across both responses. It is observation, not causal attribution.
Character spans come from exact, append-only incremental decoding using the generation
decoder settings. Stripped whitespace and skipped special tokens have empty/clipped spans.
If decoded prefixes are rewritten or text does not match, feature line shading is unavailable;
token spellings and re-tokenization are never used to guess an alignment. The selector also
works in exported HTML without JavaScript or further model calls.

HTML/JSON exports include the exact prompt, input/output token IDs, selected features,
activations, generation settings, dictionary identity and current vector metadata.
Changing controls clears the displayed run. Observing IDs does not select a new
intervention; use the auxiliary diagnostic below for individual-feature experiments.

### Legacy A/B diagnostic API (not in the dashboard)

This study does not apply persistent steering, generate full answers, or optimize
for incorrect decisions.

1. Build a common profile in section 2. Choose a study layer and evaluation source
   through `evidence_lab.prepare_study`. SQL validation/test presets supply A/B
   descriptions. Uploaded manifests may contain `labels.A` and `labels.B`; otherwise
   enter both descriptions. Check that they match the actual pair semantics.
2. Select one to three features and **Compare features**. The shortlist uses
   calibration stability, not a claimed causal effect. Whole pairs are sampled
   reproducibly using the shared seed. Both sides are always evaluated.
3. Inspect an intervention and an example. The report separates correctness,
   false alarms, discrimination, response criterion and local activation spread.
   **Export presentation & data** writes a standalone HTML report and full JSON.

The study reuses the loaded model and SAE; `SAE_WIDTH=262k` works without a separate
runtime. IDs are checked against the actual profile/loaded dictionary width. Model,
release, SAE ID, profile ID, prompts, sample identities and control seeds are recorded.
The layer-wide vectors and strengths are not used or changed by this lab.
The report counts target inputs with an actual residual change. Runs with no such
change are not ranked above effective interventions and are explicitly unassessed
for causal relevance. An inactive feature at the decision position cannot be tested
by multiplying its zero activation, even if it varied in the common profile.

### Intervention and controls

`research/feature_evidence.py` owns the model-independent measurement functions.
For a candidate j, it measures its current encoded activity z_j at the final input
position and requests a +/-25–100% change in its decoded contribution (50% default).
Zero percent is an identity control. The residual delta is capped to an intended
2% of that position's residual norm. It adds one decoder direction to the original
residual, preserving the SAE reconstruction error. This is not exact latent clamping:
re-encoding may move the candidate differently, and may move other features too.
An inactive feature produces no perturbation, including in its random controls.

Each input/feature/sign gets two seeded random directions with the same intended
perturbation norm. Actual norms are recorded after model-dtype rounding. The lab
re-encodes the actual modified residual and reports other features crossing
`abs(after-before) > 0.001*max(abs(before),abs(after)) + 1e-6`. The target's share of
squared latent change and the six largest other changes are recorded. Neither
this threshold nor this coordinate-dependent energy share proves semantic isolation.
No dense 262k-by-262k matrix is constructed.

Hooks touch only the final input position of one layer during one forward. Earlier
prompt positions, downstream activations and generated continuations are not
controlled. Hooks are removed on failure and success. A final replay of the first
baseline case checks decision-margin reproducibility with relative tolerance 1e-4.

### Decision readout and interpretation

The prompt contains both condition descriptions and the supplied material. Case
names and expected labels are never inserted into the model prompt. Evaluation
compares the next-token logits of single-token A and B; ties are explicit errors.
`P(B | A or B)` is a constrained preference, not calibrated vulnerability confidence.
The total probability mass assigned to A/B is also displayed; low mass is flagged.
Shared temperature and generation length do not affect this one-forward readout.

B is the positive class. All target inputs are scored, including baseline failures.
Hit rate and false-alarm rate use their own class denominators. d-prime and criterion
use the equal-variance Gaussian signal-detection model with loglinear half-count
correction: H=(TP+0.5)/(N_B+1), F=(FP+0.5)/(N_A+1),
d'=Phi^-1(H)-Phi^-1(F), c=-0.5*(Phi^-1(H)+Phi^-1(F)). Ties leave these estimates
undefined. These are descriptive estimates on small samples, not proof that a
model knows or has forgotten a concept. Response wording and generation behavior
need separate experiments.

Four fixed, balanced synthetic controls check simple unrelated A/B tasks. The
control regression denominator includes only controls the baseline answered
correctly; all raw control decisions and flips are retained. These controls are
not a substitute for broader capability evaluation or a same-task control corpus.

Results are ranked by evaluation correctness, then control regressions, then the
mean change in log-odds of the correct choice. This ranking is exploratory. Two
random directions and a small sample do not justify significance or generalization
claims. Freeze feature, layer, descriptions and intervention strength before the
reserved test set; this convention is not enforced by the UI. Calibration overlap
is flagged using exact stripped code blocks (or full text) plus image hashes; absence
of exact overlap does not establish independent program families.

`sae_dashboard/evidence_lab.py` orchestrates session/model locks, paired samples,
optional persistence and explicit exports. `evidence_view.py` renders escaped HTML
and `causal_ui.py` supplies the compact interface. Changed study inputs clear stale
results; changed profiles also clear the prepared study. Automatic writes still
require the shared experiment-saving toggle. An explicit export writes files even
when that toggle is off.

The earlier `causal_interventions.py`, `causal_lab.py`, `feature_study.py` and
`feature_studies.py` APIs remain available to existing research callers, but their
old controls and manual scoring screens have been removed from Extra.

```bash
uv run python -m unittest discover -s tests -p 'test_feature_evidence.py' -v
```
