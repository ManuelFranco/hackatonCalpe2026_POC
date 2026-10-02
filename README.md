# Hackathon 2026

A shared Gradio workspace for Gemma 3 and Gemma Scope 2 activation steering.

The dashboard is in English and each browser session owns its manifests, profile,
vectors, settings, and results.

```bash
make setup

make benchmark-setup     # tokenizer data required by IFEval

make check

make test

make login               # accept Gemma's Hugging Face access terms first

make server              # NVIDIA; use make mac for Apple Silicon, make run for auto
```

Open http://127.0.0.1:7860. `PRELOAD=1` loads model/SAEs before serving.

### Windows

On Windows, use the following commands instead of the `make` targets:

```powershell
uv sync
uv run hf auth login
uv run python dashboard.py
```

The `hf auth login` step requires accepting Gemma's Hugging Face access terms first.

See [deployment](docs/deployment.md) for configuration and shared access.

## Workflow

0. **Try Gemma 3** — text or image inference without loading SAEs or building a profile.

1. **Load inputs** — upload/select JSON manifests or open the optional manual A/B editor.

   Add, edit or remove image–text pairs, independently or alongside manifests.
   Loaded pairs appear side by side with their text and images. Use the input-source
   and pair selectors to browse; only the selected pair's images are decoded for preview.

2. **Build common profile** — capture matched A/B SAE features for all pairs.

3. **Create steering vectors** — average B − A, retain all features, decode and scale.

4. **Base vs. steered** — compare using the same prompt, seed, temperature and token budget.

5. **Coherence benchmarks** — preview MMLU-Pro, MMMU, POPE or IFEval cases and references,
   evaluate/cache the base model, then evaluate steering only on base-correct cases.
   For simple multimodal samples, select **POPE → random**: one photograph and a
   yes/no question about whether an object is present. Start with 20 items and leave
   the subject field empty. Prepare the sample, evaluate base, then evaluate steered
   with your current vectors. POPE uses strict yes/no scoring; see the
   [benchmark protocol](docs/research.md#benchmarks).

6. **Export VLM** — explicitly download a portable loader, vectors and configuration;
   optionally include base weights and processor.

7. **Extra** — single-feature causal sweeps, ablation, random controls, token activation
   maps and Markdown full-response comparisons. Choose a current profile feature or a manual ID.

Generation and capture controls appear once at the top. Profile tokens default to
`all`; `last` and `non_image` remain available. Capture scope and aggregation changes
invalidate the old profile and vectors. Rebuilding a profile invalidates its vectors.

The vector baseline intentionally does **not** apply the previous common-feature
intersection; all mean B − A features are retained.

## Manifests

[`data/sql_injection/manifest.json`](data/sql_injection/manifest.json) contains
12 SQL-injection code-and-review pairs for `all + mean`, with separate validation
and test manifests requesting short explanatory reviews.
Use the same loading, profiling and comparison flow as every other use case.
See the [dataset instructions](data/sql_injection/README.md) for exact prompts
and expected review content.

[`data/physical_damage/manifest.json`](data/physical_damage/manifest.json)
loads our eight physical-damage training pairs with A = intact and B = damaged.
The three validation and five test pairs have separate manifests. See the
[dataset instructions](data/physical_damage/README.md) for loading, evaluation
and steering direction.

[`data/scripts/cwe_120/manifest.json`](data/scripts/cwe_120/manifest.json)
contains 20 normal/vulnerable code pairs. Each manifest represents one use case:

```json
{
  "version": 1,
  "name": "Code safety",
  "asset_root": "scripts/cwe_287",
  "pairs": [
    {
      "id": "CWE-287",
      "A": {"text_file": "cwe_287_A_01.txt", "image": ""},
      "B": {"text_file": "cwe_287_B_01.txt", "image": ""}
    }
  ]
}
```

`text` contains inline text; `text_file` loads UTF-8 text. Choose one of these.

`image` is a relative path. Each condition must contain text, an image, or both.

`asset_root` is relative to `GEMMA_DATA_ROOT` (default: repository `data/`). This
makes uploaded JSON files resolve identically to their repository counterparts.

Without `asset_root`, repository manifests resolve relative to their own directory;
uploaded manifests resolve relative to the data root. Absolute paths are accepted
only inside that root. Traversal and symlink escapes are rejected. Text is loaded
into the session; images are fingerprinted and checked again before capture.

Per-CWE v2 manifests with `safe_file` / `vulnerable_file` mappings are also readable.

The dataset-wide index is not a use-case manifest; select a category manifest.

The **Manual image–text pairs (optional)** accordion in section 1 accepts text, an
image, or both on each side. Empty A/B conditions are rejected. Add several pairs,
select one to edit it, or remove it. Loading JSON manifests preserves manual pairs;
the combined limit is 500 pairs. Every successful input change invalidates the profile
and vectors. Uploaded image snapshots stay in session memory (Gradio still manages its
upload cache); experiment metadata records their hashes rather than image bytes.

Refreshing/expiring the session discards manual inputs. `Manual pairs` is reserved as
a manifest name while manual input is in use.

## Parallel work and persistence

The UI lives in `sae_dashboard/ui.py`; research teams can work independently in
`research/vector_builders/` and `research/benchmarks/`. See
[research contracts](docs/research.md) for extension points and test commands.

Each session has a distinct ID and lock. A shared model lock covers temporary hooks
and RNG changes; a base/steered pair cannot interleave with another model request.

Calibration yields the model between pairs and benchmarks yield between items.

This supports simultaneous browser work while serializing inference on one model.

Sessions expire after four hours; refreshing may start a new session. This is a
session-isolated workspace, not an account/authentication system.

**Experiment recording is off by default.** The top button opts the current
session into saving new JSON results and metadata under `runs/<session-id>/experiments/`.
It does not retrospectively save previous events or export a model. Turning it off
stops subsequent writes. The default can never be enabled through an environment
variable.

**Base benchmark responses are automatically cached as JSON**, independently
of this toggle, under `.cache/base_responses` (override with `GEMMA_BASE_CACHE_DIR`).

Cache keys cover processed input tokens and image tensors, exact prompt, seed, temperature,
token budget, resolved model configuration/revision, generation defaults and runtime versions.

Atomic writes and per-key process locks support concurrent sessions. A hit skips generation;
model loading/input preparation can still be needed to verify identity. Steered benchmark
responses are recomputed every run, remain in memory and are never persisted, even with
experiment recording enabled. Causal experiments follow the recording toggle.

Gradio upload caches and Hugging Face/model/dataset caches are separate.

Only **Export steered VLM** writes model packages under
`runs/<session-id>/exports/<unique-id>.zip`, regardless of the experiment-saving
setting. Packages include safetensors vectors, generation and steering settings,
a standalone loader, versioned requirements, and optionally the base weights.

Without the weights, the loader fetches the referenced base model. No SAE is needed
for exported inference. Activation hooks cannot be merged into ordinary Gemma weights;
use the bundled `SteeredVLM` loader to retain steering behavior.

## Validation and cleanup

`make check` validates configuration, SAE registry and UI without downloading weights.

`make test` checks manifests, tensor math, hook cleanup, concurrent sessions, explicit
persistence, export reloading and benchmark scoring using small fixtures/fake models.

These checks do not measure actual Gemma behavior or benchmark quality; run section 0
and the benchmark workflow on the target GPU to do that.

The retired monolithic entry point, hardcoded presets and historical experiment fixtures
were removed from the active tree. The causal laboratory is now integrated under Extra,
using the current profile or manual selection instead of historical preset files. Earlier tracked versions remain
in Git. Use `dashboard.py`; old profile files are not imported by the new workflow.
