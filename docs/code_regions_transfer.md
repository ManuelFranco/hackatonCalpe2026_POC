# Code-region capture and transfer: step by step

The original manifest → profile → vector → comparison/export workflow is unchanged.
Two opt-in extensions focus the calibration measurements and compare frozen vectors
across reserved code families. Neither extension claims to identify a universal
vulnerability feature or to improve accuracy before an actual model evaluation.

## 1. Choose what the profile measures

Load `sql_injection/manifest.json` in **1 · Load inputs**. In the shared parameters,
choose **Profile tokens** before clicking **Build common profile**:

SQL revision `code-flow-v3-20` contains 20 calibration pairs and separate
20-pair validation/test manifests. Reload the training manifest and rebuild old
profiles/vectors to use the compact selection. The [SQL dataset guide](../data/sql_injection/README.md)
documents behavioral checks, template dependence and split use.

| Option | Selected positions | Use |
| --- | --- | --- |
| All tokens | Entire processed chat, including the assistant prefix | Existing default |
| Last token | Last input position | Existing decision-position comparison |
| Non-image tokens | Positions other than image tokens | Existing multimodal option |
| Code only | Bodies of fenced code blocks, or recognized unfenced source | Compare SQL, command injection, XSS or C examples |
| SQL construction | Query expressions and reachable preceding assignments in the same statement block | Study construction of the argument that reaches a database call |
| SQL execution calls | Python `execute`, `executemany` and `executescript` call expressions | Focus on the execution boundary |
| SQL construction + calls | Union of construction and call spans | Compare both regions together |

Code-region scopes are text-only. SQL scopes require valid Python; a SQL-specific
scope on command/XSS code produces an actionable error instead of silently measuring
all tokens. Other languages can use Code only. Explicit image inputs retain the
original all/last/non-image workflow.

**The model still receives the complete original input.** Only the positions fed to
SAE aggregation change. Mean/max aggregation and every existing vector builder keep
their behavior. The residual reference norm is measured on the selected positions.
Changing scope or aggregation invalidates the current profile and vectors.

## 2. Inspect the selection

After profiling, open **Capture details → Inspect captured input**. Choose either
side of a pair to see the selected/total token counts and highlighted source spans.
The tokenizer's full-chat IDs must exactly match the processed input IDs before
character offsets can be used. A mismatch or missing offset support stops capture.
Message templates may trim leading/trailing whitespace; source offsets account for
that trimming and exclude removed whitespace from the selection. Other source-text
changes or ambiguous matches still stop capture.
Unicode AST byte offsets are converted to character offsets; partial boundary tokens
can include adjacent whitespace. Prefix and fence-marker tokens are excluded from
the source scopes unless they share a boundary token.

SQL construction follows the latest preceding single-name assignments recursively
within the same statement block, including the expression behind a starred argument
tuple. It does not infer receiver types, trace branches or helper functions, or prove
taint/safety. Unrelated display-only interpolations are not collected unless they
are dependencies of the executed argument. This is a measurement mask, independent
of the bounded generated-code reviewer in Extra.

## 3. Build and freeze the first source

Create vectors in **3 · Steering vectors** using any existing method. Set at least
one nonzero layer strength. Open **5 · Benchmarks & transfer → Transfer across
vulnerabilities**, enter a descriptive source name, and click **Freeze current vectors**.

A frozen source copies the CPU residual vectors, strengths, last-token/all-token
policy, model revision, capture settings, profile/vector IDs and calibration hashes.
Later edits to live vectors or strength sliders cannot modify this source. Zero
interventions are rejected at freezing. Up to six sources can coexist in the session;
**Clear frozen sources** explicitly starts over. Refresh/session expiry discards them.

For a first scope comparison, freeze a SQL vector from Code only, then rebuild the
same training manifest with SQL construction + calls and freeze it under another
name. Keep the method and strengths comparable. Different masks can have different
reference norms; consult source metadata and measured residual changes when comparing
effects. Frozen directions are fixed, rather than renormalized on each evaluation case.

## 4. Add other source families, if desired

Load `command_injection/manifest.json`, select **Code only**, build the profile/vector,
set strengths and freeze another source. Repeat with `xss/manifest.json` if useful.
Loading new calibration inputs clears the live profile, vectors and evaluation,
but preserves previously frozen sources. This is how you build multiple matrix rows.

The new families each contain 6 training, 4 validation and 4 test pairs. They use
neutral context and Python code without reference answers inside the material:

- Command injection compares separate subprocess arguments or POSIX shell quoting
  with request values entering shell syntax. Test cases include an explicit
  `/bin/sh -c` call with `shell=False`, so the flag alone is insufficient.
- XSS compares escaping with raw request values in **HTML text content**. Test cases
  include reassignment and undoing escaping. This does not cover attributes,
  JavaScript, URLs, template engines, browser CSP or an entire web application.

The fixtures are synthetic and never executed by the dashboard. Tests mock every
subprocess call; no external program or injected command is launched.

## 5. Prepare and inspect the evaluation

Select frozen sources and evaluation manifests. The default targets are SQL,
command-injection and XSS **validation**. Choose pairs per evaluation set and click
**1 · Prepare transfer sample**. Whole A/B pairs are sampled deterministically using
the shared seed; both sides are always retained.

Set the limit to **20** to evaluate a complete SQL validation/test split (40
prompts); a lower limit samples that split. Select settings on validation and keep
the full test split reserved for frozen choices. The synthetic variants share
templates, so assess paired outcomes and template groups as well as prompt accuracy.

Inspect **Preview transfer input** before inference. It shows the exact model prompt
and expected label separately. Each target supplies its own `labels.A`/`labels.B`
descriptions. Filenames, expected labels and matrix source names are not inserted as
answers into the prompt. Legacy per-CWE manifests derive descriptions from the CWE
title; custom version-1 manifests need explicit descriptions.

The preview flags exact code/image overlap with each source's calibration set.
Such cells are exploratory. Absence of exact overlap does not rule out related
program templates. Uploads use the existing confined asset-loader rules; sources
and targets must have distinct names.

## 6. Run and interpret the matrix

Click **2 · Run transfer matrix**. Each input is processed once. BASE gets one
forward and is reused across all sources; each source gets one intervened forward.
Other browser sessions can use the shared model between cases. No SAE is needed
for this inference, and no full responses are generated.

The readout compares the next-token logits for single-token A/B labels. It uses all
sampled inputs, including BASE failures. Temperature and max-new-tokens have no
effect on this fixed decision readout. The original frozen hook policy determines
whether the residual vector touches the final input position or all prefill positions.

Each cell reports accuracy change in percentage points, counts of corrections and
regressions, false alarms on A, misses on B, and exact calibration overlap. In the
case inspector you can compare BASE/steered labels, the B−A logit margin, A/B mass,
actual changed-input counts, ties and source code. A/B probability mass below 50%
is flagged; normalized A/B preference is not calibrated vulnerability confidence.
Ties count as errors and are listed separately from false alarms and misses.

A correction with a regression can produce zero net accuracy change. A vector that
always favors B may increase hits and false alarms together; inspect both. These
small datasets provide exploratory transfer evidence, not statistical significance,
exploit-success rates or full-response quality. Use the existing coherence benchmarks
for capability preservation and section 4/Extra for free generation.

## 7. Freeze choices and export

Use validation to choose source/method/scope/strengths. Freeze those choices before
selecting the three `manifest.test.json` files. Prepare a new sample and rerun.
The app preserves provenance but does not enforce a blind-test policy.

**Export transfer report** writes a self-contained HTML matrix with case details and
a JSON file containing prompts, labels, margins, residual changes, settings, source
vector hashes and dataset fingerprints. Reports contain no images or model weights;
image hashes remain in JSON for custom multimodal targets. Automatic experiment
writes still require the existing saving toggle. The explicit report export writes
regardless of that toggle. Frozen vectors are session state, not persisted packages;
section 6 still exports the current live vector/model through the existing loader.

Extra now contains only the full-generation audit. The former feature A/B diagnostic
is removed from the dashboard, while its Python APIs remain available to existing
research callers.
