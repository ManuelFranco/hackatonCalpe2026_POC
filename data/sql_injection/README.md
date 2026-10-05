# SQL Injection — Code Flow

A text-only use case for the standard manifest → common profile → vectors →
base/steered workflow. Revision **code-flow-v3-20** uses **20 calibration pairs**
and reserves separate **20 validation + 20 test pairs**. It preserves all 44 original
v2 prompt assets byte for byte, the neutral input context and the original split
assignments. Reference reviews and explicit security cues remain outside model inputs.

**A:** the database call binds request values separately from SQL syntax.
**B:** the executed query contains at least one interpolated request value.
These descriptions are researcher metadata, not part of the model input. The
scope is the shown SQLite execution path, not the safety of an entire application.

## Dataset

| Manifest | Pairs | Model input | Purpose |
| --- | ---: | --- | --- |
| `manifest.json` | 20 | Shared neutral context + code | Build the profile and vectors |
| `manifest.validation.json` | 20 | Neutral context + code-flow task + code | Select settings |
| `manifest.test.json` | 20 | Neutral context + code-flow task + code | Evaluate frozen settings |

All **120 sample files** across the three splits omit reference explanations, vulnerability names, severity
labels, safe/unsafe annotations, YES/NO answers and suggestive function names.
Every pair has identical surrounding text. SQL-specific dataset names and A/B
labels remain outside the captured prompt; the normal loader supplies only the
condition text and optional image to the model.

The shared context preserves the origin of values without naming a security issue:

> SQLite. db is a database connection; all other arguments are strings from an HTTP request.

## Coverage and controls against superficial cues

The compact selection keeps **12 original calibration pairs plus eight distinct
added mechanisms**. The eight additions balance construction idioms: two each
using f-strings, concatenation, percent formatting and `str.format`. Every pair
keeps the table, request origin, surrounding context and ordinary database behavior
matched between A and B.

| Added pair | Mechanism |
| --- | --- |
| `named_dictionary_04` | Named binding versus interpolation with an unused binding dictionary |
| `like_value_02` | Bound LIKE pattern versus an interpolated pattern |
| `partial_binding_03` | Both values bound versus only one bound |
| `mapped_identifier_01` | Fixed table mapping with a bound versus interpolated predicate |
| `list_membership_04` | Generated IN placeholders versus generated SQL literals |
| `insert_value_03` | Bound versus interpolated INSERT value |
| `update_predicate_02` | Bound versus interpolated UPDATE predicate |
| `delete_predicate_01` | Bound versus interpolated DELETE predicate |

**Nine calibration pairs** provide matched controls: both conditions contain SQL
candidates, the same construction components, or the same parameter dictionary.
An unused binding dictionary on B is deliberate: sqlite3 accepts extra named
dictionary entries, so their presence alone does not protect interpolated SQL.
Some A examples legitimately format a fixed mapped identifier or a list of `?`
placeholders. These controls make formatting or parameter presence insufficient
as a label rule. They do not eliminate every possible syntactic shortcut.

The new reserved template groups add mapping packets, alias chains, aggregate
HAVING, CASE predicates, optional filters, partially bound INSERT, interpolated
SET values and cursor loops on validation; indexed/history packets, UNION,
named batch previews, LIKE wildcard escaping, joined IN lists and CTEs with mixed
binding on test. Reference mechanisms and execution fixtures for **every pair**
are in [`dataset_metadata.json`](dataset_metadata.json); the dashboard never
appends that file to model inputs.

### Preserved v2 subset

The original 12 calibration pairs remain first in the manifest and comprise:

- **Six minimal pairs:** two each using f-strings, string concatenation and percent
  formatting in B, with parameter binding in A.
- **Six matched preview controls:** both A and B construct the same fixed query,
  interpolated SQL preview and parameter tuple. A executes the fixed query with
  the tuple; B executes the preview. Three pairs call execute directly and three
  pass a selected argument tuple via `execute(*request)`. Both sides return the
  same preview alongside their rows. The mere presence of formatting, a `?`, or
  a parameter tuple is therefore insufficient to distinguish these conditions.

| Calibration group | Pair file stems | Formatting order |
| --- | --- | --- |
| Minimal | `lookup_catalog`, `lookup_tickets`, `lookup_devices` | f-string, concatenation, percent |
| Minimal | `lookup_products`, `lookup_books`, `lookup_packages` | f-string, concatenation, percent |
| Shared preview, direct call | `lookup_builds`, `lookup_hosts`, `lookup_services` | f-string, concatenation, percent |
| Shared preview, argument tuple | `lookup_projects`, `lookup_sessions`, `lookup_artifacts` | f-string, concatenation, percent |

For example, the shared code in one pair constructs `statement`, `preview` and
`parameters`; the material difference is:

```python
# A: this annotation is documentation only, not included in the sample.
rows = db.execute(statement, parameters).fetchall()

# B: this annotation is documentation only, not included in the sample.
rows = db.execute(preview).fetchall()
```

The preview is returned as text, never executed on A's database path. No behavior
of a downstream caller is assumed. B may fail on some input values; the relevant
property is the request value entering the executed SQL string.

The examples are small, related synthetic programs. There are **11 train, 12
validation and 13 test template groups**, including the preserved subset; multiple
variants of a template are not independent mechanisms. Newly added template groups
are assigned to one split, but underlying concepts overlap across splits. The
splits hold out examples and new templates, not wholly unseen mechanism families.
Exact duplicate Python ASTs and reused prompt assets are rejected by the audit;
absence of exact duplication does not establish independence.

## Behavioral audit and reproducibility

[`validation_report.json`](validation_report.json) records the dataset revision,
hashes, group counts and per-pair audit status. The audit uses fresh **in-memory
SQLite databases** and verifies:

- **120 ordinary-input comparisons:** A and B produce identical database reads
  and final table contents for two ordinary input sets per pair.
- **60 apostrophe checks:** A accepts apostrophe-bearing values without placing
  them into executed SQL text.
- **60 successful injection witnesses:** both conditions execute successfully,
  while B's parsed request syntax changes query results or table contents relative
  to A. A syntax error on B is insufficient to pass this check.
- **480 capture-region checks:** every input has nonempty `code`, `sql_query`,
  `sql_call` and `sql_flow` selections, valid Python, the exact split-specific
  context, no answer cues and a matching SHA-256 asset hash.

From the repository root, with Python 3.12:

```sh
python tools/sql_injection_dataset.py audit --report data/sql_injection/validation_report.json
python -m unittest discover -s tests -p test_sql_dataset.py -v
```

To reproduce the added assets and manifests, run
`python tools/sql_injection_dataset.py build --report data/sql_injection/validation_report.json`.
The build retains legacy text assets and refuses unexpected manually added manifest
entries. It removes unselected generated training assets only when their saved
hashes match, and refuses to remove edited assets. The offline audit executes repository-authored fixture code only; it is
not an evaluator for arbitrary uploads. The dashboard itself never executes snippets.

These checks establish local reference-label behavior. They do not measure Gemma
accuracy, SAE feature meaning, steering benefit or an entire application's safety.

## Capture and feature interpretation

Use **all + mean** and **Mean B − A (all features)**. Capture measures only the
input; no generated explanation is appended or averaged. The first mean is across
input tokens; the vector builder separately averages across pairs.

Removing reference reviews may reduce activation magnitudes and change rankings.
The earlier layer 22 / feature 11749 / 262k result was obtained on another input
distribution. Re-measure it and inspect other candidates; do not preserve a large
contrast by putting the answer back in the sample. Positive B − A is the direction
toward B's observed input activations, not a guarantee of improved reasoning.

## Load and inspect

1. Reload `sql_injection/manifest.json` in section 1, confirm **20 pairs**, and rebuild the common profile
   with the intended model, layer dictionaries and `all / mean` capture.
2. For focused calibration, optionally rebuild with **SQL construction**, **SQL
   execution calls**, or **SQL construction + calls**. Open **Capture details** to
   inspect the selected token counts and highlighted code. The complete model input
   remains unchanged; only the SAE measurement positions differ.
3. Create vectors, set strengths, then open **5 → Transfer across vulnerabilities**
   and **Freeze current vectors** under a distinct source name. Frozen vectors
   survive loading or profiling another family.
4. Select SQL validation and, optionally, command-injection/XSS validation manifests.
   Set **pairs per evaluation set to 20** to include the full SQL split (40 prompts).
   Prepare the sample, inspect its exact prompts, then run the matrix. All inputs,
   including BASE mistakes, contribute to accuracy, corrections, regressions,
   false alarms and misses.
5. Freeze settings before selecting test manifests. Export the HTML matrix and full
   JSON measurements. Use **Code only** when comparing calibration across families.

The transfer matrix measures a fixed next-token decision, including baseline failures.
It reports hits, false alarms, discrimination and response criterion. This does not
reproduce the former free-response rubric or prove vulnerability understanding.
Both descriptions appear in the prompt; the expected answer remains evaluator-only.
Check **Changed inputs** and measured residual changes before interpreting accuracy.
The matrix applies frozen layer vectors, rather than rescaling individual features.
The neutral three-sentence instructions embedded in validation/test material are
quoted as data inside the classification prompt, not used as the response task.

For a generated-code audit with per-token observations, use **7 · Extra**.
Its feature selection observes activations; it does not select individual interventions.
For full explanations, paste a validation/test input into **Base vs. steered** and
inspect mechanism, consequence and recommendation. A larger activation, more
security vocabulary, or a longer answer is not evidence of improved analysis.

See the [capture and transfer guide](../../docs/code_regions_transfer.md) for the exact
selection, intervention, denominators and limitations. The former A/B diagnostic
has been removed from Extra; its research API still supports legacy experiments.

## Evaluation protocol and preserved reference examples

Use only the 20 training pairs to select features or build vectors. Use validation
to select capture scope, layer, vector builder, intervention strength and direction.
Freeze those choices before model evaluation on test. Reloading the reduced
manifest changes its fingerprint; vectors built from an earlier selection must be
rebuilt and compared under a distinct calibration name.

Report BASE and STEERED on the same full 40 test prompts, including baseline
mistakes, corrections, regressions, false alarms and misses. Also inspect paired
outcomes and results by `template_group`; treat the pair as the smallest sampling
unit and the template group as a dependence cluster for uncertainty estimates.
Do not split A/B conditions or distribute variants of a new group across splits.
Predeclare further model evaluations on new data if test observations lead to
another tuning round. The six-pair feature-evidence panel is a sampled diagnostic;
the transfer matrix can include all 20 pairs per split.

The tables below document the preserved v2 reserved examples. References for the
added examples are available for researchers in `dataset_metadata.json`.

| Validation pair | A: expected explanation | B: expected explanation |
| --- | --- | --- |
| `cursor_lookup` | Cursor binds handle separately | Raw handle is inserted into executed SQL; bind it |
| `named_binding` | Dictionary binds :zone | Formatting places zone into SQL syntax; bind it |
| `normalized_input` | Normalized city remains bound | Trimming/lowercasing does not neutralize SQL syntax; bind it |
| `prefix_search` | LIKE pattern is bound data | Pattern is placed inside SQL text; bind it |

Choose settings on the full **40 validation prompts**, then freeze the layer and
strength before evaluating the **40 test prompts**. Keep baseline mistakes in the
review so that both corrections and regressions remain visible.

| Test pair | A: expected explanation | B: expected explanation |
| --- | --- | --- |
| `allowlisted_identifier` | Exact table allowlist plus bound name protects this path | The table allowlist does not protect interpolated name |
| `display_interpolation` | Display interpolation is separate from the bound query | The executed query also interpolates tag |
| `two_values` | Both region and status are bound | Region is bound but status is still interpolated |
| `named_reuse` | Both :value occurrences use one bound value | Both occurrences are formatted into SQL text |
| `substring_search` | The f-string builds a bound LIKE value | The pattern is then interpolated into SQL syntax |
| `query_reassignment` | The executed query retains its bound placeholder | Reassignment overwrites the placeholder query with interpolated SQL |

Held-out improvements need replication on more independently written cases.
Coherence can be assessed separately through the existing section 5 benchmarks.
The Extra generation report contains observations; section 6 exports the current
live vectors/model.

## References

- [MITRE CWE-89: SQL injection](https://cwe.mitre.org/data/definitions/89.html)
- [Python sqlite3: placeholders](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries)
