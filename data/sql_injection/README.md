# SQL Injection — Code Flow

A text-only use case for the standard manifest → common profile → vectors →
base/steered workflow. Revision **code-flow-v2** removes reference reviews from
calibration and explicit security cues from evaluation instructions.

**A:** the database call binds request values separately from SQL syntax.
**B:** the executed query contains at least one interpolated request value.
These descriptions are researcher metadata, not part of the model input. The
scope is the shown SQLite execution path, not the safety of an entire application.

## Dataset

| Manifest | Pairs | Model input | Purpose |
| --- | ---: | --- | --- |
| `manifest.json` | 12 | Shared neutral context + code | Build the profile and vectors |
| `manifest.validation.json` | 4 | Neutral context + code-flow task + code | Select settings |
| `manifest.test.json` | 6 | Neutral context + code-flow task + code | Evaluate frozen settings |

All 44 sample files omit reference explanations, vulnerability names, severity
labels, safe/unsafe annotations, YES/NO answers and suggestive function names.
Every pair has identical surrounding text. SQL-specific dataset names and A/B
labels remain outside the captured prompt; the normal loader supplies only the
condition text and optional image to the model.

The shared context preserves the origin of values without naming a security issue:

> SQLite. db is a database connection; all other arguments are strings from an HTTP request.

## Controls against superficial cues

The 12 calibration pairs comprise:

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

This design reduces explicit explanation and formatting cues. It does not remove
all syntactic shortcuts or prove semantic understanding. The examples are small,
related synthetic programs, not twelve independent real-world demonstrations.
The held-out programs preserve cursor, named binding, normalization, LIKE,
allowlisting, mixed values and reassignment cases with distinct identifiers.
Some concepts now also occur in training controls; the splits are held out at the
example level, not wholly unseen mechanism families.

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

1. Replace the full `data/sql_injection/` directory on the server, including all three
   manifests and train/validation/test files. Paths and pair counts remain unchanged.
2. Reload **SQL injection · Code flow** via `sql_injection/manifest.json` in section 1.
   Rebuild the common profile and vectors; old captures still represent code plus
   reference explanations. Clear unrelated manual pairs from this profile.
3. Start with seed **0**, temperature **0**, max new tokens **160**, **all / mean**,
   and steering strengths **0**.
4. In Extra, measure the training pairs. **Full input** and **Code + preamble** should
   now be equivalent apart from any formatting normalization: there is no trailing
   review to remove. **Text without code** is the identical neutral context for A
   and B, a context-only negative control whose paired delta should be zero under
   deterministic inference. Inspect the minimal and preview-control pairs separately
   in the detailed table; aggregate means can hide failure on the latter.
5. If revisiting feature 11749, use the exact **layer 22 / 262k** dictionary and
   re-estimate its calibration coefficient. It need not remain a useful candidate.

## Evaluate without naming the vulnerability

Load `manifest.validation.json` in **Extra**, retaining the training profile, or
paste individual validation files into **Base vs. steered**. Do not replace the
training manifest in section 1 to evaluate: that invalidates its profile/vectors.

Use **Code + preamble** in Extra and leave the shared instruction blank. The
validation/test files already contain this identical neutral instruction:

> In three sentences, explain how the supplied values reach the database call and affect the executed query. Suggest a change only if needed.

Start with `validation/named_binding_A.txt` and its B counterpart. In A, a grounded
answer identifies the `:zone` placeholder and dictionary binding and preserves it.
In B, it identifies that zone is formatted into the executed SQL, explains that
its contents can alter SQL syntax, and recommends separate binding. An answer
may name SQL injection, but that phrase is never supplied in the prompt.

First run at dose **0**, with ablation off. Base and additive-control responses
should match; this does not establish correctness. Then compare **±0.5** and **±1**
in Extra with random controls and optional ablation. For whole-vector experiments,
change one layer at a time and keep the others at zero. These are exploration
settings, not validated improvements.

Rate **mechanism**, **consequence** and **recommendation** individually:
**0 = wrong/missing**, **1 = partial**, **2 = correct and grounded in the code**.
Use their mean on a 0–2 scale; report paired changes against base and random
controls and retain concrete errors. An unsupported vulnerability claim on A is
a regression. More security vocabulary, a longer answer, or a larger feature
activation alone does not demonstrate improvement.

Freeze settings and the rubric before using the test split. Keep the original
base mistakes visible so that corrections and regressions can both be measured.
No Boolean classifier, extra loader or custom inference path is required.

## Reserved examples and reference mechanisms

| Validation pair | A: expected explanation | B: expected explanation |
| --- | --- | --- |
| `cursor_lookup` | Cursor binds handle separately | Raw handle is inserted into executed SQL; bind it |
| `named_binding` | Dictionary binds :zone | Formatting places zone into SQL syntax; bind it |
| `normalized_input` | Normalized city remains bound | Trimming/lowercasing does not neutralize SQL syntax; bind it |
| `prefix_search` | LIKE pattern is bound data | Pattern is placed inside SQL text; bind it |

Choose settings on these eight prompts, then freeze the layer and strength before
opening the twelve test prompts below. Keep base mistakes in the review so that
both corrections and regressions remain visible.

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
No new interface, runner, scorer or model export is required by this dataset.

## References

- [MITRE CWE-89: SQL injection](https://cwe.mitre.org/data/definitions/89.html)
- [Python sqlite3: placeholders](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries)
