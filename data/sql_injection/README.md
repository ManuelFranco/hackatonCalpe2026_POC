# SQL Injection — Mechanism and Repair

A text-only use case for the standard manifest → common profile → vectors →
base/steered workflow. The output is a short security review describing the input
flow, its consequences and an appropriate repair or preservation of binding.

**A:** parameterized query plus an explanation of its protection.
**B:** interpolated query plus an explanation of the vulnerability and its repair.
The scope is SQL injection in the shown SQLite query, not general program safety.

## Dataset

| Manifest | Pairs | Contents | Purpose |
| --- | ---: | --- | --- |
| `manifest.json` | 12 | Code + short reference review | Build the profile and vectors |
| `manifest.validation.json` | 4 | Code + short review instruction | Choose settings on unseen examples |
| `manifest.test.json` | 6 | Code + short review instruction | Evaluate frozen settings |

```text
sql_injection/
├── README.md
├── manifest.json
├── manifest.validation.json
├── manifest.test.json
├── train/          # 24 code-and-review texts
├── validation/     # 8 prompts without reference answers
└── test/           # 12 prompts without reference answers
```

Training uses one controlled contrast: a bound `?` parameter in A and f-string
interpolation in B. Each matched pair keeps the same function, identifiers and
query purpose. Twelve table/column combinations and four matched review phrasings
reduce dependence on one wording; these remain closely related synthetic examples,
not twelve independent demonstrations of general security understanding.

Both sides mention SQL injection and binding. There are no categorical answer
labels in the training text, and neither side includes its filename or A/B ID.
The changed code and review carry the contrast.

## Why `all + mean`

Set **Profile tokens = all** and **Token aggregation = mean**. The existing
capture averages SAE activations across the input tokens. Including a brief
reference review in the training text lets that average include security-relevant
explanations as well as code syntax. This is a design hypothesis, not a measured
guarantee of more interpretable features.

Generated answers are not included in this capture. Asking for a longer answer
alone would not make the current profile average over that answer. The reference
reviews are intentionally input text during calibration; they are omitted from
validation/test prompts to avoid giving the model the answer during evaluation.

Use **Mean B − A (all features)** for the initial vector construction. This is a
second average, across matched pairs, distinct from averaging tokens within each
input. Keep the existing implementation and UI unchanged.

## Load on the server

1. Replace `data/sql_injection/` with this directory, including its three manifests.
   Remove the previous training files when synchronizing to avoid leaving old data.
   For a custom `GEMMA_DATA_ROOT`, place `sql_injection/` under that root.
2. Reload **`sql_injection/manifest.json`** in section 1. Its name is now
   **SQL injection · Mechanism and repair**. Rebuild the profile and vectors:
   existing session captures still contain the previous dataset.
3. Use seed **0**, temperature **0**, max new tokens **160**, profile tokens **all**,
   aggregation **mean**, and every steering strength **0** initially.
4. In section 2, build the common profile. In section 3, create vectors using
   **Mean B − A (all features)**.

Validation and test manifests can be inspected with the normal loader and preview.
Do not use them to build the training profile. Loading another manifest clears
existing vectors, so copy their individual text files into section 4 for evaluation
without replacing the training manifest. Keep unrelated manifests and manual pairs
out of this profile.

## First comparison

Paste the full contents of
[`validation/named_binding_B.txt`](validation/named_binding_B.txt) into section 4,
leave the image empty and run the normal base/steered comparison. The exact shared
instruction is:

> SQLite. db is trusted; other arguments are untrusted strings. In three sentences, explain the executed query's input handling, its SQL-injection implications, and any necessary fix.

Expected **substance** for B, not an exact-match answer:

> The function formats the untrusted zone value directly into the SQL string.
> That value can alter SQL syntax, creating an SQL-injection vulnerability.
> Replace the formatted literal with a SQLite placeholder and pass zone separately
> as a bound parameter.

Then use
[`validation/named_binding_A.txt`](validation/named_binding_A.txt). Expected substance:

> The function supplies zone through the parameter dictionary for the :zone placeholder.
> SQLite treats the value as data, protecting this query from SQL injection through zone.
> Preserve the fixed query and separate binding; interpolation is not a necessary fix.

At zero strength, base and steered outputs should match. Their correctness is a
separate question: identical wrong explanations are still base errors.

Next try **layer 17 only** at **+0.5**, **+1**, **−0.5**, **−1**. Keep every other
layer at zero. If exploring layer 22, first reset layer 17 to zero and repeat the
same values with layer 22 alone. These are candidate doses, not validated settings.
Use the same prompts and generation settings throughout.

Positive B − A emphasizes the contrast toward vulnerable-code-and-repair reviews;
negative strength applies the opposite direction. It does not follow that positive
strength improves reviews or that negative strength preserves correct judgments.
Measure grounding, not the number of security-related words.

## What counts as improvement

Read both complete answers and compare three aspects:

- **Mechanism:** identifies the actual external value and whether the executed
  query interpolates it or binds it separately. On mixed examples it identifies
  which argument remains interpolated.
- **Consequence:** explains whether this specific input path can change SQL
  syntax. Does not confuse LIKE wildcard matching, display text or an allowlisted
  identifier with unprotected interpolation into SQL.
- **Recommendation:** proposes a placeholder plus a separate parameter value
  when needed; preserves correct binding when already present. Does not claim
  that a whole application is secure or add unrelated vulnerabilities.

For a compact manual record, rate each aspect **0 = wrong/missing**,
**1 = broadly correct but vague/incomplete**, **2 = correct and tied to the code**.
Report their **mean on a 0–2 scale** across all validation cases alongside concrete
errors. This manual review mean is separate from SAE token aggregation. An
unsupported vulnerability claim on a protected case remains a regression even
if the answer is longer or its average score rises. Check readability and the
three-sentence instruction separately; wording need not match the examples above.

Keep a nonzero setting only if reviews become more grounded overall without
additional invented vulnerabilities or broken repairs. Otherwise retain zero.
If the base already explains every case well, there may be no improvement to show
on this sample. A change of wording alone is not evidence of a better security review.

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
