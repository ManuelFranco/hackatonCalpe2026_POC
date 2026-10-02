# SQL Injection — Paired Code Dataset

Standard version 1 manifests for the usual dashboard workflow. Each text file is
one complete, copyable prompt: shared instructions followed by a Python/SQLite
function. **A = protected against the shown SQL-injection path (NO)**;
**B = vulnerable to SQL injection (YES)**. The target is CWE-89 only, not whether
an entire program is secure.

## Files

| Manifest | Pairs | Use |
| --- | ---: | --- |
| `manifest.json` | 12 | Build the common profile and vectors |
| `manifest.validation.json` | 4 | Inspect validation pairs; choose steering settings |
| `manifest.test.json` | 6 | Inspect held-out pairs; evaluate frozen settings |

```text
sql_injection/
├── README.md
├── manifest.json
├── manifest.validation.json
├── manifest.test.json
├── train/          # 24 complete A/B prompts
├── validation/     # 8 complete A/B prompts
└── test/           # 12 complete A/B prompts
```

Only `manifest.json` goes into profile building. The other two manifests use the
same loader and visual preview as any use case. Loading them replaces the input
selection and clears the current profile/vectors. To evaluate without clearing
vectors, copy a validation/test `.txt` file into the ordinary section 4 prompt.
Do not include manual pairs or other manifests in this experiment's profile.

Paths use `asset_root: "sql_injection"`. If `GEMMA_DATA_ROOT` is customized, place
this directory under that data root.

## First experiment: two explicit decisions

1. Set shared controls: **seed 0**, **temperature 0**, **max new tokens 8**,
   **profile tokens all**, **aggregation mean**. Set **all steering strengths to 0**.
2. In section 1, load **`sql_injection/manifest.json`**.
3. In section 2, click **Build common profile**. In section 3, choose
   **Mean B − A (all features)** and click **Create vectors**.
4. In section 4, leave the image empty and paste all of
   [`validation/named_binding_B.txt`](validation/named_binding_B.txt). Click
   **Run base + steered**. The correct answer is **YES** for both.
5. Replace the prompt with all of
   [`validation/named_binding_A.txt`](validation/named_binding_A.txt) and run again.
   The correct answer is **NO** for both.
6. Run both prompts at layer **17 = +0.5**, then **+1**, **−0.5**, **−1**. Keep
   layers 9, 22 and 29 at zero. Do not change the prompt or generation controls.
   If investigating layer 22 next, reset layer 17 to zero first and repeat the
   same four values with layer 22 alone.

The shared question is:

> Can an attacker-controlled value change SQL syntax in an executed query?

For the protected prompt, the external string is passed separately to SQLite:

```python
query = "SELECT id FROM assets WHERE zone = :zone"
return db.execute(query, {"zone": zone}).fetchall()
```

For the vulnerable prompt, it is placed directly in the executable SQL:

```python
query = "SELECT id FROM assets WHERE zone = '{}'".format(zone)
return db.execute(query).fetchall()
```

A and B are separate prompts, not two snippets to concatenate into one request.
References, filenames and these explanations should not be added to the prompt.
The `.txt` files already contain all necessary context and the exact YES/NO instruction.

## What the results mean

- **Zero steering:** base and steered responses should match. Correct decisions
  are B → YES and A → NO. If both match but are wrong, that is a base error, not
  evidence of a steering problem. If the two responses differ at zero, resolve
  the comparison/runtime issue before interpreting nonzero steering.
- **Nonzero steering:** the correct answers remain the same. A useful detector
  change corrects a base mistake without creating additional false positives.
  A protected A changing from NO to YES is a false positive.
- **Direction:** B − A points from parameterized-query examples toward vulnerable
  examples. Positive strength may increase YES answers and negative strength may
  decrease them. This is a hypothesis to measure, not an established outcome.
- **No visible change:** the sampled output did not change on these cases. That
  alone does not establish that the internal activations were unaffected.
- **A perfect base:** there is no accuracy improvement to demonstrate on that
  sample. A changed answer can demonstrate an intervention effect, but errors
  introduced by steering are not better detection.

## Choose settings on validation, then freeze them

After the first two prompts, run all eight validation prompts at zero and at the
candidate strengths. All expected answers are fixed below. Count exact YES/NO
responses; treat extra prose or any other output as an invalid answer, not as a
correct prediction inferred from a keyword.

| Validation pair | A | B | What it checks |
| --- | --- | --- | --- |
| `cursor_lookup` | NO | YES | Binding through a cursor |
| `named_binding` | NO | YES | Named placeholders versus string formatting |
| `normalized_input` | NO | YES | Trimming/lowercasing does not prevent syntax injection |
| `prefix_search` | NO | YES | A bound LIKE pattern is a value, not SQL syntax |

A candidate qualifies only if it has **more correct answers than base**, **no
additional false positives**, and **no additional invalid answers** across all
validation cases. Keep base errors in the comparison so recovered mistakes are
visible. If no candidate qualifies, keep steering at zero. Prefer the smaller
absolute strength when qualifying settings tie.

Freeze the chosen layer and strength before opening the twelve test prompts.
Do not retune from their results or use them for profile capture.

| Test pair | A | B | Control |
| --- | --- | --- | --- |
| `allowlisted_identifier` | NO | YES | Exact identifier allowlist plus a bound value |
| `display_interpolation` | NO | YES | Display f-string versus executed SQL f-string |
| `two_values` | NO | YES | Binding one input does not protect another interpolated input |
| `named_reuse` | NO | YES | Repeated named parameter versus repeated interpolation |
| `substring_search` | NO | YES | Interpolation inside a bound LIKE value is not SQL syntax injection |
| `query_reassignment` | NO | YES | Judge the query actually executed after reassignment |

The desired result is a held-out gain that preserves protected cases, not simply
more YES responses. If validation or test does not improve, record that result;
these small synthetic partitions do not establish general vulnerability-detection
performance. Coherence can then be measured with the existing section 5 workflow.

## References

- [MITRE CWE-89: SQL injection](https://cwe.mitre.org/data/definitions/89.html)
- [Python sqlite3: placeholders](https://docs.python.org/3.12/library/sqlite3.html#how-to-use-placeholders-to-bind-values-in-sql-queries)
