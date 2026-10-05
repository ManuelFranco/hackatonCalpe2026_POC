# Paired Datasets

## SQL injection

[`sql_injection/`](sql_injection/) contains **CWE-89 Python/SQLite** code-only
pairs: **A = executed query binds values**, **B = executed query interpolates a value**.
Use `all + mean` with the **20 training pairs** in `sql_injection/manifest.json`.
The **20 validation pairs** and **20 test pairs** use a neutral code-flow instruction
without vulnerability names or reference answers. Training includes nine controls
that match SQL candidates, construction components or parameter dictionaries.
All 60 pairs across the three splits pass an offline SQLite behavioral audit. Examples are synthetic
and clustered by template; keep evaluation splits separate from calibration.
Use the normal dashboard workflow; the [dataset instructions](sql_injection/README.md)
provide the coverage, evaluation protocol and audit commands.

## Command injection and XSS

[`command_injection/`](command_injection/) and [`xss/`](xss/) each contain
6 training, 4 validation and 4 test A/B pairs. Command-injection examples compare
separate subprocess arguments or POSIX shell quoting with values entering shell
syntax. XSS examples compare escaped and raw values in HTML text content.
The dashboard reads these snippets as data; it never runs them.

Build a profile with **Code only**, freeze the resulting vectors and compare them
across reserved families in **5 · Benchmarks & transfer**. See the
[capture and transfer guide](../docs/code_regions_transfer.md) for the workflow,
matrix readout and interpretation limits.

## Physical vehicle damage

[`physical_damage/`](physical_damage/) contains 16 matched image pairs:
**A = intact**, **B = physically damaged**. Its version 1 manifests preserve
8 training, 3 validation and 5 test pairs. Select `physical_damage/manifest.json`
to build a profile from training images; see the [dataset instructions](physical_damage/README.md)
for evaluation images and steering direction.

## Military vehicle condition

[`military_vehicle_condition/`](military_vehicle_condition/) provides a version 1 image-only manifest with 20 A/B slots and separate image folders: **A = not destroyed**, **B = destroyed**. Add your own images before loading it. See its [setup instructions](military_vehicle_condition/README.md) for filenames and pairing conventions.

## C CWE Paired Code Dataset

This dataset has 20 CWE categories and 20 matched safe/vulnerable C examples per category.

- [`scripts/`](scripts/) contains complete C snippets and per-CWE manifests. The top-level `manifest.json` and `manifest.csv` index these full snippets.
- [`code_lines/`](code_lines/) contains a parallel line-only dataset. Each A/B text file has exactly one original source line. Its own `manifest.json` and `manifest.csv` index the excerpts; each per-CWE manifest records original source paths, line numbers, and extraction semantics.

The full examples are near-duplicates with a small security-relevant change. The line-only data makes selected statements easier to inspect while dropping surrounding context. For flaws caused by a missing check or cleanup, A shows that omitted guard/cleanup and B shows the affected operation or exit statement. Such excerpts can serve as focused cues, but do not preserve control flow or establish vulnerability on their own.

The examples are intended for defensive research, teaching, and isolated static-analysis/classification experiments. They demonstrate local coding flaws and do not contain shell-spawning payloads, persistence, credential theft, or real-world exploitation workflows.
