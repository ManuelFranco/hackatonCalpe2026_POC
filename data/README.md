# Paired Datasets

## Military vehicle condition

[`military_vehicle_condition/`](military_vehicle_condition/) provides a version 1 image-only manifest with 20 A/B slots and separate image folders: **A = not destroyed**, **B = destroyed**. Add your own images before loading it. See its [setup instructions](military_vehicle_condition/README.md) for filenames and pairing conventions.

## C CWE Paired Code Dataset

This dataset has 20 CWE categories and 20 matched safe/vulnerable C examples per category.

- [`scripts/`](scripts/) contains complete C snippets and per-CWE manifests. The top-level `manifest.json` and `manifest.csv` index these full snippets.
- [`code_lines/`](code_lines/) contains a parallel line-only dataset. Each A/B text file has exactly one original source line. Its own `manifest.json` and `manifest.csv` index the excerpts; each per-CWE manifest records original source paths, line numbers, and extraction semantics.

The full examples are near-duplicates with a small security-relevant change. The line-only data makes selected statements easier to inspect while dropping surrounding context. For flaws caused by a missing check or cleanup, A shows that omitted guard/cleanup and B shows the affected operation or exit statement. Such excerpts can serve as focused cues, but do not preserve control flow or establish vulnerability on their own.

The examples are intended for defensive research, teaching, and isolated static-analysis/classification experiments. They demonstrate local coding flaws and do not contain shell-spawning payloads, persistence, credential theft, or real-world exploitation workflows.
