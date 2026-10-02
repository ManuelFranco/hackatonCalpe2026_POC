# C CWE Paired-Snippet Dataset v2

This dataset contains 20 CWE categories. Each `cwe_<id>/` directory contains 20 matched pairs of very small C snippets:

- `cwe_<id>_A_01.txt` ... `cwe_<id>_A_20.txt`: safe / checked versions
- `cwe_<id>_B_01.txt` ... `cwe_<id>_B_20.txt`: intentionally CWE-vulnerable versions
- `manifest.json`: per-CWE metadata and A/B pair mapping

The A/B snippets are deliberately near-duplicates so the security-relevant code difference is small. Variants change identifiers, constants, buffer sizes, messages, or sample tokens while retaining the same CWE pattern.

The vulnerable samples are designed for defensive research, teaching, static-analysis/classification experiments, and isolated lab use. They demonstrate local coding flaws and do not contain shell-spawning payloads, persistence, credential theft, or real-world exploitation workflows.

Root `manifest.json` and `manifest.csv` provide a dataset-wide index.
