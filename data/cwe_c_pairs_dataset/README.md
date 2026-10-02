# C CWE Paired-Snippet Dataset

20 paired C examples (40 code files).

- `cwe_<id>_A.txt`: normal / safer implementation
- `cwe_<id>_B.txt`: intentionally vulnerable implementation
- `manifest.csv` / `manifest.json`: labels and CWE metadata

Designed for defensive security research, classroom demonstrations, static-analysis experiments, and isolated/simulated environments. Vulnerable files intentionally contain memory-safety errors, denial-of-service conditions, resource-management flaws, or faulty security checks. They contain no shell-spawning payloads, persistence, credential theft, or real-world exploitation workflow.

Run vulnerable snippets only in an isolated lab. Consult MITRE CWE for authoritative definitions and scope when citing CWE entries in publications.
