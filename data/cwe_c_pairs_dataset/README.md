# CWE code safety manifest

Twenty matched C examples. A is the normal/reference implementation and B is the
vulnerable/comparison implementation. `manifest.json` is the dashboard's version 1
manifest. `manifest.csv` retains the source dataset's tabular metadata.

Load the repository manifest in section 1, or upload the same JSON: its `asset_root`
is relative to the configured data root, so text-file paths resolve in both cases.
The dashboard reads the C programs as text; it does not compile or execute them.
