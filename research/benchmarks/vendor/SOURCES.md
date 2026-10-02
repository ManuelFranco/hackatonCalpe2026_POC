# Upstream evaluators

- IFEval: https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval (Apache-2.0).
  Only import paths were changed to relative imports. License is in `ifeval/LICENSE`.
- MMMU: https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py .
  Removed the module-level `random.seed(42)` to avoid changing server RNG state.
  License is in `MMMU_LICENSE`. Only the deterministic open-answer parser/scorer
  is used; multiple-choice responses use the dashboard's strict letter parser.

Do not hand-edit scoring logic. Refresh from a pinned upstream revision and run tests.
