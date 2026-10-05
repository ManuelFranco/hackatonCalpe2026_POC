# Command injection — code flow

Six training pairs in `manifest.json`; four validation and four test pairs in
separate manifests. A preserves request values as command arguments; B places at
least one request value directly into executed shell syntax. All samples use
neutral context and Python code. Labels are evaluator metadata.

Only the displayed path is classified. Executable names and options are fixed;
POSIX shell quoting is assumed. Argument injection into a specific tool, Windows
shell behavior, arbitrary executable selection and the rest of an application are
outside this dataset. No commands are executed by the dashboard.

Use **Code only + mean** for calibration, create vectors, freeze a source in section
5 and evaluate against reserved manifests. See the
[step-by-step guide](../../docs/code_regions_transfer.md). A positive B−A direction
is not assumed to improve detection. The cases are small synthetic templates;
example-level held-out results do not establish real-world generalization.
