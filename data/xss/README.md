# XSS — HTML text code flow

Six training pairs in `manifest.json`; four validation and four test pairs in
separate manifests. A escapes all request values inserted into HTML text; B inserts
at least one value unescaped. Returned strings (or the `html_response` field) are
assumed to be rendered as HTML. A `text_preview` field is display-only text and is
not rendered as HTML. Labels and expected answers stay outside model material.

The scope is **HTML text content**, not attributes, URLs, scripts, template engines,
CSP, sanitization policies or an entire web application's safety. In this context,
`html.escape(value, quote=False)` is valid; quote handling alone does not determine
the label. Test cases cover aliases, reassignment, joining strings and undoing
escaping. The fixtures are synthetic and never executed by the dashboard.

Use **Code only + mean** for calibration, create vectors and freeze a source in
section 5. Evaluate on reserved manifests with the same prompts across sources.
See the [step-by-step guide](../../docs/code_regions_transfer.md). Transfer to other
families remains an empirical question; no feature is claimed to be universal.
