"""Escaped generation reports and token-level observation maps."""

import json
from hashlib import sha256

from .evidence_view import EVIDENCE_CSS, esc


GENERATION_CSS = """
.el .el-heat th {text-transform:none;letter-spacing:0;}
.el .el-heat {width:max-content;min-width:100%;table-layout:fixed;}.el .el-heat th,.el .el-heat td {padding:7px 5px;min-width:45px;text-align:center;font-size:10px;}.el .el-heat th:first-child {position:sticky;left:0;background:var(--el-panel);min-width:105px;text-align:left;z-index:1;}.el .el-heat small {display:block;color:var(--el-muted);font-size:9px;margin-bottom:4px;}.el .el-code {max-height:620px;white-space:pre;overflow:auto;padding:12px 0;}.el-code-line {display:block;padding:0 12px;min-height:20px;border-left:3px solid transparent;scroll-margin-top:100px;}.el-code-line:target {outline:2px solid #7470ec;outline-offset:-2px;}.el-code i {display:inline-block;width:32px;margin-right:12px;color:var(--el-muted);font-style:normal;user-select:none;}.el-code .el-sql-dynamic {background:rgba(222,78,84,.19);border-left-color:#de4e54;}.el-code .el-sql-unresolved {background:rgba(221,154,61,.16);border-left-color:#cb903d;}.el-code .el-sql-literal {background:rgba(21,148,119,.09);border-left-color:#159477;}.el .el-generation-pair {grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));}
.el-shade {position:absolute;opacity:0;width:1px;height:1px;}.el-shade + label {display:inline-block;padding:7px 12px;margin:3px 5px 3px 0;border:1px solid var(--el-border);border-radius:8px;cursor:pointer;font-size:12px;}.el-shade:checked + label {background:rgba(111,101,232,.14);border-color:#7470ec;}.el-shade:focus-visible + label {outline:2px solid #7470ec;outline-offset:2px;}.el-shade:disabled + label {opacity:.4;cursor:not-allowed;}.el .el-sql-key {display:inline-flex;gap:6px;align-items:center;font-size:11px;margin-right:13px;}.el-sql-key b {width:10px;height:10px;border-radius:3px;}.el-feature-key {display:none;}.el-ready a.el-badge {text-decoration:none;}
"""


def hero():
    return """<div class="el"><div class="el-hero"><div><div class="el-kicker">7 · Extra / Generation audit</div>
    <h2>Follow the response, token by token.</h2><p>Inspect code. Observe features. Test causal effects separately.</p></div>
    <div class="el-flow"><span>01<b>Generate</b></span>→<span>02<b>Observe</b></span>→<span>03<b>Validate</b></span></div></div></div>"""


def empty():
    return """<div class="el"><div class="el-empty"><h3>One response. Its code and its activations.</h3>
    <p>Use Base alone, or compare with your current vectors and shared layer strengths.</p>
    <p>Up to 3 observed features · one layer · every generation step</p></div></div>"""


def line_activity(run, width):
    trace, text = run["trace"], run["response"]
    spans = trace.get("text_spans", [])
    if not spans or len(spans) != len(trace["after"]):
        return None
    previous = 0
    for span, values in zip(spans, trace["after"]):
        if (
            len(span) != 2
            or not previous <= span[0] <= span[1] <= len(text)
            or len(values) != width
        ):
            return None
        previous = span[1]
    lines, offset = [], 0
    for line in text.splitlines(keepends=True):
        indices = [
            i
            for i, (a, b) in enumerate(spans)
            if a < offset + len(line) and b > offset and a < b
        ]
        lines.append(
            {
                "peaks": [
                    max((trace["after"][i][j] for i in indices), default=0)
                    for j in range(width)
                ],
                "tokens": f"{indices[0] + 1}–{indices[-1] + 1}" if indices else "None",
            }
        )
        offset += len(line)
    return lines


def sql_line_classes(review):
    kinds = {
        "Literal query": (1, "literal"),
        "Unresolved query": (2, "unresolved"),
        "Dynamic query · review": (3, "dynamic"),
    }
    lines = {}
    for finding in review["findings"]:
        level = kinds[finding["kind"]]
        default = [finding["line"], finding["line"]]
        source = finding.get(
            "query_lines",
            [finding.get("assignment", {}).get("line", finding["line"])] * 2,
        )
        for start, end in [source, finding.get("call_lines", default)]:
            for line in range(start, end + 1):
                if level[0] > lines.get(line, (0, ""))[0]:
                    lines[line] = level
    return {line: kind for line, (_, kind) in lines.items()}


def overview(result, behavioral=False):
    panels, features = [], result["features"]
    scope = "shade-" + sha256(str(result["id"]).encode()).hexdigest()[:12]
    activity = [line_activity(r, len(features)) for r in result["runs"]]
    scales = [
        max(
            (line["peaks"][j] for mapped in activity if mapped for line in mapped),
            default=0,
        )
        or 1
        for j in range(len(features))
    ]
    choices = f'<input class="el-shade" type="radio" name="{scope}" id="{scope}-sql" checked><label for="{scope}-sql">SQL review</label>'
    rules = []
    for j, feature in enumerate(features):
        choices += f'<input class="el-shade" type="radio" name="{scope}" id="{scope}-{j}" {"" if any(activity) else "disabled"}><label for="{scope}-{j}">Feature #{feature}</label>'
        rules.append(
            f"#{scope}-{j}:checked ~ .el-generation-pair .el-code-line {{background:rgba(111,101,232,var(--obs-{j},0));border-left-color:rgba(111,101,232,var(--obs-{j},0));}} #{scope}-{j}:checked ~ .el-legend .el-sql-key {{display:none;}} #{scope}-{j}:checked ~ .el-legend .el-feature-key {{display:inline;}}"
        )
    for index, run in enumerate(result["runs"]):
        review, trace = run["review"], run["trace"]
        flags, code = sql_line_classes(review), []
        for i, line in enumerate(run["response"].splitlines(), 1):
            measured = activity[index][i - 1] if activity[index] else None
            peaks = measured["peaks"] if measured else [0] * len(features)
            style = ";".join(
                f"--obs-{j}:{max(0, value) / scales[j] * 0.55:.4f}"
                for j, value in enumerate(peaks)
            )
            title = f"L{i} · {flags.get(i, 'No SQL site identified')}"
            if measured:
                title += f" · predicted tokens {measured['tokens']} · " + ", ".join(
                    f"#{feature}: max {value:.6g}"
                    for feature, value in zip(features, peaks)
                )
            code.append(
                f'<span class="el-code-line {"el-sql-" + flags[i] if i in flags else ""}" id="{scope}-r{index}-l{i}" style="{style}" title="{esc(title)}"><i>{i}</i>{esc(line)}</span>'
            )
        code = "".join(code)
        findings = (
            "".join(
                f'<a class="el-badge" href="#{scope}-r{index}-l{f["line"]}">L{f["line"]} · {esc(f["kind"])}'
                + (
                    f" · {esc(f['assignment']['name'])} ← L{f['assignment']['line']}"
                    if "assignment" in f
                    else ""
                )
                + "</a> "
                for f in review["findings"]
            )
            or '<span class="el-small">No execute / executemany / executescript calls identified.</span>'
        )
        warning = ""
        if features and not activity[index]:
            warning += '<div class="el-warning">Feature shading unavailable: exact token/text alignment was not established.</div>'
        if trace["termination"] == "Token limit" or any(
            b["unclosed_fence"] for b in review["blocks"]
        ):
            warning += '<div class="el-warning">Output may be incomplete. Inspect the ending.</div>'
        errors = "".join(
            f'<div class="el-warning">Block at L{b["start_line"]}: {esc(b["error"])}</div>'
            for b in review["blocks"]
            if "error" in b
        )
        checks = ""
        if behavioral:
            check = run["checks"]
            normal = check.get("normal_checks", [])
            passed = sum(c["status"] == "Passed" for c in normal)
            failed = sum(c["status"] in {"Failed", "Error"} for c in normal)
            unsupported = sum(c["status"] == "Unsupported" for c in normal)
            ordinary = (
                f"{passed}/{len(normal)} passed · {failed} failed · {unsupported} unsupported"
                if normal
                else check.get("functional_status", "Not evaluated")
            )
            reason = check.get("error") or next(
                (c["error"] for c in normal if c.get("error")),
                check.get("probe", {}).get("error", ""),
            )
            checks = (
                '<div class="el-ready">'
                f'<span class="el-badge">Ordinary inputs: {esc(ordinary)}</span>'
                f'<span class="el-badge">Evaluation: {esc(check.get("status", "Legacy evaluation"))}</span>'
                f'<span class="el-badge">{esc(check["behavior"])}</span></div>'
                + (f'<p class="el-note">{esc(reason)}</p>' if reason else "")
            )
        panels.append(
            f'<div class="el-card"><div class="el-toolbar"><h3>{esc(run["name"])}</h3><span class="el-badge">{len(trace["tokens"])} tokens · {esc(trace["termination"])}</span></div>'
            f'<div class="el-ready"><span class="el-badge">Python syntax: {review["syntax"]}</span><span class="el-badge">{review["dynamic_queries"]} dynamic · {review["unresolved_queries"]} unresolved queries</span></div>'
            f'{checks}{warning}{errors}<pre class="el-code">{code}</pre><div class="el-ready">{findings}</div></div>'
        )
    protocol = (
        "Bounded interpreter · in-memory SQL fixture"
        if behavioral
        else "Static review · code never executed"
    )
    note = (
        "Behavior checks cover three ordinary lookups and one SQL probe in a read-only fixture. Generated Python is interpreted through a limited syntax whitelist; unsupported code is inconclusive. Shading is observed activity, not causal attribution."
        if behavioral
        else "SQL review covers direct calls and immediately preceding assignments. Colors do not establish exploitability or causal influence. Other data flow is unresolved. Code execution and functional tests: not run."
    )
    return (
        f'<div class="el"><div class="el-toolbar"><h3>Generated responses · code shading</h3><span class="el-badge">{protocol}</span></div>'
        f"<style>{''.join(rules)}</style>{choices}"
        f'<div class="el-ready el-legend"><span class="el-sql-key"><b style="background:#de4e54"></b>Dynamic SQL · review</span><span class="el-sql-key"><b style="background:#cb903d"></b>Unresolved</span><span class="el-sql-key"><b style="background:#159477"></b>Literal SQL</span><span class="el-feature-key">Purple = measured activity after layer {result["layer"]} · line maximum · same scale across responses · not causal attribution</span></div>'
        f'<div class="el-split el-generation-pair">{"".join(panels)}</div>'
        f'<p class="el-note">{note}</p></div>'
    )


def activity_start(result, run_name="Base", window=32):
    trace = next(r["trace"] for r in result["runs"] if r["name"] == run_name)
    peak = max(
        range(len(trace["after"])), key=lambda i: max(trace["after"][i], default=0)
    )
    return max(1, min(peak + 1 - window // 2, len(trace["after"]) - window + 1))


def timeline(result, run_name="Base", start=None, window=32):
    run = next(r for r in result["runs"] if r["name"] == run_name)
    trace, ids = run["trace"], result["features"]
    start = activity_start(result, run_name, window) if start is None else start
    start = max(0, min(int(start) - 1, len(trace["tokens"]) - 1))
    end = min(start + window, len(trace["tokens"]))
    if not ids:
        return '<div class="el"><div class="el-empty">No active features found in Base.</div></div>'
    headers = "".join(
        f'<th title="{esc(trace["tokens"][i])} · token ID {trace["token_ids"][i]}"><small>{i + 1}</small>{esc(trace["tokens"][i][:9])}</th>'
        for i in range(start, end)
    )
    rows = []
    activity = []
    for j, feature in enumerate(ids):
        values = [row[j] for row in trace["after"]]
        peak = max(range(len(values)), key=values.__getitem__)
        active = sum(value > 0 for value in values)
        peak_label = f"peak token {peak + 1}" if active else "inactive"
        activity.append(
            f'<span class="el-badge">#{feature} · active {active}/{len(values)} · {peak_label}</span>'
        )
        scale = (
            max(
                (
                    row[j]
                    for r in result["runs"]
                    for key in ("before", "after")
                    for row in r["trace"][key]
                ),
                default=0,
            )
            or 1
        )
        for key in ["before", "after"] if run_name != "Base" else ["after"]:
            label = key.title() if run_name != "Base" else "Activity"
            cells = "".join(
                f'<td style="background:rgba(111,101,232,{max(0, row[j]) / scale * 0.55:.3f})" title="#{feature} · {label} · token {i + 1}: {row[j]:.6g}">{row[j]:.2g}</td>'
                for i, row in enumerate(trace[key][start:end], start)
            )
            rows.append(
                f"<tr><th>#{feature}<small>{label} · max {scale:.3g}</small></th>{cells}</tr>"
            )
    return (
        f'<div class="el"><div class="el-toolbar"><h3>Layer {result["layer"]} · {esc(run_name)}</h3><span class="el-badge">Tokens {start + 1}–{end} · observed features</span></div>'
        f'<div class="el-ready">{"".join(activity)}</div>'
        f'<div class="el-table-wrap"><table class="el-heat" aria-label="Feature activation timeline"><thead><tr><th>Feature</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="el-note">Each column is the state predicting the shown token. Before/after refers to the addition at this layer; earlier-layer effects are already present. Each feature uses one scale across both runs. Different responses have different prefixes, so columns across runs are not matched counterfactuals.</p></div>'
    )


def metadata(result):
    active = (
        ", ".join(f"L{k}: {v:g}" for k, v in result["strengths"].items() if v) or "None"
    )
    position = (
        "Final position at EVERY generation step"
        if result["steer_last_token_only"]
        else "All prefill positions; current position at each cached decode step"
    )
    fields = {
        "Model / SAE": f"{result['model']} · {result['sae_release']}/{result['sae_id']}",
        "Observed features": ", ".join(f"#{j}" for j in result["features"])
        or "No active features",
        "Selection": result["selection"] + ". Observation is not a causal ranking.",
        "Intervention": f"Existing vectors · {active}. Feature selection does not change the intervention.",
        "Schedule": position if result["mode"] != "Base only" else "No steering",
        "Vector / profile": f"{result['vector_id']} / {result['profile_id']}",
        "Generation": f"Seed {result['settings']['seed']} · temperature {result['settings']['temperature']} · token limit {result['settings']['max_new_tokens']}",
    }
    return (
        '<div class="el"><details><summary>Run identity & interpretation</summary><dl>'
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in fields.items())
        + f"</dl><details><summary>Exact prompt</summary><pre>{esc(result['prompt'])}</pre></details></details></div>"
    )


def standalone_report(result):
    maps = "".join(
        timeline(result, run["name"], start + 1)
        for run in result["runs"]
        for start in range(0, len(run["trace"]["tokens"]), 32)
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Generation audit · {esc(result['id'][:8])}</title><style>body{{background:#f5f7fb;color:#182236;padding:28px;}}main{{max-width:1200px;margin:auto;}}{EVIDENCE_CSS}{GENERATION_CSS}</style>"
        f'<main>{hero()}{overview(result)}{maps}{metadata(result)}<div class="el"><details><summary>Vector metadata</summary><pre>{esc(json.dumps(result["vector_metadata"], indent=2))}</pre></details></div></main></html>'
    )
