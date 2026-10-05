"""Escaped, dependency-free visual reports for the feature evidence lab."""

from html import escape
from statistics import mean

from .evidence_lab import effective_cases, feature_groups

EVIDENCE_CSS = """
.el {--el-ink:var(--body-text-color,#182236);--el-muted:var(--body-text-color-subdued,#596579);--el-border:rgba(128,140,164,.25);--el-panel:var(--block-background-fill,#fff);color:var(--el-ink);font:14px/1.5 system-ui,sans-serif;}
.el * {box-sizing:border-box;} .el h2,.el h3,.el p {margin:0;} .el h2 {font-size:27px;letter-spacing:-.04em;line-height:1.2;} .el h3 {font-size:15px;font-weight:650;}
.el-hero {border-radius:18px;background:linear-gradient(120deg,#14243a,#25305c);padding:27px 30px;color:#f5f7ff;display:flex;align-items:center;justify-content:space-between;gap:24px;}
.el .el-hero h2,.el .el-hero .el-kicker,.el .el-hero .el-flow span {color:#f5f7ff;}.el-hero p {color:#ced8ec;margin-top:8px;max-width:530px;} .el-kicker {font-size:10px;font-weight:750;letter-spacing:.15em;text-transform:uppercase;margin-bottom:9px;opacity:.8;}
.el-flow {display:flex;align-items:center;gap:12px;font-size:11px;color:#d1dbef;white-space:nowrap;}.el-flow b {display:block;color:#fff;font-size:13px;}.el-flow span {border:1px solid #61718b;border-radius:10px;padding:10px 13px;}
.el-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0;}.el-card {background:var(--el-panel);border:1px solid var(--el-border);border-radius:13px;padding:17px;min-width:0;}
.el-label {color:var(--el-muted);font-size:11px;letter-spacing:.035em;font-weight:600;}.el-value {font-size:30px;letter-spacing:-.05em;font-weight:720;line-height:1.2;margin:6px 0;}.el-small {font-size:11px;color:var(--el-muted);}.el-positive {color:#159477;}.el-negative {color:#c67528;}
.el-toolbar {display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:9px;margin:19px 0 10px;}.el-badge {display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:30px;background:rgba(110,112,245,.1);font-size:11px;color:var(--el-ink);}.el-note {color:var(--el-muted);font-size:12px;margin:10px 0!important;}.el-warning {padding:10px 13px;border-left:3px solid #d5944c;background:rgba(219,153,61,.09);border-radius:5px;font-size:12px;margin:12px 0;}
.el-table-wrap {overflow-x:auto;border:1px solid var(--el-border);border-radius:13px;background:var(--el-panel);}.el table {width:100%;border-collapse:collapse;border:0;text-align:left;white-space:nowrap;}.el th {font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--el-muted);background:rgba(128,140,164,.045);}.el td,.el th {padding:12px 16px;border:0;border-bottom:1px solid var(--el-border);}.el tbody tr:last-child td {border-bottom:0;}.el td {font-variant-numeric:tabular-nums;}.el td strong {font-weight:650;}.el .el-highlight {background:rgba(101,100,245,.065);}
.el-mini-track {width:110px;height:5px;border-radius:5px;background:rgba(128,140,164,.13);margin-top:5px;overflow:hidden;}.el-mini-track i {display:block;height:100%;background:#7470ec;border-radius:5px;}
.el-split {display:grid;grid-template-columns:1.2fr 1fr;gap:14px;margin:14px 0;}.el-bar-row {display:grid;grid-template-columns:95px minmax(0,1fr) 60px;gap:12px;align-items:center;margin:15px 0;font-size:12px;}.el-track {height:12px;background:rgba(128,140,164,.14);border-radius:6px;overflow:hidden;}.el-fill {height:100%;border-radius:6px;background:#7974ed;}.el-fill.base {background:#8894a8;}.el-fill.random {background:#b4a0cc;}.el-axis {display:flex;justify-content:space-between;font-size:10px;color:var(--el-muted);margin-left:107px;margin-right:72px;}.el-arrow {display:flex;align-items:center;justify-content:space-around;gap:14px;margin:18px 0;}.el-arrow strong {font-size:27px;letter-spacing:-.04em;}.el-arrow span {display:block;font-size:11px;color:var(--el-muted);}.el-contrast {color:#7b75ec;font-size:26px;}.el-neighbor {display:grid;grid-template-columns:62px 1fr 65px;gap:12px;align-items:center;margin:8px 0;font-size:11px;}
.el-empty {padding:34px 24px;border:1px dashed var(--el-border);border-radius:15px;text-align:center;margin-top:16px;background:linear-gradient(110deg,rgba(123,117,236,.025),transparent);}.el-empty p {color:var(--el-muted);font-size:13px;margin-top:7px;}.el-empty-steps {display:flex;justify-content:center;gap:18px;margin-bottom:18px;}.el-empty-steps span {width:32px;height:32px;border:1px solid var(--el-border);border-radius:50%;display:grid;place-items:center;color:#8a83eb;}
.el pre {white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;padding:15px;background:rgba(128,140,164,.055);border-radius:8px;max-height:400px;overflow:auto;}.el details {border:1px solid var(--el-border);border-radius:10px;padding:12px 16px;margin:12px 0;}.el summary {cursor:pointer;font-size:12px;font-weight:650;}.el dl {display:grid;grid-template-columns:150px 1fr;gap:8px;font-size:12px;}.el dt {color:var(--el-muted);}.el dd {margin:0;overflow-wrap:anywhere;}.el-ready {display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin:7px 0;}
#evidence-lab {gap:12px;} #evidence-lab .el-primary button {min-height:46px;}
@media(max-width:850px){.el-hero{display:block;padding:22px;}.el-flow{margin-top:19px;}.el-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.el-split{grid-template-columns:1fr;}}
@media(max-width:450px){.el-flow{gap:6px;white-space:normal;}.el-flow span{padding:7px;}.el-bar-row{grid-template-columns:75px minmax(0,1fr) 55px;gap:7px;}.el-axis{margin-left:82px;margin-right:62px;}.el-card{padding:13px;}.el-value{font-size:25px;}.el dl{grid-template-columns:1fr;}}
@media print{.el-hero{background:#14243a!important;color:white!important;print-color-adjust:exact;}.el-card,.el-table-wrap,.el-split{break-inside:avoid;}details{display:block;}}
"""


def esc(value):
    return escape(str(value), quote=True)


def number(value, digits=2):
    if value is None:
        return "—"
    return f"{0 if round(value, digits) == 0 else value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{value:.0%}"


def hero():
    return """<div class="el"><div class="el-hero"><div><div class="el-kicker">7 · Extra / Causal evidence</div>
    <h2>What does this feature actually change?</h2><p>Compare decisions. Check nearby activations. Keep the evidence.</p></div>
    <div class="el-flow" aria-label="Workflow"><span>01<b>Examples</b></span>→<span>02<b>One feature</b></span>→<span>03<b>Measured effect</b></span></div></div></div>"""


def empty():
    return """<div class="el"><div class="el-empty"><div class="el-empty-steps"><span>1</span><span>2</span><span>3</span></div>
    <h3>Move from a shared activation to causal evidence.</h3><p>Prepare examples, select up to three candidates, then compare.</p>
    <p>Fixed A/B decisions · individual interventions · two random controls</p></div></div>"""


def prepared_card(prepared):
    warning = (
        f'<div class="el-warning">{prepared["overlap"]} evaluation inputs overlap calibration content. This is exploratory, not independent validation.</div>'
        if prepared["overlap"]
        else ""
    )
    return (
        f'<div class="el"><div class="el-ready"><span class="el-badge">{esc(prepared["source"])}</span>'
        f'<span class="el-badge">{prepared["pairs"]} available pairs</span>'
        f'<span class="el-badge">Layer {prepared["layer"]} · {prepared["dictionary_size"]:,} features</span>'
        '<span class="el-small">Shortlist = calibration stability, not causal impact.</span></div>'
        + warning
        + "</div>"
    )


def random_groups(result, group):
    return [
        r
        for r in result["groups"]
        if r["feature"] == group["feature"]
        and r["mode"] == group["mode"]
        and r["random_seed"] is not None
    ]


def _metric(label, value, note, style=""):
    return f'<div class="el-card"><div class="el-label">{esc(label)}</div><div class="el-value {style}">{esc(value)}</div><div class="el-small">{esc(note)}</div></div>'


def overview(result):
    base = result["base_summary"]
    entries = feature_groups(result)
    best = entries[0]
    score = best["summary"]
    effective = effective_cases(best)
    gain = score["accuracy"] - base["accuracy"]
    status = (
        "Improvement to validate" if gain > 0 else "No accuracy improvement observed"
    )
    if not effective:
        status = (
            "Identity control · no intervention requested"
            if not result["activity_percent"]
            else "No effective intervention on target inputs"
        )
    if not result["baseline_replay_passed"]:
        status = "Baseline replay differs · inspect before interpreting"
    cards = "".join(
        [
            _metric(
                "BASE CORRECT",
                f"{base['tp'] + base['tn']}/{base['n']}",
                "Fixed A/B decisions",
            ),
            _metric(
                "BEST OBSERVED" if effective else "INPUTS CHANGED",
                f"{score['tp'] + score['tn']}/{score['n']}"
                if effective
                else f"0/{score['n']}",
                f"#{best['feature']} · {best['mode'].lower()} activity"
                if effective
                else "No target residual changed",
            ),
            _metric(
                "CHANGE",
                f"{gain * 100:+.0f} pp" if effective else "—",
                "Exploratory comparison" if effective else "Causal effect not assessed",
                "el-positive" if gain > 0 else "el-negative" if gain < 0 else "",
            ),
            _metric(
                "CONTROL REGRESSIONS",
                f"{score['control_regressions']}/{score['control_eligible']}",
                f"{score['control_total']} small controls · base-correct denominator",
            ),
        ]
    )
    rows = []
    for group in entries:
        s = group["summary"]
        controls = random_groups(result, group)
        random_accuracy = mean(g["summary"]["accuracy"] for g in controls)
        delta = s["accuracy"] - base["accuracy"]
        rows.append(
            f'<tr class="{"el-highlight" if group is best and effective else ""}"><td><strong>#{group["feature"]}</strong><div class="el-small">{group["mode"]} activity</div></td>'
            f'<td><strong>{s["tp"] + s["tn"]}/{s["n"]}</strong><div class="el-mini-track"><i style="width:{s["accuracy"] * 100:.2f}%"></i></div></td>'
            f'<td class="{"el-positive" if delta > 0 else "el-negative" if delta < 0 else ""}">{delta * 100:+.0f} pp</td>'
            f'<td>{random_accuracy:.0%}<div class="el-small">mean of 2 directions</div></td>'
            f"<td>{s['control_regressions']}/{s['control_eligible']}</td>"
            f"<td>{effective_cases(group)}/{s['n']}</td>"
            f'<td>{s["mean_other_features"]:,.0f}<div class="el-small">mean per target input</div></td></tr>'
        )
    warnings = []
    if result["overlap"]:
        warnings.append(
            "Evaluation content overlaps calibration. Results are exploratory."
        )
    if not result["baseline_replay_passed"]:
        warnings.append(
            "The first baseline decision did not reproduce within tolerance."
        )
    if score["mean_label_mass"] < 0.05:
        warnings.append(
            "Low probability mass on A/B. These are constrained-choice scores, not natural-answer confidence."
        )
    if not effective:
        warnings.append(
            "No target residual changed. These runs cannot establish whether the features are causally relevant."
        )
    elif score["inactive_cases"]:
        warnings.append(
            f"The selected feature was inactive on {score['inactive_cases']}/{score['n']} target inputs; rescaling an inactive feature makes no intervention."
        )
    warning_html = "".join(f'<div class="el-warning">{esc(w)}</div>' for w in warnings)
    return (
        f'<div class="el"><div class="el-toolbar"><h3>{status}</h3><span class="el-badge">{base["n"] // 2} pairs · layer {result["layer"]} · ±{result["activity_percent"]:g}% activity</span></div>'
        f'<div class="el-grid">{cards}</div>{warning_html}'
        '<div class="el-table-wrap"><table aria-label="Feature comparison"><thead><tr><th>Feature / intervention</th><th>Correct</th><th>Vs. base</th><th>Random controls</th><th>Control regressions</th><th>Inputs changed</th><th>Other features moved</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="el-note">Runs with actual target changes first, then correctness and control preservation. One decoder direction can move other SAE activations. Small samples and two random directions do not establish isolation or significance.</p></div>'
    )


def _bar(label, value, kind=""):
    return (
        f'<div class="el-bar-row"><span>{esc(label)}</span><div class="el-track" role="meter" aria-label="{esc(label)} B preference" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{value * 100:.2f}">'
        f'<div class="el-fill {kind}" style="width:{value * 100:.3f}%"></div></div><strong>{value:.2%}</strong></div>'
    )


def detail(result, group_key, case_id):
    group = next(
        g
        for g in result["groups"]
        if g["key"] == group_key and g["random_seed"] is None
    )
    row = next(r for r in group["rows"] if r["case_id"] == case_id)
    base = next(r for r in result["baseline"] if r["case_id"] == case_id)
    case = next(c for c in result["cases"] if c["case_id"] == case_id)
    s, b = group["summary"], result["base_summary"]
    cards = "".join(
        [
            _metric(
                "B CORRECTLY DETECTED",
                f"{s['tp']}/{s['n_b']}",
                f"Base: {b['tp']}/{b['n_b']} · ties count as errors",
            ),
            _metric(
                "A INCORRECTLY CALLED B",
                f"{s['fp']}/{s['n_a']}",
                f"Base: {b['fp']}/{b['n_a']} · ties: {s['ties_a']}",
            ),
            _metric(
                "DISCRIMINATION · d′",
                number(s["dprime"]),
                f"Base: {number(b['dprime'])} · corrected estimate",
            ),
            _metric(
                "RESPONSE CRITERION · c",
                number(s["criterion"]),
                f"Base: {number(b['criterion'])} · positive favors A",
            ),
        ]
    )
    bars = _bar("Base", base["p_b"], "base") + _bar(group["mode"], row["p_b"])
    for i, control in enumerate(random_groups(result, group)):
        random_row = next(r for r in control["rows"] if r["case_id"] == case_id)
        bars += _bar(f"Random {i + 1}", random_row["p_b"], "random")
    neighbors = row["neighbors"]
    maximum = max((abs(n["delta"]) for n in neighbors), default=1)
    neighbor_html = (
        "".join(
            f'<div class="el-neighbor"><span>#{n["feature"]}</span><div class="el-mini-track" style="width:100%"><i style="width:{abs(n["delta"]) / maximum * 100:.2f}%"></i></div><span>{n["delta"]:+.3g}</span></div>'
            for n in neighbors
        )
        or '<p class="el-note">No other activations crossed the measurement threshold.</p>'
    )
    return (
        f'<div class="el"><div class="el-grid">{cards}</div>'
        '<p class="el-note">d′ and c describe this fixed decision task, under the equal-variance signal-detection model. Half-count correction; ties leave them undefined. They are not proof that a concept was erased.</p>'
        f'<div class="el-toolbar"><h3>{esc(case["label"])}</h3><span class="el-badge">Expected {case["expected"]} · {esc(case["group"])}</span></div>'
        '<div class="el-split"><div class="el-card"><h3>Decision movement</h3><div class="el-small">Preference for B, conditional on A or B</div>'
        f'{bars}<div class="el-axis"><span>A ← 0%</span><span>50%</span><span>100% → B</span></div>'
        f'<p class="el-note">A/B mass: {base["label_mass"]:.1%} → {row["label_mass"]:.1%}. Prediction: {base["prediction"]} → {row["prediction"]}.</p></div>'
        f'<div class="el-card"><h3>Feature #{group["feature"]} · measured activity</h3><div class="el-arrow"><div><strong>{row["activation_before"]:.3g}</strong><span>Before</span></div><div class="el-contrast">→</div><div><strong>{row["activation_after"]:.3g}</strong><span>After re-encoding</span></div></div>'
        f'<div class="el-small">Actual residual change: {row["residual_change"]:.2%} · intended budget: 2%{" · capped" if row["budget_capped"] else ""}</div>'
        f'<p class="el-note">{row["other_features_moved"]:,} other features moved. Largest changes:</p>{neighbor_html}</div></div>'
        f'<details><summary>Exact input and expected decision</summary><p class="el-note">Expected {case["expected"]}; this label is used only by the evaluator. Images, when present, are identified by hash in JSON.</p><pre>{esc(case["prompt"])}</pre></details></div>'
    )


def metadata(result):
    fields = {
        "Model": result["model"],
        "Dictionary": f"{result['sae_release']}/{result['sae_id']}",
        "Profile": result["profile_id"],
        "Evaluation": result["source"],
        "A means": result["definitions"]["A"],
        "B means": result["definitions"]["B"],
        "Intervention": "Current activity rescaled in one decoder direction, final input position only",
        "Baseline replay": "Passed on first case"
        if result["baseline_replay_passed"]
        else "Different on first case",
        "Forwards": result["forwards"],
        "Decision protocol": "A/B next-token comparison. No free-response generation; shared temperature and token limit are not used.",
        "Scope": "Local re-encoding plus four synthetic controls; no guarantee about other prompts or layers.",
    }
    return (
        '<div class="el"><details><summary>Study identity & interpretation</summary><dl>'
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in fields.items())
        + "</dl></details></div>"
    )


def standalone_report(result):
    sections = []
    target_cases = [c for c in result["cases"] if c["group"] == "target"]
    for index, group in enumerate(feature_groups(result)):
        first = detail(result, group["key"], target_cases[0]["case_id"])
        sections.append(
            f"<details {'open' if index == 0 else ''}><summary>#{group['feature']} · {group['mode']} activity · first evaluation case</summary>{first}</details>"
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Feature evidence · {esc(result['id'][:8])}</title><style>body{{margin:0;background:#f5f7fb;padding:28px;color:#182236;}}main{{max-width:1100px;margin:auto;}}{EVIDENCE_CSS}</style>"
        f'<main class="el">{hero()}{overview(result)}{"".join(sections)}{metadata(result)}'
        '<p class="el-note">Exploratory measurements. Full per-case results, controls, prompts and dictionary identity are in the accompanying measurements.json.</p></main></html>'
    )
