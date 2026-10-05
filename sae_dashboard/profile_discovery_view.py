"""Automatic candidate overview and several input features visible together."""

from html import escape

from research.profile_discovery import source_segments


COLORS = (
    "99,102,241",
    "13,148,136",
    "217,119,6",
    "225,29,72",
    "14,165,233",
    "147,51,234",
    "101,163,13",
    "234,88,12",
    "79,70,229",
    "5,150,105",
    "192,38,211",
    "71,85,105",
)

DISCOVERY_CSS = """
.pd {--pd-border:rgba(127,127,127,.22);--pd-muted:var(--body-text-color-subdued);min-width:0;}
.pd-hero {display:flex;justify-content:space-between;gap:16px;align-items:center;padding:22px 24px;border:1px solid var(--pd-border);border-radius:14px;background:linear-gradient(115deg,rgba(99,102,241,.09),rgba(13,148,136,.04));margin:12px 0;}
.pd h3 {font-size:1.2rem;margin:0 0 6px;font-weight:750;}.pd p {font-size:.87rem;margin:6px 0;line-height:1.6;}
.pd-kicker {font-size:.68rem;font-weight:750;letter-spacing:.1em;text-transform:uppercase;color:#6366f1;margin-bottom:7px;}
.pd-count {font-size:2.2rem;font-weight:750;line-height:1;}.pd-hero small {display:block;font-size:.73rem;color:var(--pd-muted);margin-top:7px;}
.pd-scroll {overflow-x:auto;border:1px solid var(--pd-border);border-radius:12px;}
.pd-table {width:100%;min-width:850px;border-collapse:collapse;font-size:.8rem;}.pd-table th {font-size:.68rem;letter-spacing:.04em;text-transform:uppercase;text-align:left;padding:11px 13px;background:rgba(127,127,127,.07);}.pd-table td {padding:12px 13px;border-top:1px solid var(--pd-border);vertical-align:middle;font-variant-numeric:tabular-nums;}.pd-table tr:hover td {background:rgba(99,102,241,.035);}
.pd-feature {display:inline-flex;align-items:center;gap:7px;font-weight:750;white-space:nowrap;}.pd-dot {display:inline-block;width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.pd-direction {font-size:.67rem;color:var(--pd-muted);display:block;margin-top:3px;}.pd-score {height:5px;background:rgba(99,102,241,.12);border-radius:4px;margin-top:5px;width:85px;}.pd-score b {height:100%;display:block;background:#6366f1;border-radius:4px;}
.pd-deltas {display:flex;flex-wrap:wrap;gap:3px;max-width:185px;}.pd-delta {display:block;width:9px;height:16px;border-radius:2px;}
.pd-note {color:var(--pd-muted);font-size:.78rem!important;margin:12px 0!important;}.pd-empty {padding:18px;border:1px dashed var(--pd-border);border-radius:12px;color:var(--pd-muted);}
.pd-choice {position:absolute;opacity:0;width:1px;height:1px;}.pd-choice + label {display:inline-flex;align-items:center;gap:6px;padding:7px 10px;margin:8px 6px 14px 0;border:1px solid var(--pd-border);border-radius:8px;cursor:pointer;font-size:.78rem;}.pd-choice:checked + label {border-color:#6366f1;background:rgba(99,102,241,.1);}.pd-choice:focus-visible + label {outline:2px solid #6366f1;outline-offset:3px;}
.pd-inputs {display:grid;grid-template-columns:1fr 1fr;gap:16px;}.pd-input {border:1px solid var(--pd-border);border-radius:12px;min-width:0;overflow:hidden;}.pd-input-head {padding:13px 16px;border-bottom:1px solid var(--pd-border);display:flex;justify-content:space-between;gap:10px;align-items:center;}.pd-input-head b {font-size:.9rem;}.pd-input-head small {font-size:.72rem;color:var(--pd-muted);}.pd-source {font-family:ui-monospace,SFMono-Regular,monospace;font-size:.8rem;line-height:1.95;white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;max-height:480px;overflow-y:auto;margin:0;}.pd-piece {border-radius:2px;background:var(--pd-mix,transparent);}.pd-outside {color:var(--pd-muted);}.pd-context {border-top:1px solid var(--pd-border);padding:11px 16px;font-size:.75rem;}.pd-context summary {cursor:pointer;font-weight:650;}.pd-context pre {white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8;margin-top:10px;font-size:.75rem;}.pd-context-note {display:block;margin-top:8px;color:var(--pd-muted);}
@media(max-width:750px) {.pd-inputs {grid-template-columns:1fr;}.pd-hero {padding:17px;}.pd-table td,.pd-table th {padding:10px;}}
"""


def empty():
    return '<div class="pd"><div class="pd-empty">Build a common profile to discover candidates automatically.</div></div>'


def render_candidates(rows, layer, pair_labels=()):
    if not rows:
        return '<div class="pd"><div class="pd-empty">No stable nonzero contrasts found in this layer.</div></div>'
    largest = max(r["score"] for r in rows)
    table = []
    for i, row in enumerate(rows):
        deltas = row["pair_deltas"]
        peak = max((abs(v) for v in deltas), default=0) or 1
        squares = []
        for j, value in enumerate(deltas[:36]):
            color = (
                "13,148,136"
                if value > 0
                else "225,29,72"
                if value < 0
                else "127,127,127"
            )
            label = pair_labels[j] if j < len(pair_labels) else f"Pair {j + 1}"
            title = escape(f"{label} · B − A {value:+.6g}", quote=True)
            squares.append(
                f'<span class="pd-delta" style="background:rgba({color},{0.15 + 0.85 * abs(value) / peak:.3f})" title="{title}"></span>'
            )
        if len(deltas) > 36:
            squares.append(
                f'<span title="First 36 pairs shown">+{len(deltas) - 36}</span>'
            )
        table.append(
            "<tr>"
            f'<td><span class="pd-feature"><i class="pd-dot" style="background:rgb({COLORS[i % len(COLORS)]})"></i>#{row["feature"]}</span>'
            f'<span class="pd-direction">{"B higher" if row["contrast"] > 0 else "A higher"}</span></td>'
            f"<td>{row['mean_a']:.4g}</td><td>{row['mean_b']:.4g}</td>"
            f"<td><strong>{row['contrast']:+.4g}</strong></td>"
            f"<td>{row['same_sign']}/{row['pairs']}</td>"
            f"<td>{row['coverage']:.0%}</td>"
            f'<td>{row["score"]:.4g}<div class="pd-score"><b style="width:{100 * row["score"] / largest:.2f}%"></b></div></td>'
            f'<td><div class="pd-deltas">{"".join(squares)}</div></td></tr>'
        )
    return (
        '<div class="pd"><div class="pd-hero"><div><div class="pd-kicker">Common profile · automatic discovery</div>'
        "<h3>Find the features in your inputs.</h3>"
        f"<p>Layer {int(layer)} · {rows[0]['pairs']} matched pairs · no feature IDs needed.</p></div>"
        f'<div><span class="pd-count">{len(rows)}</span><small>candidates</small></div></div>'
        '<div class="pd-scroll"><table class="pd-table"><thead><tr><th>Candidate</th><th>Mean A</th><th>Mean B</th><th>B − A</th><th>Same direction</th><th>Coverage</th><th>Rank score</th><th>Across pairs</th></tr></thead><tbody>'
        + "".join(table)
        + '</tbody></table></div><p class="pd-note">Score = |mean B − A| × (|mean| / RMS difference) × contrast coverage. '
        "Same direction counts all pairs, including zero contrasts in the denominator. "
        "Teal: B higher; rose: A higher. Scores describe these inputs; they are not causal evidence or calibrated confidence.</p></div>"
    )


def _piece(text, values, positions, selected, features, scales):
    if not values or not selected:
        title = "Outside the captured scope" if positions else "No aligned model token"
        return f'<span class="pd-outside" title="{title}">{escape(text)}</span>'
    normalized = [max(0, v) / scale for v, scale in zip(values, scales)]
    dominant = max(range(len(values)), key=normalized.__getitem__)
    intensity = min(1, normalized[dominant])
    styles = [f"--pd-mix:rgba({COLORS[dominant % len(COLORS)]},{0.65 * intensity:.3f})"]
    for j, value in enumerate(normalized):
        styles.append(
            f"--pd-f{j}:rgba({COLORS[j % len(COLORS)]},{0.65 * min(1, value):.3f})"
        )
    title = "Input positions " + ", ".join(str(i + 1) for i in positions)
    title += "\n" + "\n".join(f"#{j}: {v:.6g}" for j, v in zip(features, values))
    return f'<span class="pd-piece" style="{";".join(styles)}" title="{escape(title, quote=True)}">{escape(text)}</span>'


def render_observation(result):
    features, conditions = result["features"], result["conditions"]
    scope = "pd-" + result["profile_id"] + f"-{result['layer']}-{result['pair_index']}"
    scales = [
        max(
            [
                float(row[j])
                for c in conditions
                for row, chosen in zip(c["values"], c["selected"])
                if chosen
            ]
            + [1e-8]
        )
        for j in range(len(features))
    ]
    choices = f'<input class="pd-choice" type="radio" name="{scope}" id="{scope}-all" checked><label for="{scope}-all">All candidates</label>'
    styles = []
    for j, feature in enumerate(features):
        choices += f'<input class="pd-choice" type="radio" name="{scope}" id="{scope}-{j}"><label for="{scope}-{j}"><i class="pd-dot" style="background:rgb({COLORS[j % len(COLORS)]})"></i>#{feature}</label>'
        styles.append(
            f"#{scope}-{j}:checked ~ .pd-inputs .pd-piece {{background:var(--pd-f{j},transparent);}}"
        )
    panels = []
    for c in conditions:
        if c["spans"]:
            segments = source_segments(
                c["text"], c["spans"], c["values"], c["selected"]
            )
            source = "".join(
                _piece(
                    s["text"],
                    s["values"],
                    s["positions"],
                    s["selected"],
                    features,
                    scales,
                )
                for s in segments
            )
            extra = [
                i
                for i, span in enumerate(c["spans"])
                if span[0] == span[1] and c["selected"][i] and not c["image_mask"][i]
            ]
            context = "".join(
                _piece(
                    c["tokens"][i] + " ", c["values"][i], [i], True, features, scales
                )
                for i in extra
            )
            context = (
                f'<details class="pd-context"><summary>{len(extra)} captured chat positions outside the source</summary><pre>{context}</pre>'
                '<span class="pd-context-note">Chat delimiters and the assistant prefix are also included in All tokens.</span></details>'
                if extra
                else ""
            )
        else:
            source = "".join(
                _piece(
                    token + " ", c["values"][i], [i], c["selected"][i], features, scales
                )
                for i, token in enumerate(c["tokens"])
                if not c["image_mask"][i]
            )
            context = f'<div class="pd-context">{escape(c["alignment"])}</div>'
        visual = sum(bool(v and s) for v, s in zip(c["image_mask"], c["selected"]))
        if visual:
            context += f'<div class="pd-context">{visual} captured image-token positions are not mapped to pixels.</div>'
        panels.append(
            '<div class="pd-input"><div class="pd-input-head">'
            f"<b>{escape(c['name'])} · Input</b><small>{sum(c['selected'])} / {len(c['tokens'])} tokens captured</small>"
            f'</div><pre class="pd-source">{source}</pre>{context}</div>'
        )
    return (
        f'<div class="pd"><style>{"".join(styles)}</style><h3>{escape(result["pair_label"])}</h3>'
        "<p>All candidates shows the strongest normalized candidate at each captured position. Hover for exact activations.</p>"
        + choices
        + f'<div class="pd-inputs">{"".join(panels)}</div>'
        + '<p class="pd-note">Each feature uses one activation scale across A and B. Colors from different features do not compare raw magnitudes. '
        "Only positions included in Profile tokens are shaded. These are input observations; generated responses and causal effects require separate measurements.</p></div>"
    )
