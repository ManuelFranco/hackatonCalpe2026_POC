"""Escaped, local transfer-matrix reports with explicit paired denominators."""

from html import escape
import json

TRANSFER_CSS = """
.transfer-view {color:var(--body-text-color,#182236);line-height:1.5;overflow-wrap:anywhere;}
.transfer-view table {width:100%;border-collapse:collapse;margin:14px 0;}
.transfer-view th,.transfer-view td {padding:12px;text-align:left;border-bottom:1px solid rgba(128,128,128,.25);vertical-align:top;}
.transfer-view .matrix-wrap {overflow-x:auto;}
.transfer-view .matrix td {min-width:160px;}
.transfer-view .matrix a {display:block;color:inherit;text-decoration:none;}
.transfer-view .matrix strong {display:block;font-size:22px;}
.transfer-view .gain {background:rgba(16,185,129,.09);}
.transfer-view .loss {background:rgba(245,158,11,.1);}
.transfer-view .overlap {display:block;font-weight:600;}
.transfer-view small {display:block;opacity:.8;}
.transfer-view pre {white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:14px;}
.transfer-view .reading {display:flex;gap:24px;flex-wrap:wrap;margin:12px 0;}
.transfer-view details {margin:12px 0;}
.transfer-view summary {cursor:pointer;font-weight:600;}
"""


def esc(value):
    return escape(str(value), quote=True)


def source_choices(session):
    return [(s.metadata["name"], key) for key, s in session.transfer_sources.items()]


def sources(session):
    if session is None or not session.transfer_sources:
        return '<div class="transfer-view">No frozen vectors yet. Build vectors in section 3, set strengths, then freeze a source here.</div>'
    rows = []
    for source in session.transfer_sources.values():
        m = source.metadata
        strengths = ", ".join(f"L{k}: {v:+g}" for k, v in m["strengths"].items() if v)
        rows.append(
            f"<tr><td>{esc(m['name'])}</td><td>{esc(m['capture']['token_scope'])} / {esc(m['capture']['aggregation'])}</td><td>{esc(strengths)}</td></tr>"
        )
    return (
        '<div class="transfer-view"><table><thead><tr><th>Frozen source</th><th>Capture</th><th>Fixed strengths</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def prepared(plan, session):
    rows = []
    for target in plan["targets"]:
        overlap = max(target["overlap"].values(), default=0)
        rows.append(
            f"<tr><td>{esc(target['name'])}</td><td>{esc(target['split'])}</td><td>{len(target['cases']) // 2} pairs / {len(target['cases'])} inputs</td><td>{overlap} exact overlaps (max per source)</td></tr>"
        )
    forwards = sum(len(t["cases"]) for t in plan["targets"]) * (
        1 + len(plan["source_ids"])
    )
    return (
        '<div class="transfer-view"><table><thead><tr><th>Evaluation set</th><th>Split</th><th>Frozen sample</th><th>Calibration overlap</th></tr></thead><tbody>'
        + "".join(rows)
        + f"</tbody></table><small>Seed {plan['seed']} · {len(plan['source_ids'])} sources · {forwards} forwards. Exact overlap is exploratory; absence of overlap does not prove independent templates.</small></div>"
    )


def preview(plan, case_id):
    if not plan or not case_id:
        return ""
    for target in plan["targets"]:
        case = next((c for c in target["cases"] if c["case_id"] == case_id), None)
        if case:
            return f'<div class="transfer-view"><p><strong>{esc(target["name"])} · {esc(case["pair_id"])} · expected {case["expected"]}</strong></p><pre>{esc(case["prompt"])}</pre><small>The expected label is evaluator metadata; it is not inserted into the model prompt.</small></div>'
    raise ValueError("Select a case from the prepared sample.")


def empty():
    return '<div class="transfer-view">Prepare the sample, inspect its cases, then run the matrix. Every source is evaluated on the same inputs.</div>'


def matrix(result, links=False):
    if not result:
        return empty()
    cells = {(c["source_id"], c["target_id"]): c for c in result["cells"]}
    header = "".join(f"<th>{esc(t['name'])}</th>" for t in result["targets"])
    rows = []
    for source in result["sources"]:
        entries = []
        for target in result["targets"]:
            cell = cells[source["id"], target["id"]]
            s = cell["summary"]
            delta = s["delta_accuracy"] * 100
            style = "gain" if delta > 0 else "loss" if delta < 0 else ""
            overlap = (
                f'<small class="overlap">Exploratory · {cell["overlap"]} calibration overlaps</small>'
                if cell["overlap"]
                else ""
            )
            start = f'<a href="#cell-{esc(cell["id"])}">' if links else "<div>"
            end = "</a>" if links else "</div>"
            entries.append(
                f'<td class="{style}">{start}<strong>{delta:+.1f} pp</strong><small>Base {s["base"]["tp"] + s["base"]["tn"]}/{s["n"]} → steered {s["tp"] + s["tn"]}/{s["n"]}</small><small>{s["corrections"]} corrections · {s["regressions"]} regressions</small><small>False alarms {s["fp"]}/{s["n_a"]} · misses {s["fn"]}/{s["n_b"]}</small>{overlap}{end}</td>'
            )
        rows.append(f"<tr><th>{esc(source['name'])}</th>{''.join(entries)}</tr>")
    return f'<div class="transfer-view"><div class="matrix-wrap"><table class="matrix"><thead><tr><th>Source → evaluation set</th>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div><small>Change in accuracy over all sampled inputs. Ties count as errors and are reported in the details. Cells measure decision transfer, not exploit success or free-response quality.</small></div>'


def cell_choices(result):
    if not result:
        return []
    names = {s["id"]: s["name"] for s in result["sources"]} | {
        t["id"]: t["name"] for t in result["targets"]
    }
    return [
        (f"{names[c['source_id']]} → {names[c['target_id']]}", c["id"])
        for c in result["cells"]
    ]


def case_choices(result, cell_id):
    if not result or not cell_id:
        return []
    cell = next(c for c in result["cells"] if c["id"] == cell_id)
    target = next(t for t in result["targets"] if t["id"] == cell["target_id"])
    return [
        (f"{c['pair_id']} · {c['expected']}", c["case_id"]) for c in target["cases"]
    ]


def detail(result, cell_id, case_id):
    if not result or not cell_id or not case_id:
        return ""
    cell = next(c for c in result["cells"] if c["id"] == cell_id)
    target = next(t for t in result["targets"] if t["id"] == cell["target_id"])
    case = next(c for c in target["cases"] if c["case_id"] == case_id)
    base = next(r for r in target["baseline"] if r["case_id"] == case_id)
    treated = next(r for r in cell["rows"] if r["case_id"] == case_id)
    source = next(s for s in result["sources"] if s["id"] == cell["source_id"])
    s = cell["summary"]
    readings = []
    for name, row in (("Base", base), ("Steered", treated)):
        correct = (
            "Correct"
            if row["prediction"] == case["expected"]
            else "Tie · counted as error"
            if row["prediction"] == "Tie"
            else "Incorrect"
        )
        readings.append(
            f"<div><strong>{name}: {row['prediction']} · {correct}</strong><small>B − A logit margin: {row['margin']:+.4f} · A/B probability mass: {row['label_mass']:.1%}</small></div>"
        )
    return (
        f'<div class="transfer-view" id="cell-{esc(cell_id)}"><p><strong>{esc(source["name"])} → {esc(target["name"])}</strong></p>'
        f"<p>Changed inputs: {s['changed_inputs']}/{s['n']} · low A/B mass (&lt;50%): {s['low_label_mass']}/{s['n']} · ties: {s['ties_a'] + s['ties_b']}</p>"
        f'<p>Expected {case["expected"]}: {esc(target["labels"][case["expected"]])}</p><div class="reading">{"".join(readings)}</div>'
        f"<pre>{esc(case['text'])}</pre><small>The margin is a forced A/B preference, not calibrated vulnerability confidence. Frozen vectors use their original strengths and token policy.</small></div>"
    )


def standalone_report(result):
    sections = []
    for cell in result["cells"]:
        choices = case_choices(result, cell["id"])
        sections.append(detail(result, cell["id"], choices[0][1]))
        for label, identifier in choices[1:]:
            # Avoid duplicate anchor IDs for the same cell.
            content = detail(result, cell["id"], identifier).replace(
                f' id="cell-{esc(cell["id"])}"', ""
            )
            sections.append(
                f"<details><summary>{esc(label)}</summary>{content}</details>"
            )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Transfer matrix</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:14px/1.5 system-ui,sans-serif;} "
        + TRANSFER_CSS
        + "</style><h1>Frozen-vector transfer matrix</h1>"
        + matrix(result, links=True)
        + "".join(sections)
        + f"<details><summary>Protocol and provenance</summary><pre>{esc(json.dumps({k: result[k] for k in ('model_id', 'seed', 'readout', 'sources')}, indent=2))}</pre></details></html>"
    )
