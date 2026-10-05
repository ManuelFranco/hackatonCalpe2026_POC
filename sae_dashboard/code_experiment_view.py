"""Compact generated-code comparisons with complete escaped report exports."""

import json

from . import generation_view
from .code_experiment_lab import rank_groups
from .evidence_view import EVIDENCE_CSS, esc
from .profile_discovery_view import DISCOVERY_CSS


RESERVED = "Reserved continuations"
FULL = "Full generation"


def empty():
    return '<div class="pd"><div class="pd-empty">Build a profile, then test its candidates on generated SQL code.</div></div>'


def choices(result):
    return [
        (g["name"], g["id"])
        for g in rank_groups(result["groups"])
        if g["kind"] != "Random"
    ] + [(RESERVED, RESERVED), (FULL, FULL)]


def case_choices(result, selection):
    split = "Reserved" if selection in {RESERVED, FULL} else "Search"
    return [
        (c["id"].removeprefix("lookup-"), c["id"])
        for c in result["cases"]
        if c["split"] == split
    ]


def overview(result):
    selected = next(g for g in result["groups"] if g["id"] == result["selected_id"])
    base, held = result["base_summary"], result["reserved_summary"]
    legacy = result.get("evaluator_version", 1) < 2
    incomplete = any(
        s["assessed"] < s["cases"] or s.get("comparable", 0) < s["cases"]
        for s in [held, *result["reserved_random"]]
    )
    if legacy:
        status = "Outdated evaluation · re-evaluate the generated code"
    elif any(s.get("budget_violations", 0) for s in [held, *result["reserved_random"]]):
        status = "Exploratory · measured intervention exceeded the requested budget"
    elif result["confirmed"]:
        status = "Effect observed on reserved continuations"
    elif incomplete:
        status = "Inconclusive · incomplete behavioral evaluation"
    else:
        status = "No confirmed gain over the reserved controls"
    selection = (
        "Candidate selected on search tasks"
        if result.get("search_gain_observed")
        else "Exploratory candidate · no complete positive gain on search"
    )
    if result.get("selection_preserved"):
        selection += " · frozen selection preserved during re-evaluation"

    def hits(measured):
        if not measured["assessed"]:
            return "— · no assessed cases"
        return f"{measured['target_hits']}/{measured['assessed']} assessed"

    rows = []
    for group in rank_groups(result["groups"]):
        s = group["summary"]
        schedule = (
            "All tokens"
            if group["steps"] is None
            else "First token"
            if group["steps"] == 1
            else f"First {group['steps']} tokens"
        )
        name = " + ".join(f"#{f}" for f in group["features"]) or group["name"]
        changes = (
            f"↑ {s['new_hits']} · ↓ {s['lost_hits']}" if s.get("comparable", 0) else "—"
        )
        unknown = s.get("functional_unknown", 0)
        working = f"{s['functional']}/{s['cases']}"
        if unknown:
            working += f" · {unknown} unknown"
        goal_hits = f"{s['target_hits']}/{s['assessed']}" if s["assessed"] else "—"
        assessment = f"{s['assessed']}/{s['cases']} outputs evaluated"
        comparable = f"{s.get('comparable', 0)}/{s['cases']} comparable with Base"
        rows.append(
            f"<tr><td>{esc(name)}{' · selected' if group['id'] == selected['id'] else ''}"
            f"<br><small>{schedule}{' · budget exceeded' if s.get('budget_violations') else ''}</small></td>"
            f"<td>{working}</td><td title='{assessment}'>{goal_hits}</td>"
            f"<td title='{comparable}'>{changes}</td></tr>"
        )
    replay = (
        "Token replay matches"
        if result["baseline_replay_passed"]
        else "MISMATCH · exploratory results"
    )
    return (
        '<div class="pd"><div class="pd-hero"><div><div class="pd-kicker">Generated code · measured interventions</div>'
        f"<h3>{status}</h3><p>{selection}: {esc(selected['name'])} · {'all tokens' if selected['steps'] is None else str(selected['steps']) + ' continuation tokens'}.</p>"
        f"<p>Goal: {esc(result['goal'])} · layer {result['layer']} · {result['budget_percent']:g}% residual budget per changed token.</p></div>"
        f'<div class="pd-count">{held["new_hits"] if held.get("comparable", 0) else "—"}<small>new reserved goal hits · {held.get("comparable", 0)}/{held["cases"]} comparable</small></div></div>'
        '<div class="el"><div class="el-ready">'
        f'<span class="el-badge">Reserved: {held["functional"]}/{held["cases"]} functional · {held.get("functional_unknown", 0)} unknown · {held["assessed"]}/{held["cases"]} assessed</span>'
        f'<span class="el-badge">Reserved random hits: {" · ".join(hits(s) for s in result["reserved_random"])}</span>'
        f'<span class="el-badge">Full generation goal hits: {hits(result["full_summary"])}</span>'
        f'<span class="el-badge">Reserved actual change: {100 * held["maximum_relative_change"]:.3f}% maximum</span>'
        f'<span class="el-badge">Baseline replay: {replay}</span></div></div>'
        f'<p class="pd-note">Search baseline goal hits: {hits(base)}. Unsupported code is unknown, not a functional failure. New/lost hits require both Base and the intervention to be assessed. Reserved tasks do not select features or timing.</p>'
        '<details><summary>Search results</summary><div class="pd-scroll"><table class="pd-table"><thead><tr>'
        "<th>Candidate · timing</th><th>Working code</th><th>SQL goal hits</th><th>vs Base</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="pd-note">Working code = ordinary lookups passed. Goal hits = successes / evaluated outputs. ↑ new hits · ↓ lost hits vs Base. — = not evaluated or not comparable.</p></details>'
        '<p class="pd-note">The intervention window limits changed generation steps, not response length. Baseline replay checks reproducibility of token IDs, not code correctness. Pair comparisons share one total norm budget; a pair gain does not establish feature synergy. Evidence applies to these synthetic tasks and this SQL probe.</p></div>'
    )


def comparison(result, selection, case_id):
    if not result:
        return ""
    names, rows = [], []
    if selection in {RESERVED, FULL}:
        measured = next(
            r
            for r in result[
                "validation" if selection == RESERVED else "full_generation"
            ]
            if r["case_id"] == case_id
        )
        names = ["Base", "Frozen candidate", "Random control 1", "Random control 2"]
        rows = [measured["base"], measured["selected"], *measured["random"]]
        if selection == RESERVED:
            names.append("Reverse direction")
            rows.append(measured["reverse"])
        note = (
            "Frozen direction and schedule; reserved tasks were excluded from search."
            if selection == RESERVED
            else "No forced function prefix. Frozen direction applied continuously; this is a separate check with a different schedule."
        )
    else:
        groups = {g["id"]: g for g in result["groups"]}
        group = groups[selection]
        rows = [next(r for r in result["base"] if r["case_id"] == case_id)]
        names = ["Base"]
        identifiers = group.get("single_ids", []) + [selection]
        if group["kind"] == "Single":
            identifiers += ["random-0", "random-1"]
        for identifier in identifiers:
            g = groups[identifier]
            rows.append(next(r for r in g["rows"] if r["case_id"] == case_id))
            names.append(g["name"])
        note = "Shared function prefix, seed and per-token residual budget. Continuations may diverge after the first changed token."
        if group["kind"] == "Pair":
            gain = group["gain_over_best_single"]
            note += (
                f" Pair goal-hit gain over the best single on search: {gain:+d}."
                if gain is not None
                else " Pair gain is inconclusive: assessments are incomplete."
            )
    observed = {
        "id": result["id"] + selection + case_id,
        "layer": result["layer"],
        "features": result["features"],
        "runs": [{**r, "name": n} for n, r in zip(names, rows)],
    }
    timelines = "".join(
        f"<details><summary>{esc(name)} · token activations</summary>{generation_view.timeline(observed, name)}</details>"
        for name in names
    )
    case = next(c for c in result["cases"] if c["id"] == case_id)
    return (
        f'<div class="pd"><p>{esc(note)}</p></div>'
        + generation_view.overview(observed, behavioral=True)
        + f'<div class="el">{timelines}<details><summary>Exact task and evaluation</summary><pre>{esc(case["prompt"])}</pre>'
        + f"<pre>{esc(json.dumps([{'condition': n, 'checks': r['checks'], 'intervention': {k: v for k, v in r['intervention'].items() if k != 'steps'}} for n, r in zip(names, rows)], indent=2))}</pre></details></div>"
    )


def standalone_report(result):
    comparisons = "".join(
        f"<details><summary>{esc(label)} · {esc(case_id)}</summary>{comparison(result, selection, case_id)}</details>"
        for label, selection in choices(result)
        for _, case_id in case_choices(result, selection)
    )
    identity = {
        k: result[k]
        for k in (
            "model",
            "sae_release",
            "sae_id",
            "profile_id",
            "settings",
            "candidates",
            "selection_rule",
            "limitations",
            "generations",
        )
    }
    identity.update(
        {
            k: result.get(k)
            for k in (
                "protocol",
                "evaluator_version",
                "reevaluated_from",
                "assessment_updated_unix",
                "selection_preserved",
            )
        }
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Generated code experiment · {esc(result['id'][:8])}</title>"
        "<style>body{background:#f5f7fb;color:#182236;padding:24px}main{max-width:1250px;margin:auto}details{margin:12px 0}summary{cursor:pointer}pre{overflow:auto}"
        + EVIDENCE_CSS
        + generation_view.GENERATION_CSS
        + DISCOVERY_CSS
        + f"</style><main>{overview(result)}{comparisons}<details><summary>Identity and protocol</summary><pre>{esc(json.dumps(identity, indent=2))}</pre></details></main></html>"
    )
