"""Automatic, session-isolated experiments on generated SQLite lookup functions."""

from dataclasses import asdict
from copy import deepcopy
from itertools import combinations
import time
import uuid

import torch

from research.causal_interventions import prepare_case_inputs
from research.code_review import review_python
from research.generation_interventions import (
    ResidualIntervention,
    feature_direction,
    random_direction,
)
from research.generation_trace import GenerationTrace
from research.profile_discovery import rank_candidates
from research.sql_generation_checks import EVALUATOR_VERSION, evaluate, tasks

from . import model_runtime as runtime
from .artifact_store import record_event
from .workflow import require_profile


GOALS = ("B · unintended SQL row", "A · block the SQL probe")
POLARITIES = ("B higher", "A higher", "Both signs")


def cancel(session):
    with session.lock:
        session.code_experiment_job = None


def summary(rows, baseline, goal):
    if [r["case_id"] for r in rows] != [r["case_id"] for r in baseline]:
        raise ValueError("Behavior summaries require the same ordered tasks.")
    target = "B · unintended row" if goal == GOALS[0] else "A · probe blocked"
    hits = [r["checks"]["assessed"] and r["checks"]["behavior"] == target for r in rows]
    base_hits = [
        r["checks"]["assessed"] and r["checks"]["behavior"] == target for r in baseline
    ]
    comparable = [
        r["checks"]["assessed"] and b["checks"]["assessed"]
        for r, b in zip(rows, baseline)
    ]
    return {
        "cases": len(rows),
        "functional": sum(r["checks"]["functional"] is True for r in rows),
        "functional_failed": sum(r["checks"]["functional"] is False for r in rows),
        "functional_unknown": sum(r["checks"]["functional"] is None for r in rows),
        "assessed": sum(r["checks"]["assessed"] for r in rows),
        "comparable": sum(comparable),
        "unsupported": sum(
            r["checks"].get("status") == "Unsupported"
            or any(
                c["status"] == "Unsupported"
                for c in r["checks"].get("normal_checks", [])
            )
            for r in rows
        ),
        "target_hits": sum(hits),
        "new_hits": sum(
            c and h and not b for h, b, c in zip(hits, base_hits, comparable)
        ),
        "lost_hits": sum(
            c and b and not h for h, b, c in zip(hits, base_hits, comparable)
        ),
        "functional_regressions": sum(
            b["checks"]["functional"] is True and r["checks"]["functional"] is False
            for r, b in zip(rows, baseline)
        ),
        "uncertain_regressions": sum(
            b["checks"]["functional"] is True and r["checks"]["functional"] is None
            for r, b in zip(rows, baseline)
        ),
        "changed_outputs": sum(
            r["trace"]["token_ids"] != b["trace"]["token_ids"]
            for r, b in zip(rows, baseline)
        ),
        "maximum_relative_change": max(
            (r["intervention"]["maximum_relative_change"] for r in rows), default=0.0
        ),
        "changed_tokens": sum(r["intervention"]["changed_tokens"] for r in rows),
        "budget_violations": sum(
            r["intervention"]["maximum_relative_change"]
            > r["intervention"]["requested_fraction"] * 1.00001
            for r in rows
        ),
    }


def rank_groups(groups):
    return sorted(
        groups,
        key=lambda g: (
            g["summary"].get("budget_violations", 0),
            g["summary"]["functional_regressions"],
            g["summary"].get("uncertain_regressions", 0),
            -g["summary"].get("comparable", 0),
            -g["summary"]["new_hits"],
            -g["summary"]["target_hits"],
            -g["summary"]["functional"],
            -g["summary"]["assessed"],
            len(g["features"]),
            g["steps"] if g["steps"] is not None else 10**9,
            g["id"],
        ),
    )


def complete_comparison(measured):
    return (
        measured["cases"] > 0
        and measured["assessed"] == measured["cases"]
        and measured["comparable"] == measured["cases"]
        and not measured["functional_regressions"]
        and not measured["uncertain_regressions"]
        and not measured.get("budget_violations", 0)
    )


def confirmed_effect(selected, controls, replay_ok):
    return bool(
        replay_ok
        and complete_comparison(selected)
        and all(complete_comparison(s) for s in controls)
        and selected["new_hits"]
        and selected["target_hits"] > max(s["target_hits"] for s in controls)
    )


def pair_gain(pair, singles):
    if not all(complete_comparison(s) for s in [pair, *singles]):
        return None
    return pair["target_hits"] - max(s["target_hits"] for s in singles)


def reevaluate_result(measured):
    """Refresh checks on existing responses without inference or reselection."""
    result = deepcopy(measured)
    cases = {c["id"]: c for c in result["cases"]}

    def update(row):
        row["checks"] = evaluate(row["response"], cases[row["case_id"]])

    for row in result["base"]:
        update(row)
    for group in result["groups"]:
        for row in group["rows"]:
            update(row)
        group["summary"] = summary(group["rows"], result["base"], result["goal"])
    by_id = {g["id"]: g for g in result["groups"]}
    for group in result["groups"]:
        if group["kind"] == "Pair":
            group["gain_over_best_single"] = pair_gain(
                group["summary"], [by_id[i]["summary"] for i in group["single_ids"]]
            )
    for block in result["validation"] + result["full_generation"]:
        for row in [block["base"], block["selected"], *block["random"]]:
            update(row)
        if "reverse" in block:
            update(block["reverse"])
    reserved_base = [b["base"] for b in result["validation"]]
    result["base_summary"] = summary(result["base"], result["base"], result["goal"])
    result["reserved_summary"] = summary(
        [b["selected"] for b in result["validation"]], reserved_base, result["goal"]
    )
    result["reserved_random"] = [
        summary(
            [b["random"][i] for b in result["validation"]],
            reserved_base,
            result["goal"],
        )
        for i in range(2)
    ]
    result["full_summary"] = summary(
        [b["selected"] for b in result["full_generation"]],
        [b["base"] for b in result["full_generation"]],
        result["goal"],
    )
    selected = by_id[result["selected_id"]]["summary"]
    result["search_gain_observed"] = bool(
        complete_comparison(selected) and selected["new_hits"]
    )
    result["confirmed"] = confirmed_effect(
        result["reserved_summary"],
        result["reserved_random"],
        result["baseline_replay_passed"],
    )
    result["reevaluated_from"] = result["id"]
    result["id"] = uuid.uuid4().hex
    result["assessment_updated_unix"] = time.time()
    result["evaluator_version"] = EVALUATOR_VERSION
    result["protocol"] = 2
    result["selection_preserved"] = True
    return result


def _generate(task, settings, layer, features, direction, fraction, steps, seed):
    # Caller owns MODEL_LOCK. Runtime owns hook cleanup and seeding.
    inputs, input_ids, length = prepare_case_inputs(runtime, task)
    trace = GenerationTrace(layer)
    intervention = (
        None
        if direction is None
        else ResidualIntervention(layer, direction, fraction, steps)
    )
    runtime.generate_answer(
        inputs,
        length,
        settings.max_new_tokens,
        settings.temperature,
        seed=seed,
        trace=trace,
        residual_intervention=intervention,
    )
    suffix = runtime.processor.decode(
        trace.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    measured = trace.measure(runtime, features, settings.max_new_tokens, suffix.strip())
    prefix = task.get("assistant_prefix", "")
    response = prefix + suffix
    if measured["text_spans"]:
        offset = len(prefix) + len(suffix) - len(suffix.lstrip())
        measured["text_spans"] = [
            [a + offset, b + offset] for a, b in measured["text_spans"]
        ]
    return {
        "case_id": task["id"],
        "response": response,
        "input_ids": input_ids.tolist(),
        "seed": seed,
        "checks": evaluate(response, task),
        "review": review_python(response),
        "trace": measured,
        "intervention": intervention.metadata()
        if intervention
        else {
            "requested_fraction": 0,
            "window_tokens": None,
            "changed_tokens": 0,
            "maximum_relative_change": 0.0,
            "steps": [],
        },
    }


def run_experiment(
    session,
    settings,
    layer,
    polarity="B higher",
    goal=GOALS[0],
    candidate_count=6,
    case_count=2,
    budget_percent=2.0,
    window=8,
    progress=None,
):
    settings.validate()
    if polarity not in POLARITIES or goal not in GOALS:
        raise ValueError("Choose the candidate sign and SQL test goal.")
    if int(candidate_count) != candidate_count or not 1 <= candidate_count <= 6:
        raise ValueError("Choose 1–6 candidate features.")
    if int(case_count) != case_count or not 1 <= case_count <= 4:
        raise ValueError("Choose 1–4 examples per split.")
    if int(window) != window or not 1 <= window <= 64:
        raise ValueError("Choose a continuation window of 1–64 tokens.")
    # Also validates finiteness; done before any model load.
    fraction = float(budget_percent) / 100
    ResidualIntervention(layer, torch.ones(1), fraction, int(window))
    with session.lock:
        require_profile(session, settings)
        if layer not in runtime.LAYERS or layer not in session.profile:
            raise ValueError("Choose a layer from the current common profile.")
        profile = session.profile[layer]
        candidates = rank_candidates(profile, int(candidate_count), polarity)
        if not candidates:
            raise ValueError(
                f"No nonzero {polarity} candidates in this layer. Choose another sign or layer."
            )
        profile_id, width = session.profile_id, profile.a.shape[1]
        job = session.code_experiment_job = uuid.uuid4().hex
    cases = tasks(int(case_count))
    search = [c for c in cases if c["split"] == "Search"]
    reserved = [c for c in cases if c["split"] == "Reserved"]
    features = [r["feature"] for r in candidates]
    completed, total = 0, int(case_count) * (len(features) + 17) + 1

    def check_current():
        with session.lock:
            if (
                session.code_experiment_job != job
                or session.profile_id != profile_id
                or session.capture_key != settings.capture_key
            ):
                raise ValueError(
                    "Code experiment stopped: its inputs or profile changed."
                )

    def generate(case, direction=None, steps=None, label="Base"):
        nonlocal completed
        check_current()
        if progress:
            progress(
                min(completed / total, 0.99),
                desc=f"{case['split']} · {label} · {case['id']}",
            )
        check_current()
        with runtime.MODEL_LOCK:
            # Do not acquire session.lock here: profile capture uses the other order.
            if session.code_experiment_job != job or session.profile_id != profile_id:
                raise ValueError("Code experiment stopped before the next generation.")
            runtime.ensure_models_loaded(sae_layers=[layer])
            if runtime.saes[layer].W_dec.shape[0] != width:
                raise ValueError(
                    "The SAE dictionary differs from this common profile. Rebuild it."
                )
            row = _generate(
                case,
                settings,
                layer,
                features,
                direction,
                fraction,
                steps,
                settings.seed,
            )
        check_current()
        completed += 1
        return row

    try:
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded(sae_layers=[layer])
            decoder = runtime.saes[layer].W_dec
            if decoder.shape[0] != width:
                raise ValueError(
                    "The SAE dictionary differs from this common profile. Rebuild it."
                )
            directions = {
                r["feature"]: feature_direction(decoder, [r], goal == GOALS[0])
                for r in candidates
            }
        base = [generate(c) for c in search]
        groups, by_id = [], {}

        def group(identifier, name, ids, direction, steps, kind):
            rows = [generate(c, direction, steps, name) for c in search]
            measured = {
                "id": identifier,
                "name": name,
                "features": ids,
                "kind": kind,
                "steps": steps,
                "rows": rows,
                "summary": summary(rows, base, goal),
            }
            groups.append(measured)
            by_id[identifier] = (measured, direction)
            return measured

        for feature in features:
            group(
                f"feature-{feature}",
                f"Feature #{feature}",
                [feature],
                directions[feature],
                int(window),
                "Single",
            )
        randoms = [
            random_direction(decoder.shape[1], (settings.seed + 104729 + i) % (2**32))
            for i in range(2)
        ]
        for i, direction in enumerate(randoms):
            group(
                f"random-{i}",
                f"Random control {i + 1}",
                [],
                direction,
                int(window),
                "Random",
            )
        top = rank_groups([g for g in groups if g["kind"] == "Single"])[:3]
        for first, second in combinations(top, 2):
            ids = first["features"] + second["features"]
            direction = sum(directions[j] / directions[j].norm() for j in ids)
            if float(direction.norm()) <= 1e-8:
                continue
            pair = group(
                "pair-" + "-".join(map(str, ids)),
                " + ".join(f"#{j}" for j in ids),
                ids,
                direction,
                int(window),
                "Pair",
            )
            pair["single_ids"] = [first["id"], second["id"]]
            pair["gain_over_best_single"] = pair_gain(
                pair["summary"], [first["summary"], second["summary"]]
            )
        contenders = [g for g in groups if g["kind"] != "Random"]
        early = rank_groups(contenders)[0]
        direction = by_id[early["id"]][1]
        for steps, title in ((1, "First token"), (None, "All continuation tokens")):
            if steps == early["steps"]:
                continue
            group(
                f"timing-{steps}",
                f"{early['name']} · {title}",
                early["features"],
                direction,
                steps,
                "Timing",
            )
        selected = rank_groups([g for g in groups if g["kind"] != "Random"])[0]
        direction = by_id[selected["id"]][1]
        # Selection ends here. Reserved results never influence direction/schedule.
        validation = []
        for case in reserved:
            normal = generate(case)
            chosen = generate(case, direction, selected["steps"], "Frozen candidate")
            reverse = generate(case, -direction, selected["steps"], "Reverse direction")
            controls = [
                generate(case, d, selected["steps"], f"Random control {i + 1}")
                for i, d in enumerate(randoms)
            ]
            validation.append(
                {
                    "case_id": case["id"],
                    "base": normal,
                    "selected": chosen,
                    "reverse": reverse,
                    "random": controls,
                }
            )
        # A separate confirmation removes the forced function prefix. Its schedule
        # is continuous: it does not claim to locate a query during free generation.
        full = []
        for case in reserved:
            free_case = {**case, "assistant_prefix": ""}
            normal = generate(free_case)
            chosen = generate(
                free_case, direction, None, "Full generation · continuous"
            )
            controls = [
                generate(free_case, d, None, f"Full random {i + 1}")
                for i, d in enumerate(randoms)
            ]
            full.append(
                {
                    "case_id": case["id"],
                    "base": normal,
                    "selected": chosen,
                    "random": controls,
                }
            )
        replay = generate(search[0], label="Baseline replay")
        replay_ok = replay["trace"]["token_ids"] == base[0]["trace"]["token_ids"]
        reserved_base = [r["base"] for r in validation]
        selected_summary = summary(
            [r["selected"] for r in validation], reserved_base, goal
        )
        random_summaries = [
            summary([r["random"][i] for r in validation], reserved_base, goal)
            for i in range(2)
        ]
        result = {
            "kind": "code_experiment",
            "protocol": 2,
            "evaluator_version": EVALUATOR_VERSION,
            "id": job,
            "created_unix": time.time(),
            "profile_id": profile_id,
            "model": runtime.MODEL_ID,
            "sae_release": runtime.SAE_RELEASE,
            "sae_id": runtime.SAE_IDS[layer],
            "layer": layer,
            "dictionary_size": width,
            "settings": asdict(settings),
            "polarity": polarity,
            "goal": goal,
            "features": features,
            "candidates": candidates,
            "budget_percent": float(budget_percent),
            "search_window": int(window),
            "cases": cases,
            "base": base,
            "base_summary": summary(base, base, goal),
            "groups": groups,
            "selected_id": selected["id"],
            "search_gain_observed": bool(
                complete_comparison(selected["summary"])
                and selected["summary"]["new_hits"]
            ),
            "frozen_direction": (direction / direction.norm()).tolist(),
            "random_direction_seeds": [
                (settings.seed + 104729 + i) % (2**32) for i in range(2)
            ],
            "intervention_protocol": "Add a normalized decoder direction at the final position; preserve the residual reconstruction error. Equal per-token norm budgets, with actual rounded changes recorded. SAE features can coactivate; a decoder intervention is not guaranteed to change only the named feature.",
            "validation": validation,
            "reserved_summary": selected_summary,
            "reserved_random": random_summaries,
            "confirmed": confirmed_effect(
                selected_summary, random_summaries, replay_ok
            ),
            "full_generation": full,
            "full_summary": summary(
                [r["selected"] for r in full], [r["base"] for r in full], goal
            ),
            "baseline_replay_passed": replay_ok,
            "generations": completed,
            "selection_rule": "Fewest known/unknown functional regressions, then most comparable cases, new goal hits, total goal hits, functional/assessed cases, fewest features and shortest window; search tasks only. Selection without a complete positive gain is exploratory.",
            "limitations": "Synthetic lookup tasks and one SQL probe. Unsupported code is inconclusive. Reserved means excluded from this experiment's search, not proven disjoint from profile inputs. Pair gains at equal total norm are not proof of feature synergy. Full generation tests the frozen direction continuously, with a different schedule from localized continuations.",
        }
        with session.lock:
            check_current()
            record_event(session, "code_experiment", result)
            session.code_experiment_job = None
        if progress:
            progress(1, desc="Generated-code experiment ready")
        return result
    finally:
        with session.lock:
            if session.code_experiment_job == job:
                session.code_experiment_job = None
