"""Session-owned paired studies, independent of UI layout and vector builders."""

from dataclasses import asdict
import hashlib
import uuid
from research.feature_studies import input_view, measure_feature
from research.causal_interventions import generate_feature_report
from . import causal_lab, model_runtime as runtime
from .artifact_store import record_event


def cases_for_pair(pair, view, instruction=""):
    cases = []
    for side, condition in (("A", pair.a), ("B", pair.b)):
        text = input_view(condition.text, view)
        if instruction.strip():
            text += "\n\n" + instruction.strip()
        cases.append((side, causal_lab.make_case(text, condition.load_image(), "")))
    return cases


def check_dictionary(candidate, expected):
    if expected and candidate["dictionary_size"] != int(expected):
        raise ValueError(
            f"Expected {int(expected):,} SAE features, but loaded {candidate['dictionary_size']:,} "
            f"({candidate['sae_id']}). Feature IDs do not transfer between dictionaries. "
            "For the 262k reference, configure SAE_WIDTH=262k and restart the server."
        )


def candidate_info(candidate):
    c = candidate
    text = (
        f"**Layer {c['layer']} · feature #{c['feature_id']}** · "
        f"{c['dictionary_size']:,} features\n\n"
        f"Model: `{c['model_id']}` · SAE: `{c['sae_release']}/{c['sae_id']}`\n\n"
        f"Native coefficient per dose: **{c['intervention_step']:+.5g}**."
    )
    if "mean_delta" in c:
        text += (
            f" Mean A: **{c['mean_a']:.4g}** · Mean B: **{c['mean_b']:.4g}** · "
            f"B − A: **{c['mean_delta']:+.4g}** · same direction: "
            f"**{c['same_direction_pairs']}/{c['pair_count']}** (calibration pairs)."
        )
    return text


def activation_study(
    session, settings, selection, expected, manifests, views, limit, progress=None
):
    if not manifests or not views:
        raise ValueError("Load research inputs and select at least one input view.")
    limit = int(limit)
    if not 1 <= limit <= 24:
        raise ValueError("Choose 1–24 pairs.")
    pairs = [(m, p) for m in manifests for p in m.pairs][:limit]
    # Preflight all transformations before any GPU work; no silently omitted cases.
    planned = [(m, p, v, cases_for_pair(p, v)) for m, p in pairs for v in views]
    rows = []
    with session.lock, runtime.MODEL_LOCK:
        candidate = causal_lab.select_candidate(session, settings, *selection)
        check_dictionary(candidate, expected)
        for index, (manifest, pair, view, cases) in enumerate(planned):
            measured = [
                measure_feature(runtime, case, candidate, settings.token_scope)
                for _, case in cases
            ]
            a, b = measured
            rows.append(
                {
                    "dataset": manifest.name,
                    "pair": pair.id,
                    "view": view,
                    "mean_a": a["mean"],
                    "mean_b": b["mean"],
                    "delta": b["mean"] - a["mean"],
                    "active_a": a["active_fraction"],
                    "active_b": b["active_fraction"],
                    "tokens_a": a["selected_tokens"],
                    "tokens_b": b["selected_tokens"],
                }
            )
            if progress:
                progress((index + 1) / len(planned), desc=f"{pair.id} · {view}")
        record_event(
            session,
            "feature_activation_study",
            {
                "candidate": candidate,
                "settings": asdict(settings),
                "measurement": "arithmetic mean over selected input tokens; no generated tokens",
                "manifests": [m.metadata() for m in manifests],
                "pair_limit": limit,
                "views": list(views),
                "rows": rows,
            },
        )
    return candidate, rows


def response_study(
    session,
    settings,
    selection,
    expected,
    cases,
    dose,
    schedule,
    ablation,
    random_count,
    progress=None,
):
    dose = float(dose)
    if not 0 <= dose <= 4 or int(random_count) not in (1, 2, 3):
        raise ValueError("Choose a dose from 0 to 4 and 1–3 random controls.")
    with session.lock, runtime.MODEL_LOCK:
        candidate = causal_lab.select_candidate(session, settings, *selection)
        check_dictionary(candidate, expected)
        conditions = [
            ("Base", 0, None, False, None),
            ("Toward B", dose, None, False, None),
            ("Toward A", -dose, None, False, None),
        ]
        if ablation:
            conditions.append(("Feature ablation", 0, None, True, None))
        for index in range(int(random_count)):
            direction_seed = settings.seed + index
            direction = causal_lab.random_direction(candidate, direction_seed)
            conditions.append(
                (f"Random {index + 1}", dose, direction, False, direction_seed)
            )
        results = []
        study_id = uuid.uuid4().hex
        for case_name, case in cases:
            for name, amount, direction, remove, direction_seed in conditions:
                result = generate_feature_report(
                    runtime,
                    case,
                    candidate,
                    amount,
                    settings.max_new_tokens,
                    direction=direction,
                    schedule=schedule,
                    temperature=settings.temperature,
                    seed=settings.seed,
                    ablate=remove,
                )
                results.append(
                    {
                        "study_id": study_id,
                        "case": case_name,
                        "condition": name,
                        "direction_seed": direction_seed,
                        **result,
                    }
                )
                if progress:
                    progress(
                        len(results) / (len(cases) * len(conditions)),
                        desc=f"{case_name} · {name}",
                    )
        zero_ok = None
        if dose == 0:
            zero_ok = all(
                len(
                    {
                        r["answer"]
                        for r in results
                        if r["case"] == name and not r["ablation"]
                    }
                )
                == 1
                for name, _ in cases
            )
        record_event(
            session,
            "feature_response_study",
            {
                "study_id": study_id,
                "candidate": candidate,
                "settings": asdict(settings),
                "dose": dose,
                "schedule": schedule,
                "cases": [
                    {
                        "name": name,
                        **{k: v for k, v in case.items() if k != "image"},
                        "image": image_metadata(case["image"]),
                    }
                    for name, case in cases
                ],
                "results": results,
                "zero_control_passed": zero_ok,
            },
        )
    return candidate, results, zero_ok


def image_metadata(image):
    if image is None:
        return None
    return {
        "mode": image.mode,
        "size": list(image.size),
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
