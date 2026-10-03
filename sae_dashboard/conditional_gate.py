"""Session-scoped conditional steering: calibrate a visual gate, compare, measure leakage.

READ (gate)  : linear probe on mean image-token residuals at one layer.
WRITE (vector): the session's existing steering vectors (e.g. damaged - intact).

    h' = h + alpha * g(h) * v

The gate is stored next to, not inside, the A/B profile: rebuilding the profile or
vectors keeps a valid gate, and changing the gate data or settings discards only the
gate. All model work runs under MODEL_LOCK with request-local GateTrace objects.
"""

from __future__ import annotations
import hashlib
import math
from typing import Any
import torch
from research.causal_interventions import label_ids, readout
from research.conditional_steering import (
    GateTrace,
    LinearGate,
    classification_metrics,
    fit_gate,
    leakage_metrics,
    pool_visual_tokens,
    validate_gate_layer_order,
)
from . import model_runtime as runtime
from .artifact_store import record_event
from .gate_manifests import (
    GateManifest,
    check_disjoint,
    check_not_reserved,
    load_gate_manifest,
    reserved_evaluation_hashes,
)
from .session import Session, Settings

# Yes/no readouts. The INTACT question is a control: steering that only biases the
# model toward "Yes" raises both log-odds, while damage semantics lowers the second.
DAMAGE_QUESTION = "Is the main object in this image physically damaged? Answer Yes or No."
INTACT_QUESTION = (
    "Is the main object in this image intact and undamaged? Answer Yes or No."
)
OPEN_QUESTION = "Describe the main visible object and its physical condition."
POSITIVE_LABEL, NEGATIVE_LABEL = "Yes", "No"


def _path(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return getattr(value, "name", value)


def set_gate_manifests(session: Session, train_path, validation_path) -> list[list]:
    """Load gate calibration data. Never touches the A/B profile or vectors."""
    train_path, validation_path = _path(train_path), _path(validation_path)
    if not train_path or not validation_path:
        raise ValueError("Select both a gate training and a gate validation manifest.")
    train = load_gate_manifest(train_path)
    validation = load_gate_manifest(validation_path)
    if train.split != "train" or validation.split != "validation":
        raise ValueError(
            'Gate manifests must declare "split": "train" and "split": "validation".'
        )
    if train.target != validation.target:
        raise ValueError(
            f"Gate target mismatch: {train.target!r} vs {validation.target!r}."
        )
    check_disjoint(train, validation)
    reserved = reserved_evaluation_hashes()
    for manifest in (train, validation):
        check_not_reserved(manifest, reserved)
        if len(set(manifest.labels)) < 2:
            raise ValueError(
                f"The gate {manifest.split} manifest needs target and non-target images."
            )
    with session.lock:
        session.gate_manifests = {"train": train, "validation": validation}
        session.invalidate_gate()
    return manifest_table(session)


def manifest_table(session: Session) -> list[list]:
    return [
        [
            m.split,
            m.name,
            m.target,
            s["target_examples"],
            s["non_target_examples"],
            s["hard_negatives"],
        ]
        for m in session.gate_manifests.values()
        for s in [m.summary()]
    ]


def capture_visual_features(
    manifest: GateManifest, layer: int, progress=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean image-token residual at `layer` for every example (no steering)."""
    rows = []
    for index, example in enumerate(manifest.examples):
        if progress is not None:
            progress(
                index / len(manifest.examples),
                desc=f"Gate {manifest.split}: {example.id}",
            )
        image = example.condition.load_image()
        # One example per lock acquisition so other sessions can interleave.
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded(with_saes=False)
            inputs, ids, _ = runtime.prepare_inputs(image, example.condition.text)
            captured = runtime.capture_prompt_residuals(inputs, layers=[layer])
            pooled = pool_visual_tokens(captured[layer], runtime.get_image_mask(ids))
        if pooled is None:
            raise ValueError(f"Gate example {example.id} produced no image tokens.")
        rows.append(pooled)
    return torch.stack(rows), torch.tensor(manifest.labels, dtype=torch.float32)


def build_gate(
    session: Session,
    layer: int,
    mode: str = "hard",
    temperature: float = 1.0,
    max_false_positive_rate: float = 0.0,
    l2: float = 1e-2,
    seed: int = 0,
    progress=None,
) -> LinearGate:
    layer = int(layer)
    if layer < 0:
        raise ValueError("The gate layer must be nonnegative.")
    with session.lock:
        manifests = dict(session.gate_manifests)
    if set(manifests) != {"train", "validation"}:
        raise ValueError("Load gate training and validation manifests first.")
    train, validation = manifests["train"], manifests["validation"]
    train_x, train_y = capture_visual_features(train, layer, progress)
    val_x, val_y = capture_visual_features(validation, layer, progress)
    gate = fit_gate(
        train_x,
        train_y,
        val_x,
        val_y,
        layer=layer,
        target=train.target,
        mode=mode,
        temperature=float(temperature),
        max_false_positive_rate=float(max_false_positive_rate),
        l2=float(l2),
        seed=int(seed),
        metadata={
            "model_id": runtime.MODEL_ID,
            "train_manifest": train.summary(),
            "validation_manifest": validation.summary(),
            "prompt": train.prompt,
        },
    )
    gate_id = hashlib.sha256(
        repr(
            (
                train.fingerprint,
                validation.fingerprint,
                layer,
                mode,
                float(temperature),
                float(max_false_positive_rate),
                float(l2),
                int(seed),
                runtime.MODEL_ID,
            )
        ).encode()
    ).hexdigest()[:16]
    with session.lock:
        if session.gate_manifests.get("train") is not train:
            raise ValueError("Gate manifests changed during calibration. Rebuild.")
        session.conditional_gate = gate
        session.conditional_gate_id = gate_id
        # A new gate must be enabled explicitly before it changes any steering.
        session.gate_enabled = False
    record_event(session, "conditional_gate", {"gate": gate.config()})
    return gate


def set_gate_enabled(session: Session, enabled: bool) -> str:
    with session.lock:
        if enabled and session.conditional_gate is None:
            session.gate_enabled = False
            raise ValueError("Build the vehicle gate before enabling it.")
        session.gate_enabled = bool(enabled)
    return gate_status(session)


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def gate_status(session: Session) -> str:
    gate = session.conditional_gate
    if gate is None:
        loaded = ", ".join(
            f"{m.split}: {len(m.examples)}" for m in session.gate_manifests.values()
        )
        return "**Gate status:** not built" + (f" · loaded {loaded}" if loaded else "")
    val = gate.metadata.get("validation_metrics", {})
    state = (
        "**enabled** — steering is conditional"
        if session.gate_enabled
        else "built but **disabled** — legacy global steering"
    )
    return (
        f"**Gate status:** {state}  \n"
        f"Target: `{gate.target}` · layer {gate.layer} · mode {gate.mode}"
        + (f" (T={_fmt(gate.temperature)})" if gate.mode == "soft" else "")
        + f" · threshold {_fmt(gate.threshold)} · id `{session.conditional_gate_id}`  \n"
        f"Validation TPR {_fmt(val.get('tpr'))} · FPR {_fmt(val.get('fpr'))} · "
        f"TNR {_fmt(val.get('tnr'))} ({val.get('positives', 0)} target / "
        f"{val.get('negatives', 0)} non-target)"
    )


def gate_metrics_table(session: Session) -> list[list]:
    gate = session.conditional_gate
    if gate is None:
        return []
    rows = []
    for split in ("train", "validation"):
        m = gate.metadata.get(f"{split}_metrics", {})
        rows.append(
            [
                split,
                m.get("tp"),
                m.get("fp"),
                m.get("tn"),
                m.get("fn"),
                _fmt(m.get("tpr")),
                _fmt(m.get("fpr")),
                _fmt(m.get("tnr")),
            ]
        )
    return rows


def trace_summary(gate: LinearGate | None, trace: GateTrace | None) -> str:
    if gate is None or trace is None:
        return "Gate disabled · global steering: h' = h + αv for every input."
    if trace.score is None:
        reason = "no image tokens, so the gate is closed"
    else:
        reason = f"score {_fmt(trace.score)} vs threshold {_fmt(gate.threshold)}"
    applied = "applied" if trace.value else "**not applied** (zero intervention)"
    return (
        f"Gate `{gate.target}` · {reason} · g = {_fmt(trace.value)} · "
        f"steering {applied}."
    )


def _steering_snapshot(session: Session, settings: Settings, strengths):
    from .workflow import require_profile

    settings.validate()
    with session.lock:
        require_profile(session, settings)
        if not session.vectors:
            raise ValueError("Create steering vectors before conditional steering.")
        gate = session.conditional_gate
        if gate is None:
            raise ValueError("Build the vehicle gate first.")
        directions = {k: v.direction for k, v in session.vectors.items()}
    strengths = {int(k): float(v) for k, v in strengths.items()}
    if not all(math.isfinite(v) for v in strengths.values()):
        raise ValueError("Steering strengths must be finite.")
    if not any(strengths.get(k, 0.0) for k in directions):
        raise ValueError("Set a nonzero strength for at least one vector layer.")
    validate_gate_layer_order(gate.layer, strengths)
    return gate, directions, strengths


def compare_gated(
    session: Session, settings: Settings, strengths, text: str, image=None
) -> tuple[str, str, str]:
    """Base vs. gated-steered answers with the same prompt, seed and token budget."""
    if not (text or "").strip() and image is None:
        raise ValueError("Provide a prompt, an image, or both.")
    gate, directions, strengths = _steering_snapshot(session, settings, strengths)
    trace = GateTrace()
    with runtime.MODEL_LOCK:
        runtime.ensure_models_loaded(with_saes=False)
        inputs, _, length = runtime.prepare_inputs(image, text or "")
        common = {
            "inputs": inputs,
            "input_len": length,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "seed": settings.seed,
        }
        base = runtime.generate_answer(**common)
        steered = runtime.generate_answer(
            **common,
            steering_directions=directions,
            strengths=strengths,
            gate=gate,
            gate_trace=trace,
        )
    summary = trace_summary(gate, trace)
    with session.lock:
        record_event(
            session,
            "gated_compare",
            {
                "prompt": text,
                "has_image": image is not None,
                "strengths": {str(k): v for k, v in strengths.items()},
                "gate": trace.as_dict(),
                "base": base,
                "steered": steered,
            },
        )
    return base, steered, summary


def _log_odds(inputs, ids, **steering) -> float:
    return readout(runtime.next_token_logits(inputs, **steering), ids)["log_odds"]


def evaluate_leakage(
    session: Session,
    settings: Settings,
    strengths,
    manifest_path,
    generate_text: bool = False,
    progress=None,
) -> tuple[str, list[list], dict[str, Any]]:
    """Target effect vs. off-target leakage for global and gated steering.

    Per image, damage_shift = Δlog-odds(Yes|damaged?) − Δlog-odds(Yes|intact?), with
    Δ = steered − base. A pure "Yes" bias cancels out of damage_shift.
    """
    manifest = load_gate_manifest(_path(manifest_path))
    gate, directions, strengths = _steering_snapshot(session, settings, strengths)
    with session.lock:
        calibration = tuple(session.gate_manifests.values())
    check_disjoint(*calibration, manifest)
    rows, records = [], []
    for index, example in enumerate(manifest.examples):
        if progress is not None:
            progress(index / len(manifest.examples), desc=f"Leakage: {example.id}")
        image = example.condition.load_image()
        record = example.metadata()
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded(with_saes=False)
            ids = label_ids(runtime, POSITIVE_LABEL, NEGATIVE_LABEL)
            for key, question in (
                ("damaged", DAMAGE_QUESTION),
                ("intact", INTACT_QUESTION),
            ):
                inputs, _, _ = runtime.prepare_inputs(image, question)
                trace = GateTrace()
                base = _log_odds(inputs, ids)
                global_ = _log_odds(
                    inputs, ids, steering_directions=directions, strengths=strengths
                )
                gated = _log_odds(
                    inputs,
                    ids,
                    steering_directions=directions,
                    strengths=strengths,
                    gate=gate,
                    gate_trace=trace,
                )
                record[key] = {
                    "base": base,
                    "global": global_,
                    "gated": gated,
                    "gate": trace.as_dict(),
                }
            # Image tokens precede the question, so both questions share one gate.
            record["gate"] = record["damaged"]["gate"]
            if generate_text:
                inputs, _, length = runtime.prepare_inputs(image, OPEN_QUESTION)
                common = {
                    "inputs": inputs,
                    "input_len": length,
                    "max_new_tokens": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "seed": settings.seed,
                }
                record["open_ended"] = {
                    "base": runtime.generate_answer(**common),
                    "global": runtime.generate_answer(
                        **common, steering_directions=directions, strengths=strengths
                    ),
                    "gated": runtime.generate_answer(
                        **common,
                        steering_directions=directions,
                        strengths=strengths,
                        gate=gate,
                        gate_trace=GateTrace(),
                    ),
                }
        for mode in ("global", "gated"):
            d = record["damaged"][mode] - record["damaged"]["base"]
            i = record["intact"][mode] - record["intact"]["base"]
            record[f"{mode}_delta_damaged_q"] = d
            record[f"{mode}_delta_intact_q"] = i
            record[f"{mode}_damage_shift"] = d - i
        records.append(record)
        rows.append(
            [
                example.id,
                example.category,
                "target"
                if example.label
                else ("hard negative" if example.hard_negative else "non-target"),
                _fmt(record["gate"]["score"]),
                _fmt(record["gate"]["value"]),
                _fmt(record["global_damage_shift"]),
                _fmt(record["gated_damage_shift"]),
            ]
        )
    summary = summarize_leakage(gate, manifest, records)
    with session.lock:
        record_event(
            session,
            "gate_leakage_eval",
            {
                "manifest": manifest.summary(),
                "strengths": {str(k): v for k, v in strengths.items()},
                "questions": {
                    "damaged": DAMAGE_QUESTION,
                    "intact": INTACT_QUESTION,
                    "open": OPEN_QUESTION if generate_text else None,
                },
                "summary": summary,
                "records": records,
            },
        )
    return leakage_markdown(gate, summary), rows, summary


def summarize_leakage(gate: LinearGate, manifest: GateManifest, records) -> dict:
    """Pure aggregation over per-example records (testable without a model)."""
    is_target = [bool(r["label"]) for r in records]
    scores = torch.tensor(
        [
            -math.inf if r["gate"]["score"] is None else r["gate"]["score"]
            for r in records
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor(is_target)
    gate_metrics = classification_metrics(scores, labels, gate.threshold)
    hard = [r for r in records if r["hard_negative"]]
    hard_open = sum(1 for r in hard if r["gate"]["value"])
    closed = [r for r in records if not r["gate"]["value"]]
    return {
        "manifest": manifest.name,
        "split": manifest.split,
        "target": gate.target,
        "global": leakage_metrics([r["global_damage_shift"] for r in records], is_target),
        "gated": leakage_metrics([r["gated_damage_shift"] for r in records], is_target),
        "gate_classification": gate_metrics,
        "hard_negatives": len(hard),
        "hard_negative_fpr": hard_open / len(hard) if hard else None,
        "closed_gate_examples": len(closed),
        # Software invariant: a closed hard gate must produce exactly zero change.
        "closed_gate_max_abs_delta": max(
            (
                abs(r[f"gated_delta_{q}_q"])
                for r in closed
                for q in ("damaged", "intact")
            ),
            default=None,
        ),
    }


def leakage_markdown(gate: LinearGate, summary: dict) -> str:
    g, c, m = summary["global"], summary["gated"], summary["gate_classification"]
    return (
        f"**Leakage evaluation** · `{summary['manifest']}` ({summary['split']}) · "
        f"target `{gate.target}`\n\n"
        "| Steering | E_target | E_leak | max leak | selectivity |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| global | {_fmt(g['E_target'])} | {_fmt(g['E_leak'])} | "
        f"{_fmt(g['max_abs_leak'])} | {_fmt(g['selectivity'])} |\n"
        f"| gated | {_fmt(c['E_target'])} | {_fmt(c['E_leak'])} | "
        f"{_fmt(c['max_abs_leak'])} | {_fmt(c['selectivity'])} |\n\n"
        f"Gate on this set: TPR {_fmt(m['tpr'])} · FPR {_fmt(m['fpr'])} · "
        f"TNR {_fmt(m['tnr'])} · hard-negative FPR {_fmt(summary['hard_negative_fpr'])}. "
        "Damage shift = Δlog-odds(Yes | damaged?) − Δlog-odds(Yes | intact?). "
        "These are measurements on the loaded images, not general claims."
    )
