"""Session-owned snapshots and paired transfer matrices. Model access is serialized."""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from research.causal_interventions import label_ids
from research.transfer import (
    classification_prompt,
    content_key,
    decision_readout,
    summarize_cell,
)
from . import model_runtime as runtime
from .artifact_store import ARTIFACT_ROOT, record_event
from .manifests import load_manifest, MAX_MANIFEST_BYTES
from .workflow import require_profile, validate_strengths

MAX_SOURCES = 6
MAX_TARGETS = 6
DEFAULT_TARGETS = [
    "sql_injection/manifest.validation.json",
    "command_injection/manifest.validation.json",
    "xss/manifest.validation.json",
]


@dataclass(frozen=True)
class FrozenSteering:
    metadata: dict
    directions: dict
    strengths: dict
    calibration_keys: frozenset


def freeze_current(session, settings, strengths, name):
    with session.lock:
        require_profile(session, settings, vectors=True)
        validate_strengths(strengths)
        if not any(
            strengths.get(layer, 0) and bool(v.direction.count_nonzero())
            for layer, v in session.vectors.items()
        ):
            raise ValueError(
                "Set at least one nonzero layer strength with a nonzero vector before freezing it."
            )
        if len(session.transfer_sources) >= MAX_SOURCES:
            raise ValueError(
                f"Keep at most {MAX_SOURCES} frozen sources. Clear them to start a new matrix."
            )
        name = (name or "").strip() or " + ".join(m.name for m in session.manifests)
        if not name or len(name) > 160:
            raise ValueError("Source names must contain 1–160 characters.")
        if name in {s.metadata["name"] for s in session.transfer_sources.values()}:
            raise ValueError(
                "Use a distinct source name so matrix rows stay unambiguous."
            )
        directions = {
            k: v.direction.detach().float().cpu().clone()
            for k, v in session.vectors.items()
        }
        with runtime.MODEL_LOCK:
            model_metadata = runtime.runtime_metadata()
        identifier = uuid.uuid4().hex
        metadata = {
            "id": identifier,
            "name": name,
            "model_id": runtime.MODEL_ID,
            "model_commit": model_metadata["model_commit"],
            "sae_release": runtime.SAE_RELEASE,
            "sae_ids": {str(k): runtime.SAE_IDS[k] for k in directions},
            "profile_id": session.profile_id,
            "vector_id": session.vector_id,
            "capture": {
                "token_scope": settings.token_scope,
                "aggregation": settings.aggregation,
            },
            "strengths": {str(k): float(v) for k, v in strengths.items()},
            "last_token_only": runtime.STEER_LAST_TOKEN_ONLY,
            "manifests": [
                {"name": m.name, "fingerprint": m.fingerprint}
                for m in session.manifests
            ],
            "vectors": {
                str(k): {
                    "sha256": hashlib.sha256(
                        v.contiguous().numpy().tobytes()
                    ).hexdigest(),
                    "norm": float(v.norm()),
                    "metadata": deepcopy(session.vectors[k].metadata),
                }
                for k, v in directions.items()
            },
        }
        snapshot = FrozenSteering(
            metadata,
            directions,
            dict(strengths),
            frozenset(
                content_key(c)
                for m in session.manifests
                for p in m.pairs
                for c in (p.a, p.b)
            ),
        )
        session.transfer_sources[identifier] = snapshot
        reset_evaluation(session)
        record_event(session, "transfer_source", metadata)
        return identifier


def reset_evaluation(session):
    session.transfer_plan = session.transfer_result = None


def clear_sources(session):
    with session.lock:
        session.transfer_sources.clear()
        reset_evaluation(session)


def prepare_transfer(session, selected_sources, paths, pair_limit, seed):
    if int(pair_limit) != pair_limit or not 1 <= pair_limit <= 20:
        raise ValueError("Choose 1–20 pairs per evaluation set.")
    if (
        not isinstance(seed, (int, float))
        or int(seed) != seed
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("Choose a valid integer seed.")
    selected_sources = list(dict.fromkeys(selected_sources or []))
    paths = list(dict.fromkeys(paths or []))
    if not 1 <= len(paths) <= MAX_TARGETS:
        raise ValueError(f"Select 1–{MAX_TARGETS} evaluation manifests.")
    with session.lock:
        if not selected_sources or any(
            k not in session.transfer_sources for k in selected_sources
        ):
            raise ValueError("Freeze and select at least one source vector first.")
        sources = [session.transfer_sources[k] for k in selected_sources]
        targets = []
        for path in paths:
            path = Path(path)
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("Evaluation manifests must be smaller than 5 MB.")
            raw = json.loads(path.read_text(encoding="utf-8"))
            manifest = load_manifest(path)
            labels = raw.get("labels", {}) if isinstance(raw, dict) else {}
            if not isinstance(labels, dict):
                raise ValueError(
                    f"{manifest.name}: labels must be an object with A/B descriptions."
                )
            if not labels and isinstance(raw, dict) and raw.get("cwe_id"):
                title = raw.get("title", raw["cwe_id"])
                labels = {
                    "A": f"The shown path avoids {title}.",
                    "B": f"The shown path exhibits {title}.",
                }
            if any(not isinstance(labels.get(k), str) for k in ("A", "B")):
                raise ValueError(
                    f"{manifest.name}: labels.A and labels.B must be text descriptions."
                )
            a, b = (labels[k].strip() for k in ("A", "B"))
            if not a or not b or a == b or max(len(a), len(b)) > 800:
                raise ValueError(
                    f"{manifest.name}: add distinct labels.A and labels.B descriptions (1–800 characters) to the manifest."
                )
            if any(t["name"] == manifest.name for t in targets):
                raise ValueError("Evaluation manifests must have distinct names.")
            pairs = sorted(
                manifest.pairs,
                key=lambda p: hashlib.sha256(
                    f"{int(seed)}:{manifest.fingerprint}:{p.id}".encode()
                ).hexdigest(),
            )[: int(pair_limit)]
            target_id = f"target-{len(targets)}"
            cases = [
                {
                    "case_id": f"{target_id}:{index}:{side}",
                    "pair_id": pair.id,
                    "expected": side,
                    "text": condition.text,
                    "prompt": classification_prompt(condition.text, a, b),
                    "condition": condition,
                    "content_key": content_key(condition),
                }
                for index, pair in enumerate(pairs)
                for side, condition in (("A", pair.a), ("B", pair.b))
            ]
            targets.append(
                {
                    "id": target_id,
                    "name": manifest.name,
                    "fingerprint": manifest.fingerprint,
                    "split": raw.get("split", "unspecified")
                    if isinstance(raw, dict)
                    else "unspecified",
                    "labels": {"A": a, "B": b},
                    "cases": cases,
                    "overlap": {
                        s.metadata["id"]: sum(
                            c["content_key"] in s.calibration_keys for c in cases
                        )
                        for s in sources
                    },
                }
            )
        plan = {
            "id": uuid.uuid4().hex,
            "source_ids": selected_sources,
            "seed": int(seed),
            "pair_limit": int(pair_limit),
            "targets": targets,
        }
        session.transfer_plan, session.transfer_result = plan, None
        return plan


def run_transfer(session, progress=None):
    with session.lock:
        plan = session.transfer_plan
        if plan is None:
            raise ValueError("Prepare the transfer sample first.")
        sources = [session.transfer_sources[k] for k in plan["source_ids"]]
        with runtime.MODEL_LOCK:
            runtime.ensure_models_loaded(with_saes=False)
            metadata = runtime.runtime_metadata()
            if any(
                s.metadata["model_id"] != runtime.MODEL_ID
                or s.metadata["model_commit"] != metadata["model_commit"]
                for s in sources
            ):
                raise ValueError(
                    "Frozen sources belong to another model revision. Clear them and rebuild on this model."
                )
            labels = label_ids(runtime, "A", "B")
        total = sum(len(t["cases"]) for t in plan["targets"]) * (1 + len(sources))
        completed, cells, targets = 0, [], []
        for target in plan["targets"]:
            baseline, treatments = [], {s.metadata["id"]: [] for s in sources}
            for case in target["cases"]:
                with runtime.MODEL_LOCK:
                    inputs, _, _ = runtime.prepare_inputs(
                        case["condition"].load_image(), case["prompt"]
                    )
                    shared = {k: case[k] for k in ("case_id", "pair_id", "expected")}
                    baseline.append(
                        {**shared, **decision_readout(runtime, inputs, labels)}
                    )
                    completed += 1
                    if progress:
                        progress(completed / total, desc=f"{target['name']} · base")
                    for source in sources:
                        treatments[source.metadata["id"]].append(
                            {
                                **shared,
                                **decision_readout(
                                    runtime,
                                    inputs,
                                    labels,
                                    source.directions,
                                    source.strengths,
                                    source.metadata["last_token_only"],
                                ),
                            }
                        )
                        completed += 1
                        if progress:
                            progress(
                                completed / total,
                                desc=f"{source.metadata['name']} → {target['name']}",
                            )
                    del inputs
            targets.append(
                {k: v for k, v in target.items() if k != "cases"}
                | {
                    "cases": [
                        {k: v for k, v in c.items() if k != "condition"}
                        | {"image_sha256": c["condition"].image_sha256}
                        for c in target["cases"]
                    ],
                    "baseline": baseline,
                }
            )
            for source in sources:
                source_id = source.metadata["id"]
                cells.append(
                    {
                        "id": f"{source_id}:{target['id']}",
                        "source_id": source_id,
                        "target_id": target["id"],
                        "overlap": target["overlap"][source_id],
                        "rows": treatments[source_id],
                        "summary": summarize_cell(baseline, treatments[source_id]),
                    }
                )
        result = {
            "kind": "transfer_matrix",
            "protocol": 1,
            "id": uuid.uuid4().hex,
            "model_id": runtime.MODEL_ID,
            "runtime": metadata,
            "seed": plan["seed"],
            "readout": "Single next-token A/B decision; all sampled cases; fixed prompts; no free-response generation",
            "sources": [s.metadata for s in sources],
            "targets": targets,
            "cells": cells,
            "forwards": total,
        }
        session.transfer_result = result
        record_event(session, "transfer_matrix", result)
        return result


def export_report(session):
    from .transfer_view import standalone_report

    with session.lock:
        result = session.transfer_result
        if not result:
            raise ValueError("Run a transfer matrix before exporting it.")
        directory = ARTIFACT_ROOT / session.id / "transfer_reports" / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        json_path, html_path = (
            directory / "measurements.json",
            directory / "report.html",
        )
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        html_path.write_text(standalone_report(result), encoding="utf-8")
        return [str(html_path), str(json_path)]
