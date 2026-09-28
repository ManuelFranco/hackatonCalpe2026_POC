"""Create a single-feature experiment from a saved section-1 calibration."""

import json
from pathlib import Path
import shutil
import time
import uuid

import torch

from sae_dashboard.causal_lab import scale_candidate, generate_baseline_prefix
from sae_dashboard.security_reports import report_labels


def snapshot_image(image, directory, name="input"):
    if not image:
        return None
    source = Path(image)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (name + source.suffix.lower())
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return str(target.resolve())


def experiment_from_calibration(app, session, key, query_image=None, query_prompt="", calibration_id=None):
    """Read actual calibration scores; never import discovery-suite candidates.

    The caller owns MODEL_LOCK and has loaded the matching model and SAEs.
    A/B inputs have no inferred truth labels or independent validation results.
    """
    if not session:
        raise ValueError("Build a profile in section 1 before selecting a feature.")
    if calibration_id != session["run_id"]:
        raise ValueError("This row belongs to an older calibration. Select a row from the current profile.")
    layer, feature = (int(part) for part in key.split(":"))
    if layer not in app.LAYERS:
        raise ValueError("The selected layer is not part of this calibration.")
    root = Path(session["run_dir"])
    profile_path = Path(session["profile_path"])
    app.load_steering_directions(str(profile_path), token_scope=session["feature_token_scope"])
    metadata = json.loads((root / "experiment.json").read_text())
    if metadata["run_id"] != session["run_id"]:
        raise ValueError("Calibration identity does not match the selected row.")
    if metadata["runtime"]["model_commit"] != app.runtime_metadata()["model_commit"]:
        raise ValueError("The model revision changed. Build a new profile in section 1.")
    profile = torch.load(profile_path.parent / f"layer_{layer}_common_profile.pt",
                         map_location="cpu", weights_only=True)
    mask = profile["effective_mask"]
    if not 0 <= feature < mask.numel() or not bool(mask[feature]):
        raise ValueError("Select an included feature with a nonzero calibration mean.")
    a, b, cases = [], [], []
    for pair in metadata["pairs"]:
        index = pair["pair_index"]
        for side, scores in (("A", a), ("B", b)):
            condition = pair[f"condition_{side}"]
            saved = torch.load(root / "pairs" / f"pair_{index:02d}" / f"condition_{side}"
                               / f"layer_{layer}.pt", map_location="cpu", weights_only=True)
            scores.append(float(saved["feature_score"][feature]))
            image = condition.get("image_path")
            if condition.get("image_filename") and (not image or not Path(image).is_file()):
                raise ValueError("Calibration images are unavailable. Rebuild the profile to preserve them.")
            pos, neg = report_labels(condition["prompt"])
            cases.append({"id": f"calibration_pair_{index:02d}_{side}", "split": "calibration",
                          "side": side, "prompt": condition["prompt"], "image": image,
                          "assistant_prefix": condition.get("assistant_prefix", ""),
                          "assistant_prefix_ids": condition.get("assistant_prefix_ids"),
                          "image_sha256": condition.get("image_sha256"),
                          "positive_label": pos, "negative_label": neg})
    deltas = profile["pair_deltas_B_minus_A"][:, feature].float()
    mean = float(profile["feature_delta_common_only"][feature])
    epsilon = profile["feature_presence_epsilon"]
    candidate = {"layer": layer, "feature_id": feature,
                 "mean_A": sum(a) / len(a), "mean_B": sum(b) / len(b),
                 "natural_delta": mean, "pair_deltas": deltas.tolist(),
                 "train_agreement": float((deltas * (1 if mean > 0 else -1) > epsilon).float().mean()),
                 "activation_p95": float(torch.quantile(torch.tensor(a + b), .95))}
    candidate = scale_candidate(candidate, profile["reference_residual_norm"],
                                float(app.saes[layer].W_dec[feature].float().norm()))
    output = root / "causal_features" / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]) / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if query_image or (query_prompt or "").strip():
        copied_image = snapshot_image(query_image, output.parent, "query")
        prefix_data = generate_baseline_prefix(app, query_image, query_prompt or "", metadata.get("prefix_marker", ""))
        pos, neg = report_labels(query_prompt or "")
        cases.append({"id": "query_from_section_2", "split": "query", **prefix_data,
                      "prompt": query_prompt or "", "image": copied_image,
                      "image_sha256": app.sha256_file(copied_image),
                      "positive_label": pos, "negative_label": neg})
        default = "query_from_section_2"
    else:
        default = cases[0]["id"]
    report = {
        "format_version": 2, "mode": "calibration", "created_unix": time.time(),
        "model_id": metadata["model_id"], "runtime": metadata["runtime"],
        "sae_release": app.SAE_RELEASE, "sae_ids": metadata["sae_ids"],
        "prefix_marker": metadata.get("prefix_marker", ""),
        "intervention_schedule": metadata.get("intervention_schedule", "Every decoding step"),
        "source_profile": {"run_id": metadata["run_id"], "path": str(profile_path.resolve()),
                           "sha256": app.sha256_file(str(profile_path)),
                           "num_pairs": metadata["num_pairs"],
                           "feature_token_scope": metadata["feature_token_scope"],
                           "aggregation": metadata["aggregation"]},
        "selection_method": "User-selected effective common feature from section 1; no independent validation.",
        "protocol": "Saved calibration feature scores determine means, sign agreement and p95 dose; the single decoder direction is applied at the final input position.",
        "selected": candidate, "candidates": [candidate], "cases": cases, "default_case": default,
        "validation": [], "baseline": {}, "validation_gate_passed": False,
        "artifact_path": str(output.resolve()),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
