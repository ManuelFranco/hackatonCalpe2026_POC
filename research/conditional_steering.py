"""Conditional (gated) steering: a READ direction decides whether a WRITE direction is added.

    h' = h + alpha * g(h) * v_write

v_write is an ordinary steering vector (for example mean(damaged - intact) decoded
through the SAE). g is a gate computed by a small residual-space linear probe from
the *visual-token* states of the prompt at one gate layer:

    z     = mean_{t in image tokens} h_t^(gate layer)
    s     = w . ((z - center) / scale) + b
    hard: g = 1 if s >= threshold else 0
    soft: g = sigmoid((s - threshold) / temperature)

The probe is a detector, the steering vector is a writer; they are trained and stored
separately. Both are operationally identified directions, not proofs that a
coordinate "is" an object or an attribute.

Pure tensor code only: no model loading, no UI, no global state. The caller creates
one GateTrace per request; the hooks fill it during prefill and reuse it while decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

GATE_MODES = ("hard", "soft")
GATE_FORMAT_VERSION = 1


@dataclass(frozen=True, eq=False)
class LinearGate:
    """Residual-space linear probe over mean-pooled visual tokens at one layer."""

    layer: int
    weight: torch.Tensor  # [hidden]
    bias: float
    threshold: float
    center: torch.Tensor  # [hidden]; subtracted before projection
    scale: float  # scalar normalization applied after centering
    target: str
    mode: str = "hard"
    temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    @property
    def hidden_size(self) -> int:
        return int(self.weight.shape[0])

    def validate(self, hidden_size: int | None = None) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise ValueError("The gate layer must be an integer.")
        if self.layer < 0:
            raise ValueError("The gate layer must be nonnegative.")
        for name, tensor in (("weight", self.weight), ("center", self.center)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                raise ValueError(f"The gate {name} must be a 1-D tensor.")
            if not tensor.numel():
                raise ValueError(f"The gate {name} is empty.")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"The gate {name} contains NaN or infinite values.")
        if self.center.shape != self.weight.shape:
            raise ValueError("The gate center and weight shapes differ.")
        if hidden_size is not None and self.hidden_size != int(hidden_size):
            raise ValueError(
                f"The gate width {self.hidden_size} does not match the model width "
                f"{hidden_size}."
            )
        for name, value in (
            ("bias", self.bias),
            ("threshold", self.threshold),
            ("scale", self.scale),
            ("temperature", self.temperature),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"The gate {name} must be a finite number.")
        if self.scale <= 0:
            raise ValueError("The gate scale must be positive.")
        if self.mode not in GATE_MODES:
            raise ValueError(f"The gate mode must be one of {GATE_MODES}.")
        if self.temperature <= 0:
            raise ValueError("The soft-gate temperature must be positive.")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("The gate target category must be a nonempty string.")

    def score(self, pooled: torch.Tensor) -> float:
        pooled = pooled.detach().float().cpu().reshape(-1)
        if pooled.shape != self.weight.shape:
            raise ValueError("Pooled visual state and gate weight shapes differ.")
        normalized = (pooled - self.center.float()) / float(self.scale)
        return float(normalized @ self.weight.float()) + float(self.bias)

    def value(self, score: float | None) -> float:
        return gate_value(score, self.threshold, self.mode, self.temperature)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "weight": self.weight.detach().float().cpu().contiguous(),
            "center": self.center.detach().float().cpu().contiguous(),
        }

    def config(self) -> dict[str, Any]:
        return {
            "format_version": GATE_FORMAT_VERSION,
            "kind": "linear_visual_gate",
            "layer": self.layer,
            "bias": float(self.bias),
            "threshold": float(self.threshold),
            "scale": float(self.scale),
            "target": self.target,
            "mode": self.mode,
            "temperature": float(self.temperature),
            "hidden_size": self.hidden_size,
            "pooling": "mean_over_image_tokens",
            "metadata": self.metadata,
        }

    @classmethod
    def from_artifacts(
        cls, tensors: dict[str, torch.Tensor], config: dict[str, Any]
    ) -> "LinearGate":
        if not isinstance(config, dict):
            raise ValueError("The gate configuration must be an object.")
        if config.get("format_version") != GATE_FORMAT_VERSION:
            raise ValueError("Unsupported conditional gate format.")
        if set(tensors) != {"weight", "center"}:
            raise ValueError("The gate tensors must be exactly weight and center.")
        try:
            gate = cls(
                layer=config["layer"],
                weight=tensors["weight"].float(),
                bias=config["bias"],
                threshold=config["threshold"],
                center=tensors["center"].float(),
                scale=config["scale"],
                target=config["target"],
                mode=config.get("mode", "hard"),
                temperature=config.get("temperature", 1.0),
                metadata=config.get("metadata", {}),
            )
        except KeyError as error:
            raise ValueError(f"The gate configuration is missing {error}.") from error
        if config.get("hidden_size", gate.hidden_size) != gate.hidden_size:
            raise ValueError("The gate configuration width does not match its tensors.")
        return gate


@dataclass
class GateTrace:
    """Request-local gate state. Create one per generation; never share it."""

    score: float | None = None
    value: float | None = None
    image_tokens: int = 0

    @property
    def computed(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "value": self.value,
            "image_tokens": self.image_tokens,
        }


def gate_value(
    score: float | None, threshold: float, mode: str = "hard", temperature: float = 1.0
) -> float:
    """Map a probe score to [0, 1]. A missing score (no image tokens) closes the gate."""
    if score is None:
        return 0.0
    if not math.isfinite(score):
        raise ValueError("The gate score is not finite.")
    if mode == "hard":
        return 1.0 if score >= threshold else 0.0
    if mode == "soft":
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("The soft-gate temperature must be finite and positive.")
        return float(torch.sigmoid(torch.tensor((score - threshold) / temperature)))
    raise ValueError(f"The gate mode must be one of {GATE_MODES}.")


def pool_visual_tokens(
    hidden: torch.Tensor, image_mask: torch.Tensor
) -> torch.Tensor | None:
    """Mean of image-token states only. Text-token states never contribute.

    hidden: [seq, d] or [1, seq, d]; image_mask: [seq] bool. Returns CPU float32 [d],
    or None when the prompt has no image tokens.
    """
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError("Conditional gating supports batch size 1 only.")
        hidden = hidden[0]
    if hidden.ndim != 2:
        raise ValueError("Hidden states must have shape [seq, hidden].")
    mask = image_mask.reshape(-1).bool()
    if mask.numel() != hidden.shape[0]:
        raise ValueError(
            "The image mask length does not match the hidden sequence; the gate "
            "must be computed from the full prefill."
        )
    if not bool(mask.any()):
        return None
    pooled = hidden[mask.to(hidden.device)].float().mean(dim=0)
    pooled = pooled.detach().cpu()
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("Pooled visual state contains NaN or infinite values.")
    return pooled


def observe_gate(
    gate: LinearGate,
    hidden: torch.Tensor,
    image_mask: torch.Tensor,
    trace: GateTrace,
) -> GateTrace:
    """Fill a request-local trace from the unmodified prefill output of the gate layer."""
    pooled = pool_visual_tokens(hidden, image_mask)
    trace.image_tokens = int(image_mask.bool().sum())
    trace.score = None if pooled is None else gate.score(pooled)
    trace.value = gate.value(trace.score)
    return trace


def validate_gate_layer_order(gate_layer: int, strengths: dict[int, float]) -> None:
    """Steering may only happen at or after the gate layer, never before it."""
    early = sorted(
        int(layer)
        for layer, alpha in strengths.items()
        if float(alpha) != 0.0 and int(layer) < int(gate_layer)
    )
    if early:
        raise ValueError(
            f"Steering at layer(s) {early} would run before the gate layer "
            f"{gate_layer} is computed. Set those strengths to zero or choose an "
            "earlier gate layer."
        )


def apply_gated_delta(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    gate: float,
    last_token_only: bool,
) -> torch.Tensor | None:
    """Return hidden + alpha * gate * direction, or None when the intervention is zero."""
    effective = float(alpha) * float(gate)
    if effective == 0.0:
        return None
    delta = direction.to(device=hidden.device, dtype=hidden.dtype) * effective
    if last_token_only:
        steered = hidden.clone()
        steered[:, -1, :] = steered[:, -1, :] + delta
        return steered
    return hidden + delta.view(1, 1, -1)


# ----------------------------------------------------------------------------
# Calibration: logistic-regression probe, threshold choice and metrics.
# ----------------------------------------------------------------------------


def _check_xy(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or not features.shape[0]:
        raise ValueError("Features must have nonempty shape [examples, hidden].")
    labels = labels.reshape(-1)
    if labels.numel() != features.shape[0]:
        raise ValueError("Features and labels have different lengths.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Gate features contain NaN or infinite values.")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Gate labels must be 0 (non-target) or 1 (target).")
    if not bool((labels == 1).any()) or not bool((labels == 0).any()):
        raise ValueError("Gate calibration needs both target and non-target examples.")
    return features.detach().double().cpu(), labels.detach().double().cpu()


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    l2: float = 1e-2,
    max_iter: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Class-balanced L2 logistic regression, full batch, float64 LBFGS from zero.

    Deterministic: zero initialization, CPU float64 and no data shuffling.
    Returns weight [d], bias, center [d], scale and training diagnostics.
    """
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("The L2 penalty must be finite and nonnegative.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be positive.")
    x, y = _check_xy(features, labels)
    torch.manual_seed(int(seed))
    center = x.mean(dim=0)
    centered = x - center
    scale = float(centered.norm(dim=1).mean()) / math.sqrt(x.shape[1])
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    xn = centered / scale
    positives = float(y.sum())
    negatives = float(y.numel() - positives)
    sample_weight = torch.where(
        y == 1, y.numel() / (2 * positives), y.numel() / (2 * negatives)
    )
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def objective():
        optimizer.zero_grad()
        logits = xn @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, weight=sample_weight
        ) + 0.5 * l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(objective)
        final_loss = float(objective())
    weight = weight.detach()
    bias_value = float(bias.detach())
    if not bool(torch.isfinite(weight).all()) or not math.isfinite(bias_value):
        raise ValueError("Probe training produced non-finite parameters.")
    train_scores = (xn @ weight + bias_value).float()
    return {
        "weight": weight.float(),
        "bias": bias_value,
        "center": center.float(),
        "scale": float(scale),
        "train_scores": train_scores,
        "train_loss": final_loss,
        "l2": float(l2),
        "max_iter": int(max_iter),
        "seed": int(seed),
    }


def classification_metrics(
    scores: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, Any]:
    scores = scores.detach().double().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1).bool()
    if scores.numel() != labels.numel():
        raise ValueError("Scores and labels have different lengths.")
    predicted = scores >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    tn = int((~predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    pos, neg = tp + fn, fp + tn
    return {
        "threshold": float(threshold),
        "positives": pos,
        "negatives": neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / pos if pos else None,
        "fpr": fp / neg if neg else None,
        "tnr": tn / neg if neg else None,
    }


def choose_threshold(
    scores: torch.Tensor, labels: torch.Tensor, max_false_positive_rate: float = 0.0
) -> tuple[float, dict[str, Any]]:
    """Pick the threshold with the best TPR subject to FPR <= the budget.

    Candidates are midpoints between consecutive distinct validation scores (plus one
    point beyond each end), so the chosen threshold keeps a margin on both sides.
    A budget of 0 means no validation non-target may open the gate.
    """
    if not math.isfinite(max_false_positive_rate) or not 0 <= max_false_positive_rate < 1:
        raise ValueError("The false-positive budget must be in [0, 1).")
    scores = scores.detach().double().cpu().reshape(-1)
    labels = labels.detach().cpu().reshape(-1)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Validation scores contain NaN or infinite values.")
    if not bool((labels == 1).any()) or not bool((labels == 0).any()):
        raise ValueError("Threshold selection needs target and non-target validation examples.")
    distinct = torch.unique(scores)  # sorted
    candidates = [float(distinct[0]) - 1.0]
    candidates += [float(v) for v in (distinct[:-1] + distinct[1:]) / 2]
    candidates.append(float(distinct[-1]) + 1.0)
    best = None
    for threshold in candidates:
        metrics = classification_metrics(scores, labels, threshold)
        if metrics["fpr"] > max_false_positive_rate + 1e-12:
            continue
        key = (metrics["tpr"], -metrics["fpr"])
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    # The candidate beyond the largest score always has FPR 0, so best exists.
    assert best is not None
    return best[1], best[2]


def fit_gate(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    layer: int,
    target: str,
    mode: str = "hard",
    temperature: float = 1.0,
    max_false_positive_rate: float = 0.0,
    l2: float = 1e-2,
    max_iter: int = 500,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> LinearGate:
    """Fit the probe on training examples; choose the threshold on validation only."""
    probe = fit_linear_probe(train_features, train_labels, l2, max_iter, seed)
    _check_xy(validation_features, validation_labels)
    if validation_features.shape[1] != train_features.shape[1]:
        raise ValueError("Training and validation feature widths differ.")
    provisional = LinearGate(
        layer=int(layer),
        weight=probe["weight"],
        bias=probe["bias"],
        threshold=0.0,
        center=probe["center"],
        scale=probe["scale"],
        target=target,
        mode=mode,
        temperature=float(temperature),
    )
    validation_scores = torch.tensor(
        [provisional.score(row) for row in validation_features.float()]
    )
    threshold, validation_metrics = choose_threshold(
        validation_scores, validation_labels, max_false_positive_rate
    )
    train_metrics = classification_metrics(probe["train_scores"], train_labels, threshold)
    return LinearGate(
        layer=int(layer),
        weight=probe["weight"],
        bias=probe["bias"],
        threshold=float(threshold),
        center=probe["center"],
        scale=probe["scale"],
        target=target,
        mode=mode,
        temperature=float(temperature),
        metadata={
            **(metadata or {}),
            "probe": "class_balanced_l2_logistic_regression",
            "l2": probe["l2"],
            "max_iter": probe["max_iter"],
            "seed": probe["seed"],
            "train_loss": probe["train_loss"],
            "max_false_positive_rate": float(max_false_positive_rate),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        },
    )


# ----------------------------------------------------------------------------
# Leakage evaluation.
# ----------------------------------------------------------------------------


def leakage_metrics(
    deltas: torch.Tensor | list[float],
    is_target: torch.Tensor | list[bool],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Separate on-target effect from off-target leakage.

    E_target = mean(delta | target), E_leak = mean(|delta| | non-target),
    selectivity = E_target / (E_leak + eps). delta is any per-example change in a
    damage score caused by steering (steered - base).
    """
    deltas = torch.as_tensor(deltas, dtype=torch.float64).reshape(-1)
    is_target = torch.as_tensor(is_target).reshape(-1).bool()
    if deltas.numel() != is_target.numel():
        raise ValueError("Deltas and target labels have different lengths.")
    if not bool(torch.isfinite(deltas).all()):
        raise ValueError("Steering deltas contain NaN or infinite values.")
    target = deltas[is_target]
    other = deltas[~is_target]
    e_target = float(target.mean()) if target.numel() else None
    e_leak = float(other.abs().mean()) if other.numel() else None
    selectivity = (
        e_target / (e_leak + eps) if e_target is not None and e_leak is not None else None
    )
    return {
        "target_examples": int(target.numel()),
        "non_target_examples": int(other.numel()),
        "E_target": e_target,
        "E_leak": e_leak,
        "max_abs_leak": float(other.abs().max()) if other.numel() else None,
        "selectivity": selectivity,
        "eps": eps,
    }
