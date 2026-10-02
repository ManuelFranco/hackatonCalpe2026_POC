"""Current baseline: mean(B - A), retaining every feature."""

import math
import torch
from .base import LayerProfile, VectorResult


class MeanDifference:
    name = "Mean B − A (all features)"

    def build(
        self, profile: LayerProfile, decoder: torch.Tensor, fraction: float
    ) -> VectorResult:
        a, b = profile.a, profile.b
        if a.ndim != 2 or a.shape != b.shape or not a.shape[0]:
            raise ValueError(
                "A and B must have matching, nonempty [pairs, features] shapes."
            )
        if decoder.ndim != 2 or decoder.shape[0] != a.shape[1]:
            raise ValueError("The decoder must have shape [features, residual width].")
        if not all(bool(torch.isfinite(t).all()) for t in (a, b, decoder)):
            raise ValueError("Profile and decoder values must be finite.")
        if not math.isfinite(profile.reference_norm) or profile.reference_norm < 0:
            raise ValueError("The reference norm must be finite and non-negative.")
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("The steering fraction must be finite and positive.")
        delta = (b.float().cpu() - a.float().cpu()).mean(dim=0)
        raw = delta @ decoder.detach().float().cpu()
        norm = float(raw.norm())
        # Avoid promoting floating-point cancellation to a full-strength vector.
        direction = (
            raw * (profile.reference_norm * fraction / norm)
            if norm > 1e-12
            else torch.zeros_like(raw)
        )
        if not bool(torch.isfinite(direction).all()):
            raise ValueError("The resulting direction is not finite.")
        return VectorResult(
            delta,
            direction,
            {
                "method": "mean_difference",
                "formula": "mean(B - A) @ W_dec",
                "feature_count": delta.numel(),
                "nonzero_features": int(torch.count_nonzero(delta)),
                "raw_norm": norm,
                "reference_norm": profile.reference_norm,
                "direction_norm": float(direction.norm()),
                "fraction": fraction,
            },
        )
