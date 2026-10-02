"""B-only features: keep features active in B and exactly zero in A."""

import math
import torch
from .base import LayerProfile, VectorResult


class BOnlyFeatures:
    name = "B-only features"

    def build(
        self, profile: LayerProfile, decoder: torch.Tensor, fraction: float
    ) -> VectorResult:
        a, b = profile.a, profile.b

        if a.ndim != 2 or a.shape != b.shape or not a.shape[0]:
            raise ValueError(
                "A and B must have matching, nonempty [pairs, features] shapes."
            )

        if decoder.ndim != 2 or decoder.shape[0] != a.shape[1]:
            raise ValueError(
                "The decoder must have shape [features, residual width]."
            )

        if not all(bool(torch.isfinite(t).all()) for t in (a, b, decoder)):
            raise ValueError("Profile and decoder values must be finite.")

        if not math.isfinite(profile.reference_norm) or profile.reference_norm < 0:
            raise ValueError(
                "The reference norm must be finite and non-negative."
            )

        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError(
                "The steering fraction must be finite and positive."
            )

        a = a.float().cpu()
        b = b.float().cpu()

        # Feature qualifies iff:
        #   A == 0
        #   B != 0
        #
        # Activation magnitude is completely ignored.
        b_only = (a == 0) & (b != 0)

        # Fraction of pairs for which each feature is B-only.
        #
        # Example:
        # feature 42 qualifies in 8/10 pairs -> delta[42] = 0.8
        delta = b_only.float().mean(dim=0)

        raw = delta @ decoder.detach().float().cpu()
        norm = float(raw.norm())

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
                "method": "b_only_features",
                "formula": "mean(1[A == 0 and B != 0]) @ W_dec",
                "pair_count": a.shape[0],
                "feature_count": delta.numel(),
                "selected_features": int(torch.count_nonzero(delta)),
                "qualifying_pair_features": int(b_only.sum()),
                "raw_norm": norm,
                "reference_norm": profile.reference_norm,
                "direction_norm": float(direction.norm()),
                "fraction": fraction,
            },
        )