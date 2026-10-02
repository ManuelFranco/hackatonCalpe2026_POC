"""Gram-Schmidt difference: retain only novel orthogonal feature components."""

import math
import torch
from .base import LayerProfile, VectorResult


class GramSchmidtDifference:
    name = "Gram-Schmidt B − A"

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

        # One feature-space difference vector per pair.
        differences = b.float().cpu() - a.float().cpu()

        basis = []
        components = []

        eps = 1e-12

        for v in differences:
            # Remove everything already represented by previous directions.
            orthogonal = v.clone()

            for q in basis:
                orthogonal = orthogonal - torch.dot(orthogonal, q) * q

            orthogonal_norm = orthogonal.norm()

            if orthogonal_norm > eps:
                # q is only used as an orthonormal basis for later projections.
                q = orthogonal / orthogonal_norm
                basis.append(q)

                # Keep the original magnitude of the novel component.
                components.append(orthogonal)

        if components:
            delta = torch.stack(components).mean(dim=0)
        else:
            delta = torch.zeros(
                a.shape[1],
                dtype=torch.float32,
            )

        raw = delta @ decoder.detach().float().cpu()
        norm = float(raw.norm())

        direction = (
            raw * (profile.reference_norm * fraction / norm)
            if norm > eps
            else torch.zeros_like(raw)
        )

        if not bool(torch.isfinite(direction).all()):
            raise ValueError("The resulting direction is not finite.")

        return VectorResult(
            delta,
            direction,
            {
                "method": "gram_schmidt_difference",
                "formula": "mean(GS(B_i - A_i)) @ W_dec",
                "pair_count": differences.shape[0],
                "feature_count": delta.numel(),
                "orthogonal_components": len(components),
                "discarded_components": differences.shape[0] - len(components),
                "nonzero_features": int(torch.count_nonzero(delta)),
                "raw_norm": norm,
                "reference_norm": profile.reference_norm,
                "direction_norm": float(direction.norm()),
                "fraction": fraction,
            },
        )