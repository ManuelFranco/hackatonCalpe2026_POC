"""Small additive interventions at the state predicting a continuation token."""

import math

import torch


class ResidualIntervention:
    """A unit direction, a per-step norm budget and a zero-based token window.

    Never reconstructs the residual from the SAE. Each experiment creates a new
    instance. The runtime owns hook installation, tracing, RNG and cleanup.
    """

    def __init__(self, layer, direction, fraction=0.02, steps=None):
        direction = direction.detach().float().cpu().clone()
        if (
            direction.ndim != 1
            or not bool(torch.isfinite(direction).all())
            or not bool(torch.isfinite(direction.norm()))
            or float(direction.norm()) <= 1e-12
        ):
            raise ValueError("Intervention direction must be finite and nonzero.")
        if not math.isfinite(fraction) or not 0 < fraction <= 0.05:
            raise ValueError("The residual budget must be between 0 and 5%.")
        if steps is not None and (not isinstance(steps, int) or not 1 <= steps <= 64):
            raise ValueError("Choose a window of 1–64 continuation tokens.")
        self.layers = (layer,)
        self.direction = direction / direction.norm()
        self.fraction, self.steps = float(fraction), steps
        self.cache = {}
        self.measurements = []

    def apply(self, layer, hidden, step):
        if layer not in self.layers or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(
                "Interventions require one sequence at the selected layer."
            )
        if self.steps is not None and step >= self.steps:
            return hidden
        if hidden.shape[-1] != len(self.direction):
            raise ValueError("Decoder direction and residual dimensions differ.")
        before = hidden[0, -1].float()
        norm = before.norm()
        if not bool(torch.isfinite(norm)):
            raise ValueError("Non-finite residual during intervention.")
        key = (str(hidden.device), str(hidden.dtype))
        if key not in self.cache:
            self.cache[key] = self.direction.to(hidden.device)
        intended = self.cache[key] * (norm * self.fraction)
        changed = hidden.clone()
        changed[:, -1, :] += intended.to(hidden.dtype)
        delta = changed[0, -1].float() - before
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Non-finite residual change during intervention.")
        # Lower-precision rounding can exceed the requested budget. Reduce the
        # addition; if it cannot be represented within budget, leave it unchanged.
        target = norm * self.fraction
        for _ in range(3):
            actual = delta.norm()
            if bool(actual <= target * 1.00001):
                break
            intended = intended * (target / actual.clamp_min(1e-12)) * 0.99
            changed = hidden.clone()
            changed[:, -1, :] += intended.to(hidden.dtype)
            delta = changed[0, -1].float() - before
        if bool(delta.norm() > target * 1.00001):
            changed, delta = hidden, torch.zeros_like(before)
        self.measurements.append(
            {
                "token": step + 1,
                "residual_norm": float(norm),
                "delta_norm": float(delta.norm()),
                "relative_change": float(delta.norm() / norm.clamp_min(1e-12)),
            }
        )
        return changed

    def metadata(self):
        return {
            "requested_fraction": self.fraction,
            "window_tokens": self.steps,
            "changed_tokens": sum(m["delta_norm"] > 0 for m in self.measurements),
            "maximum_relative_change": max(
                (m["relative_change"] for m in self.measurements), default=0.0
            ),
            "steps": self.measurements,
        }


def feature_direction(decoder, rows, toward_b=True):
    """Equal unit contributions; normalize the combined direction, not each budget."""
    directions = []
    for row in rows:
        direction = decoder[row["feature"]].detach().float().cpu()
        if (
            not bool(torch.isfinite(direction).all())
            or float(direction.norm()) <= 1e-12
        ):
            raise ValueError(f"Feature #{row['feature']} has an unusable decoder.")
        sign = math.copysign(1.0, row["contrast"]) * (1 if toward_b else -1)
        directions.append(sign * direction / direction.norm())
    return torch.stack(directions).sum(0)


def random_direction(size, seed):
    return torch.randn(size, generator=torch.Generator().manual_seed(seed))
