"""Stable tensor contract between capture, research, and inference."""

from dataclasses import dataclass, field
from typing import Any, Protocol
import torch


@dataclass(frozen=True)
class LayerProfile:
    # Rows are matching A/B pairs; columns are SAE features. CPU float32.
    a: torch.Tensor
    b: torch.Tensor
    reference_norm: float


@dataclass(frozen=True)
class VectorResult:
    feature_delta: torch.Tensor
    direction: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorBuilder(Protocol):
    name: str

    def build(
        self, profile: LayerProfile, decoder: torch.Tensor, fraction: float
    ) -> VectorResult:
        """Return a feature delta and a residual direction without mutating inputs.

        Future filters, weighting, and robust estimators belong behind this
        interface. Methods must be deterministic and must not persist artifacts.
        """
        ...
