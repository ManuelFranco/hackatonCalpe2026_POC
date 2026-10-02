"""Replace or register vector strategies here; the dashboard discovers the registry."""

from .base import LayerProfile, VectorBuilder, VectorResult
from .difference import MeanDifference

BUILDERS: dict[str, VectorBuilder] = {"mean_difference": MeanDifference()}
__all__ = ["BUILDERS", "LayerProfile", "VectorBuilder", "VectorResult"]
