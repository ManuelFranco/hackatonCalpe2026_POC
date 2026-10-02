"""Replace or register vector strategies here; the dashboard discovers the registry."""

from .base import LayerProfile, VectorBuilder, VectorResult
from .difference import MeanDifference
from .gs import GramSchmidtDifference
from .bOnly import BOnlyFeatures


BUILDERS: dict[str, VectorBuilder] = {"mean_difference": MeanDifference(), 
                                      "gramSchmidt": GramSchmidtDifference(), 
                                      "bOnly": BOnlyFeatures()}

__all__ = ["BUILDERS", "LayerProfile", "VectorBuilder", "VectorResult"]
