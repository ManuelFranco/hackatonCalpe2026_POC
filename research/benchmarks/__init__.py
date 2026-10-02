"""Public benchmark interface. Add adapters without editing dashboard callbacks."""

from .adapters import BenchmarkAdapter, IFEval, MMLUPro, MMMU
from .runner import BenchmarkRequest, run_benchmark

BENCHMARKS = {"mmlu_pro": MMLUPro(), "mmmu": MMMU(), "ifeval": IFEval()}
__all__ = ["BENCHMARKS", "BenchmarkAdapter", "BenchmarkRequest", "run_benchmark"]
