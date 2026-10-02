"""Public benchmark interface. Add adapters without editing dashboard callbacks."""

from .adapters import BenchmarkAdapter, IFEval, MMLUPro, MMMU, POPE
from .runner import (
    BenchmarkRequest,
    run_benchmark,
    prepare_benchmark,
    evaluate_base,
    evaluate_steered,
    summarize,
)

BENCHMARKS = {
    "mmlu_pro": MMLUPro(),
    "mmmu": MMMU(),
    "pope": POPE(),
    "ifeval": IFEval(),
}
__all__ = [
    "BENCHMARKS",
    "BenchmarkAdapter",
    "BenchmarkRequest",
    "run_benchmark",
    "prepare_benchmark",
    "evaluate_base",
    "evaluate_steered",
    "summarize",
]
