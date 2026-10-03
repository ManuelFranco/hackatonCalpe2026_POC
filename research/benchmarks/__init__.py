"""Public benchmark interface. Add adapters without editing dashboard callbacks."""

from .adapters import (
    BenchmarkAdapter,
    IFEval,
    MBPP,
    MMLUPro,
    MMLUProStratifiedEasy,
    MMMU,
    POPE,
)
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
    "mmlu_pro_stratified_easy": MMLUProStratifiedEasy(),
    "mmmu": MMMU(),
    "pope": POPE(),
    "ifeval": IFEval(),
    "mbpp": MBPP(),
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
