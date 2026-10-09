"""Phase 4 评估包。"""

from .eval_pipeline import build_evaluation_rows, load_golden_dataset, run_evaluation, run_rag_pipeline

__all__ = [
    "build_evaluation_rows",
    "load_golden_dataset",
    "run_evaluation",
    "run_rag_pipeline",
]
