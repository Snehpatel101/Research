"""
Evaluation Pipeline Stage - Post-training model evaluation.

This stage runs after model training to generate comprehensive evaluation
reports including:
- Performance metrics (accuracy, precision, recall, F1)
- Financial metrics (Sharpe, returns, drawdown)
- Confusion matrices and feature importance charts
- Trade analysis and rolling performance

Delegates to existing evaluation utilities in src.models.evaluation.
"""

from .run import EvaluationStageResult, run_evaluation

__all__ = [
    "run_evaluation",
    "EvaluationStageResult",
]
