"""
Evaluation methods package.

Provides the CPCV-PBO evaluator (Combinatorially Purged Cross-Validation with
Probability of Backtest Overfitting):

    from src.validation.evaluation import CPCVPBOEvaluator
"""

from .cpcv_pbo_evaluator import CPCVPBOEvaluator

__all__ = [
    "CPCVPBOEvaluator",
]
