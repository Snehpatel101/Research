"""
Evaluation methods package.

Import paths:
    # New (preferred):
    from src.validation.evaluation import CVEvaluator

    # Legacy (still works, deprecation warning):
    from src.evaluation import CVEvaluator

Provides model evaluation strategies:
- cv: Cross-validation with PurgedKFold
- walk_forward: Walk-forward validation
- cpcv_pbo: CPCV-PBO (Combinatorially Purged Cross-Validation with Probability of Backtest Overfitting)
"""

from .cv_evaluator import CVEvaluator
from .walk_forward_evaluator import WalkForwardEvaluator
from .cpcv_pbo_evaluator import CPCVPBOEvaluator

__all__ = [
    "CVEvaluator",
    "WalkForwardEvaluator",
    "CPCVPBOEvaluator",
]
