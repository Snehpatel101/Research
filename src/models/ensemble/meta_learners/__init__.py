"""
Meta-learner models for stacking ensembles.

REORG-003: Meta-learners have been moved to src/models/ensemble/ (flattened).
This module re-exports from parent for backward compatibility.
New code should import directly from src.models.ensemble.

Available Meta-Learners:
- RidgeMetaLearner: Ridge regression for linear combination of predictions
- MLPMetaLearner: Multi-layer perceptron for non-linear combinations
- CalibratedMetaLearner: Calibration wrapper using Isotonic or Platt scaling
- XGBoostMeta: XGBoost as meta-learner for gradient boosted stacking
"""

# Re-export from parent directory (backward compatibility)
from ..calibrated_meta import CalibratedMetaLearner
from ..mlp_meta import MLPMetaLearner
from ..ridge_meta import RidgeMetaLearner
from ..xgboost_meta import XGBoostMeta

__all__ = [
    "RidgeMetaLearner",
    "MLPMetaLearner",
    "CalibratedMetaLearner",
    "XGBoostMeta",
]
