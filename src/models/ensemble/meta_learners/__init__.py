"""
Meta-learner models for stacking ensembles.

This module provides specialized meta-learner implementations designed to
combine out-of-fold (OOF) predictions from base models. These models accept
input of shape (n_samples, n_base_models * n_classes) and produce final
3-class predictions.

Available Meta-Learners:
- RidgeMetaLearner: Ridge regression for linear combination of predictions
- MLPMetaLearner: Multi-layer perceptron for non-linear combinations
- CalibratedMetaLearner: Calibration wrapper using Isotonic or Platt scaling
- XGBoostMeta: XGBoost as meta-learner for gradient boosted stacking

All meta-learners implement the BaseModel interface and support:
- Sample weights for imbalanced data
- Training on stacking datasets from OOF predictions
- Returning PredictionOutput with probabilities and confidence
"""

from .calibrated_meta import CalibratedMetaLearner
from .mlp_meta import MLPMetaLearner
from .ridge_meta import RidgeMetaLearner
from .xgboost_meta import XGBoostMeta

__all__ = [
    "RidgeMetaLearner",
    "MLPMetaLearner",
    "CalibratedMetaLearner",
    "XGBoostMeta",
]
