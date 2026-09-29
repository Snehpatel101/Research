"""
Optimization Package.

Subpackages and modules:
    feature_selection: MDA/walk-forward feature selection, filtering, and
        feature-governance utilities used by the training pipeline.
    scoring: ``get_score_fn()`` metric dispatcher used by the live Optuna
        tuner (``src.validation.cv.TimeSeriesOptunaTuner``).

Hyperparameter search spaces for the live tuner live in
``src.validation.cv.param_spaces``.
"""

from src.optimization.scoring import get_score_fn

__all__ = [
    "get_score_fn",
]
