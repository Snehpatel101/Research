"""
Ensemble model implementations.

Meta-learner models that combine predictions from multiple
base models for improved performance.

Models:
- VotingEnsemble: Soft or hard voting
- StackingEnsemble: Stacked generalization with meta-learner (OOF predictions)
- BlendingEnsemble: Blending with holdout set

All models auto-register with ModelRegistry on import.

Example:
    # Create via registry
    from src.models import ModelRegistry

    # Voting ensemble
    voting = ModelRegistry.create("voting", config={
        "voting": "soft",
        "base_model_names": ["xgboost", "lightgbm"],
    })

    # Stacking ensemble
    stacking = ModelRegistry.create("stacking", config={
        "base_model_names": ["xgboost", "lightgbm", "random_forest"],
        "meta_learner_name": "logistic",
        "n_folds": 5,
    })

    # Blending ensemble
    blending = ModelRegistry.create("blending", config={
        "base_model_names": ["xgboost", "lightgbm"],
        "meta_learner_name": "logistic",
        "holdout_fraction": 0.2,
    })
"""
from .voting import VotingEnsemble
from .stacking import StackingEnsemble
from .blending import BlendingEnsemble

__all__ = [
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
]
