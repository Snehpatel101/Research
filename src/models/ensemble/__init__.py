"""
Ensemble model implementations.

Meta-learner models that combine predictions from multiple
base models for improved performance.

Models:
- VotingEnsemble: Soft or hard voting
- StackingEnsemble: Stacked generalization with meta-learner (OOF predictions)
- BlendingEnsemble: Blending with holdout set

Validators:
- validate_ensemble_config: Check if base models are compatible
- validate_base_model_compatibility: Raise error if models are incompatible
- get_compatible_models: Get list of models compatible with a reference model

All models auto-register with ModelRegistry on import.

IMPORTANT: Ensembles require all base models to have the same input shape:
- Tabular models (2D): xgboost, lightgbm, catboost, random_forest, logistic, svm
- Sequence models (3D): lstm, gru, tcn, transformer
- Mixed ensembles are NOT supported (will raise EnsembleCompatibilityError)

Example:
    # Create via registry
    from src.models import ModelRegistry

    # Voting ensemble (tabular models only)
    voting = ModelRegistry.create("voting", config={
        "voting": "soft",
        "base_model_names": ["xgboost", "lightgbm", "catboost"],
    })

    # Stacking ensemble (sequence models only)
    stacking = ModelRegistry.create("stacking", config={
        "base_model_names": ["lstm", "gru", "tcn"],
        "meta_learner_name": "logistic",
        "n_folds": 5,
    })

    # Blending ensemble (tabular models only)
    blending = ModelRegistry.create("blending", config={
        "base_model_names": ["xgboost", "random_forest"],
        "meta_learner_name": "logistic",
        "holdout_fraction": 0.2,
    })
"""
from .voting import VotingEnsemble
from .stacking import StackingEnsemble
from .blending import BlendingEnsemble
from .validator import (
    validate_ensemble_config,
    validate_base_model_compatibility,
    get_compatible_models,
    EnsembleCompatibilityError,
)

__all__ = [
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "validate_ensemble_config",
    "validate_base_model_compatibility",
    "get_compatible_models",
    "EnsembleCompatibilityError",
]
