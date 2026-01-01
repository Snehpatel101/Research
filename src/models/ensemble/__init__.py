"""
Ensemble model implementations.

Meta-learner models that combine predictions from multiple
base models for improved performance.

Ensemble Models:
- VotingEnsemble: Soft or hard voting
- StackingEnsemble: Stacked generalization with meta-learner (OOF predictions)
- BlendingEnsemble: Blending with holdout set

Meta-Learners (for stacking/blending):
- RidgeMetaLearner: Ridge regression for linear combination of predictions
- MLPMetaLearner: Multi-layer perceptron for non-linear combinations
- CalibratedMetaLearner: Calibration wrapper using Isotonic/Platt scaling
- XGBoostMeta: XGBoost gradient boosting as meta-learner

Validators:
- validate_ensemble_config: Check if base models are compatible
- validate_base_model_compatibility: Raise error if models are incompatible
- get_compatible_models: Get list of models compatible with a reference model
- is_heterogeneous_ensemble: Check if config has mixed tabular + sequence models
- classify_base_models: Separate models into tabular and sequence categories

All models auto-register with ModelRegistry on import.

ENSEMBLE COMPATIBILITY:
- Tabular models (2D): xgboost, lightgbm, catboost, random_forest, logistic, svm
- Sequence models (3D): lstm, gru, tcn, transformer

HOMOGENEOUS ENSEMBLES (Voting/Blending):
- Require all base models to have the same input shape
- Mixed models will raise EnsembleCompatibilityError

HETEROGENEOUS ENSEMBLES (Stacking only):
- Stacking supports mixed tabular + sequence models
- The meta-learner receives 2D OOF predictions regardless of base model type
- This enables combining XGBoost with LSTM, CatBoost with TCN, etc.

Example:
    # Create via registry
    from src.models import ModelRegistry

    # Voting ensemble (tabular models only - homogeneous)
    voting = ModelRegistry.create("voting", config={
        "voting": "soft",
        "base_model_names": ["xgboost", "lightgbm", "catboost"],
    })

    # Stacking ensemble (sequence models only - homogeneous)
    stacking = ModelRegistry.create("stacking", config={
        "base_model_names": ["lstm", "gru", "tcn"],
        "meta_learner_name": "logistic",
        "n_folds": 5,
    })

    # HETEROGENEOUS stacking (mixed tabular + sequence)
    # The meta-learner receives 2D OOF predictions from all models
    heterogeneous_stacking = ModelRegistry.create("stacking", config={
        "base_model_names": ["xgboost", "lstm", "catboost", "tcn"],
        "meta_learner_name": "logistic",
        "n_folds": 5,
    })
    # For heterogeneous training, provide both 2D and 3D data:
    # metrics = heterogeneous_stacking.fit(
    #     X_train, y_train, X_val, y_val,
    #     X_train_seq=X_train_3d, X_val_seq=X_val_3d
    # )

    # Blending ensemble (tabular models only - homogeneous)
    blending = ModelRegistry.create("blending", config={
        "base_model_names": ["xgboost", "random_forest"],
        "meta_learner_name": "logistic",
        "holdout_fraction": 0.2,
    })

    # Use specialized meta-learner for stacking
    stacking_xgb = ModelRegistry.create("stacking", config={
        "base_model_names": ["xgboost", "lightgbm", "catboost"],
        "meta_learner_name": "xgboost_meta",  # XGBoost as meta-learner
        "n_folds": 5,
    })
"""
from .blending import BlendingEnsemble
from .meta_learners import (
    CalibratedMetaLearner,
    MLPMetaLearner,
    RidgeMetaLearner,
    XGBoostMeta,
)
from .stacking import StackingEnsemble
from .validator import (
    HETEROGENEOUS_ENSEMBLE_TYPES,
    HOMOGENEOUS_ENSEMBLE_TYPES,
    EnsembleCompatibilityError,
    classify_base_models,
    get_compatible_models,
    is_heterogeneous_ensemble,
    validate_base_model_compatibility,
    validate_ensemble_config,
)
from .voting import VotingEnsemble

__all__ = [
    # Ensemble models
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    # Meta-learners
    "RidgeMetaLearner",
    "MLPMetaLearner",
    "CalibratedMetaLearner",
    "XGBoostMeta",
    # Validators
    "validate_ensemble_config",
    "validate_base_model_compatibility",
    "get_compatible_models",
    "is_heterogeneous_ensemble",
    "classify_base_models",
    "EnsembleCompatibilityError",
    "HETEROGENEOUS_ENSEMBLE_TYPES",
    "HOMOGENEOUS_ENSEMBLE_TYPES",
]
