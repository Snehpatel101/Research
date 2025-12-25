"""
Model-aware configuration for Phase 1 data preparation.

This module defines data preparation requirements for each model type,
enabling Phase 1 to prepare appropriate datasets for Phase 2 training.

The model factory architecture requires Phase 1 to produce standardized
datasets that satisfy the requirements of all target model types.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class ModelFamily(str, Enum):
    """Model family classification."""
    BOOSTING = "boosting"
    NEURAL = "neural"
    TRANSFORMER = "transformer"
    CLASSICAL = "classical"
    ENSEMBLE = "ensemble"


class ScalerType(str, Enum):
    """Supported scaler types for data normalization."""
    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    QUANTILE = "quantile"


@dataclass(frozen=True)
class ModelDataRequirements:
    """
    Data preparation requirements for a specific model type.

    These requirements inform Phase 1 about what data format
    each model type in Phase 2 will expect.

    Attributes:
        model_name: Unique identifier for the model type
        family: Model family classification
        feature_set: Default feature set to use
        requires_scaling: Whether features need normalization
        scaler_type: Type of scaler if scaling required
        requires_sequences: Whether data should be formatted as sequences
        sequence_length: Length of sequences if required
        max_features: Suggested maximum feature count (for regularization)
        supports_categorical: Whether model handles categorical features natively
        supports_missing: Whether model handles missing values natively
        description: Human-readable description
    """
    model_name: str
    family: ModelFamily
    feature_set: str
    requires_scaling: bool = False
    scaler_type: ScalerType = ScalerType.NONE
    requires_sequences: bool = False
    sequence_length: int = 60
    max_features: Optional[int] = None
    supports_categorical: bool = False
    supports_missing: bool = False
    description: str = ""


# =============================================================================
# MODEL DATA REQUIREMENTS
# =============================================================================
# Define data preparation needs for each supported model type.
# This allows Phase 1 to prepare appropriate datasets for Phase 2 training.

MODEL_DATA_REQUIREMENTS: Dict[str, ModelDataRequirements] = {
    # -------------------------------------------------------------------------
    # BOOSTING MODELS (tree-based gradient boosting)
    # -------------------------------------------------------------------------
    "xgboost": ModelDataRequirements(
        model_name="xgboost",
        family=ModelFamily.BOOSTING,
        feature_set="boosting_optimal",
        requires_scaling=False,
        scaler_type=ScalerType.NONE,
        requires_sequences=False,
        max_features=100,
        supports_categorical=True,
        supports_missing=True,
        description="XGBoost gradient boosting. Handles raw features, missing values, and high correlation.",
    ),
    "lightgbm": ModelDataRequirements(
        model_name="lightgbm",
        family=ModelFamily.BOOSTING,
        feature_set="boosting_optimal",
        requires_scaling=False,
        scaler_type=ScalerType.NONE,
        requires_sequences=False,
        max_features=100,
        supports_categorical=True,
        supports_missing=True,
        description="LightGBM gradient boosting. Fast training with leaf-wise growth.",
    ),
    "catboost": ModelDataRequirements(
        model_name="catboost",
        family=ModelFamily.BOOSTING,
        feature_set="boosting_optimal",
        requires_scaling=False,
        scaler_type=ScalerType.NONE,
        requires_sequences=False,
        max_features=100,
        supports_categorical=True,
        supports_missing=True,
        description="CatBoost gradient boosting. Excellent categorical feature handling.",
    ),

    # -------------------------------------------------------------------------
    # NEURAL NETWORK MODELS (sequential/recurrent)
    # -------------------------------------------------------------------------
    "lstm": ModelDataRequirements(
        model_name="lstm",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="LSTM recurrent network. Captures long-term dependencies in sequences.",
    ),
    "gru": ModelDataRequirements(
        model_name="gru",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="GRU recurrent network. Simpler than LSTM, often similar performance.",
    ),
    "tcn": ModelDataRequirements(
        model_name="tcn",
        family=ModelFamily.NEURAL,
        feature_set="tcn_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=120,
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="Temporal Convolutional Network. Dilated convolutions for long-range dependencies.",
    ),
    "mlp": ModelDataRequirements(
        model_name="mlp",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=False,
        max_features=100,
        supports_categorical=False,
        supports_missing=False,
        description="Multi-Layer Perceptron. Simple feedforward network for tabular data.",
    ),

    # -------------------------------------------------------------------------
    # TRANSFORMER MODELS (attention-based)
    # -------------------------------------------------------------------------
    "transformer": ModelDataRequirements(
        model_name="transformer",
        family=ModelFamily.TRANSFORMER,
        feature_set="transformer_raw",
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=True,
        sequence_length=128,
        max_features=50,
        supports_categorical=False,
        supports_missing=False,
        description="Vanilla Transformer encoder. Self-attention for time series.",
    ),
    "patchtst": ModelDataRequirements(
        model_name="patchtst",
        family=ModelFamily.TRANSFORMER,
        feature_set="transformer_raw",
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=True,
        sequence_length=256,
        max_features=30,
        supports_categorical=False,
        supports_missing=False,
        description="PatchTST Transformer. Patches input sequences for efficiency.",
    ),
    "informer": ModelDataRequirements(
        model_name="informer",
        family=ModelFamily.TRANSFORMER,
        feature_set="transformer_raw",
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=True,
        sequence_length=192,
        max_features=40,
        supports_categorical=False,
        supports_missing=False,
        description="Informer model. ProbSparse attention for long sequences.",
    ),

    # -------------------------------------------------------------------------
    # CLASSICAL ML MODELS
    # -------------------------------------------------------------------------
    "random_forest": ModelDataRequirements(
        model_name="random_forest",
        family=ModelFamily.CLASSICAL,
        feature_set="boosting_optimal",
        requires_scaling=False,
        scaler_type=ScalerType.NONE,
        requires_sequences=False,
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="Random Forest ensemble. Robust baseline with feature importance.",
    ),
    "logistic": ModelDataRequirements(
        model_name="logistic",
        family=ModelFamily.CLASSICAL,
        feature_set="neural_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=False,
        max_features=50,
        supports_categorical=False,
        supports_missing=False,
        description="Logistic Regression. Simple interpretable baseline.",
    ),
    "svm": ModelDataRequirements(
        model_name="svm",
        family=ModelFamily.CLASSICAL,
        feature_set="neural_optimal",
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=False,
        max_features=50,
        supports_categorical=False,
        supports_missing=False,
        description="Support Vector Machine. Kernel-based classification.",
    ),
}


# =============================================================================
# ENSEMBLE CONFIGURATIONS
# =============================================================================
# Pre-defined ensemble combinations for common use cases.

@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for an ensemble of models."""
    name: str
    description: str
    base_models: List[str]
    meta_learner: str = "logistic"
    stacking_method: str = "soft"  # 'soft' (probabilities) or 'hard' (predictions)


ENSEMBLE_CONFIGS: Dict[str, EnsembleConfig] = {
    "boosting_ensemble": EnsembleConfig(
        name="boosting_ensemble",
        description="Ensemble of tree-based boosting models",
        base_models=["xgboost", "lightgbm", "catboost"],
        meta_learner="logistic",
        stacking_method="soft",
    ),
    "neural_ensemble": EnsembleConfig(
        name="neural_ensemble",
        description="Ensemble of recurrent neural networks",
        base_models=["lstm", "gru", "tcn"],
        meta_learner="logistic",
        stacking_method="soft",
    ),
    "transformer_ensemble": EnsembleConfig(
        name="transformer_ensemble",
        description="Ensemble of transformer-based models",
        base_models=["transformer", "patchtst", "informer"],
        meta_learner="logistic",
        stacking_method="soft",
    ),
    "hybrid_ensemble": EnsembleConfig(
        name="hybrid_ensemble",
        description="Hybrid ensemble mixing boosting and neural models",
        base_models=["xgboost", "lstm", "transformer"],
        meta_learner="xgboost",
        stacking_method="soft",
    ),
    "full_ensemble": EnsembleConfig(
        name="full_ensemble",
        description="Full ensemble using all model families",
        base_models=["xgboost", "lightgbm", "lstm", "gru", "transformer"],
        meta_learner="xgboost",
        stacking_method="soft",
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_requirements(model_name: str) -> ModelDataRequirements:
    """
    Get data requirements for a specific model.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., 'xgboost', 'lstm')

    Returns
    -------
    ModelDataRequirements
        Requirements for the specified model

    Raises
    ------
    ValueError
        If model_name is not recognized
    """
    model_name = model_name.lower().strip()
    if model_name not in MODEL_DATA_REQUIREMENTS:
        valid = sorted(MODEL_DATA_REQUIREMENTS.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Valid models: {valid}")
    return MODEL_DATA_REQUIREMENTS[model_name]


def get_ensemble_config(ensemble_name: str) -> EnsembleConfig:
    """
    Get configuration for a pre-defined ensemble.

    Parameters
    ----------
    ensemble_name : str
        Ensemble identifier (e.g., 'boosting_ensemble')

    Returns
    -------
    EnsembleConfig
        Configuration for the specified ensemble

    Raises
    ------
    ValueError
        If ensemble_name is not recognized
    """
    ensemble_name = ensemble_name.lower().strip()
    if ensemble_name not in ENSEMBLE_CONFIGS:
        valid = sorted(ENSEMBLE_CONFIGS.keys())
        raise ValueError(f"Unknown ensemble: '{ensemble_name}'. Valid ensembles: {valid}")
    return ENSEMBLE_CONFIGS[ensemble_name]


def get_models_by_family(family: ModelFamily) -> List[str]:
    """
    Get all model names belonging to a specific family.

    Parameters
    ----------
    family : ModelFamily
        Model family to filter by

    Returns
    -------
    List[str]
        List of model names in the specified family
    """
    return [
        name for name, req in MODEL_DATA_REQUIREMENTS.items()
        if req.family == family
    ]


def get_combined_requirements(model_names: List[str]) -> Dict:
    """
    Get combined data requirements for multiple models.

    When training multiple models, Phase 1 should prepare data that
    satisfies ALL model requirements (union of features, strictest scaling, etc.)

    Parameters
    ----------
    model_names : List[str]
        List of model names to combine requirements for

    Returns
    -------
    Dict
        Combined requirements with keys:
        - feature_sets: Set of all required feature sets
        - requires_scaling: True if any model requires scaling
        - scaler_types: Set of all required scaler types
        - requires_sequences: True if any model requires sequences
        - max_sequence_length: Maximum sequence length required
        - min_max_features: Minimum of all max_features constraints
    """
    if not model_names:
        raise ValueError("At least one model name required")

    requirements = [get_model_requirements(m) for m in model_names]

    return {
        "feature_sets": set(r.feature_set for r in requirements),
        "requires_scaling": any(r.requires_scaling for r in requirements),
        "scaler_types": set(r.scaler_type for r in requirements if r.requires_scaling),
        "requires_sequences": any(r.requires_sequences for r in requirements),
        "max_sequence_length": max(
            (r.sequence_length for r in requirements if r.requires_sequences),
            default=0
        ),
        "min_max_features": min(
            (r.max_features for r in requirements if r.max_features),
            default=None
        ),
    }


def validate_model_config(model_names: List[str]) -> List[str]:
    """
    Validate a list of model names.

    Parameters
    ----------
    model_names : List[str]
        List of model names to validate

    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    if not model_names:
        errors.append("At least one model must be specified")
        return errors

    for name in model_names:
        name_lower = name.lower().strip()
        if name_lower not in MODEL_DATA_REQUIREMENTS:
            valid = sorted(MODEL_DATA_REQUIREMENTS.keys())
            errors.append(f"Unknown model: '{name}'. Valid models: {valid}")

    return errors


def get_all_model_names() -> List[str]:
    """Get list of all supported model names."""
    return sorted(MODEL_DATA_REQUIREMENTS.keys())


def get_all_ensemble_names() -> List[str]:
    """Get list of all pre-defined ensemble names."""
    return sorted(ENSEMBLE_CONFIGS.keys())


__all__ = [
    # Enums
    "ModelFamily",
    "ScalerType",
    # Dataclasses
    "ModelDataRequirements",
    "EnsembleConfig",
    # Data
    "MODEL_DATA_REQUIREMENTS",
    "ENSEMBLE_CONFIGS",
    # Functions
    "get_model_requirements",
    "get_ensemble_config",
    "get_models_by_family",
    "get_combined_requirements",
    "validate_model_config",
    "get_all_model_names",
    "get_all_ensemble_names",
]
