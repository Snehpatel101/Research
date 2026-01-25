"""
Model data requirements configuration.

This module defines data preparation requirements for each model type,
enabling the data pipeline (Phase 1) to prepare appropriate datasets for
model training (Phase 2).

The model factory architecture requires Phase 1 to produce standardized
datasets that satisfy the requirements of all target model types.

This is the CANONICAL location for model data requirements. Import from here:
    from src.models.config import MODEL_DATA_REQUIREMENTS, ModelFamily
    from src.models.config.data_requirements import get_model_requirements
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.types import ModelFamily


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
        feature_selection_method: Method for feature selection ("mda", "mdi", "hybrid", "none")
        feature_selection_n_features: Number of features to select (0 = use max_features)
        description: Human-readable description

        # Phase 1 SNwH fields:
        input_rank: Input dimensionality (2=tabular, 3=sequence, 4=multi-TF)
        feature_mode: Feature mode ("engineered", "raw", "hybrid")
        mtf_mode: Multi-timeframe mode ("none", "indicators", "multi_stream")
        primary_timeframe: Default primary training timeframe
        mtf_timeframes: Additional timeframes for multi_stream mode
        min_features: Minimum features required
    """

    model_name: str
    family: ModelFamily
    feature_set: str
    requires_scaling: bool = False
    scaler_type: ScalerType = ScalerType.NONE
    requires_sequences: bool = False
    sequence_length: int = 60
    max_features: int | None = None
    supports_categorical: bool = False
    supports_missing: bool = False
    feature_selection_method: str = "mda"
    feature_selection_n_features: int = 0
    description: str = ""

    # Phase 1 SNwH: Per-model configuration fields
    input_rank: int = 2  # 2D (tabular), 3D (sequence), or 4D (multi-TF)
    feature_mode: str = "engineered"  # engineered, raw, hybrid
    mtf_mode: str = "none"  # none, indicators, multi_stream
    primary_timeframe: str = "5min"  # Default primary TF
    mtf_timeframes: tuple[str, ...] = ()  # Additional TFs for multi_stream
    min_features: int = 4  # Minimum features required

    @property
    def adapter_id(self) -> str:
        """Get adapter ID based on input_rank."""
        if self.input_rank == 2:
            return "tabular"
        elif self.input_rank == 3:
            return "sequence"
        elif self.input_rank == 4:
            return "multi_stream"
        return "tabular"


# =============================================================================
# MODEL DATA REQUIREMENTS
# =============================================================================
# Define data preparation needs for each supported model type.
# This allows Phase 1 to prepare appropriate datasets for Phase 2 training.

MODEL_DATA_REQUIREMENTS: dict[str, ModelDataRequirements] = {
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="15min",
        min_features=40,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="15min",
        min_features=40,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="15min",
        min_features=40,
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
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="5min",
        min_features=30,
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
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="5min",
        min_features=30,
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
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=30,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="5min",
        min_features=30,
    ),
    # -------------------------------------------------------------------------
    # TRANSFORMER MODELS (attention-based)
    # -------------------------------------------------------------------------
    # FEATURE SET SELECTION RATIONALE:
    # - transformer_raw: Minimal features (~10-15) for models designed to learn
    #   patterns from raw data. Use when model has strong representation learning.
    # - neural_optimal: Pre-computed features (~40-60) for models that benefit
    #   from engineered signals. Use when you want faster convergence.
    #
    # Current recommendations are based on empirical testing:
    # - transformer: Uses transformer_raw (vanilla attention learns features well)
    # - patchtst: Uses neural_optimal (patch-based benefits from pre-computed)
    # - itransformer: Uses neural_optimal (channel attention on features)
    # - tft: Uses neural_optimal (variable selection works with rich features)
    #
    # Override via model YAML config if needed. See feature_sets.py for details.
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
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="5min",
        min_features=20,
    ),
    "patchtst": ModelDataRequirements(
        model_name="patchtst",
        family=ModelFamily.NEURAL,  # Registry uses "neural" for all neural models
        feature_set="neural_optimal",  # Matches patchtst.yaml (was transformer_raw - MOD-002 fix)
        requires_scaling=True,
        scaler_type=ScalerType.STANDARD,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="PatchTST Transformer. Patches input sequences for efficiency.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="raw",
        mtf_mode="multi_stream",
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        min_features=4,
    ),
    # -------------------------------------------------------------------------
    # ADDITIONAL NEURAL MODELS (MOD-001: added missing models)
    # -------------------------------------------------------------------------
    "itransformer": ModelDataRequirements(
        model_name="itransformer",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",  # Matches itransformer.yaml
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="iTransformer. Channel-wise attention for multivariate time series.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="raw",
        mtf_mode="multi_stream",
        primary_timeframe="1min",
        mtf_timeframes=("5min", "15min"),
        min_features=4,
    ),
    "tft": ModelDataRequirements(
        model_name="tft",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",  # Matches tft.yaml
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="Temporal Fusion Transformer. Interpretable attention and variable selection.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="5min",
        min_features=30,
    ),
    "nbeats": ModelDataRequirements(
        model_name="nbeats",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",  # Matches nbeats.yaml
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="N-BEATS. Interpretable trend and seasonality decomposition.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=30,
    ),
    "inceptiontime": ModelDataRequirements(
        model_name="inceptiontime",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",  # Matches inceptiontime.yaml
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="InceptionTime CNN. Multi-scale temporal convolutions with residual connections.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=30,
    ),
    "resnet1d": ModelDataRequirements(
        model_name="resnet1d",
        family=ModelFamily.NEURAL,
        feature_set="neural_optimal",  # Matches resnet1d.yaml
        requires_scaling=True,
        scaler_type=ScalerType.ROBUST,
        requires_sequences=True,
        sequence_length=60,  # Default from YAML; can be overridden via config
        max_features=80,
        supports_categorical=False,
        supports_missing=False,
        description="ResNet1D. 1D residual network for deep temporal feature extraction.",
        # Phase 1 SNwH fields
        input_rank=3,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=30,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="indicators",
        primary_timeframe="10min",
        min_features=30,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="15min",
        min_features=20,
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
        # Phase 1 SNwH fields
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="15min",
        min_features=20,
    ),
    # -------------------------------------------------------------------------
    # ENSEMBLE MODELS (MOD-001: added missing models)
    # Note: Ensemble feature_set depends on base models; defaults to boosting_optimal
    # -------------------------------------------------------------------------
    "voting": ModelDataRequirements(
        model_name="voting",
        family=ModelFamily.ENSEMBLE,
        feature_set="boosting_optimal",  # Matches voting.yaml; depends on base models
        requires_scaling=False,  # Depends on base models
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Depends on base models
        max_features=None,  # No limit; depends on base models
        supports_categorical=True,  # Depends on base models
        supports_missing=True,  # Depends on base models
        description="Voting ensemble. Combines predictions via majority vote or averaging.",
        # Phase 1 SNwH fields (ensemble inherits from base models)
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=4,
    ),
    "stacking": ModelDataRequirements(
        model_name="stacking",
        family=ModelFamily.ENSEMBLE,
        feature_set="boosting_optimal",  # Matches stacking.yaml; depends on base models
        requires_scaling=False,  # Depends on base models
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Meta-learner receives 2D OOF predictions
        max_features=None,  # No limit; depends on base models
        supports_categorical=True,  # Depends on base models
        supports_missing=True,  # Depends on base models
        description="Stacking ensemble. OOF predictions feed meta-learner.",
        # Phase 1 SNwH fields (ensemble inherits from base models)
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=4,
    ),
    "blending": ModelDataRequirements(
        model_name="blending",
        family=ModelFamily.ENSEMBLE,
        feature_set="boosting_optimal",  # Matches blending.yaml; depends on base models
        requires_scaling=False,  # Depends on base models
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Meta-learner receives 2D holdout predictions
        max_features=None,  # No limit; depends on base models
        supports_categorical=True,  # Depends on base models
        supports_missing=True,  # Depends on base models
        description="Blending ensemble. Holdout predictions feed meta-learner.",
        # Phase 1 SNwH fields (ensemble inherits from base models)
        input_rank=2,
        feature_mode="engineered",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=4,
    ),
    # -------------------------------------------------------------------------
    # META-LEARNER MODELS (MOD-001: added missing models)
    # Meta-learners receive 2D OOF predictions: (n_samples, n_base_models * n_classes)
    # -------------------------------------------------------------------------
    "ridge_meta": ModelDataRequirements(
        model_name="ridge_meta",
        family=ModelFamily.META_LEARNER,
        feature_set="meta_learner",  # Special feature set: OOF predictions
        requires_scaling=False,  # OOF predictions are already probability-like
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Always receives 2D OOF predictions
        max_features=None,  # Depends on number of base models
        supports_categorical=False,
        supports_missing=False,
        description="Ridge meta-learner. L2-regularized linear stacking.",
        # Phase 1 SNwH fields (meta-learners receive OOF predictions)
        input_rank=2,
        feature_mode="oof_probs",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=2,
    ),
    "mlp_meta": ModelDataRequirements(
        model_name="mlp_meta",
        family=ModelFamily.META_LEARNER,
        feature_set="meta_learner",  # Special feature set: OOF predictions
        requires_scaling=True,  # MLPs benefit from scaled inputs
        scaler_type=ScalerType.STANDARD,
        requires_sequences=False,  # Always receives 2D OOF predictions
        max_features=None,  # Depends on number of base models
        supports_categorical=False,
        supports_missing=False,
        description="MLP meta-learner. Non-linear neural network stacking.",
        # Phase 1 SNwH fields (meta-learners receive OOF predictions)
        input_rank=2,
        feature_mode="oof_probs",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=2,
    ),
    "calibrated_meta": ModelDataRequirements(
        model_name="calibrated_meta",
        family=ModelFamily.META_LEARNER,
        feature_set="meta_learner",  # Special feature set: OOF predictions
        requires_scaling=False,  # Calibration handles probability scaling
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Always receives 2D OOF predictions
        max_features=None,  # Depends on number of base models
        supports_categorical=False,
        supports_missing=False,
        description="Calibrated meta-learner. Isotonic/Platt calibration for stacking.",
        # Phase 1 SNwH fields (meta-learners receive OOF predictions)
        input_rank=2,
        feature_mode="oof_probs",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=2,
    ),
    "xgboost_meta": ModelDataRequirements(
        model_name="xgboost_meta",
        family=ModelFamily.META_LEARNER,
        feature_set="meta_learner",  # Special feature set: OOF predictions
        requires_scaling=False,  # XGBoost handles raw features well
        scaler_type=ScalerType.NONE,
        requires_sequences=False,  # Always receives 2D OOF predictions
        max_features=None,  # Depends on number of base models
        supports_categorical=False,
        supports_missing=True,  # XGBoost handles missing values
        description="XGBoost meta-learner. Gradient boosting for non-linear stacking.",
        # Phase 1 SNwH fields (meta-learners receive OOF predictions)
        input_rank=2,
        feature_mode="oof_probs",
        mtf_mode="none",
        primary_timeframe="5min",
        min_features=2,
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
    base_models: list[str]
    meta_learner: str = "logistic"
    stacking_method: str = "soft"  # 'soft' (probabilities) or 'hard' (predictions)


ENSEMBLE_CONFIGS: dict[str, EnsembleConfig] = {
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
        base_models=[
            "transformer",
            "patchtst",
            "itransformer",
        ],  # informer removed (not in registry)
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


def get_models_by_family(family: ModelFamily) -> list[str]:
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
    return [name for name, req in MODEL_DATA_REQUIREMENTS.items() if req.family == family]


def get_combined_requirements(model_names: list[str]) -> dict[str, Any]:
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
        "feature_sets": {r.feature_set for r in requirements},
        "requires_scaling": any(r.requires_scaling for r in requirements),
        "scaler_types": {r.scaler_type for r in requirements if r.requires_scaling},
        "requires_sequences": any(r.requires_sequences for r in requirements),
        "max_sequence_length": max(
            (r.sequence_length for r in requirements if r.requires_sequences), default=0
        ),
        "min_max_features": min(
            (r.max_features for r in requirements if r.max_features), default=None
        ),
    }


def validate_model_config(model_names: list[str]) -> list[str]:
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


def get_all_model_names() -> list[str]:
    """Get list of all supported model names."""
    return sorted(MODEL_DATA_REQUIREMENTS.keys())


def get_all_ensemble_names() -> list[str]:
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
