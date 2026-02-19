"""
Consolidated Ensemble Configuration Classes.

This module contains all ensemble-related configuration classes:
- EnsembleConfig: Main ensemble configuration
- MetaLearnerConfig: Meta-learner specific configuration
- StackingConfig: Stacking ensemble configuration
- VotingConfig: Voting ensemble configuration
- BlendingConfig: Blending ensemble configuration

These are the CANONICAL locations for ensemble configs.
Import from here:
    from src.config.ensemble import EnsembleConfig, MetaLearnerConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.config.base import BaseConfig

# =============================================================================
# ENUMS
# =============================================================================


class EnsembleMethod(StrEnum):
    """Supported ensemble methods."""

    STACKING = "stacking"
    VOTING = "voting"
    BLENDING = "blending"
    WEIGHTED_AVERAGE = "weighted_average"


class VotingType(StrEnum):
    """Voting types for voting ensemble."""

    HARD = "hard"
    SOFT = "soft"


class MetaLearnerType(StrEnum):
    """Supported meta-learner types."""

    RIDGE = "ridge"
    LOGISTIC = "logistic"
    MLP = "mlp"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CALIBRATED = "calibrated"


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================


@dataclass
class EnsembleConfig(BaseConfig):
    """
    Main configuration for ensemble models.

    This is the CANONICAL EnsembleConfig. Use this instead of:
    - EnsembleConfig in src/models/config/data_requirements.py (deprecated)

    Attributes:
        name: Ensemble name
        method: Ensemble method ('stacking', 'voting', 'blending')
        base_models: List of base model names
        meta_learner: Meta-learner type
        use_probabilities: Whether to use probabilities (soft voting)
        cv_folds: CV folds for OOF generation

    Example:
        config = EnsembleConfig(
            name="my_ensemble",
            method="stacking",
            base_models=["xgboost", "lightgbm", "lstm"],
            meta_learner="ridge",
        )
    """

    name: str = ""
    description: str = ""
    method: str = "stacking"
    base_models: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    meta_learner: str = "ridge"
    use_probabilities: bool = True
    cv_folds: int = 5

    # Alignment settings for heterogeneous models
    alignment_strategy: str = "intersection"  # intersection, union, majority
    n_classes: int = 3

    # Calibration
    calibrate_base_models: bool = True
    calibrate_ensemble: bool = True

    def validate(self) -> list[str]:
        """Validate ensemble configuration."""
        issues = super().validate()

        valid_methods = [m.value for m in EnsembleMethod]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

        if len(self.base_models) < 2:
            issues.append("At least 2 base models required for ensemble")

        if self.cv_folds < 2:
            issues.append(f"cv_folds must be >= 2, got {self.cv_folds}")

        return issues


# =============================================================================
# META-LEARNER CONFIGURATION
# =============================================================================


@dataclass
class MetaLearnerConfig(BaseConfig):
    """
    Configuration for meta-learner in stacking ensemble.

    This is the CANONICAL MetaLearnerConfig. Use this instead of:
    - MetaLearnerConfig in src/models/ensemble/meta_factory.py (deprecated)

    Attributes:
        learner_type: Type of meta-learner
        regularization: Regularization strength (for ridge/logistic)
        hidden_sizes: Hidden layer sizes (for MLP)
        learning_rate: Learning rate (for neural meta-learners)
        max_iter: Maximum iterations

    Example:
        config = MetaLearnerConfig(
            learner_type="ridge",
            regularization=1.0,
        )
    """

    learner_type: str = "ridge"
    regularization: float = 1.0
    hidden_sizes: list[int] = field(default_factory=lambda: [32])
    learning_rate: float = 0.001
    max_iter: int = 1000

    # Calibration
    apply_calibration: bool = True
    calibration_method: str = "isotonic"

    # Feature engineering on OOF predictions
    add_confidence_features: bool = True
    add_agreement_features: bool = True

    def validate(self) -> list[str]:
        """Validate meta-learner configuration."""
        issues = super().validate()

        valid_types = [t.value for t in MetaLearnerType]
        if self.learner_type not in valid_types:
            issues.append(f"learner_type must be one of {valid_types}, got '{self.learner_type}'")

        if self.regularization < 0:
            issues.append(f"regularization must be non-negative, got {self.regularization}")

        if self.learning_rate <= 0:
            issues.append(f"learning_rate must be positive, got {self.learning_rate}")

        if self.max_iter <= 0:
            issues.append(f"max_iter must be positive, got {self.max_iter}")

        return issues


# =============================================================================
# STACKING CONFIGURATION
# =============================================================================


@dataclass
class StackingConfig(BaseConfig):
    """
    Configuration for stacking ensemble.

    Stacking uses out-of-fold predictions from base models to train
    a meta-learner.

    Attributes:
        cv_folds: Number of CV folds for OOF generation
        meta_learner: Meta-learner configuration
        passthrough: Whether to include original features
        stack_method: How to stack predictions ('auto', 'predict_proba', 'predict')

    Example:
        config = StackingConfig(
            cv_folds=5,
            meta_learner=MetaLearnerConfig(learner_type="ridge"),
        )
    """

    cv_folds: int = 5
    meta_learner: MetaLearnerConfig = field(default_factory=MetaLearnerConfig)
    passthrough: bool = False
    stack_method: str = "auto"

    # OOF handling
    purge_bars: int = 60
    embargo_bars: int = 1440

    def validate(self) -> list[str]:
        """Validate stacking configuration."""
        issues = super().validate()

        if self.cv_folds < 2:
            issues.append(f"cv_folds must be >= 2, got {self.cv_folds}")

        issues.extend(self.meta_learner.validate())

        return issues


# =============================================================================
# VOTING CONFIGURATION
# =============================================================================


@dataclass
class VotingConfig(BaseConfig):
    """
    Configuration for voting ensemble.

    Voting combines predictions via majority vote (hard) or
    probability averaging (soft).

    Attributes:
        voting_type: 'hard' or 'soft'
        weights: Optional weights for each model
        flatten_transform: Whether to flatten in transform

    Example:
        config = VotingConfig(
            voting_type="soft",
            weights=[1.0, 1.0, 2.0],
        )
    """

    voting_type: str = "soft"
    weights: list[float] | None = None
    flatten_transform: bool = True

    def validate(self) -> list[str]:
        """Validate voting configuration."""
        issues = super().validate()

        valid_types = [t.value for t in VotingType]
        if self.voting_type not in valid_types:
            issues.append(f"voting_type must be one of {valid_types}, got '{self.voting_type}'")

        if self.weights is not None and any(w < 0 for w in self.weights):
            issues.append("All weights must be non-negative")

        return issues


# =============================================================================
# BLENDING CONFIGURATION
# =============================================================================


@dataclass
class BlendingConfig(BaseConfig):
    """
    Configuration for blending ensemble.

    Blending uses a held-out set for training the meta-learner
    instead of out-of-fold predictions.

    Attributes:
        holdout_fraction: Fraction of data for meta-learner training
        meta_learner: Meta-learner configuration
        stratify: Whether to stratify the holdout split

    Example:
        config = BlendingConfig(
            holdout_fraction=0.2,
            meta_learner=MetaLearnerConfig(learner_type="logistic"),
        )
    """

    holdout_fraction: float = 0.2
    meta_learner: MetaLearnerConfig = field(default_factory=MetaLearnerConfig)
    stratify: bool = True

    def validate(self) -> list[str]:
        """Validate blending configuration."""
        issues = super().validate()

        if not 0 < self.holdout_fraction < 1:
            issues.append(f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}")

        issues.extend(self.meta_learner.validate())

        return issues


# =============================================================================
# OOF ALIGNMENT CONFIGURATION
# =============================================================================


@dataclass
class OOFAlignmentConfig(BaseConfig):
    """
    Configuration for OOF prediction alignment.

    When base models have different sequence lengths or training
    requirements, their OOF predictions need alignment.

    Attributes:
        strategy: Alignment strategy ('intersection', 'union', 'majority')
        n_classes: Number of output classes
        fill_value: Value for missing predictions (union strategy)
        min_coverage: Minimum coverage required for a sample

    Example:
        config = OOFAlignmentConfig(
            strategy="intersection",
            n_classes=3,
        )
    """

    strategy: str = "intersection"
    n_classes: int = 3
    fill_value: float = 0.33333  # Uniform distribution for missing
    min_coverage: float = 0.5

    # Per-model settings
    model_offsets: dict[str, int] = field(default_factory=dict)
    sequence_lengths: dict[str, int] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate OOF alignment configuration."""
        issues = super().validate()

        valid_strategies = ["intersection", "union", "majority"]
        if self.strategy not in valid_strategies:
            issues.append(f"strategy must be one of {valid_strategies}, got '{self.strategy}'")

        if self.n_classes < 2:
            issues.append(f"n_classes must be >= 2, got {self.n_classes}")

        if not 0 <= self.min_coverage <= 1:
            issues.append(f"min_coverage must be in [0, 1], got {self.min_coverage}")

        return issues


# =============================================================================
# PREDEFINED ENSEMBLE CONFIGURATIONS
# =============================================================================

# Common ensemble configurations for convenience
PREDEFINED_ENSEMBLES: dict[str, EnsembleConfig] = {
    "boosting_ensemble": EnsembleConfig(
        name="boosting_ensemble",
        description="Ensemble of tree-based boosting models",
        base_models=["xgboost", "lightgbm", "catboost"],
        meta_learner="ridge",
    ),
    "neural_ensemble": EnsembleConfig(
        name="neural_ensemble",
        description="Ensemble of recurrent neural networks",
        base_models=["lstm", "gru", "tcn"],
        meta_learner="ridge",
    ),
    "hybrid_ensemble": EnsembleConfig(
        name="hybrid_ensemble",
        description="Hybrid ensemble mixing boosting and neural models",
        base_models=["xgboost", "lstm", "transformer"],
        meta_learner="xgboost",
    ),
    "full_ensemble": EnsembleConfig(
        name="full_ensemble",
        description="Full ensemble using all model families",
        base_models=["xgboost", "lightgbm", "lstm", "gru", "transformer"],
        meta_learner="xgboost",
    ),
}


def get_predefined_ensemble(name: str) -> EnsembleConfig:
    """
    Get a predefined ensemble configuration by name.

    Args:
        name: Ensemble name (e.g., 'boosting_ensemble')

    Returns:
        EnsembleConfig copy

    Raises:
        ValueError: If ensemble name not found
    """
    if name not in PREDEFINED_ENSEMBLES:
        valid = list(PREDEFINED_ENSEMBLES.keys())
        raise ValueError(f"Unknown ensemble: '{name}'. Valid: {valid}")

    return PREDEFINED_ENSEMBLES[name].copy()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EnsembleMethod",
    "VotingType",
    "MetaLearnerType",
    # Configs
    "EnsembleConfig",
    "MetaLearnerConfig",
    "StackingConfig",
    "VotingConfig",
    "BlendingConfig",
    "OOFAlignmentConfig",
    # Predefined
    "PREDEFINED_ENSEMBLES",
    "get_predefined_ensemble",
]
