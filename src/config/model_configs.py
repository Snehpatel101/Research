"""
Consolidated Model Configuration Classes.

This module contains model-specific configuration classes:
- ModelConfig: Generic model configuration
- XGBoostConfig: XGBoost-specific configuration
- LightGBMConfig: LightGBM-specific configuration
- LSTMConfig: LSTM-specific configuration
- TransformerConfig: Transformer-specific configuration
- PerModelConfig: Per-model customization

These are the CANONICAL locations for model configs.
Import from here:
    from src.config.model_configs import XGBoostConfig, LSTMConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from src.config.base import BaseConfig
from src.core.types import ModelFamily

# =============================================================================
# ENUMS
# =============================================================================


class ActivationType(StrEnum):
    """Neural network activation functions."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"


# =============================================================================
# BASE MODEL CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig(BaseConfig):
    """
    Base configuration for all model types.

    Attributes:
        model_name: Name/type of the model
        family: Model family classification
        n_classes: Number of output classes
        random_state: Random seed
        verbose: Verbosity level

    Example:
        config = ModelConfig(
            model_name="xgboost",
            family="boosting",
            n_classes=3,
        )
    """

    model_name: str = ""
    family: str = "boosting"
    n_classes: int = 3
    random_state: int = 42
    verbose: int = 0

    # Feature requirements
    min_features: int = 4
    max_features: int | None = None
    requires_scaling: bool = False

    def validate(self) -> list[str]:
        """Validate model configuration."""
        issues = super().validate()

        if not self.model_name:
            issues.append("model_name is required")

        valid_families = [f.value for f in ModelFamily]
        if self.family not in valid_families:
            issues.append(f"family must be one of {valid_families}, got '{self.family}'")

        if self.n_classes < 2:
            issues.append(f"n_classes must be >= 2, got {self.n_classes}")

        if self.min_features < 1:
            issues.append(f"min_features must be >= 1, got {self.min_features}")

        return issues


# =============================================================================
# BOOSTING MODEL CONFIGURATIONS
# =============================================================================


@dataclass
class XGBoostConfig(BaseConfig):
    """
    Configuration for XGBoost models.

    Attributes:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        min_child_weight: Minimum sum of instance weight
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns for each tree
        reg_alpha: L1 regularization term
        reg_lambda: L2 regularization term
        scale_pos_weight: Balance of positive and negative weights

    Example:
        config = XGBoostConfig(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
        )
    """

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0

    # Additional settings
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int | None = 50
    tree_method: str = "auto"  # auto, exact, approx, hist, gpu_hist

    def validate(self) -> list[str]:
        """Validate XGBoost configuration."""
        issues = super().validate()

        if self.n_estimators <= 0:
            issues.append(f"n_estimators must be positive, got {self.n_estimators}")

        if self.max_depth <= 0:
            issues.append(f"max_depth must be positive, got {self.max_depth}")

        if self.learning_rate <= 0:
            issues.append(f"learning_rate must be positive, got {self.learning_rate}")

        if not 0 < self.subsample <= 1:
            issues.append(f"subsample must be in (0, 1], got {self.subsample}")

        if not 0 < self.colsample_bytree <= 1:
            issues.append(f"colsample_bytree must be in (0, 1], got {self.colsample_bytree}")

        return issues

    def to_xgb_params(self) -> dict[str, Any]:
        """Convert to XGBoost parameter dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
        }


@dataclass
class LightGBMConfig(BaseConfig):
    """
    Configuration for LightGBM models.

    Attributes:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth (-1 = no limit)
        learning_rate: Learning rate
        num_leaves: Maximum number of leaves per tree
        min_child_samples: Minimum samples in leaf
        subsample: Subsample ratio (bagging_fraction)
        colsample_bytree: Feature fraction
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization

    Example:
        config = LightGBMConfig(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.1,
        )
    """

    n_estimators: int = 500
    max_depth: int = -1
    learning_rate: float = 0.1
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    # Additional settings
    objective: str = "multiclass"
    metric: str = "multi_logloss"
    boosting_type: str = "gbdt"  # gbdt, dart, goss
    early_stopping_rounds: int | None = 50

    def validate(self) -> list[str]:
        """Validate LightGBM configuration."""
        issues = super().validate()

        if self.n_estimators <= 0:
            issues.append(f"n_estimators must be positive, got {self.n_estimators}")

        if self.num_leaves <= 1:
            issues.append(f"num_leaves must be > 1, got {self.num_leaves}")

        if self.learning_rate <= 0:
            issues.append(f"learning_rate must be positive, got {self.learning_rate}")

        return issues

    def to_lgb_params(self) -> dict[str, Any]:
        """Convert to LightGBM parameter dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
        }


@dataclass
class CatBoostConfig(BaseConfig):
    """
    Configuration for CatBoost models.

    Attributes:
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        l2_leaf_reg: L2 regularization coefficient
        random_strength: Random strength for scoring splits
        bagging_temperature: Bayesian bootstrap parameter

    Example:
        config = CatBoostConfig(
            iterations=500,
            depth=6,
            learning_rate=0.1,
        )
    """

    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0

    # Additional settings
    loss_function: str = "MultiClass"
    eval_metric: str = "MultiClass"
    early_stopping_rounds: int | None = 50
    verbose: bool = False

    def validate(self) -> list[str]:
        """Validate CatBoost configuration."""
        issues = super().validate()

        if self.iterations <= 0:
            issues.append(f"iterations must be positive, got {self.iterations}")

        if self.depth <= 0:
            issues.append(f"depth must be positive, got {self.depth}")

        if self.learning_rate <= 0:
            issues.append(f"learning_rate must be positive, got {self.learning_rate}")

        return issues


# =============================================================================
# NEURAL NETWORK CONFIGURATIONS
# =============================================================================


@dataclass
class LSTMConfig(BaseConfig):
    """
    Configuration for LSTM models.

    Attributes:
        hidden_size: Size of LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        batch_first: Whether input is (batch, seq, features)

    Example:
        config = LSTMConfig(
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
        )
    """

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    batch_first: bool = True

    # Head configuration
    head_hidden_sizes: list[int] = field(default_factory=lambda: [64])
    head_dropout: float = 0.1
    head_activation: str = "relu"

    # Training
    sequence_length: int = 60

    def validate(self) -> list[str]:
        """Validate LSTM configuration."""
        issues = super().validate()

        if self.hidden_size <= 0:
            issues.append(f"hidden_size must be positive, got {self.hidden_size}")

        if self.num_layers <= 0:
            issues.append(f"num_layers must be positive, got {self.num_layers}")

        if not 0 <= self.dropout < 1:
            issues.append(f"dropout must be in [0, 1), got {self.dropout}")

        if self.sequence_length <= 0:
            issues.append(f"sequence_length must be positive, got {self.sequence_length}")

        return issues


@dataclass
class GRUConfig(BaseConfig):
    """
    Configuration for GRU models.

    Same structure as LSTM but for GRU architecture.
    """

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    batch_first: bool = True
    head_hidden_sizes: list[int] = field(default_factory=lambda: [64])
    head_dropout: float = 0.1
    head_activation: str = "relu"
    sequence_length: int = 60

    def validate(self) -> list[str]:
        """Validate GRU configuration."""
        issues = super().validate()

        if self.hidden_size <= 0:
            issues.append(f"hidden_size must be positive, got {self.hidden_size}")

        if self.num_layers <= 0:
            issues.append(f"num_layers must be positive, got {self.num_layers}")

        return issues


@dataclass
class TCNConfig(BaseConfig):
    """
    Configuration for Temporal Convolutional Network models.

    Attributes:
        num_channels: List of channel sizes for each TCN block
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        dilation_base: Base for exponential dilation

    Example:
        config = TCNConfig(
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.2,
        )
    """

    num_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.2
    dilation_base: int = 2
    sequence_length: int = 120

    def validate(self) -> list[str]:
        """Validate TCN configuration."""
        issues = super().validate()

        if not self.num_channels:
            issues.append("num_channels must have at least one element")

        if self.kernel_size <= 0:
            issues.append(f"kernel_size must be positive, got {self.kernel_size}")

        return issues


# =============================================================================
# TRANSFORMER CONFIGURATIONS
# =============================================================================


@dataclass
class TransformerConfig(BaseConfig):
    """
    Configuration for Transformer models.

    Attributes:
        d_model: Model/embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_seq_len: Maximum sequence length

    Example:
        config = TransformerConfig(
            d_model=128,
            n_heads=8,
            n_layers=4,
        )
    """

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128

    # Additional settings
    activation: str = "gelu"
    pre_norm: bool = True
    use_cls_token: bool = True

    def validate(self) -> list[str]:
        """Validate Transformer configuration."""
        issues = super().validate()

        if self.d_model <= 0:
            issues.append(f"d_model must be positive, got {self.d_model}")

        if self.n_heads <= 0:
            issues.append(f"n_heads must be positive, got {self.n_heads}")

        if self.d_model % self.n_heads != 0:
            issues.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        if self.n_layers <= 0:
            issues.append(f"n_layers must be positive, got {self.n_layers}")

        return issues


@dataclass
class PatchTSTConfig(BaseConfig):
    """
    Configuration for PatchTST models.

    Attributes:
        patch_size: Size of each patch
        stride: Stride between patches
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers

    Example:
        config = PatchTSTConfig(
            patch_size=16,
            stride=8,
            d_model=128,
        )
    """

    patch_size: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    sequence_length: int = 60

    def validate(self) -> list[str]:
        """Validate PatchTST configuration."""
        issues = super().validate()

        if self.patch_size <= 0:
            issues.append(f"patch_size must be positive, got {self.patch_size}")

        if self.stride <= 0:
            issues.append(f"stride must be positive, got {self.stride}")

        if self.d_model % self.n_heads != 0:
            issues.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        return issues


# =============================================================================
# PER-MODEL CONFIGURATION
# =============================================================================


@dataclass
class PerModelConfig(BaseConfig):
    """
    Per-model customization configuration.

    This is the CANONICAL PerModelConfig. Use this instead of:
    - PerModelConfig in src/models/config/per_model_config.py (deprecated)

    Allows customizing settings for specific models in the ensemble.

    Attributes:
        model_name: Name of the model to customize
        enabled: Whether this model is enabled
        hyperparams: Model-specific hyperparameters
        feature_set: Feature set to use
        timeframe: Primary timeframe for training

    Example:
        config = PerModelConfig(
            model_name="xgboost",
            hyperparams={"max_depth": 8},
            feature_set="boosting_optimal",
        )
    """

    model_name: str = ""
    enabled: bool = True
    hyperparams: dict[str, Any] = field(default_factory=dict)
    feature_set: str | None = None
    timeframe: str | None = None
    sequence_length: int | None = None
    batch_size: int | None = None

    def validate(self) -> list[str]:
        """Validate per-model configuration."""
        issues = super().validate()

        if not self.model_name:
            issues.append("model_name is required")

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ModelFamily",
    "ActivationType",
    # Base
    "ModelConfig",
    # Boosting
    "XGBoostConfig",
    "LightGBMConfig",
    "CatBoostConfig",
    # Neural
    "LSTMConfig",
    "GRUConfig",
    "TCNConfig",
    # Transformer
    "TransformerConfig",
    "PatchTSTConfig",
    # Per-model
    "PerModelConfig",
]
