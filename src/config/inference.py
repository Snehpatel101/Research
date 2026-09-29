"""
Consolidated Inference Configuration Classes.

This module contains all inference-related configuration classes:
- InferenceConfig: Main inference configuration
- BundleConfig: Model bundle configuration
- BacktestConfig: Backtesting configuration
- PreprocessingConfig: Preprocessing graph configuration

These are the CANONICAL locations for inference configs.
Import from here:
    from src.config.inference import InferenceConfig, BacktestConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.config.base import BaseConfig

# =============================================================================
# ENUMS
# =============================================================================


class InferenceMode(StrEnum):
    """Inference execution modes."""

    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


class PositionSizingMethod(StrEnum):
    """Position sizing methods."""

    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    CONFIDENCE = "confidence"


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================


@dataclass
class InferenceConfig(BaseConfig):
    """
    Main configuration for model inference.

    Attributes:
        mode: Inference mode ('single', 'batch', 'streaming')
        batch_size: Batch size for batch inference
        use_gpu: Whether to use GPU for inference
        num_workers: Number of data loading workers
        calibrate: Whether to apply probability calibration

    Example:
        config = InferenceConfig(
            mode="batch",
            batch_size=256,
            use_gpu=True,
        )
    """

    mode: str = "single"
    batch_size: int = 256
    use_gpu: bool = True
    num_workers: int = 4
    calibrate: bool = True

    # Timeout settings
    timeout_seconds: float = 30.0
    max_retries: int = 3

    # Output format
    output_probabilities: bool = True
    output_confidence: bool = True
    output_classes: bool = True

    def validate(self) -> list[str]:
        """Validate inference configuration."""
        issues = super().validate()

        valid_modes = [m.value for m in InferenceMode]
        if self.mode not in valid_modes:
            issues.append(f"mode must be one of {valid_modes}, got '{self.mode}'")

        if self.batch_size <= 0:
            issues.append(f"batch_size must be positive, got {self.batch_size}")

        if self.timeout_seconds <= 0:
            issues.append(f"timeout_seconds must be positive, got {self.timeout_seconds}")

        return issues


# =============================================================================
# BUNDLE CONFIGURATION
# =============================================================================


@dataclass
class BundleConfig(BaseConfig):
    """
    Configuration for model bundles.

    Bundles contain everything needed for inference:
    model, scaler, feature config, etc.

    Attributes:
        bundle_path: Path to the bundle directory
        model_name: Name of the model in the bundle
        horizon: Prediction horizon
        symbol: Trading symbol
        version: Bundle version

    Example:
        config = BundleConfig(
            bundle_path="./bundles/xgboost_h20",
            model_name="xgboost",
            horizon=20,
        )
    """

    bundle_path: str | Path = ""
    model_name: str = ""
    horizon: int = 20
    symbol: str = ""
    version: str = "1.0.0"

    # Contents
    has_scaler: bool = True
    has_calibrator: bool = True
    has_feature_config: bool = True

    def validate(self) -> list[str]:
        """Validate bundle configuration."""
        issues = super().validate()

        if not self.bundle_path:
            issues.append("bundle_path is required")

        if not self.model_name:
            issues.append("model_name is required")

        if self.horizon <= 0:
            issues.append(f"horizon must be positive, got {self.horizon}")

        return issues


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================


@dataclass
class BacktestConfig(BaseConfig):
    """
    Configuration for backtesting.

    This is the CANONICAL BacktestConfig. Use this instead of:
    - BacktestConfig in src/backtesting/backtest.py (deprecated)
    - BacktestConfig in src/inference/backtesting/backtest.py (deprecated)

    Attributes:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        commission: Commission per trade (fraction)
        slippage: Slippage per trade (fraction)
        position_sizing: Position sizing method

    Example:
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=100000,
        )
    """

    start_date: str | None = None
    end_date: str | None = None
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    position_sizing: str = "fixed"

    # Position limits
    max_position_size: float = 1.0
    max_positions: int = 1
    allow_short: bool = True
    max_leverage: float = 1.0

    # Risk management
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None

    # Execution
    fill_at: str = "close"  # close, open, midpoint (formerly vwap)
    delay_bars: int = 0

    def validate(self) -> list[str]:
        """Validate backtest configuration."""
        issues = super().validate()

        if self.initial_capital <= 0:
            issues.append(f"initial_capital must be positive, got {self.initial_capital}")

        if self.commission < 0:
            issues.append(f"commission must be non-negative, got {self.commission}")

        if self.slippage < 0:
            issues.append(f"slippage must be non-negative, got {self.slippage}")

        valid_sizing = [m.value for m in PositionSizingMethod]
        if self.position_sizing not in valid_sizing:
            issues.append(
                f"position_sizing must be one of {valid_sizing}, " f"got '{self.position_sizing}'"
            )

        if self.max_position_size <= 0:
            issues.append(f"max_position_size must be positive, got {self.max_position_size}")

        return issues


# =============================================================================
# POSITION SIZER CONFIGURATION
# =============================================================================


@dataclass
class PositionSizerConfig(BaseConfig):
    """
    Configuration for position sizing.

    This is the CANONICAL PositionSizerConfig. Use this instead of:
    - PositionSizerConfig in src/backtesting/position_sizing.py (deprecated)
    - PositionSizerConfig in src/inference/backtesting/position_sizing.py (deprecated)

    Attributes:
        method: Position sizing method
        base_size: Base position size
        kelly_fraction: Kelly fraction for Kelly criterion
        volatility_target: Target volatility for vol-based sizing

    Example:
        config = PositionSizerConfig(
            method="kelly",
            kelly_fraction=0.25,
        )
    """

    method: str = "fixed"
    base_size: float = 1.0
    kelly_fraction: float = 0.25
    volatility_target: float = 0.15
    confidence_threshold: float = 0.6

    # Risk limits
    max_size: float = 1.0
    min_size: float = 0.0

    def validate(self) -> list[str]:
        """Validate position sizer configuration."""
        issues = super().validate()

        valid_methods = [m.value for m in PositionSizingMethod]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

        if self.base_size < 0:
            issues.append(f"base_size must be non-negative, got {self.base_size}")

        if not 0 < self.kelly_fraction <= 1:
            issues.append(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")

        if self.volatility_target <= 0:
            issues.append(f"volatility_target must be positive, got {self.volatility_target}")

        return issues


# =============================================================================
# PREPROCESSING GRAPH CONFIGURATION
# =============================================================================


@dataclass
class PreprocessingGraphConfig(BaseConfig):
    """
    Configuration for preprocessing graph at inference time.

    This is the CANONICAL PreprocessingGraphConfig. Use this instead of:
    - PreprocessingGraphConfig in src/inference/preprocessing_graph.py (deprecated)

    Attributes:
        enable_cleaning: Whether to apply data cleaning
        enable_indicators: Whether to compute indicators
        enable_mtf: Whether to compute MTF features
        enable_scaling: Whether to apply scaling
        enable_regime: Whether to compute regime features

    Example:
        config = PreprocessingGraphConfig(
            enable_cleaning=True,
            enable_indicators=True,
            enable_scaling=True,
        )
    """

    enable_cleaning: bool = True
    enable_indicators: bool = True
    enable_mtf: bool = True
    enable_scaling: bool = True
    enable_regime: bool = False
    enable_wavelet: bool = False

    # Step-specific configs
    cleaning_config: dict[str, Any] = field(default_factory=dict)
    indicator_config: dict[str, Any] = field(default_factory=dict)
    mtf_config: dict[str, Any] = field(default_factory=dict)
    scaling_config: dict[str, Any] = field(default_factory=dict)
    regime_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate preprocessing graph configuration."""
        issues = super().validate()
        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "InferenceMode",
    "PositionSizingMethod",
    # Configs
    "InferenceConfig",
    "BundleConfig",
    "BacktestConfig",
    "PositionSizerConfig",
    "PreprocessingGraphConfig",
]
