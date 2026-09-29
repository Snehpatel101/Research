"""
Data-section configuration classes for ExperimentConfig.

Every field of these classes reaches the pipeline (see
``ExperimentConfig.to_pipeline_config()`` and ``MLFactory._run_data_pipeline``):

- FeatureConfig: feature-selection switch
- LabelingConfig: triple-barrier overrides + binary mode
- SequenceConfig: window length for sequence models
- MTFConfig: multi-timeframe feature switch + timeframes
- SplitConfig: chronological train/val/test ratios

The enums below are shared with the data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.config.base import BaseConfig

# =============================================================================
# ENUMS
# =============================================================================


class ScalerType(StrEnum):
    """Supported scaler types for data normalization."""

    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    QUANTILE = "quantile"


class FeatureCategory(StrEnum):
    """Feature categories for scaling strategy selection."""

    RETURNS = "returns"
    OSCILLATOR = "oscillator"
    PRICE_LEVEL = "price_level"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TEMPORAL = "temporal"
    BINARY = "binary"
    UNKNOWN = "unknown"


class MTFMode(StrEnum):
    """Multi-timeframe aggregation modes."""

    NONE = "none"
    BARS = "bars"
    INDICATORS = "indicators"
    BOTH = "both"
    MULTI_STREAM = "multi_stream"


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================


@dataclass
class FeatureConfig(BaseConfig):
    """
    Feature configuration.

    The feature set itself is fixed (``FeatureEngineer`` computes every
    family; MTF is controlled by ``MTFConfig``). What is configurable is
    whether the MDA feature-selection pipeline prunes it per model (feature
    counts come from each model's contract).

    Attributes:
        selection_enabled: Run train-only MDA feature selection per model
    """

    selection_enabled: bool = True


# =============================================================================
# LABELING CONFIGURATION
# =============================================================================


@dataclass
class LabelingConfig(BaseConfig):
    """
    Triple-barrier labeling configuration.

    Attributes:
        upper_mult: Upper barrier multiplier in ATR units. None (default) means
            "auto": use the per-symbol/per-horizon BARRIER_PARAMS table, which is
            the same source the backtester uses — keeping labels and backtest
            playing the same game. Set explicitly to override both.
        lower_mult: Lower barrier multiplier (ATR units). None = auto, see upper_mult.
        atr_period: ATR calculation period
        max_holding_bars: Maximum holding period (time barrier, bars).
            None = auto from BARRIER_PARAMS, see upper_mult.
        binary_mode: Remap labels to {0: time-out, 1: barrier hit}

    Example:
        config = LabelingConfig(
            upper_mult=2.0,
            lower_mult=2.0,
            max_holding_bars=20,
        )
    """

    upper_mult: float | None = None
    lower_mult: float | None = None
    atr_period: int = 14
    max_holding_bars: int | None = None

    # Binary classification mode
    binary_mode: bool = False  # If True, remap labels to binary: 0=neutral, 1=significant_move

    def validate(self) -> list[str]:
        """Validate labeling configuration."""
        issues = super().validate()

        if self.upper_mult is not None and self.upper_mult <= 0:
            issues.append(f"upper_mult must be positive, got {self.upper_mult}")

        if self.lower_mult is not None and self.lower_mult <= 0:
            issues.append(f"lower_mult must be positive, got {self.lower_mult}")

        if self.atr_period <= 0:
            issues.append(f"atr_period must be positive, got {self.atr_period}")

        if self.max_holding_bars is not None and self.max_holding_bars <= 0:
            issues.append(f"max_holding_bars must be positive, got {self.max_holding_bars}")

        return issues


# =============================================================================
# SEQUENCE CONFIGURATION
# =============================================================================


@dataclass
class SequenceConfig(BaseConfig):
    """
    Sequence-window configuration for 3D/4D models.

    Attributes:
        seq_len: Window length (time steps per sample) for EVERY sequence model.
            None (default) gives each model its contract length (e.g. TCN 64,
            Transformer 128, most others 60).
    """

    seq_len: int | None = None

    def validate(self) -> list[str]:
        """Validate sequence configuration."""
        issues = super().validate()

        if self.seq_len is not None and self.seq_len <= 0:
            issues.append(f"seq_len must be positive, got {self.seq_len}")

        return issues


# =============================================================================
# MTF CONFIGURATION
# =============================================================================


@dataclass
class MTFConfig(BaseConfig):
    """
    Multi-timeframe feature configuration.

    Attributes:
        enabled: Whether MTF features are computed
        timeframes: Higher timeframes to aggregate (also the streams fed to
            multi-stream 4D models)

    Example:
        config = MTFConfig(enabled=True, timeframes=["15min", "60min"])
    """

    enabled: bool = True
    timeframes: list[str] = field(default_factory=lambda: ["5min", "15min", "60min"])


# =============================================================================
# SPLITS CONFIGURATION
# =============================================================================


@dataclass
class SplitConfig(BaseConfig):
    """
    Chronological train/val/test split ratios.

    The purge/embargo gaps between splits come from
    ``training.purge_bars`` / ``training.embargo_bars``.

    Attributes:
        train_ratio: Proportion for training
        val_ratio: Proportion for validation (calibration, early stopping)
        test_ratio: Proportion held out for testing
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self) -> list[str]:
        """Validate split configuration."""
        issues = super().validate()

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            issues.append(f"Split ratios must sum to 1.0, got {total}")

        if self.train_ratio <= 0:
            issues.append(f"train_ratio must be positive, got {self.train_ratio}")

        if self.val_ratio < 0:
            issues.append(f"val_ratio must be non-negative, got {self.val_ratio}")

        if self.test_ratio < 0:
            issues.append(f"test_ratio must be non-negative, got {self.test_ratio}")

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ScalerType",
    "FeatureCategory",
    "MTFMode",
    # Configs
    "FeatureConfig",
    "LabelingConfig",
    "SequenceConfig",
    "MTFConfig",
    "SplitConfig",
]
