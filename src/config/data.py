"""
Consolidated Data Configuration Classes.

This module contains all data-related configuration classes:
- FeatureConfig: Feature engineering settings
- LabelingConfig: Triple-barrier and labeling settings
- ScalerConfig: Data scaling/normalization settings
- SequenceConfig: Sequence dataset settings
- MTFConfig: Multi-timeframe feature settings
- DataPipelineConfig: Data pipeline orchestration

These are the CANONICAL locations for these configs.
Import from here:
    from src.config.data import FeatureConfig, LabelingConfig, ScalerConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.config.base import BaseConfig
from src.core.types import LabelingMethod

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
    Configuration for feature engineering.

    Controls which features are generated and their parameters.

    Attributes:
        mode: Feature generation mode ('full', 'minimal', 'custom')
        families: Feature families to include
        sma_periods: Periods for SMA calculations
        ema_periods: Periods for EMA calculations
        atr_periods: Periods for ATR calculations
        rsi_period: Period for RSI calculation
        macd_params: MACD parameters (fast, slow, signal)
        bb_period: Bollinger Bands period
        bb_std: Bollinger Bands standard deviation

    Example:
        config = FeatureConfig(
            mode="full",
            families=["momentum", "volatility", "trend"],
        )
    """

    mode: str = "full"
    families: list[str] = field(
        default_factory=lambda: ["price", "momentum", "volatility", "volume", "trend"]
    )
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50])
    atr_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    rsi_period: int = 14
    macd_params: dict[str, int] = field(
        default_factory=lambda: {"fast": 12, "slow": 26, "signal": 9}
    )
    bb_period: int = 20
    bb_std: float = 2.0

    # Feature selection
    selection_enabled: bool = True
    selection_method: str = "mda"  # mda, mdi, hybrid
    selection_n_features: int = 50
    selection_cv_splits: int = 5
    selection_min_frequency: float = 0.6  # Min fraction of CV folds a feature must appear in

    def validate(self) -> list[str]:
        """Validate feature configuration."""
        issues = super().validate()

        valid_modes = ["full", "minimal", "custom"]
        if self.mode not in valid_modes:
            issues.append(f"mode must be one of {valid_modes}, got '{self.mode}'")

        if self.rsi_period <= 0:
            issues.append(f"rsi_period must be positive, got {self.rsi_period}")

        if self.bb_std <= 0:
            issues.append(f"bb_std must be positive, got {self.bb_std}")

        valid_selection_methods = ["mda", "mdi", "hybrid"]
        if self.selection_method not in valid_selection_methods:
            issues.append(
                f"selection_method must be one of {valid_selection_methods}, "
                f"got '{self.selection_method}'"
            )

        return issues


# =============================================================================
# LABELING CONFIGURATION
# =============================================================================


@dataclass
class LabelingConfig(BaseConfig):
    """
    Configuration for label generation.

    Supports triple-barrier labeling and other methods.

    Attributes:
        method: Labeling method ('triple_barrier', 'directional', 'threshold')
        upper_mult: Upper barrier multiplier in ATR units. None (default) means
            "auto": use the per-symbol/per-horizon BARRIER_PARAMS table, which is
            the same source the backtester uses — keeping labels and backtest
            playing the same game. Set explicitly to override both.
        lower_mult: Lower barrier multiplier (ATR units). None = auto, see upper_mult.
        atr_period: ATR calculation period
        max_holding_bars: Maximum holding period (time barrier, bars).
            None = auto from BARRIER_PARAMS, see upper_mult.
        min_return_threshold: Minimum return for non-zero label

    Example:
        config = LabelingConfig(
            method="triple_barrier",
            upper_mult=2.0,
            lower_mult=2.0,
            max_holding_bars=20,
        )
    """

    method: str = "triple_barrier"
    upper_mult: float | None = None
    lower_mult: float | None = None
    atr_period: int = 14
    max_holding_bars: int | None = None
    min_return_threshold: float = 0.0

    # Triple-barrier specific
    use_dynamic_barriers: bool = True
    barrier_touch_is_exit: bool = True
    vertical_barrier_enabled: bool = True

    # Binary classification mode
    binary_mode: bool = False  # If True, remap labels to binary: 0=neutral, 1=significant_move

    # Optimization
    optimize_barriers: bool = True
    optimization_trials: int = 100
    target_class_balance: dict[str, float] | None = None

    def validate(self) -> list[str]:
        """Validate labeling configuration."""
        issues = super().validate()

        valid_methods = [m.value for m in LabelingMethod]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

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
# SCALER CONFIGURATION
# =============================================================================


@dataclass
class ScalerConfig(BaseConfig):
    """
    Configuration for data scaling/normalization.

    This is the CANONICAL ScalerConfig. Use this instead of duplicates in:
    - src/adapters/scaling.py (deprecated)
    - src/data/adapters/scaling.py (deprecated)
    - src/config/global_config.py (deprecated)
    - src/data/pipeline/stages/scaling/core.py (deprecated)

    Attributes:
        scaler_type: Type of scaler ('robust', 'standard', 'minmax', 'none')
        clip_outliers: Whether to clip scaled values
        clip_range: Range for clipping (min, max)
        per_feature: Whether to use per-feature scaling strategies

    Example:
        config = ScalerConfig(
            scaler_type="robust",
            clip_outliers=True,
            clip_range=(-5.0, 5.0),
        )
    """

    scaler_type: str = "robust"
    clip_outliers: bool = True
    clip_range: tuple[float, float] = (-5.0, 5.0)
    per_feature: bool = False

    # Category-specific scaling
    category_scalers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Normalize clip_range — YAML/JSON round-trips deliver it as a list."""
        if isinstance(self.clip_range, (list, tuple)) and len(self.clip_range) == 2:
            self.clip_range = (float(self.clip_range[0]), float(self.clip_range[1]))

    def validate(self) -> list[str]:
        """Validate scaler configuration."""
        issues = super().validate()

        valid_types = [t.value for t in ScalerType]
        if self.scaler_type not in valid_types:
            issues.append(f"scaler_type must be one of {valid_types}, got '{self.scaler_type}'")

        if self.clip_range[0] >= self.clip_range[1]:
            issues.append(f"clip_range min must be < max, got {self.clip_range}")

        return issues

    @classmethod
    def for_model_family(cls, family: str) -> ScalerConfig:
        """
        Get recommended scaler config for a model family.

        Args:
            family: Model family ('boosting', 'neural', 'transformer', 'classical')

        Returns:
            ScalerConfig with recommended settings
        """
        recommendations = {
            "boosting": cls(scaler_type="none"),
            "neural": cls(scaler_type="robust", clip_outliers=True),
            "transformer": cls(scaler_type="standard", clip_outliers=True),
            "classical": cls(scaler_type="standard", clip_outliers=True),
        }
        return recommendations.get(family, cls())


# =============================================================================
# SEQUENCE CONFIGURATION
# =============================================================================


@dataclass
class SequenceConfig(BaseConfig):
    """
    Configuration for sequence dataset generation.

    This is the CANONICAL SequenceConfig. Use this instead of duplicates in:
    - src/core/datasets/sequences.py (deprecated)
    - src/data/pipeline/stages/datasets/sequences.py (deprecated)
    - src/pipeline/stages/datasets/sequences.py (deprecated)

    Attributes:
        seq_len: Sequence length (number of time steps)
        stride: Step size between sequences
        feature_columns: List of feature column names
        label_column: Target column name
        weight_column: Sample weight column name
        symbol_column: Column for symbol isolation

    Example:
        config = SequenceConfig(
            seq_len=60,
            stride=1,
            label_column="label_h20",
        )
    """

    seq_len: int = 60
    stride: int = 1
    feature_columns: list[str] | None = None
    label_column: str = "label_h20"
    weight_column: str | None = "sample_weight_h20"
    symbol_column: str | None = "symbol"

    def __post_init__(self) -> None:
        """Initialize default feature columns."""
        if self.feature_columns is None:
            self.feature_columns = []

    def validate(self) -> list[str]:
        """Validate sequence configuration."""
        issues = super().validate()

        if self.seq_len <= 0:
            issues.append(f"seq_len must be positive, got {self.seq_len}")

        if self.stride <= 0:
            issues.append(f"stride must be positive, got {self.stride}")

        return issues


# =============================================================================
# MTF CONFIGURATION
# =============================================================================


@dataclass
class MTFConfig(BaseConfig):
    """
    Configuration for multi-timeframe features.

    This is the CANONICAL MTFConfig. Use this instead of duplicates in:
    - src/config/global_config.py (deprecated)
    - src/data/features/compute/mtf.py (deprecated)
    - src/features/compute/mtf.py (deprecated)
    - src/inference/preprocessing_graph.py (deprecated)

    Attributes:
        enabled: Whether MTF features are enabled
        mode: MTF mode ('none', 'indicators', 'bars', 'both', 'multi_stream')
        timeframes: List of additional timeframes
        primary_timeframe: Primary/base timeframe

    Example:
        config = MTFConfig(
            enabled=True,
            mode="indicators",
            timeframes=["5min", "15min", "1h"],
        )
    """

    enabled: bool = True
    mode: str = "indicators"
    timeframes: list[str] = field(default_factory=lambda: ["5min", "15min", "60min"])
    primary_timeframe: str = "5min"

    # Aggregation settings
    aggregate_ohlcv: bool = True
    aggregate_indicators: bool = True

    def validate(self) -> list[str]:
        """Validate MTF configuration."""
        issues = super().validate()

        valid_modes = [m.value for m in MTFMode]
        if self.mode not in valid_modes:
            issues.append(f"mode must be one of {valid_modes}, got '{self.mode}'")

        return issues


# =============================================================================
# MULTI-RESOLUTION CONFIGURATION
# =============================================================================


@dataclass
class MultiResolutionConfig(BaseConfig):
    """
    Configuration for multi-resolution/4D data adapters.

    This is the CANONICAL MultiResolutionConfig. Use this instead of:
    - MultiResolution4DConfig in src/adapters/multi_resolution.py (deprecated)
    - MultiResolution4DConfig in src/data/adapters/multi_resolution.py (deprecated)
    - MultiResolution4DConfig in src/data/pipeline/stages/datasets/adapters/multi_resolution.py (deprecated)

    Attributes:
        timeframes: List of timeframes for multi-resolution
        sequence_length: Sequence length for each timeframe
        feature_columns: Feature columns to include
        alignment_strategy: How to align different timeframes

    Example:
        config = MultiResolutionConfig(
            timeframes=["1min", "5min", "15min"],
            sequence_length=60,
        )
    """

    timeframes: list[str] = field(default_factory=lambda: ["1min", "5min", "15min"])
    sequence_length: int = 60
    feature_columns: list[str] | None = None
    alignment_strategy: str = "forward_fill"

    def validate(self) -> list[str]:
        """Validate multi-resolution configuration."""
        issues = super().validate()

        if not self.timeframes:
            issues.append("At least one timeframe must be specified")

        if self.sequence_length <= 0:
            issues.append(f"sequence_length must be positive, got {self.sequence_length}")

        return issues


# =============================================================================
# SESSION CONFIGURATION
# =============================================================================


@dataclass
class SessionConfig(BaseConfig):
    """
    Configuration for a single trading session.

    Attributes:
        name: Session name (e.g., 'rth', 'overnight')
        start_time: Session start time (HH:MM format)
        end_time: Session end time (HH:MM format)
        timezone: Timezone for the session
        is_primary: Whether this is the primary trading session
    """

    name: str = "rth"
    start_time: str = "09:30"
    end_time: str = "16:00"
    timezone: str = "America/New_York"
    is_primary: bool = True


@dataclass
class SessionsConfig(BaseConfig):
    """
    Configuration for all trading sessions.

    This is the CANONICAL SessionsConfig. Use this instead of:
    - SessionsConfig in src/data/pipeline/stages/sessions/config.py (deprecated)
    - SessionsConfig in src/pipeline/stages/sessions/config.py (deprecated)

    Attributes:
        sessions: List of session configurations
        filter_to_sessions: Whether to filter data to session times only
        add_session_features: Whether to add session indicator features
    """

    sessions: list[SessionConfig] = field(default_factory=list)
    filter_to_sessions: bool = False
    add_session_features: bool = True

    def get_session(self, name: str) -> SessionConfig | None:
        """Get a session by name."""
        for session in self.sessions:
            if session.name == name:
                return session
        return None


# =============================================================================
# SPLITS CONFIGURATION
# =============================================================================


@dataclass
class SplitConfig(BaseConfig):
    """
    Configuration for train/val/test splits.

    Attributes:
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        purge_bars: Bars to purge between splits
        embargo_bars: Bars to embargo after test
        auto_scale_purge_embargo: Whether to auto-calculate purge/embargo
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int = 60
    embargo_bars: int = 1440
    auto_scale_purge_embargo: bool = True

    def validate(self) -> list[str]:
        """Validate split configuration."""
        issues = super().validate()

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            issues.append(f"Split ratios must sum to 1.0, got {total}")

        if self.train_ratio <= 0:
            issues.append(f"train_ratio must be positive, got {self.train_ratio}")

        if self.val_ratio < 0:
            issues.append(f"val_ratio must be non-negative, got {self.val_ratio}")

        if self.test_ratio <= 0:
            issues.append(f"test_ratio must be positive, got {self.test_ratio}")

        return issues


# =============================================================================
# BAR CONFIGURATION
# =============================================================================


@dataclass
class BarConfig(BaseConfig):
    """
    Configuration for bar building.

    This is the CANONICAL BarConfig. Use this instead of:
    - BarConfig in src/data/pipeline/stages/clean/bar_builders/factory.py (deprecated)
    - BarConfig in src/pipeline/stages/clean/bar_builders/factory.py (deprecated)

    Attributes:
        bar_type: Type of bar ('time', 'tick', 'volume', 'dollar')
        timeframe: Target timeframe for time bars
        threshold: Threshold for tick/volume/dollar bars
        include_overnight: Whether to include overnight data
    """

    bar_type: str = "time"
    timeframe: str = "5min"
    threshold: int | None = None
    include_overnight: bool = True

    def validate(self) -> list[str]:
        """Validate bar configuration."""
        issues = super().validate()

        valid_types = ["time", "tick", "volume", "dollar"]
        if self.bar_type not in valid_types:
            issues.append(f"bar_type must be one of {valid_types}, got '{self.bar_type}'")

        if self.bar_type != "time" and self.threshold is None:
            issues.append(f"threshold required for {self.bar_type} bars")

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ScalerType",
    "FeatureCategory",
    "LabelingMethod",
    "MTFMode",
    # Configs
    "FeatureConfig",
    "LabelingConfig",
    "ScalerConfig",
    "SequenceConfig",
    "MTFConfig",
    "MultiResolutionConfig",
    "SessionConfig",
    "SessionsConfig",
    "SplitConfig",
    "BarConfig",
]
