"""
Central constants - re-exports from src.core.common and local modules.

This module provides unified access to constants from src.core.common
and local constant definitions (default_periods, thresholds).

Usage:
    from src.config.constants import CANONICAL_TIMEFRAMES, DEFAULT_SPLIT_RATIOS
    from src.config.constants import is_valid_timeframe, get_timeframe_minutes
    from src.config.constants import HorizonConfig, validate_horizons

    # Technical indicator defaults
    from src.config.constants import RSI_PERIOD, SMA_PERIOD, ATR_PERIOD

    # Validation thresholds
    from src.config.constants import LEAKAGE_CORRELATION_THRESHOLD, MAX_NAN_RATIO
"""

# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# =============================================================================
# HORIZON CONFIGURATION
# =============================================================================
# =============================================================================
# DEFAULT PERIODS (technical indicators)
# =============================================================================
from src.config.constants.default_periods import (
    # Volatility
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    # Lookback
    DEFAULT_LOOKBACK,
    # Sequences
    DEFAULT_SEQUENCE_LENGTH,
    EMA_PERIOD,
    EMA_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
    MAX_LOOKBACK,
    MAX_SEQUENCE_LENGTH,
    MIN_LOOKBACK,
    MIN_SEQUENCE_LENGTH,
    OBV_PERIOD,
    # Oscillators
    RSI_PERIOD,
    # Moving averages
    SMA_PERIOD,
    STOCH_PERIOD,
    STOCH_SMOOTH,
    # Volume
    VWAP_PERIOD,
)

# =============================================================================
# THRESHOLDS (validation and filtering)
# =============================================================================
from src.config.constants.thresholds import (
    DEFAULT_CONFIDENCE,
    FEATURE_CORRELATION_WARN,
    HIGH_CONFIDENCE_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
    # Correlation
    LEAKAGE_CORRELATION_THRESHOLD,
    MAX_CLASS_IMBALANCE,
    MAX_MISSING_PCT,
    # Data quality
    MAX_NAN_RATIO,
    # Performance
    MIN_ACCURACY,
    # Class balance
    MIN_CLASS_RATIO,
    MIN_F1_SCORE,
    MIN_ROWS,
    # Signal
    MIN_SIGNAL_RATIO,
    MIN_TRAIN_SAMPLES,
    MIN_VOLATILITY,
    OVERFITTING_GAP,
    TARGET_CLASS_BALANCE,
    VOL_RATIO_MAX,
    # Volatility
    VOL_RATIO_MIN,
)
from src.core.common.horizon_config import (
    # Horizon lists
    ACTIVE_HORIZONS,
    # Purge/embargo configuration
    DEFAULT_TIMEFRAME_MINUTES,
    EMBARGO_MULTIPLIER,
    EMBARGO_TIME_MINUTES,
    HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    MIN_EMBARGO_BARS,
    PURGE_MULTIPLIER,
    SUPPORTED_HORIZONS,
    # HorizonConfig dataclass
    HorizonConfig,
    # Functions
    auto_scale_purge_embargo,
    compute_embargo_bars,
    get_default_barrier_params_for_horizon,
    get_scaled_horizons,
    validate_horizons,
)

# =============================================================================
# SPLIT RATIO CONFIGURATION
# =============================================================================
from src.core.common.split_ratios import (
    DEFAULT_SPLIT_RATIOS,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    validate_split_ratios,
)
from src.core.common.timeframes import (
    # Core timeframe lists
    ALL_CANONICAL_TIMEFRAMES,
    CANONICAL_TIMEFRAMES,
    EXTENDED_TIMEFRAMES,
    FULL_9TF_LADDER,
    SUPPORTED_TIMEFRAMES,
    # Mappings
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
    # Functions
    get_canonical_from_suffix,
    get_timeframe_minutes,
    get_timeframe_suffix,
    is_valid_timeframe,
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_to_minutes,
    validate_timeframe,
)

__all__ = [
    # Timeframe lists
    "CANONICAL_TIMEFRAMES",
    "FULL_9TF_LADDER",
    "EXTENDED_TIMEFRAMES",
    "ALL_CANONICAL_TIMEFRAMES",
    "SUPPORTED_TIMEFRAMES",
    # Timeframe mappings
    "TIMEFRAME_ALIASES",
    "TIMEFRAME_TO_MINUTES",
    "TIMEFRAME_TO_FREQ",
    # Timeframe functions
    "normalize_timeframe",
    "normalize_timeframe_list",
    "get_timeframe_minutes",
    "timeframe_to_minutes",
    "is_valid_timeframe",
    "validate_timeframe",
    "get_timeframe_suffix",
    "get_canonical_from_suffix",
    # Split ratios
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "validate_split_ratios",
    # Horizon configuration
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "LABEL_HORIZONS",
    "LOOKBACK_HORIZONS",
    # Purge/embargo configuration
    "PURGE_MULTIPLIER",
    "EMBARGO_MULTIPLIER",
    "EMBARGO_TIME_MINUTES",
    "DEFAULT_TIMEFRAME_MINUTES",
    "MIN_EMBARGO_BARS",
    # HorizonConfig dataclass
    "HorizonConfig",
    # Horizon functions
    "validate_horizons",
    "get_scaled_horizons",
    "auto_scale_purge_embargo",
    "compute_embargo_bars",
    "get_default_barrier_params_for_horizon",
    # ==========================================================================
    # DEFAULT PERIODS (technical indicators)
    # ==========================================================================
    # Moving averages
    "SMA_PERIOD",
    "EMA_PERIOD",
    "EMA_SLOW_PERIOD",
    # Oscillators
    "RSI_PERIOD",
    "STOCH_PERIOD",
    "STOCH_SMOOTH",
    "MACD_SIGNAL_PERIOD",
    # Volatility
    "ATR_PERIOD",
    "BB_PERIOD",
    "BB_STD",
    # Volume
    "VWAP_PERIOD",
    "OBV_PERIOD",
    # Lookback
    "DEFAULT_LOOKBACK",
    "MIN_LOOKBACK",
    "MAX_LOOKBACK",
    # Sequences
    "DEFAULT_SEQUENCE_LENGTH",
    "MIN_SEQUENCE_LENGTH",
    "MAX_SEQUENCE_LENGTH",
    # ==========================================================================
    # THRESHOLDS (validation and filtering)
    # ==========================================================================
    # Signal
    "MIN_SIGNAL_RATIO",
    "DEFAULT_CONFIDENCE",
    "HIGH_CONFIDENCE_THRESHOLD",
    # Volatility
    "VOL_RATIO_MIN",
    "VOL_RATIO_MAX",
    "MIN_VOLATILITY",
    # Data quality
    "MAX_NAN_RATIO",
    "MIN_ROWS",
    "MIN_TRAIN_SAMPLES",
    "MAX_MISSING_PCT",
    # Correlation
    "LEAKAGE_CORRELATION_THRESHOLD",
    "HIGH_CORRELATION_THRESHOLD",
    "FEATURE_CORRELATION_WARN",
    # Class balance
    "MIN_CLASS_RATIO",
    "MAX_CLASS_IMBALANCE",
    "TARGET_CLASS_BALANCE",
    # Performance
    "MIN_ACCURACY",
    "MIN_F1_SCORE",
    "OVERFITTING_GAP",
]
