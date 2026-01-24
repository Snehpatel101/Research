"""
Base Feature Sets - Default feature configurations per model family.

Phase 3 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.

This module defines sensible default feature sets for each model family that
Optuna can use as starting points during feature selection optimization.
Different model families work best with different feature combinations.

Feature Categories:
    - RAW: OHLCV passthrough (5 features)
    - MOMENTUM: RSI, MACD, Stochastic, etc. (23 features)
    - MOVING_AVERAGE: SMA, EMA, crossovers (16 features)
    - VOLATILITY: ATR, Bollinger Bands, Keltner (25 features)
    - VOLUME: OBV, VWAP, dollar volume (15 features)
    - TREND: ADX, Supertrend (6 features)
    - PRICE: Returns, ratios, autocorrelation (12 features)

Usage:
    from src.optimization.base_feature_sets import (
        get_base_features,
        get_all_features,
        get_feature_category,
        BASE_FEATURE_SETS,
    )

    # Get base features for a model family
    features = get_base_features(ModelFamily.BOOSTING)

    # Get all available features
    all_features = get_all_features()

    # Get category for a feature
    category = get_feature_category("rsi_14")  # "momentum"
"""

from __future__ import annotations

from typing import Any

from src.core.types import ModelFamily

# =============================================================================
# FEATURE CATEGORIES
# Using actual feature names from src/data/features/compute/ modules
# =============================================================================

# Momentum features (from src/data/features/compute/momentum.py)
MOMENTUM_FEATURES: list[str] = [
    # RSI
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "rsi_overbought",
    "rsi_oversold",
    # MACD
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "macd_cross_up",
    "macd_cross_down",
    # Stochastic
    "stoch_k",
    "stoch_d",
    "stoch_overbought",
    "stoch_oversold",
    # Williams %R
    "williams_r",
    # ROC
    "roc_5",
    "roc_10",
    "roc_20",
    # CCI
    "cci_14",
    "cci_20",
    # MFI
    "mfi_14",
    "mfi_overbought",
    "mfi_oversold",
]

# Trend features (from src/data/features/compute/trend.py)
TREND_FEATURES: list[str] = [
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "adx_strong_trend",
    "supertrend",
    "supertrend_direction",
]

# Volatility features (from src/data/features/compute/volatility.py)
VOLATILITY_FEATURES: list[str] = [
    # ATR
    "atr_7",
    "atr_14",
    "atr_21",
    "atr_pct_14",
    # Bollinger Bands
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",
    "close_bb_zscore",
    # Keltner Channels
    "kc_upper",
    "kc_middle",
    "kc_lower",
    "kc_position",
    # Historical Volatility
    "hvol_10",
    "hvol_20",
    "hvol_60",
    # Advanced Volatility
    "parkinson_vol",
    "gk_vol",
    "rs_vol",
    "yz_vol",
    # Return Distribution
    "return_skew_20",
    "return_kurt_20",
    # GARCH (stubs)
    "garch_vol_forecast",
    "garch_vol_ratio",
]

# Volume features (from src/data/features/compute/volume.py)
VOLUME_FEATURES: list[str] = [
    # OBV
    "obv",
    "obv_sma_20",
    # Volume SMA/Ratio
    "volume_sma_20",
    "volume_ratio",
    "volume_zscore",
    # VWAP
    "vwap",
    "price_to_vwap",
    # TWAP
    "twap",
    "twap_10",
    "twap_20",
    "price_to_twap_10",
    # Dollar Volume
    "dollar_volume",
    "dollar_volume_sma_10",
    "dollar_volume_sma_20",
    "dollar_volume_ratio",
]

# Raw/Price features (from src/data/features/compute/raw.py)
RAW_FEATURES: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Price-derived features (from src/data/features/compute/price.py)
PRICE_FEATURES: list[str] = [
    # Returns
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "log_return_1",
    "log_return_5",
    # Ratios
    "hl_ratio",
    "co_ratio",
    "range_pct",
    "clv",
    # Autocorrelation
    "autocorr_lag1",
    "autocorr_lag5",
]

# Moving average features (from src/data/features/compute/moving_average.py)
MOVING_AVERAGE_FEATURES: list[str] = [
    # SMA
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    # EMA
    "ema_9",
    "ema_12",
    "ema_21",
    "ema_26",
    "ema_50",
    # Price ratios
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_ema_21",
    # Crossovers
    "sma_cross_10_50",
    "sma_cross_20_200",
    "ema_cross_9_21",
]


# =============================================================================
# BASE FEATURE SETS PER MODEL FAMILY
# =============================================================================

# Core momentum indicators that work well universally
_CORE_MOMENTUM: list[str] = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci_14",
    "mfi_14",
]

# Core volatility indicators
_CORE_VOLATILITY: list[str] = [
    "atr_14",
    "atr_pct_14",
    "bb_width",
    "bb_position",
    "close_bb_zscore",
    "hvol_20",
]

# Core trend indicators
_CORE_TREND: list[str] = [
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "supertrend_direction",
]

# Core volume indicators
_CORE_VOLUME: list[str] = [
    "obv",
    "volume_ratio",
    "volume_zscore",
    "vwap",
    "price_to_vwap",
]

# Core moving averages
_CORE_MA: list[str] = [
    "sma_20",
    "sma_50",
    "ema_21",
    "price_to_sma_20",
    "price_to_sma_50",
]

# Core price features
_CORE_PRICE: list[str] = [
    "return_1",
    "return_5",
    "log_return_1",
    "range_pct",
    "clv",
]

BASE_FEATURE_SETS: dict[ModelFamily, list[str]] = {
    # Boosting models (XGBoost, LightGBM, CatBoost) work well with all feature types
    # They handle categorical-like features well and benefit from comprehensive coverage
    ModelFamily.BOOSTING: [
        *_CORE_MOMENTUM,
        *_CORE_TREND,
        *_CORE_VOLATILITY,
        *_CORE_VOLUME,
        *_CORE_MA,
        *_CORE_PRICE,
        # Additional useful features for tree-based models
        "rsi_7",
        "rsi_21",
        "roc_10",
        "roc_20",
        "atr_7",
        "atr_21",
        "sma_10",
        "sma_100",
        "ema_9",
        "ema_50",
    ],
    # Classical models (Logistic Regression, SVM) need normalized, lower-dimensional sets
    ModelFamily.CLASSICAL: [
        *_CORE_MOMENTUM,
        *_CORE_VOLATILITY,
        *_CORE_PRICE,
        # Fewer features for regularization stability
    ],
    # Neural networks (LSTM, GRU, TCN) prefer normalized, sequential-aware features
    # Avoid features that are difficult to normalize or highly sparse
    ModelFamily.NEURAL: [
        *_CORE_MOMENTUM,
        *_CORE_VOLATILITY,
        *_CORE_PRICE,
        *_CORE_MA,
        # Neural networks handle these well
        "bb_position",
        "kc_position",
        "close_bb_zscore",
        "volume_zscore",
        "return_10",
        "return_20",
        "log_return_5",
    ],
    # Transformers (PatchTST, iTransformer, TFT) can handle raw price + patterns
    # They benefit from multi-scale features and can learn complex relationships
    ModelFamily.TRANSFORMER: [
        # Include raw features - transformers can learn from them
        "close",
        "high",
        "low",
        "volume",
        *_CORE_MOMENTUM,
        *_CORE_VOLATILITY,
        *_CORE_PRICE,
        *_CORE_MA,
        # Multi-scale ATR
        "atr_7",
        "atr_14",
        "atr_21",
        # Multi-scale returns
        "return_1",
        "return_5",
        "return_10",
        "return_20",
        # Autocorrelation features
        "autocorr_lag1",
        "autocorr_lag5",
    ],
    # Ensemble models use a broad set for diversity
    ModelFamily.ENSEMBLE: [
        *_CORE_MOMENTUM,
        *_CORE_TREND,
        *_CORE_VOLATILITY,
        *_CORE_VOLUME,
        *_CORE_MA,
        *_CORE_PRICE,
    ],
    # Meta-learner models typically use predictions from other models as features
    # They also benefit from some market structure features
    ModelFamily.META_LEARNER: [
        *_CORE_VOLATILITY,
        *_CORE_TREND,
        *_CORE_PRICE,
        "volume_ratio",
        "volume_zscore",
    ],
}


# =============================================================================
# DEFAULT FEATURE PARAMETERS
# =============================================================================

DEFAULT_FEATURE_PARAMS: dict[str, dict[str, Any]] = {
    # RSI
    "rsi_7": {"period": 7},
    "rsi_14": {"period": 14},
    "rsi_21": {"period": 21},
    # Stochastic
    "stoch_k": {"k_period": 14, "d_period": 3},
    "stoch_d": {"k_period": 14, "d_period": 3},
    # MACD
    "macd_line": {"fast": 12, "slow": 26, "signal": 9},
    "macd_signal": {"fast": 12, "slow": 26, "signal": 9},
    "macd_histogram": {"fast": 12, "slow": 26, "signal": 9},
    # ATR
    "atr_7": {"period": 7},
    "atr_14": {"period": 14},
    "atr_21": {"period": 21},
    "atr_pct_14": {"period": 14},
    # Bollinger Bands
    "bb_upper": {"period": 20, "std": 2.0},
    "bb_middle": {"period": 20},
    "bb_lower": {"period": 20, "std": 2.0},
    "bb_width": {"period": 20, "std": 2.0},
    "bb_position": {"period": 20, "std": 2.0},
    "close_bb_zscore": {"period": 20},
    # Keltner Channels
    "kc_upper": {"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
    "kc_middle": {"ema_period": 20},
    "kc_lower": {"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
    "kc_position": {"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
    # Historical Volatility
    "hvol_10": {"period": 10},
    "hvol_20": {"period": 20},
    "hvol_60": {"period": 60},
    # ADX
    "adx_14": {"period": 14},
    "plus_di_14": {"period": 14},
    "minus_di_14": {"period": 14},
    # Supertrend
    "supertrend": {"period": 10, "multiplier": 3.0},
    "supertrend_direction": {"period": 10, "multiplier": 3.0},
    # Moving Averages
    "sma_10": {"period": 10},
    "sma_20": {"period": 20},
    "sma_50": {"period": 50},
    "sma_100": {"period": 100},
    "sma_200": {"period": 200},
    "ema_9": {"span": 9},
    "ema_12": {"span": 12},
    "ema_21": {"span": 21},
    "ema_26": {"span": 26},
    "ema_50": {"span": 50},
    # ROC
    "roc_5": {"period": 5},
    "roc_10": {"period": 10},
    "roc_20": {"period": 20},
    # CCI
    "cci_14": {"period": 14},
    "cci_20": {"period": 20},
    # MFI
    "mfi_14": {"period": 14},
    # Williams %R
    "williams_r": {"period": 14},
    # Volume
    "volume_sma_20": {"period": 20},
    "obv_sma_20": {"period": 20},
    "twap_10": {"period": 10},
    "twap_20": {"period": 20},
    "dollar_volume_sma_10": {"period": 10},
    "dollar_volume_sma_20": {"period": 20},
    # Returns
    "return_1": {"period": 1},
    "return_5": {"period": 5},
    "return_10": {"period": 10},
    "return_20": {"period": 20},
    "log_return_1": {"period": 1},
    "log_return_5": {"period": 5},
    # Autocorrelation
    "autocorr_lag1": {"lag": 1, "window": 20},
    "autocorr_lag5": {"lag": 5, "window": 30},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_base_features(model_family: ModelFamily) -> list[str]:
    """
    Get the base feature set for a model family.

    Args:
        model_family: The model family to get features for

    Returns:
        List of feature names for the given model family.
        Falls back to BOOSTING if family not found.
    """
    return list(BASE_FEATURE_SETS.get(model_family, BASE_FEATURE_SETS[ModelFamily.BOOSTING]))


def get_all_features() -> list[str]:
    """
    Get all available features across all categories.

    Returns:
        Sorted list of all unique feature names
    """
    all_features = set(
        MOMENTUM_FEATURES
        + TREND_FEATURES
        + VOLATILITY_FEATURES
        + VOLUME_FEATURES
        + RAW_FEATURES
        + PRICE_FEATURES
        + MOVING_AVERAGE_FEATURES
    )
    return sorted(all_features)


def get_feature_category(feature: str) -> str:
    """
    Get the category for a given feature.

    Args:
        feature: The feature name

    Returns:
        Category name: "momentum", "trend", "volatility", "volume",
        "raw", "price", "moving_average", or "unknown"
    """
    if feature in MOMENTUM_FEATURES:
        return "momentum"
    if feature in TREND_FEATURES:
        return "trend"
    if feature in VOLATILITY_FEATURES:
        return "volatility"
    if feature in VOLUME_FEATURES:
        return "volume"
    if feature in RAW_FEATURES:
        return "raw"
    if feature in PRICE_FEATURES:
        return "price"
    if feature in MOVING_AVERAGE_FEATURES:
        return "moving_average"
    return "unknown"


def get_features_by_category(category: str) -> list[str]:
    """
    Get all features in a specific category.

    Args:
        category: The category name (momentum, trend, volatility, etc.)

    Returns:
        List of feature names in that category

    Raises:
        ValueError: If category is unknown
    """
    category_map = {
        "momentum": MOMENTUM_FEATURES,
        "trend": TREND_FEATURES,
        "volatility": VOLATILITY_FEATURES,
        "volume": VOLUME_FEATURES,
        "raw": RAW_FEATURES,
        "price": PRICE_FEATURES,
        "moving_average": MOVING_AVERAGE_FEATURES,
    }
    if category not in category_map:
        raise ValueError(
            f"Unknown category '{category}'. " f"Valid categories: {list(category_map.keys())}"
        )
    return list(category_map[category])


def get_feature_params(feature: str) -> dict[str, Any]:
    """
    Get the default parameters for a feature.

    Args:
        feature: The feature name

    Returns:
        Dictionary of parameter names to values.
        Empty dict if feature has no configurable parameters.
    """
    return dict(DEFAULT_FEATURE_PARAMS.get(feature, {}))


def get_minimal_feature_set() -> list[str]:
    """
    Get a minimal feature set for quick experimentation.

    This is a curated set of the most important indicators
    that covers momentum, volatility, trend, and volume.

    Returns:
        List of ~15 core features
    """
    return [
        # Momentum
        "rsi_14",
        "macd_histogram",
        "stoch_k",
        # Volatility
        "atr_14",
        "bb_position",
        "hvol_20",
        # Trend
        "adx_14",
        "supertrend_direction",
        # Volume
        "volume_ratio",
        "vwap",
        # Price
        "return_1",
        "return_5",
        "clv",
        # Moving Average
        "price_to_sma_20",
        "ema_cross_9_21",
    ]


def get_extensive_feature_set() -> list[str]:
    """
    Get an extensive feature set for thorough optimization.

    This includes all features that are non-redundant and
    computationally feasible (excludes GARCH stubs).

    Returns:
        List of ~80 features
    """
    extensive = set(get_all_features())
    # Remove GARCH stubs as they return NaN
    extensive.discard("garch_vol_forecast")
    extensive.discard("garch_vol_ratio")
    return sorted(extensive)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Feature category lists
    "MOMENTUM_FEATURES",
    "TREND_FEATURES",
    "VOLATILITY_FEATURES",
    "VOLUME_FEATURES",
    "RAW_FEATURES",
    "PRICE_FEATURES",
    "MOVING_AVERAGE_FEATURES",
    # Base feature sets
    "BASE_FEATURE_SETS",
    "DEFAULT_FEATURE_PARAMS",
    # Helper functions
    "get_base_features",
    "get_all_features",
    "get_feature_category",
    "get_features_by_category",
    "get_feature_params",
    "get_minimal_feature_set",
    "get_extensive_feature_set",
]
