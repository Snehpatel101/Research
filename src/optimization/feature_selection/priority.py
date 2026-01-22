"""
Feature Priority Rankings.

Provides interpretability/priority rankings for feature selection.
Higher scores indicate more interpretable/fundamental features that
should be preferred when removing correlated pairs.

This module consolidates priority constants from:
- src/phase1/utils/feature_selection.py
"""

from __future__ import annotations

# Feature interpretability ranking - higher is more interpretable/fundamental
# This guides which feature to keep from correlated pairs
FEATURE_PRIORITY: dict[str, int] = {
    # Price-based returns (most fundamental)
    "log_return": 100,
    "simple_return": 95,
    "high_low_range": 90,
    "close_open_range": 85,
    # RSI and momentum (classic indicators)
    "rsi": 90,
    "rsi_oversold": 85,
    "rsi_overbought": 85,
    "stoch_k": 80,
    "stoch_d": 75,
    "williams_r": 70,  # Essentially same as stoch_k
    # Moving averages - prefer simpler/shorter
    "sma_10": 80,
    "sma_20": 78,
    "sma_50": 75,
    "sma_100": 72,
    "sma_200": 70,
    "ema_9": 75,
    "ema_21": 73,
    "ema_50": 70,
    # Price relative to moving averages (more useful than raw MA values)
    "close_to_sma_10": 85,
    "close_to_sma_20": 83,
    "close_to_sma_50": 80,
    "close_to_sma_100": 78,
    "close_to_sma_200": 76,
    "close_to_ema_9": 82,
    "close_to_ema_21": 80,
    "close_to_ema_50": 78,
    # MACD components
    "macd": 85,
    "macd_signal": 80,
    "macd_hist": 90,  # Most useful - the difference
    "macd_crossover": 75,
    # Bollinger Bands - prefer derived metrics
    "bb_position": 90,  # Most useful - normalized position
    "bb_width": 85,
    "bb_upper": 60,  # Raw values less useful
    "bb_lower": 60,
    # ATR - prefer percentage versions
    "atr_7_pct": 85,
    "atr_14_pct": 83,
    "atr_21_pct": 80,
    "atr_7": 70,
    "atr_14": 68,
    "atr_21": 65,
    # ADX and directional indicators
    "adx": 85,
    "plus_di": 75,
    "minus_di": 75,
    # Volume indicators
    "volume_ratio": 85,
    "volume_zscore": 80,
    "volume_sma_20": 70,
    "obv": 65,
    "obv_sma_20": 60,
    # VWAP
    "close_to_vwap": 85,
    "vwap": 60,  # Raw VWAP less useful
    # Rate of change - prefer shorter periods
    "roc_5": 80,
    "roc_10": 78,
    "roc_20": 75,
    # Time features
    "hour_sin": 70,
    "hour_cos": 70,
    "dow_sin": 70,
    "dow_cos": 70,
    "is_rth": 80,
    # Regime features
    "vol_regime": 85,
    "trend_regime": 85,
}

# Default priority for unknown features
DEFAULT_PRIORITY: int = 50


def get_feature_priority(feature_name: str) -> int:
    """
    Get the priority/interpretability score for a feature.

    Higher scores indicate more interpretable/fundamental features that
    should be preferred when removing correlated pairs.

    Args:
        feature_name: Name of the feature

    Returns:
        Priority score (0-100)
    """
    return FEATURE_PRIORITY.get(feature_name, DEFAULT_PRIORITY)


__all__ = [
    "DEFAULT_PRIORITY",
    "FEATURE_PRIORITY",
    "get_feature_priority",
]
