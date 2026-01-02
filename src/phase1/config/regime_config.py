"""
Regime detection and adaptive barrier configuration.
"""

from __future__ import annotations

from typing import Any

from src.phase1.config.barriers_config import get_barrier_params

REGIME_CONFIG: dict[str, dict[str, Any]] = {
    "volatility": {
        "enabled": True,
        "atr_period": 14,
        "lookback": 100,
        "low_percentile": 25.0,
        "high_percentile": 75.0,
    },
    "trend": {
        "enabled": True,
        "adx_period": 14,
        "sma_period": 50,
        "adx_threshold": 25.0,
    },
    "structure": {
        "enabled": True,
        "lookback": 100,
        "min_lag": 2,
        "max_lag": 20,
        "mean_reverting_threshold": 0.4,
        "trending_threshold": 0.6,
    },
}


REGIME_BARRIER_ADJUSTMENTS: dict[str, dict[str, dict[str, float]]] = {
    "volatility": {
        "high": {"k_up_mult": 1.2, "k_down_mult": 1.2, "max_bars_mult": 1.1},
        "low": {"k_up_mult": 0.9, "k_down_mult": 0.9, "max_bars_mult": 0.9},
        "normal": {"k_up_mult": 1.0, "k_down_mult": 1.0, "max_bars_mult": 1.0},
    },
    "trend": {
        "uptrend": {"k_up_mult": 1.05, "k_down_mult": 0.95, "max_bars_mult": 1.0},
        "downtrend": {"k_up_mult": 0.95, "k_down_mult": 1.05, "max_bars_mult": 1.0},
        "sideways": {"k_up_mult": 1.0, "k_down_mult": 1.0, "max_bars_mult": 0.95},
    },
    "structure": {
        "trending": {"k_up_mult": 1.05, "k_down_mult": 1.05, "max_bars_mult": 1.1},
        "mean_reverting": {"k_up_mult": 0.95, "k_down_mult": 0.95, "max_bars_mult": 0.9},
        "random": {"k_up_mult": 1.0, "k_down_mult": 1.0, "max_bars_mult": 1.0},
    },
}


def _normalize_regime(value: Any, mapping: dict[str, str]) -> str:
    if value is None:
        return "normal"
    normalized = str(value).strip().lower().replace(" ", "_")
    return mapping.get(normalized, normalized)


def get_regime_adjusted_barriers(
    symbol: str,
    horizon: int,
    volatility_regime: str,
    trend_regime: str,
    structure_regime: str,
) -> dict[str, Any]:
    """
    Apply regime multipliers to base barrier parameters.
    """
    base_params = get_barrier_params(symbol, horizon)
    k_up = base_params["k_up"]
    k_down = base_params["k_down"]
    max_bars = base_params["max_bars"]

    regime_map = {
        "volatility": _normalize_regime(
            volatility_regime,
            {"high_vol": "high", "low_vol": "low", "medium": "normal"},
        ),
        "trend": _normalize_regime(
            trend_regime,
            {"up": "uptrend", "down": "downtrend", "flat": "sideways", "neutral": "sideways"},
        ),
        "structure": _normalize_regime(
            structure_regime,
            {"meanreverting": "mean_reverting", "mean-reverting": "mean_reverting"},
        ),
    }

    adjustments: dict[str, dict[str, float | str]] = {}

    for category, regime_value in regime_map.items():
        multipliers = REGIME_BARRIER_ADJUSTMENTS.get(category, {}).get(
            regime_value,
            {"k_up_mult": 1.0, "k_down_mult": 1.0, "max_bars_mult": 1.0},
        )
        k_up *= multipliers["k_up_mult"]
        k_down *= multipliers["k_down_mult"]
        max_bars *= multipliers["max_bars_mult"]
        adjustments[category] = {
            "regime": regime_value,
            **multipliers,
        }

    adjusted_params = {
        "k_up": round(k_up, 4),
        "k_down": round(k_down, 4),
        "max_bars": max(1, int(round(max_bars))),
        "base_params": base_params,
        "adjustments": adjustments,
    }

    return adjusted_params


__all__ = [
    "REGIME_CONFIG",
    "REGIME_BARRIER_ADJUSTMENTS",
    "get_regime_adjusted_barriers",
]
