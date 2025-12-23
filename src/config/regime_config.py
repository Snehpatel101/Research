"""
Regime detection and adaptive barrier configuration.

This module contains configuration for market regime detection
(volatility, trend, structure) and regime-adaptive barrier adjustments.
"""

from .barriers_config import get_barrier_params


# =============================================================================
# REGIME DETECTION CONFIGURATION
# =============================================================================
# Regime detection classifies market conditions into discrete states.
# These states can be used as:
# 1. Features: Add regime columns to feature DataFrame for model training
# 2. Filters: Use regime to select which model to apply (model per regime)
# 3. Adaptive Parameters: Adjust triple-barrier parameters per regime
#
# Each regime type has configurable parameters:
# - volatility: Based on ATR percentile (low/normal/high)
# - trend: Based on ADX + SMA alignment (uptrend/downtrend/sideways)
# - structure: Based on Hurst exponent (mean_reverting/random/trending)

REGIME_CONFIG = {
    # =========================================================================
    # VOLATILITY REGIME
    # =========================================================================
    # Classifies volatility into low, normal, high based on ATR percentile
    # within a rolling lookback window.
    'volatility': {
        'enabled': True,
        'atr_period': 14,           # Period for ATR calculation
        'lookback': 100,            # Rolling window for percentile calculation
        'low_percentile': 25.0,     # Below this = low volatility
        'high_percentile': 75.0,    # Above this = high volatility
        'atr_column': 'atr_14',     # Use pre-computed ATR if available
    },
    # =========================================================================
    # TREND REGIME
    # =========================================================================
    # Classifies trend into uptrend, downtrend, sideways based on
    # ADX strength and price position relative to SMA.
    'trend': {
        'enabled': True,
        'adx_period': 14,           # Period for ADX calculation
        'sma_period': 50,           # Period for SMA calculation
        'adx_threshold': 25.0,      # ADX above this = trending market
        'adx_column': 'adx_14',     # Use pre-computed ADX if available
        'sma_column': 'sma_50',     # Use pre-computed SMA if available
    },
    # =========================================================================
    # MARKET STRUCTURE REGIME
    # =========================================================================
    # Classifies market structure based on Hurst exponent:
    # - H < 0.4: Mean-reverting (anti-persistent)
    # - 0.4 <= H <= 0.6: Random walk
    # - H > 0.6: Trending (persistent)
    'structure': {
        'enabled': True,
        'lookback': 100,            # Rolling window for Hurst calculation
        'min_lag': 2,               # Minimum lag for R/S analysis
        'max_lag': 20,              # Maximum lag for R/S analysis
        'mean_reverting_threshold': 0.4,  # H below this = mean-reverting
        'trending_threshold': 0.6,        # H above this = trending
    },
}

# =============================================================================
# REGIME-ADAPTIVE BARRIER ADJUSTMENTS
# =============================================================================
# Optional multipliers to adjust barrier parameters based on regime.
# Applied as: k_adjusted = k_base * multiplier
#
# Example: In high volatility, widen barriers to avoid noise triggers
# Example: In trending regime, tighten barriers to capture momentum

REGIME_BARRIER_ADJUSTMENTS = {
    # Volatility-based adjustments
    'volatility': {
        'low': {'k_multiplier': 0.8, 'max_bars_multiplier': 0.8},
        'normal': {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0},
        'high': {'k_multiplier': 1.3, 'max_bars_multiplier': 1.2},
    },
    # Trend-based adjustments
    'trend': {
        'uptrend': {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0},
        'downtrend': {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0},
        'sideways': {'k_multiplier': 1.2, 'max_bars_multiplier': 0.8},
    },
    # Structure-based adjustments
    'structure': {
        'mean_reverting': {'k_multiplier': 0.9, 'max_bars_multiplier': 0.7},
        'random': {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0},
        'trending': {'k_multiplier': 1.1, 'max_bars_multiplier': 1.3},
    },
}


def get_regime_adjusted_barriers(
    symbol: str,
    horizon: int,
    volatility_regime: str = 'normal',
    trend_regime: str = 'sideways',
    structure_regime: str = 'random'
) -> dict:
    """
    Get barrier parameters adjusted for current market regime.

    Applies regime-based multipliers to base barrier parameters.

    Parameters
    ----------
    symbol : str
        Symbol (e.g., 'MES', 'MGC')
    horizon : int
        Labeling horizon (e.g., 5, 20)
    volatility_regime : str
        Current volatility regime ('low', 'normal', 'high')
    trend_regime : str
        Current trend regime ('uptrend', 'downtrend', 'sideways')
    structure_regime : str
        Current structure regime ('mean_reverting', 'random', 'trending')

    Returns
    -------
    dict
        Adjusted barrier parameters with keys:
        - k_up: Adjusted upper barrier multiplier
        - k_down: Adjusted lower barrier multiplier
        - max_bars: Adjusted maximum bars
        - description: Description of adjustments applied
        - base_params: Original unadjusted parameters
        - adjustments: Dict of regime values used
    """
    # Get base parameters
    base_params = get_barrier_params(symbol, horizon)

    # Start with base values
    k_up = base_params['k_up']
    k_down = base_params['k_down']
    max_bars = base_params['max_bars']

    # Apply volatility adjustment
    vol_adj = REGIME_BARRIER_ADJUSTMENTS['volatility'].get(
        volatility_regime, {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0}
    )
    k_up *= vol_adj['k_multiplier']
    k_down *= vol_adj['k_multiplier']
    max_bars = int(max_bars * vol_adj['max_bars_multiplier'])

    # Apply trend adjustment
    trend_adj = REGIME_BARRIER_ADJUSTMENTS['trend'].get(
        trend_regime, {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0}
    )
    k_up *= trend_adj['k_multiplier']
    k_down *= trend_adj['k_multiplier']
    max_bars = int(max_bars * trend_adj['max_bars_multiplier'])

    # Apply structure adjustment
    struct_adj = REGIME_BARRIER_ADJUSTMENTS['structure'].get(
        structure_regime, {'k_multiplier': 1.0, 'max_bars_multiplier': 1.0}
    )
    k_up *= struct_adj['k_multiplier']
    k_down *= struct_adj['k_multiplier']
    max_bars = int(max_bars * struct_adj['max_bars_multiplier'])

    # Ensure max_bars stays reasonable
    max_bars = max(5, min(max_bars, 200))

    return {
        'k_up': round(k_up, 2),
        'k_down': round(k_down, 2),
        'max_bars': max_bars,
        'description': (
            f'{symbol} H{horizon}: Regime-adjusted '
            f'(vol={volatility_regime}, trend={trend_regime}, struct={structure_regime})'
        ),
        'base_params': base_params,
        'adjustments': {
            'volatility': volatility_regime,
            'trend': trend_regime,
            'structure': structure_regime
        }
    }


def get_regime_config() -> dict:
    """
    Get a copy of the regime configuration.

    Returns
    -------
    dict
        Copy of REGIME_CONFIG
    """
    import copy
    return copy.deepcopy(REGIME_CONFIG)


def validate_regime_config(config: dict = None) -> list[str]:
    """
    Validate regime configuration values.

    Parameters
    ----------
    config : dict, optional
        Regime config dict to validate. Uses REGIME_CONFIG if not provided.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    if config is None:
        config = REGIME_CONFIG

    errors = []

    # Validate volatility config
    vol_config = config.get('volatility', {})
    if vol_config.get('enabled', False):
        if vol_config.get('atr_period', 0) <= 0:
            errors.append("REGIME_CONFIG['volatility']['atr_period'] must be positive")
        if vol_config.get('lookback', 0) <= 0:
            errors.append("REGIME_CONFIG['volatility']['lookback'] must be positive")
        low_pct = vol_config.get('low_percentile', 0)
        high_pct = vol_config.get('high_percentile', 100)
        if not (0 < low_pct < high_pct < 100):
            errors.append(
                f"REGIME_CONFIG['volatility'] percentiles invalid: "
                f"low={low_pct}, high={high_pct}. Need 0 < low < high < 100"
            )

    # Validate trend config
    trend_config = config.get('trend', {})
    if trend_config.get('enabled', False):
        if trend_config.get('adx_period', 0) <= 0:
            errors.append("REGIME_CONFIG['trend']['adx_period'] must be positive")
        if trend_config.get('sma_period', 0) <= 0:
            errors.append("REGIME_CONFIG['trend']['sma_period'] must be positive")
        if trend_config.get('adx_threshold', 0) <= 0:
            errors.append("REGIME_CONFIG['trend']['adx_threshold'] must be positive")

    # Validate structure config
    struct_config = config.get('structure', {})
    if struct_config.get('enabled', False):
        if struct_config.get('lookback', 0) <= 0:
            errors.append("REGIME_CONFIG['structure']['lookback'] must be positive")
        min_lag = struct_config.get('min_lag', 0)
        max_lag = struct_config.get('max_lag', 0)
        if not (min_lag > 0 and max_lag > min_lag):
            errors.append(
                f"REGIME_CONFIG['structure'] lags invalid: "
                f"min_lag={min_lag}, max_lag={max_lag}. Need min_lag > 0 and max_lag > min_lag"
            )
        mr_thresh = struct_config.get('mean_reverting_threshold', 0.5)
        tr_thresh = struct_config.get('trending_threshold', 0.5)
        if not (0 < mr_thresh < tr_thresh < 1):
            errors.append(
                f"REGIME_CONFIG['structure'] thresholds invalid: "
                f"mean_reverting={mr_thresh}, trending={tr_thresh}. Need 0 < mr < trending < 1"
            )

    return errors
