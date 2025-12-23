"""
Barrier configuration for triple-barrier labeling.

This module contains symbol-specific and default barrier parameters
for triple-barrier labeling, including transaction costs and tick values.
"""

# =============================================================================
# TRANSACTION COSTS - Critical for realistic fitness evaluation
# =============================================================================
# Round-trip costs in ticks (entry + exit):
# - MES: ~0.5 ticks (1 tick spread + commission, 0.25 per side)
# - MGC: ~0.3 ticks (tighter spread on gold micro)
TRANSACTION_COSTS = {
    'MES': 0.5,  # ticks round-trip
    'MGC': 0.3   # ticks round-trip
}

# Tick values in dollars (for P&L calculation)
TICK_VALUES = {
    'MES': 1.25,  # $1.25 per tick (micro E-mini S&P)
    'MGC': 1.00   # $1.00 per tick (micro Gold)
}

# =============================================================================
# SYMBOL-SPECIFIC BARRIER PARAMETERS
# =============================================================================
# These parameters control triple-barrier label distribution.
#
# CRITICAL FIX (2024-12): SYMBOL-SPECIFIC BARRIERS
# -----------------------------------------------------------------------------
# PROBLEM 1: Previous neutral rate was <2% - models predict every bar, creating
# excessive trading. Target neutral rate: 20-30% for realistic trading.
#
# PROBLEM 2: Same asymmetric barriers for all symbols ignores market structure.
# MES (S&P 500 futures) has structural upward drift from equity risk premium.
# MGC (Gold futures) does NOT have this drift - gold is a store of value.
#
# SOLUTION:
# - MES: ASYMMETRIC barriers (k_up > k_down) to counteract equity drift
# - MGC: SYMMETRIC barriers (k_up = k_down) since gold lacks directional bias
# - WIDER barriers across both to achieve 20-30% neutral rate
#
# Key principles:
# 1. MES: k_up > k_down to counteract bullish market bias (upper barrier harder)
# 2. MGC: k_up = k_down for unbiased mean-reverting asset
# 3. Wider k values = more neutrals, more selective signals
# 4. max_bars reduction = more timeouts = more neutrals
#
# MATHEMATICAL REASONING (2025-12 correction):
# - MES drifts UP due to equity risk premium (~7% annually)
# - This makes the UPPER barrier naturally EASIER to hit
# - To balance long/short signals, make UPPER barrier HARDER (k_up > k_down)
# - Previous config had k_down > k_up which AMPLIFIED long bias (wrong direction)

BARRIER_PARAMS = {
    # =========================================================================
    # MES (S&P 500 Micro Futures) - ASYMMETRIC for equity drift
    # =========================================================================
    # MES has structural upward drift from equity risk premium (~7% annually).
    # Use asymmetric barriers: k_up > k_down to counteract long bias.
    # Making the UPPER barrier harder to hit reduces the number of long signals,
    # which counterbalances the inherent upward drift of equities.
    # WIDER barriers to achieve 20-30% neutral rate.
    'MES': {
        # Horizon 5: Short-term (25 minutes) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # MES drifts UP, so upper barrier is naturally easier to hit.
        # Making k_up LARGER (harder upper barrier) balances long/short signals.
        5: {
            'k_up': 1.50,
            'k_down': 1.00,
            'max_bars': 12,
            'description': 'MES H5: Asymmetric (k_up>k_down) to counteract equity drift'
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # Upper barrier harder to hit counters the structural long bias
        20: {
            'k_up': 3.00,
            'k_down': 2.10,
            'max_bars': 50,
            'description': 'MES H20: Asymmetric (k_up>k_down) to counteract equity drift'
        }
    },
    # =========================================================================
    # MGC (Gold Micro Futures) - SYMMETRIC for mean-reverting asset
    # =========================================================================
    # Gold lacks the structural drift of equities. It's a store of value with
    # mean-reverting characteristics. Use SYMMETRIC barriers for unbiased signals.
    # WIDER barriers to achieve 20-30% neutral rate.
    'MGC': {
        # Horizon 5: Short-term (25 minutes) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Wide barriers (1.20/1.20) for ~25% neutral
        5: {
            'k_up': 1.20,
            'k_down': 1.20,
            'max_bars': 12,
            'description': 'MGC H5: Symmetric barriers, ~25% neutral target'
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Wide barriers (2.50/2.50) for ~25% neutral
        20: {
            'k_up': 2.50,
            'k_down': 2.50,
            'max_bars': 50,
            'description': 'MGC H20: Symmetric barriers, ~25% neutral target'
        }
    }
}

# =============================================================================
# LEGACY BARRIER PARAMS (for backward compatibility)
# =============================================================================
# Default parameters used when symbol-specific not available
# Also used for H1 horizon (excluded from active trading)
BARRIER_PARAMS_DEFAULT = {
    1: {
        'k_up': 0.30,
        'k_down': 0.30,
        'max_bars': 4,
        'description': 'H1: Ultra-short, NON-VIABLE after transaction costs'
    },
    5: {
        'k_up': 1.35,
        'k_down': 1.10,
        'max_bars': 12,
        'description': 'H5: Default asymmetric, ~25% neutral target'
    },
    20: {
        'k_up': 2.75,
        'k_down': 2.30,
        'max_bars': 50,
        'description': 'H20: Default asymmetric, ~25% neutral target'
    }
}

# Alternative: Percentage-based barriers (ATR-independent)
# Useful when ATR calculation is inconsistent across instruments
# NOTE: Also uses asymmetric barriers to correct long bias (pct_up > pct_down)
PERCENTAGE_BARRIER_PARAMS = {
    1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15% symmetric (H1 not traded)
    5: {'pct_up': 0.0030, 'pct_down': 0.0020, 'max_bars': 15},  # 0.30%/0.20% asymmetric
    20: {'pct_up': 0.0060, 'pct_down': 0.0042, 'max_bars': 60}  # 0.60%/0.42% asymmetric
}


def get_barrier_params(
    symbol: str,
    horizon: int,
    get_default_for_horizon: callable = None
) -> dict:
    """
    Get barrier parameters for a specific symbol and horizon.

    This function supports dynamic horizons by generating defaults for horizons
    not explicitly defined in BARRIER_PARAMS or BARRIER_PARAMS_DEFAULT.

    Parameters
    ----------
    symbol : str
        Symbol name ('MES' or 'MGC')
    horizon : int
        Horizon value (any value in SUPPORTED_HORIZONS)
    get_default_for_horizon : callable, optional
        Function to generate default params for non-standard horizons.
        If not provided, falls back to BARRIER_PARAMS_DEFAULT only.

    Returns
    -------
    dict
        Dict with 'k_up', 'k_down', 'max_bars', 'description'

    Notes
    -----
    Priority order:
    1. Symbol-specific params (BARRIER_PARAMS[symbol][horizon])
    2. Default params (BARRIER_PARAMS_DEFAULT[horizon])
    3. Auto-generated defaults (get_default_for_horizon if provided)
    """
    # Check symbol-specific params first
    if symbol in BARRIER_PARAMS and horizon in BARRIER_PARAMS[symbol]:
        return BARRIER_PARAMS[symbol][horizon]

    # Fall back to default params
    if horizon in BARRIER_PARAMS_DEFAULT:
        return BARRIER_PARAMS_DEFAULT[horizon]

    # Auto-generate defaults for non-standard horizons if callback provided
    if get_default_for_horizon is not None:
        return get_default_for_horizon(horizon)

    # Ultimate fallback - generate sensible defaults based on horizon
    k_base = 1.0 + (horizon / 20.0) * 1.5  # Scale k with horizon
    return {
        'k_up': round(k_base, 2),
        'k_down': round(k_base, 2),
        'max_bars': max(5, horizon * 3),
        'description': f'H{horizon}: Auto-generated default'
    }


def validate_barrier_params() -> list[str]:
    """
    Validate all barrier parameter configurations.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Validate symbol-specific barrier params
    for symbol, horizons in BARRIER_PARAMS.items():
        for horizon, params in horizons.items():
            if params.get('k_up', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_up'] must be positive")
            if params.get('k_down', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_down'] must be positive")
            if params.get('max_bars', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['max_bars'] must be positive")

    # Validate default barrier params
    for horizon, params in BARRIER_PARAMS_DEFAULT.items():
        if params.get('k_up', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_up'] must be positive")
        if params.get('k_down', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_down'] must be positive")
        if params.get('max_bars', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['max_bars'] must be positive")

    # Validate transaction costs
    for symbol, cost in TRANSACTION_COSTS.items():
        if cost < 0:
            errors.append(f"TRANSACTION_COSTS['{symbol}'] must be non-negative, got {cost}")

    # Validate tick values
    for symbol, value in TICK_VALUES.items():
        if value <= 0:
            errors.append(f"TICK_VALUES['{symbol}'] must be positive, got {value}")

    return errors


def get_max_bars_across_all_params() -> tuple[int, str]:
    """
    Get the maximum max_bars value across all barrier configurations.

    Returns
    -------
    tuple[int, str]
        (max_max_bars, source) - The maximum value and its source location
    """
    max_max_bars = 0
    max_bars_source = None

    # Check symbol-specific barrier params
    for symbol, horizons in BARRIER_PARAMS.items():
        for horizon, params in horizons.items():
            mb = params.get('max_bars', 0)
            if mb > max_max_bars:
                max_max_bars = mb
                max_bars_source = f"BARRIER_PARAMS['{symbol}'][{horizon}]"

    # Check default barrier params
    for horizon, params in BARRIER_PARAMS_DEFAULT.items():
        mb = params.get('max_bars', 0)
        if mb > max_max_bars:
            max_max_bars = mb
            max_bars_source = f"BARRIER_PARAMS_DEFAULT[{horizon}]"

    # Check percentage barrier params
    for horizon, params in PERCENTAGE_BARRIER_PARAMS.items():
        mb = params.get('max_bars', 0)
        if mb > max_max_bars:
            max_max_bars = mb
            max_bars_source = f"PERCENTAGE_BARRIER_PARAMS[{horizon}]"

    return max_max_bars, max_bars_source
