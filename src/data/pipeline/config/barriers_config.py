"""
Barrier configuration for triple-barrier labeling.

This module contains symbol-specific and default barrier parameters
for triple-barrier labeling, including transaction costs and tick values.

SYMBOL CONFIGURATION
--------------------
The pipeline supports any symbol. Known symbols (MES, MGC) have pre-tuned
parameters. Unknown symbols use sensible defaults that can be overridden
via barrier_overrides in PipelineConfig.

ADDING NEW SYMBOLS
------------------
To add symbol-specific parameters for a new symbol, add entries to:
1. BARRIER_PARAMS - Barrier parameters per horizon
2. TRANSACTION_COSTS - Round-trip commission in ticks
3. SLIPPAGE_TICKS - Slippage per fill by volatility regime
4. TICK_VALUES - Dollar value per tick
"""

# =============================================================================
# TRANSACTION COSTS - Critical for realistic fitness evaluation
# =============================================================================
# Commission costs in ticks (entry + exit round-trip).
# Known symbols have specific values; unknown symbols use DEFAULT_TRANSACTION_COST.
DEFAULT_TRANSACTION_COST = 0.5  # Default for unknown symbols (ticks round-trip)

TRANSACTION_COSTS = {
    "MES": 0.5,  # Micro E-mini S&P 500: 0.25 per side
    "MGC": 0.3,  # Micro Gold: tighter spread
}

# =============================================================================
# SLIPPAGE COSTS - Regime-adaptive slippage modeling
# =============================================================================
# Slippage per fill (entry or exit) in ticks.
# Total slippage = 2 * SLIPPAGE_TICKS (round-trip: entry + exit)
#
# Slippage varies by volatility regime:
# - Low volatility: Tight markets, minimal slippage
# - High volatility: Wide markets, higher slippage

# Default slippage for unknown symbols
DEFAULT_SLIPPAGE_TICKS = {
    "low_vol": 0.5,  # Conservative default
    "high_vol": 1.0,  # Conservative default
}

SLIPPAGE_TICKS = {
    "MES": {
        "low_vol": 0.5,  # Calm market, tight spreads
        "high_vol": 1.0,  # Volatile market, wide spreads
    },
    "MGC": {
        "low_vol": 0.75,  # Calm market, less liquid than MES
        "high_vol": 1.5,  # Volatile market, significant slippage
    },
}

# Tick values in dollars (for P&L calculation)
DEFAULT_TICK_VALUE = 1.00  # Default for unknown symbols

TICK_VALUES = {
    "MES": 1.25,  # $1.25 per tick (micro E-mini S&P)
    "MGC": 1.00,  # $1.00 per tick (micro Gold)
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
    "MES": {
        # Horizon 5: Short-term (25 minutes) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # MES drifts UP, so upper barrier is naturally easier to hit.
        # Making k_up LARGER (harder upper barrier) balances long/short signals.
        5: {
            "k_up": 1.50,
            "k_down": 1.00,
            "max_bars": 12,
            "description": "MES H5: Asymmetric (k_up>k_down) to counteract equity drift",
        },
        # Horizon 10: Medium-short-term (50 minutes) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # Intermediate between H5 and H20 for ensemble diversity
        10: {
            "k_up": 2.00,
            "k_down": 1.40,
            "max_bars": 25,
            "description": "MES H10: Asymmetric (k_up>k_down) to counteract equity drift",
        },
        # Horizon 15: Medium-term (75 minutes) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # Intermediate between H10 and H20 for ensemble diversity
        15: {
            "k_up": 2.50,
            "k_down": 1.75,
            "max_bars": 38,
            "description": "MES H15: Asymmetric (k_up>k_down) to counteract equity drift",
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # CORRECTED (2025-12): k_up > k_down to counteract upward equity drift
        # Upper barrier harder to hit counters the structural long bias
        20: {
            "k_up": 3.00,
            "k_down": 2.10,
            "max_bars": 50,
            "description": "MES H20: Asymmetric (k_up>k_down) to counteract equity drift",
        },
    },
    # =========================================================================
    # MGC (Gold Micro Futures) - SYMMETRIC for mean-reverting asset
    # =========================================================================
    # Gold lacks the structural drift of equities. It's a store of value with
    # mean-reverting characteristics. Use SYMMETRIC barriers for unbiased signals.
    # WIDER barriers to achieve 20-30% neutral rate.
    "MGC": {
        # Horizon 5: Short-term (25 minutes) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Wide barriers (1.20/1.20) for ~25% neutral
        5: {
            "k_up": 1.20,
            "k_down": 1.20,
            "max_bars": 12,
            "description": "MGC H5: Symmetric barriers, ~25% neutral target",
        },
        # Horizon 10: Medium-short-term (50 minutes) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Intermediate between H5 and H20 for ensemble diversity
        10: {
            "k_up": 1.60,
            "k_down": 1.60,
            "max_bars": 25,
            "description": "MGC H10: Symmetric barriers, ~25% neutral target",
        },
        # Horizon 15: Medium-term (75 minutes) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Intermediate between H10 and H20 for ensemble diversity
        15: {
            "k_up": 2.00,
            "k_down": 2.00,
            "max_bars": 38,
            "description": "MGC H15: Symmetric barriers, ~25% neutral target",
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # Symmetric: equal probability of hitting upper/lower
        # Wide barriers (2.50/2.50) for ~25% neutral
        20: {
            "k_up": 2.50,
            "k_down": 2.50,
            "max_bars": 50,
            "description": "MGC H20: Symmetric barriers, ~25% neutral target",
        },
    },
}

# =============================================================================
# LEGACY BARRIER PARAMS (for backward compatibility)
# =============================================================================
# Default parameters used when symbol-specific not available
# Also used for H1 horizon (excluded from active trading)
BARRIER_PARAMS_DEFAULT = {
    1: {
        "k_up": 0.30,
        "k_down": 0.30,
        "max_bars": 4,
        "description": "H1: Ultra-short, NON-VIABLE after transaction costs",
    },
    5: {
        "k_up": 1.35,
        "k_down": 1.10,
        "max_bars": 12,
        "description": "H5: Default asymmetric, ~25% neutral target",
    },
    10: {
        "k_up": 1.80,
        "k_down": 1.35,
        "max_bars": 25,
        "description": "H10: Default asymmetric, ~25% neutral target",
    },
    15: {
        "k_up": 2.25,
        "k_down": 1.70,
        "max_bars": 38,
        "description": "H15: Default asymmetric, ~25% neutral target",
    },
    20: {
        "k_up": 2.75,
        "k_down": 2.30,
        "max_bars": 50,
        "description": "H20: Default asymmetric, ~25% neutral target",
    },
}

# Alternative: Percentage-based barriers (ATR-independent)
# Useful when ATR calculation is inconsistent across instruments
# NOTE: Also uses asymmetric barriers to correct long bias (pct_up > pct_down)
PERCENTAGE_BARRIER_PARAMS = {
    1: {"pct_up": 0.0015, "pct_down": 0.0015, "max_bars": 5},  # 0.15% symmetric (H1 not traded)
    5: {"pct_up": 0.0030, "pct_down": 0.0020, "max_bars": 15},  # 0.30%/0.20% asymmetric
    20: {"pct_up": 0.0060, "pct_down": 0.0042, "max_bars": 60},  # 0.60%/0.42% asymmetric
}


def get_barrier_params(symbol: str, horizon: int, get_default_for_horizon: callable = None) -> dict:
    """
    Get barrier parameters for a specific symbol and horizon.

    This function supports any symbol. Known symbols (MES, MGC) have pre-tuned
    parameters. Unknown symbols use sensible defaults.

    Parameters
    ----------
    symbol : str
        Symbol name (any valid symbol)
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
    4. Sensible fallback defaults based on horizon
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
        "k_up": round(k_base, 2),
        "k_down": round(k_base, 2),
        "max_bars": max(5, horizon * 3),
        "description": f"H{horizon}: Auto-generated default",
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
            if params.get("k_up", 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_up'] must be positive")
            if params.get("k_down", 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_down'] must be positive")
            if params.get("max_bars", 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['max_bars'] must be positive")

    # Validate default barrier params
    for horizon, params in BARRIER_PARAMS_DEFAULT.items():
        if params.get("k_up", 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_up'] must be positive")
        if params.get("k_down", 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_down'] must be positive")
        if params.get("max_bars", 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['max_bars'] must be positive")

    # Validate transaction costs
    for symbol, cost in TRANSACTION_COSTS.items():
        if cost < 0:
            errors.append(f"TRANSACTION_COSTS['{symbol}'] must be non-negative, got {cost}")

    # Validate slippage costs
    for symbol, regimes in SLIPPAGE_TICKS.items():
        if not isinstance(regimes, dict):
            errors.append(
                f"SLIPPAGE_TICKS['{symbol}'] must be a dict with 'low_vol' and 'high_vol' keys"
            )
            continue

        for regime in ("low_vol", "high_vol"):
            if regime not in regimes:
                errors.append(f"SLIPPAGE_TICKS['{symbol}'] missing '{regime}' regime")
            elif regimes[regime] < 0:
                errors.append(
                    f"SLIPPAGE_TICKS['{symbol}']['{regime}'] must be non-negative, got {regimes[regime]}"
                )

        # High volatility slippage should be >= low volatility slippage
        if "low_vol" in regimes and "high_vol" in regimes:
            if regimes["high_vol"] < regimes["low_vol"]:
                errors.append(
                    f"SLIPPAGE_TICKS['{symbol}']['high_vol'] ({regimes['high_vol']}) should be >= "
                    f"'low_vol' ({regimes['low_vol']})"
                )

    # Validate tick values
    for symbol, value in TICK_VALUES.items():
        if value <= 0:
            errors.append(f"TICK_VALUES['{symbol}'] must be positive, got {value}")

    return errors


def get_slippage_ticks(symbol: str, regime: str = "low_vol") -> float:
    """
    Get slippage estimate in ticks for a symbol and volatility regime.

    Parameters
    ----------
    symbol : str
        Symbol name (any valid symbol)
    regime : str, optional
        Volatility regime: 'low_vol' or 'high_vol' (default: 'low_vol')

    Returns
    -------
    float
        Slippage in ticks per fill (entry or exit)

    Notes
    -----
    Total round-trip slippage = 2 * get_slippage_ticks() (entry + exit)
    Unknown symbols use DEFAULT_SLIPPAGE_TICKS.
    """
    if regime not in ("low_vol", "high_vol"):
        # Default to low_vol if regime invalid
        regime = "low_vol"

    if symbol in SLIPPAGE_TICKS:
        return SLIPPAGE_TICKS[symbol][regime]

    # Use defaults for unknown symbols
    return DEFAULT_SLIPPAGE_TICKS[regime]


def get_total_trade_cost(
    symbol: str, regime: str = "low_vol", include_slippage: bool = True
) -> float:
    """
    Calculate total round-trip trade cost in ticks (commission + slippage).

    Parameters
    ----------
    symbol : str
        Symbol name (any valid symbol)
    regime : str, optional
        Volatility regime: 'low_vol' or 'high_vol' (default: 'low_vol')
    include_slippage : bool, optional
        Whether to include slippage in calculation (default: True)

    Returns
    -------
    float
        Total round-trip cost in ticks (entry + exit)

    Notes
    -----
    - Commission is round-trip (entry + exit)
    - Slippage applies to both entry and exit (2x per-fill slippage)
    - Total cost = commission + 2 * slippage_per_fill
    - Unknown symbols use DEFAULT_TRANSACTION_COST and DEFAULT_SLIPPAGE_TICKS
    """
    commission = TRANSACTION_COSTS.get(symbol, DEFAULT_TRANSACTION_COST)

    if not include_slippage:
        return commission

    slippage_per_fill = get_slippage_ticks(symbol, regime)
    # Round-trip slippage = entry slippage + exit slippage
    total_slippage = 2 * slippage_per_fill

    return commission + total_slippage


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
            mb = params.get("max_bars", 0)
            if mb > max_max_bars:
                max_max_bars = mb
                max_bars_source = f"BARRIER_PARAMS['{symbol}'][{horizon}]"

    # Check default barrier params
    for horizon, params in BARRIER_PARAMS_DEFAULT.items():
        mb = params.get("max_bars", 0)
        if mb > max_max_bars:
            max_max_bars = mb
            max_bars_source = f"BARRIER_PARAMS_DEFAULT[{horizon}]"

    # Check percentage barrier params
    for horizon, params in PERCENTAGE_BARRIER_PARAMS.items():
        mb = params.get("max_bars", 0)
        if mb > max_max_bars:
            max_max_bars = mb
            max_bars_source = f"PERCENTAGE_BARRIER_PARAMS[{horizon}]"

    return max_max_bars, max_bars_source


def get_tick_value(symbol: str) -> float:
    """
    Get the tick value in dollars for a symbol.

    Parameters
    ----------
    symbol : str
        Symbol name (any valid symbol)

    Returns
    -------
    float
        Tick value in dollars

    Notes
    -----
    Unknown symbols use DEFAULT_TICK_VALUE.
    """
    return TICK_VALUES.get(symbol, DEFAULT_TICK_VALUE)


def get_transaction_cost(symbol: str) -> float:
    """
    Get the round-trip transaction cost in ticks for a symbol.

    Parameters
    ----------
    symbol : str
        Symbol name (any valid symbol)

    Returns
    -------
    float
        Transaction cost in ticks (round-trip)

    Notes
    -----
    Unknown symbols use DEFAULT_TRANSACTION_COST.
    """
    return TRANSACTION_COSTS.get(symbol, DEFAULT_TRANSACTION_COST)
