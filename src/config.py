"""
Configuration for ensemble trading project
"""
from pathlib import Path

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across the entire pipeline.

    This function sets seeds for:
    - Python's random module
    - NumPy's random module
    - PyTorch (if available)
    - TensorFlow (if available)

    Call this at the start of any pipeline run to ensure reproducible results.

    Parameters:
    -----------
    seed : int
        Random seed value (default: RANDOM_SEED = 42)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Try to set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Try to set TensorFlow seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
SPLITS_DIR = DATA_DIR / "splits"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
ENSEMBLE_MODELS_DIR = MODELS_DIR / "ensemble"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, CLEAN_DATA_DIR, FEATURES_DIR, PROCESSED_DATA_DIR,
                 FINAL_DATA_DIR, SPLITS_DIR, BASE_MODELS_DIR, ENSEMBLE_MODELS_DIR,
                 RESULTS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Trading parameters
SYMBOLS = ['MES', 'MGC']  # MES (S&P 500 futures) and MGC (Gold futures) for cross-asset analysis
BAR_RESOLUTION = '5min'
LOOKBACK_HORIZONS = [1, 5, 20]  # bars

# Split parameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# PURGE AND EMBARGO CONFIGURATION - CRITICAL FOR LEAKAGE PREVENTION
# =============================================================================
# PURGE_BARS: Number of bars to remove at split boundaries to prevent look-ahead bias.
# CRITICAL: Must equal max(max_bars) across all horizons to fully prevent leakage.
# H20 uses max_bars=60, therefore PURGE_BARS must be at least 60.
# Previous value of 20 was INSUFFICIENT and allowed label leakage from future data.
PURGE_BARS = 60  # = max_bars for H20 (CRITICAL: prevents leakage)

# EMBARGO_BARS: Buffer between splits to account for serial correlation in features.
# 288 bars = 1 day for 5-min data (reasonable for daily feature decay).
EMBARGO_BARS = 288  # ~1 day for 5-min data

# =============================================================================
# ACTIVE HORIZONS - EXCLUDING NON-VIABLE HORIZONS
# =============================================================================
# H1 (5-minute horizon) is NOT viable for production trading because:
# 1. Transaction costs (~0.5 ticks round-trip) exceed expected profit
# 2. Typical H1 profit target is ~0.25 ATR = 1-2 ticks for MGC
# 3. After transaction costs, expected value is negative
# Use ACTIVE_HORIZONS for model training/evaluation; keep LOOKBACK_HORIZONS for labeling.
ACTIVE_HORIZONS = [5, 20]  # H1 excluded due to transaction cost impact

# =============================================================================
# BARRIER CONFIGURATION - SYMBOL-SPECIFIC FOR BALANCED DISTRIBUTION
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
# 1. MES: k_up > k_down to counteract bullish market bias
# 2. MGC: k_up = k_down for unbiased mean-reverting asset
# 3. Wider k values = more neutrals, more selective signals
# 4. max_bars reduction = more timeouts = more neutrals

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
BARRIER_PARAMS = {
    # =========================================================================
    # MES (S&P 500 Micro Futures) - ASYMMETRIC for equity drift
    # =========================================================================
    # MES has structural upward drift from equity risk premium (~7% annually).
    # Use asymmetric barriers: k_up > k_down to counteract long bias.
    # WIDER barriers to achieve 20-30% neutral rate.
    'MES': {
        # Horizon 5: Short-term (25 minutes) - ACTIVE
        # Asymmetric: upper barrier 47% harder to hit than lower
        # Wider barriers (1.50/1.00) for ~25% neutral vs old (1.10/0.75) <2%
        5: {
            'k_up': 1.50,
            'k_down': 1.00,
            'max_bars': 12,
            'description': 'MES H5: Asymmetric (k_up>k_down), ~25% neutral target'
        },
        # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
        # Asymmetric: upper barrier 40% harder to hit than lower
        # Wider barriers (3.00/2.10) for ~25% neutral
        20: {
            'k_up': 3.00,
            'k_down': 2.10,
            'max_bars': 50,
            'description': 'MES H20: Asymmetric (k_up>k_down), ~25% neutral target'
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


def get_barrier_params(symbol: str, horizon: int) -> dict:
    """
    Get barrier parameters for a specific symbol and horizon.

    Parameters:
    -----------
    symbol : str - 'MES' or 'MGC'
    horizon : int - 1, 5, or 20

    Returns:
    --------
    dict with 'k_up', 'k_down', 'max_bars', 'description'
    """
    # Check symbol-specific params first
    if symbol in BARRIER_PARAMS and horizon in BARRIER_PARAMS[symbol]:
        return BARRIER_PARAMS[symbol][horizon]

    # Fall back to default params
    if horizon in BARRIER_PARAMS_DEFAULT:
        return BARRIER_PARAMS_DEFAULT[horizon]

    # Ultimate fallback
    return {
        'k_up': 1.0,
        'k_down': 1.0,
        'max_bars': horizon * 3,
        'description': 'Fallback: symmetric barriers'
    }

# Alternative: Percentage-based barriers (ATR-independent)
# Useful when ATR calculation is inconsistent across instruments
# NOTE: Also uses asymmetric barriers to correct long bias (pct_up > pct_down)
PERCENTAGE_BARRIER_PARAMS = {
    1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15% symmetric (H1 not traded)
    5: {'pct_up': 0.0030, 'pct_down': 0.0020, 'max_bars': 15},  # 0.30%/0.20% asymmetric
    20: {'pct_up': 0.0060, 'pct_down': 0.0042, 'max_bars': 60}  # 0.60%/0.42% asymmetric
}

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================
# CORRELATION_THRESHOLD: Maximum allowed correlation between features.
# Features with correlation above this threshold will be removed (keeping the
# most interpretable feature from each correlated group).
#
# CRITICAL: The previous default of 0.95 was too lenient. Features with 0.80+
# correlation still cause multicollinearity issues during ML training.
# Lowering to 0.85 provides more aggressive pruning which is desirable.
#
# Lower values = more aggressive pruning = fewer features = less multicollinearity
# Higher values = less pruning = more features = potential multicollinearity
CORRELATION_THRESHOLD = 0.85

# VARIANCE_THRESHOLD: Minimum variance for a feature to be retained.
# Features with variance below this threshold are considered near-constant
# and provide no discriminative power.
VARIANCE_THRESHOLD = 0.01

# =============================================================================
# CROSS-ASSET FEATURE CONFIGURATION
# =============================================================================
# Cross-asset features capture relationships between MES (S&P 500 futures) and
# MGC (Gold futures), which often exhibit interesting correlation dynamics:
# - Risk-on/risk-off regimes: MES up, MGC down (and vice versa)
# - Flight to safety: MES down, MGC up during market stress
# - Inflation hedging: Both assets may move together during inflation concerns
#
# These features are computed when both symbols are present in the data.

CROSS_ASSET_FEATURES = {
    'enabled': True,
    'symbols': ['MES', 'MGC'],  # Symbol pair for cross-asset features
    'features': {
        'mes_mgc_correlation_20': {
            'description': '20-bar rolling correlation between MES and MGC returns',
            'lookback': 20
        },
        'mes_mgc_spread_zscore': {
            'description': 'Z-score of spread between normalized MES and MGC prices',
            'lookback': 20
        },
        'mes_mgc_beta': {
            'description': 'Rolling beta of MES returns vs MGC returns',
            'lookback': 20
        },
        'relative_strength': {
            'description': 'MES return minus MGC return (momentum divergence)',
            'lookback': 20
        }
    }
}


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> None:
    """
    Validate configuration values for consistency and correctness.

    Raises:
        ValueError: If configuration values are invalid or inconsistent

    This function ensures:
    1. PURGE_BARS >= max(max_bars) across all horizons (prevents label leakage)
    2. Split ratios sum to 1.0
    3. All barrier parameters are positive
    4. Transaction costs are non-negative
    5. Tick values are positive
    """
    import numpy as np

    errors = []

    # === Validate purge_bars >= max_bars (CRITICAL for leakage prevention) ===
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

    if PURGE_BARS < max_max_bars:
        errors.append(
            f"PURGE_BARS ({PURGE_BARS}) must be >= max_bars ({max_max_bars}) from {max_bars_source} "
            f"to prevent label leakage. Set PURGE_BARS >= {max_max_bars}."
        )

    # === Validate split ratios sum to 1.0 ===
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not np.isclose(total_ratio, 1.0):
        errors.append(
            f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO})"
        )

    # === Validate individual split ratios are positive ===
    if TRAIN_RATIO <= 0:
        errors.append(f"TRAIN_RATIO must be positive, got {TRAIN_RATIO}")
    if VAL_RATIO <= 0:
        errors.append(f"VAL_RATIO must be positive, got {VAL_RATIO}")
    if TEST_RATIO <= 0:
        errors.append(f"TEST_RATIO must be positive, got {TEST_RATIO}")

    # === Validate purge and embargo bars are non-negative ===
    if PURGE_BARS < 0:
        errors.append(f"PURGE_BARS must be non-negative, got {PURGE_BARS}")
    if EMBARGO_BARS < 0:
        errors.append(f"EMBARGO_BARS must be non-negative, got {EMBARGO_BARS}")

    # === Validate barrier parameters are positive ===
    for symbol, horizons in BARRIER_PARAMS.items():
        for horizon, params in horizons.items():
            if params.get('k_up', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_up'] must be positive")
            if params.get('k_down', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['k_down'] must be positive")
            if params.get('max_bars', 0) <= 0:
                errors.append(f"BARRIER_PARAMS['{symbol}'][{horizon}]['max_bars'] must be positive")

    for horizon, params in BARRIER_PARAMS_DEFAULT.items():
        if params.get('k_up', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_up'] must be positive")
        if params.get('k_down', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['k_down'] must be positive")
        if params.get('max_bars', 0) <= 0:
            errors.append(f"BARRIER_PARAMS_DEFAULT[{horizon}]['max_bars'] must be positive")

    # === Validate transaction costs are non-negative ===
    for symbol, cost in TRANSACTION_COSTS.items():
        if cost < 0:
            errors.append(f"TRANSACTION_COSTS['{symbol}'] must be non-negative, got {cost}")

    # === Validate tick values are positive ===
    for symbol, value in TICK_VALUES.items():
        if value <= 0:
            errors.append(f"TICK_VALUES['{symbol}'] must be positive, got {value}")

    # === Validate correlation and variance thresholds ===
    if not (0 < CORRELATION_THRESHOLD <= 1.0):
        errors.append(f"CORRELATION_THRESHOLD must be in (0, 1.0], got {CORRELATION_THRESHOLD}")
    if VARIANCE_THRESHOLD < 0:
        errors.append(f"VARIANCE_THRESHOLD must be non-negative, got {VARIANCE_THRESHOLD}")

    # === Raise all errors at once for comprehensive feedback ===
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)


# Run validation at module import time
validate_config()
