"""
Configuration for ensemble trading project
"""
from pathlib import Path

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
# BARRIER CONFIGURATION - ASYMMETRIC FOR BALANCED LONG/SHORT DISTRIBUTION
# =============================================================================
# These parameters control triple-barrier label distribution.
#
# CRITICAL FIX (2024-12): ASYMMETRIC BARRIERS TO CORRECT LONG BIAS
# -----------------------------------------------------------------------------
# PROBLEM: Symmetric barriers (k_up = k_down) in a historically bullish market
# (2008-2025) produce 87-91% long signals, creating dangerous asymmetric risk
# in bear markets. The model becomes overconfident in longs and underweights
# short opportunities.
#
# SOLUTION: Asymmetric barriers that make the lower barrier EASIER to hit:
# - k_up > k_down: Upper barrier is farther away (harder to hit)
# - k_down < k_up: Lower barrier is closer (easier to hit)
#
# This compensates for the structural upward drift in equity/commodity markets,
# producing closer to 50/50 long/short distribution instead of 87/13.
#
# RATIONALE BY HORIZON:
# - H5:  k_up=1.10, k_down=0.75 (ratio 1.47:1) - short-term mean reversion
# - H20: k_up=2.40, k_down=1.70 (ratio 1.41:1) - medium-term trend following
#
# The asymmetry ratio (~1.4-1.5x) was empirically calibrated to achieve
# approximately balanced label distribution given the historical upward drift.
#
# Key principles:
# 1. k_up > k_down to counteract bullish market bias
# 2. max_bars should be 3-5x horizon for barriers to realistically hit
# 3. Smaller k values = more directional signals, fewer neutrals
#
# Tune these based on your instrument's volatility:
# - Higher volatility: can use larger k values
# - Lower volatility: use smaller k values

BARRIER_PARAMS = {
    # -------------------------------------------------------------------------
    # Horizon 1: Ultra-short term (5 minutes) - EXCLUDED FROM ACTIVE TRADING
    # -------------------------------------------------------------------------
    # NOTE: H1 is NOT in ACTIVE_HORIZONS due to transaction cost impact.
    # Kept for labeling completeness but should NOT be used for production models.
    # Transaction costs (~0.5 ticks) exceed typical H1 profit (1-2 ticks).
    # Symmetric barriers are acceptable here since H1 is not used for trading.
    1: {
        'k_up': 0.25,
        'k_down': 0.25,
        'max_bars': 5,
        'description': 'Ultra-short: tight barriers, NON-VIABLE after transaction costs'
    },
    # -------------------------------------------------------------------------
    # Horizon 5: Short-term (25 minutes) - ACTIVE, ASYMMETRIC
    # -------------------------------------------------------------------------
    # ASYMMETRIC BARRIERS to correct long bias in bullish markets.
    # k_up=1.10, k_down=0.75 makes lower barrier 47% easier to hit than upper.
    # This counteracts the structural upward drift that caused 87%+ long labels.
    # max_bars=15 gives sufficient time window (75 min) for barriers to hit.
    5: {
        'k_up': 1.10,
        'k_down': 0.75,
        'max_bars': 15,
        'description': 'Short-term: ASYMMETRIC barriers (k_up>k_down) to reduce long bias'
    },
    # -------------------------------------------------------------------------
    # Horizon 20: Medium-term (~1.5 hours) - ACTIVE, ASYMMETRIC
    # -------------------------------------------------------------------------
    # ASYMMETRIC BARRIERS to correct long bias in bullish markets.
    # k_up=2.40, k_down=1.70 makes lower barrier 41% easier to hit than upper.
    # This counteracts the structural upward drift that caused 87%+ long labels.
    # max_bars=60 is CRITICAL: PURGE_BARS must equal this value to prevent leakage.
    20: {
        'k_up': 2.40,
        'k_down': 1.70,
        'max_bars': 60,
        'description': 'Medium-term: ASYMMETRIC barriers (k_up>k_down) to reduce long bias'
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
