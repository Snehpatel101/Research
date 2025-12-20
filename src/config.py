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
SYMBOLS = ['MGC']  # Focus on gold only
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
# BARRIER CONFIGURATION - CALIBRATED FOR BALANCED LABELS
# =============================================================================
# These parameters control triple-barrier label distribution.
# CRITICAL: Previous defaults (k=1.0-2.0, max_bars=horizon) were too wide,
# causing 99%+ neutral labels. These calibrated values target ~35/30/35 distribution.
#
# Key principles:
# 1. k_up = k_down for symmetric (balanced) predictions
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
    1: {
        'k_up': 0.25,
        'k_down': 0.25,
        'max_bars': 5,
        'description': 'Ultra-short: tight barriers, NON-VIABLE after transaction costs'
    },
    # -------------------------------------------------------------------------
    # Horizon 5: Short-term (25 minutes) - ACTIVE
    # -------------------------------------------------------------------------
    # Empirically calibrated: k=0.90 produces ~30-35% neutral labels.
    # Previous k=0.5 was too tight, causing excessive directional signals.
    # max_bars=15 gives sufficient time window (75 min) for barriers to hit.
    5: {
        'k_up': 0.90,
        'k_down': 0.90,
        'max_bars': 15,
        'description': 'Short-term: empirically calibrated for 30-35% neutral rate'
    },
    # -------------------------------------------------------------------------
    # Horizon 20: Medium-term (~1.5 hours) - ACTIVE
    # -------------------------------------------------------------------------
    # Empirically calibrated: k=2.0 produces ~30-35% neutral labels.
    # Previous k=0.75 was too tight, causing excessive directional signals.
    # max_bars=60 is CRITICAL: PURGE_BARS must equal this value to prevent leakage.
    20: {
        'k_up': 2.00,
        'k_down': 2.00,
        'max_bars': 60,
        'description': 'Medium-term: empirically calibrated for 30-35% neutral rate'
    }
}

# Alternative: Percentage-based barriers (ATR-independent)
# Useful when ATR calculation is inconsistent across instruments
PERCENTAGE_BARRIER_PARAMS = {
    1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15%
    5: {'pct_up': 0.0025, 'pct_down': 0.0025, 'max_bars': 15},  # 0.25%
    20: {'pct_up': 0.0050, 'pct_down': 0.0050, 'max_bars': 60}  # 0.50%
}
