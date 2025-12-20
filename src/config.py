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
PURGE_BARS = 20
EMBARGO_BARS = 288  # ~1 day for 5-min data

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
    # Horizon 1: Ultra-short term (5 minutes)
    # 0.3 ATR barrier, 5-bar (25 min) window
    1: {
        'k_up': 0.3,
        'k_down': 0.3,
        'max_bars': 5,
        'description': 'Ultra-short: tight barriers for quick signals'
    },
    # Horizon 5: Short-term (25 minutes)
    # 0.5 ATR barrier, 15-bar (75 min) window
    5: {
        'k_up': 0.5,
        'k_down': 0.5,
        'max_bars': 15,
        'description': 'Short-term: moderate barriers, extended window'
    },
    # Horizon 20: Medium-term (~1.5 hours)
    # 0.75 ATR barrier, 60-bar (5 hour) window
    20: {
        'k_up': 0.75,
        'k_down': 0.75,
        'max_bars': 60,
        'description': 'Medium-term: wider barriers, long window'
    }
}

# Alternative: Percentage-based barriers (ATR-independent)
# Useful when ATR calculation is inconsistent across instruments
PERCENTAGE_BARRIER_PARAMS = {
    1: {'pct_up': 0.0015, 'pct_down': 0.0015, 'max_bars': 5},   # 0.15%
    5: {'pct_up': 0.0025, 'pct_down': 0.0025, 'max_bars': 15},  # 0.25%
    20: {'pct_up': 0.0050, 'pct_down': 0.0050, 'max_bars': 60}  # 0.50%
}
