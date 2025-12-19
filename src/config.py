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
SYMBOLS = ['MES', 'MGC']
BAR_RESOLUTION = '5min'
LOOKBACK_HORIZONS = [1, 5, 20]  # bars

# Split parameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PURGE_BARS = 20
EMBARGO_BARS = 288  # ~1 day for 5-min data
