"""
Threshold constants for validation and filtering.

These constants define sensible defaults for data quality checks,
signal detection, and validation thresholds.

Usage:
    from src.config.constants.thresholds import LEAKAGE_CORRELATION_THRESHOLD

    if correlation > LEAKAGE_CORRELATION_THRESHOLD:
        raise LeakageDetectedError(f"Correlation {correlation:.3f} exceeds threshold")
"""

# =============================================================================
# SIGNAL THRESHOLDS
# =============================================================================
MIN_SIGNAL_RATIO = 0.10  # Minimum ratio of signals to total samples
DEFAULT_CONFIDENCE = 0.5  # Default prediction confidence threshold
HIGH_CONFIDENCE_THRESHOLD = 0.7  # Threshold for "high confidence" predictions

# =============================================================================
# VOLATILITY THRESHOLDS
# =============================================================================
VOL_RATIO_MIN = 0.5  # Minimum volatility ratio (vs historical)
VOL_RATIO_MAX = 2.0  # Maximum volatility ratio (vs historical)
MIN_VOLATILITY = 1e-8  # Minimum volatility to avoid division by zero

# =============================================================================
# DATA QUALITY THRESHOLDS
# =============================================================================
MAX_NAN_RATIO = 0.01  # Maximum allowed NaN ratio (1%)
MIN_ROWS = 100  # Minimum rows required for valid dataset
MIN_TRAIN_SAMPLES = 50  # Minimum training samples per class
MAX_MISSING_PCT = 0.05  # Maximum missing data percentage (5%)

# =============================================================================
# CORRELATION THRESHOLDS
# =============================================================================
LEAKAGE_CORRELATION_THRESHOLD = 0.8  # Correlation threshold for leakage detection
HIGH_CORRELATION_THRESHOLD = 0.95  # Threshold for near-duplicate features
FEATURE_CORRELATION_WARN = 0.9  # Correlation level for warnings

# =============================================================================
# CLASS BALANCE THRESHOLDS
# =============================================================================
MIN_CLASS_RATIO = 0.05  # Minimum class ratio (5% of samples)
MAX_CLASS_IMBALANCE = 10.0  # Maximum class imbalance ratio (majority/minority)
TARGET_CLASS_BALANCE = 0.33  # Target balance for 3-class problems

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================
MIN_ACCURACY = 0.40  # Minimum acceptable accuracy (for 3-class)
MIN_F1_SCORE = 0.35  # Minimum acceptable F1 score
OVERFITTING_GAP = 0.15  # Max train-val gap before overfitting warning

__all__ = [
    # Signal
    "MIN_SIGNAL_RATIO",
    "DEFAULT_CONFIDENCE",
    "HIGH_CONFIDENCE_THRESHOLD",
    # Volatility
    "VOL_RATIO_MIN",
    "VOL_RATIO_MAX",
    "MIN_VOLATILITY",
    # Data quality
    "MAX_NAN_RATIO",
    "MIN_ROWS",
    "MIN_TRAIN_SAMPLES",
    "MAX_MISSING_PCT",
    # Correlation
    "LEAKAGE_CORRELATION_THRESHOLD",
    "HIGH_CORRELATION_THRESHOLD",
    "FEATURE_CORRELATION_WARN",
    # Class balance
    "MIN_CLASS_RATIO",
    "MAX_CLASS_IMBALANCE",
    "TARGET_CLASS_BALANCE",
    # Performance
    "MIN_ACCURACY",
    "MIN_F1_SCORE",
    "OVERFITTING_GAP",
]
