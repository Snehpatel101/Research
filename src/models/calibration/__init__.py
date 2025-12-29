"""
Probability Calibration Module.

Provides tools for calibrating probability outputs from ML models,
particularly important for boosting models (XGBoost, LightGBM, CatBoost)
which are known to produce miscalibrated probabilities.

Key Components:
- ProbabilityCalibrator: Main calibration class (isotonic/sigmoid)
- CalibrationConfig: Configuration dataclass
- CalibrationMetrics: Before/after calibration metrics
- Metrics: Brier score, ECE, reliability bins

Example:
    >>> from src.models.calibration import ProbabilityCalibrator, CalibrationConfig
    >>> calibrator = ProbabilityCalibrator(CalibrationConfig(method="isotonic"))
    >>> metrics = calibrator.fit(y_val, probas_val)
    >>> calibrated = calibrator.calibrate(probas_test)
"""
from src.models.calibration.calibrator import (
    CalibrationConfig,
    CalibrationMetrics,
    ProbabilityCalibrator,
)
from src.models.calibration.conformal import (
    ConformalConfig,
    ConformalMetrics,
    ConformalPredictor,
    validate_coverage,
)
from src.models.calibration.metrics import (
    ReliabilityBins,
    compute_brier_score,
    compute_ece,
    compute_reliability_bins,
)

__all__ = [
    # Calibrator
    "ProbabilityCalibrator",
    "CalibrationConfig",
    "CalibrationMetrics",
    # Metrics
    "ReliabilityBins",
    "compute_brier_score",
    "compute_ece",
    "compute_reliability_bins",
    # Conformal Prediction
    "ConformalPredictor",
    "ConformalConfig",
    "ConformalMetrics",
    "validate_coverage",
]
