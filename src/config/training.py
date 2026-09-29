"""
Training-section configuration classes for ExperimentConfig.

Every field here reaches the pipeline through
``ExperimentConfig.to_pipeline_config()``:

- OptunaConfig: hyperparameter-search budget and objective
- CalibrationConfig: post-training probability calibration

The operational twins (the Optuna tuner, ProbabilityCalibrator's own
CalibrationConfig, neural CheckpointConfig/OOMConfig, ...) live next to their
implementations under ``src/models`` and ``src/validation``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.base import BaseConfig

# Methods understood by src.models.calibration.ProbabilityCalibrator
CALIBRATION_METHODS = ("auto", "isotonic", "sigmoid")

# Metrics understood by the live tuner (TimeSeriesOptunaTuner)
OPTUNA_METRICS = (
    "f1_weighted",
    "accuracy",
    "sharpe_ratio",
    "sortino_ratio",
    "profit_factor",
    "precision",
    "recall",
    "roc_auc",
    "log_loss",
)


@dataclass
class OptunaConfig(BaseConfig):
    """
    Optuna hyperparameter optimization budget.

    Attributes:
        n_trials: Trials per model (0 disables hyperparameter tuning)
        timeout: Wall-clock cap per study in seconds (0 = no timeout)
        metric: Objective the tuner maximizes

    Example:
        config = OptunaConfig(n_trials=100, timeout=3600)
    """

    n_trials: int = 100
    timeout: int = 43200  # 12 hours default (prevents runaway optimization)
    metric: str = "f1_weighted"

    def validate(self) -> list[str]:
        """Validate Optuna configuration."""
        issues = super().validate()

        if self.n_trials < 0:
            issues.append(f"n_trials must be >= 0 (0 disables tuning), got {self.n_trials}")

        if self.timeout < 0:
            issues.append(f"timeout must be non-negative, got {self.timeout}")

        if self.metric not in OPTUNA_METRICS:
            issues.append(f"metric must be one of {list(OPTUNA_METRICS)}, got '{self.metric}'")

        return issues


@dataclass
class CalibrationConfig(BaseConfig):
    """
    Post-training probability calibration.

    Fitted on each model's validation split and shipped in its bundle.

    Attributes:
        enabled: Whether calibration is fitted
        method: 'auto', 'isotonic' or 'sigmoid' (Platt)

    Example:
        config = CalibrationConfig(enabled=True, method="isotonic")
    """

    enabled: bool = True
    method: str = "auto"

    def validate(self) -> list[str]:
        """Validate calibration configuration."""
        issues = super().validate()

        if self.method not in CALIBRATION_METHODS:
            issues.append(f"method must be one of {list(CALIBRATION_METHODS)}, got '{self.method}'")

        return issues


__all__ = [
    "OptunaConfig",
    "CalibrationConfig",
]
