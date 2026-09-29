"""
Walk-forward validation configuration (ExperimentConfig.training.walk_forward).

Every field here reaches the pipeline through
``ExperimentConfig.to_pipeline_config()`` (``wf_*`` PipelineConfig fields).
The walk-forward train/test gap and embargo are NOT configured here: they come
from ``training.purge_bars`` / ``training.embargo_bars`` so walk-forward and
purged k-fold always use the same leakage guards.

The operational CV configs (PurgedKFoldConfig, CPCVConfig, PBOConfig, ...)
live next to their implementations in ``src/validation/cv/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.config.base import BaseConfig


class WindowType(StrEnum):
    """Walk-forward window types."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


@dataclass
class WalkForwardConfig(BaseConfig):
    """
    Configuration for walk-forward training (training_mode="walk_forward").

    Attributes:
        n_windows: Number of walk-forward windows
        window_type: 'expanding' (growing train) or 'rolling' (fixed train)
        min_train_pct: Fraction of data in the first training window
        test_pct: Fraction of data per test window

    Example:
        config = WalkForwardConfig(
            n_windows=5,
            window_type="expanding",
            min_train_pct=0.4,
        )
    """

    n_windows: int = 5
    window_type: str = "expanding"
    min_train_pct: float = 0.4
    test_pct: float = 0.1

    def validate(self) -> list[str]:
        """Validate walk-forward configuration."""
        issues = super().validate()

        if self.n_windows < 1:
            issues.append(f"n_windows must be >= 1, got {self.n_windows}")

        valid_types = [t.value for t in WindowType]
        if self.window_type not in valid_types:
            issues.append(f"window_type must be one of {valid_types}, got '{self.window_type}'")

        if not 0 < self.min_train_pct < 1:
            issues.append(f"min_train_pct must be in (0, 1), got {self.min_train_pct}")

        if not 0 < self.test_pct < 1:
            issues.append(f"test_pct must be in (0, 1), got {self.test_pct}")

        # Check total doesn't exceed 100%
        if self.min_train_pct + self.n_windows * self.test_pct > 1.0:
            issues.append(
                "min_train_pct + n_windows * test_pct exceeds 1.0. Reduce n_windows or test_pct."
            )

        return issues


__all__ = [
    "WindowType",
    "WalkForwardConfig",
]
