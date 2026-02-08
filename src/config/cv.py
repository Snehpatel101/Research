"""
Consolidated Cross-Validation Configuration Classes.

This module contains all CV-related configuration classes:
- CVConfig: Base cross-validation configuration
- PurgedKFoldConfig: Purged K-fold configuration
- CPCVConfig: Combinatorial Purged CV configuration
- WalkForwardConfig: Walk-forward validation configuration
- PBOConfig: Probability of Backtest Overfitting configuration

These are the CANONICAL locations for these configs.
Import from here:
    from src.config.cv import CVConfig, CPCVConfig, WalkForwardConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.config.base import BaseConfig

# =============================================================================
# ENUMS
# =============================================================================


class CVMethod(str, Enum):
    """Supported cross-validation methods."""

    PURGED_KFOLD = "purged_kfold"
    CPCV = "cpcv"
    WALK_FORWARD = "walk_forward"
    PBO = "pbo"
    STANDARD = "standard"


class WindowType(str, Enum):
    """Walk-forward window types."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


# =============================================================================
# BASE CV CONFIGURATION
# =============================================================================


@dataclass
class CVConfig(BaseConfig):
    """
    Base configuration for cross-validation.

    Attributes:
        method: CV method to use
        n_splits: Number of cross-validation splits
        shuffle: Whether to shuffle (not recommended for time series)
        random_state: Random seed for shuffling

    Example:
        config = CVConfig(
            method="purged_kfold",
            n_splits=5,
        )
    """

    method: str = "purged_kfold"
    n_splits: int = 5
    shuffle: bool = False
    random_state: int = 42

    def validate(self) -> list[str]:
        """Validate CV configuration."""
        issues = super().validate()

        valid_methods = [m.value for m in CVMethod]
        if self.method not in valid_methods:
            issues.append(f"method must be one of {valid_methods}, got '{self.method}'")

        if self.n_splits < 2:
            issues.append(f"n_splits must be >= 2, got {self.n_splits}")

        return issues


# =============================================================================
# PURGE/EMBARGO CONFIGURATION
# =============================================================================


@dataclass
class PurgeEmbargoConfig(BaseConfig):
    """
    Configuration for purge and embargo settings.

    Purge: Remove samples near split boundaries to prevent label leakage.
    Embargo: Skip samples after test period to break serial correlation.

    Attributes:
        purge_bars: Number of bars to purge before test set
        embargo_bars: Number of bars to embargo after test set
        purge_multiplier: Multiplier for auto-scaling purge from horizon
        embargo_time_minutes: Embargo duration in calendar time
        auto_scale: Whether to auto-calculate from horizon

    Example:
        config = PurgeEmbargoConfig(
            purge_bars=60,
            embargo_bars=1440,
        )
    """

    purge_bars: int = 60
    embargo_bars: int = 1440
    purge_multiplier: float = 3.0
    embargo_time_minutes: int = 7200  # 5 days
    auto_scale: bool = True

    def validate(self) -> list[str]:
        """Validate purge/embargo configuration."""
        issues = super().validate()

        if self.purge_bars < 0:
            issues.append(f"purge_bars must be non-negative, got {self.purge_bars}")

        if self.embargo_bars < 0:
            issues.append(f"embargo_bars must be non-negative, got {self.embargo_bars}")

        if self.purge_multiplier <= 0:
            issues.append(f"purge_multiplier must be positive, got {self.purge_multiplier}")

        return issues

    def calculate_for_horizon(
        self, max_horizon: int, timeframe_minutes: int = 5
    ) -> tuple[int, int]:
        """
        Calculate purge and embargo bars for a given horizon.

        Args:
            max_horizon: Maximum prediction horizon
            timeframe_minutes: Bar timeframe in minutes

        Returns:
            Tuple of (purge_bars, embargo_bars)
        """
        if not self.auto_scale:
            return self.purge_bars, self.embargo_bars

        purge = int(max_horizon * self.purge_multiplier)
        embargo = int(self.embargo_time_minutes / timeframe_minutes)

        return purge, max(embargo, 1)


# =============================================================================
# PURGED K-FOLD CONFIGURATION
# =============================================================================


@dataclass
class PurgedKFoldConfig(BaseConfig):
    """
    Configuration for Purged K-Fold cross-validation.

    This is the CANONICAL PurgedKFoldConfig. Use this instead of:
    - PurgedKFoldConfig in src/cross_validation/purged_kfold.py (deprecated)
    - PurgedKFoldConfig in src/validation/cv/purged_kfold.py (deprecated)

    Attributes:
        n_splits: Number of folds
        purge_bars: Bars to purge before test
        embargo_bars: Bars to embargo after test
        use_label_end_times: Use label end times for purging

    Example:
        config = PurgedKFoldConfig(
            n_splits=5,
            purge_bars=60,
            embargo_bars=1440,
        )
    """

    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440
    use_label_end_times: bool = True

    def validate(self) -> list[str]:
        """Validate purged k-fold configuration."""
        issues = super().validate()

        if self.n_splits < 2:
            issues.append(f"n_splits must be >= 2, got {self.n_splits}")

        if self.purge_bars < 0:
            issues.append(f"purge_bars must be non-negative, got {self.purge_bars}")

        if self.embargo_bars < 0:
            issues.append(f"embargo_bars must be non-negative, got {self.embargo_bars}")

        return issues


# =============================================================================
# CPCV CONFIGURATION
# =============================================================================


@dataclass
class CPCVConfig(BaseConfig):
    """
    Configuration for Combinatorially Purged Cross-Validation.

    CPCV generates multiple train/test paths by combining different fold
    combinations, enabling PBO computation.

    This is the CANONICAL CPCVConfig. Use this instead of:
    - CPCVConfig in src/cross_validation/cpcv.py (deprecated)
    - CPCVConfig in src/validation/cv/cpcv.py (deprecated)

    Attributes:
        n_groups: Number of sequential time groups
        n_test_groups: Number of groups held out as test per combination
        max_combinations: Maximum combinations to evaluate
        purge_pct: Percentage of data to purge (0-0.5)
        embargo_pct: Percentage of data to embargo (0-0.5)

    Example:
        config = CPCVConfig(
            n_groups=6,
            n_test_groups=2,
            max_combinations=20,
        )
    """

    n_groups: int = 6
    n_test_groups: int = 2
    max_combinations: int = 20
    purge_pct: float = 0.01
    embargo_pct: float = 0.01

    def validate(self) -> list[str]:
        """Validate CPCV configuration."""
        issues = super().validate()

        if self.n_groups < 2:
            issues.append(f"n_groups must be >= 2, got {self.n_groups}")

        if self.n_test_groups < 1:
            issues.append(f"n_test_groups must be >= 1, got {self.n_test_groups}")

        if self.n_test_groups >= self.n_groups:
            issues.append(
                f"n_test_groups ({self.n_test_groups}) must be < n_groups ({self.n_groups})"
            )

        if self.max_combinations < 1:
            issues.append(f"max_combinations must be >= 1, got {self.max_combinations}")

        if not 0 <= self.purge_pct < 0.5:
            issues.append(f"purge_pct must be in [0, 0.5), got {self.purge_pct}")

        if not 0 <= self.embargo_pct < 0.5:
            issues.append(f"embargo_pct must be in [0, 0.5), got {self.embargo_pct}")

        return issues

    @property
    def n_train_groups(self) -> int:
        """Number of groups used for training in each combination."""
        return self.n_groups - self.n_test_groups

    @property
    def total_combinations(self) -> int:
        """Total possible combinations C(n_groups, n_test_groups)."""
        from math import comb

        return comb(self.n_groups, self.n_test_groups)


# =============================================================================
# WALK-FORWARD CONFIGURATION
# =============================================================================


@dataclass
class WalkForwardConfig(BaseConfig):
    """
    Configuration for walk-forward validation.

    Walk-forward is more realistic than k-fold for trading as it respects
    temporal ordering.

    This is the CANONICAL WalkForwardConfig. Use this instead of:
    - WalkForwardConfig in src/cross_validation/walk_forward.py (deprecated)
    - WalkForwardConfig in src/validation/cv/walk_forward.py (deprecated)
    - WalkForwardTrainerConfig in src/models/training/modes/walk_forward.py (deprecated)

    Attributes:
        n_windows: Number of walk-forward windows
        window_type: 'expanding' (growing train) or 'rolling' (fixed train)
        min_train_pct: Minimum training data percentage
        test_pct: Percentage of data per test window
        embargo_bars: Bars to skip after each test period
        gap_bars: Bars to skip between train and test

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
    embargo_bars: int = 60
    gap_bars: int = 60

    # Training-specific settings
    retrain_on_each_window: bool = True
    warm_start: bool = False

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
                "min_train_pct + n_windows * test_pct exceeds 1.0. " "Reduce n_windows or test_pct."
            )

        if self.embargo_bars < 0:
            issues.append(f"embargo_bars must be non-negative, got {self.embargo_bars}")

        if self.gap_bars < 0:
            issues.append(f"gap_bars must be non-negative, got {self.gap_bars}")

        return issues


# =============================================================================
# PBO CONFIGURATION
# =============================================================================


@dataclass
class PBOConfig(BaseConfig):
    """
    Configuration for Probability of Backtest Overfitting calculation.

    PBO estimates the probability that a strategy's performance is due
    to overfitting rather than genuine alpha.

    This is the CANONICAL PBOConfig. Use this instead of:
    - PBOConfig in src/cross_validation/pbo.py (deprecated)
    - PBOConfig in src/validation/cv/pbo.py (deprecated)

    Attributes:
        n_partitions: Number of data partitions for CSCV
        sharpe_threshold: Threshold for considering strategy profitable
        risk_free_rate: Risk-free rate for Sharpe calculation
        annualization_factor: Factor for annualizing returns

    Example:
        config = PBOConfig(
            n_partitions=16,
            sharpe_threshold=0.0,
        )
    """

    n_partitions: int = 16
    sharpe_threshold: float = 0.0
    risk_free_rate: float = 0.0
    annualization_factor: float = 252.0

    # CSCV settings
    n_train_partitions: int = 8

    def validate(self) -> list[str]:
        """Validate PBO configuration."""
        issues = super().validate()

        if self.n_partitions < 2:
            issues.append(f"n_partitions must be >= 2, got {self.n_partitions}")

        if self.n_train_partitions < 1:
            issues.append(f"n_train_partitions must be >= 1, got {self.n_train_partitions}")

        if self.n_train_partitions >= self.n_partitions:
            issues.append(
                f"n_train_partitions ({self.n_train_partitions}) must be < "
                f"n_partitions ({self.n_partitions})"
            )

        if self.annualization_factor <= 0:
            issues.append(f"annualization_factor must be positive, got {self.annualization_factor}")

        return issues


# =============================================================================
# DSR CONFIGURATION
# =============================================================================


@dataclass
class DSRConfig(BaseConfig):
    """
    Configuration for Deflated Sharpe Ratio calculation.

    DSR adjusts the Sharpe ratio for multiple testing bias.

    This is the CANONICAL DSRConfig. Use this instead of:
    - DSRConfig in src/validation/deflated_sharpe.py (deprecated)

    Attributes:
        n_trials: Number of strategy trials/backtests performed
        variance_of_sharpes: Variance of Sharpe ratios across trials
        benchmark_sharpe: Benchmark Sharpe ratio to beat
        skewness: Skewness of returns (0 if unknown)
        kurtosis: Excess kurtosis of returns (0 if unknown)

    Example:
        config = DSRConfig(
            n_trials=100,
            benchmark_sharpe=0.5,
        )
    """

    n_trials: int = 1
    variance_of_sharpes: float | None = None
    benchmark_sharpe: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    def validate(self) -> list[str]:
        """Validate DSR configuration."""
        issues = super().validate()

        if self.n_trials < 1:
            issues.append(f"n_trials must be >= 1, got {self.n_trials}")

        return issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CVMethod",
    "WindowType",
    # Base
    "CVConfig",
    "PurgeEmbargoConfig",
    # Specific CV methods
    "PurgedKFoldConfig",
    "CPCVConfig",
    "WalkForwardConfig",
    "PBOConfig",
    "DSRConfig",
]
