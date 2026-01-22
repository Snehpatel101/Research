"""
CVOrchestrator - Unified cross-validation orchestration.

PHASE_3: Training Orchestration - Single interface for all CV methods.

This module provides a unified interface to all cross-validation methods
with proper purging and embargo handling. It integrates with PipelineConfig
from src/core for seamless configuration.

Supported CV methods:
- PurgedKFold: Standard purged k-fold with label-aware purging
- CPCV: Combinatorial Purged Cross-Validation for robust validation
- WalkForward: Expanding/rolling window for production simulation
- PBO: Probability of Backtest Overfitting detection

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core import CVMethod, PipelineConfig, MODEL_DATA_RANKS
from .purged_kfold import PurgedKFold, PurgedKFoldConfig
from .cpcv import CombinatorialPurgedCV, CPCVConfig
from .walk_forward import WalkForwardEvaluator, WalkForwardConfig
from .pbo import compute_pbo, PBOConfig, PBOResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CVFoldResult:
    """
    Result from a single CV fold.

    Attributes:
        fold_idx: Zero-based fold index
        train_indices: Array of training sample indices
        test_indices: Array of test sample indices
        train_size: Number of training samples
        test_size: Number of test samples
        purged_size: Number of samples removed by purging
        embargoed_size: Number of samples removed by embargo
    """

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int
    purged_size: int = 0
    embargoed_size: int = 0


@dataclass
class CVSplitInfo:
    """
    Information about CV splits.

    Provides summary statistics and detailed fold information
    for debugging and analysis.

    Attributes:
        cv_method: Name of CV method used
        n_splits: Number of CV folds/paths
        n_samples: Total number of samples
        purge_bars: Number of bars purged before test
        embargo_bars: Number of bars embargoed after test
        folds: List of CVFoldResult with per-fold details
    """

    cv_method: str
    n_splits: int
    n_samples: int
    purge_bars: int
    embargo_bars: int
    folds: List[CVFoldResult] = field(default_factory=list)

    @property
    def total_train_samples(self) -> int:
        """Total training samples across all folds."""
        return sum(f.train_size for f in self.folds)

    @property
    def total_test_samples(self) -> int:
        """Total test samples across all folds."""
        return sum(f.test_size for f in self.folds)

    @property
    def avg_train_size(self) -> float:
        """Average training set size per fold."""
        return self.total_train_samples / len(self.folds) if self.folds else 0

    @property
    def avg_test_size(self) -> float:
        """Average test set size per fold."""
        return self.total_test_samples / len(self.folds) if self.folds else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cv_method": self.cv_method,
            "n_splits": self.n_splits,
            "n_samples": self.n_samples,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "total_train_samples": self.total_train_samples,
            "total_test_samples": self.total_test_samples,
            "avg_train_size": self.avg_train_size,
            "avg_test_size": self.avg_test_size,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_size": f.train_size,
                    "test_size": f.test_size,
                    "purged_size": f.purged_size,
                    "embargoed_size": f.embargoed_size,
                }
                for f in self.folds
            ],
        }


# =============================================================================
# CV ORCHESTRATOR
# =============================================================================


class CVOrchestrator:
    """
    Unified cross-validation orchestration.

    Provides single interface to all CV methods:
    - PurgedKFold: Standard purged k-fold for general use
    - CPCV: Combinatorial Purged Cross-Validation for robust validation
    - WalkForward: Walk-forward validation for production simulation
    - PBO: Probability of Backtest Overfitting detection

    The orchestrator handles proper purging and embargo for all methods,
    preventing information leakage in time-series data.

    Usage:
        from src.core import PipelineConfig
        from src.cross_validation import CVOrchestrator

        config = PipelineConfig(...)
        cv_orch = CVOrchestrator.from_config(config)

        # Get splits
        for train_idx, test_idx in cv_orch.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])

        # Get split info
        info = cv_orch.get_split_info(X, y)
        print(f"Average train size: {info.avg_train_size}")

    Attributes:
        cv_method: CVMethod enum value
        n_splits: Number of CV folds
        purge_bars: Bars to purge before test
        embargo_bars: Bars to embargo after test
        n_groups: Number of groups for CPCV
        n_test_groups: Number of test groups for CPCV
        n_windows: Number of windows for walk-forward
        window_type: "expanding" or "rolling" for walk-forward
        min_train_size: Minimum train size for walk-forward
    """

    def __init__(
        self,
        cv_method: CVMethod,
        n_splits: int = 5,
        purge_bars: int = 60,
        embargo_bars: int = 1440,
        # CPCV-specific
        n_groups: int = 6,
        n_test_groups: int = 2,
        max_combinations: int = 20,
        # Walk-forward specific
        n_windows: int = 5,
        window_type: str = "expanding",
        min_train_size: Optional[int] = None,
        min_train_pct: float = 0.4,
        test_pct: float = 0.1,
    ) -> None:
        """
        Initialize CVOrchestrator.

        Args:
            cv_method: CV method to use (from CVMethod enum)
            n_splits: Number of folds for PurgedKFold (default: 5)
            purge_bars: Bars to purge before test set (default: 60)
            embargo_bars: Bars to embargo after test set (default: 1440)
            n_groups: Number of groups for CPCV (default: 6)
            n_test_groups: Number of test groups per CPCV combination (default: 2)
            max_combinations: Maximum CPCV combinations to evaluate (default: 20)
            n_windows: Number of windows for walk-forward (default: 5)
            window_type: "expanding" or "rolling" for walk-forward (default: "expanding")
            min_train_size: Minimum train size for walk-forward (optional, uses min_train_pct if None)
            min_train_pct: Minimum train percentage for walk-forward (default: 0.4)
            test_pct: Test percentage per walk-forward window (default: 0.1)
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

        # CPCV params
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.max_combinations = max_combinations

        # Walk-forward params
        self.n_windows = n_windows
        self.window_type = window_type
        self.min_train_size = min_train_size
        self.min_train_pct = min_train_pct
        self.test_pct = test_pct

        # Create underlying CV object
        self._cv = self._create_cv()

        logger.info(
            f"CVOrchestrator initialized: {cv_method.value}, "
            f"purge={purge_bars}, embargo={embargo_bars}"
        )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "CVOrchestrator":
        """
        Create CVOrchestrator from PipelineConfig.

        This is the recommended way to create a CVOrchestrator when using
        the unified pipeline configuration.

        Args:
            config: PipelineConfig instance with CV settings

        Returns:
            CVOrchestrator instance configured from PipelineConfig

        Example:
            config = PipelineConfig(
                symbol="MES",
                data_path="./data/mes.parquet",
                output_dir="./output",
                cv_method="purged_kfold",
                n_splits=5,
            )
            cv_orch = CVOrchestrator.from_config(config)
        """
        return cls(
            cv_method=CVMethod(config.cv_method),
            n_splits=config.n_splits,
            purge_bars=config.purge_bars,
            embargo_bars=config.embargo_bars,
        )

    def _create_cv(self) -> Any:
        """
        Create the underlying CV object based on cv_method.

        Returns:
            Configured CV object (PurgedKFold, CombinatorialPurgedCV, or WalkForwardEvaluator)

        Raises:
            ValueError: If cv_method is unknown
        """
        if self.cv_method == CVMethod.PURGED_KFOLD:
            cv_config = PurgedKFoldConfig(
                n_splits=self.n_splits,
                purge_bars=self.purge_bars,
                embargo_bars=self.embargo_bars,
            )
            return PurgedKFold(cv_config)

        elif self.cv_method == CVMethod.CPCV:
            # Convert bars to percentages for CPCV
            # CPCV uses percentage-based purge/embargo
            cv_config = CPCVConfig(
                n_groups=self.n_groups,
                n_test_groups=self.n_test_groups,
                max_combinations=self.max_combinations,
                purge_pct=0.01,  # 1% default, will be adjusted in split()
                embargo_pct=0.01,  # 1% default, will be adjusted in split()
            )
            return CombinatorialPurgedCV(cv_config)

        elif self.cv_method == CVMethod.WALK_FORWARD:
            cv_config = WalkForwardConfig(
                n_windows=self.n_windows,
                window_type=self.window_type,
                min_train_pct=self.min_train_pct,
                test_pct=self.test_pct,
                embargo_bars=self.embargo_bars,
                gap_bars=self.purge_bars,  # Use purge_bars as gap
            )
            return WalkForwardEvaluator(cv_config)

        elif self.cv_method == CVMethod.PBO:
            # PBO uses CPCV underneath for generating paths
            cv_config = CPCVConfig(
                n_groups=self.n_groups,
                n_test_groups=self.n_test_groups,
                max_combinations=self.max_combinations,
                purge_pct=0.01,
                embargo_pct=0.01,
            )
            return CombinatorialPurgedCV(cv_config)

        else:
            raise ValueError(f"Unknown CV method: {self.cv_method}")

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
        groups: Optional[np.ndarray] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for CV.

        This is the main interface for obtaining CV splits. It works
        with all CV methods through a unified interface.

        Args:
            X: Feature data (DataFrame or array)
            y: Labels (optional for some CV methods)
            groups: Group labels for grouped CV (optional)
            label_end_times: When each label's outcome is known (optional)
                If provided, enables proper purging for overlapping labels

        Yields:
            (train_indices, test_indices) tuples for each fold

        Example:
            for train_idx, test_idx in cv_orch.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
        """
        if self.cv_method == CVMethod.WALK_FORWARD:
            # Walk-forward uses the split method directly
            yield from self._cv.split(X, y, groups, label_end_times)

        elif self.cv_method in (CVMethod.CPCV, CVMethod.PBO):
            # CPCV/PBO split yields (train_idx, test_idx, path_id)
            for train_idx, test_idx, _ in self._cv.split(X, y, groups, label_end_times):
                yield train_idx, test_idx

        else:
            # PurgedKFold standard split
            yield from self._cv.split(X, y, groups, label_end_times)

    def get_n_splits(
        self,
        X: Union[pd.DataFrame, np.ndarray, None] = None,
        y: Union[pd.Series, np.ndarray, None] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Get number of splits for this CV configuration.

        Args:
            X: Feature data (optional, required for some CV methods)
            y: Labels (optional)
            groups: Group labels (optional)

        Returns:
            Number of CV splits/folds/paths
        """
        if self.cv_method in (CVMethod.CPCV, CVMethod.PBO):
            return self._cv.get_n_splits(X, y, groups)
        elif self.cv_method == CVMethod.WALK_FORWARD:
            return self.n_windows
        else:
            return self._cv.get_n_splits(X, y, groups)

    def get_split_info(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> CVSplitInfo:
        """
        Get detailed information about CV splits.

        Useful for debugging, analysis, and understanding the CV structure.

        Args:
            X: Feature data
            y: Labels (optional)
            label_end_times: When each label's outcome is known (optional)

        Returns:
            CVSplitInfo with fold details and summary statistics

        Example:
            info = cv_orch.get_split_info(X, y)
            print(f"CV Method: {info.cv_method}")
            print(f"Number of splits: {info.n_splits}")
            print(f"Average train size: {info.avg_train_size:.0f}")
            print(f"Average test size: {info.avg_test_size:.0f}")
        """
        n_samples = len(X)
        folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.split(X, y, label_end_times=label_end_times)
        ):
            fold_result = CVFoldResult(
                fold_idx=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                train_size=len(train_idx),
                test_size=len(test_idx),
                purged_size=self.purge_bars,
                embargoed_size=self.embargo_bars,
            )
            folds.append(fold_result)

        return CVSplitInfo(
            cv_method=self.cv_method.value,
            n_splits=len(folds),
            n_samples=n_samples,
            purge_bars=self.purge_bars,
            embargo_bars=self.embargo_bars,
            folds=folds,
        )

    def compute_pbo(
        self,
        strategy_returns: np.ndarray,
        n_partitions: int = 16,
    ) -> PBOResult:
        """
        Compute Probability of Backtest Overfitting.

        PBO quantifies the probability that a backtest-optimal strategy
        will underperform out-of-sample. High PBO indicates overfitting.

        Only valid when cv_method is PBO or CPCV.

        Args:
            strategy_returns: Matrix of returns (n_strategies, n_paths)
                Each row is a strategy, each column is a path/period
            n_partitions: Number of partitions for CSCV (default: 16)

        Returns:
            PBOResult with overfitting probability and related metrics

        Raises:
            ValueError: If cv_method is not CPCV or PBO

        Example:
            # strategy_returns shape: (n_strategies, n_paths)
            pbo_result = cv_orch.compute_pbo(strategy_returns)
            if pbo_result.is_overfit:
                print(f"Warning: PBO = {pbo_result.pbo:.3f}")
        """
        if self.cv_method not in (CVMethod.PBO, CVMethod.CPCV):
            raise ValueError(
                f"PBO computation requires CPCV or PBO cv_method, got {self.cv_method.value}"
            )

        pbo_config = PBOConfig(n_partitions=n_partitions)
        return compute_pbo(strategy_returns, pbo_config)

    def validate_coverage(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
    ) -> Dict[str, Any]:
        """
        Validate that CV covers all samples appropriately.

        Checks for gaps, overlaps, and coverage statistics.

        Args:
            X: Feature data
            y: Labels (optional)

        Returns:
            Dict with coverage statistics
        """
        n_samples = len(X)
        test_coverage = np.zeros(n_samples, dtype=int)
        train_coverage = np.zeros(n_samples, dtype=int)

        for train_idx, test_idx in self.split(X, y):
            test_coverage[test_idx] += 1
            train_coverage[train_idx] += 1

        return {
            "total_samples": n_samples,
            "samples_in_test": int((test_coverage > 0).sum()),
            "test_coverage_fraction": float((test_coverage > 0).mean()),
            "samples_never_tested": int((test_coverage == 0).sum()),
            "samples_in_multiple_tests": int((test_coverage > 1).sum()),
            "avg_test_appearances": float(test_coverage.mean()),
            "avg_train_appearances": float(train_coverage[train_coverage > 0].mean()),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CVOrchestrator("
            f"method={self.cv_method.value}, "
            f"n_splits={self.n_splits}, "
            f"purge={self.purge_bars}, "
            f"embargo={self.embargo_bars})"
        )


# =============================================================================
# MODEL-AWARE CV FACTORY
# =============================================================================


def get_cv_for_model(
    model_name: str,
    config: PipelineConfig,
) -> CVOrchestrator:
    """
    Get appropriate CV orchestrator for a specific model.

    Neural models may need fewer folds due to longer training times.
    This function automatically adjusts CV settings based on model type.

    Args:
        model_name: Name of the model (e.g., "xgboost", "lstm", "transformer")
        config: PipelineConfig with base CV settings

    Returns:
        CVOrchestrator configured appropriately for the model

    Example:
        config = PipelineConfig(...)

        # XGBoost gets standard 5-fold
        xgb_cv = get_cv_for_model("xgboost", config)

        # LSTM gets reduced 3-fold (rank >= 3)
        lstm_cv = get_cv_for_model("lstm", config)
    """
    rank = MODEL_DATA_RANKS.get(model_name.lower(), 2)

    # Neural models (rank >= 3) use fewer folds for efficiency
    # Rank 3 = sequence models (LSTM, GRU, TCN, etc.)
    # Rank 4 = multi-stream models (PatchTST, iTransformer)
    n_splits = 3 if rank >= 3 else config.n_splits

    return CVOrchestrator(
        cv_method=CVMethod(config.cv_method),
        n_splits=n_splits,
        purge_bars=config.purge_bars,
        embargo_bars=config.embargo_bars,
    )


def create_cv_orchestrator(
    cv_method: str = "purged_kfold",
    n_splits: int = 5,
    purge_bars: int = 60,
    embargo_bars: int = 1440,
    **kwargs: Any,
) -> CVOrchestrator:
    """
    Factory function to create CVOrchestrator.

    Convenience function for creating CVOrchestrator with string-based
    cv_method specification.

    Args:
        cv_method: CV method name ("purged_kfold", "cpcv", "walk_forward", "pbo")
        n_splits: Number of CV folds
        purge_bars: Bars to purge before test
        embargo_bars: Bars to embargo after test
        **kwargs: Additional method-specific parameters

    Returns:
        Configured CVOrchestrator instance

    Example:
        cv = create_cv_orchestrator(
            cv_method="walk_forward",
            n_windows=5,
            window_type="expanding",
        )
    """
    return CVOrchestrator(
        cv_method=CVMethod(cv_method),
        n_splits=n_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CVOrchestrator",
    "CVFoldResult",
    "CVSplitInfo",
    "get_cv_for_model",
    "create_cv_orchestrator",
]
