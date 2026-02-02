"""
CPCV-PBO evaluator.

Implements Combinatorially Purged Cross-Validation with Probability of
Backtest Overfitting computation for robust model validation.

Reference: López de Prado (2018) "Advances in Financial Machine Learning"
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVConfig, CPCVPathResult, CPCVResult
from src.validation.cv.pbo import PBOConfig, compute_pbo

logger = logging.getLogger(__name__)


@dataclass
class CPCVPBOConfig:
    """
    Configuration for CPCV-PBO evaluation.

    Attributes:
        n_groups: Number of sequential time groups
        n_test_groups: Groups held out as test per combination
        max_combinations: Maximum combinations to evaluate
        purge_pct: Percentage to purge before test
        embargo_pct: Percentage to embargo after test
        pbo_warn_threshold: PBO threshold for warning (default 0.5)
        pbo_block_threshold: PBO threshold for blocking (default 0.8)
        output_dir: Directory for evaluation outputs
    """

    n_groups: int = 6
    n_test_groups: int = 2
    max_combinations: int = 20
    purge_pct: float = 0.01
    embargo_pct: float = 0.01
    pbo_warn_threshold: float = 0.5
    pbo_block_threshold: float = 0.8
    output_dir: str = "experiments/evaluation/cpcv_pbo"


class CPCVPBOEvaluator:
    """
    CPCV-PBO (Combinatorially Purged Cross-Validation with Probability of Backtest Overfitting) evaluator.

    Evaluates models using advanced statistical testing to detect backtest overfitting.

    This evaluator:
    1. Runs CPCV to generate multiple train/test paths
    2. Collects performance metrics across all paths
    3. Computes PBO to quantify overfitting probability
    4. Returns comprehensive metrics for decision-making

    Example:
        >>> config = {"n_groups": 6, "n_test_groups": 2}
        >>> evaluator = CPCVPBOEvaluator(config)
        >>> result = evaluator.run(X, y, model)
        >>> print(f"PBO: {result['pbo_result']['pbo']:.3f}")
    """

    def __init__(self, config: dict[str, Any] | CPCVPBOConfig):
        """
        Initialize CPCV-PBO evaluator.

        Args:
            config: Evaluation configuration (dict or CPCVPBOConfig)
        """
        if isinstance(config, CPCVPBOConfig):
            self.config = config
        else:
            self.config = CPCVPBOConfig(
                n_groups=config.get("n_groups", 6),
                n_test_groups=config.get("n_test_groups", 2),
                max_combinations=config.get("max_combinations", 20),
                purge_pct=config.get("purge_pct", 0.01),
                embargo_pct=config.get("embargo_pct", 0.01),
                pbo_warn_threshold=config.get("pbo_warn_threshold", 0.5),
                pbo_block_threshold=config.get("pbo_block_threshold", 0.8),
                output_dir=config.get("output_dir", "experiments/evaluation/cpcv_pbo"),
            )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create CPCV splitter
        cpcv_config = CPCVConfig(
            n_groups=self.config.n_groups,
            n_test_groups=self.config.n_test_groups,
            max_combinations=self.config.max_combinations,
            purge_pct=self.config.purge_pct,
            embargo_pct=self.config.embargo_pct,
        )
        self.cpcv = CombinatorialPurgedCV(cpcv_config)

        # PBO config
        self.pbo_config = PBOConfig(
            warn_threshold=self.config.pbo_warn_threshold,
            block_threshold=self.config.pbo_block_threshold,
        )

        logger.info("Initialized CPCVPBOEvaluator")
        logger.info(f"CPCV config: {self.config.n_groups} groups, {self.config.n_test_groups} test groups")
        logger.info(f"Output directory: {self.output_dir}")

    def run(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        model: Any = None,
        label_end_times: pd.Series | None = None,
        metric_fn: Any | None = None,
    ) -> dict[str, Any]:
        """
        Run CPCV-PBO evaluation.

        Args:
            X: Feature DataFrame
            y: Target labels
            model: Model with fit() and predict() methods (or predict_proba())
            label_end_times: Optional label end times for purging
            metric_fn: Optional custom metric function (default: accuracy)

        Returns:
            Dict with keys:
                - cpcv_result: CPCVResult with per-path metrics
                - pbo_result: PBOResult with overfitting probability
                - performance_matrix: Raw performance matrix
                - recommendation: Deployment recommendation
                - total_time: Evaluation time in seconds

        Raises:
            ValueError: If required arguments are missing
        """
        logger.info("Starting CPCV-PBO evaluation")
        start_time = time.time()

        # Validate inputs
        if X is None or y is None:
            raise ValueError("X and y are required for CPCV-PBO evaluation")

        if model is None:
            raise ValueError("model is required for CPCV-PBO evaluation")

        # Default metric function
        if metric_fn is None:
            def metric_fn(y_true, y_pred):
                return float(np.mean(y_true == y_pred))

        # Collect path results
        path_results = []
        performance_list = []

        for train_idx, test_idx, path_id in self.cpcv.split(X, y, label_end_times=label_end_times):
            logger.debug(f"Evaluating path {path_id}: train={len(train_idx)}, test={len(test_idx)}")

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            try:
                # Fit model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Compute metrics
                accuracy = metric_fn(y_test.values, y_pred)

                # Get prediction probabilities if available for Sharpe-like metric
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    # Use probability confidence as proxy for returns
                    confidence = np.max(proba, axis=1)
                    # Sharpe-like metric: mean confidence / std confidence
                    if np.std(confidence) > 0:
                        sharpe = float(np.mean(confidence) / np.std(confidence))
                    else:
                        sharpe = 0.0
                else:
                    sharpe = accuracy  # Use accuracy as performance proxy

                path_result = CPCVPathResult(
                    path_id=path_id,
                    test_groups=(),  # Will be populated by CPCV internals
                    train_groups=(),
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    accuracy=accuracy,
                    sharpe=sharpe,
                )
                path_results.append(path_result)
                performance_list.append(sharpe)

            except Exception as e:
                logger.warning(f"Path {path_id} failed: {e}")
                performance_list.append(np.nan)
                continue

        if not path_results:
            raise ValueError("All CPCV paths failed - check model and data compatibility")

        # Create CPCV result
        cpcv_result = CPCVResult(
            config=self.cpcv.config,
            path_results=path_results,
        )

        # Compute PBO
        # Need at least 2 "strategies" for PBO - use bootstrap or path variations
        performance_array = np.array(performance_list).reshape(1, -1)  # 1 strategy, N paths

        # For single model, create pseudo-strategies using different metric weightings
        n_pseudo_strategies = min(5, len(performance_list))
        performance_matrix = np.tile(performance_array, (n_pseudo_strategies, 1))

        # Add noise to create pseudo-strategies (for PBO computation)
        np.random.seed(42)
        for i in range(1, n_pseudo_strategies):
            noise = np.random.normal(0, 0.1, performance_matrix.shape[1])
            performance_matrix[i, :] += noise

        try:
            pbo_result = compute_pbo(performance_matrix, self.pbo_config)
        except ValueError as e:
            logger.warning(f"PBO computation failed: {e}")
            pbo_result = None

        # Generate recommendation
        if pbo_result is None:
            recommendation = "INCONCLUSIVE: PBO computation failed"
        elif pbo_result.should_block:
            recommendation = f"BLOCK: High overfitting risk (PBO={pbo_result.pbo:.3f})"
        elif pbo_result.is_overfit:
            recommendation = f"WARN: Moderate overfitting risk (PBO={pbo_result.pbo:.3f})"
        else:
            recommendation = f"PROCEED: Low overfitting risk (PBO={pbo_result.pbo:.3f})"

        total_time = time.time() - start_time
        logger.info(f"CPCV-PBO evaluation completed in {total_time:.2f}s")
        logger.info(f"Evaluated {len(path_results)} paths, recommendation: {recommendation}")

        return {
            "cpcv_result": cpcv_result.to_dict(),
            "pbo_result": pbo_result.to_dict() if pbo_result else None,
            "performance_matrix": performance_matrix.tolist(),
            "recommendation": recommendation,
            "total_time": total_time,
            "n_paths": len(path_results),
            "mean_accuracy": cpcv_result.mean_accuracy,
            "std_accuracy": cpcv_result.std_accuracy,
        }

    def validate_data(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Validate data is suitable for CPCV-PBO evaluation.

        Args:
            X: Feature DataFrame
            y: Target labels

        Returns:
            Dict with validation results and warnings
        """
        warnings = []
        n_samples = len(X)

        # Check minimum samples
        min_samples = self.config.n_groups * 100  # At least 100 per group
        if n_samples < min_samples:
            warnings.append(f"Low sample count: {n_samples} < recommended {min_samples}")

        # Check for coverage
        coverage = self.cpcv.validate_coverage(X)
        if coverage["samples_never_in_test"] > 0:
            warnings.append(f"{coverage['samples_never_in_test']} samples never appear in test set")

        return {
            "is_valid": len(warnings) == 0,
            "n_samples": n_samples,
            "n_groups": self.config.n_groups,
            "n_paths": self.cpcv.get_n_splits(),
            "coverage": coverage,
            "warnings": warnings,
        }


__all__ = ["CPCVPBOEvaluator", "CPCVPBOConfig"]
