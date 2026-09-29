"""
CPCV-PBO evaluator.

Runs Combinatorial Purged Cross-Validation for one or more candidate models,
assembles each model's out-of-sample predictions into the phi CPCV backtest
paths, turns them into per-bar strategy returns, and -- when at least two
candidates are compared -- estimates the Probability of Backtest Overfitting
with CSCV over the T x N matrix of path-averaged per-bar returns.

References:
    Lopez de Prado (2018) "Advances in Financial Machine Learning", Ch. 12
    Bailey, Borwein, Lopez de Prado, Zhu (2017) "The Probability of Backtest Overfitting"
"""

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVConfig, CPCVPathResult, CPCVResult
from src.validation.cv.pbo import PBOConfig, compute_pbo, directional_strategy_returns
from src.validation.deflated_sharpe import sharpe_ratio_per_period

logger = logging.getLogger(__name__)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of exact class matches."""
    return float(np.mean(y_true == y_pred))


@dataclass
class CPCVPBOConfig:
    """
    Configuration for CPCV-PBO evaluation.

    Attributes:
        n_groups: Number of contiguous time groups N
        n_test_groups: Test groups per CPCV split k
        purge_bars: Label span in bars (purge window around test groups)
        embargo_bars: Extra rows embargoed after each test group
        pbo_partitions: CSCV row blocks S (even)
        pbo_warn_threshold: PBO threshold for warning (default 0.5)
        pbo_block_threshold: PBO threshold for blocking (default 0.8)
        output_dir: Directory for evaluation outputs
    """

    n_groups: int = 6
    n_test_groups: int = 2
    purge_bars: int = 60
    embargo_bars: int = 0
    pbo_partitions: int = 16
    pbo_warn_threshold: float = 0.5
    pbo_block_threshold: float = 0.8
    output_dir: str = "experiments/evaluation/cpcv_pbo"


class CPCVPBOEvaluator:
    """
    CPCV path backtests + CSCV PBO for model selection.

    Example:
        >>> evaluator = CPCVPBOEvaluator({"n_groups": 6, "n_test_groups": 2, "purge_bars": 20})
        >>> result = evaluator.run(X, y, {"xgb": xgb, "lgbm": lgbm}, forward_returns=fwd)
        >>> print(result["pbo_result"]["pbo"])
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
            defaults = CPCVPBOConfig()
            self.config = CPCVPBOConfig(
                **{k: config.get(k, getattr(defaults, k)) for k in defaults.__dict__}
            )

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cpcv = CombinatorialPurgedCV(
            CPCVConfig(
                n_groups=self.config.n_groups,
                n_test_groups=self.config.n_test_groups,
                purge_bars=self.config.purge_bars,
                embargo_bars=self.config.embargo_bars,
            )
        )
        self.pbo_config = PBOConfig(
            n_partitions=self.config.pbo_partitions,
            warn_threshold=self.config.pbo_warn_threshold,
            block_threshold=self.config.pbo_block_threshold,
        )
        logger.info(f"Initialized CPCVPBOEvaluator: {self.cpcv}")

    def _evaluate_model(
        self,
        name: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: np.ndarray | None,
        label_end_times: pd.Series | None,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> tuple[CPCVResult, np.ndarray | None]:
        """CPCV one model; return its path results and path-averaged per-bar returns."""
        n_samples = len(X)
        split_preds: dict[int, np.ndarray] = {}
        for train_idx, test_idx, split_id in self.cpcv.split(X, y, label_end_times=label_end_times):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            split_preds[split_id] = np.asarray(model.predict(X.iloc[test_idx]), dtype=np.float64)

        pred_paths = self.cpcv.assemble_paths(split_preds, n_samples)
        assignments = self.cpcv.get_path_assignments()
        y_true = y.to_numpy()

        path_results = []
        path_returns = []
        for p in range(self.cpcv.n_paths):
            returns = None
            sharpe = 0.0
            if forward_returns is not None:
                returns = directional_strategy_returns(pred_paths[p], forward_returns)
                sharpe = sharpe_ratio_per_period(returns)
                path_returns.append(returns)
            path_results.append(
                CPCVPathResult(
                    path_id=p,
                    split_ids=tuple(int(s) for s in assignments[p]),
                    n_samples=n_samples,
                    accuracy=float(metric_fn(y_true, pred_paths[p])),
                    sharpe=sharpe,
                    returns=returns,
                )
            )

        mean_returns = np.mean(path_returns, axis=0) if path_returns else None
        return (
            CPCVResult(config=self.cpcv.config, path_results=path_results, model_name=name),
            mean_returns,
        )

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Mapping[str, Any] | Any,
        forward_returns: pd.Series | np.ndarray | None = None,
        label_end_times: pd.Series | None = None,
        metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> dict[str, Any]:
        """
        Run CPCV for every candidate and PBO across candidates.

        Args:
            X: Feature DataFrame (rows in time order)
            y: Directional labels in {-1, 0, +1}
            models: One model, or a mapping name -> model (fit/predict API).
                Each candidate is one strategy configuration for PBO.
            forward_returns: Return from bar t to t+1 aligned with X. Required
                for path Sharpe ratios and PBO.
            label_end_times: Optional label end times for label-aware purging
            metric_fn: Path metric on (y_true, y_pred) (default: accuracy)

        Returns:
            Dict with per-model CPCV results, the PBO result (None with fewer
            than two candidates or no forward returns), and a recommendation.
        """
        start_time = time.time()
        candidates = dict(models) if isinstance(models, Mapping) else {"model": models}
        if not candidates:
            raise ValueError("At least one model is required for CPCV-PBO evaluation")

        score_fn = metric_fn if metric_fn is not None else _accuracy
        fwd = None if forward_returns is None else np.asarray(forward_returns, dtype=np.float64)
        if fwd is not None and len(fwd) != len(X):
            raise ValueError(f"forward_returns length {len(fwd)} != len(X) {len(X)}")

        cpcv_results: dict[str, CPCVResult] = {}
        columns: list[np.ndarray] = []
        for name, model in candidates.items():
            result, mean_returns = self._evaluate_model(
                name, model, X, y, fwd, label_end_times, score_fn
            )
            cpcv_results[name] = result
            if mean_returns is not None:
                columns.append(mean_returns)

        pbo_result = None
        if len(columns) >= 2:
            pbo_result = compute_pbo(np.column_stack(columns), self.pbo_config)

        if pbo_result is None:
            recommendation = (
                "INCONCLUSIVE: PBO needs >= 2 candidate strategies and forward returns. "
                "CPCV path metrics reported for reference."
            )
        elif pbo_result.should_block:
            recommendation = f"BLOCK: High overfitting risk (PBO={pbo_result.pbo:.3f})"
        elif pbo_result.is_overfit:
            recommendation = f"WARN: Moderate overfitting risk (PBO={pbo_result.pbo:.3f})"
        else:
            recommendation = f"PROCEED: Low overfitting risk (PBO={pbo_result.pbo:.3f})"

        total_time = time.time() - start_time
        logger.info(f"CPCV-PBO evaluation completed in {total_time:.2f}s: {recommendation}")

        return {
            "cpcv_results": {name: r.to_dict() for name, r in cpcv_results.items()},
            "pbo_result": pbo_result.to_dict() if pbo_result else None,
            "recommendation": recommendation,
            "total_time": total_time,
            "n_paths": self.cpcv.n_paths,
            "n_splits": self.cpcv.get_n_splits(),
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

        min_samples = self.config.n_groups * 100  # At least 100 per group
        if n_samples < min_samples:
            warnings.append(f"Low sample count: {n_samples} < recommended {min_samples}")

        coverage = self.cpcv.validate_coverage(X)
        if coverage["samples_never_in_test"] > 0:
            warnings.append(f"{coverage['samples_never_in_test']} samples never appear in test set")

        return {
            "is_valid": len(warnings) == 0,
            "n_samples": n_samples,
            "n_groups": self.config.n_groups,
            "n_splits": self.cpcv.get_n_splits(),
            "n_paths": self.cpcv.n_paths,
            "coverage": coverage,
            "warnings": warnings,
        }


__all__ = ["CPCVPBOEvaluator", "CPCVPBOConfig"]
