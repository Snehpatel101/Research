"""
Evaluation stage runner - Generate post-training evaluation reports.

This stage acts as a thin wrapper that calls the existing evaluation
utilities from src.models.evaluation to generate comprehensive reports
after training completes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationStageResult:
    """Result from running the evaluation stage."""

    model_name: str
    horizon: int
    symbol: str
    success: bool = True
    error: str | None = None

    # Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Financial metrics (if prices provided)
    sharpe_ratio: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None

    # Artifacts
    output_dir: Path | None = None
    report_path: Path | None = None
    chart_paths: dict[str, Path] = field(default_factory=dict)


def run_evaluation(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "model",
    output_dir: str | Path = "./evaluation",
    horizon: int = 20,
    symbol: str = "UNKNOWN",
    prices: np.ndarray | pd.Series | None = None,
    y_proba: np.ndarray | pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    feature_importance: np.ndarray | dict[str, float] | None = None,
) -> EvaluationStageResult:
    """
    Run evaluation stage to generate model evaluation metrics.

    This function computes classification metrics and optionally financial
    metrics if prices are provided. It's designed as a simple wrapper
    that can be used as a pipeline stage.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        model_name: Name of the model being evaluated
        output_dir: Directory to save evaluation artifacts
        horizon: Prediction horizon used during training
        symbol: Trading symbol (e.g., "MES", "MGC")
        prices: Price series for financial metrics (optional)
        y_proba: Prediction probabilities (optional)
        feature_names: Feature names for importance (optional)
        feature_importance: Feature importance scores (optional)

    Returns:
        EvaluationStageResult with metrics and artifact paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running evaluation for {model_name}")
    logger.info(f"Output directory: {output_dir}")

    # Convert to numpy if needed
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    result = EvaluationStageResult(
        model_name=model_name,
        horizon=horizon,
        symbol=symbol,
        output_dir=output_dir,
    )

    try:
        # Compute classification metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        result.accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
        result.precision = float(
            precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        )
        result.recall = float(
            recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        )
        result.f1_score = float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        )

        logger.info(
            f"Classification metrics: acc={result.accuracy:.3f}, " f"f1={result.f1_score:.3f}"
        )

        # Compute financial metrics if prices provided
        if prices is not None:
            try:
                prices_arr = np.asarray(prices)
                result = _compute_financial_metrics(result, y_true_arr, y_pred_arr, prices_arr)
            except Exception as e:
                logger.warning(f"Failed to compute financial metrics: {e}")

        # Generate financial report if available
        if prices is not None:
            try:
                from src.models.evaluation.financial_report import generate_financial_report

                prices_arr = np.asarray(prices)
                report = generate_financial_report(
                    model_name=model_name,
                    horizon=horizon,
                    predictions=y_pred_arr,
                    y_true=y_true_arr,
                    prices=prices_arr,
                    output_dir=output_dir / "financial",
                    feature_names=feature_names,
                    feature_importance=(
                        np.asarray(list(feature_importance.values()))
                        if isinstance(feature_importance, dict)
                        else feature_importance
                    ),
                )
                result.report_path = output_dir / "financial"
                logger.info(f"Generated financial report: Sharpe={report.sharpe_ratio:.2f}")
            except Exception as e:
                logger.warning(f"Failed to generate financial report: {e}")

        result.success = True

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        result.success = False
        result.error = str(e)

    return result


def _compute_financial_metrics(
    result: EvaluationStageResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
) -> EvaluationStageResult:
    """Compute basic financial metrics from predictions and prices."""
    # Simple returns calculation
    price_returns = np.diff(prices) / prices[:-1]

    # Align predictions with returns (shift by 1 for realistic simulation)
    if len(y_pred) > len(price_returns):
        aligned_preds = y_pred[1 : len(price_returns) + 1]
    else:
        aligned_preds = y_pred[: len(price_returns)]
        price_returns = price_returns[: len(aligned_preds)]

    # Strategy returns (long when pred > 0, short when pred < 0)
    strategy_returns = aligned_preds * price_returns

    # Win rate
    wins = np.sum(strategy_returns > 0)
    total_trades = np.sum(aligned_preds != 0)
    result.win_rate = float(wins / total_trades) if total_trades > 0 else 0.0

    # Total return
    result.total_return = float(np.sum(strategy_returns))

    # Sharpe ratio (annualized, assuming daily)
    if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
        result.sharpe_ratio = float(
            np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
        )
    else:
        result.sharpe_ratio = 0.0

    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    result.max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    logger.info(
        f"Financial metrics: sharpe={result.sharpe_ratio:.2f}, "
        f"return={result.total_return:.2%}, win_rate={result.win_rate:.2%}"
    )

    return result


__all__ = [
    "run_evaluation",
    "EvaluationStageResult",
]
