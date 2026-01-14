#!/usr/bin/env python3
"""
Walk-Forward Evaluation CLI.

Runs walk-forward analysis on trained models to evaluate performance
across expanding or rolling time windows. More realistic than k-fold
for trading applications as it respects temporal ordering.

Usage:
    # Basic walk-forward with 5 windows
    python scripts/run_walk_forward.py --models xgboost --horizons 20

    # Rolling window with custom settings
    python scripts/run_walk_forward.py --models xgboost,lightgbm \\
        --window-type rolling --n-windows 10 --min-train-pct 0.3

    # All models, all horizons
    python scripts/run_walk_forward.py --models all --horizons all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardResult,
    WindowMetrics,
)
from src.cross_validation.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
from src.models.registry import ModelRegistry
from src.models.base import PredictionOutput
from src.phase1.stages.datasets.container import TimeSeriesDataContainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "splits" / "scaled"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "walk_forward"
DEFAULT_HORIZONS = [5, 10, 15, 20]
DEFAULT_N_WINDOWS = 5
DEFAULT_MIN_TRAIN_PCT = 0.4
DEFAULT_TEST_PCT = 0.1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def parse_model_list(model_arg: str) -> List[str]:
    """Parse model argument into list of model names."""
    if model_arg.lower() == "all":
        return ModelRegistry.list_all()

    models = [m.strip().lower() for m in model_arg.split(",")]

    # Validate models exist
    available = ModelRegistry.list_all()
    invalid = [m for m in models if m not in available]
    if invalid:
        logger.error(f"Unknown models: {invalid}. Available: {available}")
        sys.exit(1)

    return models


def parse_horizon_list(horizon_arg: str) -> List[int]:
    """Parse horizon argument into list of integers."""
    if horizon_arg.lower() == "all":
        return DEFAULT_HORIZONS

    try:
        return [int(h.strip()) for h in horizon_arg.split(",")]
    except ValueError as e:
        logger.error(f"Invalid horizon format: {e}")
        sys.exit(1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# =============================================================================
# WALK-FORWARD RUNNER
# =============================================================================


def run_walk_forward_evaluation(
    container: TimeSeriesDataContainer,
    model_name: str,
    config: WalkForwardConfig,
    sample_weights: Optional[pd.Series] = None,
    label_end_times: Optional[pd.Series] = None,
) -> WalkForwardResult:
    """
    Run walk-forward evaluation for a single model.

    Args:
        container: Data container with train/val/test splits
        model_name: Name of the model to evaluate
        config: WalkForwardConfig
        sample_weights: Optional sample weights
        label_end_times: Optional label end times for purging

    Returns:
        WalkForwardResult with metrics and predictions
    """
    start_time = time.time()

    # Get training data (we use train split for walk-forward)
    X, y, weights = container.get_sklearn_arrays("train", return_df=True)

    if sample_weights is None:
        sample_weights = weights

    n_samples = len(X)
    n_classes = 3  # short, neutral, long

    # Initialize prediction storage
    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full((n_samples, n_classes), np.nan)
    all_confidence = np.full(n_samples, np.nan)

    # Create evaluator
    wf = WalkForwardEvaluator(config)
    window_metrics: List[WindowMetrics] = []

    # Get scaling method for model
    scaling_method = get_scaling_method_for_model(model_name)

    logger.info(f"Running walk-forward for {model_name} ({config.n_windows} windows)")

    for window_idx, (train_idx, test_idx) in enumerate(
        wf.split(X, y, label_end_times=label_end_times)
    ):
        window_start = time.time()

        logger.debug(f"  Window {window_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")

        # Extract window data
        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fold-aware scaling
        scaler = FoldAwareScaler(method=scaling_method)
        scaling_result = scaler.fit_transform_fold(X_train_raw.values, X_test_raw.values)
        X_train_scaled = scaling_result.X_train_scaled
        X_test_scaled = scaling_result.X_val_scaled

        # Handle sample weights
        w_train = None
        if sample_weights is not None:
            w_train = sample_weights.iloc[train_idx].values

        # Create and train model
        model = ModelRegistry.create(model_name)
        training_metrics = model.fit(
            X_train=X_train_scaled,
            y_train=y_train.values,
            X_val=X_test_scaled,
            y_val=y_test.values,
            sample_weights=w_train,
        )

        # Generate predictions
        prediction_output: PredictionOutput = model.predict(X_test_scaled)

        # Store predictions
        all_preds[test_idx] = prediction_output.class_predictions
        all_probs[test_idx] = prediction_output.class_probabilities
        all_confidence[test_idx] = prediction_output.confidence

        # Compute metrics
        metrics = compute_metrics(y_test.values, prediction_output.class_predictions)
        window_time = time.time() - window_start

        # Build window metrics
        has_datetime = isinstance(X.index, pd.DatetimeIndex)
        window_metric = WindowMetrics(
            window=window_idx,
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_start_idx=int(train_idx[0]),
            train_end_idx=int(train_idx[-1]),
            test_start_idx=int(test_idx[0]),
            test_end_idx=int(test_idx[-1]),
            train_start_time=X.index[train_idx[0]] if has_datetime else None,
            train_end_time=X.index[train_idx[-1]] if has_datetime else None,
            test_start_time=X.index[test_idx[0]] if has_datetime else None,
            test_end_time=X.index[test_idx[-1]] if has_datetime else None,
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            training_time=window_time,
        )
        window_metrics.append(window_metric)

        logger.info(
            f"  Window {window_idx + 1}: acc={metrics['accuracy']:.3f}, "
            f"f1={metrics['f1']:.3f}, time={window_time:.1f}s"
        )

    # Build predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "datetime": X.index if isinstance(X.index, pd.DatetimeIndex) else range(len(X)),
            f"{model_name}_pred": all_preds,
            f"{model_name}_prob_short": all_probs[:, 0],
            f"{model_name}_prob_neutral": all_probs[:, 1],
            f"{model_name}_prob_long": all_probs[:, 2],
            f"{model_name}_confidence": all_confidence,
            "y_true": y.values,
        }
    )

    total_time = time.time() - start_time

    return WalkForwardResult(
        model_name=model_name,
        horizon=container.horizon,
        window_metrics=window_metrics,
        predictions=predictions_df,
        config=config,
        total_time=total_time,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward evaluation on ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, single horizon
  python scripts/run_walk_forward.py --models xgboost --horizons 20

  # Multiple models with rolling window
  python scripts/run_walk_forward.py --models xgboost,lightgbm \\
      --window-type rolling --n-windows 10

  # All models, all horizons
  python scripts/run_walk_forward.py --models all --horizons all

  # Custom window configuration
  python scripts/run_walk_forward.py --models xgboost \\
      --n-windows 8 --min-train-pct 0.3 --test-pct 0.08
        """,
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names or 'all'",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,15,20",
        help="Comma-separated horizons or 'all' (default: 5,10,15,20)",
    )

    # Walk-forward configuration
    parser.add_argument(
        "--n-windows",
        type=int,
        default=DEFAULT_N_WINDOWS,
        help=f"Number of walk-forward windows (default: {DEFAULT_N_WINDOWS})",
    )
    parser.add_argument(
        "--window-type",
        type=str,
        choices=["expanding", "rolling"],
        default="expanding",
        help="Window type: 'expanding' or 'rolling' (default: expanding)",
    )
    parser.add_argument(
        "--min-train-pct",
        type=float,
        default=DEFAULT_MIN_TRAIN_PCT,
        help=f"Minimum training data percentage (default: {DEFAULT_MIN_TRAIN_PCT})",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=DEFAULT_TEST_PCT,
        help=f"Test window percentage (default: {DEFAULT_TEST_PCT})",
    )
    parser.add_argument(
        "--gap-bars",
        type=int,
        default=0,
        help="Gap bars between train and test (default: 0)",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to scaled data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse arguments
    models = parse_model_list(args.models)
    horizons = parse_horizon_list(args.horizons)

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = WalkForwardConfig(
        n_windows=args.n_windows,
        window_type=args.window_type,
        min_train_pct=args.min_train_pct,
        test_pct=args.test_pct,
        gap_bars=args.gap_bars,
    )

    logger.info("=" * 60)
    logger.info("WALK-FORWARD EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Config: {config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    all_results: List[WalkForwardResult] = []

    for horizon in horizons:
        logger.info("-" * 60)
        logger.info(f"HORIZON {horizon}")
        logger.info("-" * 60)

        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=args.data_dir,
                horizon=horizon,
            )
            logger.info(f"Loaded container: {container}")
        except Exception as e:
            logger.error(f"Failed to load data for H{horizon}: {e}")
            continue

        # Get label end times if available
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info("  Using label_end_times for overlap-aware purging")

        for model_name in models:
            try:
                result = run_walk_forward_evaluation(
                    container=container,
                    model_name=model_name,
                    config=config,
                    label_end_times=label_end_times,
                )
                all_results.append(result)

                logger.info(
                    f"  {model_name} H{horizon}: "
                    f"mean_acc={result.mean_accuracy:.3f} "
                    f"(std={result.std_accuracy:.3f}), "
                    f"mean_f1={result.mean_f1:.3f}, "
                    f"time={result.total_time:.1f}s"
                )

                # Save individual result
                result_path = args.output_dir / f"wf_{model_name}_h{horizon}.json"
                with open(result_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)

                # Save predictions
                pred_path = args.output_dir / f"wf_preds_{model_name}_h{horizon}.parquet"
                result.predictions.to_parquet(pred_path, index=False)

            except Exception as e:
                logger.error(f"  Failed {model_name} H{horizon}: {e}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                continue

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if all_results:
        summary_data = []
        for r in all_results:
            summary_data.append(
                {
                    "model": r.model_name,
                    "horizon": r.horizon,
                    "mean_acc": f"{r.mean_accuracy:.3f}",
                    "std_acc": f"{r.std_accuracy:.3f}",
                    "mean_f1": f"{r.mean_f1:.3f}",
                    "n_windows": r.n_windows,
                    "time_s": f"{r.total_time:.1f}",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Save summary
        summary_path = args.output_dir / "walk_forward_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.warning("No results generated")

    logger.info(f"Results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
