#!/usr/bin/env python3
"""
CPCV and PBO Evaluation CLI.

Runs Combinatorially Purged Cross-Validation (CPCV) and computes
Probability of Backtest Overfitting (PBO) for model selection gating.

Usage:
    # Basic CPCV with PBO computation
    python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizons 20

    # More groups for higher resolution
    python scripts/run_cpcv_pbo.py --models all --n-groups 8 --n-test-groups 2

    # Strict PBO thresholds
    python scripts/run_cpcv_pbo.py --models xgboost --pbo-warn 0.4 --pbo-block 0.7
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

from src.cross_validation.cpcv import (
    CPCVConfig,
    CombinatorialPurgedCV,
    CPCVResult,
    CPCVPathResult,
)
from src.cross_validation.pbo import (
    PBOConfig,
    PBOResult,
    compute_pbo,
    pbo_gate,
    analyze_overfitting_risk,
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "cpcv_pbo"
DEFAULT_HORIZONS = [5, 10, 15, 20]
DEFAULT_N_GROUPS = 6
DEFAULT_N_TEST_GROUPS = 2
DEFAULT_MAX_COMBINATIONS = 15


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


def compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.nanmean(returns)
    std_ret = np.nanstd(returns, ddof=1)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))


# =============================================================================
# CPCV RUNNER
# =============================================================================

def run_cpcv_evaluation(
    container: TimeSeriesDataContainer,
    model_name: str,
    cpcv_config: CPCVConfig,
    label_end_times: Optional[pd.Series] = None,
) -> CPCVResult:
    """
    Run CPCV evaluation for a single model.

    Args:
        container: Data container
        model_name: Name of model to evaluate
        cpcv_config: CPCVConfig
        label_end_times: Optional label end times for purging

    Returns:
        CPCVResult with path metrics
    """
    # Get training data
    X, y, weights = container.get_sklearn_arrays("train", return_df=True)

    cpcv = CombinatorialPurgedCV(cpcv_config)
    path_results: List[CPCVPathResult] = []

    # Get scaling method for model
    scaling_method = get_scaling_method_for_model(model_name)

    logger.info(f"Running CPCV for {model_name} ({cpcv.get_n_splits()} paths)")

    for train_idx, test_idx, path_id in cpcv.split(X, y, label_end_times=label_end_times):
        logger.debug(f"  Path {path_id}: train={len(train_idx)}, test={len(test_idx)}")

        # Extract path data
        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        w_train = weights.iloc[train_idx].values

        # Fold-aware scaling
        scaler = FoldAwareScaler(method=scaling_method)
        scaling_result = scaler.fit_transform_fold(
            X_train_raw.values, X_test_raw.values
        )

        # Create and train model
        model = ModelRegistry.create(model_name)
        training_metrics = model.fit(
            X_train=scaling_result.X_train_scaled,
            y_train=y_train.values,
            X_val=scaling_result.X_val_scaled,
            y_val=y_test.values,
            sample_weights=w_train,
        )

        # Generate predictions
        prediction_output: PredictionOutput = model.predict(scaling_result.X_val_scaled)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = float(accuracy_score(y_test.values, prediction_output.class_predictions))
        f1 = float(f1_score(y_test.values, prediction_output.class_predictions,
                           average="weighted", zero_division=0))

        # Compute simulated returns for PBO
        # Assume: correct prediction = +1%, incorrect = -1%
        correct = (prediction_output.class_predictions == y_test.values).astype(float)
        returns = np.where(correct, 0.01, -0.01)
        sharpe = compute_sharpe(returns)

        # Determine test groups from path structure
        test_groups = tuple()  # Would need to extract from CPCV internals

        path_result = CPCVPathResult(
            path_id=path_id,
            test_groups=test_groups,
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_groups=tuple(),
            accuracy=accuracy,
            f1=f1,
            sharpe=sharpe,
            returns=returns,
        )
        path_results.append(path_result)

        logger.debug(f"    Path {path_id}: acc={accuracy:.3f}, f1={f1:.3f}, sharpe={sharpe:.2f}")

    return CPCVResult(
        config=cpcv_config,
        path_results=path_results,
        model_name=model_name,
        horizon=container.horizon,
    )


def compute_model_pbo(
    cpcv_results: Dict[str, CPCVResult],
    pbo_config: PBOConfig,
) -> PBOResult:
    """
    Compute PBO from multiple model CPCV results.

    Treats each model as a "strategy" for PBO computation.

    Args:
        cpcv_results: Dict mapping model_name to CPCVResult
        pbo_config: PBOConfig

    Returns:
        PBOResult
    """
    model_names = list(cpcv_results.keys())
    n_models = len(model_names)

    if n_models < 2:
        logger.warning("PBO requires at least 2 models for comparison")
        return PBOResult(
            pbo=0.0,
            logit_distribution=np.array([]),
            performance_degradation=1.0,
            rank_correlation=1.0,
            is_overfit=False,
            should_block=False,
            n_paths_evaluated=0,
            best_is_strategy_idx=0,
            best_is_oos_rank=0.5,
            config=pbo_config,
        )

    # Build performance matrix (n_models x n_paths)
    # Use Sharpe ratios as performance metric
    first_result = cpcv_results[model_names[0]]
    n_paths = first_result.n_paths

    perf_matrix = np.zeros((n_models, n_paths))
    for i, model_name in enumerate(model_names):
        result = cpcv_results[model_name]
        for j, path_result in enumerate(result.path_results):
            if j < n_paths:
                perf_matrix[i, j] = path_result.sharpe

    # Compute PBO
    return compute_pbo(perf_matrix, pbo_config)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CPCV and PBO evaluation for model selection gating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CPCV with PBO
  python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizons 20

  # More groups for higher resolution
  python scripts/run_cpcv_pbo.py --models all --n-groups 8

  # Strict PBO thresholds
  python scripts/run_cpcv_pbo.py --models xgboost --pbo-warn 0.4 --pbo-block 0.7
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
        default="20",
        help="Comma-separated horizons (default: 20)",
    )

    # CPCV configuration
    parser.add_argument(
        "--n-groups",
        type=int,
        default=DEFAULT_N_GROUPS,
        help=f"Number of time groups (default: {DEFAULT_N_GROUPS})",
    )
    parser.add_argument(
        "--n-test-groups",
        type=int,
        default=DEFAULT_N_TEST_GROUPS,
        help=f"Groups held out as test (default: {DEFAULT_N_TEST_GROUPS})",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=DEFAULT_MAX_COMBINATIONS,
        help=f"Maximum combinations to evaluate (default: {DEFAULT_MAX_COMBINATIONS})",
    )

    # PBO thresholds
    parser.add_argument(
        "--pbo-warn",
        type=float,
        default=0.5,
        help="PBO warning threshold (default: 0.5)",
    )
    parser.add_argument(
        "--pbo-block",
        type=float,
        default=0.8,
        help="PBO blocking threshold (default: 0.8)",
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
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
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

    # Build configs
    cpcv_config = CPCVConfig(
        n_groups=args.n_groups,
        n_test_groups=args.n_test_groups,
        max_combinations=args.max_combinations,
    )

    pbo_config = PBOConfig(
        warn_threshold=args.pbo_warn,
        block_threshold=args.pbo_block,
    )

    logger.info("=" * 60)
    logger.info("CPCV + PBO EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"CPCV: {cpcv_config.n_groups} groups, {cpcv_config.n_test_groups} test")
    logger.info(f"PBO thresholds: warn={args.pbo_warn}, block={args.pbo_block}")

    all_results = []

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

        # Run CPCV for each model
        cpcv_results: Dict[str, CPCVResult] = {}
        for model_name in models:
            try:
                result = run_cpcv_evaluation(
                    container=container,
                    model_name=model_name,
                    cpcv_config=cpcv_config,
                    label_end_times=label_end_times,
                )
                cpcv_results[model_name] = result

                logger.info(
                    f"  {model_name}: mean_acc={result.mean_accuracy:.3f}, "
                    f"std={result.std_accuracy:.3f}, mean_sharpe={result.mean_sharpe:.2f}"
                )

            except Exception as e:
                logger.error(f"  Failed {model_name}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        # Compute PBO across models
        if len(cpcv_results) >= 2:
            pbo_result = compute_model_pbo(cpcv_results, pbo_config)

            logger.info("-" * 40)
            logger.info(f"PBO Analysis for H{horizon}:")
            logger.info(f"  PBO: {pbo_result.pbo:.3f}")
            logger.info(f"  Risk Level: {pbo_result.get_risk_level()}")
            logger.info(f"  Performance Degradation: {pbo_result.performance_degradation:.2f}")
            logger.info(f"  Rank Correlation: {pbo_result.rank_correlation:.3f}")

            # Gate check
            should_proceed, reason = pbo_gate(pbo_result, strict=False)
            logger.info(f"  Gate Decision: {'PASS' if should_proceed else 'FAIL'}")
            logger.info(f"  Reason: {reason}")

            all_results.append({
                "horizon": horizon,
                "n_models": len(cpcv_results),
                "pbo": pbo_result.pbo,
                "is_overfit": pbo_result.is_overfit,
                "should_block": pbo_result.should_block,
                "risk_level": pbo_result.get_risk_level(),
                "gate_pass": should_proceed,
            })

            # Save PBO result
            pbo_path = args.output_dir / f"pbo_h{horizon}.json"
            with open(pbo_path, "w") as f:
                json.dump(pbo_result.to_dict(), f, indent=2)

        # Save CPCV results
        for model_name, result in cpcv_results.items():
            result_path = args.output_dir / f"cpcv_{model_name}_h{horizon}.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))

        # Save summary
        summary_path = args.output_dir / "cpcv_pbo_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")

        # Check for any failures
        any_blocked = any(r["should_block"] for r in all_results)
        if any_blocked:
            logger.warning("Some horizons have PBO > block threshold!")
            return 1
    else:
        logger.warning("No results generated")

    logger.info(f"Results saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
