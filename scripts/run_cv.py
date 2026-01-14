#!/usr/bin/env python3
"""
Cross-Validation Runner CLI.

Run purged k-fold cross-validation for Phase 3.
Generates out-of-fold predictions for ensemble stacking in Phase 4.

Usage:
    python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,20
    python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100
    python scripts/run_cv.py --models all --horizons all --no-feature-selection

Requirements:
    - Phase 1 data in data/splits/scaled/
    - Phase 2 models registered in ModelRegistry
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig
from src.cross_validation.cv_runner import (
    CrossValidationRunner,
    analyze_cv_stability,
)
from src.cross_validation.oof_generator import analyze_prediction_correlation
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.models.registry import ModelRegistry

# Import models to register them
import src.models  # noqa: F401

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "stacking"
DEFAULT_HORIZONS = [5, 10, 15, 20]
DEFAULT_PURGE_BARS = 60  # 3x max horizon (20)
DEFAULT_EMBARGO_BARS = 1440  # 5 trading days at 5-min


def generate_cv_run_id() -> str:
    """
    Generate unique run ID for CV output directory.

    Format: {timestamp_with_ms}_{random_suffix}
    Example: 20251228_143025_789456_a3f9

    Prevents collision between parallel CV runs.
    """
    import secrets
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
    return f"{timestamp}_{random_suffix}"


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run purged k-fold cross-validation for Phase 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CV for XGBoost and LightGBM on all horizons
  python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,15,20

  # Run CV with hyperparameter tuning
  python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100

  # Run CV for all available models
  python scripts/run_cv.py --models all --horizons all

  # Run without feature selection (use all features)
  python scripts/run_cv.py --models xgboost --no-feature-selection
        """,
    )

    # Model and horizon selection
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model names or 'all' for all registered models",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,15,20",
        help="Comma-separated horizons or 'all' for default horizons (default: 5,10,15,20)",
    )

    # CV configuration
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5, reduced to 3 for neural models)",
    )
    parser.add_argument(
        "--purge-bars",
        type=int,
        default=DEFAULT_PURGE_BARS,
        help=f"Purge bars before test set (default: {DEFAULT_PURGE_BARS})",
    )
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=DEFAULT_EMBARGO_BARS,
        help=f"Embargo bars after test set (default: {DEFAULT_EMBARGO_BARS})",
    )

    # Feature selection
    parser.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Disable walk-forward feature selection",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=50,
        help="Number of features to select per fold (default: 50)",
    )

    # Hyperparameter tuning
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna hyperparameter tuning",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )

    # Input/output paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Input data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom subdirectory name for this CV run (default: auto-generated timestamp)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


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


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse model and horizon lists
    models = parse_model_list(args.models)
    horizons = parse_horizon_list(args.horizons)

    logger.info(f"Models: {models}")
    logger.info(f"Horizons: {horizons}")

    # Validate data directory exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Run Phase 1 pipeline first to generate scaled data")
        return 1

    # Generate unique run ID for this CV run
    cv_run_id = args.output_name if args.output_name else generate_cv_run_id()
    cv_output_dir = args.output_dir / cv_run_id

    # Create run-specific output directory
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CV output directory: {cv_output_dir}")

    # Configure CV
    cv_config = PurgedKFoldConfig(
        n_splits=args.n_splits,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )
    cv = PurgedKFold(cv_config)

    logger.info(f"CV config: {cv}")

    # Process each horizon
    all_results = {}
    all_stacking_datasets = {}

    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing horizon H{horizon}")
        logger.info(f"{'='*60}")

        # Load data container
        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=args.data_dir,
                horizon=horizon,
            )
            logger.info(f"Loaded container: {container}")
        except Exception as e:
            logger.error(f"Failed to load data for H{horizon}: {e}")
            continue

        # Create CV runner
        runner = CrossValidationRunner(
            cv=cv,
            models=models,
            horizons=[horizon],  # Process one horizon at a time
            tune_hyperparams=args.tune,
            select_features=not args.no_feature_selection,
            n_features_to_select=args.n_features,
            tuning_trials=args.n_trials,
        )

        # Run CV
        try:
            cv_results = runner.run(container)
            all_results.update(cv_results)

            # Build stacking dataset
            stacking_datasets = runner.build_stacking_datasets(cv_results, container)
            all_stacking_datasets.update(stacking_datasets)

        except Exception as e:
            logger.error(f"CV failed for H{horizon}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue

    if not all_results:
        logger.error("No CV results generated. Check errors above.")
        return 1

    # Analyze stability
    logger.info("\n" + "=" * 60)
    logger.info("STABILITY ANALYSIS")
    logger.info("=" * 60)

    stability_df = analyze_cv_stability(all_results)
    print("\n" + stability_df.to_string(index=False))

    # Analyze prediction correlation (if multiple models)
    if len(models) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION CORRELATION ANALYSIS")
        logger.info("=" * 60)

        for horizon, stacking_ds in all_stacking_datasets.items():
            logger.info(f"\nHorizon H{horizon}:")
            corr_df = analyze_prediction_correlation(
                stacking_ds.data,
                stacking_ds.model_names,
            )
            print(corr_df.to_string(index=False))

    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    # Create fresh CV runner for saving (with all horizons)
    save_runner = CrossValidationRunner(
        cv=cv,
        models=models,
        horizons=horizons,
        tune_hyperparams=args.tune,
        select_features=not args.no_feature_selection,
    )
    save_runner.save_results(all_results, all_stacking_datasets, cv_output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for (model_name, horizon), result in all_results.items():
        logger.info(
            f"{model_name}/H{horizon}: "
            f"F1={result.mean_f1:.3f} (+/- {result.std_f1:.3f}), "
            f"Stability={result.get_stability_score():.3f}, "
            f"Time={result.total_time:.1f}s"
        )

    logger.info(f"\nCV Run ID: {cv_run_id}")
    logger.info(f"Results saved to: {cv_output_dir}")
    logger.info(f"Stacking datasets saved to: {cv_output_dir / 'stacking'}")
    logger.info(f"\nTo use in Phase 4:")
    logger.info(
        f"  python scripts/train_model.py --model stacking --horizon <H> --stacking-data {cv_run_id}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
