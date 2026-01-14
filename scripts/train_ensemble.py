#!/usr/bin/env python3
"""
Train heterogeneous stacking ensembles from the Model Factory.

Dedicated CLI for training heterogeneous ensembles that mix tabular and sequence
models. Automatically handles dual data loading (2D + 3D) and routes appropriate
data formats to each base model.

Examples:
    # Standard 3-base heterogeneous ensemble
    python scripts/train_ensemble.py --base-models catboost,tcn,patchtst \
        --meta-learner logistic --horizon 20

    # 4-base ensemble with custom meta-learner
    python scripts/train_ensemble.py --base-models xgboost,lightgbm,lstm,transformer \
        --meta-learner ridge_meta --horizon 20

    # Minimal 2-base ensemble for quick testing
    python scripts/train_ensemble.py --base-models xgboost,lstm \
        --meta-learner logistic --horizon 20 --n-folds 3

    # List available base models and meta-learners
    python scripts/train_ensemble.py --list-models

Usage:
    python scripts/train_ensemble.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for training."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train heterogeneous stacking ensembles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 3-base heterogeneous ensemble (CatBoost + TCN + PatchTST)
  python scripts/train_ensemble.py --base-models catboost,tcn,patchtst \\
      --meta-learner logistic --horizon 20

  # 4-base maximum diversity ensemble
  python scripts/train_ensemble.py --base-models lightgbm,tcn,tft,random_forest \\
      --meta-learner ridge_meta --horizon 20

  # Quick 2-base ensemble for prototyping
  python scripts/train_ensemble.py --base-models xgboost,lstm \\
      --meta-learner logistic --horizon 20 --n-folds 3

  # List available models
  python scripts/train_ensemble.py --list-models
        """,
    )

    # Required arguments
    parser.add_argument(
        "--base-models",
        type=str,
        required=False,
        help="Comma-separated base model names (e.g., catboost,tcn,patchtst)",
    )
    parser.add_argument(
        "--meta-learner",
        type=str,
        default="logistic",
        help="Meta-learner model name (default: logistic). Options: logistic, ridge_meta, mlp_meta, calibrated_meta, xgboost_meta",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Label horizon (default: 20)",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        help=(
            "Feature set name (e.g., 'boosting_optimal', 'ensemble_base'). "
            "If not specified, uses 'ensemble_base' for heterogeneous stacking."
        ),
    )

    # Stacking configuration
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds for OOF generation (default: 5)",
    )
    parser.add_argument(
        "--purge-bars",
        type=int,
        default=60,
        help="Purge bars for CV splits (default: 60)",
    )
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=1440,
        help="Embargo bars for CV splits (default: 1440)",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/splits/scaled"),
        help="Path to scaled splits directory (default: data/splits/scaled)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Output directory for artifacts (default: experiments/runs)",
    )

    # Sequence model arguments
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length for sequence models (default: 60)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )

    # Test set evaluation
    parser.add_argument(
        "--evaluate-test",
        dest="evaluate_test",
        action="store_true",
        default=True,
        help="Evaluate on test set (default: True)",
    )
    parser.add_argument(
        "--no-evaluate-test",
        dest="evaluate_test",
        action="store_false",
        help="Skip test set evaluation",
    )

    # Utility arguments
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available base models and meta-learners",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip saving artifacts (for quick testing)",
    )

    return parser.parse_args()


def list_models() -> None:
    """Print available base models and meta-learners."""
    from src.models.registry import ModelRegistry
    import src.models  # noqa: F401

    print("\nHeterogeneous Ensemble Components")
    print("=" * 70)

    # Get models by family
    models_by_family = ModelRegistry.list_models()

    print("\nTABULAR BASE MODELS (2D input):")
    print("-" * 40)
    for family in ["boosting", "classical"]:
        if family in models_by_family:
            for model_name in sorted(models_by_family[family]):
                if not model_name.endswith("_meta"):
                    info = ModelRegistry.get_model_info(model_name)
                    desc = info.get("description", "")[:50]
                    print(f"  {model_name:20} - {desc}")

    print("\nSEQUENCE BASE MODELS (3D input):")
    print("-" * 40)
    for family in ["neural", "cnn", "mlp", "advanced"]:
        if family in models_by_family:
            for model_name in sorted(models_by_family[family]):
                info = ModelRegistry.get_model_info(model_name)
                desc = info.get("description", "")[:50]
                print(f"  {model_name:20} - {desc}")

    print("\nMETA-LEARNERS (for stacking):")
    print("-" * 40)
    meta_learners = ["logistic", "ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]
    for name in meta_learners:
        if ModelRegistry.is_registered(name):
            info = ModelRegistry.get_model_info(name)
            desc = info.get("description", "")[:50]
            print(f"  {name:20} - {desc}")

    print("\nRECOMMENDED CONFIGURATIONS:")
    print("-" * 40)
    print("  3-Base Standard:    catboost,tcn,patchtst + logistic")
    print("  4-Base Maximum:     lightgbm,tcn,tft,random_forest + ridge_meta")
    print("  2-Base Minimal:     xgboost,lstm + logistic")

    print(f"\nTotal registered models: {ModelRegistry.count()}")


def classify_models(model_names: list[str]) -> tuple[list[str], list[str]]:
    """Classify models into tabular and sequence families."""
    from src.models.registry import ModelRegistry
    from src.models.ensemble.validator import classify_base_models

    return classify_base_models(model_names)


def build_stacking_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build stacking ensemble configuration from CLI args."""
    base_models = [m.strip().lower() for m in args.base_models.split(",") if m.strip()]

    config = {
        "base_model_names": base_models,
        "meta_learner_name": args.meta_learner.strip().lower(),
        "n_folds": args.n_folds,
        "purge_bars": args.purge_bars,
        "embargo_bars": args.embargo_bars,
        "use_probabilities": True,
        "passthrough": False,
        "use_default_configs_for_oof": True,
    }

    # Add feature set if specified (defaults to 'ensemble_base' for heterogeneous stacking)
    if args.feature_set:
        config["feature_set"] = args.feature_set.strip().lower()

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Handle utility commands
    if args.list_models:
        list_models()
        return 0

    # Validate required arguments
    if not args.base_models:
        print("Error: --base-models is required")
        print("Example: --base-models catboost,tcn,patchtst")
        print("Use --list-models to see available models")
        return 1

    # Import models to ensure registration
    logger.info("Loading model registry...")
    from src.models.registry import ModelRegistry
    import src.models  # noqa: F401

    # Parse and validate base models
    base_model_names = [m.strip().lower() for m in args.base_models.split(",") if m.strip()]
    if len(base_model_names) < 2:
        print("Error: At least 2 base models required for stacking")
        return 1

    invalid = [m for m in base_model_names if not ModelRegistry.is_registered(m)]
    if invalid:
        print(f"Error: Unknown base models: {invalid}")
        print(f"Available: {ModelRegistry.list_all()}")
        return 1

    # Validate meta-learner
    if not ModelRegistry.is_registered(args.meta_learner):
        print(f"Error: Unknown meta-learner '{args.meta_learner}'")
        print("Available: logistic, ridge_meta, mlp_meta, calibrated_meta, xgboost_meta")
        return 1

    # Classify base models
    tabular_models, sequence_models = classify_models(base_model_names)
    is_heterogeneous = len(tabular_models) > 0 and len(sequence_models) > 0

    print("\n" + "=" * 70)
    print("HETEROGENEOUS STACKING ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"Base Models: {base_model_names}")
    print(f"  Tabular ({len(tabular_models)}): {tabular_models}")
    print(f"  Sequence ({len(sequence_models)}): {sequence_models}")
    print(f"Meta-Learner: {args.meta_learner}")
    print(f"Heterogeneous: {is_heterogeneous}")
    print(f"CV Folds: {args.n_folds}")
    print(f"Horizon: {args.horizon}")
    print("=" * 70 + "\n")

    # Load data
    logger.info(f"Loading data from {args.data_dir}...")
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

    data_path = PROJECT_ROOT / args.data_dir
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return 1

    try:
        container = TimeSeriesDataContainer.from_parquet_dir(
            path=data_path,
            horizon=args.horizon,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    logger.info(f"Loaded: {container}")

    # Build stacking configuration
    stacking_config = build_stacking_config(args)

    # Create trainer config
    from src.models.config import create_trainer_config

    try:
        trainer_config = create_trainer_config(
            model_name="stacking",
            horizon=args.horizon,
            cli_args=stacking_config,
            config_file=None,
        )
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Override settings from CLI
    if args.output_dir:
        trainer_config.output_dir = PROJECT_ROOT / args.output_dir
    trainer_config.device = args.device
    trainer_config.evaluate_test_set = args.evaluate_test

    # Add sequence length to model config for sequence models
    if sequence_models:
        trainer_config.model_config["sequence_length"] = args.seq_len
        logger.info(f"Using sequence length: {args.seq_len} for sequence models")

    # Run training
    from src.models.trainer import Trainer

    logger.info("Starting heterogeneous stacking training...")
    trainer = Trainer(trainer_config)

    try:
        results = trainer.run(container, skip_save=args.skip_save)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Run ID: {results['run_id']}")
    print(f"Model: stacking ({len(base_model_names)} bases + {args.meta_learner})")
    print(f"Horizon: {results['horizon']}")

    print(f"\nValidation Metrics:")
    eval_metrics = results["evaluation_metrics"]
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {eval_metrics['macro_f1']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")

    # Trading metrics (validation)
    if "trading" in eval_metrics:
        trading = eval_metrics["trading"]
        print(f"\n  Trading Metrics (Validation):")
        print(f"    Position Win Rate: {trading.get('position_win_rate', 0):.4f}")
        print(f"    Long Accuracy: {trading.get('long_accuracy', 0):.4f}")
        print(f"    Short Accuracy: {trading.get('short_accuracy', 0):.4f}")

    print(f"\nPer-Class F1:")
    for cls, f1 in eval_metrics.get("per_class_f1", {}).items():
        print(f"  {cls}: {f1:.4f}")

    # Test set metrics (if evaluated)
    if results.get("test_metrics") is not None:
        print("\n" + "=" * 70)
        print("TEST SET RESULTS (ONE-SHOT GENERALIZATION ESTIMATE)")
        print("=" * 70)

        test_metrics = results["test_metrics"]
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")

    print(f"\nTraining Time: {results['total_time_seconds']:.1f}s")
    if not args.skip_save:
        print(f"Output: {results['output_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
