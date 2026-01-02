#!/usr/bin/env python3
"""
Train any registered model from the Model Factory.

CLI interface to train models using Phase 1 datasets and the Trainer class.
Supports all registered model types (boosting, neural, etc.) with config
loading from YAML files and CLI overrides.

Examples:
    # Train XGBoost with defaults
    python scripts/train_model.py --model xgboost --horizon 20

    # Train a voting ensemble from scratch
    python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

    # Train LSTM with custom config
    python scripts/train_model.py --model lstm --horizon 20 --hidden-size 256 --seq-len 60

    # Train TCN with longer sequences
    python scripts/train_model.py --model tcn --horizon 20 --seq-len 120

    # Train with custom data path
    python scripts/train_model.py --model xgboost --horizon 20 --data-dir data/splits/scaled

Usage:
    python scripts/train_model.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def load_phase3_stacking_data(
    cv_run_id: str,
    horizon: int,
    phase3_base_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load Phase 3 stacking dataset for ensemble training.

    Args:
        cv_run_id: CV run ID (e.g., '20251228_143025_789456_a3f9')
        horizon: Label horizon
        phase3_base_dir: Base directory for Phase 3 outputs

    Returns:
        Dict with 'data' (DataFrame) and 'metadata' (dict), or None if not found

    Raises:
        FileNotFoundError: If stacking data does not exist
        ValueError: If data format is invalid
    """
    import json
    import pandas as pd

    logger = logging.getLogger(__name__)

    # Construct paths
    stacking_dir = PROJECT_ROOT / phase3_base_dir / cv_run_id / "stacking"
    parquet_path = stacking_dir / f"stacking_dataset_h{horizon}.parquet"
    metadata_path = stacking_dir / f"stacking_metadata_h{horizon}.json"

    # Validate existence
    if not stacking_dir.exists():
        raise FileNotFoundError(
            f"Phase 3 CV run not found: {cv_run_id}\n"
            f"Expected directory: {stacking_dir}\n"
            f"Available runs: {list((PROJECT_ROOT / phase3_base_dir).glob('*'))}"
        )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Stacking dataset not found for H{horizon}\n"
            f"Expected: {parquet_path}\n"
            f"Available: {list(stacking_dir.glob('stacking_dataset_*.parquet'))}"
        )

    # Load data
    logger.info(f"Loading Phase 3 stacking data from {parquet_path}")
    data = pd.read_parquet(parquet_path)

    # Load metadata if available
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Validate data structure
    if "y_true" not in data.columns:
        raise ValueError(
            f"Invalid stacking dataset: missing 'y_true' column\n"
            f"Columns: {data.columns.tolist()}"
        )

    logger.info(
        f"Loaded stacking dataset: {len(data)} samples, "
        f"{len(data.columns) - 1} features, "
        f"models: {metadata.get('model_names', 'unknown')}"
    )

    return {
        "data": data,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train any registered model from the Model Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost (boosting)
  python scripts/train_model.py --model xgboost --horizon 20

  # Train LSTM (neural)
  python scripts/train_model.py --model lstm --horizon 20 --hidden-size 256

  # Train TCN (neural)
  python scripts/train_model.py --model tcn --horizon 20 --seq-len 120

  # Train a voting ensemble from scratch
  python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

  # Train stacking ensemble using Phase 3 CV data (recommended workflow)
  python scripts/train_model.py --model stacking --horizon 20 --stacking-data 20251228_143025_789456_a3f9

  # List available models
  python scripts/train_model.py --list-models
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., xgboost, lstm, tcn, gru)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Label horizon (default: 20)",
    )

    # Ensemble arguments
    parser.add_argument(
        "--base-models",
        type=str,
        help="Comma-separated base model names for ensemble models (voting/stacking/blending)",
    )
    parser.add_argument(
        "--meta-learner",
        type=str,
        help="Meta-learner model name for stacking/blending (default: logistic)",
    )
    parser.add_argument(
        "--voting",
        type=str,
        choices=["soft", "hard"],
        help="Voting strategy for VotingEnsemble (soft/hard)",
    )

    # Phase 3→4 integration (stacking from CV)
    parser.add_argument(
        "--stacking-data",
        type=str,
        help="CV run ID for loading Phase 3 stacking data (e.g., '20251228_143025_789456_a3f9')",
    )
    parser.add_argument(
        "--phase3-output",
        type=Path,
        default=Path("data/stacking"),
        help="Base directory for Phase 3 CV outputs (default: data/stacking)",
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

    # Training arguments (apply to all models)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Early stopping patience (epochs)",
    )

    # Sequence model arguments
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Sequence length for sequential models (LSTM, GRU, TCN)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="Hidden size for RNN/TCN models",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers for RNN models",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate",
    )

    # TCN-specific arguments
    parser.add_argument(
        "--num-channels",
        type=str,
        help="Comma-separated channel sizes for TCN (e.g., '64,64,64,64')",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        help="Kernel size for TCN",
    )

    # Boosting-specific arguments
    parser.add_argument(
        "--n-estimators",
        type=int,
        help="Number of estimators for boosting models",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Max tree depth for boosting models",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )

    # Test set evaluation
    parser.add_argument(
        "--evaluate-test",
        dest="evaluate_test",
        action="store_true",
        default=True,
        help="Evaluate on test set (default: True, one-shot generalization estimate)",
    )
    parser.add_argument(
        "--no-evaluate-test",
        dest="evaluate_test",
        action="store_false",
        help="Skip test set evaluation (use during development/iteration)",
    )

    # Utility arguments
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file (overrides defaults)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--model-info",
        type=str,
        metavar="MODEL",
        help="Show detailed info about a specific model",
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


def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build config overrides from CLI arguments."""
    overrides: Dict[str, Any] = {}

    # Map CLI args to config keys
    arg_mapping = {
        "batch_size": "batch_size",
        "max_epochs": "max_epochs",
        "learning_rate": "learning_rate",
        "early_stopping_patience": "early_stopping_patience",
        "seq_len": "sequence_length",
        "hidden_size": "hidden_size",
        "num_layers": "num_layers",
        "dropout": "dropout",
        "kernel_size": "kernel_size",
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
    }

    for arg_name, config_key in arg_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_key] = value

    # Handle special cases
    if args.num_channels:
        channels = [int(c.strip()) for c in args.num_channels.split(",")]
        overrides["num_channels"] = channels

    if args.device != "auto":
        overrides["device"] = args.device

    if args.no_mixed_precision:
        overrides["mixed_precision"] = False

    # Ensemble helpers
    if args.base_models:
        overrides["base_model_names"] = [
            m.strip().lower() for m in args.base_models.split(",") if m.strip()
        ]
    if args.meta_learner:
        overrides["meta_learner_name"] = args.meta_learner.strip().lower()
    if args.voting:
        overrides["voting"] = args.voting

    return overrides


def list_models() -> None:
    """Print all available models."""
    from src.models.registry import ModelRegistry

    # Ensure models are imported/registered
    import src.models  # noqa: F401

    print("\nAvailable Models:")
    print("=" * 60)

    models_by_family = ModelRegistry.list_models()

    for family, models in sorted(models_by_family.items()):
        print(f"\n{family.upper()}:")
        for model_name in sorted(models):
            try:
                meta = ModelRegistry.get_metadata(model_name)
                desc = meta.get("description", "")
                print(f"  - {model_name}: {desc}")
            except Exception:
                print(f"  - {model_name}")

    print(f"\nTotal: {ModelRegistry.count()} models")


def show_model_info(model_name: str) -> None:
    """Print detailed info about a model."""
    from src.models.registry import ModelRegistry

    # Ensure models are imported/registered
    import src.models  # noqa: F401

    try:
        info = ModelRegistry.get_model_info(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nModel: {info['name']}")
    print("=" * 60)
    print(f"Family: {info['family']}")
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Requires Scaling: {info['requires_scaling']}")
    print(f"Requires Sequences: {info['requires_sequences']}")
    print("\nDefault Configuration:")
    for key, value in sorted(info["default_config"].items()):
        print(f"  {key}: {value}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Handle utility commands
    if args.list_models:
        list_models()
        return 0

    if args.model_info:
        show_model_info(args.model_info)
        return 0

    # Validate required arguments
    if not args.model:
        print("Error: --model is required")
        print("Use --list-models to see available models")
        return 1

    # Import models to ensure registration
    logger.info("Loading model registry...")
    from src.models.registry import ModelRegistry
    import src.models  # noqa: F401

    # Validate model exists
    if not ModelRegistry.is_registered(args.model):
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {ModelRegistry.list_all()}")
        return 1

    # Validate base models if provided (ensemble convenience)
    config_overrides = build_config_overrides(args)
    base_model_names = config_overrides.get("base_model_names")
    if base_model_names:
        invalid = [m for m in base_model_names if not ModelRegistry.is_registered(m)]
        if invalid:
            print(f"Error: Unknown base models: {invalid}")
            print(f"Available: {ModelRegistry.list_all()}")
            return 1

    # Phase 3→4 workflow: Load stacking data if provided
    stacking_data = None
    if args.stacking_data:
        if args.model not in ["stacking", "blending"]:
            logger.warning(
                f"--stacking-data is only used for stacking/blending models, "
                f"but model is '{args.model}'. Ignoring stacking data."
            )
        else:
            try:
                stacking_data = load_phase3_stacking_data(
                    cv_run_id=args.stacking_data,
                    horizon=args.horizon,
                    phase3_base_dir=args.phase3_output,
                )
                logger.info(
                    f"Using Phase 3 stacking data from CV run: {args.stacking_data}\n"
                    f"This data contains leakage-safe OOF predictions from cross-validation.\n"
                    f"Skipping fresh OOF generation (already done in Phase 3)."
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load Phase 3 stacking data: {e}")
                return 1

    # Load data (Phase 1 datasets OR Phase 3 stacking data)
    container = None
    if not stacking_data:
        # Standard workflow: Load Phase 1 datasets
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
    else:
        # Phase 3→4 workflow: Stacking data already loaded
        logger.info(
            f"Using Phase 3 stacking data (skipping Phase 1 container load)\n"
            f"Samples: {len(stacking_data['data'])}, "
            f"Models: {stacking_data['metadata'].get('model_names', [])}"
        )

    # Get model info for sequence length
    model_info = ModelRegistry.get_model_info(args.model)
    if model_info["requires_sequences"]:
        if "sequence_length" not in config_overrides:
            # Use default from model config
            config_overrides["sequence_length"] = model_info["default_config"].get(
                "sequence_length", 60
            )
        logger.info(f"Using sequence length: {config_overrides.get('sequence_length')}")

    # Create trainer config
    from src.models.config import create_trainer_config

    # Pass config_file to create_trainer_config (will FAIL HARD if invalid)
    config_file = PROJECT_ROOT / args.config if args.config else None

    try:
        trainer_config = create_trainer_config(
            model_name=args.model,
            horizon=args.horizon,
            cli_args=config_overrides,
            config_file=config_file,
        )
    except Exception as e:
        # ConfigError or other exceptions from config loading
        logger.error(f"Configuration error: {e}")
        return 1

    # Override output_dir from CLI if provided
    if args.output_dir:
        trainer_config.output_dir = PROJECT_ROOT / args.output_dir

    # Override device settings from CLI
    trainer_config.device = args.device
    trainer_config.mixed_precision = not args.no_mixed_precision

    # Override test set evaluation from CLI
    trainer_config.evaluate_test_set = args.evaluate_test

    # Run training (different path for stacking with Phase 3 data)
    from src.models.trainer import Trainer

    logger.info(f"Starting training: model={args.model}, horizon={args.horizon}")

    if stacking_data:
        # Phase 3→4 workflow: Train directly on stacking data
        logger.info("Training meta-learner on Phase 3 OOF predictions...")

        # Extract features and labels
        stacking_df = stacking_data["data"]
        feature_cols = [c for c in stacking_df.columns if c not in ("y_true", "datetime")]
        X_stacking = stacking_df[feature_cols].values
        y_stacking = stacking_df["y_true"].values

        # Split into train/val (80/20 on the stacking data)
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_stacking,
            y_stacking,
            test_size=0.2,
            random_state=42,
            stratify=y_stacking,
        )

        logger.info(
            f"Stacking data split: train={len(X_train)}, val={len(X_val)}\n"
            f"Training meta-learner (this should be fast - no OOF generation needed)"
        )

        # Create trainer and train meta-learner directly
        trainer = Trainer(trainer_config)

        # Train the model directly (bypassing container)
        try:
            # Get the meta-learner from the ensemble model
            from src.models.base import PredictionOutput
            import numpy as np

            # Train directly
            training_metrics = trainer.model.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                sample_weights=None,  # No weights for stacking data
                config=trainer_config.model_config,
            )

            # Evaluate
            val_predictions = trainer.model.predict(X_val)
            from src.models.trainer import compute_classification_metrics

            eval_metrics = compute_classification_metrics(
                y_true=y_val,
                y_pred=val_predictions.class_predictions,
                y_proba=val_predictions.class_probabilities,
            )

            # Build results dict
            results = {
                "run_id": trainer.run_id,
                "model_name": trainer_config.model_name,
                "horizon": trainer_config.horizon,
                "training_metrics": training_metrics.to_dict(),
                "evaluation_metrics": eval_metrics,
                "output_path": str(trainer.output_path),
                "total_time_seconds": 0.0,  # Not tracked for stacking workflow
                "val_predictions": val_predictions.class_predictions,
                "val_true": y_val,
                "stacking_source": args.stacking_data,
            }

            # Save artifacts
            if not args.skip_save:
                trainer._setup_output_dir()
                trainer._save_config()
                trainer._save_artifacts(training_metrics, eval_metrics, val_predictions)
                trainer._save_model()

        except Exception as e:
            logger.exception(f"Meta-learner training failed: {e}")
            return 1
    else:
        # Standard workflow: Use container
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
    print(f"Model: {results['model_name']}")
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
        print(f"    Position Sharpe: {trading.get('position_sharpe', 0):.4f}")
        print(f"    Total Positions: {trading.get('total_positions', 0)}")

    print(f"\nPer-Class F1:")
    for cls, f1 in eval_metrics.get("per_class_f1", {}).items():
        print(f"  {cls}: {f1:.4f}")

    # Test set metrics (if evaluated)
    if results.get("test_metrics") is not None:
        print("\n" + "=" * 70)
        print("⚠️  TEST SET RESULTS (ONE-SHOT GENERALIZATION ESTIMATE)")
        print("=" * 70)
        print("WARNING: Do NOT iterate on these results. If you do, you're overfitting to test.")
        print("=" * 70)

        test_metrics = results["test_metrics"]
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")

        # Trading metrics (test)
        if "trading" in test_metrics:
            trading = test_metrics["trading"]
            print(f"\n  Trading Metrics (Test):")
            print(f"    Position Win Rate: {trading.get('position_win_rate', 0):.4f}")
            print(f"    Long Accuracy: {trading.get('long_accuracy', 0):.4f}")
            print(f"    Short Accuracy: {trading.get('short_accuracy', 0):.4f}")
            print(f"    Position Sharpe: {trading.get('position_sharpe', 0):.4f}")
            print(f"    Total Positions: {trading.get('total_positions', 0)}")

        print(f"\nPer-Class F1 (Test):")
        for cls, f1 in test_metrics.get("per_class_f1", {}).items():
            print(f"  {cls}: {f1:.4f}")

        print("\n" + "=" * 70)
        print("If test results are disappointing: DO NOT tune and re-evaluate.")
        print("Move on to the next experiment. Test set discipline is critical.")
        print("=" * 70)

    print(f"\nTraining Time: {results['total_time_seconds']:.1f}s")
    if not args.skip_save:
        print(f"Output: {results['output_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
