#!/usr/bin/env python3
"""
Train any registered model from the Model Factory.

CLI interface to train models using Phase 1 datasets and the Trainer class.
Supports all registered model types (boosting, neural, etc.) with config
loading from YAML files and CLI overrides.

Examples:
    # Train XGBoost with defaults
    python scripts/train_model.py --model xgboost --horizon 20

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
        "-v", "--verbose",
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

    return overrides


def list_models() -> None:
    """Print all available models."""
    from src.models.registry import ModelRegistry

    # Ensure models are imported/registered
    import src.models.boosting  # noqa: F401
    import src.models.neural  # noqa: F401

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
    import src.models.boosting  # noqa: F401
    import src.models.neural  # noqa: F401

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
    import src.models.boosting  # noqa: F401
    import src.models.neural  # noqa: F401

    # Validate model exists
    if not ModelRegistry.is_registered(args.model):
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {ModelRegistry.list_all()}")
        return 1

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

    # Build config
    config_overrides = build_config_overrides(args)

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

    # Run training
    from src.models.trainer import Trainer

    logger.info(f"Starting training: model={args.model}, horizon={args.horizon}")
    trainer = Trainer(trainer_config)

    try:
        results = trainer.run(container, skip_save=args.skip_save)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID: {results['run_id']}")
    print(f"Model: {results['model_name']}")
    print(f"Horizon: {results['horizon']}")
    print(f"\nValidation Metrics:")
    eval_metrics = results["evaluation_metrics"]
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {eval_metrics['macro_f1']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"\nPer-Class F1:")
    for cls, f1 in eval_metrics.get("per_class_f1", {}).items():
        print(f"  {cls}: {f1:.4f}")
    print(f"\nTraining Time: {results['total_time_seconds']:.1f}s")
    if not args.skip_save:
        print(f"Output: {results['output_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
