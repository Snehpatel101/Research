"""
Train Commands - train, train-ensemble.

Commands for training individual models and ensemble models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer
    from src.models.config.trainer_config import TrainerConfig

from src.cli.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    console,
    setup_logging,
    show_error,
    show_info,
    show_warning,
)

logger = logging.getLogger(__name__)

train_app = typer.Typer(
    name="train",
    help="Model training commands",
    no_args_is_help=True,
)


def _load_phase3_stacking_data(
    cv_run_id: str,
    horizon: int,
    phase3_base_dir: Path,
) -> dict[str, Any] | None:
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
    import pandas as pd  # type: ignore[import-untyped]

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
        with open(metadata_path) as f:
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


def _build_config_overrides(
    batch_size: int | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    seq_len: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    num_channels: str | None = None,
    kernel_size: int | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    feature_set: str | None = None,
    device: str = "auto",
    no_mixed_precision: bool = False,
    base_models: str | None = None,
    meta_learner: str | None = None,
    voting: str | None = None,
) -> dict[str, Any]:
    """Build config overrides from CLI arguments."""
    overrides: dict[str, Any] = {}

    # Map CLI args to config keys
    arg_mapping = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "early_stopping_patience": early_stopping_patience,
        "sequence_length": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "kernel_size": kernel_size,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "feature_set": feature_set,
    }

    for config_key, value in arg_mapping.items():
        if value is not None:
            overrides[config_key] = value

    # Handle special cases
    if num_channels:
        channels = [int(c.strip()) for c in num_channels.split(",")]
        overrides["num_channels"] = channels

    if device != "auto":
        overrides["device"] = device

    if no_mixed_precision:
        overrides["mixed_precision"] = False

    # Ensemble helpers
    if base_models:
        overrides["base_model_names"] = [
            m.strip().lower() for m in base_models.split(",") if m.strip()
        ]
    if meta_learner:
        overrides["meta_learner_name"] = meta_learner.strip().lower()
    if voting:
        overrides["voting"] = voting

    return overrides


def _generate_post_training_report(
    results: dict,
    trainer_config: TrainerConfig,
    container: TimeSeriesDataContainer,
) -> None:
    """Generate financial report after training completion."""
    logger = logging.getLogger(__name__)

    try:
        import numpy as np

        from src.models.evaluation import generate_financial_report

        # Get predictions and true labels from results
        val_predictions = results.get("val_predictions")
        val_true = results.get("val_true")

        if val_predictions is None or val_true is None:
            logger.warning("No predictions available for financial report generation")
            return

        # Get price data for trade simulation
        val_split = container.get_split("val")
        prices = None
        timestamps = None

        if hasattr(val_split, "df"):
            val_df = val_split.df
            # Try to get close prices
            price_col = None
            for col in ["close", "Close", "price", "Price"]:
                if col in val_df.columns:
                    price_col = col
                    break

            if price_col:
                # Align prices with predictions (may have been trimmed)
                if len(val_df) > len(val_predictions):
                    offset = len(val_df) - len(val_predictions)
                    prices = val_df[price_col].iloc[offset:].values
                    if val_df.index is not None:
                        timestamps = val_df.index[offset:]
                else:
                    prices = val_df[price_col].values
                    timestamps = val_df.index

        # Use dummy prices if none available
        if prices is None:
            prices = np.cumsum(np.random.randn(len(val_predictions)) * 0.001) + 100
            logger.warning("No price data found, using simulated prices for report")

        # Generate report
        output_path = PROJECT_ROOT / results["output_path"] / "reports"
        output_path.mkdir(parents=True, exist_ok=True)

        report = generate_financial_report(
            model_name=results["model_name"],
            horizon=results["horizon"],
            predictions=np.asarray(val_predictions),
            y_true=np.asarray(val_true),
            prices=np.asarray(prices),
            timestamps=timestamps,
            output_dir=output_path,
            run_id=results["run_id"],
        )

        console.print("\n[bold green]Financial Report Generated:[/bold green]")
        console.print(f"  HTML: {report.html_path}")
        console.print(f"  JSON: {report.json_path}")
        console.print(f"  Charts: {report.charts_dir}")

    except Exception as e:
        logger.warning(f"Failed to generate financial report: {e}")
        console.print(f"[yellow]Warning: Could not generate financial report: {e}[/yellow]")


def _list_models() -> None:
    """Print all available models."""
    import src.models  # noqa: F401
    from src.models.registry import ModelRegistry

    console.print("\n[bold]Available Models:[/bold]")
    console.print("=" * 60)

    models_by_family = ModelRegistry.list_models()

    for family, models in sorted(models_by_family.items()):
        console.print(f"\n[bold cyan]{family.upper()}:[/bold cyan]")
        for model_name in sorted(models):
            try:
                meta = ModelRegistry.get_metadata(model_name)
                desc = meta.get("description", "")
                console.print(f"  - {model_name}: {desc}")
            except Exception as e:
                logger.warning(f"Failed to get metadata for {model_name}: {e}")
                console.print(f"  - {model_name}")

    console.print(f"\n[bold]Total: {ModelRegistry.count()} models[/bold]")


def _show_model_info(model_name: str) -> None:
    """Print detailed info about a model."""
    import src.models  # noqa: F401
    from src.models.registry import ModelRegistry

    try:
        info = ModelRegistry.get_model_info(model_name)
    except ValueError as e:
        show_error(str(e))
        raise typer.Exit(1) from None

    console.print(f"\n[bold]Model: {info['name']}[/bold]")
    console.print("=" * 60)
    console.print(f"Family: {info['family']}")
    console.print(f"Description: {info.get('description', 'N/A')}")
    console.print(f"Requires Scaling: {info['requires_scaling']}")
    console.print(f"Requires Sequences: {info['requires_sequences']}")
    console.print(f"Requires 4D: {info.get('requires_4d', False)}")
    console.print("\n[bold]Default Configuration:[/bold]")
    for key, value in sorted(info["default_config"].items()):
        console.print(f"  {key}: {value}")


@train_app.command("model")
def train_model(
    model: str = typer.Option(None, "--model", "-m", help="Model name (e.g., xgboost, lstm, tcn)"),
    horizon: int = typer.Option(20, "--horizon", "-h", help="Label horizon"),
    feature_set: str | None = typer.Option(None, "--feature-set", help="Feature set name"),
    # Ensemble arguments
    base_models: str | None = typer.Option(
        None, "--base-models", help="Comma-separated base models for ensemble"
    ),
    meta_learner: str | None = typer.Option(
        None, "--meta-learner", help="Meta-learner for stacking/blending"
    ),
    voting: str | None = typer.Option(None, "--voting", help="Voting strategy: soft/hard"),
    # Phase 3->4 integration
    stacking_data: str | None = typer.Option(
        None, "--stacking-data", help="CV run ID for Phase 3 stacking data"
    ),
    phase3_output: Path = typer.Option(
        Path("data/stacking"), "--phase3-output", help="Phase 3 outputs directory"
    ),
    # Data arguments
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", "-d", help="Path to scaled splits directory"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", "-o", help="Output directory for artifacts"
    ),
    # Training arguments
    batch_size: int | None = typer.Option(None, "--batch-size", help="Training batch size"),
    max_epochs: int | None = typer.Option(None, "--max-epochs", help="Maximum training epochs"),
    learning_rate: float | None = typer.Option(
        None, "--learning-rate", "--lr", help="Learning rate"
    ),
    early_stopping_patience: int | None = typer.Option(
        None, "--early-stopping-patience", help="Early stopping patience"
    ),
    # Sequence model arguments
    seq_len: int | None = typer.Option(
        None, "--seq-len", help="Sequence length for sequential models"
    ),
    hidden_size: int | None = typer.Option(
        None, "--hidden-size", help="Hidden size for RNN/TCN models"
    ),
    num_layers: int | None = typer.Option(
        None, "--num-layers", help="Number of layers for RNN models"
    ),
    dropout: float | None = typer.Option(None, "--dropout", help="Dropout rate"),
    # TCN-specific
    num_channels: str | None = typer.Option(
        None, "--num-channels", help="Comma-separated channel sizes for TCN"
    ),
    kernel_size: int | None = typer.Option(None, "--kernel-size", help="Kernel size for TCN"),
    # Boosting-specific
    n_estimators: int | None = typer.Option(None, "--n-estimators", help="Number of estimators"),
    max_depth: int | None = typer.Option(None, "--max-depth", help="Max tree depth"),
    # Device
    device: str = typer.Option("auto", "--device", help="Device: cuda, cpu, auto"),
    no_mixed_precision: bool = typer.Option(
        False, "--no-mixed-precision", help="Disable mixed precision"
    ),
    # Test evaluation
    evaluate_test: bool = typer.Option(
        True, "--evaluate-test/--no-evaluate-test", help="Evaluate on test set"
    ),
    # Utility
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    list_models: bool = typer.Option(False, "--list-models", help="List available models and exit"),
    model_info: str | None = typer.Option(
        None, "--model-info", help="Show info about a specific model"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    skip_save: bool = typer.Option(False, "--skip-save", help="Skip saving artifacts"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip financial report generation"),
):
    """
    Train any registered model from the Model Factory.

    Supports all registered model types (boosting, neural, etc.) with config
    loading from YAML files and CLI overrides.

    Examples:

        ml train model --model xgboost --horizon 20

        ml train model --model lstm --horizon 20 --hidden-size 256

        ml train model --model voting --base-models xgboost,lightgbm,catboost --horizon 20
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Handle utility commands
    if list_models:
        _list_models()
        raise typer.Exit(0)

    if model_info:
        _show_model_info(model_info)
        raise typer.Exit(0)

    # Validate required arguments
    if not model:
        show_error("--model is required")
        show_info("Use --list-models to see available models")
        raise typer.Exit(1) from None

    # Import models to ensure registration
    logger.info("Loading model registry...")
    import src.models  # noqa: F401
    from src.models.registry import ModelRegistry

    # Validate model exists
    if not ModelRegistry.is_registered(model):
        show_error(f"Unknown model '{model}'")
        show_info(f"Available: {ModelRegistry.list_all()}")
        raise typer.Exit(1) from None

    # Build config overrides
    config_overrides = _build_config_overrides(
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_channels=num_channels,
        kernel_size=kernel_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        feature_set=feature_set,
        device=device,
        no_mixed_precision=no_mixed_precision,
        base_models=base_models,
        meta_learner=meta_learner,
        voting=voting,
    )

    # Validate base models if provided
    base_model_names = config_overrides.get("base_model_names")
    if base_model_names:
        invalid = [m for m in base_model_names if not ModelRegistry.is_registered(m)]
        if invalid:
            show_error(f"Unknown base models: {invalid}")
            show_info(f"Available: {ModelRegistry.list_all()}")
            raise typer.Exit(1) from None

    # Phase 3->4 workflow: Load stacking data if provided
    stacking_data_dict = None
    if stacking_data:
        if model not in ["stacking", "blending"]:
            show_warning(
                f"--stacking-data is only used for stacking/blending models, "
                f"but model is '{model}'. Ignoring stacking data."
            )
        else:
            try:
                stacking_data_dict = _load_phase3_stacking_data(
                    cv_run_id=stacking_data,
                    horizon=horizon,
                    phase3_base_dir=phase3_output,
                )
                logger.info(
                    f"Using Phase 3 stacking data from CV run: {stacking_data}\n"
                    f"This data contains leakage-safe OOF predictions from cross-validation."
                )
            except (FileNotFoundError, ValueError) as e:
                show_error(f"Failed to load Phase 3 stacking data: {e}")
                raise typer.Exit(1) from None

    # Load data (Phase 1 datasets OR Phase 3 stacking data)
    container = None
    if not stacking_data_dict:
        # Standard workflow: Load Phase 1 datasets
        logger.info(f"Loading data from {data_dir}...")
        from src.core.container import TimeSeriesDataContainer

        data_path = PROJECT_ROOT / data_dir
        if not data_path.exists():
            show_error(f"Data directory not found: {data_path}")
            raise typer.Exit(1) from None

        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=data_path,
                horizon=horizon,
            )
        except Exception as e:
            show_error(f"Error loading data: {e}")
            raise typer.Exit(1) from None

        logger.info(f"Loaded: {container}")
    else:
        logger.info(
            f"Using Phase 3 stacking data (skipping Phase 1 container load)\n"
            f"Samples: {len(stacking_data_dict['data'])}, "
            f"Models: {stacking_data_dict['metadata'].get('model_names', [])}"
        )

    # Get model info for sequence length
    model_info_dict = ModelRegistry.get_model_info(model)
    if model_info_dict["requires_sequences"]:
        if "sequence_length" not in config_overrides:
            config_overrides["sequence_length"] = model_info_dict["default_config"].get(
                "sequence_length", 60
            )
        logger.info(f"Using sequence length: {config_overrides.get('sequence_length')}")

    # Create trainer config
    from src.models.config import create_trainer_config

    config_file = PROJECT_ROOT / config if config else None

    try:
        trainer_config = create_trainer_config(
            model_name=model,
            horizon=horizon,
            cli_args=config_overrides,
            config_file=config_file,
        )
    except Exception as e:
        show_error(f"Configuration error: {e}")
        raise typer.Exit(1) from None

    # Override settings from CLI
    if output_dir:
        trainer_config.output_dir = PROJECT_ROOT / output_dir
    trainer_config.device = device
    trainer_config.mixed_precision = not no_mixed_precision
    trainer_config.evaluate_test_set = evaluate_test

    # Run training
    from src.models.trainer import Trainer

    logger.info(f"Starting training: model={model}, horizon={horizon}")
    logger.info(
        "Using MODEL_DATA_REQUIREMENTS for validation. "
        "Note: MLFactory uses MODEL_CONTRACTS (src/core/contracts/) which may have different limits."
    )

    if stacking_data_dict and stacking_data:
        # Phase 3->4 workflow: Train directly on stacking data
        results = _train_on_stacking_data(
            stacking_data_dict, trainer_config, skip_save, stacking_data
        )
    else:
        # Standard workflow: Use container
        if container is None:
            show_error("No data available for training")
            raise typer.Exit(1) from None
        trainer = Trainer(trainer_config)
        try:
            results = trainer.run(container, skip_save=skip_save)
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise typer.Exit(1) from None

    # Generate financial report if not skipped
    if not no_report and not skip_save:
        _generate_post_training_report(results, trainer_config, container)

    # Print results
    _print_training_results(results, skip_save)
    raise typer.Exit(0)


def _train_on_stacking_data(
    stacking_data_dict: dict,
    trainer_config,
    skip_save: bool,
    stacking_source: str,
) -> dict:
    """Train meta-learner on Phase 3 OOF predictions."""
    from src.models.trainer import Trainer, compute_classification_metrics

    logger = logging.getLogger(__name__)
    logger.info("Training meta-learner on Phase 3 OOF predictions...")

    # Extract features and labels
    stacking_df = stacking_data_dict["data"]
    feature_cols = [c for c in stacking_df.columns if c not in ("y_true", "datetime")]
    X_stacking = stacking_df[feature_cols].values
    y_stacking = stacking_df["y_true"].values

    # Time-based split into train/val (80/20) to prevent data leakage
    if len(X_stacking) < 2:
        raise ValueError(
            f"Cannot split stacking data with {len(X_stacking)} samples (minimum 2 required)"
        )
    split_idx = int(len(X_stacking) * 0.8)
    split_idx = max(1, split_idx)  # Ensure at least 1 training sample
    X_train, X_val = X_stacking[:split_idx], X_stacking[split_idx:]
    y_train, y_val = y_stacking[:split_idx], y_stacking[split_idx:]

    logger.info(
        f"Stacking data split: train={len(X_train)}, val={len(X_val)}\n"
        f"Training meta-learner (this should be fast - no OOF generation needed)"
    )

    # Create trainer and train meta-learner directly
    trainer = Trainer(trainer_config)

    try:
        # Train directly
        training_metrics = trainer.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weights=None,
            config=trainer_config.model_config,
        )

        # Evaluate
        val_predictions = trainer.model.predict(X_val)

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
            "total_time_seconds": 0.0,
            "stacking_source": stacking_source,
        }

        # Save artifacts
        if not skip_save:
            trainer._setup_output_dir()
            trainer._save_config()
            trainer._save_artifacts(training_metrics, eval_metrics, val_predictions)
            trainer._save_model()

        return results

    except Exception as e:
        logger.exception(f"Meta-learner training failed: {e}")
        raise typer.Exit(1) from None


def _print_training_results(results: dict, skip_save: bool) -> None:
    """Print training results to console."""
    console.print("\n" + "=" * 70)
    console.print("[bold green]TRAINING COMPLETE[/bold green]")
    console.print("=" * 70)
    console.print(f"Run ID: {results['run_id']}")
    console.print(f"Model: {results['model_name']}")
    console.print(f"Horizon: {results['horizon']}")

    console.print("\n[bold]Validation Metrics:[/bold]")
    eval_metrics = results["evaluation_metrics"]
    console.print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    console.print(f"  Macro F1: {eval_metrics['macro_f1']:.4f}")
    console.print(f"  Precision: {eval_metrics['precision']:.4f}")
    console.print(f"  Recall: {eval_metrics['recall']:.4f}")

    # Trading metrics (validation)
    if "trading" in eval_metrics:
        trading = eval_metrics["trading"]
        console.print("\n  [bold]Trading Metrics (Validation):[/bold]")
        console.print(f"    Position Win Rate: {trading.get('position_win_rate', 0):.4f}")
        console.print(f"    Long Accuracy: {trading.get('long_accuracy', 0):.4f}")
        console.print(f"    Short Accuracy: {trading.get('short_accuracy', 0):.4f}")
        console.print(f"    Position Sharpe: {trading.get('position_sharpe', 0):.4f}")
        console.print(f"    Total Positions: {trading.get('total_positions', 0)}")

    console.print("\n[bold]Per-Class F1:[/bold]")
    for cls, f1 in eval_metrics.get("per_class_f1", {}).items():
        console.print(f"  {cls}: {f1:.4f}")

    # Test set metrics (if evaluated)
    if results.get("test_metrics") is not None:
        console.print("\n" + "=" * 70)
        console.print(
            "[bold yellow]TEST SET RESULTS (ONE-SHOT GENERALIZATION ESTIMATE)[/bold yellow]"
        )
        console.print("=" * 70)
        console.print(
            "[yellow]WARNING: Do NOT iterate on these results. If you do, you're overfitting to test.[/yellow]"
        )
        console.print("=" * 70)

        test_metrics = results["test_metrics"]
        console.print("\n[bold]Test Metrics:[/bold]")
        console.print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        console.print(f"  Precision: {test_metrics['precision']:.4f}")
        console.print(f"  Recall: {test_metrics['recall']:.4f}")

        if "trading" in test_metrics:
            trading = test_metrics["trading"]
            console.print("\n  [bold]Trading Metrics (Test):[/bold]")
            console.print(f"    Position Win Rate: {trading.get('position_win_rate', 0):.4f}")
            console.print(f"    Long Accuracy: {trading.get('long_accuracy', 0):.4f}")
            console.print(f"    Short Accuracy: {trading.get('short_accuracy', 0):.4f}")
            console.print(f"    Position Sharpe: {trading.get('position_sharpe', 0):.4f}")
            console.print(f"    Total Positions: {trading.get('total_positions', 0)}")

        console.print("\n[bold]Per-Class F1 (Test):[/bold]")
        for cls, f1 in test_metrics.get("per_class_f1", {}).items():
            console.print(f"  {cls}: {f1:.4f}")

        console.print("\n" + "=" * 70)
        console.print(
            "[yellow]If test results are disappointing: DO NOT tune and re-evaluate.[/yellow]"
        )
        console.print(
            "[yellow]Move on to the next experiment. Test set discipline is critical.[/yellow]"
        )
        console.print("=" * 70)

    console.print(f"\nTraining Time: {results['total_time_seconds']:.1f}s")
    if not skip_save:
        console.print(f"Output: {results['output_path']}")


@train_app.command("ensemble")
def train_ensemble(
    base_models: str = typer.Option(
        ..., "--base-models", "-b", help="Comma-separated base model names"
    ),
    meta_learner: str = typer.Option(
        "logistic", "--meta-learner", "-m", help="Meta-learner model name"
    ),
    horizon: int = typer.Option(20, "--horizon", "-h", help="Label horizon"),
    feature_set: str | None = typer.Option(None, "--feature-set", help="Feature set name"),
    # Stacking configuration
    n_folds: int = typer.Option(5, "--n-folds", help="Number of CV folds for OOF generation"),
    purge_bars: int = typer.Option(60, "--purge-bars", help="Purge bars for CV splits"),
    embargo_bars: int = typer.Option(1440, "--embargo-bars", help="Embargo bars for CV splits"),
    # Data arguments
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", "-d", help="Path to scaled splits directory"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", "-o", help="Output directory"
    ),
    # Sequence model arguments
    seq_len: int = typer.Option(60, "--seq-len", help="Sequence length for sequence models"),
    # Device
    device: str = typer.Option("auto", "--device", help="Device: cuda, cpu, auto"),
    # Test evaluation
    evaluate_test: bool = typer.Option(
        True, "--evaluate-test/--no-evaluate-test", help="Evaluate on test set"
    ),
    # Utility
    list_models: bool = typer.Option(
        False, "--list-models", help="List available base models and meta-learners"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    skip_save: bool = typer.Option(False, "--skip-save", help="Skip saving artifacts"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip financial report generation"),
):
    """
    Train heterogeneous stacking ensembles.

    Automatically handles dual data loading (2D + 3D) and routes appropriate
    data formats to each base model.

    Examples:

        ml train ensemble --base-models catboost,tcn,patchtst --meta-learner logistic --horizon 20

        ml train ensemble --base-models xgboost,lstm --meta-learner logistic --horizon 20 --n-folds 3
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Handle utility commands
    if list_models:
        _list_ensemble_models()
        raise typer.Exit(0)

    # Import models to ensure registration
    logger.info("Loading model registry...")
    import src.models  # noqa: F401
    from src.models.registry import ModelRegistry

    # Parse and validate base models
    base_model_names = [m.strip().lower() for m in base_models.split(",") if m.strip()]
    if len(base_model_names) < 2:
        show_error("At least 2 base models required for stacking")
        raise typer.Exit(1) from None

    invalid = [m for m in base_model_names if not ModelRegistry.is_registered(m)]
    if invalid:
        show_error(f"Unknown base models: {invalid}")
        show_info(f"Available: {ModelRegistry.list_all()}")
        raise typer.Exit(1) from None

    # Validate meta-learner
    if not ModelRegistry.is_registered(meta_learner):
        show_error(f"Unknown meta-learner '{meta_learner}'")
        show_info("Available: logistic, ridge_meta, mlp_meta, calibrated_meta, xgboost_meta")
        raise typer.Exit(1) from None

    # Classify base models
    from src.models.ensemble.validator import classify_base_models

    tabular_models, sequence_models = classify_base_models(base_model_names)
    is_heterogeneous = len(tabular_models) > 0 and len(sequence_models) > 0

    console.print("\n" + "=" * 70)
    console.print("[bold]HETEROGENEOUS STACKING ENSEMBLE TRAINING[/bold]")
    console.print("=" * 70)
    console.print(f"Base Models: {base_model_names}")
    console.print(f"  Tabular ({len(tabular_models)}): {tabular_models}")
    console.print(f"  Sequence ({len(sequence_models)}): {sequence_models}")
    console.print(f"Meta-Learner: {meta_learner}")
    console.print(f"Heterogeneous: {is_heterogeneous}")
    console.print(f"CV Folds: {n_folds}")
    console.print(f"Horizon: {horizon}")
    console.print("=" * 70 + "\n")

    # Load data
    logger.info(f"Loading data from {data_dir}...")
    from src.core.container import TimeSeriesDataContainer

    data_path = PROJECT_ROOT / data_dir
    if not data_path.exists():
        show_error(f"Data directory not found: {data_path}")
        raise typer.Exit(1) from None

    try:
        container = TimeSeriesDataContainer.from_parquet_dir(
            path=data_path,
            horizon=horizon,
        )
    except Exception as e:
        show_error(f"Error loading data: {e}")
        raise typer.Exit(1) from None

    logger.info(f"Loaded: {container}")

    # Build stacking configuration
    stacking_config = {
        "base_model_names": base_model_names,
        "meta_learner_name": meta_learner.strip().lower(),
        "n_folds": n_folds,
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
        "use_probabilities": True,
        "passthrough": False,
        "use_default_configs_for_oof": True,
    }

    if feature_set:
        stacking_config["feature_set"] = feature_set.strip().lower()

    # Create trainer config
    from src.models.config import create_trainer_config

    try:
        trainer_config = create_trainer_config(
            model_name="stacking",
            horizon=horizon,
            cli_args=stacking_config,
            config_file=None,
        )
    except Exception as e:
        show_error(f"Configuration error: {e}")
        raise typer.Exit(1) from None

    # Override settings from CLI
    if output_dir:
        trainer_config.output_dir = PROJECT_ROOT / output_dir
    trainer_config.device = device
    trainer_config.evaluate_test_set = evaluate_test

    # Add sequence length to model config for sequence models
    if sequence_models:
        trainer_config.model_config["sequence_length"] = seq_len
        logger.info(f"Using sequence length: {seq_len} for sequence models")

    # Run training
    from src.models.trainer import Trainer

    logger.info("Starting heterogeneous stacking training...")
    trainer = Trainer(trainer_config)

    try:
        results = trainer.run(container, skip_save=skip_save)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise typer.Exit(1) from None

    # Generate financial report if not skipped
    if not no_report and not skip_save:
        _generate_post_training_report(results, trainer_config, container)

    # Print results
    console.print("\n" + "=" * 70)
    console.print("[bold green]TRAINING COMPLETE[/bold green]")
    console.print("=" * 70)
    console.print(f"Run ID: {results['run_id']}")
    console.print(f"Model: stacking ({len(base_model_names)} bases + {meta_learner})")
    console.print(f"Horizon: {results['horizon']}")

    console.print("\n[bold]Validation Metrics:[/bold]")
    eval_metrics = results["evaluation_metrics"]
    console.print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    console.print(f"  Macro F1: {eval_metrics['macro_f1']:.4f}")
    console.print(f"  Precision: {eval_metrics['precision']:.4f}")
    console.print(f"  Recall: {eval_metrics['recall']:.4f}")

    if "trading" in eval_metrics:
        trading = eval_metrics["trading"]
        console.print("\n  [bold]Trading Metrics (Validation):[/bold]")
        console.print(f"    Position Win Rate: {trading.get('position_win_rate', 0):.4f}")
        console.print(f"    Long Accuracy: {trading.get('long_accuracy', 0):.4f}")
        console.print(f"    Short Accuracy: {trading.get('short_accuracy', 0):.4f}")

    console.print("\n[bold]Per-Class F1:[/bold]")
    for cls, f1 in eval_metrics.get("per_class_f1", {}).items():
        console.print(f"  {cls}: {f1:.4f}")

    # Test set metrics
    if results.get("test_metrics") is not None:
        console.print("\n" + "=" * 70)
        console.print(
            "[bold yellow]TEST SET RESULTS (ONE-SHOT GENERALIZATION ESTIMATE)[/bold yellow]"
        )
        console.print("=" * 70)

        test_metrics = results["test_metrics"]
        console.print("\n[bold]Test Metrics:[/bold]")
        console.print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        console.print(f"  Precision: {test_metrics['precision']:.4f}")
        console.print(f"  Recall: {test_metrics['recall']:.4f}")

    console.print(f"\nTraining Time: {results['total_time_seconds']:.1f}s")
    if not skip_save:
        console.print(f"Output: {results['output_path']}")

    raise typer.Exit(0)


def _list_ensemble_models() -> None:
    """Print available base models and meta-learners."""
    import src.models  # noqa: F401
    from src.models.registry import ModelRegistry

    console.print("\n[bold]Heterogeneous Ensemble Components[/bold]")
    console.print("=" * 70)

    # Get models by family
    models_by_family = ModelRegistry.list_models()

    console.print("\n[bold cyan]TABULAR BASE MODELS (2D input):[/bold cyan]")
    console.print("-" * 40)
    for family in ["boosting", "classical"]:
        if family in models_by_family:
            for model_name in sorted(models_by_family[family]):
                if not model_name.endswith("_meta"):
                    info = ModelRegistry.get_model_info(model_name)
                    desc = info.get("description", "")[:50]
                    console.print(f"  {model_name:20} - {desc}")

    console.print("\n[bold cyan]SEQUENCE BASE MODELS (3D input):[/bold cyan]")
    console.print("-" * 40)
    for family in ["neural", "cnn", "mlp", "advanced"]:
        if family in models_by_family:
            for model_name in sorted(models_by_family[family]):
                info = ModelRegistry.get_model_info(model_name)
                desc = info.get("description", "")[:50]
                console.print(f"  {model_name:20} - {desc}")

    console.print("\n[bold cyan]META-LEARNERS (for stacking):[/bold cyan]")
    console.print("-" * 40)
    meta_learners = ["logistic", "ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]
    for name in meta_learners:
        if ModelRegistry.is_registered(name):
            info = ModelRegistry.get_model_info(name)
            desc = info.get("description", "")[:50]
            console.print(f"  {name:20} - {desc}")

    console.print("\n[bold]RECOMMENDED CONFIGURATIONS:[/bold]")
    console.print("-" * 40)
    console.print("  3-Base Standard:    catboost,tcn,patchtst + logistic")
    console.print("  4-Base Maximum:     lightgbm,tcn,tft,random_forest + ridge_meta")
    console.print("  2-Base Minimal:     xgboost,lstm + logistic")

    console.print(f"\nTotal registered models: {ModelRegistry.count()}")
