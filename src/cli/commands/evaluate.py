"""
Evaluate Commands - cv, walk-forward, cpcv-pbo.

Commands for model evaluation including cross-validation, walk-forward analysis,
and CPCV/PBO for model selection gating.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from src.cli.utils import (
    DEFAULT_CPCV_OUTPUT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_STACKING_OUTPUT_DIR,
    DEFAULT_WALK_FORWARD_OUTPUT_DIR,
    console,
    generate_run_id,
    parse_horizon_list,
    parse_model_list,
    setup_logging,
    show_error,
    show_warning,
    validate_data_dir,
)

evaluate_app = typer.Typer(
    name="evaluate",
    help="Model evaluation commands",
    no_args_is_help=True,
)


# =============================================================================
# CROSS-VALIDATION COMMAND
# =============================================================================


@evaluate_app.command("cv")
def run_cv(
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    horizons: str = typer.Option(
        "5,10,15,20", "--horizons", "-h", help="Comma-separated horizons or 'all'"
    ),
    # CV configuration
    n_splits: int = typer.Option(5, "--n-splits", help="Number of CV folds"),
    purge_bars: int = typer.Option(60, "--purge-bars", help="Purge bars before test set"),
    embargo_bars: int = typer.Option(1440, "--embargo-bars", help="Embargo bars after test set"),
    # Feature selection
    no_feature_selection: bool = typer.Option(
        False, "--no-feature-selection", help="Disable feature selection"
    ),
    n_features: int = typer.Option(
        50, "--n-features", help="Number of features to select per fold"
    ),
    # Hyperparameter tuning
    tune: bool = typer.Option(False, "--tune", help="Enable Optuna hyperparameter tuning"),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials per model"),
    # Paths
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", "-d", help="Input data directory"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_STACKING_OUTPUT_DIR, "--output-dir", "-o", help="Output directory"
    ),
    output_name: str | None = typer.Option(
        None, "--output-name", help="Custom subdirectory name for this CV run"
    ),
    # Verbosity
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run purged k-fold cross-validation for model evaluation.

    Generates out-of-fold predictions for ensemble stacking (Phase 4).

    Examples:

        ml cv --models xgboost,lightgbm --horizons 5,10,20

        ml cv --models xgboost --horizons 20 --tune --n-trials 100

        ml cv --models all --horizons all --no-feature-selection
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse model and horizon lists
    model_list = parse_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    logger.info(f"Models: {model_list}")
    logger.info(f"Horizons: {horizon_list}")

    # Validate data directory exists
    if not validate_data_dir(data_dir):
        raise typer.Exit(1) from None

    # Import CV modules
    from src.core.container import TimeSeriesDataContainer
    from src.validation.cv.cv_runner import CrossValidationRunner, analyze_cv_stability
    from src.validation.cv.oof_generator import analyze_prediction_correlation
    from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

    # Generate unique run ID for this CV run
    cv_run_id = output_name if output_name else generate_run_id()
    cv_output_dir = output_dir / cv_run_id

    # Create run-specific output directory
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CV output directory: {cv_output_dir}")

    # Configure CV
    cv_config = PurgedKFoldConfig(
        n_splits=n_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    cv = PurgedKFold(cv_config)

    logger.info(f"CV config: {cv}")

    # Process each horizon
    all_results = {}
    all_stacking_datasets = {}

    for horizon in horizon_list:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Processing horizon H{horizon}[/bold]")
        console.print("=" * 60)

        # Load data container
        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=data_dir,
                horizon=horizon,
            )
            logger.info(f"Loaded container: {container}")
        except Exception as e:
            show_error(f"Failed to load data for H{horizon}: {e}")
            continue

        # Create CV runner
        runner = CrossValidationRunner(
            cv=cv,
            models=model_list,
            horizons=[horizon],
            tune_hyperparams=tune,
            select_features=not no_feature_selection,
            n_features_to_select=n_features,
            tuning_trials=n_trials,
        )

        # Run CV
        try:
            cv_results = runner.run(container)
            all_results.update(cv_results)

            # Build stacking dataset
            stacking_datasets = runner.build_stacking_datasets(cv_results, container)
            all_stacking_datasets.update(stacking_datasets)

        except Exception as e:
            show_error(f"CV failed for H{horizon}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    if not all_results:
        show_error("No CV results generated. Check errors above.")
        raise typer.Exit(1) from None

    # Analyze stability
    console.print("\n" + "=" * 60)
    console.print("[bold]STABILITY ANALYSIS[/bold]")
    console.print("=" * 60)

    stability_df = analyze_cv_stability(all_results)
    console.print("\n" + stability_df.to_string(index=False))

    # Analyze prediction correlation (if multiple models)
    if len(model_list) > 1:
        console.print("\n" + "=" * 60)
        console.print("[bold]PREDICTION CORRELATION ANALYSIS[/bold]")
        console.print("=" * 60)

        for horizon_key, stacking_ds in all_stacking_datasets.items():
            console.print(f"\nHorizon H{horizon_key}:")
            corr_df = analyze_prediction_correlation(
                stacking_ds.data,
                stacking_ds.model_names,
            )
            console.print(corr_df.to_string(index=False))

    # Save results
    console.print("\n" + "=" * 60)
    console.print("[bold]SAVING RESULTS[/bold]")
    console.print("=" * 60)

    # Create fresh CV runner for saving (with all horizons)
    save_runner = CrossValidationRunner(
        cv=cv,
        models=model_list,
        horizons=horizon_list,
        tune_hyperparams=tune,
        select_features=not no_feature_selection,
    )
    save_runner.save_results(all_results, all_stacking_datasets, cv_output_dir)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 60)

    for (model_name, horizon), result in all_results.items():
        console.print(
            f"{model_name}/H{horizon}: "
            f"F1={result.mean_f1:.3f} (+/- {result.std_f1:.3f}), "
            f"Stability={result.get_stability_score():.3f}, "
            f"Time={result.total_time:.1f}s"
        )

    console.print(f"\nCV Run ID: {cv_run_id}")
    console.print(f"Results saved to: {cv_output_dir}")
    console.print(f"Stacking datasets saved to: {cv_output_dir / 'stacking'}")
    console.print("\n[bold]To use in Phase 4:[/bold]")
    console.print(f"  ml train model --model stacking --horizon <H> --stacking-data {cv_run_id}")

    raise typer.Exit(0)


# =============================================================================
# WALK-FORWARD COMMAND
# =============================================================================


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _run_walk_forward_for_model(
    container,
    model_name: str,
    config,
    label_end_times=None,
):
    """Run walk-forward evaluation for a single model."""
    from src.models.base import PredictionOutput
    from src.models.registry import ModelRegistry
    from src.validation.cv.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
    from src.validation.cv.walk_forward import (
        WalkForwardEvaluator,
        WalkForwardResult,
        WindowMetrics,
    )

    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Get training data
    X, y, weights = container.get_sklearn_arrays("train", return_df=True)

    n_samples = len(X)
    n_classes = 3

    # Initialize prediction storage
    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full((n_samples, n_classes), np.nan)
    all_confidence = np.full(n_samples, np.nan)

    # Create evaluator
    wf = WalkForwardEvaluator(config)
    window_metrics: list = []

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
        if weights is not None:
            w_train = weights.iloc[train_idx].values

        # Create and train model
        model = ModelRegistry.create(model_name)
        model.fit(
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
        metrics = _compute_metrics(y_test.values, prediction_output.class_predictions)
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


@evaluate_app.command("walk-forward")
def run_walk_forward(
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    horizons: str = typer.Option(
        "5,10,15,20", "--horizons", "-h", help="Comma-separated horizons or 'all'"
    ),
    # Walk-forward configuration
    n_windows: int = typer.Option(5, "--n-windows", help="Number of walk-forward windows"),
    window_type: str = typer.Option(
        "expanding", "--window-type", help="Window type: expanding or rolling"
    ),
    min_train_pct: float = typer.Option(
        0.4, "--min-train-pct", help="Minimum training data percentage"
    ),
    test_pct: float = typer.Option(0.1, "--test-pct", help="Test window percentage"),
    gap_bars: int = typer.Option(0, "--gap-bars", help="Gap bars between train and test"),
    # Paths
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", "-d", help="Path to scaled data directory"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_WALK_FORWARD_OUTPUT_DIR, "--output-dir", "-o", help="Output directory"
    ),
    # Verbosity
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run walk-forward evaluation on ML models.

    More realistic than k-fold for trading applications as it respects temporal ordering.

    Examples:

        ml walk-forward --models xgboost --horizons 20

        ml walk-forward --models xgboost,lightgbm --window-type rolling --n-windows 10

        ml walk-forward --models all --horizons all
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse arguments
    model_list = parse_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    # Validate data directory
    if not validate_data_dir(data_dir):
        raise typer.Exit(1) from None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    from src.core.container import TimeSeriesDataContainer
    from src.validation.cv.walk_forward import WalkForwardConfig, WalkForwardResult

    # Build config
    config = WalkForwardConfig(
        n_windows=n_windows,
        window_type=window_type,
        min_train_pct=min_train_pct,
        test_pct=test_pct,
        gap_bars=gap_bars,
    )

    console.print("=" * 60)
    console.print("[bold]WALK-FORWARD EVALUATION[/bold]")
    console.print("=" * 60)
    console.print(f"Models: {model_list}")
    console.print(f"Horizons: {horizon_list}")
    console.print(f"Config: {config}")
    console.print(f"Data dir: {data_dir}")
    console.print(f"Output dir: {output_dir}")

    all_results: list[WalkForwardResult] = []

    for horizon in horizon_list:
        console.print("-" * 60)
        console.print(f"[bold]HORIZON {horizon}[/bold]")
        console.print("-" * 60)

        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=data_dir,
                horizon=horizon,
            )
            logger.info(f"Loaded container: {container}")
        except Exception as e:
            show_error(f"Failed to load data for H{horizon}: {e}")
            continue

        # Get label end times if available
        label_end_times = container.get_label_end_times("train")
        if label_end_times is not None:
            logger.info("  Using label_end_times for overlap-aware purging")

        for model_name in model_list:
            try:
                result = _run_walk_forward_for_model(
                    container=container,
                    model_name=model_name,
                    config=config,
                    label_end_times=label_end_times,
                )
                all_results.append(result)

                console.print(
                    f"  {model_name} H{horizon}: "
                    f"mean_acc={result.mean_accuracy:.3f} "
                    f"(std={result.std_accuracy:.3f}), "
                    f"mean_f1={result.mean_f1:.3f}, "
                    f"time={result.total_time:.1f}s"
                )

                # Save individual result
                result_path = output_dir / f"wf_{model_name}_h{horizon}.json"
                with open(result_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)

                # Save predictions
                pred_path = output_dir / f"wf_preds_{model_name}_h{horizon}.parquet"
                result.predictions.to_parquet(pred_path, index=False)

            except Exception as e:
                show_error(f"Failed {model_name} H{horizon}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                continue

    # Summary
    console.print("=" * 60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 60)

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
        console.print(summary_df.to_string(index=False))

        # Save summary
        summary_path = output_dir / "walk_forward_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        console.print(f"\nSummary saved to: {summary_path}")
    else:
        show_warning("No results generated")

    console.print(f"Results saved to: {output_dir}")
    raise typer.Exit(0)


# =============================================================================
# CPCV-PBO COMMAND
# =============================================================================


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.nanmean(returns)
    std_ret = np.nanstd(returns, ddof=1)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(252))


def _run_cpcv_for_model(container, model_name: str, cpcv_config, label_end_times=None):
    """Run CPCV evaluation for a single model."""
    from sklearn.metrics import accuracy_score, f1_score

    from src.models.base import PredictionOutput
    from src.models.registry import ModelRegistry
    from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVPathResult, CPCVResult
    from src.validation.cv.fold_scaling import FoldAwareScaler, get_scaling_method_for_model

    logger = logging.getLogger(__name__)

    # Get training data
    X, y, weights = container.get_sklearn_arrays("train", return_df=True)

    cpcv = CombinatorialPurgedCV(cpcv_config)
    path_results: list = []

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
        scaling_result = scaler.fit_transform_fold(X_train_raw.values, X_test_raw.values)

        # Create and train model
        model = ModelRegistry.create(model_name)
        model.fit(
            X_train=scaling_result.X_train_scaled,
            y_train=y_train.values,
            X_val=scaling_result.X_val_scaled,
            y_val=y_test.values,
            sample_weights=w_train,
        )

        # Generate predictions
        prediction_output: PredictionOutput = model.predict(scaling_result.X_val_scaled)

        # Compute metrics
        accuracy = float(accuracy_score(y_test.values, prediction_output.class_predictions))
        f1 = float(
            f1_score(
                y_test.values,
                prediction_output.class_predictions,
                average="weighted",
                zero_division=0,
            )
        )

        # Compute simulated returns for PBO
        correct = (prediction_output.class_predictions == y_test.values).astype(float)
        returns = np.where(correct, 0.01, -0.01)
        sharpe = _compute_sharpe(returns)

        path_result = CPCVPathResult(
            path_id=path_id,
            test_groups=(),
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_groups=(),
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


def _compute_model_pbo(cpcv_results: dict, pbo_config):
    """Compute PBO from multiple model CPCV results."""
    from src.validation.cv.pbo import PBOResult, compute_pbo

    model_names = list(cpcv_results.keys())
    n_models = len(model_names)

    if n_models < 2:
        logging.getLogger(__name__).warning("PBO requires at least 2 models for comparison")
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
    first_result = cpcv_results[model_names[0]]
    n_paths = first_result.n_paths

    perf_matrix = np.zeros((n_models, n_paths))
    for i, model_name in enumerate(model_names):
        result = cpcv_results[model_name]
        for j, path_result in enumerate(result.path_results):
            if j < n_paths:
                perf_matrix[i, j] = path_result.sharpe

    return compute_pbo(perf_matrix, pbo_config)


@evaluate_app.command("cpcv-pbo")
def run_cpcv_pbo(
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Comma-separated horizons"),
    # CPCV configuration
    n_groups: int = typer.Option(6, "--n-groups", help="Number of time groups"),
    n_test_groups: int = typer.Option(2, "--n-test-groups", help="Groups held out as test"),
    max_combinations: int = typer.Option(
        15, "--max-combinations", help="Maximum combinations to evaluate"
    ),
    # PBO thresholds
    pbo_warn: float = typer.Option(0.5, "--pbo-warn", help="PBO warning threshold"),
    pbo_block: float = typer.Option(0.8, "--pbo-block", help="PBO blocking threshold"),
    # Paths
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir", "-d", help="Path to scaled data directory"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_CPCV_OUTPUT_DIR, "--output-dir", "-o", help="Output directory"
    ),
    # Verbosity
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run CPCV and PBO evaluation for model selection gating.

    Computes Combinatorially Purged Cross-Validation (CPCV) and
    Probability of Backtest Overfitting (PBO) metrics.

    Examples:

        ml cpcv-pbo --models xgboost,lightgbm --horizons 20

        ml cpcv-pbo --models all --n-groups 8 --n-test-groups 2

        ml cpcv-pbo --models xgboost --pbo-warn 0.4 --pbo-block 0.7
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # Parse arguments
    model_list = parse_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    # Validate data directory
    if not validate_data_dir(data_dir):
        raise typer.Exit(1) from None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    from src.core.container import TimeSeriesDataContainer
    from src.validation.cv.cpcv import CPCVConfig, CPCVResult
    from src.validation.cv.pbo import PBOConfig, pbo_gate

    # Build configs
    cpcv_config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        max_combinations=max_combinations,
    )

    pbo_config = PBOConfig(
        warn_threshold=pbo_warn,
        block_threshold=pbo_block,
    )

    console.print("=" * 60)
    console.print("[bold]CPCV + PBO EVALUATION[/bold]")
    console.print("=" * 60)
    console.print(f"Models: {model_list}")
    console.print(f"Horizons: {horizon_list}")
    console.print(f"CPCV: {cpcv_config.n_groups} groups, {cpcv_config.n_test_groups} test")
    console.print(f"PBO thresholds: warn={pbo_warn}, block={pbo_block}")

    all_results = []

    for horizon in horizon_list:
        console.print("-" * 60)
        console.print(f"[bold]HORIZON {horizon}[/bold]")
        console.print("-" * 60)

        try:
            container = TimeSeriesDataContainer.from_parquet_dir(
                path=data_dir,
                horizon=horizon,
            )
            logger.info(f"Loaded container: {container}")
        except Exception as e:
            show_error(f"Failed to load data for H{horizon}: {e}")
            continue

        # Get label end times if available
        label_end_times = container.get_label_end_times("train")

        # Run CPCV for each model
        cpcv_results: dict[str, CPCVResult] = {}
        for model_name in model_list:
            try:
                result = _run_cpcv_for_model(
                    container=container,
                    model_name=model_name,
                    cpcv_config=cpcv_config,
                    label_end_times=label_end_times,
                )
                cpcv_results[model_name] = result

                console.print(
                    f"  {model_name}: mean_acc={result.mean_accuracy:.3f}, "
                    f"std={result.std_accuracy:.3f}, mean_sharpe={result.mean_sharpe:.2f}"
                )

            except Exception as e:
                show_error(f"Failed {model_name}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                continue

        # Compute PBO across models
        if len(cpcv_results) >= 2:
            pbo_result = _compute_model_pbo(cpcv_results, pbo_config)

            console.print("-" * 40)
            console.print(f"[bold]PBO Analysis for H{horizon}:[/bold]")
            console.print(f"  PBO: {pbo_result.pbo:.3f}")
            console.print(f"  Risk Level: {pbo_result.get_risk_level()}")
            console.print(f"  Performance Degradation: {pbo_result.performance_degradation:.2f}")
            console.print(f"  Rank Correlation: {pbo_result.rank_correlation:.3f}")

            # Gate check
            should_proceed, reason = pbo_gate(pbo_result, strict=False)
            console.print(
                f"  Gate Decision: {'[green]PASS[/green]' if should_proceed else '[red]FAIL[/red]'}"
            )
            console.print(f"  Reason: {reason}")

            all_results.append(
                {
                    "horizon": horizon,
                    "n_models": len(cpcv_results),
                    "pbo": pbo_result.pbo,
                    "is_overfit": pbo_result.is_overfit,
                    "should_block": pbo_result.should_block,
                    "risk_level": pbo_result.get_risk_level(),
                    "gate_pass": should_proceed,
                }
            )

            # Save PBO result
            pbo_path = output_dir / f"pbo_h{horizon}.json"
            with open(pbo_path, "w") as f:
                json.dump(pbo_result.to_dict(), f, indent=2)

        # Save CPCV results
        for model_name, result in cpcv_results.items():
            result_path = output_dir / f"cpcv_{model_name}_h{horizon}.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

    # Summary
    console.print("=" * 60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 60)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        console.print(summary_df.to_string(index=False))

        # Save summary
        summary_path = output_dir / "cpcv_pbo_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        console.print(f"\nSummary saved to: {summary_path}")

        # Check for any failures
        any_blocked = any(r["should_block"] for r in all_results)
        if any_blocked:
            show_warning("Some horizons have PBO > block threshold!")
            raise typer.Exit(1) from None
    else:
        show_warning("No results generated")

    console.print(f"Results saved to: {output_dir}")
    raise typer.Exit(0)
