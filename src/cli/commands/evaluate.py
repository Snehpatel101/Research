"""
Evaluate Commands - cv, walk-forward, cpcv-pbo.

Standalone model evaluation on MLFactory-prepared data. Each command loads raw
OHLCV bars, runs the same FeatureEngineer + triple-barrier labeling step as
``ml run`` (labels carry their resolution bar, so purging follows the actual
label spans), and evaluates tabular models on the chronological TRAIN split with
per-fold scaling. Validation and test rows are never touched.

- ``cv``:          purged k-fold cross-validation (fold stability, stacking datasets)
- ``walk-forward``: expanding/rolling walk-forward windows
- ``cpcv-pbo``:    combinatorial purged CV backtest paths + PBO overfitting gate

Sequence models are evaluated through ``ml run --training-mode walk_forward``.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from src.cli.utils import (
    DEFAULT_OUTPUT_DIR,
    build_experiment_config,
    console,
    parse_horizon_list,
    parse_tabular_model_list,
    setup_logging,
    show_error,
    show_warning,
)
from src.core.reproducibility import apply_model_seed


def _load_evaluation_data(
    *,
    data_path: Path,
    output_dir: Path,
    symbol: str,
    horizons: list[int],
    models: list[str],
    bar_timeframe: str | None,
    mtf: bool,
    n_splits: int | None,
    purge_bars: int | None,
    embargo_bars: int | None,
    command: str,
    random_seed: int,
):
    """
    Prepare data through MLFactory and open the command's result directory.

    Returns:
        (factory, {horizon: TimeSeriesDataContainer}, results_dir)
    """
    from src.factory import MLFactory

    config = build_experiment_config(
        data_path=data_path,
        output_dir=output_dir,
        symbol=symbol,
        horizons=horizons,
        models=models,
        bar_timeframe=bar_timeframe,
        mtf=mtf,
        n_splits=n_splits,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        name=f"{symbol}_{command}",
        random_seed=random_seed,
    )
    factory = MLFactory(config, enable_checkpoints=False)
    try:
        containers = factory.build_evaluation_containers()
    except Exception:
        # A failed data step leaves no half-made run directory behind
        shutil.rmtree(factory.output_dir, ignore_errors=True)
        raise
    results_dir = factory.output_dir / command
    results_dir.mkdir(parents=True, exist_ok=True)
    return factory, containers, results_dir


# =============================================================================
# CROSS-VALIDATION COMMAND
# =============================================================================


def run_cv(
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Raw OHLCV file (parquet or csv)"
    ),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option(
        "5,10,15,20", "--horizons", "-h", help="Comma-separated horizons or 'all'"
    ),
    bar_timeframe: str | None = typer.Option(
        None, "--bar-timeframe", help="Resample input bars first, e.g. 5min"
    ),
    mtf: bool = typer.Option(True, "--mtf/--no-mtf", help="Multi-timeframe features"),
    # CV configuration
    n_splits: int = typer.Option(5, "--n-splits", help="Number of CV folds"),
    purge_bars: int | None = typer.Option(
        None, "--purge-bars", help="Purge bars before each test fold (default: label span)"
    ),
    embargo_bars: int | None = typer.Option(
        None, "--embargo-bars", help="Embargo bars after each test fold (default: one day)"
    ),
    # Feature selection
    no_feature_selection: bool = typer.Option(
        False, "--no-feature-selection", help="Disable per-fold feature selection"
    ),
    n_features: int = typer.Option(
        50, "--n-features", help="Number of features to select per fold"
    ),
    # Hyperparameter tuning
    tune: bool = typer.Option(False, "--tune", help="Enable Optuna hyperparameter tuning"),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials per model"),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", "-o", help="Output root; writes <dir>/<run_id>/cv/"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed (models, Optuna, feature selection)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run purged k-fold cross-validation for tabular models.

    Reports per-fold F1 and stability, prediction correlation between models, and
    saves the out-of-fold stacking datasets.

    Examples:

        ml cv -d data/mes_5min.parquet -m xgboost,lightgbm --horizons 5,20

        ml cv -d data/mes_5min.parquet -m xgboost --horizons 20 --tune --n-trials 100
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    model_list = parse_tabular_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    logger.info(f"Models: {model_list}")
    logger.info(f"Horizons: {horizon_list}")

    from src.validation.cv.cv_runner import CrossValidationRunner, analyze_cv_stability
    from src.validation.cv.oof_generator import analyze_prediction_correlation
    from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

    try:
        factory, containers, cv_output_dir = _load_evaluation_data(
            data_path=data_path,
            output_dir=output_dir,
            symbol=symbol,
            horizons=horizon_list,
            models=model_list,
            bar_timeframe=bar_timeframe,
            mtf=mtf,
            n_splits=n_splits,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            command="cv",
            random_seed=seed,
        )
    except Exception as e:
        show_error(f"Data preparation failed: {e}")
        raise typer.Exit(1) from None

    purge, embargo = factory.cv_gaps
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=n_splits, purge_bars=purge, embargo_bars=embargo))
    logger.info(f"CV config: {cv}")

    all_results = {}
    all_stacking_datasets = {}

    for horizon in horizon_list:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Processing horizon H{horizon}[/bold]")
        console.print("=" * 60)

        container = containers[horizon]
        runner = CrossValidationRunner(
            cv=cv,
            models=model_list,
            horizons=[horizon],
            tune_hyperparams=tune,
            select_features=not no_feature_selection,
            n_features_to_select=n_features,
            tuning_trials=n_trials,
            seed=seed,
        )

        try:
            cv_results = runner.run(container)
            all_results.update(cv_results)
            all_stacking_datasets.update(runner.build_stacking_datasets(cv_results, container))
        except Exception as e:
            show_error(f"CV failed for H{horizon}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    if not all_results:
        show_error("No CV results generated. Check errors above.")
        raise typer.Exit(1) from None

    console.print("\n" + "=" * 60)
    console.print("[bold]STABILITY ANALYSIS[/bold]")
    console.print("=" * 60)

    stability_df = analyze_cv_stability(all_results)
    console.print("\n" + stability_df.to_string(index=False))

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

    console.print("\n" + "=" * 60)
    console.print("[bold]SAVING RESULTS[/bold]")
    console.print("=" * 60)

    save_runner = CrossValidationRunner(
        cv=cv,
        models=model_list,
        horizons=horizon_list,
        tune_hyperparams=tune,
        select_features=not no_feature_selection,
    )
    save_runner.save_results(all_results, all_stacking_datasets, cv_output_dir)

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

    console.print(f"\nResults saved to: {cv_output_dir}", soft_wrap=True)
    console.print(f"Stacking datasets saved to: {cv_output_dir / 'stacking'}", soft_wrap=True)


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
    seed: int,
    label_spans=None,
):
    """Run walk-forward evaluation for a single model."""
    from src.models.base import PredictionResult
    from src.models.registry import ModelRegistry
    from src.validation.cv.early_stopping_split import carve_early_stopping_split
    from src.validation.cv.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
    from src.validation.cv.oof_core import build_oof_frame
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
    n_classes = container.n_classes

    # Initialize prediction storage
    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full((n_samples, n_classes), np.nan)
    all_confidence = np.full(n_samples, np.nan)
    all_window_ids = np.full(n_samples, -1, dtype=int)

    # Create evaluator
    wf = WalkForwardEvaluator(config)
    window_metrics: list = []

    # Get scaling method for model
    scaling_method = get_scaling_method_for_model(model_name)

    logger.info(f"Running walk-forward for {model_name} ({config.n_windows} windows)")

    for window_idx, (train_idx, test_idx) in enumerate(wf.split(X, y, label_spans=label_spans)):
        window_start = time.time()

        logger.debug(f"  Window {window_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")

        # Early stopping selects on a purged tail of the window's train rows,
        # never on the test window being evaluated
        es_split = carve_early_stopping_split(train_idx, config.gap_bars)
        train_idx, es_idx = es_split.fit_idx, es_split.es_idx

        # Extract window data
        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_es = y.iloc[es_idx]
        y_test = y.iloc[test_idx]

        # Fold-aware scaling (fit on the fit rows only)
        scaler = FoldAwareScaler(method=scaling_method)
        scaling_result = scaler.fit_transform_fold(
            X_train_raw.values, np.vstack([X.iloc[es_idx].values, X_test_raw.values])
        )
        X_train_scaled = scaling_result.X_train_scaled
        X_es_scaled = scaling_result.X_val_scaled[: len(es_idx)]
        X_test_scaled = scaling_result.X_val_scaled[len(es_idx) :]

        # Handle sample weights
        w_train = None
        if weights is not None:
            w_train = weights.iloc[train_idx].values

        # Create and train model
        model = ModelRegistry.create(
            model_name, config=apply_model_seed({"n_classes": n_classes}, seed)
        )
        model.fit(
            X_train=X_train_scaled,
            y_train=y_train.values,
            X_val=X_es_scaled,
            y_val=y_es.values,
            sample_weights=w_train,
        )

        # Generate predictions
        prediction_output: PredictionResult = model.predict(X_test_scaled)

        # Store predictions
        all_preds[test_idx] = prediction_output.class_predictions
        all_probs[test_idx] = prediction_output.class_probabilities
        all_confidence[test_idx] = prediction_output.confidence
        all_window_ids[test_idx] = window_idx

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
    predictions_df = build_oof_frame(
        model_name, X.index, y.values, all_probs, all_preds, all_confidence, all_window_ids
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


def run_walk_forward(
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Raw OHLCV file (parquet or csv)"
    ),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option(
        "5,10,15,20", "--horizons", "-h", help="Comma-separated horizons or 'all'"
    ),
    bar_timeframe: str | None = typer.Option(
        None, "--bar-timeframe", help="Resample input bars first, e.g. 5min"
    ),
    mtf: bool = typer.Option(True, "--mtf/--no-mtf", help="Multi-timeframe features"),
    # Walk-forward configuration
    n_windows: int = typer.Option(5, "--n-windows", help="Number of walk-forward windows"),
    window_type: str = typer.Option(
        "expanding", "--window-type", help="Window type: expanding or rolling"
    ),
    min_train_pct: float = typer.Option(
        0.4, "--min-train-pct", help="Minimum training data percentage"
    ),
    test_pct: float = typer.Option(0.1, "--test-pct", help="Test window percentage"),
    gap_bars: int | None = typer.Option(
        None, "--gap-bars", help="Gap bars between train and test (default: label span)"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output root; writes <dir>/<run_id>/walk-forward/",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed (models, Optuna, feature selection)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run walk-forward evaluation on tabular models.

    More realistic than k-fold for trading applications as it respects temporal
    ordering. Sequence models: `ml run --training-mode walk_forward`.

    Examples:

        ml walk-forward -d data/mes_5min.parquet -m xgboost --horizons 20

        ml walk-forward -d data/mes_5min.parquet -m xgboost,lightgbm --window-type rolling
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    model_list = parse_tabular_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    from src.validation.cv.walk_forward import WalkForwardConfig, WalkForwardResult

    try:
        factory, containers, results_dir = _load_evaluation_data(
            data_path=data_path,
            output_dir=output_dir,
            symbol=symbol,
            horizons=horizon_list,
            models=model_list,
            bar_timeframe=bar_timeframe,
            mtf=mtf,
            n_splits=None,
            purge_bars=gap_bars,
            embargo_bars=None,
            command="walk-forward",
            random_seed=seed,
        )
    except Exception as e:
        show_error(f"Data preparation failed: {e}")
        raise typer.Exit(1) from None

    purge, embargo = factory.cv_gaps
    config = WalkForwardConfig(
        n_windows=n_windows,
        window_type=window_type,
        min_train_pct=min_train_pct,
        test_pct=test_pct,
        embargo_bars=embargo,
        gap_bars=purge,
    )

    console.print("=" * 60)
    console.print("[bold]WALK-FORWARD EVALUATION[/bold]")
    console.print("=" * 60)
    console.print(f"Models: {model_list}")
    console.print(f"Horizons: {horizon_list}")
    console.print(f"Config: {config}")
    console.print(f"Output dir: {results_dir}", soft_wrap=True)

    all_results: list[WalkForwardResult] = []

    for horizon in horizon_list:
        console.print("-" * 60)
        console.print(f"[bold]HORIZON {horizon}[/bold]")
        console.print("-" * 60)

        container = containers[horizon]
        label_spans = container.get_label_spans("train")
        if label_spans is not None:
            logger.info("  Using label spans for overlap-aware purging")

        for model_name in model_list:
            try:
                result = _run_walk_forward_for_model(
                    container=container,
                    model_name=model_name,
                    config=config,
                    seed=seed,
                    label_spans=label_spans,
                )
                all_results.append(result)

                console.print(
                    f"  {model_name} H{horizon}: "
                    f"mean_acc={result.mean_accuracy:.3f} "
                    f"(std={result.std_accuracy:.3f}), "
                    f"mean_f1={result.mean_f1:.3f}, "
                    f"time={result.total_time:.1f}s"
                )

                result_path = results_dir / f"wf_{model_name}_h{horizon}.json"
                with open(result_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)

                pred_path = results_dir / f"wf_preds_{model_name}_h{horizon}.parquet"
                result.predictions.to_parquet(pred_path, index=False)

            except Exception as e:
                show_error(f"Failed {model_name} H{horizon}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                continue

    console.print("=" * 60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 60)

    if not all_results:
        show_error("No results generated: every model failed")
        raise typer.Exit(1)

    summary_df = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "horizon": r.horizon,
                "mean_acc": f"{r.mean_accuracy:.3f}",
                "std_acc": f"{r.std_accuracy:.3f}",
                "mean_f1": f"{r.mean_f1:.3f}",
                "n_windows": r.n_windows,
                "time_s": f"{r.total_time:.1f}",
            }
            for r in all_results
        ]
    )
    console.print(summary_df.to_string(index=False))

    summary_path = results_dir / "walk_forward_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    console.print(f"\nSummary saved to: {summary_path}", soft_wrap=True)
    console.print(f"Results saved to: {results_dir}", soft_wrap=True)


# =============================================================================
# CPCV-PBO COMMAND
# =============================================================================


def _forward_returns_and_costs(
    split_df: pd.DataFrame,
    symbol_column: str,
    include_costs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Next-bar returns, per-bar cost per unit turnover, and symbol ids.

    forward_return_t = close_{t+1} / close_t - 1 within each symbol (NaN on the
    last bar of a symbol, treated as flat). The cost of one side of a trade is
    half the symbol's round-trip cost (commission + slippage, in ticks) times
    its tick size, expressed as a fraction of close_t.
    """
    from src.config.symbol import SymbolConfig
    from src.data.pipeline.config.barriers_config import get_total_trade_cost

    logger = logging.getLogger(__name__)

    if "close" not in split_df.columns:
        raise ValueError(
            "cpcv-pbo needs a 'close' column in the data to compute per-bar strategy returns"
        )
    close = split_df["close"].astype(float)
    symbols = split_df[symbol_column] if symbol_column in split_df.columns else None

    next_close = close.groupby(symbols).shift(-1) if symbols is not None else close.shift(-1)
    forward_returns = (next_close / close - 1.0).to_numpy()

    cost = np.zeros(len(split_df))
    if include_costs:
        if symbols is None:
            logger.warning("No symbol column; strategy returns are computed without costs")
        else:
            close_values = close.to_numpy()
            for symbol in symbols.unique():
                try:
                    tick_size = SymbolConfig.from_symbol(str(symbol)).tick_size
                except ValueError:
                    logger.warning(f"Unknown symbol {symbol!r}; no costs applied to it")
                    continue
                per_side_price = get_total_trade_cost(str(symbol)) / 2.0 * tick_size
                rows = (symbols == symbol).to_numpy()
                cost[rows] = per_side_price / close_values[rows]

    groups = symbols.to_numpy() if symbols is not None else None
    return forward_returns, cost, groups


def _require_directional_labels(container) -> None:
    """CPCV/PBO backtests sign(prediction) as a position; binary labels have no sign."""
    if container.n_classes == 2:
        raise ValueError(
            "cpcv-pbo needs directional labels {-1, 0, +1}: binary labels {0: no move, "
            "1: move} carry no trade direction, so strategy returns are undefined"
        )


def _run_cpcv_for_model(
    container,
    model_name: str,
    cpcv_config,
    forward_returns: np.ndarray,
    cost_per_turnover: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    label_spans=None,
):
    """
    CPCV one model and backtest every assembled path.

    Returns:
        (CPCVResult with one entry per path, path-averaged per-bar returns)

    Raises:
        ValueError: For binary labels, which carry no trade direction.
    """
    _require_directional_labels(container)

    from sklearn.metrics import accuracy_score, f1_score

    from src.models.base import PredictionResult
    from src.models.registry import ModelRegistry
    from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVPathResult, CPCVResult
    from src.validation.cv.early_stopping_split import carve_early_stopping_split
    from src.validation.cv.fold_scaling import FoldAwareScaler, get_scaling_method_for_model
    from src.validation.cv.pbo import directional_strategy_returns
    from src.validation.deflated_sharpe import sharpe_ratio_per_period

    logger = logging.getLogger(__name__)

    X, y, weights = container.get_sklearn_arrays("train", return_df=True)
    n_samples = len(X)

    cpcv = CombinatorialPurgedCV(cpcv_config)
    # Early stopping selects on a purged tail of each split's train rows,
    # never on the test groups being evaluated
    purge_bars = cpcv_config.purge_bars
    scaling_method = get_scaling_method_for_model(model_name)

    logger.info(
        f"Running CPCV for {model_name} ({cpcv.get_n_splits()} splits, {cpcv.n_paths} paths)"
    )

    split_preds: dict[int, np.ndarray] = {}
    for train_idx, test_idx, split_id in cpcv.split(X, y, label_spans=label_spans):
        logger.debug(f"  Split {split_id}: train={len(train_idx)}, test={len(test_idx)}")

        es_split = carve_early_stopping_split(train_idx, purge_bars)
        fit_idx, es_idx = es_split.fit_idx, es_split.es_idx

        # Fold-aware scaling (fit on the fit rows only)
        scaler = FoldAwareScaler(method=scaling_method)
        scaling_result = scaler.fit_transform_fold(
            X.iloc[fit_idx].values, np.vstack([X.iloc[es_idx].values, X.iloc[test_idx].values])
        )
        X_es_scaled = scaling_result.X_val_scaled[: len(es_idx)]
        X_test_scaled = scaling_result.X_val_scaled[len(es_idx) :]

        model = ModelRegistry.create(
            model_name, config=apply_model_seed({"n_classes": container.n_classes}, seed)
        )
        model.fit(
            X_train=scaling_result.X_train_scaled,
            y_train=y.iloc[fit_idx].values,
            X_val=X_es_scaled,
            y_val=y.iloc[es_idx].values,
            sample_weights=weights.iloc[fit_idx].values,
        )
        prediction_output: PredictionResult = model.predict(X_test_scaled)
        split_preds[split_id] = np.asarray(prediction_output.class_predictions, dtype=np.float64)

    pred_paths = cpcv.assemble_paths(split_preds, n_samples)
    assignments = cpcv.get_path_assignments()
    y_true = y.to_numpy()

    path_results = []
    path_returns = []
    for p in range(cpcv.n_paths):
        preds = pred_paths[p]
        returns = directional_strategy_returns(preds, forward_returns, cost_per_turnover, groups)
        path_returns.append(returns)
        path_results.append(
            CPCVPathResult(
                path_id=p,
                split_ids=tuple(int(s) for s in assignments[p]),
                n_samples=n_samples,
                accuracy=float(accuracy_score(y_true, preds)),
                f1=float(f1_score(y_true, preds, average="weighted", zero_division=0)),
                sharpe=sharpe_ratio_per_period(returns),
                returns=returns,
            )
        )
        logger.debug(
            f"    Path {p}: acc={path_results[-1].accuracy:.3f}, "
            f"sharpe/bar={path_results[-1].sharpe:.4f}"
        )

    result = CPCVResult(
        config=cpcv_config,
        path_results=path_results,
        model_name=model_name,
        horizon=container.horizon,
    )
    return result, np.mean(path_returns, axis=0)


def run_cpcv_pbo(
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Raw OHLCV file (parquet or csv)"
    ),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated models or 'all'"),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Comma-separated horizons"),
    bar_timeframe: str | None = typer.Option(
        None, "--bar-timeframe", help="Resample input bars first, e.g. 5min"
    ),
    mtf: bool = typer.Option(True, "--mtf/--no-mtf", help="Multi-timeframe features"),
    # CPCV configuration
    n_groups: int = typer.Option(6, "--n-groups", help="Number of time groups"),
    n_test_groups: int = typer.Option(2, "--n-test-groups", help="Groups held out as test"),
    purge_bars: int | None = typer.Option(
        None, "--purge-bars", help="Bars purged around each test group (default: label span)"
    ),
    embargo_bars: int | None = typer.Option(
        None, "--embargo-bars", help="Bars embargoed after each test group (default: one day)"
    ),
    # PBO configuration
    n_partitions: int = typer.Option(16, "--n-partitions", help="CSCV row blocks S for PBO (even)"),
    pbo_warn: float = typer.Option(0.5, "--pbo-warn", help="PBO warning threshold"),
    pbo_block: float = typer.Option(0.8, "--pbo-block", help="PBO blocking threshold"),
    no_costs: bool = typer.Option(
        False, "--no-costs", help="Compute strategy returns without transaction costs"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output root; writes <dir>/<run_id>/cpcv-pbo/",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed (models, Optuna, feature selection)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Run CPCV and PBO evaluation for model selection gating.

    Each model is evaluated with Combinatorial Purged Cross-Validation; its
    out-of-sample predictions are assembled into the CPCV backtest paths and
    traded as position = sign(prediction) on the next-bar return (minus
    per-symbol costs). PBO is estimated with CSCV over the T x N matrix of
    path-averaged per-bar returns of the N models, so it needs at least two models.

    Examples:

        ml cpcv-pbo -d data/mes_5min.parquet -m xgboost,lightgbm --horizons 20

        ml cpcv-pbo -d data/mes_5min.parquet -m xgboost,catboost --pbo-warn 0.4 --pbo-block 0.7
    """
    setup_logging(verbose)

    model_list = parse_tabular_model_list(models)
    horizon_list = parse_horizon_list(horizons)

    from src.validation.cv.cpcv import CPCVConfig, CPCVResult
    from src.validation.cv.pbo import PBOConfig, compute_pbo, pbo_gate

    try:
        factory, containers, results_dir = _load_evaluation_data(
            data_path=data_path,
            output_dir=output_dir,
            symbol=symbol,
            horizons=horizon_list,
            models=model_list,
            bar_timeframe=bar_timeframe,
            mtf=mtf,
            n_splits=None,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            command="cpcv-pbo",
            random_seed=seed,
        )
    except Exception as e:
        show_error(f"Data preparation failed: {e}")
        raise typer.Exit(1) from None

    purge, embargo = factory.cv_gaps
    cpcv_config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_bars=purge,
        embargo_bars=embargo,
    )
    pbo_config = PBOConfig(
        n_partitions=n_partitions,
        warn_threshold=pbo_warn,
        block_threshold=pbo_block,
    )

    console.print("=" * 60)
    console.print("[bold]CPCV + PBO EVALUATION[/bold]")
    console.print("=" * 60)
    console.print(f"Models: {model_list}")
    console.print(f"Horizons: {horizon_list}")
    console.print(
        f"CPCV: {n_groups} groups, {n_test_groups} test -> "
        f"{cpcv_config.total_combinations} splits, {cpcv_config.n_paths} paths "
        f"(purge={purge}, embargo={embargo})"
    )
    console.print(f"PBO: S={n_partitions}, thresholds warn={pbo_warn}, block={pbo_block}")

    all_results = []
    n_models_evaluated = 0

    for horizon in horizon_list:
        console.print("-" * 60)
        console.print(f"[bold]HORIZON {horizon}[/bold]")
        console.print("-" * 60)

        container = containers[horizon]
        try:
            _require_directional_labels(container)
        except ValueError as e:
            show_error(str(e))
            raise typer.Exit(1) from None
        try:
            split = container.get_split("train")
            forward_returns, costs, groups = _forward_returns_and_costs(
                split.df, split.symbol_column, include_costs=not no_costs
            )
            label_spans = container.get_label_spans("train")
        except Exception as e:
            show_error(f"Failed to prepare H{horizon}: {e}")
            continue

        cpcv_results: dict[str, CPCVResult] = {}
        model_returns: dict[str, np.ndarray] = {}
        for model_name in model_list:
            try:
                result, mean_returns = _run_cpcv_for_model(
                    container=container,
                    model_name=model_name,
                    cpcv_config=cpcv_config,
                    forward_returns=forward_returns,
                    cost_per_turnover=costs,
                    groups=groups,
                    seed=seed,
                    label_spans=label_spans,
                )
                cpcv_results[model_name] = result
                model_returns[model_name] = mean_returns

                console.print(
                    f"  {model_name}: mean_acc={result.mean_accuracy:.3f}, "
                    f"std={result.std_accuracy:.3f}, "
                    f"mean_sharpe_per_bar={result.mean_sharpe:.4f}"
                )

            except Exception as e:
                show_error(f"Failed {model_name}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                continue

        if len(model_returns) >= 2:
            names = list(model_returns)
            returns_matrix = np.column_stack([model_returns[n] for n in names])
            pbo_result = compute_pbo(returns_matrix, pbo_config)

            console.print("-" * 40)
            console.print(f"[bold]PBO Analysis for H{horizon}:[/bold]")
            console.print(f"  PBO: {pbo_result.pbo:.3f} ({pbo_result.n_combinations} CSCV splits)")
            console.print(f"  Risk Level: {pbo_result.get_risk_level()}")
            console.print(f"  Degradation slope (OOS~IS): {pbo_result.performance_degradation:.3f}")
            console.print(f"  P(OOS loss): {pbo_result.prob_oos_loss:.3f}")
            console.print(f"  Rank Correlation: {pbo_result.rank_correlation:.3f}")

            should_proceed, reason = pbo_gate(pbo_result, strict=False)
            console.print(
                f"  Gate Decision: {'[green]PASS[/green]' if should_proceed else '[red]FAIL[/red]'}"
            )
            console.print(f"  Reason: {reason}")

            all_results.append(
                {
                    "horizon": horizon,
                    "n_models": len(names),
                    "pbo": pbo_result.pbo,
                    "is_overfit": pbo_result.is_overfit,
                    "should_block": pbo_result.should_block,
                    "risk_level": pbo_result.get_risk_level(),
                    "gate_pass": should_proceed,
                }
            )

            pbo_payload = pbo_result.to_dict()
            pbo_payload["strategies"] = names
            with open(results_dir / f"pbo_h{horizon}.json", "w") as f:
                json.dump(pbo_payload, f, indent=2)
        elif model_returns:
            show_warning("PBO needs at least 2 successfully evaluated models; skipped")

        n_models_evaluated += len(cpcv_results)
        for model_name, result in cpcv_results.items():
            result_path = results_dir / f"cpcv_{model_name}_h{horizon}.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

    console.print("=" * 60)
    console.print("[bold green]SUMMARY[/bold green]")
    console.print("=" * 60)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        console.print(summary_df.to_string(index=False))

        summary_path = results_dir / "cpcv_pbo_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        console.print(f"\nSummary saved to: {summary_path}", soft_wrap=True)

        if any(r["should_block"] for r in all_results):
            show_warning("Some horizons have PBO > block threshold!")
            raise typer.Exit(1) from None
    elif n_models_evaluated == 0:
        show_error("No results generated: every model failed")
        raise typer.Exit(1)
    else:
        show_warning("No PBO results generated (needs 2+ evaluated models per horizon)")

    console.print(f"Results saved to: {results_dir}", soft_wrap=True)
