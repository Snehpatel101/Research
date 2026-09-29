"""
Pipeline Commands - run, data, status, models.

Every command runs on MLFactory: raw OHLCV bars -> FeatureEngineer -> triple-barrier
labels (with label spans) -> models -> ensemble -> backtest -> deploy artifact.

- ``run``:    the full pipeline (train one model, several, or an ensemble; any training mode)
- ``data``:   the data step only - features and labels written to parquet
- ``status``: checkpoint progress of a run directory
- ``models``: registered models and their default configuration
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from src.cli.utils import (
    DEFAULT_OUTPUT_DIR,
    build_experiment_config,
    console,
    parse_horizon_list,
    setup_logging,
    show_error,
    show_info,
)

logger = logging.getLogger(__name__)

RUN_CONFIG_FILE = "experiment_config.yaml"


def _print_run_result(result) -> None:
    """Print the summary of a finished MLFactory run."""
    console.print("\n" + "=" * 70)
    console.print("[bold green]PIPELINE COMPLETED SUCCESSFULLY[/bold green]")
    console.print("=" * 70)
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output: {result.output_dir}", soft_wrap=True)
    console.print(f"Duration: {result.duration_seconds:.1f}s")

    if result.best_model:
        console.print(f"\nBest model: {result.best_model}")

    for title, values in (("Metrics", result.metrics), ("Backtest", result.backtest_metrics)):
        scalars = {k: v for k, v in (values or {}).items() if isinstance(v, int | float | str)}
        if scalars:
            console.print(f"\n{title}:")
            for k, v in scalars.items():
                console.print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def run_pipeline(
    data_path: Path | None = typer.Option(
        None, "--data-path", "-d", help="Raw OHLCV file (parquet or csv)"
    ),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    models: str = typer.Option(
        "xgboost", "--models", "-m", help="Models to train (comma-separated; see `ml models`)"
    ),
    training_mode: str = typer.Option(
        "standard",
        "--training-mode",
        help="Training mode: standard, walk_forward, regime_aware, meta_labeling",
    ),
    build_ensemble: bool = typer.Option(
        False, "--build-ensemble", help="Stack the base models into an ensemble"
    ),
    meta_learner: str = typer.Option(
        "ridge_meta",
        "--meta-learner",
        help="Stacking meta-learner: ridge_meta, xgboost_meta, mlp_meta, calibrated_meta, voting_meta",
    ),
    bar_timeframe: str | None = typer.Option(
        None, "--bar-timeframe", help="Resample input bars before training, e.g. 5min"
    ),
    mtf: bool = typer.Option(
        True, "--mtf/--no-mtf", help="Multi-timeframe features (shift(1) anti-lookahead)"
    ),
    n_splits: int | None = typer.Option(None, "--n-splits", help="Purged CV folds (default 5)"),
    purge_bars: int | None = typer.Option(
        None,
        "--purge-bars",
        help="CV purge gap in bars (default: longest label span, max_bars over horizons; "
        "smaller values are raised to it)",
    ),
    embargo_bars: int | None = typer.Option(
        None,
        "--embargo-bars",
        help="CV embargo gap in bars (default: one trading day at the bar timeframe, "
        "capped at 25% of a CV fold)",
    ),
    n_trials: int = typer.Option(
        0, "--n-trials", help="Optuna hyperparameter trials per model (0 = no tuning)"
    ),
    max_epochs: int | None = typer.Option(None, "--max-epochs", help="Neural max epochs"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Neural batch size"),
    backtest: bool = typer.Option(
        False,
        "--backtest/--no-backtest",
        help="Backtest the predictions with barrier-aligned exits",
    ),
    deploy: bool = typer.Option(
        True, "--deploy/--no-deploy", help="Write the bundle and deploy artifact"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", "-o", help="Output root; the run writes <dir>/<run_id>/"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="ExperimentConfig YAML (replaces the options above)"
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        help="Run directory to resume from its last checkpoint (settings come from the "
        "run's experiment_config.yaml)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """
    Run the full pipeline: features, labels, training, ensemble, backtest, deploy.

    One model or many, any training mode. This is the single training entry point.

    Examples:

        ml run -d data/mes_5min.parquet -m xgboost

        ml run -d data/mes_1m.parquet --bar-timeframe 5min \\
            -m xgboost,lstm,patchtst --build-ensemble --meta-learner voting_meta

        ml run -d data/mes_5min.parquet -m xgboost --training-mode walk_forward --backtest

        ml run --resume experiments/<run_id>
    """
    from src.config.experiment import ExperimentConfig
    from src.factory import MLFactory

    setup_logging(verbose)

    try:
        if resume is not None:
            config_path = resume / RUN_CONFIG_FILE
            if not config_path.exists():
                raise FileNotFoundError(f"No {RUN_CONFIG_FILE} in run directory {resume}")
            ml_config = ExperimentConfig.from_yaml(config_path)
        elif config is not None:
            ml_config = ExperimentConfig.from_yaml(config)
            if data_path is not None:
                ml_config.data.data_path = data_path
        else:
            if data_path is None:
                raise ValueError("--data-path is required (or use --config / --resume)")
            ml_config = build_experiment_config(
                data_path=data_path,
                output_dir=output_dir,
                symbol=symbol,
                horizons=parse_horizon_list(horizons),
                models=[m.strip() for m in models.split(",") if m.strip()],
                bar_timeframe=bar_timeframe,
                mtf=mtf,
                training_mode=training_mode,
                build_ensemble=build_ensemble,
                meta_learner=meta_learner,
                n_trials=n_trials,
                max_epochs=max_epochs,
                batch_size=batch_size,
                n_splits=n_splits,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                run_backtest=backtest,
                deploy=deploy,
            )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        show_error(f"Configuration error: {e}")
        raise typer.Exit(1) from None

    console.print("\n[bold]Starting ML pipeline[/bold]")
    console.print(f"Symbol: {ml_config.symbol}")
    console.print(f"Horizons: {ml_config.horizons}")
    console.print(f"Models: {ml_config.models}")
    console.print(f"Mode: {ml_config.training.training_mode}")
    console.print(f"Data path: {ml_config.data.data_path}")
    console.print(f"Output dir: {ml_config.output_dir}")
    console.print()

    try:
        factory = MLFactory(ml_config)
        result = factory.resume_from_checkpoint() if resume is not None else factory.run()
    except Exception as e:
        show_error(f"Pipeline failed: {e}")
        raise typer.Exit(1) from None

    _print_run_result(result)


def run_data(
    data_path: Path = typer.Option(
        ..., "--data-path", "-d", help="Raw OHLCV file (parquet or csv)"
    ),
    symbol: str = typer.Option("MES", "--symbol", "-s", help="Trading symbol"),
    horizons: str = typer.Option("20", "--horizons", "-h", help="Label horizons (comma-separated)"),
    bar_timeframe: str | None = typer.Option(
        None, "--bar-timeframe", help="Resample input bars first, e.g. 5min"
    ),
    mtf: bool = typer.Option(True, "--mtf/--no-mtf", help="Multi-timeframe features"),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", "-o", help="Output root; writes <dir>/<run_id>/"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
):
    """
    Compute features and triple-barrier labels and write them to parquet (no training).

    The frame is exactly what `ml run` trains on: engineered features, one
    `label_h<H>` column per horizon and the `label_end_h<H>` row positions at
    which each label resolves (used for purging).

    Example:

        ml data -d data/mes_5min.parquet --horizons 5,20
    """
    from src.factory import MLFactory

    setup_logging(verbose)

    try:
        ml_config = build_experiment_config(
            data_path=data_path,
            output_dir=output_dir,
            symbol=symbol,
            horizons=parse_horizon_list(horizons),
            models=["xgboost"],
            bar_timeframe=bar_timeframe,
            mtf=mtf,
            name=f"{symbol}_data",
        )
        factory = MLFactory(ml_config, enable_checkpoints=False)
        df, _ = factory.prepare_data()
    except Exception as e:
        show_error(f"Data step failed: {e}")
        raise typer.Exit(1) from None

    out_path = factory.output_dir / "features_labels.parquet"
    df.to_parquet(out_path)

    console.print("\n[bold green]DATA STEP COMPLETED[/bold green]")
    console.print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    for horizon in ml_config.horizons:
        counts = df[f"label_h{horizon}"].value_counts().to_dict()
        console.print(f"label_h{horizon}: {counts}")
    console.print(f"Written: {out_path}", soft_wrap=True)


def show_status(
    run_dir: Path = typer.Argument(..., help="Run directory (<output_dir>/<run_id>)"),
):
    """
    Show checkpoint progress of a run.

    Lists the stages `ml run` completed (data, training, evaluation, bundling) and
    what `ml run --resume <run_dir>` would do next.

    Example:

        ml status experiments/20260929_101500_ab12cd
    """
    from src.core.checkpoint import PipelineCheckpointManager
    from src.factory import MLFactory

    if not run_dir.is_dir():
        show_error(f"Run directory not found: {run_dir}")
        raise typer.Exit(1)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = (
        PipelineCheckpointManager(run_dir).get_all_checkpoints() if checkpoint_dir.is_dir() else []
    )

    console.print("=" * 70)
    console.print(f"[bold]RUN STATUS[/bold]  {run_dir}")
    console.print("=" * 70)
    if not checkpoints:
        console.print("No checkpoints: the run has not completed its first stage.")
        show_info(f"Start it with `ml run --resume {run_dir}` (needs {RUN_CONFIG_FILE})")
        return

    for state in checkpoints:
        console.print(
            f"  [green]+ {state.stage_name}[/green]  "
            f"{state.completed_at:%Y-%m-%d %H:%M:%S}  "
            + " ".join(f"{k}={v}" for k, v in state.metadata.items())
        )
    last = checkpoints[-1].stage_index
    if last >= MLFactory.STAGE_BUNDLING:
        console.print("\nAll stages completed.")
    else:
        console.print(f"\nNext: stage {last + 1} - `ml run --resume {run_dir}`")


def list_models(
    model: str | None = typer.Argument(
        None, help="Model name for its default configuration; omit to list all models"
    ),
):
    """
    List registered models, or show one model's details and default configuration.

    Examples:

        ml models

        ml models xgboost
    """
    import src.models  # noqa: F401 - registers models
    from src.models.registry import ModelRegistry

    if model is None:
        console.print("\n[bold]Available Models:[/bold]")
        console.print("=" * 60)
        for family, names in sorted(ModelRegistry.list_models().items()):
            console.print(f"\n[bold cyan]{family.upper()}:[/bold cyan]")
            for name in sorted(names):
                description = ModelRegistry.get_metadata(name).get("description", "")
                console.print(f"  - {name}: {description}")
        console.print(f"\n[bold]Total: {ModelRegistry.count()} models[/bold]")
        return

    try:
        info = ModelRegistry.get_model_info(model)
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
