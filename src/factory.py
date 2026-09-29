"""
MLFactory - Unified Entry Point for ML Factory Operations.

This is THE single entry point for the ML Factory system, coordinating:
- Data preparation (in-process: raw bars -> FeatureEngineer -> TripleBarrierLabeler;
  the standalone PipelineRunner behind `ml data` is not used here)
- Training (via UnifiedTrainingOrchestrator)
- Evaluation (optional)
- Bundling (via BundleBuilder)

The factory pattern provides a clean, high-level API while delegating
heavy lifting to specialized components.

Supports checkpointing for pipeline recovery (Phase 17A):
- Enable with `enable_checkpoints=True`
- Resume failed runs with `resume_from_checkpoint()`
- Automatic config change detection

Example:
    from src.factory import MLFactory
    from src.config.experiment import ExperimentConfig

    config = ExperimentConfig(
        name="mes_xgboost_experiment",
        symbol="MES",
        models=["xgboost", "lightgbm"],
        horizons=[5, 10, 15, 20],
    )

    factory = MLFactory(config, enable_checkpoints=True)
    result = factory.run()

    print(f"Trained {result.n_models} models")
    print(f"Best model: {result.best_model}")
    print(f"Bundle path: {result.bundle_path}")

    # Resume a failed run
    factory = MLFactory(config, enable_checkpoints=True)
    result = factory.resume_from_checkpoint()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.config.symbol import SymbolConfig

if TYPE_CHECKING:
    from src.models.training.unified_orchestrator import TrainingRunResult
from src.core.checkpoint import PipelineCheckpointManager, compute_config_hash

logger = logging.getLogger(__name__)

# Data-checkpoint file holding the labeler's barrier cost term per horizon
LABEL_COSTS_FILE = "label_costs.json"


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class ExperimentResult:
    """
    Result from a complete MLFactory experiment run.

    Attributes:
        run_id: Unique identifier for this run
        config: ExperimentConfig used
        success: Whether the run completed successfully
        duration_seconds: Total wall-clock time
        n_models: Number of models trained
        best_model: Name of best-performing model
        metrics: Model performance metrics
        ensemble_metrics: Ensemble performance metrics (if built)
        backtest_metrics: Backtest results (if run); ``strategy`` names the
            signals replayed and ``horizon`` the horizon whose barriers were used
        bundle_path: Path to deployment bundle (if created)
        output_dir: Directory containing all artifacts
        error_message: Error message if failed
    """

    run_id: str
    config: ExperimentConfig
    success: bool
    duration_seconds: float = 0.0
    n_models: int = 0
    best_model: str | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    backtest_metrics: dict[str, Any] = field(default_factory=dict)
    bundle_path: Path | None = None
    deploy_path: Path | None = None
    output_dir: Path | None = None
    error_message: str | None = None
    training_result: TrainingRunResult | None = None
    equity_curve: Any = None
    backtest_trades: list = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "=" * 60,
            f"ML Factory Experiment: {status}",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"Models Trained: {self.n_models}",
        ]

        if self.best_model:
            lines.append(f"Best Model: {self.best_model}")

        if self.metrics:
            lines.append("\nModel Performance:")
            for model_name, model_metrics in self.metrics.items():
                f1 = model_metrics.get("val_f1", 0.0)
                acc = model_metrics.get("val_accuracy", 0.0)
                lines.append(f"  {model_name}: F1={f1:.4f}, Acc={acc:.4f}")

        if self.ensemble_metrics:
            lines.append("\nEnsemble Performance:")
            for k, v in self.ensemble_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")

        if self.backtest_metrics:
            lines.append("\nBacktest Results:")
            for k, v in self.backtest_metrics.items():
                if isinstance(v, int | float):
                    lines.append(f"  {k}: {v}")

        if self.bundle_path:
            lines.append(f"\nBundle: {self.bundle_path}")

        if self.deploy_path:
            lines.append(f"Deploy: {self.deploy_path}")

        if self.output_dir:
            lines.append(f"Output: {self.output_dir}")

        if not self.success and self.error_message:
            lines.append(f"\nError: {self.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# ML FACTORY
# =============================================================================


class MLFactory:
    """
    Unified entry point for ML Factory operations.

    This class coordinates the full ML pipeline:
    1. Data Pipeline: Load, clean, engineer features
    2. Training: Train models, build ensemble
    3. Evaluation: Compute metrics, run backtests (optional)
    4. Bundling: Package for deployment (optional)

    The factory delegates to specialized components:
    - FeatureEngineer + TripleBarrierLabeler: in-process data preparation
    - UnifiedTrainingOrchestrator: Model training
    - BundleBuilder: Inference artifact creation

    Supports checkpointing (Phase 17A):
    - Enable with enable_checkpoints=True
    - Saves state after each stage for recovery
    - Detects config changes and handles appropriately

    Example:
        factory = MLFactory(config, enable_checkpoints=True)
        result = factory.run()

        if result.success:
            print(f"Trained {result.n_models} models")
            print(f"Best: {result.best_model}")

        # Resume from failure
        factory = MLFactory(config, enable_checkpoints=True)
        result = factory.resume_from_checkpoint()
    """

    # Stage definitions for checkpointing
    STAGE_DATA_PIPELINE = 0
    STAGE_TRAINING = 1
    STAGE_EVALUATION = 2
    STAGE_BUNDLING = 3

    def __init__(
        self,
        config: ExperimentConfig,
        verbose: int | None = None,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize MLFactory with experiment configuration.

        Args:
            config: ExperimentConfig defining the experiment
            verbose: Logging verbosity (0=silent, 1=info, 2=debug).
                None (default) uses ``config.verbose``.
            enable_checkpoints: Whether to save checkpoints after each stage
                (default: True). Enables resume_from_checkpoint() on failure.
        """
        self.config = config
        self.verbose = config.verbose if verbose is None else verbose
        self.enable_checkpoints = enable_checkpoints

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint manager
        self._checkpoint_manager: PipelineCheckpointManager | None = None
        self._config_hash: str = ""
        if enable_checkpoints:
            self._checkpoint_manager = PipelineCheckpointManager(self.output_dir)
            self._config_hash = compute_config_hash(config)

        # Cache for intermediate results (used during resume)
        self._cached_df: pd.DataFrame | None = None

        # Raw-OHLCV -> features recipe (bar timeframe + FeatureEngineer spec),
        # recorded by the data pipeline and baked into bundles for inference.
        self._feature_pipeline: dict[str, Any] | None = None
        # (purge_bars, embargo_bars) resolved once the labeled frame is known
        self._cv_gaps: tuple[int, int] | None = None

        # Backtest artifacts (populated by _run_evaluation)
        self._last_equity_curve: Any = None
        self._last_backtest_trades: list = []
        # Barrier cost term (ATR units) the labeler applied per horizon, handed
        # to the backtester so its stop/TP distances equal the label barriers
        self._label_cost_in_atr: dict[int, float] = {}

        # Save config
        config_path = self.output_dir / "experiment_config.yaml"
        config.save_yaml(config_path)

        if self.verbose >= 1:
            logger.info(f"MLFactory initialized: {self.output_dir.name}")
            logger.info(f"Config saved to: {config_path}")
            if enable_checkpoints:
                logger.info("Checkpointing enabled")

    def run(self, resume: bool = False) -> ExperimentResult:
        """
        Execute the complete ML Factory pipeline.

        Phases:
        1. Data Pipeline: Prepare features and labels
        2. Training: Train models and ensemble
        3. Evaluation: Compute metrics (optional backtest)
        4. Bundling: Create deployment artifacts (optional)

        Args:
            resume: If True, attempt to resume from last checkpoint
                (only works if enable_checkpoints=True)

        Returns:
            ExperimentResult with metrics and artifacts

        Raises:
            Exception: If any critical phase fails
        """
        start_time = datetime.now()
        self._log("=" * 60)
        self._log(f"Starting ML Factory Experiment: {self.config.name}")
        self._log("=" * 60)

        # Determine resume stage
        resume_from_stage = 0
        if resume and self._checkpoint_manager:
            resume_from_stage = self._get_resume_stage()
            if resume_from_stage > 0:
                self._log(f"Resuming from stage {resume_from_stage}")

        try:
            # Phase 1: Data Pipeline
            if resume_from_stage <= self.STAGE_DATA_PIPELINE:
                self._log("\n[Phase 1/4] Data Pipeline")
                df, additional_dfs = self._run_data_pipeline()
                self._save_checkpoint_data_pipeline(df, additional_dfs)
            else:
                self._log("\n[Phase 1/4] Data Pipeline (cached)")
                df = self._load_cached_data()
                additional_dfs = self._load_cached_additional_dfs()
                pipeline_path = self.output_dir / "cache" / "feature_pipeline.json"
                if pipeline_path.exists():
                    with open(pipeline_path) as f:
                        self._feature_pipeline = json.load(f)
                costs_path = self.output_dir / "cache" / LABEL_COSTS_FILE
                if costs_path.exists():
                    with open(costs_path) as f:
                        self._label_cost_in_atr = {int(h): c for h, c in json.load(f).items()}

            # Resolve purge/embargo for this data, then validate sufficiency
            self._pipeline_config(n_rows=len(df))
            self._validate_data_sufficiency(df)

            # Phase 2: Training
            if resume_from_stage <= self.STAGE_TRAINING:
                self._log("\n[Phase 2/4] Model Training")
                training_result = self._run_training(df, additional_dfs=additional_dfs)
                self._save_checkpoint_training(training_result)
            else:
                self._log("\n[Phase 2/4] Model Training (cached)")
                training_result = self._load_cached_training()

            # Phase 3: Evaluation (optional)
            if resume_from_stage <= self.STAGE_EVALUATION:
                self._log("\n[Phase 3/4] Evaluation")
                backtest_metrics = self._run_evaluation(df, training_result)
                self._save_checkpoint_evaluation(backtest_metrics)
            else:
                self._log("\n[Phase 3/4] Evaluation (cached)")
                backtest_metrics = self._load_cached_evaluation()

            # Phase 4: Bundling (optional)
            self._log("\n[Phase 4/4] Bundling")
            bundle_path = self._create_bundle(training_result)
            self._save_checkpoint_bundling(bundle_path)

            # Phase 4b: Deploy artifact (optional)
            deploy_path = self._create_deploy(training_result, bundle_path)

            # Build result
            duration = (datetime.now() - start_time).total_seconds()
            result = ExperimentResult(
                run_id=self.config.run_id,
                config=self.config,
                success=True,
                duration_seconds=duration,
                n_models=training_result.n_models,
                best_model=training_result.best_model,
                metrics=training_result.get_metrics_summary(),
                ensemble_metrics=self._extract_ensemble_metrics(training_result),
                backtest_metrics=backtest_metrics,
                bundle_path=bundle_path,
                deploy_path=deploy_path,
                output_dir=self.output_dir,
                training_result=training_result,
                equity_curve=self._last_equity_curve,
                backtest_trades=self._last_backtest_trades,
            )

            self._log("\n" + result.summary())
            return result

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.exception("MLFactory experiment failed")

            result = ExperimentResult(
                run_id=self.config.run_id,
                config=self.config,
                success=False,
                duration_seconds=duration,
                output_dir=self.output_dir,
                error_message=str(e),
            )

            self._log("\n" + result.summary())
            raise

    def resume_from_checkpoint(self) -> ExperimentResult:
        """
        Resume experiment from the last successful checkpoint.

        This method checks for existing checkpoints and resumes from
        the last successfully completed stage. If the configuration
        has changed since the checkpoint was created, checkpoints are
        cleared and execution starts fresh.

        Returns:
            ExperimentResult from the resumed run

        Raises:
            ValueError: If checkpointing is not enabled
            Exception: If the resumed run fails

        Example:
            factory = MLFactory(config, enable_checkpoints=True)
            try:
                result = factory.run()
            except Exception:
                # Something failed, resume later
                result = factory.resume_from_checkpoint()
        """
        if not self._checkpoint_manager:
            raise ValueError(
                "Cannot resume: checkpointing not enabled. "
                "Initialize with enable_checkpoints=True"
            )

        if not self._checkpoint_manager.has_checkpoint():
            self._log("No checkpoint found, starting fresh run")
            return self.run(resume=False)

        # Validate config hasn't changed
        if not self._checkpoint_manager.validate_config(self._config_hash):
            self._log("Config changed since checkpoint, starting fresh")
            self._checkpoint_manager.clear_checkpoints()
            return self.run(resume=False)

        return self.run(resume=True)

    def _get_resume_stage(self) -> int:
        """Get the stage to resume from based on checkpoints."""
        if not self._checkpoint_manager:
            return 0
        return self._checkpoint_manager.get_resume_stage()

    def _save_checkpoint_data_pipeline(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Save checkpoint after data pipeline stage."""
        if not self._checkpoint_manager:
            return

        # Save DataFrame to cache
        data_cache_path = self.output_dir / "cache" / "data_pipeline.parquet"
        data_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_cache_path)

        if self._feature_pipeline is not None:
            with open(self.output_dir / "cache" / "feature_pipeline.json", "w") as f:
                json.dump(self._feature_pipeline, f, indent=2)
        # The labeler's barrier cost term per horizon, reused by the backtest
        with open(self.output_dir / "cache" / LABEL_COSTS_FILE, "w") as f:
            json.dump({str(h): c for h, c in self._label_cost_in_atr.items()}, f)

        # Save additional_dfs (multi-stream data for 4D models like PatchTST)
        if additional_dfs:
            for tf_key, tf_df in additional_dfs.items():
                mtf_path = self.output_dir / "cache" / f"mtf_{tf_key}.parquet"
                tf_df.to_parquet(mtf_path)

        self._checkpoint_manager.save_checkpoint(
            stage_name="data_pipeline",
            stage_index=self.STAGE_DATA_PIPELINE,
            artifacts={"data": data_cache_path},
            config_hash=self._config_hash,
            n_rows=len(df),
            n_columns=len(df.columns),
        )

    def _save_checkpoint_training(self, training_result: TrainingRunResult) -> None:
        """Save checkpoint after training stage."""
        if not self._checkpoint_manager:
            return

        import pickle

        # Save training result
        training_cache_path = self.output_dir / "cache" / "training_result.pkl"
        training_cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(training_cache_path, "wb") as f:
                pickle.dump(training_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (RuntimeError, pickle.PicklingError) as e:
            # torch.compile'd models cannot be pickled; save without models
            logger.warning(
                f"Could not pickle training result (likely torch.compile'd model): {e}. "
                "Saving checkpoint metadata only — training cache will not be resumable."
            )
            training_cache_path.unlink(missing_ok=True)

        self._checkpoint_manager.save_checkpoint(
            stage_name="training",
            stage_index=self.STAGE_TRAINING,
            artifacts={"training_result": training_cache_path},
            config_hash=self._config_hash,
            n_models=training_result.n_models,
            best_model=training_result.best_model,
        )

    def _save_checkpoint_evaluation(self, backtest_metrics: dict) -> None:
        """Save checkpoint after evaluation stage."""
        if not self._checkpoint_manager:
            return

        # Save backtest metrics
        eval_cache_path = self.output_dir / "cache" / "evaluation.json"
        eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_cache_path, "w") as f:
            json.dump(backtest_metrics, f)

        self._checkpoint_manager.save_checkpoint(
            stage_name="evaluation",
            stage_index=self.STAGE_EVALUATION,
            artifacts={"evaluation": eval_cache_path},
            config_hash=self._config_hash,
        )

    def _save_checkpoint_bundling(self, bundle_path: Path | None) -> None:
        """Save checkpoint after bundling stage."""
        if not self._checkpoint_manager:
            return

        artifacts = {}
        if bundle_path:
            artifacts["bundle"] = bundle_path

        self._checkpoint_manager.save_checkpoint(
            stage_name="bundling",
            stage_index=self.STAGE_BUNDLING,
            artifacts=artifacts,
            config_hash=self._config_hash,
        )

    def _load_cached_data(self) -> pd.DataFrame:
        """Load cached data from checkpoint."""
        data_cache_path = self.output_dir / "cache" / "data_pipeline.parquet"
        if data_cache_path.exists():
            return pd.read_parquet(data_cache_path)
        raise FileNotFoundError(f"Cached data not found at {data_cache_path}")

    def _load_cached_additional_dfs(self) -> dict[str, pd.DataFrame] | None:
        """Load cached multi-stream DataFrames from checkpoint.

        Falls back to regenerating from raw data if cache files are missing
        (backward compatibility with checkpoints saved before this fix).
        """
        if not self._needs_multi_stream():
            return None

        cache_dir = self.output_dir / "cache"
        mtf_files = sorted(cache_dir.glob("mtf_*.parquet"))

        if mtf_files:
            additional_dfs = {}
            for mtf_path in mtf_files:
                # Extract timeframe key from filename: mtf_15min.parquet -> 15min
                tf_key = mtf_path.stem.removeprefix("mtf_")
                additional_dfs[tf_key] = pd.read_parquet(mtf_path)
                self._log(f"  Loaded cached MTF: {tf_key} ({len(additional_dfs[tf_key])} bars)")
            return additional_dfs

        # Backward compat: regenerate from raw source file
        # (cached data_pipeline.parquet has features/labels, not raw OHLCV)
        if self.config.data.data_path and Path(self.config.data.data_path).exists():
            self._log("  No cached MTF data — regenerating from raw OHLCV source")
            raw_df, _bar_tf = self._load_raw_bars()
            return self._generate_additional_dfs(raw_df)

        self._log("  WARNING: 4D models need MTF data but none available")
        return None

    def _load_cached_training(self) -> TrainingRunResult:
        """Load cached training result from checkpoint."""
        from src.core.utils.safe_pickle import safe_pickle_load

        training_cache_path = self.output_dir / "cache" / "training_result.pkl"
        if training_cache_path.exists():
            return safe_pickle_load(training_cache_path)
        raise FileNotFoundError(f"Cached training result not found at {training_cache_path}")

    def _load_cached_evaluation(self) -> dict:
        """Load cached evaluation metrics from checkpoint."""

        eval_cache_path = self.output_dir / "cache" / "evaluation.json"
        if eval_cache_path.exists():
            with open(eval_cache_path) as f:
                return json.load(f)
        return {}

    def _needs_multi_stream(self) -> bool:
        """Check if any configured model requires MULTI_TF_4D input."""
        from src.core.contracts import get_model_contract
        from src.core.types import DataRank

        for model_name in self.config.training.models:
            try:
                contract = get_model_contract(model_name)
                if contract.input_rank == DataRank.MULTI_TF_4D:
                    return True
            except Exception:
                continue
        return False

    def _generate_additional_dfs(self, raw_df: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
        """
        Generate resampled OHLCV DataFrames for multi-stream transformer models.

        Args:
            raw_df: Raw OHLCV DataFrame with DatetimeIndex

        Returns:
            Dict mapping timeframe strings to resampled DataFrames, or None
        """
        if not self._needs_multi_stream():
            return None

        timeframes = self.config.data.mtf.timeframes
        self._log(f"  Generating multi-stream data for timeframes: {timeframes}")

        ohlcv_agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        from src.core.common.timeframes import normalize_timeframe

        additional_dfs: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            resampled = (
                raw_df[list(ohlcv_agg.keys())]
                .resample(tf, closed="left", label="left")
                .agg(ohlcv_agg)
                .dropna()
            )
            # Shift by 1 bar to prevent lookahead: bar N should only see
            # the COMPLETED higher-TF bar (bar N-1), not the current one
            # which may contain up to 59 minutes of future data.
            resampled = resampled.shift(1).dropna()
            normalized_key = normalize_timeframe(tf)
            additional_dfs[normalized_key] = resampled
            self._log(f"    {normalized_key}: {len(resampled)} bars (shifted)")

        return additional_dfs

    def _validate_data_sufficiency(self, df: pd.DataFrame) -> None:
        """
        Validate that the dataset is large enough for the CV configuration.

        Checks that embargo_bars + purge_bars leave enough room for
        meaningful train/test splits in PurgedKFold. Raises early with a
        clear message instead of failing partway through expensive training.

        Args:
            df: Prepared DataFrame (after feature engineering and labeling).

        Raises:
            ValueError: If data is insufficient for the CV configuration.
        """
        n_splits = self.config.training.n_splits
        purge_bars, embargo_bars = self._cv_gaps or self.config.resolve_cv_gaps(
            self._bar_timeframe(), len(df)
        )

        n_samples = len(df)
        # Each fold removes (purge + embargo) bars from the usable training set.
        # We need at least min_samples_per_fold usable samples in each fold.
        min_samples_per_fold = 100
        gap_per_fold = purge_bars + embargo_bars
        min_required = n_splits * min_samples_per_fold + n_splits * gap_per_fold

        if n_samples < min_required:
            raise ValueError(
                f"Insufficient data ({n_samples} bars) for CV configuration: "
                f"embargo={embargo_bars}, purge={purge_bars}, splits={n_splits}. "
                f"Need at least {min_required} bars."
            )

        self._log(
            f"  Data sufficiency check: {n_samples} bars >= {min_required} required "
            f"(embargo={embargo_bars}, purge={purge_bars}, splits={n_splits}) — OK"
        )

    def _resolve_barrier_params(self, horizon: int) -> tuple[float, float, int, str]:
        """Triple-barrier (k_up, k_down, max_bars, source) for one horizon.

        Delegates to ``ExperimentConfig.resolve_barrier_params`` — the single
        source of truth shared by labeling, the backtester and the derived CV
        purge (``resolve_cv_gaps``).
        """
        return self.config.resolve_barrier_params(horizon)

    def _pipeline_config(self, n_rows: int | None = None) -> Any:
        """PipelineConfig with the CV gaps resolved for this run's data.

        Pass ``n_rows`` (rows of the labeled frame) to resolve and log
        purge/embargo from the label span and the training bar timeframe;
        later calls reuse the resolved gaps.
        """
        if self._cv_gaps is None or n_rows is not None:
            self._cv_gaps = self.config.resolve_cv_gaps(self._bar_timeframe(), n_rows)
            self._log(
                f"  CV gaps: purge_bars={self._cv_gaps[0]} (longest label span "
                f"{self.config.label_span_bars()} bars), embargo_bars={self._cv_gaps[1]} "
                f"(bar timeframe {self._bar_timeframe()})"
            )
        return self.config.to_pipeline_config(
            cv_gaps=self._cv_gaps, bar_timeframe=self._bar_timeframe()
        )

    def _bar_timeframe(self) -> str | None:
        """Training bar timeframe recorded by the data pipeline (None before it ran)."""
        if self._feature_pipeline is not None:
            return self._feature_pipeline.get("bar_timeframe")
        return self.config.data.bar_timeframe

    def _load_raw_bars(self) -> tuple[pd.DataFrame, str]:
        """
        Load raw OHLCV as a sorted DatetimeIndex frame at the training bar timeframe.

        Applies ``data.start_date`` / ``data.end_date`` filtering and, when
        ``data.bar_timeframe`` is coarser than the source bars, resamples to it
        (same resample code the inference PreprocessingGraph uses).

        Returns:
            (raw OHLCV frame, bar timeframe string such as "5min")
        """
        from src.core.common.timeframes import detect_timeframe, get_timeframe_minutes
        from src.data.pipeline.stages.clean.utils import resample_ohlcv

        if not self.config.data.data_path:
            raise ValueError("data_path must be provided in config")

        data_path = Path(self.config.data.data_path)
        if data_path.suffix.lower() == ".csv":
            raw_df = pd.read_csv(data_path)
        else:
            raw_df = pd.read_parquet(data_path)
        self._log(f"  Loaded: {len(raw_df)} rows from {data_path}")

        # Normalize column names
        raw_df.columns = [str(c).lower().strip() for c in raw_df.columns]

        # Ensure datetime index
        if "datetime" in raw_df.columns:
            raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
            raw_df = raw_df.set_index("datetime").sort_index()
        elif "date" in raw_df.columns:
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df.set_index("date").sort_index()
        elif not isinstance(raw_df.index, pd.DatetimeIndex):
            raw_df.index = pd.to_datetime(raw_df.index)
            raw_df = raw_df.sort_index()
        else:
            raw_df = raw_df.sort_index()
        raw_df.index.name = "datetime"

        # Check for OHLCV columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        raw_df = raw_df[required]

        # Date range filtering
        if self.config.data.start_date:
            raw_df = raw_df[raw_df.index >= pd.Timestamp(self.config.data.start_date)]
        if self.config.data.end_date:
            raw_df = raw_df[raw_df.index <= pd.Timestamp(self.config.data.end_date)]
        if raw_df.empty:
            raise ValueError(
                f"No rows left after date filtering "
                f"(start={self.config.data.start_date}, end={self.config.data.end_date})"
            )

        source_tf = detect_timeframe(raw_df)
        if source_tf is None:
            raise ValueError(
                "Could not detect the bar timeframe of the input data (median bar "
                "spacing is not a whole number of minutes)."
            )

        bar_tf = self.config.data.bar_timeframe or source_tf
        if bar_tf != source_tf:
            if get_timeframe_minutes(bar_tf) < get_timeframe_minutes(source_tf):
                raise ValueError(
                    f"data.bar_timeframe={bar_tf} is finer than the {source_tf} input bars"
                )
            resampled = resample_ohlcv(raw_df.reset_index(), bar_tf, include_metadata=False)
            raw_df = resampled.set_index("datetime")
            self._log(f"  Resampled {source_tf} -> {bar_tf}: {len(raw_df)} bars")
        return raw_df, bar_tf

    def _run_data_pipeline(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame] | None]:
        """
        Run data pipeline to prepare features and labels.

        Returns:
            Tuple of (DataFrame with features/labels, additional_dfs for multi-stream or None)
        """
        self._log("Running data pipeline...")

        raw_df, bar_timeframe = self._load_raw_bars()

        # Generate additional_dfs for multi-stream models BEFORE feature engineering
        # (needs raw OHLCV with DatetimeIndex)
        additional_dfs = self._generate_additional_dfs(raw_df)

        # =====================================================================
        # STEP 1: Feature Engineering
        # =====================================================================
        self._log("  Generating features...")
        from src.data.pipeline.stages.features import FeatureEngineer

        # Reset index to make datetime a column (FeatureEngineer expects this)
        df_for_features = raw_df.reset_index()
        if df_for_features.columns[0] == "index":
            df_for_features = df_for_features.rename(columns={"index": "datetime"})

        mtf = self.config.data.mtf
        engineer = FeatureEngineer(
            output_dir=self.output_dir,
            timeframe=bar_timeframe,
            enable_mtf=mtf.enabled,
            mtf_timeframes=list(mtf.timeframes),
        )
        # Recorded so bundles replay the exact same transform at inference
        self._feature_pipeline = {
            "bar_timeframe": bar_timeframe,
            "engineer": engineer.to_spec(),
        }
        df_features, _report = engineer.engineer_features(
            df_for_features,
            symbol=self.config.data.symbol,
        )
        self._log(f"  Features: {len(df_features.columns)} columns")

        # =====================================================================
        # STEP 2: Triple Barrier Labeling
        # =====================================================================
        self._log("  Generating labels...")
        from src.core.label_spans import label_end_column, label_end_positions, remap_label_ends
        from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler

        # Create labels for each horizon. Barrier params come from a single
        # source of truth (_resolve_barrier_params) shared with the backtester,
        # so labels and backtest always play the same game.
        labeling = self.config.data.labeling
        symbol = self.config.data.symbol.upper()
        for horizon in self.config.training.horizons:
            k_up, k_down, max_bars, barrier_source = self._resolve_barrier_params(horizon)
            label_config = TripleBarrierConfig(
                horizon=max_bars,
                upper_mult=k_up,
                lower_mult=k_down,
                atr_period=labeling.atr_period,
                # ATR computed inline from OHLCV (Wilder, same as the backtest) — not
                # a feature column, whose period and lag follow feature engineering
                atr_column=None,
                symbol=symbol,
                # Price cost -> ATR units with the TRAINING split's median ATR
                # (the split prepare() makes); val/test volatility must not
                # shape the training labels. The backtest reuses this scalar.
                cost_calibration_fraction=self.config.data.splits.train_ratio,
            )
            labeler = TripleBarrierLabeler(label_config)
            label_result = labeler.compute_labels(df_features, horizon=max_bars)
            labels = pd.Series(label_result.labels, index=df_features.index, name="label")
            label_ends = label_end_positions(
                label_result.labels, label_result.metadata["bars_to_hit"]
            )
            cost_meta = label_result.metadata.get("cost_in_atr")
            self._label_cost_in_atr[horizon] = float(cost_meta[0]) if cost_meta is not None else 0.0
            df_features[f"label_h{horizon}"] = labels
            # Row at which the label resolves (-1 = invalid): drives CV purging
            # and sample-uniqueness weights downstream
            df_features[label_end_column(f"label_h{horizon}")] = label_ends
            self._log(
                f"    label_h{horizon}: k_up={k_up} k_down={k_down} "
                f"max_bars={max_bars} [{barrier_source}] "
                f"{labels.value_counts().to_dict()}"
            )

        # Also create a default 'label' column using first horizon
        first_horizon = self.config.training.horizons[0]
        df_features["label"] = df_features[f"label_h{first_horizon}"]
        df_features[label_end_column("label")] = df_features[
            label_end_column(f"label_h{first_horizon}")
        ]

        # Binary mode: remap {-1, 0, 1} -> {0, 1} (significant move vs no move)
        if self.config.data.labeling.binary_mode:
            # Binary labels: 1 = barrier hit either way (a move), 0 = time-out.
            # They carry no direction, so the directional backtest is skipped.
            label_remap = {-1: 1, 0: 0, 1: 1, -99: -99}
            label_cols = [f"label_h{h}" for h in self.config.training.horizons] + ["label"]
            for col in label_cols:
                if col in df_features.columns:
                    df_features[col] = df_features[col].map(label_remap)
            self._log("  Binary mode: remapped labels {-1,0,1} -> {0,1}")

        # Drop rows with NaN labels (label-end positions are re-mapped to the
        # surviving rows so every span still covers exactly the same bars)
        initial_len = len(df_features)
        kept = df_features["label"].notna().to_numpy()
        if not kept.all():
            df_features = df_features.loc[kept]
            kept_rows = np.flatnonzero(kept)
            label_cols = [f"label_h{h}" for h in self.config.training.horizons] + ["label"]
            for col in map(label_end_column, label_cols):
                df_features[col] = remap_label_ends(df_features[col].to_numpy(), kept_rows)
            self._log(f"  Dropped {initial_len - len(df_features)} rows with NaN labels")

        self._log(
            f"  Pipeline complete: {len(df_features)} rows, {len(df_features.columns)} columns"
        )
        self._log(f"  Label distribution: {df_features['label'].value_counts().to_dict()}")

        # Restore DatetimeIndex for multi-stream adapter compatibility
        # (FeatureEngineer needs datetime as column, MultiStreamAdapter needs DatetimeIndex)
        if "datetime" in df_features.columns and not isinstance(
            df_features.index, pd.DatetimeIndex
        ):
            df_features = df_features.set_index("datetime")
            if not df_features.index.is_monotonic_increasing:
                # Label-end columns are row positions: reordering would break them
                raise ValueError("Feature frame is not in chronological order after labeling")

        # Downcast float64 → float32 at the source to halve the DataFrame memory
        # that stays alive for the entire training + evaluation run.
        # (949K × 227 cols: 1.72 GB float64 → 0.86 GB float32)
        float64_cols = df_features.select_dtypes(include=["float64"]).columns
        if len(float64_cols) > 0:
            df_features = df_features.astype(dict.fromkeys(float64_cols, np.float32))

        # Also downcast additional_dfs (multi-stream resampled OHLCV) — they stay
        # alive until multi-stream adapter casts them, so halve early.
        if additional_dfs:
            for tf_key, tf_df in additional_dfs.items():
                f64 = tf_df.select_dtypes(include=["float64"]).columns
                if len(f64) > 0:
                    additional_dfs[tf_key] = tf_df.astype(dict.fromkeys(f64, np.float32))

        return df_features, additional_dfs

    def _run_training(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> TrainingRunResult:
        """
        Train models using UnifiedTrainingOrchestrator.

        Args:
            df: Prepared DataFrame with features and labels
            additional_dfs: Resampled OHLCV DataFrames for multi-stream models

        Returns:
            TrainingRunResult from orchestrator
        """
        from src.models.training import UnifiedTrainingOrchestrator

        self._log(f"Training {len(self.config.training.models)} models...")
        self._log(f"  Models: {self.config.training.models}")
        self._log(f"  Mode: {self.config.training.training_mode}")
        if additional_dfs:
            self._log(f"  Multi-stream timeframes: {list(additional_dfs.keys())}")

        pipeline_config = self._pipeline_config()
        orchestrator = UnifiedTrainingOrchestrator(pipeline_config)
        result = orchestrator.train(df, additional_dfs=additional_dfs)

        self._log(f"  Trained: {result.n_models} models")
        self._log(f"  Best: {result.best_model}")

        return result

    def _run_evaluation(
        self, df: pd.DataFrame, training_result: TrainingRunResult
    ) -> dict[str, Any]:
        """
        Run evaluation (backtesting, metrics computation).

        Backtests the deployed strategy of the first horizon with that
        horizon's barriers (see ``_extract_predictions``); the metrics carry
        ``strategy`` and ``horizon`` so the number is never mistaken for
        another model's or horizon's.

        Args:
            df: Raw OHLCV data
            training_result: Result from training phase

        Returns:
            Dictionary of backtest metrics
        """
        if not self.config.evaluation.run_backtest:
            self._log("  Backtest disabled, skipping")
            return {}
        if self.config.data.labeling.binary_mode:
            # Binary labels are {0: no move, 1: move}; they carry no direction,
            # so there is no long/short signal to backtest.
            logger.warning("Binary mode: directional backtest skipped (labels carry no direction)")
            return {}

        try:
            from src.inference.backtesting import BacktestConfig, Backtester

            # Out-of-sample signals of the deployed strategy (primary horizon)
            predictions_df, strategy = self._extract_predictions(df, training_result)
            if predictions_df is None or len(predictions_df) == 0:
                self._log("  No predictions available for backtest")
                return {}

            # Backtester merges on "timestamp" column — rename "datetime" if needed
            if "datetime" in predictions_df.columns and "timestamp" not in predictions_df.columns:
                predictions_df = predictions_df.rename(columns={"datetime": "timestamp"})

            # Pass OHLCV-only prices. The featured df also carries 'label'
            # columns; the Backtester injects its own 'label' into predictions
            # and its merge would rename the colliding columns to
            # label_pred/label_price, crashing run() at data['label'] — which
            # the except below then silently swallowed (backtest_metrics was
            # always {}). Regression test: tests/test_factory_e2e.py.
            ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            prices_df = df[ohlcv_cols].copy()
            if "datetime" in df.columns:
                prices_df["timestamp"] = df["datetime"].values
            elif "timestamp" in df.columns:
                prices_df["timestamp"] = df["timestamp"].values
            elif isinstance(df.index, pd.DatetimeIndex):
                prices_df["timestamp"] = df.index

            # Map evaluation.position_sizing short names to the Backtester's
            # PositionSizingMethod values.
            position_sizing_map = {
                "fixed": "fixed_contracts",
                "volatility": "volatility_targeted",
                "confidence": "bet_sizing",
                # "kelly" stays as "kelly"
            }
            canonical_sizing = self.config.evaluation.position_sizing
            local_sizing = position_sizing_map.get(canonical_sizing, canonical_sizing)

            # Select contract specs based on symbol
            symbol = self.config.data.symbol.upper()
            sym_config = SymbolConfig.from_symbol(symbol)

            # Build kwargs with optional transaction cost overrides from config
            bt_kwargs: dict[str, Any] = {
                "position_sizing": local_sizing,
                "initial_equity": self.config.evaluation.initial_equity,
            }
            if self.config.evaluation.commission_per_contract is not None:
                bt_kwargs["commission_per_contract"] = (
                    self.config.evaluation.commission_per_contract
                )
            if self.config.evaluation.slippage_ticks is not None:
                bt_kwargs["slippage_ticks"] = self.config.evaluation.slippage_ticks

            # Wire triple-barrier params from the SAME resolution the labeler
            # used (Phase 86 parity guarantee: labels and backtest play the
            # same game — see _resolve_barrier_params). The signals are the
            # first horizon's, so are the barriers.
            first_horizon = self.config.training.horizons[0]
            k_up, k_down, max_bars, _src = self._resolve_barrier_params(first_horizon)
            bt_kwargs["barrier_k_up"] = k_up
            bt_kwargs["barrier_k_down"] = k_down
            bt_kwargs["max_holding_period"] = max_bars
            # Same cost term the labeler added (persisted with the data
            # checkpoint; None only for checkpoints that predate it — the
            # backtester then derives it with the labeler's own helper)
            bt_kwargs["barrier_cost_in_atr"] = self._label_cost_in_atr.get(first_horizon)

            backtest_config = BacktestConfig.from_symbol_config(
                sym_config,
                **bt_kwargs,
            )
            backtester = Backtester(
                predictions=predictions_df, prices=prices_df, config=backtest_config
            )

            # Run backtest
            bt_result = backtester.run()
            metrics: dict[str, Any] = {
                **bt_result.summary(),
                "strategy": strategy,
                "horizon": first_horizon,
            }

            # Store equity curve and trades for notebook visualizations
            self._last_equity_curve = bt_result.equity_curve
            self._last_backtest_trades = list(bt_result.trades)

            self._log(
                f"  Backtest ({strategy}, h{first_horizon}) complete: "
                f"{metrics.get('total_trades', 0)} trades"
            )
            self._log(f"  Win rate: {metrics.get('win_rate_pct', 0):.1f}%")
            self._log(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

            return metrics

        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {}

    def _create_bundle(self, training_result: TrainingRunResult) -> Path | None:
        """
        Create deployment bundle with trained models and artifacts.

        Args:
            training_result: Result from training phase

        Returns:
            Path to bundle directory, or None if bundling disabled
        """
        if not self.config.bundling.create_bundle:
            self._log("  Bundling disabled, skipping")
            return None

        try:
            from src.inference.builder import BundleBuilder

            pipeline_config = self._pipeline_config()
            builder = BundleBuilder(pipeline_config, feature_pipeline=self._feature_pipeline)

            bundle_result = builder.build_from_training_result(training_result)
            ensemble_path = builder.build_ensemble_from_training_result(
                training_result, base_bundles=bundle_result.bundle_paths
            )
            bundle_path = self.output_dir / "bundles"
            bundle_path.mkdir(exist_ok=True)

            self._log(f"  Created {bundle_result.n_bundles} bundles")
            if ensemble_path is not None:
                self._log(f"  Created ensemble bundle: {ensemble_path.name}")
            self._log(f"  Bundle path: {bundle_path}")

            return bundle_path

        except Exception as e:
            logger.warning(f"Bundling failed: {e}")
            return None

    def _create_deploy(
        self, training_result: TrainingRunResult, bundle_path: Path | None
    ) -> Path | None:
        """
        Create deploy artifact directory with manifest.

        Scans the bundles directory for all saved bundles and produces
        a deploy/manifest.json indexing them by horizon.

        Args:
            training_result: Result from training phase.
            bundle_path: Path to bundles directory (from _create_bundle).

        Returns:
            Path to deploy directory, or None if disabled or no bundles.
        """
        if not getattr(self.config.bundling, "deploy_artifact", True):
            self._log("  Deploy artifact disabled, skipping")
            return None

        if bundle_path is None or not bundle_path.exists():
            return None

        try:
            from datetime import datetime as dt

            from src.inference.deploy import (
                DEPLOY_MANIFEST_FILE,
                DeployManifest,
                HorizonArtifactEntry,
                HorizonManifest,
                describe_bundle,
            )

            deploy_dir = self.output_dir / "deploy"
            deploy_dir.mkdir(parents=True, exist_ok=True)

            # Scan bundles directory for saved model bundles
            horizons: dict[int, HorizonManifest] = {}

            best_score: dict[int, float] = {}
            for item in sorted(bundle_path.iterdir()):
                info = describe_bundle(item) if item.is_dir() else None
                if info is None:
                    continue

                # Build relative path from deploy dir to bundle
                try:
                    rel_path = str(item.relative_to(deploy_dir))
                except ValueError:
                    rel_path = str(item.relative_to(self.output_dir))

                entry = HorizonArtifactEntry(
                    model_name=info.model_name,
                    bundle_path=rel_path,
                    is_ensemble=info.kind == "ensemble",
                    metrics=info.metrics,
                )
                h_manifest = horizons.setdefault(
                    info.horizon, HorizonManifest(horizon=info.horizon)
                )
                h_manifest.entries.append(entry)

                # Primary: the ensemble when one exists, else best validation score
                current = h_manifest.primary_model
                current_is_ensemble = any(
                    e.is_ensemble for e in h_manifest.entries if e.model_name == current
                )
                if entry.is_ensemble or (
                    not current_is_ensemble
                    and (not current or info.score > best_score.get(info.horizon, -1.0))
                ):
                    h_manifest.primary_model = info.model_name
                    best_score[info.horizon] = info.score

            if not horizons:
                self._log("  No bundles found for deploy manifest")
                return None

            manifest = DeployManifest(
                created_at=dt.now().isoformat(),
                symbol=self.config.data.symbol,
                horizons=horizons,
            )
            manifest.save(deploy_dir / DEPLOY_MANIFEST_FILE)

            self._log(
                f"  Deploy artifact: {len(horizons)} horizons, "
                f"{sum(len(h.entries) for h in horizons.values())} bundles"
            )
            return deploy_dir

        except Exception as e:
            logger.warning(f"Deploy artifact creation failed: {e}")
            return None

    def _extract_predictions(
        self, df: pd.DataFrame, training_result: TrainingRunResult
    ) -> tuple[pd.DataFrame | None, str]:
        """
        Out-of-sample signals of the DEPLOYED strategy at the primary horizon.

        The backtest replays what ships for ``horizons[0]`` (the horizon whose
        barriers it uses):

        - Stacking ensemble (deployed whenever a meta-learner was trained):
          the meta-learner's predictions on its purged holdout — built from
          out-of-fold base predictions and never seen by the evaluation fit
          (strategy ``"stacking_holdout"``).
        - Otherwise the horizon's primary model (manifest rule,
          ``TrainingRunResult.best_model``) and its out-of-fold predictions
          (strategy ``"oof:<model key>"``).

        Args:
            df: Labeled feature frame (DatetimeIndex)
            training_result: TrainingRunResult

        Returns:
            (DataFrame with datetime, prediction {-1, 0, 1}, confidence — or
            None when there is nothing to backtest, strategy label)
        """
        ensemble = training_result.ensemble_result
        if ensemble is not None and ensemble.trainer is not None:
            holdout = ensemble.metadata.get("holdout_predictions")
            if holdout is not None and len(holdout) > 0:
                rows = holdout["row"].to_numpy(dtype=np.int64)
                return (
                    pd.DataFrame(
                        {
                            "datetime": df.index[rows].values,
                            "prediction": holdout["prediction"].to_numpy(dtype=np.int64),
                            "confidence": holdout["confidence"].to_numpy(dtype=float),
                        }
                    ),
                    "stacking_holdout",
                )
            logger.warning("Ensemble has no holdout predictions; backtesting the best single model")

        best_model = training_result.best_model
        model_result = training_result.model_results.get(best_model) if best_model else None
        oof = model_result.oof_prediction if model_result is not None else None
        if oof is None:
            return None, "none"
        # Accessors return FULL-LENGTH arrays (NaN where no prediction);
        # original_indices marks the valid rows.
        preds = oof.get_class_predictions()  # 1D array of {-1, 0, 1}
        probs = oof.get_probabilities()  # (n_samples, n_classes)
        indices = oof.original_indices
        if indices is None:
            # Legacy producers leave original_indices unset — derive the valid
            # rows from non-NaN predictions so NaN gaps don't become fabricated
            # neutral signals.
            indices = np.where(~np.isnan(preds))[0]
        return (
            pd.DataFrame(
                {
                    "datetime": df.index[indices].values,
                    "prediction": np.nan_to_num(preds[indices], nan=0.0).astype(int),
                    # Subset BEFORE the row-max: covered rows only
                    "confidence": probs[indices].max(axis=1),
                }
            ),
            f"oof:{best_model}",
        )

    def _extract_ensemble_metrics(self, training_result: TrainingRunResult) -> dict[str, float]:
        """
        Extract ensemble metrics from training result.

        Args:
            training_result: TrainingRunResult

        Returns:
            Dictionary of ensemble metrics
        """
        if hasattr(training_result, "ensemble_result") and training_result.ensemble_result:
            return training_result.ensemble_result.metrics
        return {}

    def _log(self, message: str) -> None:
        """Log message based on verbosity setting."""
        if self.verbose >= 1:
            print(message)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MLFactory",
    "ExperimentResult",
]
