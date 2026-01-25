"""
MLFactory - Unified Entry Point for ML Factory Operations.

This is THE single entry point for the ML Factory system, coordinating:
- Data Pipeline (via PipelineRunner)
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

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.experiment import ExperimentConfig
from src.core.checkpoint import PipelineCheckpointManager, compute_config_hash

logger = logging.getLogger(__name__)


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
        backtest_metrics: Backtest results (if run)
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
    backtest_metrics: dict[str, float] = field(default_factory=dict)
    bundle_path: Path | None = None
    output_dir: Path | None = None
    error_message: str | None = None

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
    - PipelineRunner: Data pipeline execution
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
        verbose: int = 1,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize MLFactory with experiment configuration.

        Args:
            config: ExperimentConfig defining the experiment
            verbose: Logging verbosity (0=silent, 1=info, 2=debug)
            enable_checkpoints: Whether to save checkpoints after each stage
                (default: True). Enables resume_from_checkpoint() on failure.
        """
        self.config = config
        self.verbose = verbose
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
        self._cached_training_result: Any = None

        # Save config
        config_path = self.output_dir / "experiment_config.yaml"
        config.save_yaml(config_path)

        if verbose >= 1:
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
                df = self._run_data_pipeline()
                self._save_checkpoint_data_pipeline(df)
            else:
                self._log("\n[Phase 1/4] Data Pipeline (cached)")
                df = self._load_cached_data()

            # Phase 2: Training
            if resume_from_stage <= self.STAGE_TRAINING:
                self._log("\n[Phase 2/4] Model Training")
                training_result = self._run_training(df)
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
                output_dir=self.output_dir,
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

    def _save_checkpoint_data_pipeline(self, df: pd.DataFrame) -> None:
        """Save checkpoint after data pipeline stage."""
        if not self._checkpoint_manager:
            return

        # Save DataFrame to cache
        data_cache_path = self.output_dir / "cache" / "data_pipeline.parquet"
        data_cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_cache_path)

        self._checkpoint_manager.save_checkpoint(
            stage_name="data_pipeline",
            stage_index=self.STAGE_DATA_PIPELINE,
            artifacts={"data": data_cache_path},
            config_hash=self._config_hash,
            n_rows=len(df),
            n_columns=len(df.columns),
        )

    def _save_checkpoint_training(self, training_result: Any) -> None:
        """Save checkpoint after training stage."""
        if not self._checkpoint_manager:
            return

        import pickle

        # Save training result
        training_cache_path = self.output_dir / "cache" / "training_result.pkl"
        training_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(training_cache_path, "wb") as f:
            pickle.dump(training_result, f)

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

        import json

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

    def _load_cached_training(self) -> Any:
        """Load cached training result from checkpoint."""
        import pickle

        training_cache_path = self.output_dir / "cache" / "training_result.pkl"
        if training_cache_path.exists():
            with open(training_cache_path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(f"Cached training result not found at {training_cache_path}")

    def _load_cached_evaluation(self) -> dict:
        """Load cached evaluation metrics from checkpoint."""
        import json

        eval_cache_path = self.output_dir / "cache" / "evaluation.json"
        if eval_cache_path.exists():
            with open(eval_cache_path) as f:
                return json.load(f)
        return {}

    def _run_data_pipeline(self) -> pd.DataFrame:
        """
        Run data pipeline to prepare features and labels.

        Returns:
            DataFrame with features, labels, and metadata
        """
        self._log("Loading raw data...")

        # If data path provided, load it
        if self.config.data.data_path:
            df = pd.read_parquet(self.config.data.data_path)
            self._log(f"  Loaded: {len(df)} rows from {self.config.data.data_path}")
            return df

        # Otherwise, run full pipeline
        from src.data.pipeline import PipelineRunner

        pipeline_config = self.config.to_pipeline_config()
        runner = PipelineRunner(pipeline_config)

        self._log("  Running pipeline stages...")
        runner.run()

        # Load the processed data
        # Note: This assumes the pipeline saves to a canonical location
        # You may need to adjust based on actual pipeline output
        processed_path = pipeline_config.run_canonical_dir / "processed.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(
                f"Pipeline output not found at {processed_path}. "
                "Ensure PipelineRunner saves processed data."
            )

        df = pd.read_parquet(processed_path)
        self._log(f"  Pipeline complete: {len(df)} rows")
        return df

    def _run_training(self, df: pd.DataFrame) -> Any:
        """
        Train models using UnifiedTrainingOrchestrator.

        Args:
            df: Prepared DataFrame with features and labels

        Returns:
            TrainingRunResult from orchestrator
        """
        from src.models.training import UnifiedTrainingOrchestrator

        self._log(f"Training {len(self.config.training.models)} models...")
        self._log(f"  Models: {self.config.training.models}")
        self._log(f"  Mode: {self.config.training.training_mode}")

        pipeline_config = self.config.to_pipeline_config()
        orchestrator = UnifiedTrainingOrchestrator(pipeline_config)
        result = orchestrator.train(df)

        self._log(f"  Trained: {result.n_models} models")
        self._log(f"  Best: {result.best_model}")

        return result

    def _run_evaluation(self, df: pd.DataFrame, training_result: Any) -> dict[str, float]:
        """
        Run evaluation (backtesting, metrics computation).

        Args:
            df: Raw OHLCV data
            training_result: Result from training phase

        Returns:
            Dictionary of backtest metrics
        """
        if not self.config.evaluation.run_backtest:
            self._log("  Backtest disabled, skipping")
            return {}

        try:
            from src.inference.backtesting import Backtester

            # Extract predictions from training result
            predictions_df = self._extract_predictions(df, training_result)
            if predictions_df is None or len(predictions_df) == 0:
                self._log("  No predictions available for backtest")
                return {}

            # Configure backtester
            backtest_config = self.config.to_backtest_config()
            backtester = Backtester(predictions=predictions_df, prices=df, config=backtest_config)

            # Run backtest
            bt_result = backtester.run()
            metrics = bt_result.summary()

            self._log(f"  Backtest complete: {metrics.get('total_trades', 0)} trades")
            self._log(f"  Win rate: {metrics.get('win_rate_pct', 0):.1f}%")
            self._log(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

            return metrics

        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return {}

    def _create_bundle(self, training_result: Any) -> Path | None:
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

            pipeline_config = self.config.to_pipeline_config()
            builder = BundleBuilder(pipeline_config)

            bundle_result = builder.build_from_training_result(training_result)
            bundle_path = self.output_dir / "bundles"
            bundle_path.mkdir(exist_ok=True)

            self._log(f"  Created {bundle_result.n_bundles} bundles")
            self._log(f"  Bundle path: {bundle_path}")

            return bundle_path

        except Exception as e:
            logger.warning(f"Bundling failed: {e}")
            return None

    def _extract_predictions(self, df: pd.DataFrame, training_result: Any) -> pd.DataFrame | None:
        """
        Extract predictions from training result for backtesting.

        Args:
            df: Raw OHLCV data
            training_result: TrainingRunResult

        Returns:
            DataFrame with datetime, prediction, confidence columns
        """
        # Try aligned OOF from ensemble
        if hasattr(training_result, "aligned_oof") and training_result.aligned_oof is not None:
            oof = training_result.aligned_oof
            return pd.DataFrame(
                {
                    "datetime": df["datetime"].iloc[oof.common_indices],
                    "prediction": oof.ensemble_predictions,
                    "confidence": (
                        oof.ensemble_probabilities.max(axis=1)
                        if oof.ensemble_probabilities is not None
                        else None
                    ),
                }
            )

        # Try OOF from best model
        best_model = training_result.best_model
        if best_model and hasattr(training_result, "model_results"):
            model_result = training_result.model_results.get(best_model)
            if model_result and hasattr(model_result, "oof_prediction"):
                oof = model_result.oof_prediction
                if oof is not None:
                    return pd.DataFrame(
                        {
                            "datetime": df["datetime"].iloc[oof.indices],
                            "prediction": oof.predictions,
                            "confidence": (
                                oof.probabilities.max(axis=1)
                                if hasattr(oof, "probabilities") and oof.probabilities is not None
                                else None
                            ),
                        }
                    )

        return None

    def _extract_ensemble_metrics(self, training_result: Any) -> dict[str, float]:
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
