"""
MLPipeline - THE orchestrator for the ML pipeline.

This is the top-level orchestrator that delegates to specialized components:

ARCHITECTURE HIERARCHY:
    MLPipeline (this file)
    └── Phase 1-4: Data loading (raw OHLCV from data_path)
    └── Phase 5-6: UnifiedTrainingOrchestrator (training + ensemble)
         └── UnifiedDataPreparation (features, labels, splits)
         └── Model trainers (XGBoost, LSTM, etc.)
         └── Ensemble builder
    └── Phase 7: Evaluation (metrics from TrainingRunResult)
    └── Phase 8: Backtester (strategy simulation)
    └── Phase 9: BundleBuilder (inference artifacts)

DELEGATED WORK:
    - Features/Labels/Splits: Handled internally by UnifiedTrainingOrchestrator
      via UnifiedDataPreparation adapters. MLPipeline just loads raw OHLCV.
    - Training + Ensemble: UnifiedTrainingOrchestrator.train(df) handles both.
    - Backtesting: Delegates to src.backtesting.Backtester
    - Bundling: Delegates to src.inference.BundleBuilder

Usage:
    from src import MLPipeline, PipelineConfig

    config = PipelineConfig(symbol="MES", models=["xgboost", "lstm"])
    result = MLPipeline(config).run()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import pandas as pd

from src.pipeline_config import PipelineConfig

if TYPE_CHECKING:
    from src.training.unified_orchestrator import TrainingRunResult, UnifiedTrainingOrchestrator
    from src.inference.backtesting.backtest import Backtester, BacktestConfig, BacktestResult
    from src.inference.builder import BundleBuilder, BundleBuildResult

logger = logging.getLogger(__name__)


class PipelinePhase(Enum):
    """Pipeline phases."""
    DATA_PREP = "data_prep"
    FEATURES = "features"
    LABELING = "labeling"
    SPLITS = "splits"
    TRAINING = "training"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    BACKTEST = "backtest"
    BUNDLING = "bundling"


@dataclass
class PhaseResult:
    """Result from a single phase."""
    phase: PipelinePhase
    success: bool
    duration_seconds: float
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineResult:
    """Result from the complete pipeline."""
    run_id: str
    config: PipelineConfig
    success: bool
    phases: dict[str, PhaseResult] = field(default_factory=dict)
    duration_seconds: float = 0.0
    best_model: str | None = None
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    backtest_metrics: dict[str, float] = field(default_factory=dict)
    models_dir: Path | None = None
    bundle_path: Path | None = None
    # Store the training result for downstream use
    training_result: Any = None

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Pipeline Result: {status}",
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Phases:",
        ]
        for name, result in self.phases.items():
            s = "OK" if result.success else "FAIL"
            lines.append(f"  {name}: {s} ({result.duration_seconds:.1f}s)")

        if self.best_model:
            lines.append(f"\nBest Model: {self.best_model}")

        if self.ensemble_metrics:
            lines.append("\nEnsemble Metrics:")
            for k, v in self.ensemble_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        if self.backtest_metrics:
            lines.append("\nBacktest Metrics:")
            for k, v in self.backtest_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        return "\n".join(lines)


class MLPipeline:
    """
    THE orchestrator for the ML pipeline.

    This is the high-level entry point that coordinates all pipeline phases.
    It delegates heavy lifting to specialized components:

    - Phases 1-4 (data): Load raw OHLCV data. Features/labels/splits are
      handled internally by UnifiedTrainingOrchestrator via adapters.
    - Phases 5-6 (training): Delegates to UnifiedTrainingOrchestrator.train(df)
      which handles both training AND ensemble building.
    - Phase 7 (evaluation): Extracts metrics from TrainingRunResult.
    - Phase 8 (backtest): Delegates to Backtester for strategy simulation.
    - Phase 9 (bundling): Delegates to BundleBuilder for inference artifacts.

    Example:
        from src import MLPipeline, PipelineConfig

        config = PipelineConfig(symbol="MES")
        result = MLPipeline(config).run()

    Advanced usage (pre-loaded data):
        df = pd.read_parquet("my_data.parquet")
        result = MLPipeline(config, df=df).run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        df: pd.DataFrame | None = None,
        verbose: int = 1,
    ):
        self.config = config
        self._df = df
        self._verbose = verbose
        self._completed_phases: set[PipelinePhase] = set()
        self._phase_results: dict[PipelinePhase, PhaseResult] = {}
        self._artifacts: dict[str, Any] = {}

        # Create run directory
        self._run_dir = config.get_run_dir()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        config.save(self._run_dir / "config.json")

        if verbose >= 1:
            logger.info(f"MLPipeline initialized: {config.run_id}")

    def run(self) -> PipelineResult:
        """Run the complete pipeline."""
        start_time = datetime.now()
        self._log(f"Starting pipeline: {self.config.run_id}")

        # Run all phases
        self.run_data()
        self.run_train()

        if self.config.run_evaluation:
            self.run_evaluate()

        if self.config.run_backtest:
            self._run_backtest_phase()

        if self.config.save_bundle:
            self.run_bundle()

        duration = (datetime.now() - start_time).total_seconds()

        result = PipelineResult(
            run_id=self.config.run_id,
            config=self.config,
            success=True,
            phases={p.value: r for p, r in self._phase_results.items()},
            duration_seconds=duration,
            best_model=self._artifacts.get("best_model"),
            ensemble_metrics=self._artifacts.get("ensemble_metrics", {}),
            backtest_metrics=self._artifacts.get("backtest_metrics", {}),
            models_dir=self._run_dir / "models" if (self._run_dir / "models").exists() else None,
            bundle_path=self._artifacts.get("bundle_path"),
            training_result=self._artifacts.get("training_result"),
        )

        self._save_result(result)
        self._log(f"Pipeline completed in {duration:.1f}s")
        return result

    def run_data(self) -> None:
        """Run data phases (1-4).

        NOTE: Phases 2-4 (features, labeling, splits) are handled internally
        by UnifiedTrainingOrchestrator via UnifiedDataPreparation adapters.
        MLPipeline just loads raw OHLCV data in phase 1.
        """
        self._run_phase(PipelinePhase.DATA_PREP, self._phase_data_prep)
        # Features, labeling, and splits are handled internally by
        # UnifiedTrainingOrchestrator, but we run these phases for tracking
        self._run_phase(PipelinePhase.FEATURES, self._phase_features)
        self._run_phase(PipelinePhase.LABELING, self._phase_labeling)
        self._run_phase(PipelinePhase.SPLITS, self._phase_splits)

    def run_train(self) -> None:
        """Run training phases (5-6).

        Delegates to UnifiedTrainingOrchestrator.train() which handles
        both training AND ensemble building.
        """
        self._run_phase(PipelinePhase.TRAINING, self._phase_training)
        if self.config.build_ensemble:
            self._run_phase(PipelinePhase.ENSEMBLE, self._phase_ensemble)

    def run_evaluate(self) -> None:
        """Run evaluation phase (7)."""
        self._run_phase(PipelinePhase.EVALUATION, self._phase_evaluation)

    def run_bundle(self) -> None:
        """Run bundling phase (9)."""
        self._run_phase(PipelinePhase.BUNDLING, self._phase_bundling)

    def _run_backtest_phase(self) -> None:
        """Run backtest phase (8)."""
        self._run_phase(PipelinePhase.BACKTEST, self._phase_backtest)

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    def _phase_data_prep(self) -> PhaseResult:
        """Phase 1: Data preparation - load raw OHLCV data."""
        self._log("Phase 1: Data Preparation")

        if self._df is None:
            self._log(f"  Loading: {self.config.data_path}")
            self._df = pd.read_parquet(self.config.data_path)

        self._log(f"  Rows: {len(self._df)}")
        self._artifacts["df"] = self._df

        return PhaseResult(
            phase=PipelinePhase.DATA_PREP,
            success=True,
            duration_seconds=0,
            metrics={"rows": len(self._df)},
        )

    def _phase_features(self) -> PhaseResult:
        """Phase 2: Feature engineering.

        NOTE: Actual feature engineering is handled internally by
        UnifiedTrainingOrchestrator via UnifiedDataPreparation adapters.
        This phase just tracks that we're ready for training.
        """
        self._log("Phase 2: Feature Engineering")
        self._log("  (Delegated to UnifiedTrainingOrchestrator)")

        return PhaseResult(
            phase=PipelinePhase.FEATURES,
            success=True,
            duration_seconds=0,
            metrics={"note": "delegated_to_training_orchestrator"},
        )

    def _phase_labeling(self) -> PhaseResult:
        """Phase 3: Labeling.

        NOTE: Actual labeling is handled internally by
        UnifiedTrainingOrchestrator via UnifiedDataPreparation adapters.
        This phase just tracks that we're ready for training.
        """
        self._log("Phase 3: Labeling")
        self._log("  (Delegated to UnifiedTrainingOrchestrator)")

        return PhaseResult(
            phase=PipelinePhase.LABELING,
            success=True,
            duration_seconds=0,
            metrics={"method": self.config.labeling_method},
        )

    def _phase_splits(self) -> PhaseResult:
        """Phase 4: Data splitting.

        NOTE: Actual splitting is handled internally by
        UnifiedTrainingOrchestrator via UnifiedDataPreparation adapters.
        This phase just tracks that we're ready for training.
        """
        self._log("Phase 4: Splits")
        self._log("  (Delegated to UnifiedTrainingOrchestrator)")

        return PhaseResult(
            phase=PipelinePhase.SPLITS,
            success=True,
            duration_seconds=0,
            metrics={
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
            },
        )

    def _phase_training(self) -> PhaseResult:
        """Phase 5: Model training.

        Delegates to UnifiedTrainingOrchestrator.train() which:
        - Prepares data via UnifiedDataPreparation adapters
        - Trains all specified models
        - Generates OOF predictions
        - Optionally builds ensemble (phase 6)
        """
        self._log("Phase 5: Training")
        self._log(f"  Models: {self.config.models}")
        self._log(f"  Mode: {self.config.training_mode}")

        # Import here to avoid circular imports
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        # Get raw OHLCV DataFrame
        df = self._artifacts.get("df", self._df)
        if df is None:
            raise ValueError("No data available for training. Run data phases first.")

        # Create orchestrator with our config
        # Note: We need to adapt PipelineConfig to the format expected by UnifiedTrainingOrchestrator
        # The UnifiedTrainingOrchestrator expects a PipelineConfig from src.core
        try:
            from src.core import PipelineConfig as CorePipelineConfig
            # Convert our config to core config format
            core_config = CorePipelineConfig(
                symbol=self.config.symbol,
                data_path=str(self.config.data_path),
                output_dir=self._run_dir,
                models=self.config.models,
                horizons=self.config.horizons,
                build_ensemble=self.config.build_ensemble,
                ensemble_method=self.config.ensemble_method,
                training_mode=self.config.training_mode,
                cv_method=self.config.cv_method,
                n_splits=self.config.n_splits,
                purge_bars=self.config.purge_bars,
                embargo_bars=self.config.embargo_bars,
                random_seed=self.config.random_seed,
                save_oof=self.config.save_oof,
            )
            orchestrator = UnifiedTrainingOrchestrator(core_config)
        except ImportError:
            # If src.core doesn't exist, try using our config directly
            self._log("  Warning: src.core not available, using local config")
            orchestrator = UnifiedTrainingOrchestrator(self.config)

        # Run training (this handles features, labels, splits internally)
        training_result = orchestrator.train(df)

        # Store result for downstream phases
        self._artifacts["training_result"] = training_result
        self._artifacts["best_model"] = training_result.best_model

        return PhaseResult(
            phase=PipelinePhase.TRAINING,
            success=True,
            duration_seconds=training_result.total_time_seconds,
            metrics={
                "n_models": training_result.n_models,
                "best_model": training_result.best_model,
                **training_result.get_metrics_summary(),
            },
        )

    def _phase_ensemble(self) -> PhaseResult:
        """Phase 6: Ensemble building.

        NOTE: Ensemble is built as part of UnifiedTrainingOrchestrator.train()
        if config.build_ensemble=True. This phase extracts ensemble metrics.
        """
        self._log("Phase 6: Ensemble")
        self._log(f"  Method: {self.config.ensemble_method}")

        training_result = self._artifacts.get("training_result")

        if training_result is None:
            return PhaseResult(
                phase=PipelinePhase.ENSEMBLE,
                success=False,
                duration_seconds=0,
                error="No training result available",
            )

        # Extract ensemble metrics from training result
        ensemble_metrics = {}
        if training_result.ensemble_result is not None:
            ensemble_metrics = training_result.ensemble_result.metrics
            self._artifacts["ensemble_metrics"] = ensemble_metrics
            self._log(f"  Ensemble trained successfully")
        else:
            self._log("  No ensemble built (single model or disabled)")

        return PhaseResult(
            phase=PipelinePhase.ENSEMBLE,
            success=True,
            duration_seconds=0,
            metrics=ensemble_metrics,
        )

    def _phase_evaluation(self) -> PhaseResult:
        """Phase 7: Evaluation.

        Extracts and summarizes metrics from TrainingRunResult.
        """
        self._log("Phase 7: Evaluation")

        training_result = self._artifacts.get("training_result")

        if training_result is None:
            return PhaseResult(
                phase=PipelinePhase.EVALUATION,
                success=False,
                duration_seconds=0,
                error="No training result available",
            )

        # Collect all metrics
        eval_metrics = {}

        # Model metrics
        for model_key, model_result in training_result.model_results.items():
            for metric_name, value in model_result.metrics.items():
                eval_metrics[f"{model_key}_{metric_name}"] = value

        # Ensemble metrics
        if training_result.ensemble_result is not None:
            for metric_name, value in training_result.ensemble_result.metrics.items():
                eval_metrics[f"ensemble_{metric_name}"] = value

        self._log(f"  Evaluated {len(training_result.model_results)} models")

        # Save evaluation report
        eval_path = self._run_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2, default=str)

        return PhaseResult(
            phase=PipelinePhase.EVALUATION,
            success=True,
            duration_seconds=0,
            metrics=eval_metrics,
            artifacts={"evaluation": eval_path},
        )

    def _phase_backtest(self) -> PhaseResult:
        """Phase 8: Backtesting.

        Delegates to Backtester for realistic strategy simulation.
        """
        self._log("Phase 8: Backtest")

        training_result = self._artifacts.get("training_result")
        df = self._artifacts.get("df", self._df)

        if training_result is None or df is None:
            return PhaseResult(
                phase=PipelinePhase.BACKTEST,
                success=False,
                duration_seconds=0,
                error="No training result or data available",
            )

        try:
            from src.inference.backtesting import Backtester, BacktestConfig

            # Get predictions from training result
            # Use ensemble if available, otherwise use best model
            predictions_df = self._get_predictions_for_backtest(training_result, df)

            if predictions_df is None or len(predictions_df) == 0:
                self._log("  No predictions available for backtesting")
                return PhaseResult(
                    phase=PipelinePhase.BACKTEST,
                    success=True,
                    duration_seconds=0,
                    metrics={"note": "no_predictions_available"},
                )

            # Create backtest config based on symbol
            symbol = self.config.symbol.upper()
            if symbol == "MES":
                bt_config = BacktestConfig.for_mes()
            elif symbol == "MGC":
                bt_config = BacktestConfig.for_mgc()
            else:
                bt_config = BacktestConfig()

            # Run backtest
            backtester = Backtester(
                predictions=predictions_df,
                prices=df,
                config=bt_config,
            )
            bt_result = backtester.run()

            # Extract metrics
            backtest_metrics = bt_result.summary()
            self._artifacts["backtest_metrics"] = backtest_metrics
            self._artifacts["backtest_result"] = bt_result

            self._log(f"  Total trades: {backtest_metrics.get('total_trades', 0)}")
            self._log(f"  Win rate: {backtest_metrics.get('win_rate_pct', 0):.1f}%")
            self._log(f"  Sharpe: {backtest_metrics.get('sharpe_ratio', 0):.2f}")

            return PhaseResult(
                phase=PipelinePhase.BACKTEST,
                success=True,
                duration_seconds=0,
                metrics=backtest_metrics,
            )

        except ImportError as e:
            self._log(f"  Backtesting not available: {e}")
            return PhaseResult(
                phase=PipelinePhase.BACKTEST,
                success=False,
                duration_seconds=0,
                error=str(e),
            )

    def _phase_bundling(self) -> PhaseResult:
        """Phase 9: Bundling.

        Delegates to BundleBuilder to create inference artifacts.
        """
        self._log("Phase 9: Bundling")

        training_result = self._artifacts.get("training_result")

        if training_result is None:
            return PhaseResult(
                phase=PipelinePhase.BUNDLING,
                success=False,
                duration_seconds=0,
                error="No training result available",
            )

        try:
            from src.inference.builder import BundleBuilder

            # Create bundle builder
            try:
                from src.core import PipelineConfig as CorePipelineConfig
                # Use the core config if available
                core_config = CorePipelineConfig(
                    symbol=self.config.symbol,
                    output_dir=self._run_dir,
                )
                builder = BundleBuilder(core_config)
            except ImportError:
                builder = BundleBuilder(self.config)

            # Build bundles from training result
            bundle_result = builder.build_from_training_result(training_result)

            bundle_path = self._run_dir / "bundle"
            bundle_path.mkdir(exist_ok=True)
            self._artifacts["bundle_path"] = bundle_path
            self._artifacts["bundle_result"] = bundle_result

            self._log(f"  Created {bundle_result.n_bundles} bundles")
            self._log(f"  Total size: {bundle_result.total_size_mb:.1f} MB")

            return PhaseResult(
                phase=PipelinePhase.BUNDLING,
                success=True,
                duration_seconds=0,
                artifacts={"bundle": bundle_path},
                metrics={
                    "n_bundles": bundle_result.n_bundles,
                    "total_size_mb": bundle_result.total_size_mb,
                },
            )

        except ImportError as e:
            self._log(f"  Bundling not available: {e}")
            # Fallback: just create directory
            bundle_path = self._run_dir / "bundle"
            bundle_path.mkdir(exist_ok=True)
            self._artifacts["bundle_path"] = bundle_path

            return PhaseResult(
                phase=PipelinePhase.BUNDLING,
                success=True,
                duration_seconds=0,
                artifacts={"bundle": bundle_path},
                metrics={"note": "minimal_bundle_created"},
            )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _get_predictions_for_backtest(
        self,
        training_result: Any,
        df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Extract predictions from training result for backtesting."""
        # Try to get OOF predictions from ensemble or best model
        if training_result.aligned_oof is not None:
            # Use aligned OOF from ensemble
            oof = training_result.aligned_oof
            return pd.DataFrame({
                "datetime": df["datetime"].iloc[oof.common_indices],
                "prediction": oof.ensemble_predictions,
                "confidence": oof.ensemble_probabilities.max(axis=1) if oof.ensemble_probabilities is not None else None,
            })

        # Try to get OOF from best model
        best_model = training_result.best_model
        if best_model and best_model in training_result.model_results:
            model_result = training_result.model_results[best_model]
            if model_result.oof_prediction is not None:
                oof = model_result.oof_prediction
                return pd.DataFrame({
                    "datetime": df["datetime"].iloc[oof.indices],
                    "prediction": oof.predictions,
                    "confidence": oof.probabilities.max(axis=1) if oof.probabilities is not None else None,
                })

        return None

    def _run_phase(
        self,
        phase: PipelinePhase,
        handler: Callable[[], PhaseResult],
    ) -> PhaseResult:
        """Run a phase with timing."""
        if phase in self._completed_phases:
            return self._phase_results[phase]

        start = datetime.now()
        try:
            result = handler()
        except Exception as e:
            logger.exception(f"Phase {phase.value} failed")
            result = PhaseResult(
                phase=phase,
                success=False,
                duration_seconds=0,
                error=str(e),
            )
        result.duration_seconds = (datetime.now() - start).total_seconds()

        self._completed_phases.add(phase)
        self._phase_results[phase] = result
        return result

    def _log(self, message: str) -> None:
        """Log a message."""
        if self._verbose >= 1:
            print(message)

    def _save_result(self, result: PipelineResult) -> None:
        """Save result to disk."""
        result_path = self._run_dir / "result.json"
        data = {
            "run_id": result.run_id,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "best_model": result.best_model,
            "phases": {
                name: {"success": r.success, "duration": r.duration_seconds}
                for name, r in result.phases.items()
            },
        }
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def resume(cls, run_dir: str | Path) -> "MLPipeline":
        """Resume from a previous run."""
        run_dir = Path(run_dir)
        config = PipelineConfig.load(run_dir / "config.json")
        return cls(config)
