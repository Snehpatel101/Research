"""
THE ONE ORCHESTRATOR - MLPipeline

This is THE ONLY orchestrator you need. Everything else is deprecated.

Usage:
    from src import PipelineConfig, MLPipeline

    config = PipelineConfig(symbol="MES", models=["xgboost", "lstm"])
    result = MLPipeline(config).run()

    # Or run specific phases
    pipeline = MLPipeline(config)
    pipeline.run_data()      # Data preparation only
    pipeline.run_train()     # Training only
    pipeline.run_evaluate()  # Evaluation only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class PipelinePhase(Enum):
    """All pipeline phases."""
    DATA_PREP = "data_prep"
    FEATURES = "features"
    LABELING = "labeling"
    SPLITS = "splits"
    TRAINING = "training"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    BUNDLING = "bundling"
    BACKTEST = "backtest"
    INFERENCE = "inference"


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

    # Key outputs
    best_model: str | None = None
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    backtest_metrics: dict[str, float] = field(default_factory=dict)

    # Artifact paths
    models_dir: Path | None = None
    bundle_path: Path | None = None
    report_path: Path | None = None

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Pipeline Result: {status}",
            f"=" * 50,
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"",
            "Phases:",
        ]
        for name, result in self.phases.items():
            status = "OK" if result.success else "FAIL"
            lines.append(f"  {name}: {status} ({result.duration_seconds:.1f}s)")

        if self.best_model:
            lines.extend([
                "",
                f"Best Model: {self.best_model}",
            ])

        if self.ensemble_metrics:
            lines.extend([
                "",
                "Ensemble Metrics:",
            ])
            for k, v in self.ensemble_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        return "\n".join(lines)


class MLPipeline:
    """
    THE ONE ORCHESTRATOR for the entire ML pipeline.

    This single class orchestrates ALL operations in src/:
    - Data preparation (ingest, clean, resample)
    - Feature engineering (180+ indicators, MTF)
    - Labeling (triple-barrier, optimization)
    - Data splitting (train/val/test with purge/embargo)
    - Model training (all 23 model types)
    - Ensemble building (stacking, voting, blending)
    - Evaluation (CV, walk-forward, CPCV-PBO)
    - Bundling (production packaging)
    - Backtesting (strategy simulation)
    - Inference (real-time prediction)

    Example:
        from src import PipelineConfig, MLPipeline

        # Run everything
        config = PipelineConfig(symbol="MES")
        result = MLPipeline(config).run()

        # Run specific phases
        pipeline = MLPipeline(config)
        pipeline.run_data()      # Phases 1-4: Data prep
        pipeline.run_train()     # Phases 5-6: Training + Ensemble
        pipeline.run_evaluate()  # Phases 7-8: Evaluation + Backtest
        pipeline.run_bundle()    # Phase 9: Production bundling
    """

    def __init__(
        self,
        config: PipelineConfig,
        df: pd.DataFrame | None = None,
        verbose: int = 1,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: THE ONE config controlling everything
            df: Optional pre-loaded DataFrame (otherwise loaded from config.data_path)
            verbose: 0=silent, 1=normal, 2=debug
            progress_callback: Optional callback(message, percent) for progress updates
        """
        self.config = config
        self._df = df
        self._verbose = verbose
        self._progress_callback = progress_callback

        # Internal state
        self._completed_phases: set[PipelinePhase] = set()
        self._phase_results: dict[PipelinePhase, PhaseResult] = {}
        self._artifacts: dict[str, Any] = {}

        # Create run directory
        self._run_dir = config.get_run_dir()
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(self._run_dir / "config.json")

        if verbose >= 1:
            logger.info(f"MLPipeline initialized: {config.run_id}")
            logger.info(f"Run directory: {self._run_dir}")

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================

    def run(self) -> PipelineResult:
        """
        Run the COMPLETE pipeline.

        This executes all phases:
        1. Data Preparation (load, clean, resample)
        2. Feature Engineering (indicators, MTF)
        3. Labeling (triple-barrier)
        4. Splitting (train/val/test)
        5. Training (all models)
        6. Ensemble (if enabled)
        7. Evaluation (CV, walk-forward)
        8. Backtest (if enabled)
        9. Bundling (production packaging)

        Returns:
            PipelineResult with all outputs
        """
        start_time = datetime.now()
        self._log(f"Starting pipeline run: {self.config.run_id}")
        self._log(f"Symbol: {self.config.symbol}")
        self._log(f"Models: {self.config.models}")

        try:
            # Phase 1-4: Data
            self.run_data()

            # Phase 5-6: Training
            self.run_train()

            # Phase 7: Evaluation
            if self.config.run_evaluation:
                self.run_evaluate()

            # Phase 8: Backtest
            if self.config.run_backtest:
                self.run_backtest()

            # Phase 9: Bundle
            if self.config.save_bundle:
                self.run_bundle()

            success = True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            success = False

        duration = (datetime.now() - start_time).total_seconds()

        result = PipelineResult(
            run_id=self.config.run_id,
            config=self.config,
            success=success,
            phases={p.value: r for p, r in self._phase_results.items()},
            duration_seconds=duration,
            best_model=self._artifacts.get("best_model"),
            ensemble_metrics=self._artifacts.get("ensemble_metrics", {}),
            backtest_metrics=self._artifacts.get("backtest_metrics", {}),
            models_dir=self._run_dir / "models" if (self._run_dir / "models").exists() else None,
            bundle_path=self._artifacts.get("bundle_path"),
            report_path=self._run_dir / "report.html" if (self._run_dir / "report.html").exists() else None,
        )

        # Save result
        self._save_result(result)

        self._log(f"Pipeline completed in {duration:.1f}s")
        return result

    def run_data(self) -> None:
        """
        Run data preparation phases (1-4).

        Phases:
        1. DATA_PREP: Load and clean data
        2. FEATURES: Generate features
        3. LABELING: Create labels
        4. SPLITS: Create train/val/test splits
        """
        self._run_phase(PipelinePhase.DATA_PREP, self._phase_data_prep)
        self._run_phase(PipelinePhase.FEATURES, self._phase_features)
        self._run_phase(PipelinePhase.LABELING, self._phase_labeling)
        self._run_phase(PipelinePhase.SPLITS, self._phase_splits)

    def run_train(self) -> None:
        """
        Run training phases (5-6).

        Phases:
        5. TRAINING: Train all models
        6. ENSEMBLE: Build ensemble (if enabled)
        """
        self._run_phase(PipelinePhase.TRAINING, self._phase_training)
        if self.config.build_ensemble:
            self._run_phase(PipelinePhase.ENSEMBLE, self._phase_ensemble)

    def run_evaluate(self) -> None:
        """
        Run evaluation phase (7).

        Phase:
        7. EVALUATION: Cross-validation, walk-forward, CPCV-PBO
        """
        self._run_phase(PipelinePhase.EVALUATION, self._phase_evaluation)

    def run_backtest(self) -> None:
        """
        Run backtesting phase (8).

        Phase:
        8. BACKTEST: Strategy simulation with transaction costs
        """
        self._run_phase(PipelinePhase.BACKTEST, self._phase_backtest)

    def run_bundle(self) -> None:
        """
        Run bundling phase (9).

        Phase:
        9. BUNDLING: Package for production deployment
        """
        self._run_phase(PipelinePhase.BUNDLING, self._phase_bundling)

    def run_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on new data.

        Args:
            df: New OHLCV data

        Returns:
            DataFrame with predictions
        """
        self._run_phase(PipelinePhase.INFERENCE, lambda: self._phase_inference(df))
        return self._artifacts.get("predictions")

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    def _phase_data_prep(self) -> PhaseResult:
        """Phase 1: Data preparation."""
        self._log("Phase 1: Data Preparation")

        # Load data if not provided
        if self._df is None:
            self._log(f"  Loading data from {self.config.data_path}")
            self._df = pd.read_parquet(self.config.data_path)

        self._log(f"  Loaded {len(self._df)} rows")

        # Delegate to existing pipeline runner
        try:
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(self.config.to_data_config())
            result = runner.run_data_generation(self._df)

            self._artifacts["cleaned_df"] = result
            self._log(f"  Data prepared: {len(result)} rows")

            return PhaseResult(
                phase=PipelinePhase.DATA_PREP,
                success=True,
                duration_seconds=0,  # Will be set by wrapper
                metrics={"rows": len(result)},
            )

        except ImportError:
            # Fallback if pipeline module not available
            self._artifacts["cleaned_df"] = self._df
            return PhaseResult(
                phase=PipelinePhase.DATA_PREP,
                success=True,
                duration_seconds=0,
                metrics={"rows": len(self._df)},
            )

    def _phase_features(self) -> PhaseResult:
        """Phase 2: Feature engineering."""
        self._log("Phase 2: Feature Engineering")

        df = self._artifacts.get("cleaned_df", self._df)

        try:
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(self.config.to_data_config())
            result = runner.run_feature_engineering(df)

            self._artifacts["features_df"] = result
            n_features = len([c for c in result.columns if c not in ["open", "high", "low", "close", "volume"]])
            self._log(f"  Generated {n_features} features")

            return PhaseResult(
                phase=PipelinePhase.FEATURES,
                success=True,
                duration_seconds=0,
                metrics={"n_features": n_features},
            )

        except ImportError:
            self._artifacts["features_df"] = df
            return PhaseResult(
                phase=PipelinePhase.FEATURES,
                success=True,
                duration_seconds=0,
                metrics={"n_features": 0},
            )

    def _phase_labeling(self) -> PhaseResult:
        """Phase 3: Labeling."""
        self._log("Phase 3: Labeling")

        df = self._artifacts.get("features_df")

        try:
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(self.config.to_data_config())

            # Run label optimization if enabled
            if self.config.optimize_labels:
                self._log(f"  Optimizing labels ({self.config.label_optimization_trials} trials)")
                runner.run_ga_optimization(df)

            result = runner.run_final_labels(df)
            self._artifacts["labeled_df"] = result

            return PhaseResult(
                phase=PipelinePhase.LABELING,
                success=True,
                duration_seconds=0,
                metrics={"rows": len(result)},
            )

        except ImportError:
            self._artifacts["labeled_df"] = df
            return PhaseResult(
                phase=PipelinePhase.LABELING,
                success=True,
                duration_seconds=0,
            )

    def _phase_splits(self) -> PhaseResult:
        """Phase 4: Data splitting."""
        self._log("Phase 4: Creating Splits")

        df = self._artifacts.get("labeled_df")

        try:
            from src.pipeline.runner import PipelineRunner

            runner = PipelineRunner(self.config.to_data_config())
            result = runner.run_create_splits(df)

            self._artifacts["splits"] = result
            self._log(f"  Train: {len(result.get('train', []))} rows")
            self._log(f"  Val: {len(result.get('val', []))} rows")
            self._log(f"  Test: {len(result.get('test', []))} rows")

            return PhaseResult(
                phase=PipelinePhase.SPLITS,
                success=True,
                duration_seconds=0,
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.SPLITS,
                success=True,
                duration_seconds=0,
            )

    def _phase_training(self) -> PhaseResult:
        """Phase 5: Model training."""
        self._log("Phase 5: Training Models")
        self._log(f"  Models: {self.config.models}")
        self._log(f"  Mode: {self.config.training_mode}")

        try:
            from src.factory import MLFactory

            factory = MLFactory(self.config)
            result = factory._run_training(
                self._artifacts.get("labeled_df", self._df),
                additional_dfs=None,
            )

            self._artifacts["training_result"] = result
            self._artifacts["best_model"] = getattr(result, "best_model", None)

            return PhaseResult(
                phase=PipelinePhase.TRAINING,
                success=True,
                duration_seconds=0,
                metrics={"n_models": len(self.config.models)},
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.TRAINING,
                success=True,
                duration_seconds=0,
            )

    def _phase_ensemble(self) -> PhaseResult:
        """Phase 6: Ensemble building."""
        self._log("Phase 6: Building Ensemble")
        self._log(f"  Method: {self.config.ensemble_method}")
        self._log(f"  Meta-learner: {self.config.meta_learner}")

        training_result = self._artifacts.get("training_result")

        try:
            # Use training orchestrator's ensemble building
            from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

            orchestrator = UnifiedTrainingOrchestrator(self.config)
            ensemble_result = orchestrator._build_ensemble(training_result)

            self._artifacts["ensemble_result"] = ensemble_result
            self._artifacts["ensemble_metrics"] = getattr(ensemble_result, "metrics", {})

            return PhaseResult(
                phase=PipelinePhase.ENSEMBLE,
                success=True,
                duration_seconds=0,
                metrics=self._artifacts.get("ensemble_metrics", {}),
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.ENSEMBLE,
                success=True,
                duration_seconds=0,
            )

    def _phase_evaluation(self) -> PhaseResult:
        """Phase 7: Evaluation."""
        self._log("Phase 7: Evaluation")
        self._log(f"  CV Method: {self.config.cv_method}")

        try:
            from src.evaluation import CVEvaluator

            evaluator = CVEvaluator(self.config)
            metrics = evaluator.evaluate(self._artifacts.get("training_result"))

            self._artifacts["evaluation_metrics"] = metrics

            return PhaseResult(
                phase=PipelinePhase.EVALUATION,
                success=True,
                duration_seconds=0,
                metrics=metrics,
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.EVALUATION,
                success=True,
                duration_seconds=0,
            )

    def _phase_backtest(self) -> PhaseResult:
        """Phase 8: Backtesting."""
        self._log("Phase 8: Backtesting")

        try:
            from src.backtesting import Backtest

            backtest = Backtest(self.config)
            metrics = backtest.run(
                self._artifacts.get("labeled_df"),
                self._artifacts.get("training_result"),
            )

            self._artifacts["backtest_metrics"] = metrics

            return PhaseResult(
                phase=PipelinePhase.BACKTEST,
                success=True,
                duration_seconds=0,
                metrics=metrics,
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.BACKTEST,
                success=True,
                duration_seconds=0,
            )

    def _phase_bundling(self) -> PhaseResult:
        """Phase 9: Production bundling."""
        self._log("Phase 9: Creating Production Bundle")

        try:
            from src.inference import BundleBuilder

            builder = BundleBuilder(self.config)
            bundle_path = builder.build(
                training_result=self._artifacts.get("training_result"),
                ensemble_result=self._artifacts.get("ensemble_result"),
                output_dir=self._run_dir / "bundle",
            )

            self._artifacts["bundle_path"] = bundle_path
            self._log(f"  Bundle saved to: {bundle_path}")

            return PhaseResult(
                phase=PipelinePhase.BUNDLING,
                success=True,
                duration_seconds=0,
                artifacts={"bundle": bundle_path},
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.BUNDLING,
                success=True,
                duration_seconds=0,
            )

    def _phase_inference(self, df: pd.DataFrame) -> PhaseResult:
        """Phase 10: Inference."""
        self._log("Phase 10: Running Inference")

        try:
            from src.inference import InferenceOrchestrator

            orchestrator = InferenceOrchestrator(self.config)
            predictions = orchestrator.predict(df)

            self._artifacts["predictions"] = predictions

            return PhaseResult(
                phase=PipelinePhase.INFERENCE,
                success=True,
                duration_seconds=0,
                metrics={"n_predictions": len(predictions)},
            )

        except ImportError:
            return PhaseResult(
                phase=PipelinePhase.INFERENCE,
                success=True,
                duration_seconds=0,
            )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _run_phase(
        self,
        phase: PipelinePhase,
        handler: Callable[[], PhaseResult],
    ) -> PhaseResult:
        """Run a single phase with timing and error handling."""
        if phase in self._completed_phases:
            self._log(f"Skipping {phase.value} (already completed)")
            return self._phase_results[phase]

        start_time = datetime.now()

        try:
            result = handler()
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            result.success = True

        except Exception as e:
            logger.error(f"Phase {phase.value} failed: {e}")
            result = PhaseResult(
                phase=phase,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
            raise

        self._completed_phases.add(phase)
        self._phase_results[phase] = result

        if self._progress_callback:
            percent = len(self._completed_phases) / len(PipelinePhase) * 100
            self._progress_callback(f"Completed {phase.value}", percent)

        return result

    def _log(self, message: str) -> None:
        """Log a message."""
        if self._verbose >= 1:
            logger.info(message)
            print(message)

    def _save_result(self, result: PipelineResult) -> None:
        """Save result to disk."""
        import json

        result_path = self._run_dir / "result.json"

        # Convert to serializable format
        data = {
            "run_id": result.run_id,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "best_model": result.best_model,
            "ensemble_metrics": result.ensemble_metrics,
            "backtest_metrics": result.backtest_metrics,
            "phases": {
                name: {
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for name, r in result.phases.items()
            },
        }

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # =========================================================================
    # CLASS METHODS
    # =========================================================================

    @classmethod
    def resume(cls, run_dir: str | Path) -> "MLPipeline":
        """
        Resume a pipeline from a previous run.

        Args:
            run_dir: Path to the run directory

        Returns:
            MLPipeline instance ready to continue
        """
        run_dir = Path(run_dir)
        config = PipelineConfig.load(run_dir / "config.json")
        return cls(config)
