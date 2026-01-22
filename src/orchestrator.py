"""
MLPipeline - THE orchestrator for the ML pipeline.

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
from typing import Any, Callable

import pandas as pd

from src.pipeline_config import PipelineConfig

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

        return "\n".join(lines)


class MLPipeline:
    """
    THE orchestrator for the ML pipeline.

    Example:
        from src import MLPipeline, PipelineConfig

        config = PipelineConfig(symbol="MES")
        result = MLPipeline(config).run()
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
        )

        self._save_result(result)
        self._log(f"Pipeline completed in {duration:.1f}s")
        return result

    def run_data(self) -> None:
        """Run data phases (1-4)."""
        self._run_phase(PipelinePhase.DATA_PREP, self._phase_data_prep)
        self._run_phase(PipelinePhase.FEATURES, self._phase_features)
        self._run_phase(PipelinePhase.LABELING, self._phase_labeling)
        self._run_phase(PipelinePhase.SPLITS, self._phase_splits)

    def run_train(self) -> None:
        """Run training phases (5-6)."""
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
        """Phase 1: Data preparation."""
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
        """Phase 2: Feature engineering."""
        self._log("Phase 2: Feature Engineering")

        df = self._artifacts.get("df", self._df)
        # Feature engineering would happen here
        self._artifacts["features_df"] = df

        return PhaseResult(
            phase=PipelinePhase.FEATURES,
            success=True,
            duration_seconds=0,
        )

    def _phase_labeling(self) -> PhaseResult:
        """Phase 3: Labeling."""
        self._log("Phase 3: Labeling")

        df = self._artifacts.get("features_df")
        # Labeling would happen here
        self._artifacts["labeled_df"] = df

        return PhaseResult(
            phase=PipelinePhase.LABELING,
            success=True,
            duration_seconds=0,
        )

    def _phase_splits(self) -> PhaseResult:
        """Phase 4: Data splitting."""
        self._log("Phase 4: Splits")

        # Split creation would happen here
        return PhaseResult(
            phase=PipelinePhase.SPLITS,
            success=True,
            duration_seconds=0,
        )

    def _phase_training(self) -> PhaseResult:
        """Phase 5: Model training."""
        self._log("Phase 5: Training")
        self._log(f"  Models: {self.config.models}")
        self._log(f"  Mode: {self.config.training_mode}")

        # Training would happen here
        # This is where we'd call the training orchestrator directly

        return PhaseResult(
            phase=PipelinePhase.TRAINING,
            success=True,
            duration_seconds=0,
            metrics={"n_models": len(self.config.models)},
        )

    def _phase_ensemble(self) -> PhaseResult:
        """Phase 6: Ensemble building."""
        self._log("Phase 6: Ensemble")
        self._log(f"  Method: {self.config.ensemble_method}")

        # Ensemble building would happen here
        return PhaseResult(
            phase=PipelinePhase.ENSEMBLE,
            success=True,
            duration_seconds=0,
        )

    def _phase_evaluation(self) -> PhaseResult:
        """Phase 7: Evaluation."""
        self._log("Phase 7: Evaluation")

        return PhaseResult(
            phase=PipelinePhase.EVALUATION,
            success=True,
            duration_seconds=0,
        )

    def _phase_backtest(self) -> PhaseResult:
        """Phase 8: Backtesting."""
        self._log("Phase 8: Backtest")

        return PhaseResult(
            phase=PipelinePhase.BACKTEST,
            success=True,
            duration_seconds=0,
        )

    def _phase_bundling(self) -> PhaseResult:
        """Phase 9: Bundling."""
        self._log("Phase 9: Bundling")

        bundle_path = self._run_dir / "bundle"
        bundle_path.mkdir(exist_ok=True)
        self._artifacts["bundle_path"] = bundle_path

        return PhaseResult(
            phase=PipelinePhase.BUNDLING,
            success=True,
            duration_seconds=0,
            artifacts={"bundle": bundle_path},
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _run_phase(
        self,
        phase: PipelinePhase,
        handler: Callable[[], PhaseResult],
    ) -> PhaseResult:
        """Run a phase with timing."""
        if phase in self._completed_phases:
            return self._phase_results[phase]

        start = datetime.now()
        result = handler()
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
