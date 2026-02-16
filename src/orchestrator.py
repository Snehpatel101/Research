"""
MLPipeline - THE single entry point for the ML pipeline.

This is a thin orchestrator that delegates to:
- UnifiedTrainingOrchestrator: Training, ensemble, and model management
- Backtester: Strategy simulation (optional)
- BundleBuilder: Inference artifacts (optional)

Usage:
    from src import MLPipeline, PipelineConfig

    config = PipelineConfig(symbol="MES", models=["xgboost", "lstm"])
    result = MLPipeline(config).run()
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config.symbol import SymbolConfig
from src.core.config import PipelineConfig

if TYPE_CHECKING:
    from src.models.training.result import TrainingResult

warnings.warn(
    "src.orchestrator is deprecated. Use src.factory.MLFactory instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from the complete pipeline."""

    run_id: str
    config: PipelineConfig
    success: bool
    duration_seconds: float = 0.0
    best_model: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    backtest_metrics: dict[str, float] = field(default_factory=dict)
    output_dir: Path | None = None
    bundle_path: Path | None = None
    training_result: TrainingResult | None = None

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Pipeline Result: {status}",
            f"Run ID: {self.run_id}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]

        if self.best_model:
            lines.append(f"Best Model: {self.best_model}")

        if self.metrics:
            lines.append("\nModel Metrics:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        if self.ensemble_metrics:
            lines.append("\nEnsemble Metrics:")
            for k, v in self.ensemble_metrics.items():
                if isinstance(v, float):
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
    .. deprecated::
        Use :class:`src.factory.MLFactory` instead.

    THE single entry point for the ML pipeline.

    Delegates heavy lifting to:
    - UnifiedTrainingOrchestrator: Training, OOF, and ensemble
    - Backtester: Strategy simulation
    - BundleBuilder: Inference artifacts

    Example:
        from src import MLPipeline, PipelineConfig

        config = PipelineConfig(symbol="MES", models=["xgboost"])
        result = MLPipeline(config).run()

    Advanced usage:
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
        self._training_result = None

        # Create run directory
        self._run_dir = Path(config.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # config.save(self._run_dir / "config.json")

        if verbose >= 1:
            logger.info(f"MLPipeline initialized: {self._run_dir.name}")

    def run(self) -> PipelineResult:
        """Run the complete pipeline."""
        start_time = datetime.now()
        self._log(f"Starting pipeline: {self._run_dir.name}")

        # Phase 1: Load data
        df = self._load_data()

        # Phase 2: Train (delegates to UnifiedTrainingOrchestrator)
        self._training_result = self._train(df)

        # Phase 3: Backtest (optional)
        backtest_metrics: dict[str, float] = {}

        # Phase 4: Bundle (optional)
        bundle_path = None
        if self.config.save_bundle:
            bundle_path = self._bundle()

        duration = (datetime.now() - start_time).total_seconds()

        # Build result
        result = PipelineResult(
            run_id=self._run_dir.name,
            config=self.config,
            success=True,
            duration_seconds=duration,
            best_model=self._training_result.best_model if self._training_result else None,
            metrics=self._training_result.get_metrics_summary() if self._training_result else {},
            ensemble_metrics=self._get_ensemble_metrics(),
            backtest_metrics=backtest_metrics,
            output_dir=self._run_dir,
            bundle_path=bundle_path,
            training_result=self._training_result,
        )

        self._save_result(result)
        self._log(f"Pipeline completed in {duration:.1f}s")
        return result

    def _load_data(self) -> pd.DataFrame:
        """Load raw OHLCV data."""
        self._log("Loading data...")

        if self._df is not None:
            self._log(f"  Using provided DataFrame: {len(self._df)} rows")
            return self._df

        self._log(f"  Loading: {self.config.data_path}")
        df = pd.read_parquet(self.config.data_path)
        self._log(f"  Rows: {len(df)}")
        return df

    def _train(self, df: pd.DataFrame) -> Any:
        """Train models via UnifiedTrainingOrchestrator."""
        self._log("Training models...")
        self._log(f"  Models: {self.config.models}")
        self._log(f"  Mode: {self.config.training_mode}")

        from src.models.training import UnifiedTrainingOrchestrator

        # Convert config to core PipelineConfig format
        try:
            from src.core import PipelineConfig as CorePipelineConfig

            core_config = CorePipelineConfig(
                symbol=self.config.symbol,
                data_path=str(self.config.data_path),
                output_dir=self._run_dir,
                models=self.config.models,
                horizons=self.config.horizons,
                build_ensemble=self.config.build_ensemble,
                # ensemble_method doesn't exist - using meta_learner instead
                meta_learner=getattr(self.config, "ensemble_method", "ridge_meta"),
                training_mode=self.config.training_mode,
                cv_method=self.config.cv_method,
                n_splits=self.config.n_splits,
                purge_bars=self.config.purge_bars,
                embargo_bars=self.config.embargo_bars,
                random_state=getattr(self.config, "random_seed", 42),
                # save_oof doesn't exist in core config
                # save_oof=self.config.save_oof,
            )
        except (ImportError, AttributeError):
            core_config = self.config

        orchestrator = UnifiedTrainingOrchestrator(core_config)
        result = orchestrator.train(df)

        self._log(f"  Trained {result.n_models} models")
        self._log(f"  Best model: {result.best_model}")

        return result

    def _backtest(self, df: pd.DataFrame) -> dict:
        """Run backtest on predictions."""
        self._log("Backtesting...")

        if self._training_result is None:
            self._log("  No training result available")
            return {}

        try:
            from src.inference.backtesting import BacktestConfig, Backtester

            predictions_df = self._get_predictions_for_backtest(df)
            if predictions_df is None or len(predictions_df) == 0:
                self._log("  No predictions available")
                return {}

            # Symbol-specific config
            symbol = self.config.symbol.upper()
            bt_config = BacktestConfig.from_symbol_config(SymbolConfig.from_symbol(symbol))

            backtester = Backtester(predictions=predictions_df, prices=df, config=bt_config)
            bt_result = backtester.run()
            metrics = bt_result.summary()

            self._log(f"  Trades: {metrics.get('total_trades', 0)}")
            self._log(f"  Win rate: {metrics.get('win_rate_pct', 0):.1f}%")
            self._log(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")

            return metrics

        except ImportError as e:
            self._log(f"  Backtesting not available: {e}")
            return {}

    def _bundle(self) -> Path | None:
        """Create inference bundles."""
        self._log("Creating bundles...")

        if self._training_result is None:
            self._log("  No training result available")
            return None

        try:
            from src.inference.builder import BundleBuilder

            try:
                from src.core import PipelineConfig as CorePipelineConfig

                core_config = CorePipelineConfig(
                    symbol=self.config.symbol,
                    data_path=str(self.config.data_path),
                    output_dir=self._run_dir,
                )
                builder = BundleBuilder(core_config)
            except (ImportError, TypeError):
                # ImportError: core config not available
                # TypeError: config incompatibility (missing required fields)
                builder = BundleBuilder(self.config)

            bundle_result = builder.build_from_training_result(self._training_result)
            bundle_path = self._run_dir / "bundle"
            bundle_path.mkdir(exist_ok=True)

            self._log(f"  Created {bundle_result.n_bundles} bundles")
            return bundle_path

        except ImportError as e:
            self._log(f"  Bundling not available: {e}")
            bundle_path = self._run_dir / "bundle"
            bundle_path.mkdir(exist_ok=True)
            return bundle_path

    def _get_predictions_for_backtest(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Extract predictions from training result."""
        if self._training_result is None:
            return None

        # Try aligned OOF from ensemble
        if self._training_result.aligned_oof is not None:
            oof = self._training_result.aligned_oof
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
        best_model = self._training_result.best_model
        if best_model and best_model in self._training_result.model_results:
            model_result = self._training_result.model_results[best_model]
            if model_result.oof_prediction is not None:
                oof = model_result.oof_prediction
                return pd.DataFrame(
                    {
                        "datetime": df["datetime"].iloc[oof.indices],
                        "prediction": oof.predictions,
                        "confidence": (
                            oof.probabilities.max(axis=1) if oof.probabilities is not None else None
                        ),
                    }
                )

        return None

    def _get_ensemble_metrics(self) -> dict:
        """Extract ensemble metrics from training result."""
        if self._training_result is not None and self._training_result.ensemble_result is not None:
            return self._training_result.ensemble_result.metrics
        return {}

    def _log(self, message: str) -> None:
        """Log message if verbose."""
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
            "output_dir": str(result.output_dir),
        }
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def resume(cls, run_dir: str | Path) -> MLPipeline:
        """Resume from a previous run."""
        run_dir = Path(run_dir)
        config = PipelineConfig.load(run_dir / "config.json")
        return cls(config)


__all__ = ["MLPipeline", "PipelineResult"]
