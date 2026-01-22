# src/training/services/artifact_persistence.py
"""Service for managing training artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

from src.cross_validation import OOFPrediction
from src.core import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ArtifactSaveRequest:
    """Request to save training artifacts."""

    output_dir: Path
    config: PipelineConfig
    model_results: dict[str, Any]
    oof_predictions: dict[str, OOFPrediction]
    trained_models: dict[str, Any]
    save_models: bool = True
    save_oof: bool = True


class ArtifactManager:
    """
    Service for managing training artifacts.

    Responsibilities:
    - Save config, metrics, OOF predictions, and models
    - Load artifacts from previous runs

    Does NOT:
    - Train models (handled by ModelTrainingService)
    - Generate OOF (handled by OOFGenerationService)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self, request: ArtifactSaveRequest) -> None:
        """
        Save all training artifacts.

        Extracted from UnifiedTrainingOrchestrator._save_results.

        Args:
            request: ArtifactSaveRequest containing all artifacts to save
        """
        logger.info(f"\nSaving results to: {request.output_dir}")

        # Ensure output directory exists
        request.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.save_config(request.config, request.output_dir)

        # Save metrics summary
        self.save_metrics(request.model_results, request.output_dir)

        # Save OOF predictions if available and enabled
        if request.oof_predictions and request.save_oof:
            self.save_oof_predictions(request.oof_predictions, request.output_dir)

        # Save trained models if enabled
        if request.save_models:
            self.save_models(request.trained_models, request.output_dir)

        logger.info(f"Results saved to: {request.output_dir}")

    def save_config(
        self, config: PipelineConfig, output_dir: Path | None = None
    ) -> Path:
        """
        Save configuration.

        Args:
            config: PipelineConfig to save
            output_dir: Optional output directory. Uses self.output_dir if not provided.

        Returns:
            Path to saved config file
        """
        target_dir = output_dir or self.output_dir
        config_path = target_dir / "config.json"
        config.save(config_path)
        logger.info(f"  Saved config to: {config_path}")
        return config_path

    def save_metrics(
        self, model_results: dict[str, Any], output_dir: Path | None = None
    ) -> Path:
        """
        Save metrics summary.

        Args:
            model_results: Dictionary mapping model keys to TrainingResult objects
            output_dir: Optional output directory. Uses self.output_dir if not provided.

        Returns:
            Path to saved metrics file
        """
        target_dir = output_dir or self.output_dir
        metrics_path = target_dir / "metrics_summary.json"

        metrics_summary: dict[str, Any] = {}
        for key, result in model_results.items():
            metrics_summary[key] = {
                "model_name": result.model_name,
                "horizon": result.horizon,
                "metrics": result.metrics,
                "training_time": result.training_time_seconds,
                "n_features": result.n_features,
                "data_rank": result.data_rank,
            }

        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"  Saved metrics to: {metrics_path}")
        return metrics_path

    def save_oof_predictions(
        self, oof_predictions: dict[str, OOFPrediction], output_dir: Path | None = None
    ) -> None:
        """
        Save OOF predictions.

        Args:
            oof_predictions: Dictionary mapping model keys to OOFPrediction objects
            output_dir: Optional output directory. Uses self.output_dir if not provided.
        """
        target_dir = output_dir or self.output_dir
        oof_dir = target_dir / "oof"
        oof_dir.mkdir(exist_ok=True)

        for key, oof in oof_predictions.items():
            oof_path = oof_dir / f"{key}_oof.parquet"
            oof.predictions.to_parquet(oof_path)
            logger.debug(f"  Saved OOF for {key} to: {oof_path}")

        logger.info(f"  Saved {len(oof_predictions)} OOF predictions to: {oof_dir}")

    def save_models(
        self, trained_models: dict[str, Any], output_dir: Path | None = None
    ) -> None:
        """
        Save trained model files.

        Args:
            trained_models: Dictionary mapping model keys to trained model/trainer objects
            output_dir: Optional output directory. Uses self.output_dir if not provided.
        """
        target_dir = output_dir or self.output_dir
        models_dir = target_dir / "models"
        models_dir.mkdir(exist_ok=True)

        saved_count = 0
        for key, trainer in trained_models.items():
            try:
                model_path = models_dir / f"{key}.pkl"
                trainer.save(model_path)
                saved_count += 1
                logger.debug(f"  Saved model {key} to: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save model {key}: {e}")

        logger.info(f"  Saved {saved_count}/{len(trained_models)} models to: {models_dir}")

    def load_config(self, config_path: Path | None = None) -> PipelineConfig:
        """
        Load configuration from file.

        Args:
            config_path: Optional path to config file. Uses output_dir/config.json if not provided.

        Returns:
            Loaded PipelineConfig
        """
        path = config_path or (self.output_dir / "config.json")
        return PipelineConfig.load(path)

    def load_metrics(self, metrics_path: Path | None = None) -> dict[str, Any]:
        """
        Load metrics summary from file.

        Args:
            metrics_path: Optional path to metrics file. Uses output_dir/metrics_summary.json if not provided.

        Returns:
            Metrics summary dictionary
        """
        path = metrics_path or (self.output_dir / "metrics_summary.json")
        with open(path) as f:
            return json.load(f)

    def list_saved_models(self) -> list[str]:
        """
        List all saved model keys.

        Returns:
            List of model keys (e.g., ["xgboost_h20", "lightgbm_h20"])
        """
        models_dir = self.output_dir / "models"
        if not models_dir.exists():
            return []
        return [p.stem for p in models_dir.glob("*.pkl")]

    def list_saved_oof(self) -> list[str]:
        """
        List all saved OOF prediction keys.

        Returns:
            List of OOF keys (e.g., ["xgboost_h20", "lightgbm_h20"])
        """
        oof_dir = self.output_dir / "oof"
        if not oof_dir.exists():
            return []
        return [p.stem.replace("_oof", "") for p in oof_dir.glob("*_oof.parquet")]
