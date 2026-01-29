"""
InferenceOrchestrator - THE single entry point for all inference operations.

Uses PipelineConfig from src/core as the ONLY configuration source.
Integrates with PHASE_3 training and PHASE_4 ensemble results.

PHASE_5: Inference Orchestration

This module provides the unified inference orchestrator that:
1. Uses PipelineConfig as the ONLY configuration source
2. Loads ModelBundles from experiment directories or specific paths
3. Supports single model and ensemble inference
4. Integrates with PreprocessingGraph for raw OHLCV inference
5. Supports batch inference for large datasets
6. Returns structured PredictionResult

Example:
    from src.core import PipelineConfig
    from src.inference import InferenceOrchestrator

    # Load from experiment directory
    config = PipelineConfig.load("./experiments/exp_001/config.json")
    orchestrator = InferenceOrchestrator.from_experiment(config)

    # Predict with pre-computed features
    result = orchestrator.predict(X_new)

    # Predict from raw OHLCV (end-to-end)
    result = orchestrator.predict_from_raw(raw_ohlcv_df)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core import PipelineConfig

# PredictionResult: Import from canonical location (Phase 27 consolidation)
from src.core.interfaces import PredictionResult

logger = logging.getLogger(__name__)


# =============================================================================
# INFERENCE ORCHESTRATOR
# =============================================================================


class InferenceOrchestrator:
    """
    THE single entry point for all inference in the ML Factory.

    Uses PipelineConfig from src/core as the ONLY configuration source.

    Supports:
    - Single model inference
    - Ensemble inference (stacking with meta-learner)
    - Raw OHLCV inference (via PreprocessingGraph)
    - Batch inference for large datasets

    Usage:
        from src.core import PipelineConfig
        from src.inference import InferenceOrchestrator

        # Load orchestrator from saved bundles
        config = PipelineConfig.load("./experiments/exp_001/config.json")
        orchestrator = InferenceOrchestrator.from_experiment(config)

        # Or load specific bundle
        orchestrator = InferenceOrchestrator.from_bundle("./bundles/xgb_h20")

        # Predict with pre-computed features
        result = orchestrator.predict(X_new)

        # Predict from raw OHLCV (end-to-end)
        result = orchestrator.predict_from_raw(raw_ohlcv_df)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        """
        Initialize InferenceOrchestrator.

        Args:
            config: Optional PipelineConfig from src/core
        """
        self.config = config
        self._bundles: dict[str, Any] = {}  # model_name -> ModelBundle
        self._ensemble_bundle: Any | None = None  # EnsembleBundle or meta-learner
        self._preprocessing_graph: Any | None = None
        self._is_loaded = False

        logger.info("InferenceOrchestrator initialized")

    @classmethod
    def from_config(cls, config: PipelineConfig) -> InferenceOrchestrator:
        """
        Create orchestrator from PipelineConfig.

        Args:
            config: PipelineConfig instance

        Returns:
            InferenceOrchestrator ready for bundle loading
        """
        return cls(config)

    @classmethod
    def from_experiment(
        cls,
        config: PipelineConfig,
        load_ensemble: bool = True,
    ) -> InferenceOrchestrator:
        """
        Create orchestrator from experiment results.

        Loads all saved bundles from config.output_dir.

        Args:
            config: PipelineConfig with output_dir pointing to experiment
            load_ensemble: Whether to load ensemble bundle if available

        Returns:
            InferenceOrchestrator with loaded bundles
        """
        orch = cls(config)

        output_dir = Path(config.output_dir)
        bundles_dir = output_dir / "bundles"

        if bundles_dir.exists():
            orch._load_bundles_from_dir(bundles_dir)

        if load_ensemble:
            ensemble_dir = output_dir / "ensemble" / "bundle"
            if ensemble_dir.exists():
                orch._load_ensemble_bundle(ensemble_dir)
            else:
                # Try alternative ensemble location
                ensemble_meta_dir = output_dir / "ensemble" / "meta_learner"
                if ensemble_meta_dir.exists():
                    orch._load_ensemble_bundle(ensemble_meta_dir)

        return orch

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        config: PipelineConfig | None = None,
    ) -> InferenceOrchestrator:
        """
        Create orchestrator from a single bundle.

        Args:
            bundle_path: Path to saved bundle directory
            config: Optional PipelineConfig

        Returns:
            InferenceOrchestrator with loaded bundle
        """
        from src.inference.bundle import ModelBundle

        orch = cls(config)
        bundle = ModelBundle.load(bundle_path)
        orch._bundles[bundle.metadata.model_name] = bundle
        orch._is_loaded = True

        # Extract preprocessing graph if available
        if bundle.preprocessing_graph is not None:
            orch._preprocessing_graph = bundle.preprocessing_graph

        logger.info(f"Loaded bundle: {bundle.metadata.model_name}")

        return orch

    @classmethod
    def from_bundles(
        cls,
        bundle_paths: list[str | Path],
        config: PipelineConfig | None = None,
    ) -> InferenceOrchestrator:
        """
        Create orchestrator from multiple bundles.

        Args:
            bundle_paths: List of paths to saved bundles
            config: Optional PipelineConfig

        Returns:
            InferenceOrchestrator with loaded bundles
        """
        from src.inference.bundle import ModelBundle

        orch = cls(config)

        for path in bundle_paths:
            bundle = ModelBundle.load(path)
            orch._bundles[bundle.metadata.model_name] = bundle

            # Extract preprocessing graph from first bundle if available
            if orch._preprocessing_graph is None and bundle.preprocessing_graph is not None:
                orch._preprocessing_graph = bundle.preprocessing_graph

        orch._is_loaded = True
        logger.info(f"Loaded {len(orch._bundles)} bundles")

        return orch

    @classmethod
    def from_training_result(
        cls,
        training_result: Any,  # TrainingRunResult from PHASE_3
        config: PipelineConfig | None = None,
    ) -> InferenceOrchestrator:
        """
        Create orchestrator from PHASE_3 TrainingRunResult.

        This enables direct integration with UnifiedTrainingOrchestrator output.

        Args:
            training_result: TrainingRunResult from UnifiedTrainingOrchestrator
            config: Optional PipelineConfig (uses training_result.config if not provided)

        Returns:
            InferenceOrchestrator with loaded models
        """
        orch = cls(config or training_result.config)

        # Load bundles from training result output directory
        output_dir = Path(training_result.output_dir)
        bundles_dir = output_dir / "bundles"

        if bundles_dir.exists():
            orch._load_bundles_from_dir(bundles_dir)

        # Load ensemble if available
        ensemble_dir = output_dir / "ensemble" / "bundle"
        if ensemble_dir.exists():
            orch._load_ensemble_bundle(ensemble_dir)

        return orch

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        model_name: str | None = None,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        Make predictions using loaded bundle(s).

        Args:
            X: Input features (DataFrame or array)
            model_name: Specific model to use (None = use ensemble or first model)
            calibrate: Whether to apply probability calibration

        Returns:
            PredictionResult with predictions and metadata

        Raises:
            RuntimeError: If no bundles are loaded
            ValueError: If specified model_name is not loaded
        """
        self._validate_loaded()
        start_time = time.perf_counter()

        if model_name:
            # Predict with specific model
            if model_name not in self._bundles:
                raise ValueError(
                    f"Model '{model_name}' not loaded. Available: {list(self._bundles.keys())}"
                )
            bundle = self._bundles[model_name]
            output = bundle.predict(X, calibrate=calibrate)
            is_ensemble = False
        elif self._ensemble_bundle is not None:
            # Predict with ensemble
            output = self._predict_with_ensemble(X, calibrate=calibrate)
            is_ensemble = True
            model_name = "ensemble"
        else:
            # Use first available model
            model_name = next(iter(self._bundles.keys()))
            bundle = self._bundles[model_name]
            output = bundle.predict(X, calibrate=calibrate)
            is_ensemble = False

        inference_time = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            class_predictions=output.class_predictions,
            class_probabilities=output.class_probabilities,
            confidence=output.confidence,
            model_name=model_name,
            horizon=self._get_horizon(),
            inference_time_ms=inference_time,
            is_ensemble=is_ensemble,
        )

    def predict_all(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> dict[str, PredictionResult]:
        """
        Get predictions from all loaded models.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            Dict mapping model_name -> PredictionResult
        """
        self._validate_loaded()

        results = {}
        for model_name in self._bundles:
            results[model_name] = self.predict(X, model_name=model_name, calibrate=calibrate)

        if self._ensemble_bundle is not None:
            results["ensemble"] = self.predict(X, calibrate=calibrate)

        return results

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        model_name: str | None = None,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        End-to-end prediction from raw OHLCV data.

        Applies preprocessing graph to transform raw data to features,
        then makes predictions.

        Args:
            raw_df: Raw OHLCV DataFrame with columns [datetime, open, high, low, close, volume]
            model_name: Specific model to use
            calibrate: Whether to apply calibration

        Returns:
            PredictionResult

        Raises:
            RuntimeError: If no preprocessing graph is available
        """
        self._validate_loaded()

        if self._preprocessing_graph is None:
            raise RuntimeError(
                "No preprocessing graph available. Bundles must include "
                "preprocessing graph for raw data inference. "
                "Use predict() with pre-computed features instead."
            )

        # Transform raw data to features
        features = self._preprocessing_graph.transform(raw_df)

        # Predict
        return self.predict(features, model_name=model_name, calibrate=calibrate)

    def predict_batch(
        self,
        data: pd.DataFrame | Path,
        batch_size: int = 10000,
        model_name: str | None = None,
        output_path: Path | None = None,
        calibrate: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Batch inference for large datasets.

        Processes data in chunks to manage memory efficiently.

        Args:
            data: Input data (DataFrame or path to parquet file)
            batch_size: Samples per batch
            model_name: Specific model to use
            output_path: Optional path to save results
            calibrate: Whether to apply calibration
            show_progress: Whether to log progress

        Returns:
            DataFrame with predictions
        """
        self._validate_loaded()

        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data)

        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_results = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = data.iloc[start_idx:end_idx]

            result = self.predict(batch, model_name=model_name, calibrate=calibrate)
            batch_df = result.to_dataframe()

            # Preserve datetime if available
            if "datetime" in batch.columns:
                batch_df["datetime"] = batch["datetime"].values
            elif hasattr(batch.index, "name") and batch.index.name == "datetime":
                batch_df["datetime"] = batch.index.values

            all_results.append(batch_df)

            if show_progress:
                logger.info(f"Batch {i+1}/{n_batches}: {end_idx}/{n_samples} samples")

        predictions_df = pd.concat(all_results, ignore_index=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_parquet(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

        return predictions_df

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> dict[str, Any]:
        """
        Get predictions with uncertainty estimates from all models.

        Useful for understanding model agreement and prediction reliability.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            Dict with ensemble prediction and uncertainty metrics
        """
        self._validate_loaded()

        if len(self._bundles) < 2:
            result = self.predict(X, calibrate=calibrate)
            return {
                "prediction": result,
                "uncertainty": np.zeros(result.n_samples),
                "agreement": np.ones(result.n_samples),
                "individual_results": {},
            }

        # Get predictions from all models
        all_results = self.predict_all(X, calibrate=calibrate)

        # Stack probabilities for uncertainty calculation
        prob_stack = np.stack(
            [r.class_probabilities for name, r in all_results.items() if name != "ensemble"]
        )

        # Calculate uncertainty as std of probabilities
        uncertainty = np.mean(np.std(prob_stack, axis=0), axis=1)

        # Calculate agreement as proportion of models predicting same class
        pred_stack = np.stack(
            [r.class_predictions for name, r in all_results.items() if name != "ensemble"]
        )
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int) + 1, minlength=3).argmax() - 1,
            axis=0,
            arr=pred_stack,
        )
        agreement = np.mean(pred_stack == mode_predictions, axis=0)

        # Get ensemble or averaged prediction
        if "ensemble" in all_results:
            main_result = all_results["ensemble"]
        else:
            main_result = all_results[next(iter(all_results.keys()))]

        return {
            "prediction": main_result,
            "uncertainty": uncertainty,
            "agreement": agreement,
            "individual_results": {k: v for k, v in all_results.items() if k != "ensemble"},
        }

    def _predict_with_ensemble(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> Any:
        """
        Make predictions using ensemble.

        Args:
            X: Input features
            calibrate: Whether to apply calibration

        Returns:
            PredictionOutput from ensemble
        """
        # If we have a dedicated ensemble bundle, use it directly
        if self._ensemble_bundle is not None and hasattr(self._ensemble_bundle, "predict"):
            return self._ensemble_bundle.predict(X, calibrate=calibrate)

        # Otherwise, collect base model predictions and combine
        base_predictions = {}
        for name, bundle in self._bundles.items():
            output = bundle.predict(X, calibrate=calibrate)
            base_predictions[name] = output.class_probabilities

        # Simple averaging ensemble
        avg_probs = np.mean(list(base_predictions.values()), axis=0)
        class_predictions = np.argmax(avg_probs, axis=1) - 1  # Map to -1, 0, 1
        confidence = np.max(avg_probs, axis=1)

        from src.models.base import PredictionOutput

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={"method": "averaging", "n_models": len(base_predictions)},
        )

    def _load_bundles_from_dir(self, bundles_dir: Path) -> None:
        """Load all bundles from a directory."""
        from src.inference.bundle import ModelBundle

        for bundle_path in bundles_dir.iterdir():
            if bundle_path.is_dir() and (bundle_path / "manifest.json").exists():
                try:
                    bundle = ModelBundle.load(bundle_path)
                    self._bundles[bundle.metadata.model_name] = bundle

                    # Extract preprocessing graph from first bundle
                    if self._preprocessing_graph is None and bundle.preprocessing_graph is not None:
                        self._preprocessing_graph = bundle.preprocessing_graph

                    logger.info(f"Loaded bundle: {bundle.metadata.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load bundle {bundle_path}: {e}")

        self._is_loaded = len(self._bundles) > 0

    def _load_ensemble_bundle(self, ensemble_path: Path) -> None:
        """Load ensemble bundle or meta-learner."""
        try:
            # Try to load as EnsembleBundle first
            try:
                from src.inference.ensemble_bundle import EnsembleBundle

                self._ensemble_bundle = EnsembleBundle.load(ensemble_path)
                logger.info("Loaded ensemble bundle")
                return
            except ImportError:
                pass

            # Try to load as a model bundle
            from src.inference.bundle import ModelBundle

            if (ensemble_path / "manifest.json").exists():
                self._ensemble_bundle = ModelBundle.load(ensemble_path)
                logger.info("Loaded ensemble as model bundle")
                return

            # Try to load meta-learner directly
            from src.models.ensemble import (
                CalibratedMetaLearner,
                MLPMetaLearner,
                RidgeMetaLearner,
                XGBoostMeta,
            )

            # Determine meta-learner type from config
            meta_type = "ridge_meta"
            if self.config is not None:
                meta_type = self.config.meta_learner

            meta_learner_map = {
                "ridge_meta": RidgeMetaLearner,
                "mlp_meta": MLPMetaLearner,
                "xgboost_meta": XGBoostMeta,
                "calibrated_meta": CalibratedMetaLearner,
            }

            meta_class = meta_learner_map.get(meta_type, RidgeMetaLearner)
            self._ensemble_bundle = meta_class()  # type: ignore[abstract]
            self._ensemble_bundle.load(ensemble_path)
            logger.info(f"Loaded meta-learner: {meta_type}")

        except Exception as e:
            logger.warning(f"Failed to load ensemble bundle: {e}")

    def _validate_loaded(self) -> None:
        """Validate that bundles are loaded."""
        if not self._is_loaded and not self._bundles:
            raise RuntimeError(
                "No bundles loaded. Use from_bundle(), from_bundles(), "
                "from_experiment(), or from_training_result() to load bundles."
            )

    def _get_horizon(self) -> int:
        """Get prediction horizon from loaded bundle."""
        if self._bundles:
            first_bundle = next(iter(self._bundles.values()))
            return int(first_bundle.metadata.horizon)
        if self._ensemble_bundle is not None and hasattr(self._ensemble_bundle, "metadata"):
            return int(self._ensemble_bundle.metadata.horizon)
        if self.config is not None and self.config.horizons:
            return int(self.config.horizons[0])
        return 0

    @property
    def loaded_models(self) -> list[str]:
        """Get list of loaded model names."""
        models = list(self._bundles.keys())
        if self._ensemble_bundle is not None:
            models.append("ensemble")
        return models

    @property
    def has_ensemble(self) -> bool:
        """Check if ensemble is loaded."""
        return self._ensemble_bundle is not None

    @property
    def has_preprocessing_graph(self) -> bool:
        """Check if preprocessing graph is available."""
        return self._preprocessing_graph is not None

    @property
    def preprocessing_graph(self) -> Any | None:
        """Get preprocessing graph if available."""
        return self._preprocessing_graph

    def set_preprocessing_graph(self, graph: Any) -> None:
        """
        Set preprocessing graph for raw OHLCV inference.

        Args:
            graph: PreprocessingGraph instance
        """
        self._preprocessing_graph = graph
        logger.info("Preprocessing graph set")

    def validate(self) -> dict[str, Any]:
        """
        Validate orchestrator state.

        Returns:
            Dict with validation results
        """
        issues: list[str] = []

        if not self._bundles:
            issues.append("No model bundles loaded")

        for name, bundle in self._bundles.items():
            bundle_validation = bundle.validate()
            if not bundle_validation["valid"]:
                issues.extend([f"{name}: {i}" for i in bundle_validation["issues"]])

        if self._preprocessing_graph is not None:
            graph_validation = self._preprocessing_graph.validate()
            if not graph_validation["valid"]:
                issues.extend([f"preprocessing_graph: {i}" for i in graph_validation["issues"]])

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "n_models": len(self._bundles),
            "models": list(self._bundles.keys()),
            "has_ensemble": self.has_ensemble,
            "has_preprocessing_graph": self.has_preprocessing_graph,
            "horizon": self._get_horizon(),
        }

    def __repr__(self) -> str:
        return (
            f"InferenceOrchestrator("
            f"models={list(self._bundles.keys())}, "
            f"has_ensemble={self.has_ensemble}, "
            f"has_preprocessing_graph={self.has_preprocessing_graph})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_inference(
    experiment_dir: str | Path,
) -> InferenceOrchestrator:
    """
    Convenience function to load inference from experiment.

    Args:
        experiment_dir: Path to experiment directory containing config.json

    Returns:
        InferenceOrchestrator with loaded bundles
    """
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    config = PipelineConfig.load(config_path)
    return InferenceOrchestrator.from_experiment(config)


def predict_from_bundle(
    bundle_path: str | Path,
    X: pd.DataFrame | np.ndarray,
    calibrate: bool = True,
) -> PredictionResult:
    """
    Convenience function for quick predictions from a bundle.

    Args:
        bundle_path: Path to bundle directory
        X: Input features
        calibrate: Whether to apply calibration

    Returns:
        PredictionResult
    """
    orchestrator = InferenceOrchestrator.from_bundle(bundle_path)
    return orchestrator.predict(X, calibrate=calibrate)


def predict_batch_from_bundle(
    bundle_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Convenience function for batch predictions from a bundle.

    Args:
        bundle_path: Path to bundle directory
        data_path: Path to input parquet file
        output_path: Path for output predictions
        batch_size: Samples per batch

    Returns:
        DataFrame with predictions
    """
    orchestrator = InferenceOrchestrator.from_bundle(bundle_path)
    return orchestrator.predict_batch(
        data=data_path,
        batch_size=batch_size,
        output_path=Path(output_path),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "InferenceOrchestrator",
    "PredictionResult",
    "load_inference",
    "predict_from_bundle",
    "predict_batch_from_bundle",
]
