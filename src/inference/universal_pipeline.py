"""
UniversalInferencePipeline - THE single entry point for all inference.

Wraps ModelBundle and EnsembleBundle into a unified interface that handles:
- Single model inference (predict, predict_from_raw)
- Multi-model inference (predict_all)
- Ensemble inference via meta-learner (predict_ensemble)
- Batch inference (predict_batch)
- Uncertainty estimation via ensemble spread (predict_with_uncertainty)

Construction:
    from src.inference import UniversalInferencePipeline

    # From a single saved bundle
    uip = UniversalInferencePipeline.from_bundle("./bundles/xgb_h20")

    # From multiple saved bundles
    uip = UniversalInferencePipeline.from_bundles([
        "./bundles/xgb_h20",
        "./bundles/lgbm_h20",
    ])

    # From a training experiment result
    uip = UniversalInferencePipeline.from_experiment(config)

    # From a TrainingRunResult (after training)
    uip = UniversalInferencePipeline.from_training_result(run_result)

Prediction:
    result = uip.predict(X)                     # pre-shaped features
    result = uip.predict_from_raw(raw_df)       # raw OHLCV bars
    results = uip.predict_all(raw_df)           # all loaded bundles
    result = uip.predict_ensemble(raw_df)       # ensemble meta-learner
    df = uip.predict_batch(data_iter, 512)      # batched predictions
    result = uip.predict_with_uncertainty(df, 5) # uncertainty via spread
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.types import ScalingSource
from src.inference.bundle import ModelBundle
from src.inference.ensemble_bundle import EnsembleBundle
from src.inference.errors import InferenceError

if TYPE_CHECKING:
    from src.core import PipelineConfig
    from src.core.interfaces import PredictionResult
    from src.models.training.unified_orchestrator import TrainingRunResult

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class UniversalPredictionResult:
    """Result from UniversalInferencePipeline.

    Attributes:
        predictions: PredictionResult from the underlying bundle.
        model_name: Name of the model that produced the result.
        inference_time_ms: Wall-clock time for the prediction in ms.
        scaling_source: Which component handled scaling.
        metadata: Arbitrary extra information.
    """

    predictions: PredictionResult
    model_name: str
    inference_time_ms: float = 0.0
    scaling_source: ScalingSource = ScalingSource.BUNDLE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.predictions.n_samples

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with prediction columns."""
        probs = self.predictions.class_probabilities
        data: dict[str, Any] = {
            "prediction": self.predictions.class_predictions,
            "confidence": self.predictions.confidence,
        }
        if probs.ndim == 2:
            for i in range(probs.shape[1]):
                data[f"prob_class_{i}"] = probs[:, i]
        return pd.DataFrame(data)


# =============================================================================
# UNIVERSAL INFERENCE PIPELINE
# =============================================================================


class UniversalInferencePipeline:
    """
    THE single entry point for all inference in ML Factory.

    Orchestrates ModelBundle and EnsembleBundle into a clean, unified
    interface.  Construction is via class methods; the ``__init__`` is
    intentionally low-level.
    """

    def __init__(
        self,
        bundles: list[ModelBundle] | None = None,
        ensemble_bundle: EnsembleBundle | None = None,
        *,
        scaling_source: ScalingSource = ScalingSource.BUNDLE,
    ) -> None:
        if not bundles and ensemble_bundle is None:
            raise ValueError("At least one ModelBundle or an EnsembleBundle is required.")
        self._bundles: list[ModelBundle] = bundles or []
        self._ensemble_bundle: EnsembleBundle | None = ensemble_bundle
        self._scaling_source = scaling_source

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_bundle(cls, path: str | Path) -> UniversalInferencePipeline:
        """Load a single ModelBundle from *path*.

        Args:
            path: Directory containing a saved ModelBundle.

        Returns:
            Pipeline with one loaded bundle.
        """
        bundle = ModelBundle.load(path)
        scaling = ScalingSource(bundle.metadata.scaling_source)
        return cls([bundle], scaling_source=scaling)

    @classmethod
    def from_bundles(
        cls,
        paths: list[str | Path],
        ensemble_path: str | Path | None = None,
    ) -> UniversalInferencePipeline:
        """Load multiple ModelBundles (and optionally an EnsembleBundle).

        Args:
            paths: Directories for individual ModelBundles.
            ensemble_path: Optional directory for an EnsembleBundle.

        Returns:
            Pipeline with multiple loaded bundles.
        """
        bundles = [ModelBundle.load(p) for p in paths]
        ens = EnsembleBundle.load(ensemble_path) if ensemble_path else None
        return cls(bundles, ensemble_bundle=ens)

    @classmethod
    def from_experiment(
        cls,
        config: PipelineConfig,
    ) -> UniversalInferencePipeline:
        """Reconstruct pipeline from an experiment directory.

        Scans ``config.output_dir / "bundles"`` for saved model bundles
        and an optional ensemble bundle.

        Args:
            config: PipelineConfig whose *output_dir* contains bundles.

        Returns:
            Pipeline loaded from the experiment.
        """
        bundle_dir = Path(config.output_dir) / "bundles"
        if not bundle_dir.is_dir():
            raise FileNotFoundError(
                f"No bundles directory at {bundle_dir}. " "Train models first or check output_dir."
            )

        bundles: list[ModelBundle] = []
        ensemble_bundle: EnsembleBundle | None = None

        for child in sorted(bundle_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / "manifest.json"
            if not manifest.exists():
                continue
            # Distinguish ensemble bundles from model bundles
            if (child / "meta_learner").is_dir():
                try:
                    ensemble_bundle = EnsembleBundle.load(child)
                    logger.info("Loaded ensemble bundle from %s", child)
                except Exception as exc:
                    logger.warning("Failed to load ensemble bundle %s: %s", child, exc)
            else:
                try:
                    bundles.append(ModelBundle.load(child))
                except Exception as exc:
                    logger.warning("Failed to load bundle %s: %s", child, exc)

        if not bundles and ensemble_bundle is None:
            raise InferenceError(f"No valid bundles found in {bundle_dir}.")

        return cls(bundles, ensemble_bundle=ensemble_bundle)

    @classmethod
    def from_training_result(
        cls,
        result: TrainingRunResult,
    ) -> UniversalInferencePipeline:
        """Build pipeline from a fresh TrainingRunResult.

        Scans ``result.output_dir / "bundles"`` for saved bundles.  This
        is a convenience wrapper around :meth:`from_experiment`.

        Args:
            result: TrainingRunResult produced by the training orchestrator.

        Returns:
            Pipeline ready for inference.
        """
        return cls.from_experiment(result.config)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def n_models(self) -> int:
        """Number of individual model bundles loaded."""
        return len(self._bundles)

    @property
    def model_names(self) -> list[str]:
        """Names of all loaded model bundles."""
        return [b.metadata.model_name for b in self._bundles]

    @property
    def has_ensemble(self) -> bool:
        """Whether an ensemble bundle is loaded."""
        return self._ensemble_bundle is not None

    @property
    def primary_bundle(self) -> ModelBundle:
        """First loaded bundle (convenience accessor).

        Raises:
            InferenceError: If no bundles are loaded.
        """
        if not self._bundles:
            raise InferenceError("No model bundles loaded.")
        return self._bundles[0]

    @property
    def scaling_source(self) -> ScalingSource:
        """Active scaling source for the pipeline."""
        return self._scaling_source

    # -----------------------------------------------------------------
    # Core prediction methods
    # -----------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        calibrate: bool = True,
        bundle_index: int = 0,
    ) -> UniversalPredictionResult:
        """Predict from pre-shaped features using a specific bundle.

        Args:
            X: Feature array/DataFrame already shaped for the model.
            calibrate: Apply probability calibration if available.
            bundle_index: Which bundle to use (default: primary).

        Returns:
            UniversalPredictionResult wrapping the bundle's output.
        """
        bundle = self._get_bundle(bundle_index)
        start = time.perf_counter()
        pred = bundle.predict(X, calibrate=calibrate)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return UniversalPredictionResult(
            predictions=pred,
            model_name=bundle.metadata.model_name,
            inference_time_ms=elapsed_ms,
            scaling_source=self._scaling_source,
            metadata={"calibrated": calibrate and bundle.calibrator is not None},
        )

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        *,
        calibrate: bool = True,
        skip_cleaning: bool = False,
        bundle_index: int = 0,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> UniversalPredictionResult:
        """End-to-end prediction from raw OHLCV bars.

        Delegates to ``ModelBundle.predict_from_raw`` which handles
        preprocessing, adapter routing (2D/3D/4D), and scaling.

        Args:
            raw_df: Raw OHLCV DataFrame.
            calibrate: Apply calibration.
            skip_cleaning: Skip resampling if data is already clean.
            bundle_index: Which bundle to use.
            additional_dfs: Extra timeframe DataFrames for 4D models.

        Returns:
            UniversalPredictionResult.
        """
        bundle = self._get_bundle(bundle_index)
        start = time.perf_counter()
        pred = bundle.predict_from_raw(
            raw_df,
            calibrate=calibrate,
            skip_cleaning=skip_cleaning,
            additional_dfs=additional_dfs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return UniversalPredictionResult(
            predictions=pred,
            model_name=bundle.metadata.model_name,
            inference_time_ms=elapsed_ms,
            scaling_source=ScalingSource.PREPROCESSING,
            metadata={
                "calibrated": calibrate and bundle.calibrator is not None,
                "from_raw": True,
            },
        )

    def predict_all(
        self,
        raw_df: pd.DataFrame,
        *,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> list[UniversalPredictionResult]:
        """Get predictions from every loaded ModelBundle.

        Each bundle independently preprocesses + predicts.

        Args:
            raw_df: Raw OHLCV DataFrame.
            calibrate: Apply calibration.
            skip_cleaning: Skip resampling.

        Returns:
            One UniversalPredictionResult per bundle.
        """
        results: list[UniversalPredictionResult] = []
        for idx in range(len(self._bundles)):
            try:
                result = self.predict_from_raw(
                    raw_df,
                    calibrate=calibrate,
                    skip_cleaning=skip_cleaning,
                    bundle_index=idx,
                )
                results.append(result)
            except Exception as exc:
                name = self._bundles[idx].metadata.model_name
                logger.warning("predict_all: bundle '%s' failed: %s", name, exc)
        return results

    def predict_ensemble(
        self,
        raw_df: pd.DataFrame,
        *,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> UniversalPredictionResult:
        """Ensemble prediction via the loaded EnsembleBundle.

        Each base ModelBundle processes ``raw_df`` independently, then
        the EnsembleBundle's meta-learner combines their outputs.

        Args:
            raw_df: Raw OHLCV DataFrame.
            calibrate: Apply calibration.
            skip_cleaning: Skip resampling.

        Returns:
            UniversalPredictionResult from the ensemble.

        Raises:
            InferenceError: If no ensemble bundle is loaded.
        """
        if self._ensemble_bundle is None:
            raise InferenceError(
                "No EnsembleBundle loaded. Use from_bundles() with "
                "ensemble_path or from_experiment() with an experiment "
                "that includes an ensemble."
            )

        start = time.perf_counter()
        pred = self._ensemble_bundle.predict_from_raw(
            raw_df,
            calibrate=calibrate,
            skip_cleaning=skip_cleaning,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return UniversalPredictionResult(
            predictions=pred,
            model_name=self._ensemble_bundle.metadata.meta_learner_name,
            inference_time_ms=elapsed_ms,
            scaling_source=ScalingSource.PREPROCESSING,
            metadata={
                "ensemble": True,
                "n_base_models": self._ensemble_bundle.metadata.n_base_models,
                "base_models": self._ensemble_bundle.metadata.base_model_names,
            },
        )

    def predict_batch(
        self,
        data_iter: Iterable[pd.DataFrame],
        batch_size: int = 512,
        *,
        calibrate: bool = True,
        bundle_index: int = 0,
    ) -> pd.DataFrame:
        """Batched predictions from an iterable of DataFrames.

        Useful for streaming or memory-constrained scenarios.  Each
        element of *data_iter* is a chunk of pre-shaped features.

        Args:
            data_iter: Iterable yielding feature DataFrames/arrays.
            batch_size: Ignored when data_iter already yields chunks;
                retained for API symmetry.
            calibrate: Apply calibration.
            bundle_index: Which bundle to use.

        Returns:
            Concatenated DataFrame of all predictions.
        """
        frames: list[pd.DataFrame] = []
        total_samples = 0
        start = time.perf_counter()

        for chunk in data_iter:
            result = self.predict(
                chunk,
                calibrate=calibrate,
                bundle_index=bundle_index,
            )
            frames.append(result.to_dataframe())
            total_samples += result.n_samples

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "predict_batch: %d samples in %.1f ms",
            total_samples,
            elapsed_ms,
        )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def predict_with_uncertainty(
        self,
        raw_df: pd.DataFrame,
        n_samples: int = 5,
        *,
        skip_cleaning: bool = False,
    ) -> UniversalPredictionResult:
        """Estimate prediction uncertainty via ensemble spread.

        If multiple bundles are loaded, runs each and computes the mean
        and standard deviation of their probability outputs.  The
        ``metadata`` of the result contains ``"prob_std"`` (per-sample
        standard deviation across models) and ``"prob_mean"`` arrays.

        Args:
            raw_df: Raw OHLCV DataFrame.
            n_samples: Maximum number of bundles to sample.  If fewer
                bundles are loaded, all are used.
            skip_cleaning: Skip resampling.

        Returns:
            UniversalPredictionResult whose ``predictions`` reflect the
            mean probabilities and whose ``metadata`` contains spread.
        """
        from src.core.interfaces import PredictionResult

        k = min(n_samples, len(self._bundles))
        if k < 2:
            logger.warning(
                "predict_with_uncertainty needs >=2 bundles; "
                "falling back to single-model prediction."
            )
            return self.predict_from_raw(
                raw_df,
                skip_cleaning=skip_cleaning,
            )

        start = time.perf_counter()

        all_probs: list[np.ndarray] = []
        for idx in range(k):
            try:
                r = self.predict_from_raw(
                    raw_df,
                    skip_cleaning=skip_cleaning,
                    bundle_index=idx,
                )
                all_probs.append(r.predictions.class_probabilities)
            except Exception as exc:
                name = self._bundles[idx].metadata.model_name
                logger.warning("predict_with_uncertainty: '%s' failed: %s", name, exc)

        if not all_probs:
            raise InferenceError("All bundles failed during uncertainty estimation.")

        # Stack → (n_bundles, n_samples, n_classes)
        stacked = np.stack(all_probs, axis=0)
        prob_mean = stacked.mean(axis=0)
        prob_std = stacked.std(axis=0)

        class_predictions = np.argmax(prob_mean, axis=1) - 1  # -1, 0, 1
        confidence = np.max(prob_mean, axis=1)

        elapsed_ms = (time.perf_counter() - start) * 1000

        pred = PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=prob_mean,
            confidence=confidence,
            metadata={"method": "ensemble_spread", "n_models": len(all_probs)},
        )

        return UniversalPredictionResult(
            predictions=pred,
            model_name="ensemble_uncertainty",
            inference_time_ms=elapsed_ms,
            scaling_source=ScalingSource.PREPROCESSING,
            metadata={
                "prob_std": prob_std,
                "prob_mean": prob_mean,
                "n_models_sampled": len(all_probs),
            },
        )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_model_info(self) -> list[dict[str, Any]]:
        """Return metadata for every loaded bundle."""
        info: list[dict[str, Any]] = []
        for b in self._bundles:
            info.append(
                {
                    "name": b.metadata.model_name,
                    "family": b.metadata.model_family,
                    "horizon": b.metadata.horizon,
                    "n_features": b.metadata.n_features,
                    "requires_sequences": b.metadata.requires_sequences,
                    "requires_4d": b.metadata.requires_4d,
                    "has_calibrator": b.metadata.has_calibrator,
                    "has_preprocessing_graph": b.metadata.has_preprocessing_graph,
                    "scaling_source": b.metadata.scaling_source,
                }
            )
        return info

    def validate(self) -> dict[str, Any]:
        """Validate every loaded bundle and ensemble bundle."""
        results: dict[str, Any] = {}
        all_valid = True

        for bundle in self._bundles:
            v = bundle.validate()
            results[bundle.metadata.model_name] = v
            if not v["valid"]:
                all_valid = False

        if self._ensemble_bundle is not None:
            v = self._ensemble_bundle.validate()
            results["ensemble"] = v
            if not v["valid"]:
                all_valid = False

        return {
            "valid": all_valid,
            "n_models": len(self._bundles),
            "has_ensemble": self.has_ensemble,
            "models": results,
        }

    def summary(self) -> str:
        """Human-readable summary of the pipeline."""
        lines = [
            f"UniversalInferencePipeline ({self.n_models} bundles)",
            f"  Scaling source: {self._scaling_source.value}",
        ]
        for b in self._bundles:
            lines.append(
                f"  - {b.metadata.model_name} "
                f"(H{b.metadata.horizon}, {b.metadata.n_features}F, "
                f"rank={'4D' if b.metadata.requires_4d else '3D' if b.metadata.requires_sequences else '2D'})"
            )
        if self._ensemble_bundle is not None:
            em = self._ensemble_bundle.metadata
            lines.append(f"  Ensemble: {em.meta_learner_name} " f"({em.n_base_models} base models)")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_bundle(self, index: int) -> ModelBundle:
        """Retrieve a bundle by index with bounds checking."""
        if not self._bundles:
            raise InferenceError("No model bundles loaded.")
        if index < 0 or index >= len(self._bundles):
            raise InferenceError(
                f"Bundle index {index} out of range " f"(0..{len(self._bundles) - 1})."
            )
        return self._bundles[index]

    def __repr__(self) -> str:
        ens = f", ensemble={self._ensemble_bundle is not None}" if self._ensemble_bundle else ""
        return f"UniversalInferencePipeline(" f"models={self.model_names}{ens})"


__all__ = [
    "UniversalInferencePipeline",
    "UniversalPredictionResult",
]
