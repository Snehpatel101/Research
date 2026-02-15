# Phase 3B: Core Inference — Implementation Plan

**Date:** 2026-02-15
**Author:** Implementation Planner
**Depends on:** Phase 3A (BundleMetadata extensions, TrainerProtocol)
**Scope:** Tasks 3B-1 through 3B-5

---

## Overview

Phase 3B wires adapter routing into the inference path, builds the `UniversalInferencePipeline` to replace `InferencePipeline` + `InferenceOrchestrator`, fixes `EnsembleBundle` path handling, adds MTF inference data generation, and bridges type alignment between training services and inference.

---

## Task 3B-1: Adapter Routing in ModelBundle

**File:** `src/inference/bundle.py`
**Complexity:** MEDIUM (~100 lines added)

### 3B-1a: Add `_apply_adapter()` method

**Insert after line 1077** (after `predict_from_raw` method):

```python
    def _apply_adapter(
        self,
        features_2d: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """
        Reshape 2D feature DataFrame to model-required tensor shape.

        Routes based on BundleMetadata flags:
        - requires_4d → _build_4d_input (multi-stream 4D)
        - requires_sequences → _build_3d_input (sliding window 3D)
        - else → pass through 2D DataFrame

        Args:
            features_2d: 2D DataFrame from preprocessing
            raw_df: Original raw OHLCV (needed for 4D MTF generation)

        Returns:
            2D DataFrame, 3D ndarray, or 4D ndarray depending on model type
        """
        if self.metadata.requires_4d:
            return self._build_4d_input(features_2d, raw_df)
        elif self.metadata.requires_sequences:
            return self._build_3d_input(features_2d)
        else:
            return features_2d
```

### 3B-1b: Add `_build_3d_input()` method

**Insert immediately after `_apply_adapter`:**

```python
    def _build_3d_input(self, features_2d: pd.DataFrame) -> np.ndarray:
        """
        Convert 2D feature DataFrame to 3D sequences via sliding window.

        Uses numpy stride_tricks for zero-copy windowing, matching
        SequenceAdapter._build_sequences() logic but without label extraction
        (inference mode — no labels needed).

        Args:
            features_2d: DataFrame with feature columns

        Returns:
            ndarray of shape (n_sequences, sequence_length, n_features)

        Raises:
            ValueError: If insufficient rows for sequence_length
        """
        seq_len = self.metadata.sequence_length
        features = features_2d[self.feature_columns].values.astype(np.float32)
        n_rows = len(features)

        if n_rows < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows for sequence model "
                f"'{self.metadata.model_name}', got {n_rows}"
            )

        # Sliding window view: (n_windows, n_features, seq_len)
        windows = np.lib.stride_tricks.sliding_window_view(
            features, seq_len, axis=0
        )
        # Transpose to (n_sequences, seq_len, n_features) and copy to own memory
        X = windows.transpose(0, 2, 1).copy()

        logger.debug(
            f"Built 3D input: {X.shape} from {n_rows} rows "
            f"(seq_len={seq_len}, n_features={len(self.feature_columns)})"
        )
        return X
```

### 3B-1c: Add `_build_4d_input()` method

**Insert immediately after `_build_3d_input`:**

```python
    def _build_4d_input(
        self,
        features_2d: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Build 4D tensor for multi-stream models (PatchTST, iTransformer).

        Generates multi-timeframe OHLCV DataFrames from raw 1min data,
        then builds sliding windows per timeframe and stacks into 4D.

        Args:
            features_2d: 2D features (used for alignment/fallback)
            raw_df: Raw 1-minute OHLCV data for MTF generation

        Returns:
            ndarray of shape (n_sequences, n_timeframes, seq_len, n_features)

        Raises:
            ValueError: If raw_df not provided for 4D model
        """
        if raw_df is None:
            raise ValueError(
                f"4D model '{self.metadata.model_name}' requires raw OHLCV data "
                "for multi-timeframe generation. Pass raw_df to predict_from_raw()."
            )

        # Get MTF config from bundle metadata
        mtf_timeframes = self.metadata.extra.get(
            "mtf_timeframes", ["1min", "5min", "15min"]
        )
        seq_len = self.metadata.sequence_length
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        # Generate multi-TF DataFrames
        tf_dfs = self._generate_mtf_dataframes(raw_df, mtf_timeframes)

        # Build 3D windows per timeframe
        tf_windows = []
        min_sequences = float("inf")

        for tf in mtf_timeframes:
            tf_df = tf_dfs[tf]
            # Extract OHLCV columns
            available_cols = [c for c in ohlcv_cols if c in tf_df.columns]
            if len(available_cols) != len(ohlcv_cols):
                raise ValueError(
                    f"Timeframe '{tf}' missing OHLCV columns. "
                    f"Available: {list(tf_df.columns)}"
                )

            features = tf_df[ohlcv_cols].values.astype(np.float32)
            n_rows = len(features)

            if n_rows < seq_len:
                raise ValueError(
                    f"Timeframe '{tf}' has {n_rows} rows, need >= {seq_len}"
                )

            windows = np.lib.stride_tricks.sliding_window_view(
                features, seq_len, axis=0
            )
            # (n_windows, n_features, seq_len) -> (n_windows, seq_len, n_features)
            windows_3d = windows.transpose(0, 2, 1).copy()
            tf_windows.append(windows_3d)
            min_sequences = min(min_sequences, windows_3d.shape[0])

        # Truncate all timeframes to same number of sequences (align to shortest)
        aligned = [w[-int(min_sequences):] for w in tf_windows]

        # Stack: (n_sequences, n_timeframes, seq_len, n_features)
        X_4d = np.stack(aligned, axis=1)

        logger.debug(
            f"Built 4D input: {X_4d.shape} from {len(mtf_timeframes)} timeframes"
        )
        return X_4d
```

### 3B-1d: Update `predict_from_raw()` to chain preprocessing → adapter → predict

**Current code** (`bundle.py:1056-1077`):

```python
    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)
        return self.predict(features, calibrate=calibrate)
```

**Replace with:**

```python
    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """
        End-to-end prediction from raw OHLCV data.

        Pipeline: raw OHLCV → preprocess (2D) → adapter (2D/3D/4D) → predict.

        For sequence models (LSTM, GRU, TCN, etc.), the adapter builds
        sliding windows from the 2D feature DataFrame.

        For multi-stream models (PatchTST, iTransformer), the adapter
        generates multi-timeframe OHLCV from the raw data.

        Args:
            raw_df: DataFrame with raw OHLCV data
            calibrate: Whether to apply probability calibration
            skip_cleaning: If True, skip resampling step

        Returns:
            PredictionResult with predictions and probabilities
        """
        # Step 1: Feature engineering (always outputs 2D)
        features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)

        # Step 2: Adapter routing (2D → 2D/3D/4D based on model type)
        model_input = self._apply_adapter(features, raw_df=raw_df)

        # Step 3: Predict (scaling + model + calibration)
        return self.predict(model_input, calibrate=calibrate)
```

### 3B-1e: Fix `preprocess()` to pass `skip_scaling=True`

**Current code** (`bundle.py:1037-1042`):

```python
        features = self.preprocessing_graph.transform(
            raw_df,
            skip_cleaning=skip_cleaning,
            skip_scaling=False,
        )
```

**Replace with:**

```python
        features = self.preprocessing_graph.transform(
            raw_df,
            skip_cleaning=skip_cleaning,
            skip_scaling=True,  # Bundle.predict() handles scaling
        )
```

**Rationale:** `ModelBundle.predict()` already applies scaling via `self.scaler` (lines 720-752). If `PreprocessingGraph.transform()` also scales, features are double-scaled. Setting `skip_scaling=True` ensures the preprocessing graph only does feature engineering (its core responsibility), while the bundle's scaler does the scaling (same scaler from training).

---

## Task 3B-2: UniversalInferencePipeline

### NEW FILE: `src/inference/errors.py`

**Complete file content:**

```python
"""
Inference error classes.

Custom exceptions for inference pipeline errors with
structured diagnostic information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.contracts import ModelContract
    from src.core.types import DataRank


class InferenceShapeMismatchError(ValueError):
    """Raised when input shape doesn't match model contract."""

    def __init__(
        self,
        model_name: str,
        expected_rank: DataRank,
        actual_shape: tuple[int, ...],
        contract: ModelContract | None = None,
    ) -> None:
        self.model_name = model_name
        self.expected_rank = expected_rank
        self.actual_shape = actual_shape
        self.contract = contract

        details = (
            f"Model '{model_name}' expects {expected_rank.name} input "
            f"(rank {expected_rank.value}), but received shape {actual_shape}."
        )
        if contract is not None:
            details += (
                f" Contract: adapter_id='{contract.adapter_id}', "
                f"sequence_length={contract.sequence_length}, "
                f"min_features={contract.min_features}."
            )
        super().__init__(details)


__all__ = ["InferenceShapeMismatchError"]
```

### NEW FILE: `src/inference/universal_pipeline.py`

**Complete file content:**

```python
"""
UniversalInferencePipeline - THE single entry point for all inference.

Replaces both InferencePipeline and InferenceOrchestrator by:
- Supporting all 3 input modes (raw OHLCV, pre-computed features, pre-shaped tensors)
- Handling adapter routing for all 12 core model families
- Preventing double-scaling via ScalingSource enum
- Providing unified API for single, ensemble, and batch inference

Usage:
    from src.inference.universal_pipeline import UniversalInferencePipeline

    # From experiment directory
    pipeline = UniversalInferencePipeline.from_experiment(config)

    # From single bundle
    pipeline = UniversalInferencePipeline.from_bundle("./bundles/xgb_h20")

    # Raw OHLCV → prediction (all model types)
    result = pipeline.predict_from_raw(raw_ohlcv_df)

    # Pre-computed features → prediction (adapter routing automatic)
    result = pipeline.predict(features_df, model_name="lstm")

    # Pre-shaped tensor → prediction (bypass adapter)
    result = pipeline.predict(X_3d_array, model_name="lstm")
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.inference.errors import InferenceShapeMismatchError
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ScalingSource(str, Enum):
    """Controls which scaler is applied during inference.

    Resolves double-scaling risk by making scaling source explicit.

    BUNDLE: Use bundle's scaler (fitted during training pipeline Stage 7.5).
            This is the default and correct choice for most cases.
    PREPROCESSING: Scaling already applied by PreprocessingGraph.transform().
                   Pipeline should NOT apply bundle scaler.
    NONE: Caller pre-scaled the data. No scaling applied.
    """

    BUNDLE = "bundle"
    PREPROCESSING = "preprocessing"
    NONE = "none"


# =============================================================================
# RESULT CLASSES
# =============================================================================


class InferenceResult:
    """Result from a single model prediction with timing metadata."""

    __slots__ = (
        "predictions",
        "inference_time_ms",
        "model_name",
        "horizon",
        "n_samples",
        "metadata",
    )

    def __init__(
        self,
        predictions: PredictionResult,
        inference_time_ms: float,
        model_name: str,
        horizon: int,
        n_samples: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.predictions = predictions
        self.inference_time_ms = inference_time_ms
        self.model_name = model_name
        self.horizon = horizon
        self.n_samples = n_samples
        self.metadata = metadata or {}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        return pd.DataFrame(
            {
                "prediction": self.predictions.class_predictions,
                "prob_short": self.predictions.class_probabilities[:, 0],
                "prob_neutral": self.predictions.class_probabilities[:, 1],
                "prob_long": self.predictions.class_probabilities[:, 2],
                "confidence": self.predictions.confidence,
            }
        )


class EnsembleInferenceResult:
    """Result from ensemble prediction combining multiple models."""

    __slots__ = (
        "predictions",
        "individual_results",
        "voting_method",
        "inference_time_ms",
    )

    def __init__(
        self,
        predictions: PredictionResult,
        individual_results: dict[str, InferenceResult],
        voting_method: str,
        inference_time_ms: float,
    ) -> None:
        self.predictions = predictions
        self.individual_results = individual_results
        self.voting_method = voting_method
        self.inference_time_ms = inference_time_ms

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ensemble predictions to DataFrame."""
        df = pd.DataFrame(
            {
                "ensemble_prediction": self.predictions.class_predictions,
                "ensemble_prob_short": self.predictions.class_probabilities[:, 0],
                "ensemble_prob_neutral": self.predictions.class_probabilities[:, 1],
                "ensemble_prob_long": self.predictions.class_probabilities[:, 2],
                "ensemble_confidence": self.predictions.confidence,
            }
        )
        for name, result in self.individual_results.items():
            df[f"{name}_pred"] = result.predictions.class_predictions
            df[f"{name}_conf"] = result.predictions.confidence
        return df


# =============================================================================
# UNIVERSAL INFERENCE PIPELINE
# =============================================================================


class UniversalInferencePipeline:
    """
    THE single entry point for all inference in ML Factory.

    Supports 3 input modes:
      Mode 1: Raw OHLCV → features → adapt → predict  (predict_from_raw)
      Mode 2: Pre-computed features (2D DataFrame) → adapt → predict  (predict)
      Mode 3: Pre-shaped tensors (ndarray) → predict directly  (predict)

    Handles all 12 core model families via contract-driven adapter routing.

    Key design invariant:
      Pipeline calls bundle.model.predict() directly (NOT bundle.predict())
      to control scaling timing and prevent double-scaling.
    """

    def __init__(
        self,
        bundles: dict[str, ModelBundle],
        ensemble_bundle: Any | None = None,
        preprocessing_graph: Any | None = None,
        config: Any | None = None,
        scaling_source: ScalingSource = ScalingSource.BUNDLE,
    ) -> None:
        """
        Initialize UniversalInferencePipeline.

        Args:
            bundles: Dict mapping model_name -> ModelBundle
            ensemble_bundle: Optional EnsembleBundle for meta-learner
            preprocessing_graph: Optional PreprocessingGraph for raw data
            config: Optional PipelineConfig
            scaling_source: Controls which scaler is applied
        """
        if not bundles:
            raise RuntimeError(
                "No bundles provided. Use from_bundle(), from_bundles(), "
                "from_experiment(), or from_training_result() to create pipeline."
            )

        self._bundles = bundles
        self._ensemble_bundle = ensemble_bundle
        self._preprocessing_graph = preprocessing_graph
        self._config = config
        self._scaling_source = scaling_source

    # =========================================================================
    # CLASS METHODS (Constructors)
    # =========================================================================

    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        config: Any | None = None,
    ) -> UniversalInferencePipeline:
        """Load pipeline from a single bundle."""
        bundle = ModelBundle.load(path)
        preprocessing_graph = bundle.preprocessing_graph
        return cls(
            bundles={bundle.metadata.model_name: bundle},
            preprocessing_graph=preprocessing_graph,
            config=config,
        )

    @classmethod
    def from_bundles(
        cls,
        paths: list[str | Path],
        config: Any | None = None,
    ) -> UniversalInferencePipeline:
        """Load pipeline from multiple bundles."""
        bundles: dict[str, ModelBundle] = {}
        preprocessing_graph = None

        for path in paths:
            bundle = ModelBundle.load(path)
            bundles[bundle.metadata.model_name] = bundle
            if preprocessing_graph is None and bundle.preprocessing_graph is not None:
                preprocessing_graph = bundle.preprocessing_graph

        return cls(
            bundles=bundles,
            preprocessing_graph=preprocessing_graph,
            config=config,
        )

    @classmethod
    def from_experiment(
        cls,
        config: Any,
        load_ensemble: bool = True,
    ) -> UniversalInferencePipeline:
        """
        Load all bundles from experiment output directory.

        Args:
            config: PipelineConfig with output_dir
            load_ensemble: Whether to load ensemble bundle
        """
        output_dir = Path(config.output_dir)
        bundles_dir = output_dir / "bundles"
        bundles: dict[str, ModelBundle] = {}
        preprocessing_graph = None
        ensemble_bundle = None

        if bundles_dir.exists():
            for bundle_path in bundles_dir.iterdir():
                if bundle_path.is_dir() and (bundle_path / "manifest.json").exists():
                    try:
                        bundle = ModelBundle.load(bundle_path)
                        bundles[bundle.metadata.model_name] = bundle
                        if preprocessing_graph is None and bundle.preprocessing_graph is not None:
                            preprocessing_graph = bundle.preprocessing_graph
                    except Exception as e:
                        logger.warning(f"Failed to load bundle {bundle_path}: {e}")

        if load_ensemble:
            ensemble_dir = output_dir / "ensemble" / "bundle"
            if ensemble_dir.exists():
                try:
                    from src.inference.ensemble_bundle import EnsembleBundle

                    ensemble_bundle = EnsembleBundle.load(ensemble_dir)
                except Exception as e:
                    logger.warning(f"Failed to load ensemble bundle: {e}")

        if not bundles:
            raise RuntimeError(f"No bundles found in {bundles_dir}")

        return cls(
            bundles=bundles,
            ensemble_bundle=ensemble_bundle,
            preprocessing_graph=preprocessing_graph,
            config=config,
        )

    @classmethod
    def from_training_result(
        cls,
        training_result: Any,
        config: Any | None = None,
    ) -> UniversalInferencePipeline:
        """Load from TrainingRunResult output directory."""
        effective_config = config or training_result.config
        output_dir = Path(training_result.output_dir)
        bundles_dir = output_dir / "bundles"
        bundles: dict[str, ModelBundle] = {}
        preprocessing_graph = None

        if bundles_dir.exists():
            for bundle_path in bundles_dir.iterdir():
                if bundle_path.is_dir() and (bundle_path / "manifest.json").exists():
                    try:
                        bundle = ModelBundle.load(bundle_path)
                        bundles[bundle.metadata.model_name] = bundle
                        if preprocessing_graph is None and bundle.preprocessing_graph is not None:
                            preprocessing_graph = bundle.preprocessing_graph
                    except Exception as e:
                        logger.warning(f"Failed to load bundle {bundle_path}: {e}")

        if not bundles:
            raise RuntimeError(f"No bundles found in {bundles_dir}")

        return cls(
            bundles=bundles,
            preprocessing_graph=preprocessing_graph,
            config=effective_config,
        )

    # =========================================================================
    # CORE PREDICTION METHODS
    # =========================================================================

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        model_name: str | None = None,
        calibrate: bool = True,
    ) -> InferenceResult:
        """
        Predict from pre-computed features (Mode 2) or pre-shaped tensors (Mode 3).

        Routing:
        - If X is ndarray: Mode 3 — skip adapter, predict directly
        - If X is DataFrame: Mode 2 — route through adapter based on model contract

        Args:
            X: Input features (DataFrame for Mode 2, ndarray for Mode 3)
            model_name: Specific model (None = first available)
            calibrate: Whether to apply calibration

        Returns:
            InferenceResult with predictions and timing
        """
        model_name = self._resolve_model_name(model_name)
        bundle = self._bundles[model_name]
        return self._predict_single(bundle, X, calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        model_name: str | None = None,
        calibrate: bool = True,
    ) -> InferenceResult:
        """
        End-to-end: Raw OHLCV → features → adapt → predict (Mode 1).

        Args:
            raw_df: Raw OHLCV DataFrame
            model_name: Specific model (None = first available)
            calibrate: Whether to apply calibration

        Returns:
            InferenceResult
        """
        if self._preprocessing_graph is None:
            raise RuntimeError(
                "No preprocessing graph available. Bundles must include a "
                "preprocessing graph for raw data inference. "
                "Use predict() with pre-computed features instead."
            )

        model_name = self._resolve_model_name(model_name)
        bundle = self._bundles[model_name]

        # Step 1: Feature engineering (always 2D output, skip scaling)
        features = self._preprocessing_graph.transform(
            raw_df,
            skip_scaling=True,  # Bundle scaler handles scaling
        )

        # Step 2: Filter to bundle's feature columns
        available_cols = [c for c in bundle.feature_columns if c in features.columns]
        if len(available_cols) < len(bundle.feature_columns):
            missing = set(bundle.feature_columns) - set(available_cols)
            logger.warning(
                f"Missing {len(missing)} feature columns: {list(missing)[:5]}..."
            )
        features = features[available_cols]

        # Step 3: Adapter routing (2D → 2D/3D/4D)
        adapted = self._adapt_input(features, model_name, bundle, raw_df=raw_df)

        # Step 4: Scale → predict → calibrate
        return self._predict_from_adapted(bundle, adapted, calibrate)

    def predict_all(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> dict[str, InferenceResult]:
        """Get predictions from all loaded models."""
        results = {}
        for model_name, bundle in self._bundles.items():
            results[model_name] = self._predict_single(bundle, X, calibrate)
        return results

    def predict_ensemble(
        self,
        X: pd.DataFrame | np.ndarray,
        method: str = "soft_vote",
        weights: list[float] | None = None,
        calibrate: bool = True,
    ) -> EnsembleInferenceResult:
        """
        Ensemble prediction combining all models.

        If EnsembleBundle is available, uses meta-learner.
        Otherwise, falls back to soft/hard/weighted voting.
        """
        start_time = time.perf_counter()

        # Get individual predictions
        individual = self.predict_all(X, calibrate=calibrate)

        # If meta-learner ensemble available, use it
        if self._ensemble_bundle is not None:
            base_preds = {
                name: result.predictions.class_probabilities
                for name, result in individual.items()
            }
            combined_output = self._ensemble_bundle.predict(base_preds, calibrate=calibrate)
        else:
            # Voting fallback
            combined_output = self._vote(individual, method, weights)

        total_time = (time.perf_counter() - start_time) * 1000

        return EnsembleInferenceResult(
            predictions=combined_output,
            individual_results=individual,
            voting_method=method,
            inference_time_ms=total_time,
        )

    def predict_batch(
        self,
        data: pd.DataFrame | Path,
        batch_size: int = 10000,
        model_name: str | None = None,
        output_path: Path | None = None,
        calibrate: bool = True,
    ) -> pd.DataFrame:
        """Batch inference for large datasets."""
        if isinstance(data, (str, Path)):
            data = pd.read_parquet(data)

        model_name = self._resolve_model_name(model_name)
        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_results = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = data.iloc[start_idx:end_idx]

            result = self.predict(batch, model_name=model_name, calibrate=calibrate)
            batch_df = result.to_dataframe()

            if "datetime" in batch.columns:
                batch_df["datetime"] = batch["datetime"].values
            elif hasattr(batch.index, "name") and batch.index.name == "datetime":
                batch_df["datetime"] = batch.index.values

            all_results.append(batch_df)
            logger.info(f"Batch {i + 1}/{n_batches}: {end_idx}/{n_samples} samples")

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
        """Predictions with uncertainty from model disagreement."""
        if len(self._bundles) < 2:
            model_name = next(iter(self._bundles))
            result = self.predict(X, model_name=model_name, calibrate=calibrate)
            return {
                "prediction": result,
                "uncertainty": np.zeros(result.n_samples),
                "agreement": np.ones(result.n_samples),
                "individual_results": {},
            }

        all_results = self.predict_all(X, calibrate=calibrate)

        prob_stack = np.stack(
            [r.predictions.class_probabilities for r in all_results.values()]
        )
        uncertainty = np.mean(np.std(prob_stack, axis=0), axis=1)

        pred_stack = np.stack(
            [r.predictions.class_predictions for r in all_results.values()]
        )
        mode_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int) + 1, minlength=3).argmax() - 1,
            axis=0,
            arr=pred_stack,
        )
        agreement = np.mean(pred_stack == mode_predictions, axis=0)

        first_name = next(iter(all_results))
        return {
            "prediction": all_results[first_name],
            "uncertainty": uncertainty,
            "agreement": agreement,
            "individual_results": all_results,
        }

    # =========================================================================
    # INTERNAL: Adapter Routing
    # =========================================================================

    def _adapt_input(
        self,
        X: pd.DataFrame,
        model_name: str,
        bundle: ModelBundle,
        raw_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Transform 2D DataFrame to model-expected shape.

        Routes based on BundleMetadata:
        - requires_4d → _build_4d (multi-stream windowing)
        - requires_sequences → _build_3d (sliding window)
        - else → extract columns as float32 array

        Returns:
            2D/3D/4D ndarray ready for scaling + prediction
        """
        if bundle.metadata.requires_4d:
            return bundle._build_4d_input(X, raw_df)
        elif bundle.metadata.requires_sequences:
            return bundle._build_3d_input(X)
        else:
            # Tabular: extract columns as array
            return X[bundle.feature_columns].values.astype(np.float32)

    # =========================================================================
    # INTERNAL: Scaling
    # =========================================================================

    def _apply_scaling(
        self,
        X: np.ndarray,
        bundle: ModelBundle,
    ) -> np.ndarray:
        """
        Apply exactly one scaling step based on scaling_source.

        ScalingSource.BUNDLE: Use bundle.scaler (default — training scaler)
        ScalingSource.PREPROCESSING: Already scaled by PreprocessingGraph, skip
        ScalingSource.NONE: Caller pre-scaled, skip

        For 3D/4D: reshape → scale → reshape back.
        """
        if self._scaling_source != ScalingSource.BUNDLE:
            return X
        if bundle.scaler is None:
            return X

        if X.ndim == 2:
            return bundle.scaler.transform(X)
        elif X.ndim in (3, 4):
            orig_shape = X.shape
            X_flat = X.reshape(-1, orig_shape[-1])
            scaler_features = getattr(bundle.scaler, "n_features_in_", None)
            if scaler_features and scaler_features != orig_shape[-1]:
                raise ValueError(
                    f"Scaler expects {scaler_features} features but input has "
                    f"{orig_shape[-1]} features per slice."
                )
            X_scaled = bundle.scaler.transform(X_flat)
            return X_scaled.reshape(orig_shape)
        return X

    # =========================================================================
    # INTERNAL: Single Prediction
    # =========================================================================

    def _predict_single(
        self,
        bundle: ModelBundle,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool,
    ) -> InferenceResult:
        """
        Internal single-model prediction with adapter routing.

        Flow:
        1. If X is DataFrame → adapt to correct shape
        2. If X is ndarray → use directly (Mode 3)
        3. Apply scaling (respecting scaling_source)
        4. model.predict(X_shaped) → PredictionResult
        5. Apply calibration if requested
        6. Wrap in InferenceResult with timing
        """
        start_time = time.perf_counter()

        if isinstance(X, pd.DataFrame):
            # Mode 2: DataFrame → adapt → scaled array
            X_adapted = self._adapt_input(X, bundle.metadata.model_name, bundle)
        else:
            # Mode 3: Pre-shaped tensor
            X_adapted = np.asarray(X, dtype=np.float32)

        # Validate shape against metadata
        self._validate_shape(X_adapted, bundle)

        # Scale
        X_scaled = self._apply_scaling(X_adapted, bundle)

        # Predict via model directly (NOT bundle.predict — we control scaling)
        output = bundle.model.predict(X_scaled)

        # Calibrate
        if calibrate and bundle.calibrator is not None:
            output = bundle._apply_calibration(output)

        inference_time = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            predictions=output,
            inference_time_ms=inference_time,
            model_name=bundle.metadata.model_name,
            horizon=bundle.metadata.horizon,
            n_samples=output.n_samples,
            metadata={
                "calibrated": calibrate and bundle.calibrator is not None,
                "model_family": bundle.metadata.model_family,
                "scaling_source": self._scaling_source.value,
            },
        )

    def _predict_from_adapted(
        self,
        bundle: ModelBundle,
        X_adapted: np.ndarray,
        calibrate: bool,
    ) -> InferenceResult:
        """Predict from already-adapted input (post-adapter routing)."""
        start_time = time.perf_counter()

        self._validate_shape(X_adapted, bundle)
        X_scaled = self._apply_scaling(X_adapted, bundle)
        output = bundle.model.predict(X_scaled)

        if calibrate and bundle.calibrator is not None:
            output = bundle._apply_calibration(output)

        inference_time = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            predictions=output,
            inference_time_ms=inference_time,
            model_name=bundle.metadata.model_name,
            horizon=bundle.metadata.horizon,
            n_samples=output.n_samples,
            metadata={
                "calibrated": calibrate and bundle.calibrator is not None,
                "model_family": bundle.metadata.model_family,
                "scaling_source": self._scaling_source.value,
            },
        )

    # =========================================================================
    # INTERNAL: Validation
    # =========================================================================

    def _validate_shape(self, X: np.ndarray, bundle: ModelBundle) -> None:
        """Validate input array shape against bundle metadata."""
        meta = bundle.metadata

        if meta.requires_4d:
            if X.ndim != 4:
                from src.core.types import DataRank

                raise InferenceShapeMismatchError(
                    model_name=meta.model_name,
                    expected_rank=DataRank.MULTI_TF_4D,
                    actual_shape=X.shape,
                )
        elif meta.requires_sequences:
            if X.ndim != 3:
                from src.core.types import DataRank

                raise InferenceShapeMismatchError(
                    model_name=meta.model_name,
                    expected_rank=DataRank.SEQUENCE_3D,
                    actual_shape=X.shape,
                )
        else:
            if X.ndim != 2:
                from src.core.types import DataRank

                raise InferenceShapeMismatchError(
                    model_name=meta.model_name,
                    expected_rank=DataRank.TABULAR_2D,
                    actual_shape=X.shape,
                )

    # =========================================================================
    # INTERNAL: Voting
    # =========================================================================

    def _vote(
        self,
        results: dict[str, InferenceResult],
        method: str,
        weights: list[float] | None,
    ) -> PredictionResult:
        """Combine predictions via voting."""
        if method == "hard_vote":
            return self._hard_vote(results, weights)
        else:
            return self._soft_vote(results, weights)

    def _soft_vote(
        self,
        results: dict[str, InferenceResult],
        weights: list[float] | None,
    ) -> PredictionResult:
        """Average probabilities across models."""
        result_list = list(results.values())
        n_models = len(result_list)

        if weights is None:
            weights_arr = np.ones(n_models) / n_models
        else:
            weights_arr = np.array(weights) / sum(weights)

        n_samples = result_list[0].n_samples
        n_classes = result_list[0].predictions.n_classes
        avg_probs = np.zeros((n_samples, n_classes))

        for result, w in zip(result_list, weights_arr, strict=False):
            avg_probs += w * result.predictions.class_probabilities

        class_predictions = np.argmax(avg_probs, axis=1) - 1
        confidence = np.max(avg_probs, axis=1)

        return PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={"method": "soft_vote", "n_models": n_models},
        )

    def _hard_vote(
        self,
        results: dict[str, InferenceResult],
        weights: list[float] | None,
    ) -> PredictionResult:
        """Majority vote on class predictions."""
        result_list = list(results.values())
        n_models = len(result_list)
        weight_values = weights or [1.0] * n_models

        n_samples = result_list[0].n_samples
        n_classes = result_list[0].predictions.n_classes

        vote_counts = np.zeros((n_samples, n_classes))
        for result, w in zip(result_list, weight_values, strict=False):
            preds = result.predictions.class_predictions + 1  # Map to 0,1,2
            for i, pred in enumerate(preds):
                vote_counts[i, int(pred)] += w

        majority_class = np.argmax(vote_counts, axis=1)
        class_predictions = majority_class - 1

        confidence = np.max(vote_counts, axis=1) / np.sum(vote_counts, axis=1)

        avg_probs = np.zeros((n_samples, n_classes))
        for result in result_list:
            avg_probs += result.predictions.class_probabilities
        avg_probs /= n_models

        return PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={"method": "hard_vote", "n_models": n_models},
        )

    # =========================================================================
    # INTERNAL: Helpers
    # =========================================================================

    def _resolve_model_name(self, model_name: str | None) -> str:
        """Resolve model name to a loaded bundle key."""
        if model_name is not None:
            if model_name not in self._bundles:
                raise ValueError(
                    f"Model '{model_name}' not loaded. "
                    f"Available: {list(self._bundles.keys())}"
                )
            return model_name
        return next(iter(self._bundles))

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def loaded_models(self) -> list[str]:
        """List of loaded model names."""
        return list(self._bundles.keys())

    @property
    def has_ensemble(self) -> bool:
        """Whether ensemble bundle is loaded."""
        return self._ensemble_bundle is not None

    @property
    def has_preprocessing_graph(self) -> bool:
        """Whether preprocessing graph is available."""
        return self._preprocessing_graph is not None

    def validate(self) -> dict[str, Any]:
        """Validate pipeline state."""
        issues: list[str] = []

        if not self._bundles:
            issues.append("No model bundles loaded")

        for name, bundle in self._bundles.items():
            result = bundle.validate()
            if not result["valid"]:
                issues.extend([f"{name}: {i}" for i in result["issues"]])

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "n_models": len(self._bundles),
            "models": list(self._bundles.keys()),
            "has_ensemble": self.has_ensemble,
            "has_preprocessing_graph": self.has_preprocessing_graph,
            "scaling_source": self._scaling_source.value,
        }

    def __repr__(self) -> str:
        return (
            f"UniversalInferencePipeline("
            f"models={list(self._bundles.keys())}, "
            f"has_ensemble={self.has_ensemble}, "
            f"scaling={self._scaling_source.value})"
        )


__all__ = [
    "UniversalInferencePipeline",
    "ScalingSource",
    "InferenceResult",
    "EnsembleInferenceResult",
]
```

---

## Task 3B-3: EnsembleBundle Fixes

**File:** `src/inference/ensemble_bundle.py`

### 3B-3a: Fix `save()` to store relative paths

**Current code** (`ensemble_bundle.py:442-452`):

```python
        # Save base bundle paths
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        with open(bundles_path, "w") as f:
            json.dump(
                {
                    "paths": [str(p) for p in self.base_bundle_paths],
                    "model_names": self.metadata.base_model_names,
                },
                f,
                indent=2,
            )
```

**Replace with:**

```python
        # Save base bundle paths (relative to ensemble bundle root)
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        relative_paths = []
        for p in self.base_bundle_paths:
            try:
                rel = Path(p).relative_to(path.parent)
                relative_paths.append(str(rel))
            except ValueError:
                # Path not relative to ensemble parent — store absolute as fallback
                relative_paths.append(str(p))
        with open(bundles_path, "w") as f:
            json.dump(
                {
                    "paths": relative_paths,
                    "model_names": self.metadata.base_model_names,
                },
                f,
                indent=2,
            )
```

### 3B-3b: Fix `load()` to resolve relative paths with absolute fallback

**Current code** (`ensemble_bundle.py:538-543`):

```python
        # Load base bundle paths
        base_bundle_paths: list[Path] = []
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        if bundles_path.exists():
            with open(bundles_path) as f:
                base_bundle_paths = [Path(p) for p in json.load(f).get("paths", [])]
```

**Replace with:**

```python
        # Load base bundle paths (resolve relative to ensemble parent)
        base_bundle_paths: list[Path] = []
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        if bundles_path.exists():
            with open(bundles_path) as f:
                raw_paths = json.load(f).get("paths", [])
            for p_str in raw_paths:
                p = Path(p_str)
                if not p.is_absolute():
                    # Resolve relative to ensemble bundle's parent directory
                    p = (path.parent / p).resolve()
                if p.exists():
                    base_bundle_paths.append(p)
                else:
                    # Try absolute path as-is (backward compat)
                    base_bundle_paths.append(Path(p_str))
                    logger.warning(f"Base bundle path not found: {p}")
```

### 3B-3c: Add `predict_from_raw()` to EnsembleBundle

**Insert after `predict_from_base_features` method** (after line 698):

```python
    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
    ) -> Any:
        """
        End-to-end prediction from raw OHLCV data.

        Loads base bundles, calls predict_from_raw on each (which handles
        adapter routing internally), then combines via meta-learner.

        Args:
            raw_df: Raw OHLCV DataFrame
            calibrate: Whether to apply calibration

        Returns:
            PredictionResult from ensemble

        Raises:
            ValueError: If base bundles not available
        """
        self._ensure_base_bundles_loaded()

        if not self._base_bundles:
            raise ValueError(
                "No base bundles loaded. Ensure base_bundle_paths are valid."
            )

        # Get predictions from each base model via predict_from_raw
        # (each bundle handles its own adapter routing)
        base_predictions: dict[str, np.ndarray] = {}

        for model_name, bundle in self._base_bundles.items():
            try:
                output = bundle.predict_from_raw(raw_df, calibrate=False)
                base_predictions[model_name] = output.class_probabilities
            except Exception as e:
                logger.warning(
                    f"Base model '{model_name}' predict_from_raw failed: {e}"
                )

        if not base_predictions:
            raise ValueError("All base model predictions failed")

        # Combine with meta-learner
        return self.predict(base_predictions, calibrate=calibrate)
```

---

## Task 3B-4: MTF Inference Data Generation

**File:** `src/inference/bundle.py`

### Add `_generate_mtf_dataframes()` to ModelBundle

**Insert after `_build_4d_input` method:**

```python
    def _generate_mtf_dataframes(
        self,
        raw_1min_df: pd.DataFrame,
        timeframes: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Generate multi-timeframe OHLCV DataFrames from 1-minute data.

        Resamples raw 1-minute OHLCV bars to each requested timeframe
        using standard OHLCV aggregation rules.

        Args:
            raw_1min_df: DataFrame with 1-minute OHLCV bars
            timeframes: List of timeframe strings (e.g., ["1min", "5min", "15min"])

        Returns:
            Dict mapping timeframe -> resampled OHLCV DataFrame
        """
        tf_dfs: dict[str, pd.DataFrame] = {}

        # Ensure datetime index for resampling
        df = raw_1min_df.copy()
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "raw_df must have a 'datetime' column or DatetimeIndex "
                "for multi-timeframe resampling"
            )

        # Freq mapping
        freq_map = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "1h": "1h",
        }

        for tf in timeframes:
            if tf == "1min":
                tf_dfs[tf] = df.reset_index()
            else:
                pandas_freq = freq_map.get(tf, tf)
                resampled = df.resample(pandas_freq).agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                ).dropna()
                tf_dfs[tf] = resampled.reset_index()

        return tf_dfs
```

---

## Task 3B-5: Type Alignment Bridge

**File:** `src/models/training/services/ensemble_service.py`

### 3B-5a: Add `to_ensemble_result()` bridge function

**Insert at end of file, before `__all__`:**

```python
def to_ensemble_result(
    service_result: EnsembleServiceResult,
    config: PipelineConfig,
) -> Any:
    """
    Convert EnsembleServiceResult to EnsembleResult for EnsembleBundle.

    Bridges the type gap between the training service output and the
    inference bundle input format.

    Args:
        service_result: Result from EnsembleService.build_ensemble()
        config: PipelineConfig for horizon/symbol extraction

    Returns:
        Object compatible with EnsembleBundle.from_ensemble_result()
    """
    from dataclasses import dataclass, field as dataclass_field
    from typing import Any as AnyType

    @dataclass
    class _EnsembleResultBridge:
        """Minimal bridge matching EnsembleResult interface."""

        meta_learner_name: str
        base_model_names: list[str]
        n_base_models: int
        aligned_oof: Any
        stacking_dataset: Any
        coverage: float
        alignment_offset: int
        metrics: dict[str, AnyType] = dataclass_field(default_factory=dict)
        ensemble: AnyType = None

    base_model_names = []
    if service_result.aligned_oof is not None:
        base_model_names = service_result.aligned_oof.model_names

    return _EnsembleResultBridge(
        meta_learner_name=config.meta_learner,
        base_model_names=base_model_names,
        n_base_models=len(base_model_names),
        aligned_oof=service_result.aligned_oof,
        stacking_dataset=service_result.stacking_dataset,
        coverage=(
            service_result.aligned_oof.coverage
            if service_result.aligned_oof
            else 1.0
        ),
        alignment_offset=0,
        metrics=service_result.ensemble_metrics,
        ensemble=service_result.meta_learner,
    )
```

Update `__all__`:

```python
__all__ = ["EnsembleService", "EnsembleRequest", "EnsembleServiceResult", "to_ensemble_result"]
```

### 3B-5b: Update `src/inference/__init__.py`

**Add to imports** (after line 134):

```python
from src.inference.universal_pipeline import (
    UniversalInferencePipeline,
    ScalingSource,
    InferenceResult as UIPInferenceResult,
    EnsembleInferenceResult,
)
from src.inference.errors import InferenceShapeMismatchError
```

**Add to `__all__`** (after line 193):

```python
    # UniversalInferencePipeline (Phase 3B)
    "UniversalInferencePipeline",
    "ScalingSource",
    "UIPInferenceResult",
    "EnsembleInferenceResult",
    "InferenceShapeMismatchError",
```

---

## Double-Scaling Prevention Summary

The double-scaling problem is prevented at 3 levels:

| Level | Mechanism | Where |
|-------|-----------|-------|
| **PreprocessingGraph** | `skip_scaling=True` in `predict_from_raw()` | `bundle.py:1038-1042` (3B-1e) |
| **UniversalInferencePipeline** | `ScalingSource` enum controls single scaling point | `universal_pipeline.py:_apply_scaling()` |
| **ModelBundle.predict()** | Unchanged — applies `self.scaler` once | `bundle.py:720-752` (existing) |

**Flow with UIP (recommended path):**
```
raw_df → PreprocessingGraph.transform(skip_scaling=True)  → 2D unscaled features
       → _adapt_input()                                    → 2D/3D/4D unscaled
       → _apply_scaling(ScalingSource.BUNDLE)              → scaled
       → bundle.model.predict()                            → PredictionResult
```

**Flow with ModelBundle.predict_from_raw (backward compat):**
```
raw_df → PreprocessingGraph.transform(skip_scaling=True)  → 2D unscaled features
       → _apply_adapter()                                  → 2D/3D/4D unscaled
       → bundle.predict()                                  → scaled + predicted
```

---

## Adapter Routing Decision Table

| BundleMetadata Flags | Adapter Path | Output Shape |
|---------------------|--------------|--------------|
| `requires_4d=True` | `_build_4d_input()` → MTF resample → per-TF windowing → stack | `(n, tf, seq, feat)` |
| `requires_sequences=True` | `_build_3d_input()` → sliding_window_view | `(n, seq, feat)` |
| Both `False` | Extract `feature_columns` as float32 array | `(n, feat)` |

**Models per path:**
- **2D (tabular):** xgboost, lightgbm, catboost, random_forest, logistic, svm
- **3D (sequence):** lstm, gru, tcn, transformer, tft, nbeats, inceptiontime, resnet1d
- **4D (multi-stream):** patchtst, itransformer

---

## Validation Commands

```bash
# Verify adapter routing exists
python -c "
from src.inference.bundle import ModelBundle
print('_apply_adapter:', hasattr(ModelBundle, '_apply_adapter'))
print('_build_3d_input:', hasattr(ModelBundle, '_build_3d_input'))
print('_build_4d_input:', hasattr(ModelBundle, '_build_4d_input'))
print('_generate_mtf_dataframes:', hasattr(ModelBundle, '_generate_mtf_dataframes'))
"

# Verify UIP imports
python -c "
from src.inference.universal_pipeline import UniversalInferencePipeline, ScalingSource
from src.inference.errors import InferenceShapeMismatchError
print('UIP OK')
print('ScalingSource values:', [s.value for s in ScalingSource])
"

# Verify EnsembleBundle has predict_from_raw
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
print('predict_from_raw:', hasattr(EnsembleBundle, 'predict_from_raw'))
"

# Verify type bridge
python -c "
from src.models.training.services.ensemble_service import to_ensemble_result
print('to_ensemble_result OK')
"

# Verify skip_scaling in preprocess
python -c "
import inspect
from src.inference.bundle import ModelBundle
source = inspect.getsource(ModelBundle.preprocess)
assert 'skip_scaling=True' in source, 'preprocess must pass skip_scaling=True'
print('skip_scaling=True confirmed in preprocess()')
"
```

---

## File Change Summary

| File | Action | Est. Lines |
|------|--------|-----------|
| `src/inference/bundle.py` | ADD `_apply_adapter`, `_build_3d_input`, `_build_4d_input`, `_generate_mtf_dataframes`; UPDATE `predict_from_raw`, `preprocess` | ~180 |
| `src/inference/errors.py` | **NEW FILE** | ~40 |
| `src/inference/universal_pipeline.py` | **NEW FILE** | ~520 |
| `src/inference/ensemble_bundle.py` | FIX `save` (relative paths), `load` (resolve paths); ADD `predict_from_raw` | ~70 |
| `src/models/training/services/ensemble_service.py` | ADD `to_ensemble_result` bridge | ~50 |
| `src/inference/__init__.py` | ADD UIP + errors exports | ~10 |

**Total estimated: ~870 lines changed/added**

---

## Dependency Graph

```
3B-1 (adapter routing in ModelBundle) ← depends on 3A (BundleMetadata extensions)
3B-2 (UniversalInferencePipeline)     ← depends on 3B-1 (uses bundle._build_3d/4d)
3B-3 (EnsembleBundle fixes)           ← independent of 3B-1/3B-2
3B-4 (MTF data generation)            ← part of 3B-1 (_generate_mtf_dataframes)
3B-5 (type alignment)                 ← independent
```

**Implementation order:**
1. 3B-1e (fix skip_scaling) — quick, prevents double-scaling immediately
2. 3B-1a-d (adapter routing methods) — core capability
3. 3B-4 (MTF generation) — included in 3B-1c
4. 3B-3 (EnsembleBundle fixes) — independent
5. 3B-5 (type bridge) — independent
6. 3B-2 (UIP) — depends on all above

---

*End of Phase 3B implementation plan.*
