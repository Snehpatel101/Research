# PHASE 5: INFERENCE - Model Bundling and Deployment

**Status:** PLANNING
**Created:** 2026-01-17
**Purpose:** Define inference infrastructure for serializable model bundles, preprocessing graphs, and production deployment

---

## Overview

Phase 5 establishes the inference system that transforms trained models into deployable artifacts. The key innovation is the **PreprocessingGraph** - a serializable representation of the entire feature engineering pipeline that ensures train/serve parity.

**Key Principles:**
1. **Train/Serve Parity** - Exact same preprocessing at inference as training
2. **Self-Contained Bundles** - Everything needed for inference in one directory
3. **No Manual Feature Engineering** - PreprocessingGraph handles raw OHLCV to features
4. **Versioned Artifacts** - Bundle version 1.1.0 with manifest and checksums

---

## Data Flow: Training to Inference

```
TRAINING TIME                                    INFERENCE TIME
=============                                    ==============

Raw OHLCV                                        Raw OHLCV
    |                                                |
    v                                                v
FeatureRegistry.compute_all()               PreprocessingGraph.transform()
    |                                                |
    v                                                v
Adapter.transform()                         bundle.preprocess()
    |                                                |
    v                                                v
model.fit()                                 bundle.predict()
    |                                                |
    v                                                v
ModelBundle.from_training()                 PredictionOutput
    |
    v
bundle.save()
    |
    v
bundles/xgb_h20/
    manifest.json
    metadata.json
    features.json
    scaler.pkl
    calibrator.pkl
    preprocessing_graph.json
    model/
```

---

## Task 5.1: Bundle Structure and Metadata

### Bundle Directory Layout

```
bundles/xgb_h20/
    manifest.json               # File listing with checksums
    metadata.json               # Model metadata (version, horizon, etc.)
    features.json               # Feature column names and order
    scaler.pkl                  # Fitted RobustScaler/StandardScaler
    calibrator.pkl              # Probability calibrator (optional)
    preprocessing_graph.json    # Complete preprocessing config
    model/                      # Model-specific artifacts
        model.pkl               # Serialized model (or model weights)
        config.json             # Model hyperparameters
```

### File: `src/inference/bundle.py`

```python
"""
ModelBundle - Serializable container for trained model artifacts.

Version: 1.1.0 - Added PreprocessingGraph support
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

BUNDLE_VERSION = "1.1.0"


@dataclass
class BundleMetadata:
    """Metadata for a model bundle."""
    version: str
    created_at: str
    model_name: str
    model_family: str
    horizon: int
    n_features: int
    feature_hash: str
    requires_sequences: bool = False
    sequence_length: int = 0
    has_calibrator: bool = False
    has_preprocessing_graph: bool = False
    symbol: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)


class ModelBundle:
    """
    Serializable container for all inference artifacts.

    Example:
        >>> bundle = ModelBundle.from_training(
        ...     model=trained_xgb,
        ...     scaler=fitted_scaler,
        ...     feature_columns=X.columns.tolist(),
        ...     horizon=20,
        ... )
        >>> bundle.save("./bundles/xgb_h20")

        >>> bundle = ModelBundle.load("./bundles/xgb_h20")
        >>> predictions = bundle.predict(X_test)

        >>> # Raw OHLCV inference with PreprocessingGraph
        >>> predictions = bundle.predict_from_raw(raw_ohlcv_df)
    """

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_columns: list[str],
        metadata: BundleMetadata,
        calibrator: Any = None,
        preprocessing_graph: Any = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.calibrator = calibrator
        self.preprocessing_graph = preprocessing_graph

    @classmethod
    def from_training(
        cls,
        model: Any,
        scaler: Any,
        feature_columns: list[str],
        horizon: int,
        calibrator: Any = None,
        preprocessing_graph: Any = None,
        symbol: str = "",
        training_metrics: dict | None = None,
    ) -> ModelBundle:
        """Create bundle from trained components."""
        model_name = getattr(model, "_get_model_type", lambda: "unknown")()
        model_family = getattr(model, "model_family", "unknown")
        requires_sequences = getattr(model, "requires_sequences", False)
        sequence_length = getattr(model, "_config", {}).get("sequence_length", 0)

        feature_hash = hashlib.md5(",".join(feature_columns).encode()).hexdigest()[:12]

        metadata = BundleMetadata(
            version=BUNDLE_VERSION,
            created_at=datetime.now().isoformat(),
            model_name=model_name,
            model_family=model_family,
            horizon=horizon,
            n_features=len(feature_columns),
            feature_hash=feature_hash,
            requires_sequences=requires_sequences,
            sequence_length=sequence_length,
            has_calibrator=calibrator is not None,
            has_preprocessing_graph=preprocessing_graph is not None,
            symbol=symbol,
            training_metrics=training_metrics or {},
        )

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
            preprocessing_graph=preprocessing_graph,
        )

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save bundle to disk."""
        path = Path(path)
        if path.exists():
            if overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Bundle exists at {path}")
        path.mkdir(parents=True)

        # Save metadata
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata.__dict__, f, indent=2)

        # Save features
        with open(path / "features.json", "w") as f:
            json.dump({"columns": self.feature_columns}, f, indent=2)

        # Save scaler
        if self.scaler is not None:
            with open(path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        # Save calibrator
        if self.calibrator is not None:
            with open(path / "calibrator.pkl", "wb") as f:
                pickle.dump(self.calibrator, f)

        # Save preprocessing graph
        if self.preprocessing_graph is not None:
            self.preprocessing_graph.save(path / "preprocessing_graph.json")

        # Save model
        model_dir = path / "model"
        self.model.save(model_dir)

        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        """Load bundle from disk."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = BundleMetadata(**json.load(f))

        # Load features
        with open(path / "features.json") as f:
            feature_columns = json.load(f)["columns"]

        # Load scaler
        scaler = None
        if (path / "scaler.pkl").exists():
            with open(path / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

        # Load calibrator
        calibrator = None
        if (path / "calibrator.pkl").exists():
            with open(path / "calibrator.pkl", "rb") as f:
                calibrator = pickle.load(f)

        # Load preprocessing graph
        preprocessing_graph = None
        if (path / "preprocessing_graph.json").exists():
            from src.inference.preprocessing_graph import PreprocessingGraph
            preprocessing_graph = PreprocessingGraph.load(path / "preprocessing_graph.json")

        # Load model
        from src.models.registry import ModelRegistry
        model = ModelRegistry.create(metadata.model_name)
        model.load(path / "model")

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
            preprocessing_graph=preprocessing_graph,
        )

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True):
        """Make predictions with pre-computed features."""
        X_array = self._prepare_input(X)

        # Apply scaling
        if self.scaler is not None:
            if self.metadata.requires_sequences:
                orig_shape = X_array.shape
                X_flat = X_array.reshape(-1, orig_shape[-1])
                X_scaled = self.scaler.transform(X_flat)
                X_array = X_scaled.reshape(orig_shape)
            else:
                X_array = self.scaler.transform(X_array)

        # Predict
        output = self.model.predict(X_array)

        # Calibrate
        if calibrate and self.calibrator is not None:
            output = self._apply_calibration(output)

        return output

    def predict_from_raw(self, raw_df: pd.DataFrame, calibrate: bool = True):
        """End-to-end prediction from raw OHLCV data."""
        if self.preprocessing_graph is None:
            raise RuntimeError("No preprocessing graph available")

        features = self.preprocessing_graph.transform(raw_df)
        available = [c for c in self.feature_columns if c in features.columns]
        features = features[available]

        return self.predict(features, calibrate=calibrate)

    def _prepare_input(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        if isinstance(X, pd.DataFrame):
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {list(missing)[:10]}")
            X = X[self.feature_columns].values
        return np.asarray(X, dtype=np.float32)

    def _apply_calibration(self, output):
        """Apply probability calibration."""
        calibrated_probs = self.calibrator.calibrate(output.class_probabilities)
        output.class_probabilities = calibrated_probs
        output.confidence = np.max(calibrated_probs, axis=1)
        return output


__all__ = ["ModelBundle", "BundleMetadata", "BUNDLE_VERSION"]
```

---

## Task 5.2: PreprocessingGraph for Feature Lineage

### File: `src/inference/preprocessing_graph.py`

```python
"""
PreprocessingGraph - Serializable preprocessing pipeline for train/serve parity.

Captures:
1. Data cleaning configuration (resampling, gap handling)
2. Feature engineering configuration (all indicator periods)
3. MTF configuration (timeframes, mode)
4. Regime detection configuration
5. Scaling configuration (per-column parameters)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREPROCESSING_GRAPH_VERSION = "1.0.0"


@dataclass
class CleaningConfig:
    source_timeframe: str = "1min"
    target_timeframe: str = "5min"
    gap_fill_method: str = "forward"


@dataclass
class IndicatorConfig:
    rsi_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    ema_periods: list[int] = field(default_factory=lambda: [9, 12, 21, 26, 50])
    atr_periods: list[int] = field(default_factory=lambda: [7, 14, 21])
    bollinger_period: int = 20
    adx_period: int = 14


@dataclass
class MTFConfig:
    enabled: bool = True
    base_timeframe: str = "5min"
    mtf_timeframes: list[str] = field(default_factory=lambda: ["15min", "60min"])
    mode: str = "both"


@dataclass
class PreprocessingGraphConfig:
    version: str = PREPROCESSING_GRAPH_VERSION
    created_at: str = ""
    horizon: int = 20
    symbol: str = ""
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    mtf: MTFConfig = field(default_factory=MTFConfig)
    feature_columns: list[str] = field(default_factory=list)
    config_hash: str = ""


class PreprocessingGraph:
    """
    Serializable preprocessing pipeline for train/serve parity.

    Raw OHLCV --> Cleaning --> Features --> MTF --> Regime --> Scaling --> Model

    Example:
        >>> graph = PreprocessingGraph.from_pipeline_config(pipeline_config)
        >>> bundle.set_preprocessing_graph(graph)
        >>> bundle.save("./bundles/xgb_h20")

        >>> bundle = ModelBundle.load("./bundles/xgb_h20")
        >>> predictions = bundle.predict_from_raw(raw_ohlcv_df)
    """

    def __init__(self, config: PreprocessingGraphConfig) -> None:
        self.config = config
        self._scaler = None
        self._is_fitted = False

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: dict[str, Any],
        feature_columns: list[str] | None = None,
        symbol: str = "",
        horizon: int = 20,
    ) -> PreprocessingGraph:
        """Create graph from training pipeline configuration."""
        config = PreprocessingGraphConfig(
            version=PREPROCESSING_GRAPH_VERSION,
            created_at=datetime.now().isoformat(),
            horizon=horizon,
            symbol=symbol,
            feature_columns=feature_columns or [],
        )
        config.config_hash = cls._compute_hash(config)

        graph = cls(config)
        graph._is_fitted = True
        return graph

    @staticmethod
    def _compute_hash(config: PreprocessingGraphConfig) -> str:
        hash_data = asdict(config)
        hash_data.pop("created_at", None)
        hash_data.pop("config_hash", None)
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def set_scaler(self, scaler: Any) -> None:
        """Set the fitted scaler instance."""
        self._scaler = scaler
        self._is_fitted = True

    def transform(self, raw_df: pd.DataFrame, skip_scaling: bool = False) -> pd.DataFrame:
        """Apply full preprocessing pipeline to raw OHLCV data."""
        df = raw_df.copy()

        # 1. Feature engineering
        df = self._apply_features(df)

        # 2. MTF features
        if self.config.mtf.enabled:
            df = self._apply_mtf(df)

        # 3. Drop NaN
        df = df.dropna()

        # 4. Scaling
        if not skip_scaling and self._scaler is not None:
            df = self._apply_scaling(df)

        # 5. Select feature columns
        if self.config.feature_columns:
            available = [c for c in self.config.feature_columns if c in df.columns]
            df = df[available]

        return df

    def _apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        from src.features.registry import FeatureRegistry
        return FeatureRegistry.compute_all(df)

    def _apply_mtf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-timeframe features."""
        from src.features.mtf import MTFFeatureGenerator
        generator = MTFFeatureGenerator(
            base_timeframe=self.config.mtf.base_timeframe,
            mtf_timeframes=self.config.mtf.mtf_timeframes,
        )
        return generator.generate_mtf_features(df)

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler."""
        exclude = {"datetime", "date", "time", "symbol", "label"}
        numeric_cols = [c for c in df.columns if c not in exclude]
        numeric_cols = df[numeric_cols].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df[numeric_cols] = self._scaler.transform(df[numeric_cols])
        return df

    def save(self, path: Path) -> None:
        """Save preprocessing graph to JSON."""
        path = Path(path)
        self.config.config_hash = self._compute_hash(self.config)
        with open(path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PreprocessingGraph:
        """Load preprocessing graph from JSON."""
        with open(path) as f:
            data = json.load(f)
        config = PreprocessingGraphConfig(**data)
        graph = cls(config)
        graph._is_fitted = True
        return graph


__all__ = ["PreprocessingGraph", "PreprocessingGraphConfig"]
```

---

## Task 5.3: InferencePipeline

### File: `src/inference/pipeline.py`

```python
"""InferencePipeline - High-level inference orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle


@dataclass
class InferenceResult:
    """Result from single model inference."""
    predictions: Any
    inference_time_ms: float
    model_name: str
    horizon: int
    n_samples: int


@dataclass
class EnsembleResult:
    """Result from ensemble inference."""
    predictions: Any
    individual_results: list[InferenceResult]
    voting_method: str
    inference_time_ms: float


class InferencePipeline:
    """
    High-level inference orchestration for one or more model bundles.

    Example:
        >>> pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
        >>> result = pipeline.predict(X_test)

        >>> pipeline = InferencePipeline.from_bundles([
        ...     "./bundles/xgb_h20",
        ...     "./bundles/lgbm_h20",
        ... ])
        >>> result = pipeline.predict_ensemble(X_test, method="soft_vote")
    """

    def __init__(self, bundles: list[ModelBundle], default_voting: str = "soft_vote"):
        self.bundles = bundles
        self.default_voting = default_voting
        self._primary_bundle = bundles[0]

    @classmethod
    def from_bundle(cls, path: str | Path) -> InferencePipeline:
        """Create pipeline from single bundle."""
        bundle = ModelBundle.load(path)
        return cls([bundle])

    @classmethod
    def from_bundles(cls, paths: list[str | Path]) -> InferencePipeline:
        """Create pipeline from multiple bundles."""
        bundles = [ModelBundle.load(p) for p in paths]
        return cls(bundles)

    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> InferenceResult:
        """Predict using primary bundle."""
        start = time.perf_counter()
        preds = self._primary_bundle.predict(X, calibrate=calibrate)
        elapsed = (time.perf_counter() - start) * 1000
        return InferenceResult(
            predictions=preds,
            inference_time_ms=elapsed,
            model_name=self._primary_bundle.metadata.model_name,
            horizon=self._primary_bundle.metadata.horizon,
            n_samples=len(X),
        )

    def predict_ensemble(
        self,
        X: pd.DataFrame | np.ndarray,
        method: str | None = None,
        weights: list[float] | None = None,
        calibrate: bool = True,
    ) -> EnsembleResult:
        """Combine predictions from all models."""
        method = method or self.default_voting
        start = time.perf_counter()

        # Get individual predictions
        results = []
        for bundle in self.bundles:
            pred = bundle.predict(X, calibrate=calibrate)
            results.append(InferenceResult(
                predictions=pred,
                inference_time_ms=0,
                model_name=bundle.metadata.model_name,
                horizon=bundle.metadata.horizon,
                n_samples=len(X),
            ))

        # Combine
        combined = self._combine(results, method, weights)

        elapsed = (time.perf_counter() - start) * 1000
        return EnsembleResult(
            predictions=combined,
            individual_results=results,
            voting_method=method,
            inference_time_ms=elapsed,
        )

    def _combine(self, results: list[InferenceResult], method: str, weights: list[float] | None):
        """Combine predictions."""
        if method == "soft_vote":
            probs = [r.predictions.class_probabilities for r in results]
            avg_probs = np.mean(probs, axis=0)
            return np.argmax(avg_probs, axis=1) - 1
        elif method == "hard_vote":
            preds = [r.predictions.class_predictions for r in results]
            stacked = np.stack(preds, axis=0)
            return np.apply_along_axis(lambda x: np.bincount(x + 1).argmax() - 1, 0, stacked)
        else:
            raise ValueError(f"Unknown method: {method}")


__all__ = ["InferencePipeline", "InferenceResult", "EnsembleResult"]
```

---

## Task 5.4: BatchPredictor

### File: `src/inference/batch.py`

```python
"""BatchPredictor - Chunked processing for large datasets."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from src.inference.pipeline import InferencePipeline


@dataclass
class BatchProgress:
    """Progress tracking for batch inference."""
    total_samples: int
    processed_samples: int
    current_batch: int
    total_batches: int
    elapsed_seconds: float

    @property
    def progress_pct(self) -> float:
        return self.processed_samples / self.total_samples * 100

    @property
    def eta_seconds(self) -> float:
        rate = self.processed_samples / max(self.elapsed_seconds, 0.001)
        return (self.total_samples - self.processed_samples) / rate


@dataclass
class BatchResult:
    """Result from batch inference."""
    predictions_df: pd.DataFrame
    n_samples: int
    n_batches: int
    total_time_seconds: float

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        self.predictions_df.to_parquet(path, index=False)
        return path


class BatchPredictor:
    """Efficient batch inference for large datasets."""

    def __init__(self, pipeline: InferencePipeline, default_batch_size: int = 10000):
        self.pipeline = pipeline
        self.default_batch_size = default_batch_size

    @classmethod
    def from_bundle(cls, path: str | Path, batch_size: int = 10000) -> BatchPredictor:
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, default_batch_size=batch_size)

    def predict_batch(
        self,
        data: pd.DataFrame | Path,
        batch_size: int | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> BatchResult:
        """Process large dataset in batches."""
        batch_size = batch_size or self.default_batch_size

        if isinstance(data, Path):
            data = pd.read_parquet(data)

        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        start = time.time()
        all_preds = []
        processed = 0

        for batch_idx in range(n_batches):
            batch = data.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            result = self.pipeline.predict(batch)
            all_preds.append(result.predictions)
            processed += len(batch)

            if progress_callback:
                progress_callback(BatchProgress(
                    total_samples=n_samples,
                    processed_samples=processed,
                    current_batch=batch_idx + 1,
                    total_batches=n_batches,
                    elapsed_seconds=time.time() - start,
                ))

        return BatchResult(
            predictions_df=pd.concat([pd.DataFrame(p) for p in all_preds]),
            n_samples=n_samples,
            n_batches=n_batches,
            total_time_seconds=time.time() - start,
        )


__all__ = ["BatchPredictor", "BatchProgress", "BatchResult"]
```

---

## Task 5.5: FastAPI Server

### File: `src/inference/server.py`

```python
"""ModelServer - FastAPI HTTP server for model inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    max_batch_size: int = 1000


class ModelServer:
    """
    FastAPI-based HTTP server for model inference.

    Endpoints:
    - GET  /health       - Health check
    - GET  /info         - Model information
    - POST /predict      - Single/batch predictions
    """

    def __init__(self, pipeline: InferencePipeline, config: ServerConfig | None = None):
        self.pipeline = pipeline
        self.config = config or ServerConfig()
        self._app = None

    @classmethod
    def from_bundle(cls, path: str | Path, config: ServerConfig | None = None):
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, config)

    def create_app(self):
        """Create FastAPI application."""
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        app = FastAPI(title="ML Factory Inference Server")

        class PredictRequest(BaseModel):
            features: list[list[float]]
            calibrate: bool = True

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/info")
        async def info():
            return {
                "model": self.pipeline._primary_bundle.metadata.model_name,
                "horizon": self.pipeline._primary_bundle.metadata.horizon,
                "n_features": self.pipeline._primary_bundle.metadata.n_features,
            }

        @app.post("/predict")
        async def predict(request: PredictRequest):
            if len(request.features) > self.config.max_batch_size:
                raise HTTPException(400, "Batch too large")
            X = np.array(request.features, dtype=np.float32)
            result = self.pipeline.predict(X, calibrate=request.calibrate)
            return {
                "predictions": result.predictions.class_predictions.tolist(),
                "probabilities": result.predictions.class_probabilities.tolist(),
            }

        self._app = app
        return app

    def run(self, host: str | None = None, port: int | None = None):
        """Run the server."""
        import uvicorn
        if self._app is None:
            self.create_app()
        uvicorn.run(self._app, host=host or self.config.host, port=port or self.config.port)


def start_server(bundle_path: str | Path, host: str = "0.0.0.0", port: int = 8080):
    """Start model server with single bundle."""
    server = ModelServer.from_bundle(bundle_path, ServerConfig(host=host, port=port))
    server.run()


__all__ = ["ModelServer", "ServerConfig", "start_server"]
```

---

## Implementation Checklist

### Task 5.1: Bundle Structure
- [ ] Create `src/inference/bundle.py`
- [ ] `BundleMetadata` dataclass
- [ ] `ModelBundle.from_training()`
- [ ] `ModelBundle.save()` / `load()`
- [ ] `predict()` and `predict_from_raw()`

### Task 5.2: PreprocessingGraph
- [ ] Create `src/inference/preprocessing_graph.py`
- [ ] Config dataclasses
- [ ] `PreprocessingGraph.transform()`
- [ ] `save()` / `load()` with JSON

### Task 5.3: InferencePipeline
- [ ] Create `src/inference/pipeline.py`
- [ ] `InferenceResult` dataclass
- [ ] `predict()` for single model
- [ ] `predict_ensemble()` with voting

### Task 5.4: BatchPredictor
- [ ] Create `src/inference/batch.py`
- [ ] `BatchProgress` and `BatchResult`
- [ ] `predict_batch()` with callbacks

### Task 5.5: FastAPI Server
- [ ] Create `src/inference/server.py`
- [ ] `/health`, `/info`, `/predict` endpoints
- [ ] `start_server()` convenience function
