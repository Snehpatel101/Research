# Phase 3C: Integration — Concrete Implementation Plan

**Date:** 2026-02-15
**Depends on:** Phase 3B (UniversalInferencePipeline, adapter routing)
**Scope:** Wire everything into consumers: notebook, server, batch, special mode bundles

---

## Overview

Phase 3C has 5 tasks:

| Task | Summary | Files | Est. Lines |
|------|---------|-------|------------|
| 3C-1 | Colab inference demo cells | `notebooks/ml_factory_colab.ipynb` | ~200 |
| 3C-2 | server.py + batch.py migration | `src/inference/server.py`, `batch.py` | ~40 |
| 3C-3 | Special mode bundles | 4 new files + builder.py additions | ~400 |
| 3C-4 | Colab polish (config, warnings, docs) | `notebooks/ml_factory_colab.ipynb` | ~100 |
| 3C-5 | Update `__init__.py` exports | `src/inference/__init__.py` | ~15 |

---

## Task 3C-1: Colab Inference Demo

### Cell 8: Inference Demo (NEW cell after Cell 7)

Insert a new cell after the current Cell 7 (Save & Download).

```python
# =============================================================
# CELL 8: INFERENCE DEMO — Load Bundle & Predict
# =============================================================
from pathlib import Path
from src.inference.bundle import ModelBundle
from src.inference.ensemble_bundle import EnsembleBundle
import pandas as pd
import numpy as np

if "result" not in dir() or result is None or not result.success:
    print("No successful result — skip inference demo.")
else:
    bundle_dir = Path(result.output_dir) / "bundles"
    if not bundle_dir.exists():
        print(f"No bundles found at {bundle_dir}")
    else:
        # --- Discover available bundles ---
        bundle_dirs = sorted([
            d for d in bundle_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ])
        print(f"Found {len(bundle_dirs)} bundles:")
        for bd in bundle_dirs:
            print(f"  - {bd.name}")

        # --- Load best individual model bundle ---
        best_model = result.best_model
        best_bundle_dir = None
        for bd in bundle_dirs:
            if best_model and best_model.lower() in bd.name.lower():
                best_bundle_dir = bd
                break
        if best_bundle_dir is None and bundle_dirs:
            best_bundle_dir = bundle_dirs[0]  # fallback to first

        if best_bundle_dir:
            print(f"\nLoading bundle: {best_bundle_dir.name}")
            bundle = ModelBundle.load(best_bundle_dir)
            print(f"  Model:    {bundle.metadata.model_name}")
            print(f"  Family:   {bundle.metadata.model_family}")
            print(f"  Horizon:  H{bundle.metadata.horizon}")
            print(f"  Features: {bundle.metadata.n_features}")
            print(f"  Sequences: {bundle.metadata.requires_sequences}")
            print(f"  4D:       {bundle.metadata.requires_4d}")

            # --- Predict on sample data (tabular models only for now) ---
            if not bundle.metadata.requires_sequences and not bundle.metadata.requires_4d:
                sample_size = min(100, len(raw_data))
                sample_df = raw_data.tail(sample_size).copy()

                if bundle.preprocessing_graph is not None:
                    features = bundle.preprocess(sample_df)
                    preds = bundle.predict(features)
                    print(f"\n  Predictions on {len(preds.class_predictions)} samples:")
                    unique, counts = np.unique(preds.class_predictions, return_counts=True)
                    for cls, cnt in zip(unique, counts):
                        label = {0: "SHORT", 1: "HOLD", 2: "LONG"}.get(int(cls), str(cls))
                        print(f"    {label}: {cnt} ({cnt/len(preds.class_predictions)*100:.1f}%)")
                    print(f"  Mean confidence: {preds.confidence.mean():.4f}")
                else:
                    print("\n  [No preprocessing graph — raw-to-prediction requires adapter integration]")
                    print("  Use bundle.predict(X_preshaped) with pre-computed feature arrays.")
            else:
                print(f"\n  [Neural/Transformer model — requires adapter integration for raw inference]")
                print(f"  Use bundle.predict(X_3d_or_4d) with pre-shaped tensors.")

        # --- Check for ensemble bundle ---
        ensemble_dir = bundle_dir / "ensemble"
        if not ensemble_dir.exists():
            ensemble_candidates = [d for d in bundle_dirs if "ensemble" in d.name.lower()]
            if ensemble_candidates:
                ensemble_dir = ensemble_candidates[0]

        if ensemble_dir.exists() and (ensemble_dir / "manifest.json").exists():
            print(f"\nEnsemble bundle found: {ensemble_dir.name}")
            try:
                ens_bundle = EnsembleBundle.load(ensemble_dir)
                print(f"  Meta-learner:  {ens_bundle.metadata.meta_learner_name}")
                print(f"  Base models:   {ens_bundle.metadata.base_model_names}")
                print(f"  Stacking features: {ens_bundle.metadata.n_stacking_features}")
            except Exception as e:
                print(f"  [Could not load ensemble bundle: {e}]")

        print("\n--- Inference demo complete ---")
```

### Cell 9: Inference-Only Export (NEW cell after Cell 8)

```python
# =============================================================
# CELL 9: DOWNLOAD INFERENCE BUNDLE ONLY
# =============================================================
from pathlib import Path
import shutil

if "result" not in dir() or result is None or not result.success:
    print("No successful result to export.")
elif result.output_dir:
    bundle_dir = Path(result.output_dir) / "bundles"
    if not bundle_dir.exists() or not any(bundle_dir.iterdir()):
        print("No bundles found. Was bundling enabled in config?")
    else:
        bundle_subdirs = [d for d in bundle_dir.iterdir() if d.is_dir()]
        n_files = sum(1 for f in bundle_dir.rglob("*") if f.is_file())

        zip_name = f"/content/{EXPERIMENT_NAME}_inference_bundle"
        shutil.make_archive(zip_name, "zip", bundle_dir)
        zip_path = Path(f"{zip_name}.zip")

        full_zip = Path(f"/content/{EXPERIMENT_NAME}_results.zip")
        full_size = full_zip.stat().st_size / 1e6 if full_zip.exists() else 0

        print(f"Inference bundle: {zip_path}")
        print(f"  Models:   {len(bundle_subdirs)}")
        print(f"  Files:    {n_files}")
        print(f"  Size:     {zip_path.stat().st_size / 1e6:.1f} MB")
        if full_size > 0:
            ratio = zip_path.stat().st_size / full_zip.stat().st_size * 100
            print(f"  (vs full: {full_size:.1f} MB — {ratio:.0f}% of full)")

        if IN_COLAB:
            try:
                from google.colab import files
                files.download(str(zip_path))
            except Exception:
                print(f"\nManual download: files.download('{zip_path}')")
else:
    print("No output directory found.")
```

### Notebook Cell Mapping

The notebook currently has cells with IDs `cell-0` through `cell-7`. The new cells insert after cell-7:

| Cell ID | Insert Position | Content |
|---------|----------------|---------|
| (new) | After cell-7 | Cell 7b: Drive Persistence |
| (new) | After cell-7b | Cell 8: Inference Demo |
| (new) | After cell-8 | Cell 9: Inference-Only Export |

Use `NotebookEdit` with `edit_mode=insert` and `cell_id="cell-7"` for 7b, then chain subsequent inserts.

---

## Task 3C-2: server.py + batch.py Migration

### server.py Diff

**Current** (line 36):
```python
from src.inference.pipeline import InferencePipeline
```

**New**:
```python
from src.inference.pipeline import InferencePipeline

# Phase 3C: Import UniversalInferencePipeline for adapter-aware inference
try:
    from src.inference.universal_pipeline import UniversalInferencePipeline

    _HAS_UIP = True
except ImportError:
    _HAS_UIP = False
```

**Current** `ModelServer.__init__` (line 224-237):
```python
    def __init__(
        self,
        pipeline: InferencePipeline,
        config: ServerConfig | None = None,
    ) -> None:
```

**New** — Accept both pipeline types:
```python
    def __init__(
        self,
        pipeline: InferencePipeline | Any,  # Also accepts UniversalInferencePipeline
        config: ServerConfig | None = None,
    ) -> None:
```

**Current** `ModelServer.from_bundle` (line 246-254):
```python
    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        config: ServerConfig | None = None,
    ) -> ModelServer:
        """Create server from a model bundle."""
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, config)
```

**New** — Prefer UniversalInferencePipeline:
```python
    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        config: ServerConfig | None = None,
    ) -> ModelServer:
        """Create server from a model bundle."""
        if _HAS_UIP:
            pipeline = UniversalInferencePipeline.from_bundle(path)
        else:
            pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, config)
```

Same pattern for `from_bundles` (line 256-264).

**Summary of server.py changes:**
- Add conditional import of `UniversalInferencePipeline` at top
- `from_bundle()` and `from_bundles()` prefer UIP when available
- `__init__` type hint broadened
- No changes to endpoints or app logic — the `pipeline.predict()` interface is the same

### batch.py Diff

**Current** (line 30):
```python
from src.inference.pipeline import InferencePipeline, InferenceResult
```

**New**:
```python
from src.inference.pipeline import InferencePipeline, InferenceResult

try:
    from src.inference.universal_pipeline import UniversalInferencePipeline

    _HAS_UIP = True
except ImportError:
    _HAS_UIP = False
```

**Current** `BatchPredictor.__init__` (line 117-121):
```python
    def __init__(
        self,
        pipeline: InferencePipeline,
        default_batch_size: int = 10000,
    ) -> None:
```

**New**:
```python
    def __init__(
        self,
        pipeline: InferencePipeline | Any,  # Also accepts UniversalInferencePipeline
        default_batch_size: int = 10000,
    ) -> None:
```

**Current** `BatchPredictor.from_bundle` (line 132-140):
```python
    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        batch_size: int = 10000,
    ) -> BatchPredictor:
        """Create BatchPredictor from a model bundle."""
        pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, default_batch_size=batch_size)
```

**New**:
```python
    @classmethod
    def from_bundle(
        cls,
        path: str | Path,
        batch_size: int = 10000,
    ) -> BatchPredictor:
        """Create BatchPredictor from a model bundle."""
        if _HAS_UIP:
            pipeline = UniversalInferencePipeline.from_bundle(path)
        else:
            pipeline = InferencePipeline.from_bundle(path)
        return cls(pipeline, default_batch_size=batch_size)
```

Same pattern for `from_bundles` (line 142-150).

**Summary of batch.py changes:**
- Add conditional import of `UniversalInferencePipeline`
- `from_bundle()` and `from_bundles()` prefer UIP when available
- `__init__` type hint broadened
- No changes to batch processing logic — pipeline interface is compatible

---

## Task 3C-3: Special Mode Bundles

### 3C-3a: NEW FILE — `src/inference/regime_detector.py`

```python
"""
RegimeDetector — Serializable market regime detection for inference.

Extracted from RegimeAwareTrainer logic to ensure training-time regime
detection is exactly replayed at inference time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection."""

    method: str = "volatility_percentile"
    n_regimes: int = 2
    lookback: int = 60
    volatility_window: int = 20
    adx_threshold: float = 25.0
    percentile_thresholds: list[float] = field(default_factory=lambda: [50.0])
    regime_names: list[str] = field(default_factory=lambda: ["low_vol", "high_vol"])
    min_samples: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_regimes": self.n_regimes,
            "lookback": self.lookback,
            "volatility_window": self.volatility_window,
            "adx_threshold": self.adx_threshold,
            "percentile_thresholds": self.percentile_thresholds,
            "regime_names": self.regime_names,
            "min_samples": self.min_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeDetectorConfig:
        return cls(
            method=data.get("method", "volatility_percentile"),
            n_regimes=data.get("n_regimes", 2),
            lookback=data.get("lookback", 60),
            volatility_window=data.get("volatility_window", 20),
            adx_threshold=data.get("adx_threshold", 25.0),
            percentile_thresholds=data.get("percentile_thresholds", [50.0]),
            regime_names=data.get("regime_names", ["low_vol", "high_vol"]),
            min_samples=data.get("min_samples", 100),
        )


class RegimeDetector:
    """
    Detects current market regime from recent price data.

    Serializable: save/load to JSON so the exact same detection
    logic used in training is replayed at inference.
    """

    def __init__(self, config: RegimeDetectorConfig) -> None:
        self.config = config

    def detect(self, recent_ohlcv: pd.DataFrame) -> str:
        """
        Detect current regime from recent OHLCV bars.

        Args:
            recent_ohlcv: Last N bars (N >= config.lookback).
                         Must have 'close' column.

        Returns:
            Regime name string matching bundle keys (e.g., "low_vol", "high_vol").
        """
        if len(recent_ohlcv) < self.config.min_samples:
            logger.warning(
                f"Only {len(recent_ohlcv)} bars, need {self.config.min_samples}. "
                f"Defaulting to first regime: {self.config.regime_names[0]}"
            )
            return self.config.regime_names[0]

        if self.config.method == "volatility_percentile":
            return self._detect_volatility_percentile(recent_ohlcv)
        elif self.config.method == "trend_adx":
            return self._detect_trend_adx(recent_ohlcv)
        else:
            logger.warning(f"Unknown method '{self.config.method}', defaulting to first regime")
            return self.config.regime_names[0]

    def _detect_volatility_percentile(self, df: pd.DataFrame) -> str:
        """Classify regime by rolling volatility percentile."""
        returns = df["close"].pct_change().dropna()
        vol = returns.rolling(self.config.volatility_window).std().iloc[-1]

        # Compute historical vol distribution
        hist_vol = returns.rolling(self.config.volatility_window).std().dropna()
        if len(hist_vol) == 0:
            return self.config.regime_names[0]

        percentile = (hist_vol < vol).mean() * 100

        # Map percentile to regime
        for i, threshold in enumerate(self.config.percentile_thresholds):
            if percentile <= threshold:
                return self.config.regime_names[i]
        return self.config.regime_names[-1]

    def _detect_trend_adx(self, df: pd.DataFrame) -> str:
        """Classify regime by ADX (trend strength)."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Simplified ADX calculation
        n = min(14, len(close) - 1)
        if n < 2:
            return self.config.regime_names[0]

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = pd.Series(tr).rolling(n).mean().iloc[-1]

        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)

        plus_di = pd.Series(plus_dm).rolling(n).mean().iloc[-1] / max(atr, 1e-10) * 100
        minus_di = pd.Series(minus_dm).rolling(n).mean().iloc[-1] / max(atr, 1e-10) * 100

        dx = abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-10) * 100
        # dx approximates ADX for latest point

        if dx > self.config.adx_threshold:
            return "trending"
        return "ranging"

    def save(self, path: Path) -> None:
        """Save detector config to JSON."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> RegimeDetector:
        """Load detector from JSON."""
        path = Path(path)
        with open(path) as f:
            config = RegimeDetectorConfig.from_dict(json.load(f))
        return cls(config)


__all__ = ["RegimeDetector", "RegimeDetectorConfig"]
```

### 3C-3b: NEW FILE — `src/inference/walk_forward_bundle.py`

```python
"""
WalkForwardBundle — Thin wrapper around the latest-window ModelBundle.

For walk-forward training, the model trained on the largest (most recent)
window is used for inference. Window metadata is preserved for provenance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

WALK_FORWARD_BUNDLE_VERSION = "1.0.0"


@dataclass
class WindowConfig:
    """Walk-forward window metadata."""

    n_windows: int = 1
    window_type: str = "expanding"  # "expanding" or "rolling"
    window_boundaries: list[dict[str, Any]] = field(default_factory=list)
    aggregated_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": self.n_windows,
            "window_type": self.window_type,
            "window_boundaries": self.window_boundaries,
            "aggregated_metrics": self.aggregated_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowConfig:
        return cls(
            n_windows=data.get("n_windows", 1),
            window_type=data.get("window_type", "expanding"),
            window_boundaries=data.get("window_boundaries", []),
            aggregated_metrics=data.get("aggregated_metrics", {}),
        )


class WalkForwardBundle:
    """
    Walk-forward inference bundle.

    Wraps the latest-window ModelBundle with walk-forward metadata.
    Prediction delegates to self.latest_bundle.predict_from_raw().

    Implements the InferenceBundle protocol:
    - predict(X, calibrate) -> PredictionResult
    - predict_from_raw(raw_df) -> PredictionResult
    - save(path) -> Path
    - load(path) -> WalkForwardBundle
    """

    def __init__(
        self,
        latest_bundle: ModelBundle,
        window_config: WindowConfig,
        model_name: str = "",
        horizon: int = 0,
    ) -> None:
        self.latest_bundle = latest_bundle
        self.window_config = window_config
        self.model_name = model_name or latest_bundle.metadata.model_name
        self.horizon = horizon or latest_bundle.metadata.horizon

    def predict(self, X: np.ndarray | pd.DataFrame, calibrate: bool = True) -> PredictionResult:
        """Predict using the latest window's model."""
        return self.latest_bundle.predict(X, calibrate=calibrate)

    def predict_from_raw(self, raw_df: pd.DataFrame, calibrate: bool = True) -> PredictionResult:
        """End-to-end prediction from raw OHLCV using latest window model."""
        return self.latest_bundle.predict_from_raw(raw_df, calibrate=calibrate)

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save walk-forward bundle to disk."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Bundle already exists at {path}. Use overwrite=True.")
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            "version": WALK_FORWARD_BUNDLE_VERSION,
            "bundle_type": "walk_forward",
            "created_at": datetime.now().isoformat(),
            "model_name": self.model_name,
            "horizon": self.horizon,
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Save window config
        with open(path / "window_config.json", "w") as f:
            json.dump(self.window_config.to_dict(), f, indent=2)

        # Save latest bundle as sub-directory
        latest_dir = path / "latest"
        self.latest_bundle.save(latest_dir, overwrite=True)

        logger.info(f"Saved WalkForwardBundle to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> WalkForwardBundle:
        """Load walk-forward bundle from disk."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        with open(path / "manifest.json") as f:
            manifest = json.load(f)

        with open(path / "window_config.json") as f:
            window_config = WindowConfig.from_dict(json.load(f))

        latest_bundle = ModelBundle.load(path / "latest")

        return cls(
            latest_bundle=latest_bundle,
            window_config=window_config,
            model_name=manifest.get("model_name", ""),
            horizon=manifest.get("horizon", 0),
        )

    def validate(self) -> dict[str, Any]:
        """Validate bundle integrity."""
        base_validation = self.latest_bundle.validate()
        return {
            "valid": base_validation["valid"],
            "bundle_type": "walk_forward",
            "n_windows": self.window_config.n_windows,
            "latest_model": base_validation,
        }


__all__ = ["WalkForwardBundle", "WindowConfig", "WALK_FORWARD_BUNDLE_VERSION"]
```

### 3C-3c: NEW FILE — `src/inference/regime_bundle.py`

```python
"""
RegimeBundle — Per-regime model routing for inference.

Detects the current market regime from recent OHLCV data,
then routes to the correct per-regime ModelBundle.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.inference.regime_detector import RegimeDetector, RegimeDetectorConfig
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

REGIME_BUNDLE_VERSION = "1.0.0"


class RegimeBundle:
    """
    Regime-aware inference bundle.

    Stores per-regime ModelBundles + serializable RegimeDetector.
    predict_from_raw() detects regime -> routes to correct model.

    Implements the InferenceBundle protocol.
    """

    def __init__(
        self,
        regime_models: dict[str, ModelBundle],
        regime_detector: RegimeDetector,
        fallback_regime: str | None = None,
        model_name: str = "",
        horizon: int = 0,
    ) -> None:
        """
        Args:
            regime_models: Mapping of regime_name -> ModelBundle.
            regime_detector: Configured RegimeDetector instance.
            fallback_regime: Regime to use if detection fails.
                            Defaults to first key in regime_models.
            model_name: Display name for this bundle.
            horizon: Prediction horizon.
        """
        self.regime_models = regime_models
        self.regime_detector = regime_detector
        self.fallback_regime = fallback_regime or next(iter(regime_models))
        self.model_name = model_name
        self.horizon = horizon

    def predict(self, X: np.ndarray | pd.DataFrame, calibrate: bool = True) -> PredictionResult:
        """
        Predict using fallback regime model (no regime detection without raw data).

        For regime-routed prediction, use predict_from_raw() instead.
        """
        logger.warning(
            "RegimeBundle.predict() uses fallback regime '%s'. "
            "Use predict_from_raw() for regime-routed prediction.",
            self.fallback_regime,
        )
        return self.regime_models[self.fallback_regime].predict(X, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        End-to-end prediction: detect regime, route to correct model.

        Args:
            raw_df: Raw OHLCV DataFrame (needs enough history for regime detection).
            calibrate: Whether to apply probability calibration.

        Returns:
            PredictionResult from the regime-appropriate model.
        """
        # Detect regime from recent data
        regime = self.regime_detector.detect(raw_df)

        if regime not in self.regime_models:
            logger.warning(
                f"Detected regime '{regime}' not in models "
                f"{list(self.regime_models.keys())}. Using fallback '{self.fallback_regime}'."
            )
            regime = self.fallback_regime

        logger.info(f"Regime detected: '{regime}', routing to {regime} model")
        return self.regime_models[regime].predict_from_raw(raw_df, calibrate=calibrate)

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save regime bundle to disk."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Bundle already exists at {path}. Use overwrite=True.")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            "version": REGIME_BUNDLE_VERSION,
            "bundle_type": "regime",
            "created_at": datetime.now().isoformat(),
            "model_name": self.model_name,
            "horizon": self.horizon,
            "regime_names": list(self.regime_models.keys()),
            "fallback_regime": self.fallback_regime,
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Save regime detector
        self.regime_detector.save(path / "regime_config.json")

        # Save each regime model as sub-directory
        regime_dir = path / "regime_models"
        regime_dir.mkdir(exist_ok=True)
        for regime_name, bundle in self.regime_models.items():
            bundle.save(regime_dir / regime_name, overwrite=True)

        logger.info(f"Saved RegimeBundle ({len(self.regime_models)} regimes) to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> RegimeBundle:
        """Load regime bundle from disk."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        with open(path / "manifest.json") as f:
            manifest = json.load(f)

        regime_detector = RegimeDetector.load(path / "regime_config.json")

        regime_models: dict[str, ModelBundle] = {}
        regime_dir = path / "regime_models"
        for regime_name in manifest.get("regime_names", []):
            regime_path = regime_dir / regime_name
            if regime_path.exists():
                regime_models[regime_name] = ModelBundle.load(regime_path)
            else:
                logger.warning(f"Regime model directory missing: {regime_path}")

        return cls(
            regime_models=regime_models,
            regime_detector=regime_detector,
            fallback_regime=manifest.get("fallback_regime"),
            model_name=manifest.get("model_name", ""),
            horizon=manifest.get("horizon", 0),
        )

    def validate(self) -> dict[str, Any]:
        """Validate bundle integrity."""
        regime_validations = {}
        all_valid = True
        for name, bundle in self.regime_models.items():
            v = bundle.validate()
            regime_validations[name] = v
            if not v.get("valid", False):
                all_valid = False

        return {
            "valid": all_valid,
            "bundle_type": "regime",
            "n_regimes": len(self.regime_models),
            "regime_names": list(self.regime_models.keys()),
            "regime_models": regime_validations,
        }


__all__ = ["RegimeBundle", "REGIME_BUNDLE_VERSION"]
```

### 3C-3d: NEW FILE — `src/inference/meta_labeling_bundle.py`

```python
"""
MetaLabelingBundle — Primary model + meta-model for position sizing.

The primary model predicts direction, the meta-model predicts
P(primary is correct), and a threshold filters low-confidence trades.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.bundle import ModelBundle
from src.models.base import PredictionResult

logger = logging.getLogger(__name__)

META_LABELING_BUNDLE_VERSION = "1.0.0"


@dataclass
class MetaLabelingPrediction:
    """Result from meta-labeling inference."""

    directions: np.ndarray  # Primary model class predictions
    meta_probabilities: np.ndarray  # P(primary correct)
    positions: np.ndarray  # direction * meta_prob (0 where below threshold)
    trade_mask: np.ndarray  # Boolean: which samples pass threshold
    threshold: float


class MetaLabelingBundle:
    """
    Meta-labeling inference bundle.

    Combines a primary ModelBundle (direction prediction) with a
    sklearn meta-model (confidence prediction) and threshold.

    Implements the InferenceBundle protocol.
    """

    def __init__(
        self,
        primary_bundle: ModelBundle,
        meta_model: Any,  # sklearn model with predict_proba
        threshold: float = 0.5,
        model_name: str = "",
        horizon: int = 0,
        meta_model_name: str = "logistic",
    ) -> None:
        self.primary_bundle = primary_bundle
        self.meta_model = meta_model
        self.threshold = threshold
        self.model_name = model_name or f"meta_{primary_bundle.metadata.model_name}"
        self.horizon = horizon or primary_bundle.metadata.horizon
        self.meta_model_name = meta_model_name

    def predict(self, X: np.ndarray | pd.DataFrame, calibrate: bool = True) -> PredictionResult:
        """
        Predict using primary model only (no meta-labeling).

        For full meta-labeling with position sizing, use predict_meta().
        """
        return self.primary_bundle.predict(X, calibrate=calibrate)

    def predict_meta(
        self,
        X: np.ndarray | pd.DataFrame,
        calibrate: bool = True,
    ) -> MetaLabelingPrediction:
        """
        Full meta-labeling prediction: direction + confidence + position sizing.

        Args:
            X: Pre-computed features for primary model.
            calibrate: Whether to calibrate primary model probabilities.

        Returns:
            MetaLabelingPrediction with directions, meta_probs, positions, mask.
        """
        # Step 1: Primary model direction prediction
        primary_result = self.primary_bundle.predict(X, calibrate=calibrate)
        directions = primary_result.class_predictions

        # Step 2: Meta-model confidence
        if hasattr(X, "values"):
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
        else:
            X_arr = np.asarray(X)

        meta_probs = self.meta_model.predict_proba(X_arr)
        # P(primary correct) is column 1 for binary meta-model
        if meta_probs.ndim == 2 and meta_probs.shape[1] >= 2:
            confidence = meta_probs[:, 1]
        else:
            confidence = meta_probs.ravel()

        # Step 3: Apply threshold
        trade_mask = confidence >= self.threshold
        positions = directions * confidence * trade_mask.astype(float)

        return MetaLabelingPrediction(
            directions=directions,
            meta_probabilities=confidence,
            positions=positions,
            trade_mask=trade_mask,
            threshold=self.threshold,
        )

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        End-to-end prediction from raw OHLCV (primary model only).

        For full meta-labeling, preprocess manually and use predict_meta().
        """
        return self.primary_bundle.predict_from_raw(raw_df, calibrate=calibrate)

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save meta-labeling bundle to disk."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Bundle already exists at {path}. Use overwrite=True.")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest = {
            "version": META_LABELING_BUNDLE_VERSION,
            "bundle_type": "meta_labeling",
            "created_at": datetime.now().isoformat(),
            "model_name": self.model_name,
            "horizon": self.horizon,
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Save meta config
        meta_config = {
            "primary_model": self.primary_bundle.metadata.model_name,
            "meta_model": self.meta_model_name,
            "threshold": self.threshold,
            "horizon": self.horizon,
        }
        with open(path / "meta_config.json", "w") as f:
            json.dump(meta_config, f, indent=2)

        # Save primary model bundle
        self.primary_bundle.save(path / "primary_model", overwrite=True)

        # Save meta-model (sklearn pickle)
        meta_dir = path / "meta_model"
        meta_dir.mkdir(exist_ok=True)
        with open(meta_dir / "model.pkl", "wb") as f:
            pickle.dump(self.meta_model, f)

        logger.info(f"Saved MetaLabelingBundle to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> MetaLabelingBundle:
        """Load meta-labeling bundle from disk."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        with open(path / "manifest.json") as f:
            manifest = json.load(f)

        with open(path / "meta_config.json") as f:
            meta_config = json.load(f)

        primary_bundle = ModelBundle.load(path / "primary_model")

        # Load meta-model from pickle
        # SECURITY: Only load from trusted internal paths (models trained by this system)
        with open(path / "meta_model" / "model.pkl", "rb") as f:
            meta_model = pickle.load(f)  # noqa: S301

        return cls(
            primary_bundle=primary_bundle,
            meta_model=meta_model,
            threshold=meta_config.get("threshold", 0.5),
            model_name=manifest.get("model_name", ""),
            horizon=manifest.get("horizon", 0),
            meta_model_name=meta_config.get("meta_model", "logistic"),
        )

    def validate(self) -> dict[str, Any]:
        """Validate bundle integrity."""
        primary_validation = self.primary_bundle.validate()
        has_meta = self.meta_model is not None and hasattr(self.meta_model, "predict_proba")
        return {
            "valid": primary_validation["valid"] and has_meta,
            "bundle_type": "meta_labeling",
            "primary_model": primary_validation,
            "has_meta_model": has_meta,
            "threshold": self.threshold,
        }


__all__ = [
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "META_LABELING_BUNDLE_VERSION",
]
```

### 3C-3e: BundleBuilder Additions — `src/inference/builder.py`

Add three new methods to `BundleBuilder` class (insert before `_create_preprocessing_graph` at line 512):

```python
    def build_walk_forward_bundle(
        self,
        training_result: TrainingRunResult,
        model_key: str,
        window_config: dict[str, Any] | None = None,
    ) -> Path:
        """
        Build walk-forward bundle from training result.

        Uses the latest-window trainer to create the inference bundle.

        Args:
            training_result: Result from walk-forward training.
            model_key: Key identifying the model in training_result.model_results.
            window_config: Optional walk-forward window metadata dict.

        Returns:
            Path to saved WalkForwardBundle.
        """
        from src.inference.walk_forward_bundle import WalkForwardBundle, WindowConfig

        model_result = training_result.model_results.get(model_key)
        if model_result is None:
            raise ValueError(f"Model key '{model_key}' not found in training result")

        # Build base ModelBundle for the latest window
        base_result = self.build_from_training_result(training_result)
        if not base_result.bundle_paths:
            raise ValueError(f"Failed to build base bundle for {model_key}")

        # Find the bundle we just created
        model_name = model_result.model_name
        horizon = model_result.horizon
        bundle_path = self.bundles_dir / f"{model_name}_h{horizon}"

        from src.inference.bundle import ModelBundle

        latest_bundle = ModelBundle.load(bundle_path)

        wf_config = WindowConfig.from_dict(window_config or {})

        wf_bundle = WalkForwardBundle(
            latest_bundle=latest_bundle,
            window_config=wf_config,
            model_name=model_name,
            horizon=horizon,
        )

        save_path = self.bundles_dir / f"walk_forward_{model_name}_h{horizon}"
        wf_bundle.save(save_path, overwrite=True)

        logger.info(f"Built walk-forward bundle: {save_path}")
        return save_path

    def build_regime_bundle(
        self,
        regime_models: dict[str, Any],
        regime_config: dict[str, Any],
        model_name: str = "",
        horizon: int = 0,
    ) -> Path:
        """
        Build regime-aware bundle from per-regime trainers.

        Args:
            regime_models: Dict mapping regime_name -> trained ModelBundle.
            regime_config: RegimeDetectorConfig as dict.
            model_name: Display name.
            horizon: Prediction horizon.

        Returns:
            Path to saved RegimeBundle.
        """
        from src.inference.regime_bundle import RegimeBundle
        from src.inference.regime_detector import RegimeDetector, RegimeDetectorConfig

        detector = RegimeDetector(RegimeDetectorConfig.from_dict(regime_config))

        bundle = RegimeBundle(
            regime_models=regime_models,
            regime_detector=detector,
            model_name=model_name,
            horizon=horizon,
        )

        save_path = self.bundles_dir / f"regime_{model_name}_h{horizon}"
        bundle.save(save_path, overwrite=True)

        logger.info(f"Built regime bundle: {save_path}")
        return save_path

    def build_meta_labeling_bundle(
        self,
        primary_bundle: Any,
        meta_model: Any,
        threshold: float = 0.5,
        model_name: str = "",
        horizon: int = 0,
        meta_model_name: str = "logistic",
    ) -> Path:
        """
        Build meta-labeling bundle from primary model and meta-model.

        Args:
            primary_bundle: ModelBundle for primary (direction) model.
            meta_model: Sklearn model with predict_proba for meta-labeling.
            threshold: Minimum confidence to take a trade.
            model_name: Display name.
            horizon: Prediction horizon.
            meta_model_name: Name of meta-model type.

        Returns:
            Path to saved MetaLabelingBundle.
        """
        from src.inference.meta_labeling_bundle import MetaLabelingBundle

        bundle = MetaLabelingBundle(
            primary_bundle=primary_bundle,
            meta_model=meta_model,
            threshold=threshold,
            model_name=model_name,
            horizon=horizon,
            meta_model_name=meta_model_name,
        )

        save_path = self.bundles_dir / f"meta_{model_name}_h{horizon}"
        bundle.save(save_path, overwrite=True)

        logger.info(f"Built meta-labeling bundle: {save_path}")
        return save_path
```

---

## Task 3C-4: Colab Polish

### Cell 0 (Markdown) — Append to Quick Start

Add to the markdown cell (cell-0), after the existing content:

```markdown

## After Training
7. **Cell 7b** - Save bundles to Google Drive (persistent)
8. **Cell 8** - Run inference demo (load bundle -> predict)
9. **Cell 9** - Download inference-only bundle (smaller zip)
```

### Cell 1 (Setup) — Torch Version Check

Insert at end of cell-1, after the GPU check block:

```python
# Torch version check
REQUIRED_TORCH = "2.2.0"
if torch.__version__ < REQUIRED_TORCH:
    print(f"WARNING: torch {torch.__version__} installed, but >={REQUIRED_TORCH} required.")
    print(f"  Neural models may fail. Upgrade with: pip install torch>={REQUIRED_TORCH}")
```

### Cell 2 (Config) — Bundling Config Toggles

Insert before `# BUILD MODEL LIST` section in cell-2:

```python
# --- BUNDLING ---
CREATE_BUNDLE = True              # Create inference bundles after training
BUNDLE_FORMAT = "directory"       # "directory" or "tar.gz"
INCLUDE_OOF = True                # Include out-of-fold predictions in bundle
INCLUDE_FEATURE_IMPORTANCE = True  # Include feature importance scores
```

### Cell 3 (Validation) — Memory Warnings

Insert after the GPU check warnings in cell-3:

```python
# Memory estimation warning
n_neural = len([m for m in MODELS if m in NEURAL_MODELS])
if TRAINING_MODE == "walk_forward" and WF_N_WINDOWS >= 5 and len(MODELS) >= 8:
    warnings.append(
        f"Walk-forward with {WF_N_WINDOWS} windows and {len(MODELS)} models may exceed "
        f"Colab's ~12GB RAM. Consider: fewer models, fewer windows, or Colab Pro (25GB)."
    )
if n_neural >= 6 and torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem < 16:
        warnings.append(
            f"GPU has {gpu_mem:.0f}GB VRAM. Running {n_neural} neural models may OOM. "
            f"Consider training in batches (boosting first, neural second)."
        )
```

### Cell 5 (MLFactory Run) — Bundling Config + Ephemeral Warning

**Change 1:** Add bundling config to ExperimentConfig. Insert import and bundling arg:

After `from src.config.data import FeatureConfig, LabelingConfig, MTFConfig`:
```python
from src.config.experiment import BundlingSection
```

Add to `ExperimentConfig(...)` constructor, after `evaluation=...`:
```python
    bundling=BundlingSection(
        create_bundle=CREATE_BUNDLE,
        bundle_format=BUNDLE_FORMAT,
        include_oof=INCLUDE_OOF,
        include_feature_importance=INCLUDE_FEATURE_IMPORTANCE,
    ),
```

**Note:** This requires `BundlingSection` to exist in `src/config/experiment.py`. If it doesn't exist yet, this is a Phase 3A/3B dependency and should be skipped or stubbed.

**Change 2:** Add ephemeral filesystem warning before `factory.run()`:

```python
# Ephemeral filesystem warning
if IN_COLAB:
    print("NOTE: Colab filesystem is ephemeral. Save bundles to Drive (Cell 7b)")
    print("      or download them (Cell 7/9) before disconnecting.\n")
```

### Cell 6 (Results) — Bundle Summary

Append at end of the else block in cell-6 (before the final `if result.bundle_path:` block):

```python
    # --- Bundle Summary ---
    bundle_path = getattr(result, "bundle_path", None)
    if bundle_path and Path(bundle_path).exists():
        bundle_dir_path = Path(bundle_path)
        bundle_subdirs = sorted([
            d for d in bundle_dir_path.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ])
        if bundle_subdirs:
            print("-" * 40)
            print(f"Inference Bundles ({len(bundle_subdirs)} created)")
            print("-" * 40)
            for bd in bundle_subdirs:
                meta_path = bd / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    family = meta.get("model_family", "?")
                    n_feat = meta.get("n_features", "?")
                    print(f"  {bd.name}: {family}, {n_feat} features")
                else:
                    print(f"  {bd.name}")
            total_size = sum(
                f.stat().st_size for f in bundle_dir_path.rglob("*") if f.is_file()
            )
            print(f"  Total bundle size: {total_size / 1e6:.1f} MB")
```

### Cell 7b: Drive Persistence (NEW cell)

Insert after cell-7:

```python
# =============================================================
# CELL 7b: SAVE TO GOOGLE DRIVE (optional)
# =============================================================
# Run this cell to mount Google Drive and save bundles for persistence.
# Colab's filesystem is EPHEMERAL — files are lost on disconnect.

import os

if IN_COLAB:
    drive_mounted = os.path.exists("/content/drive/MyDrive")

    if not drive_mounted:
        print("Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive")
        drive_mounted = os.path.exists("/content/drive/MyDrive")

    if drive_mounted and "result" in dir() and result and result.success:
        from pathlib import Path
        import shutil

        drive_base = Path("/content/drive/MyDrive/ml_factory_results")
        drive_dest = drive_base / EXPERIMENT_NAME / "bundles"
        drive_dest.mkdir(parents=True, exist_ok=True)

        bundle_src = Path(result.output_dir) / "bundles"
        if bundle_src.exists():
            if drive_dest.exists():
                shutil.rmtree(drive_dest)
            shutil.copytree(bundle_src, drive_dest)
            print(f"Bundles saved to Drive: {drive_dest}")

            config_src = Path(result.output_dir) / "experiment_config.yaml"
            if config_src.exists():
                shutil.copy2(config_src, drive_base / EXPERIMENT_NAME / "experiment_config.yaml")

            n_bundles = sum(1 for d in drive_dest.iterdir() if d.is_dir())
            drive_size = sum(f.stat().st_size for f in drive_dest.rglob("*") if f.is_file())
            print(f"  {n_bundles} bundles, {drive_size / 1e6:.1f} MB")
            print(f"\nTo reload in a future session:")
            print(f"  from src.inference.bundle import ModelBundle")
            print(f"  bundle = ModelBundle.load('{drive_dest}/xgboost_h20')")
        else:
            print("No bundles found to save.")
    elif not drive_mounted:
        print("Drive mount failed or was cancelled.")
    else:
        print("No successful result to save.")
else:
    print("Not in Colab — files persist locally at:",
          result.output_dir if "result" in dir() and result else "N/A")
```

---

## Task 3C-5: Update `__init__.py` Exports

### Exact diff for `src/inference/__init__.py`

**Add imports** (after line 133, before `from src.inference.server import`):

```python
# Phase 3C: Special mode bundles
from src.inference.walk_forward_bundle import WalkForwardBundle, WindowConfig
from src.inference.regime_bundle import RegimeBundle
from src.inference.meta_labeling_bundle import MetaLabelingBundle, MetaLabelingPrediction
from src.inference.regime_detector import RegimeDetector, RegimeDetectorConfig
```

**Add to `__all__`** (after the InferenceOrchestrator section):

```python
    # Special Mode Bundles (Phase 3C)
    "WalkForwardBundle",
    "WindowConfig",
    "RegimeBundle",
    "MetaLabelingBundle",
    "MetaLabelingPrediction",
    "RegimeDetector",
    "RegimeDetectorConfig",
```

**Add deprecation aliases** at bottom of file (replace existing `__all__` closing bracket):

```python
# Phase 3C: Deprecation aliases for old names
def __getattr__(name: str):
    """Provide deprecation warnings for old class names."""
    import warnings

    _DEPRECATED = {
        "InferenceOrchestrator": "InferenceOrchestrator",  # already canonical
    }
    if name in _DEPRECATED:
        warnings.warn(
            f"{name} is deprecated, use {_DEPRECATED[name]} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[_DEPRECATED[name]]
    raise AttributeError(f"module 'src.inference' has no attribute {name!r}")
```

**Note:** The `UniversalInferencePipeline` export depends on Phase 3B creating the file. Add conditionally:

```python
# Phase 3B: Universal Inference Pipeline (added when 3B completes)
try:
    from src.inference.universal_pipeline import UniversalInferencePipeline
except ImportError:
    pass  # Not yet implemented — Phase 3B dependency
```

---

## Execution Order

```
1. Task 3C-3: Create 4 new files (regime_detector, walk_forward_bundle,
              regime_bundle, meta_labeling_bundle) — no dependencies on
              existing code beyond ModelBundle
2. Task 3C-3e: Add BundleBuilder methods (builder.py edits)
3. Task 3C-5: Update __init__.py exports
4. Task 3C-2: server.py + batch.py migration (depends on Phase 3B UIP)
5. Task 3C-4: Colab polish (notebook modifications)
6. Task 3C-1: Colab inference demo cells (notebook new cells)
```

Tasks 1-3 can proceed immediately. Tasks 4-6 can proceed in parallel once 3B is done.

---

## Validation Commands

```bash
# Verify new imports work
python -c "from src.inference.regime_detector import RegimeDetector; print('OK')"
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('OK')"
python -c "from src.inference.regime_bundle import RegimeBundle; print('OK')"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('OK')"

# Verify __init__.py exports
python -c "from src.inference import WalkForwardBundle, RegimeBundle, MetaLabelingBundle; print('OK')"

# Verify server/batch still import correctly
python -c "from src.inference.server import ModelServer; print('OK')"
python -c "from src.inference.batch import BatchPredictor; print('OK')"

# Linting
ruff check src/inference/regime_detector.py src/inference/walk_forward_bundle.py src/inference/regime_bundle.py src/inference/meta_labeling_bundle.py
black --check src/inference/regime_detector.py src/inference/walk_forward_bundle.py src/inference/regime_bundle.py src/inference/meta_labeling_bundle.py
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Phase 3B not complete (no UIP) | MEDIUM | server.py/batch.py use conditional import with fallback |
| BundlingSection not in config | LOW | Cell 5 change is optional, skip if class doesn't exist |
| Pickle security in meta_model | LOW | Only loads from trusted paths, add `noqa: S301` comment |
| RegimeDetector accuracy vs training | MEDIUM | Exact same config serialized/deserialized, same algorithm |
| Notebook cell ID stability | LOW | Use `cell_id` references for inserts, not line numbers |

---

## Dependencies on Other Phases

| This Task | Depends On | Hard/Soft |
|-----------|-----------|-----------|
| 3C-2 (server/batch migration) | 3B (UniversalInferencePipeline exists) | SOFT — conditional import |
| 3C-3 (special bundles) | 3B (ModelBundle.predict_from_raw works for all families) | SOFT — save/load works immediately, predict_from_raw needs 3B for neural |
| 3C-4 (BundlingSection in Cell 5) | 3A (BundlingSection in config) | SOFT — skip if not present |
| 3C-1 (inference demo) | 3C-3 (bundles exist to demo) | HARD for special modes, SOFT for standard |

---

## Files Changed/Created Summary

| File | Action | Lines Changed |
|------|--------|--------------|
| `src/inference/regime_detector.py` | **NEW** | ~160 |
| `src/inference/walk_forward_bundle.py` | **NEW** | ~150 |
| `src/inference/regime_bundle.py` | **NEW** | ~170 |
| `src/inference/meta_labeling_bundle.py` | **NEW** | ~200 |
| `src/inference/builder.py` | MODIFY (add 3 methods) | +120 |
| `src/inference/__init__.py` | MODIFY (add exports) | +20 |
| `src/inference/server.py` | MODIFY (UIP migration) | +15 |
| `src/inference/batch.py` | MODIFY (UIP migration) | +15 |
| `notebooks/ml_factory_colab.ipynb` | MODIFY (6 cell edits + 3 new cells) | +300 |
| **Total** | | **~1,150** |
