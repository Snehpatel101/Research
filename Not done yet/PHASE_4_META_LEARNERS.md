# PHASE 4: META-LEARNERS - Heterogeneous Ensemble + OOF Alignment

**Status:** PLANNING
**Created:** 2026-01-17
**Purpose:** Define heterogeneous stacking with OOF alignment across 2D/3D/4D adapters

---

## Overview

Phase 4 establishes the meta-learner system for building heterogeneous ensembles. The key challenge is **OOF alignment** - when combining tabular models (100% sample coverage) with sequence models (less than 100% coverage due to lookback windows), predictions must be aligned to a common valid range.

**Key Principles:**
1. **Heterogeneous Support** - Mix 2D tabular and 3D sequence models in same ensemble
2. **OOF Alignment** - Automatically align predictions to common valid indices
3. **4 Meta-Learners** - ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
4. **Caching** - OOFCache for efficient re-use across experiments

---

## Data Flow Diagram

```
                    Base Model OOF Predictions
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
   XGBoost (2D)          LSTM (3D)           TFT (4D)
   Coverage: 100%        Coverage: 98%       Coverage: 97%
   Offset: 0             Offset: 59          Offset: 59
         |                    |                    |
         v                    v                    v
    +-------------------------------------------+
    |          OOF ALIGNMENT VALIDATOR          |
    |  max_offset = 59                          |
    |  common_samples = N - 59                  |
    +-------------------------------------------+
                              |
                              v
    +-------------------------------------------+
    |          STACKING DATASET BUILDER         |
    |  Features: xgb_prob_*, lstm_prob_*, ...   |
    |  Derived: mean_conf, agreement, entropy   |
    +-------------------------------------------+
                              |
                              v
    +-------------------------------------------+
    |             META-LEARNER                  |
    |  ridge_meta | mlp_meta | xgboost_meta    |
    +-------------------------------------------+
                              |
                              v
                    Final Predictions
```

---

## Task 4.1: HeterogeneousStackingBuilder

### File: `src/ensemble/stacking.py`

```python
"""
HeterogeneousStackingBuilder - Build stacking datasets from heterogeneous models.

Handles alignment when mixing:
- Tabular models (XGBoost, LightGBM): 100% coverage, offset=0
- Sequence models (LSTM, TCN): ~98% coverage, offset=seq_len-1
- Multi-stream models (PatchTST): ~97% coverage, offset=seq_len-1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.cross_validation.oof_core import OOFPrediction

logger = logging.getLogger(__name__)


@dataclass
class StackingDataset:
    """Dataset for meta-learner training."""
    X: pd.DataFrame  # Stacked OOF probabilities
    y: np.ndarray    # Aligned labels
    base_models: list[str]
    alignment_offset: int
    n_original_samples: int
    feature_names: list[str] = field(default_factory=list)
    horizon: int = 20

    @property
    def coverage(self) -> float:
        return len(self.X) / self.n_original_samples

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


class HeterogeneousStackingBuilder:
    """
    Build stacking datasets from heterogeneous model OOF predictions.

    Handles the alignment challenge when stacking:
    - Tabular models: Full coverage (N samples)
    - Sequence models: Partial coverage (N - seq_len + 1 samples)

    Example:
        >>> builder = HeterogeneousStackingBuilder()
        >>> stacking_ds = builder.build(
        ...     oof_predictions={
        ...         "xgboost": xgb_oof,    # N samples
        ...         "lstm": lstm_oof,       # N-59 samples
        ...     },
        ...     y_true=labels,
        ...     horizon=20,
        ... )
        >>> print(stacking_ds.coverage)  # ~0.98
    """

    def __init__(
        self,
        add_derived_features: bool = True,
        class_names: list[str] | None = None,
    ) -> None:
        self.add_derived_features = add_derived_features
        self.class_names = class_names or ["short", "neutral", "long"]

    def build(
        self,
        oof_predictions: dict[str, OOFPrediction],
        y_true: pd.Series | np.ndarray,
        horizon: int = 20,
    ) -> StackingDataset:
        """
        Build stacking dataset with aligned OOF predictions.

        Args:
            oof_predictions: Dict of model_name -> OOFPrediction
            y_true: True labels (full length)
            horizon: Label horizon for metadata

        Returns:
            StackingDataset ready for meta-learner training
        """
        if not oof_predictions:
            raise ValueError("No OOF predictions provided")

        # Compute alignment
        alignment_info = self._compute_alignment(oof_predictions)
        max_offset = alignment_info["max_offset"]
        n_common = alignment_info["n_common_samples"]

        logger.info(f"Aligning {len(oof_predictions)} models")
        logger.info(f"  Max offset: {max_offset}")
        logger.info(f"  Common samples: {n_common}")

        # Build feature DataFrame
        features = {}
        feature_names = []

        for model_name, oof_pred in oof_predictions.items():
            # Get aligned probabilities
            probs = self._align_model_predictions(oof_pred, max_offset, n_common)

            # Add probability features
            for i, class_name in enumerate(self.class_names):
                col_name = f"{model_name}_prob_{class_name}"
                features[col_name] = probs[:, i]
                feature_names.append(col_name)

        X = pd.DataFrame(features)

        # Add derived features
        if self.add_derived_features:
            derived = self._compute_derived_features(X, list(oof_predictions.keys()))
            X = pd.concat([X, derived], axis=1)
            feature_names.extend(derived.columns.tolist())

        # Align labels
        y = np.asarray(y_true)[max_offset:max_offset + n_common]

        n_original = len(y_true)

        logger.info(f"Built stacking dataset: {X.shape}")
        logger.info(f"  Coverage: {n_common / n_original:.2%}")

        return StackingDataset(
            X=X,
            y=y,
            base_models=list(oof_predictions.keys()),
            alignment_offset=max_offset,
            n_original_samples=n_original,
            feature_names=feature_names,
            horizon=horizon,
        )

    def _compute_alignment(
        self,
        oof_predictions: dict[str, OOFPrediction],
    ) -> dict[str, Any]:
        """Compute alignment parameters."""
        max_offset = 0
        min_end = float("inf")

        for model_name, oof_pred in oof_predictions.items():
            offset = oof_pred.alignment_offset
            n_samples = oof_pred.n_total_samples or len(oof_pred.predictions)

            max_offset = max(max_offset, offset)
            min_end = min(min_end, n_samples)

        n_common = int(min_end) - max_offset

        return {
            "max_offset": max_offset,
            "common_end": int(min_end),
            "n_common_samples": n_common,
        }

    def _align_model_predictions(
        self,
        oof_pred: OOFPrediction,
        max_offset: int,
        n_common: int,
    ) -> np.ndarray:
        """Extract aligned slice from model predictions."""
        probs = oof_pred.get_probabilities()
        model_offset = oof_pred.alignment_offset
        local_start = max_offset - model_offset
        return probs[local_start:local_start + n_common]

    def _compute_derived_features(
        self,
        X: pd.DataFrame,
        model_names: list[str],
    ) -> pd.DataFrame:
        """Compute ensemble diversity features."""
        derived = {}

        # Mean confidence across models
        all_probs = []
        for model in model_names:
            probs = X[[f"{model}_prob_{c}" for c in self.class_names]].values
            all_probs.append(probs)

        stacked = np.stack(all_probs, axis=0)  # (n_models, n_samples, n_classes)

        # Max probability per model, then mean across models
        max_probs = np.max(stacked, axis=2)  # (n_models, n_samples)
        derived["mean_confidence"] = np.mean(max_probs, axis=0)

        # Prediction agreement (entropy of majority vote)
        predictions = np.argmax(stacked, axis=2)  # (n_models, n_samples)
        agreement = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            agreement.append(np.max(counts) / len(votes))
        derived["prediction_agreement"] = agreement

        # Probability entropy (uncertainty measure)
        mean_probs = np.mean(stacked, axis=0)  # (n_samples, n_classes)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        derived["prediction_entropy"] = entropy

        return pd.DataFrame(derived)


__all__ = ["HeterogeneousStackingBuilder", "StackingDataset"]
```

---

## Task 4.2: Meta-Learner Registry

### File: `src/ensemble/meta_learners/__init__.py`

```python
"""
Meta-Learners for stacking ensembles.

Available meta-learners:
- ridge_meta: L2-regularized logistic regression
- mlp_meta: 2-layer MLP with dropout
- xgboost_meta: XGBoost with calibration
- calibrated_meta: Isotonic regression calibrator
"""

from src.ensemble.meta_learners.ridge import RidgeMeta
from src.ensemble.meta_learners.mlp import MLPMeta
from src.ensemble.meta_learners.xgboost_meta import XGBoostMeta
from src.ensemble.meta_learners.calibrated import CalibratedMeta

META_LEARNER_REGISTRY = {
    "ridge_meta": RidgeMeta,
    "mlp_meta": MLPMeta,
    "xgboost_meta": XGBoostMeta,
    "calibrated_meta": CalibratedMeta,
}


def get_meta_learner(name: str, **kwargs):
    """Get meta-learner by name."""
    if name not in META_LEARNER_REGISTRY:
        raise ValueError(f"Unknown meta-learner: {name}")
    return META_LEARNER_REGISTRY[name](**kwargs)


__all__ = [
    "RidgeMeta",
    "MLPMeta",
    "XGBoostMeta",
    "CalibratedMeta",
    "META_LEARNER_REGISTRY",
    "get_meta_learner",
]
```

---

## Task 4.3: Meta-Learner Implementations

### File: `src/ensemble/meta_learners/ridge.py`

```python
"""Ridge meta-learner - L2-regularized logistic regression."""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel, PredictionOutput


class RidgeMeta(BaseModel):
    """
    Ridge-regularized meta-learner for stacking.

    Simple, fast, and robust baseline meta-learner.
    Uses CV to select regularization strength.
    """

    def __init__(self, alphas: list[float] | None = None):
        super().__init__()
        self.alphas = alphas or [0.1, 1.0, 10.0, 100.0]
        self._scaler = StandardScaler()
        self._model = RidgeClassifierCV(alphas=self.alphas)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "RidgeMeta":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionOutput:
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        # Ridge doesn't have predict_proba, use decision function
        decision = self._model.decision_function(X_scaled)
        # Softmax to get probabilities
        probs = self._softmax(decision)
        return PredictionOutput(
            class_predictions=preds,
            class_probabilities=probs,
            confidence=np.max(probs, axis=1),
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### File: `src/ensemble/meta_learners/mlp.py`

```python
"""MLP meta-learner - 2-layer neural network."""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel, PredictionOutput


class MLPMeta(BaseModel):
    """
    MLP meta-learner with dropout.

    Captures non-linear interactions between base model predictions.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_iter: int = 500,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MLPMeta":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionOutput:
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        probs = self._model.predict_proba(X_scaled)
        return PredictionOutput(
            class_predictions=preds,
            class_probabilities=probs,
            confidence=np.max(probs, axis=1),
        )
```

### File: `src/ensemble/meta_learners/xgboost_meta.py`

```python
"""XGBoost meta-learner with calibration."""

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from src.models.base import BaseModel, PredictionOutput


class XGBoostMeta(BaseModel):
    """
    XGBoost meta-learner with optional isotonic calibration.

    Strong performer when base models are diverse.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        calibrate: bool = True,
    ):
        super().__init__()
        self.calibrate = calibrate
        self._base_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self._model = (
            CalibratedClassifierCV(self._base_model, cv=3, method="isotonic")
            if calibrate else self._base_model
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "XGBoostMeta":
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionOutput:
        preds = self._model.predict(X)
        probs = self._model.predict_proba(X)
        return PredictionOutput(
            class_predictions=preds,
            class_probabilities=probs,
            confidence=np.max(probs, axis=1),
        )
```

---

## Task 4.4: OOFCache

### File: `src/ensemble/oof_cache.py`

```python
"""OOF prediction caching for efficient re-use."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from src.cross_validation.oof_core import OOFPrediction


class OOFCache:
    """
    Cache OOF predictions for efficient re-use.

    Keys are computed from:
    - Model name
    - Model config hash
    - Data hash (subset of feature hashes)
    - CV config hash
    """

    def __init__(self, cache_dir: Path | str = ".oof_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(
        self,
        model_name: str,
        model_config: dict,
        data_hash: str,
        cv_config: dict,
    ) -> str:
        """Compute cache key."""
        key_data = {
            "model": model_name,
            "model_config": model_config,
            "data": data_hash,
            "cv": cv_config,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, key: str) -> OOFPrediction | None:
        """Get cached OOF prediction."""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, key: str, oof_pred: OOFPrediction) -> None:
        """Cache OOF prediction."""
        cache_path = self.cache_dir / f"{key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(oof_pred, f)

    def clear(self) -> None:
        """Clear all cached predictions."""
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
```

---

## Task 4.5: Training Flow

### Complete Stacking Training Flow

```python
def train_heterogeneous_ensemble(
    container: TimeSeriesDataContainer,
    base_models: list[str],
    meta_learner: str = "ridge_meta",
    horizon: int = 20,
) -> dict[str, Any]:
    """
    Train heterogeneous stacking ensemble.

    Args:
        container: Data container with train/val/test
        base_models: List of base model names
        meta_learner: Meta-learner name
        horizon: Prediction horizon

    Returns:
        Dict with trained models and metrics
    """
    from src.training.orchestrator import TrainingOrchestrator, OrchestrationConfig
    from src.ensemble.stacking import HeterogeneousStackingBuilder
    from src.ensemble.meta_learners import get_meta_learner

    # 1. Train base models and generate OOF
    config = OrchestrationConfig(
        models=base_models,
        horizons=[horizon],
        generate_oof=True,
        build_ensemble=False,  # We'll build manually
    )
    orchestrator = TrainingOrchestrator(config)
    base_results = orchestrator.train(container)
    oof_predictions = base_results["oof_predictions"]

    # 2. Build stacking dataset with alignment
    builder = HeterogeneousStackingBuilder()
    X_train, y_train, _ = container.get_sklearn_arrays("train")
    stacking_ds = builder.build(
        oof_predictions=oof_predictions,
        y_true=y_train,
        horizon=horizon,
    )

    # 3. Train meta-learner
    meta = get_meta_learner(meta_learner)
    meta.fit(stacking_ds.X.values, stacking_ds.y)

    # 4. Evaluate on validation set
    # (Would need to generate aligned val predictions)

    return {
        "base_models": base_models,
        "meta_learner": meta,
        "stacking_dataset": stacking_ds,
        "alignment_offset": stacking_ds.alignment_offset,
        "coverage": stacking_ds.coverage,
    }
```

---

## Implementation Checklist

### Task 4.1: HeterogeneousStackingBuilder
- [ ] Create `src/ensemble/stacking.py`
- [ ] `StackingDataset` dataclass
- [ ] `HeterogeneousStackingBuilder.build()`
- [ ] `_compute_alignment()` method
- [ ] `_align_model_predictions()` method
- [ ] `_compute_derived_features()` method

### Task 4.2: Meta-Learner Registry
- [ ] Create `src/ensemble/meta_learners/__init__.py`
- [ ] `META_LEARNER_REGISTRY` dict
- [ ] `get_meta_learner()` factory

### Task 4.3: Meta-Learner Implementations
- [ ] `RidgeMeta` - L2-regularized
- [ ] `MLPMeta` - 2-layer neural network
- [ ] `XGBoostMeta` - with calibration
- [ ] `CalibratedMeta` - isotonic regression

### Task 4.4: OOFCache
- [ ] Create `src/ensemble/oof_cache.py`
- [ ] `get_key()` for deterministic keys
- [ ] `get()` and `put()` methods
- [ ] `clear()` method

### Task 4.5: Integration
- [ ] `train_heterogeneous_ensemble()` function
- [ ] End-to-end test with mixed models
- [ ] Validation set evaluation

---

## Summary

| Component | Purpose |
|-----------|---------|
| HeterogeneousStackingBuilder | Build aligned stacking datasets |
| OOFAlignmentValidator | Compute alignment across model types |
| RidgeMeta | Fast baseline meta-learner |
| MLPMeta | Non-linear interactions |
| XGBoostMeta | Strong with diverse bases |
| CalibratedMeta | Probability calibration |
| OOFCache | Efficient OOF re-use |
