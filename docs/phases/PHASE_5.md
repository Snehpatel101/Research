# Phase 5: Test Set Evaluation and Deployment Readiness

## Current Status: PLANNED (Not Implemented)

**IMPLEMENTATION STATUS:**
- ❌ Inference pipeline - Not implemented
- ❌ Test set evaluator - Not implemented
- ❌ Lookahead verification - Not implemented
- ❌ Drift detector - Not implemented
- ❌ Performance monitor - Not implemented
- ❌ Scripts (`scripts/evaluate_test.py`) - Do not exist
- ✅ Test data - **AVAILABLE** at `data/splits/scaled/test_scaled.parquet`

**DEPENDENCIES:**
- ✅ Phase 1 (Data Pipeline) - **COMPLETE** - Test set prepared
- ❌ Phase 2 (Model Factory) - **NOT STARTED** - Required
- ❌ Phase 3 (Cross-Validation) - **NOT STARTED** - Required
- ❌ Phase 4 (Ensemble) - **NOT STARTED** - Required

**BLOCKED BY:**
- Phase 5 requires complete ensemble from Phase 2-4
- Test set must remain untouched until Phases 2-4 are complete

**NEXT STEPS (After Phase 2-4 Complete):**
1. Implement InferencePipeline for production inference
2. Create test evaluation script: `scripts/evaluate_test.py`
3. Run lookahead verification tests
4. Evaluate on hold-out test set (ONE TIME ONLY)
5. Set up drift detection and monitoring
6. Serialize pipeline for deployment

Phase 5 evaluates the complete system on the final hold-out test set. This is the "moment of truth" - true out-of-sample performance on data never seen by any model or meta-learner. This phase also covers model serialization, inference pipeline construction, production deployment, and monitoring infrastructure.

---

## Overview

```
Trained Models (Phase 2-4)  -->  Inference Pipeline  -->  Test Set  -->  Final Metrics
                                       |                      |
                               [Feature + Model + Ensemble]   |
                                       |                      v
                                       +----------> Production Deployment
                                                           |
                                                    Monitoring & Drift Detection
```

**Key Principles:**
1. **Test set integrity** - Never touch test set until this phase
2. **Honest evaluation** - Report results without iteration
3. **Production parity** - Inference pipeline matches production
4. **Monitoring from day one** - Set up drift detection before deployment

---

## Critical Requirements

### Test Set Integrity

**The test set must be completely untouched until Phase 5.**

```python
# RULES FOR TEST SET
# 1. NO hyperparameter tuning based on test results
# 2. NO peeking at test labels during development
# 3. NO feature selection based on test performance
# 4. If test performance is poor, document honestly - do not iterate

def verify_test_set_integrity(test_df: pd.DataFrame, audit_log: Path) -> bool:
    """
    Verify test set has not been accessed before Phase 5.

    Check:
    1. File modification time < Phase 5 start
    2. No access logs before Phase 5
    3. Hash matches original
    """
    import hashlib
    import json

    with open(audit_log) as f:
        audit = json.load(f)

    current_hash = hashlib.md5(test_df.to_json().encode()).hexdigest()

    if current_hash != audit["original_hash"]:
        raise ValueError("Test set has been modified!")

    if audit["access_count"] > 0:
        raise ValueError(f"Test set accessed {audit['access_count']} times before Phase 5!")

    return True
```

### No-Lookahead Verification

Before any evaluation, verify no lookahead bias exists:

```python
import numpy as np


def verify_no_lookahead(
    pipeline: "InferencePipeline",
    test_df: pd.DataFrame,
    sample_indices: List[int] = None,
    n_samples: int = 100
) -> Dict[str, any]:
    """
    Verify predictions don't use future data.

    Method: Corrupt future data and verify predictions unchanged.
    """
    if sample_indices is None:
        # Random sample of indices (not at end of data)
        valid_indices = np.arange(100, len(test_df) - 100)
        sample_indices = np.random.choice(valid_indices, n_samples, replace=False)

    violations = []

    for idx in sample_indices:
        # Original prediction
        pred_original = pipeline.predict_single(test_df, idx)

        # Corrupt future data
        df_corrupted = test_df.copy()
        df_corrupted.iloc[idx + 1:] = 999999  # Obvious corruption

        # Prediction with corrupted future
        pred_corrupted = pipeline.predict_single(df_corrupted, idx)

        # Should be identical
        if not np.allclose(pred_original, pred_corrupted, rtol=1e-5):
            violations.append({
                "index": idx,
                "original": pred_original.tolist(),
                "corrupted": pred_corrupted.tolist()
            })

    return {
        "passed": len(violations) == 0,
        "n_tested": len(sample_indices),
        "n_violations": len(violations),
        "violations": violations[:10]  # First 10 for debugging
    }
```

---

## Inference Pipeline

### Complete Pipeline Implementation

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import json


@dataclass
class InferencePipelineConfig:
    """Configuration for inference pipeline."""
    base_models: List[str]           # ["xgboost", "lightgbm", "lstm"]
    horizons: List[int]              # [5, 10, 15, 20]
    sequence_length: int = 60        # For sequential models
    feature_set: str = "boosting_optimal"
    use_ensemble: bool = True
    models_dir: Path = Path("models")
    scalers_dir: Path = Path("data/splits/scaled")


class InferencePipeline:
    """
    Production-ready inference pipeline.

    Handles:
    - Feature creation from raw OHLCV
    - Model-specific preprocessing
    - Base model predictions
    - Ensemble aggregation
    """

    def __init__(self, config: InferencePipelineConfig):
        self.config = config
        self.base_models = {}
        self.meta_learners = {}
        self.scaler = None
        self.feature_columns = None

        self._load_models()
        self._load_scaler()

    def _load_models(self):
        """Load all trained models."""
        for model_name in self.config.base_models:
            self.base_models[model_name] = {}

            for horizon in self.config.horizons:
                model_path = (
                    self.config.models_dir /
                    f"{model_name}_h{horizon}" /
                    "best_model.pkl"
                )
                with open(model_path, "rb") as f:
                    self.base_models[model_name][horizon] = pickle.load(f)

        # Load meta-learners
        if self.config.use_ensemble:
            for horizon in self.config.horizons:
                meta_path = self.config.models_dir / f"ensemble/h{horizon}/meta_learner.pkl"
                with open(meta_path, "rb") as f:
                    self.meta_learners[horizon] = pickle.load(f)

    def _load_scaler(self):
        """Load feature scaler."""
        scaler_path = self.config.scalers_dir / "feature_scaler.pkl"
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load feature columns
        meta_path = self.config.scalers_dir / "scaling_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        return_base_predictions: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all samples in DataFrame.

        Args:
            df: DataFrame with OHLCV and features
            horizon: Prediction horizon
            return_base_predictions: Include individual base model predictions

        Returns:
            Dictionary with predictions and probabilities
        """
        # Extract features
        X = df[self.feature_columns].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        base_predictions = {}
        for model_name in self.config.base_models:
            model = self.base_models[model_name][horizon]

            # Handle sequential vs tabular models
            if hasattr(model, 'requires_sequences') and model.requires_sequences:
                X_seq = self._create_sequences(X_scaled)
                probs = model.predict_proba(X_seq)
            else:
                probs = model.predict_proba(X_scaled)

            base_predictions[model_name] = probs

        # Ensemble predictions
        if self.config.use_ensemble:
            ensemble_probs = self._apply_ensemble(base_predictions, horizon)
        else:
            # Simple averaging fallback
            ensemble_probs = np.mean(list(base_predictions.values()), axis=0)

        # Predicted classes
        ensemble_preds = ensemble_probs.argmax(axis=1) - 1  # -1, 0, 1

        result = {
            "predictions": ensemble_preds,
            "probabilities": ensemble_probs,
            "confidence": ensemble_probs.max(axis=1)
        }

        if return_base_predictions:
            result["base_predictions"] = base_predictions

        return result

    def predict_single(
        self,
        df: pd.DataFrame,
        idx: int,
        horizon: int = 20
    ) -> np.ndarray:
        """Predict for single sample (for lookahead verification)."""
        # Get features for single sample
        X = df[self.feature_columns].iloc[idx:idx+1].values
        X_scaled = self.scaler.transform(X)

        base_probs = []
        for model_name in self.config.base_models:
            model = self.base_models[model_name][horizon]
            probs = model.predict_proba(X_scaled)
            base_probs.append(probs)

        if self.config.use_ensemble:
            stacking_input = np.hstack(base_probs)
            return self.meta_learners[horizon].predict_proba(stacking_input)
        else:
            return np.mean(base_probs, axis=0)

    def _apply_ensemble(
        self,
        base_predictions: Dict[str, np.ndarray],
        horizon: int
    ) -> np.ndarray:
        """Apply meta-learner to base predictions."""
        # Stack base model probabilities
        stacking_input = np.hstack([
            base_predictions[model]
            for model in self.config.base_models
        ])

        return self.meta_learners[horizon].predict_proba(stacking_input)

    def _create_sequences(
        self,
        X: np.ndarray,
        seq_len: int = None
    ) -> np.ndarray:
        """Create sequences for sequential models."""
        if seq_len is None:
            seq_len = self.config.sequence_length

        n_samples = len(X) - seq_len + 1
        n_features = X.shape[1]

        sequences = np.zeros((n_samples, seq_len, n_features))
        for i in range(n_samples):
            sequences[i] = X[i:i + seq_len]

        return sequences
```

### Pipeline Serialization

```python
def serialize_pipeline(
    pipeline: InferencePipeline,
    output_dir: Path
) -> None:
    """
    Serialize complete inference pipeline for deployment.

    Creates self-contained artifact with all models and configs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "base_models": pipeline.config.base_models,
        "horizons": pipeline.config.horizons,
        "sequence_length": pipeline.config.sequence_length,
        "feature_set": pipeline.config.feature_set,
        "use_ensemble": pipeline.config.use_ensemble
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save scaler
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(pipeline.scaler, f)

    # Save feature columns
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(pipeline.feature_columns, f)

    # Save base models
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for model_name, horizons in pipeline.base_models.items():
        for horizon, model in horizons.items():
            model_path = models_dir / f"{model_name}_h{horizon}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    # Save meta-learners
    if pipeline.config.use_ensemble:
        ensemble_dir = output_dir / "ensemble"
        ensemble_dir.mkdir(exist_ok=True)

        for horizon, meta in pipeline.meta_learners.items():
            meta_path = ensemble_dir / f"meta_h{horizon}.pkl"
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)

    # Create manifest
    manifest = {
        "version": "1.0.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "base_models": list(pipeline.base_models.keys()),
        "horizons": pipeline.config.horizons,
        "n_features": len(pipeline.feature_columns)
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_serialized_pipeline(artifact_dir: Path) -> InferencePipeline:
    """Load serialized pipeline for inference."""
    with open(artifact_dir / "config.json") as f:
        config_dict = json.load(f)

    config = InferencePipelineConfig(**config_dict)
    config.models_dir = artifact_dir
    config.scalers_dir = artifact_dir

    return InferencePipeline(config)
```

---

## Ensemble Inference Pipeline

### Model Serving Considerations

Different model types have vastly different inference characteristics:

| Model Type | Inference Time | Memory | GPU Needed | Batch Friendly |
|------------|---------------|--------|------------|----------------|
| XGBoost | < 1ms | Low (50-200MB) | No | Yes |
| LightGBM | < 1ms | Low (50-200MB) | No | Yes |
| CatBoost | < 1ms | Low (50-200MB) | No | Yes |
| LSTM/GRU | 5-20ms | Medium (200-500MB) | Optional | Yes |
| TCN | 2-10ms | Medium (200-500MB) | Recommended | Yes |
| PatchTST | 10-50ms | High (500MB-2GB) | Required | Yes |
| iTransformer | 10-50ms | High (500MB-2GB) | Required | Yes |
| TFT | 20-100ms | High (1-3GB) | Required | Less |

### Memory/Latency Tradeoffs

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class DeploymentProfile(Enum):
    """Deployment scenarios with different constraints."""
    LOW_LATENCY = "low_latency"       # < 10ms total, no GPU
    BALANCED = "balanced"              # < 50ms total, optional GPU
    HIGH_ACCURACY = "high_accuracy"    # < 200ms total, GPU available


@dataclass
class ModelProfile:
    """Resource profile for a model."""
    name: str
    inference_ms: float
    memory_mb: float
    requires_gpu: bool
    requires_sequences: bool
    batch_size_limit: int = 512


MODEL_PROFILES = {
    "xgboost": ModelProfile("xgboost", 0.5, 150, False, False),
    "lightgbm": ModelProfile("lightgbm", 0.3, 100, False, False),
    "catboost": ModelProfile("catboost", 0.5, 150, False, False),
    "lstm": ModelProfile("lstm", 10, 300, False, True, 256),
    "gru": ModelProfile("gru", 8, 250, False, True, 256),
    "tcn": ModelProfile("tcn", 5, 350, False, True, 256),
    "patchtst": ModelProfile("patchtst", 25, 800, True, True, 128),
    "itransformer": ModelProfile("itransformer", 30, 1000, True, True, 128),
    "tft": ModelProfile("tft", 50, 1500, True, True, 64),
}


def select_models_for_profile(
    profile: DeploymentProfile,
    available_models: List[str],
    target_diversity: float = 0.3
) -> List[str]:
    """
    Select models that fit deployment constraints.

    Args:
        profile: Deployment scenario
        available_models: Models we have trained
        target_diversity: Minimum diversity between selected models

    Returns:
        List of model names that fit the profile
    """
    constraints = {
        DeploymentProfile.LOW_LATENCY: {
            "max_total_ms": 10,
            "max_memory_mb": 500,
            "gpu_allowed": False
        },
        DeploymentProfile.BALANCED: {
            "max_total_ms": 50,
            "max_memory_mb": 2000,
            "gpu_allowed": True
        },
        DeploymentProfile.HIGH_ACCURACY: {
            "max_total_ms": 200,
            "max_memory_mb": 8000,
            "gpu_allowed": True
        }
    }

    c = constraints[profile]
    selected = []
    total_ms = 0
    total_mb = 0

    # Sort by inference speed for greedy selection
    candidates = sorted(
        available_models,
        key=lambda m: MODEL_PROFILES[m].inference_ms
    )

    for model in candidates:
        p = MODEL_PROFILES[model]

        # Check constraints
        if not c["gpu_allowed"] and p.requires_gpu:
            continue
        if total_ms + p.inference_ms > c["max_total_ms"]:
            continue
        if total_mb + p.memory_mb > c["max_memory_mb"]:
            continue

        selected.append(model)
        total_ms += p.inference_ms
        total_mb += p.memory_mb

    return selected


# Example configurations
DEPLOYMENT_CONFIGS = {
    "low_latency": {
        "models": ["xgboost", "lightgbm", "catboost"],
        "meta_learner": "logistic",
        "expected_latency_ms": 3,
        "expected_memory_mb": 400,
        "description": "Boosting-only ensemble for < 5ms inference"
    },
    "balanced": {
        "models": ["xgboost", "lightgbm", "tcn"],
        "meta_learner": "logistic",
        "expected_latency_ms": 15,
        "expected_memory_mb": 600,
        "description": "Hybrid ensemble with temporal patterns"
    },
    "high_accuracy": {
        "models": ["xgboost", "lightgbm", "tcn", "patchtst"],
        "meta_learner": "xgboost",
        "expected_latency_ms": 60,
        "expected_memory_mb": 2000,
        "description": "Full hybrid for maximum accuracy"
    }
}
```

### Optimized Ensemble Inference Pipeline

```python
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import time


class OptimizedEnsemblePipeline:
    """
    Production-optimized ensemble inference.

    Optimizations:
    1. Parallel base model inference
    2. Batch processing for neural models
    3. Pre-allocated output buffers
    4. Optional GPU batching for transformers
    """

    def __init__(
        self,
        config: Dict,
        models_dir: str,
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        self.models = {}
        self.meta_learner = None
        self.scaler = None

        # Thread pool for parallel inference
        self.executor = ThreadPoolExecutor(max_workers=4)

        self._load_models(models_dir)

    def _load_models(self, models_dir: str):
        """Load all models and meta-learner."""
        import pickle
        from pathlib import Path

        models_path = Path(models_dir)

        for model_name in self.config["models"]:
            model_path = models_path / f"{model_name}" / "best_model.pkl"
            with open(model_path, "rb") as f:
                self.models[model_name] = pickle.load(f)

        # Load meta-learner
        meta_path = models_path / "ensemble" / "meta_learner.pkl"
        with open(meta_path, "rb") as f:
            self.meta_learner = pickle.load(f)

        # Load scaler
        scaler_path = models_path / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(
        self,
        X: np.ndarray,
        return_timing: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix (already scaled)
            return_timing: Include timing breakdown

        Returns:
            Dict with predictions, probabilities, confidence
        """
        timings = {}
        start = time.perf_counter()

        # 1. Generate base model predictions in parallel
        base_predictions = self._parallel_base_inference(X, timings)

        # 2. Build meta-features
        meta_start = time.perf_counter()
        meta_features = self._build_meta_features(base_predictions)
        timings["meta_features_ms"] = (time.perf_counter() - meta_start) * 1000

        # 3. Meta-learner prediction
        meta_pred_start = time.perf_counter()
        ensemble_probs = self.meta_learner.predict_proba(meta_features)
        ensemble_preds = ensemble_probs.argmax(axis=1) - 1  # -1, 0, 1
        timings["meta_prediction_ms"] = (time.perf_counter() - meta_pred_start) * 1000

        total_ms = (time.perf_counter() - start) * 1000
        timings["total_ms"] = total_ms

        result = {
            "predictions": ensemble_preds,
            "probabilities": ensemble_probs,
            "confidence": ensemble_probs.max(axis=1)
        }

        if return_timing:
            result["timings"] = timings

        return result

    def _parallel_base_inference(
        self,
        X: np.ndarray,
        timings: Dict
    ) -> Dict[str, np.ndarray]:
        """Run base model inference in parallel."""

        def run_model(model_name: str) -> tuple:
            start = time.perf_counter()
            model = self.models[model_name]
            probs = model.predict_proba(X)
            elapsed = (time.perf_counter() - start) * 1000
            return model_name, probs, elapsed

        # Submit all models to thread pool
        futures = [
            self.executor.submit(run_model, name)
            for name in self.config["models"]
        ]

        # Collect results
        base_predictions = {}
        for future in futures:
            name, probs, elapsed = future.result()
            base_predictions[name] = probs
            timings[f"{name}_ms"] = elapsed

        return base_predictions

    def _build_meta_features(
        self,
        base_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build meta-features from base predictions.

        Features:
        - Raw probabilities (n_models * 3)
        - Per-model confidence
        - Agreement features
        """
        n_samples = len(list(base_predictions.values())[0])
        n_models = len(base_predictions)

        # Pre-allocate
        n_features = n_models * 3 + n_models + 3  # probs + conf + agreement
        meta_features = np.zeros((n_samples, n_features))

        idx = 0

        # Raw probabilities
        for model_name in self.config["models"]:
            probs = base_predictions[model_name]
            meta_features[:, idx:idx+3] = probs
            idx += 3

        # Confidence (max prob per model)
        for model_name in self.config["models"]:
            probs = base_predictions[model_name]
            meta_features[:, idx] = probs.max(axis=1)
            idx += 1

        # Agreement features
        preds = np.column_stack([
            base_predictions[m].argmax(axis=1)
            for m in self.config["models"]
        ])

        # All agree
        meta_features[:, idx] = (preds[:, 0:1] == preds).all(axis=1).astype(float)
        idx += 1

        # Average confidence
        confidences = np.column_stack([
            base_predictions[m].max(axis=1)
            for m in self.config["models"]
        ])
        meta_features[:, idx] = confidences.mean(axis=1)
        idx += 1

        # Confidence spread
        meta_features[:, idx] = confidences.std(axis=1)

        return meta_features

    def benchmark(self, n_samples: int = 1000, n_iterations: int = 10) -> Dict:
        """Benchmark inference performance."""
        # Generate random test data
        n_features = 150  # Typical feature count
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        latencies = []
        for _ in range(n_iterations):
            result = self.predict(X, return_timing=True)
            latencies.append(result["timings"]["total_ms"])

        return {
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_samples_per_sec": n_samples / (np.mean(latencies) / 1000)
        }
```

### GPU Batching for Transformers

For transformer models, batch GPU inference significantly:

```python
import torch


class GPUBatchedTransformerInference:
    """
    Optimized transformer inference with GPU batching.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        max_batch_size: int = 128
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch_size = max_batch_size

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference with automatic chunking.
        """
        n_samples = len(X)
        all_probs = []

        for i in range(0, n_samples, self.max_batch_size):
            batch = X[i:i + self.max_batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(self.device)

            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)
```

---

## Test Set Evaluation

### Comprehensive Evaluation

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    horizon: int
    symbol: Optional[str]

    # Classification metrics
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str

    # Trading metrics
    sharpe_ratio: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float

    # Regime breakdown
    regime_performance: Dict[str, Dict]

    # Comparison
    val_metrics: Dict[str, float]
    val_test_gap: Dict[str, float]


class TestSetEvaluator:
    """
    Comprehensive test set evaluation.
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        horizons: List[int] = None
    ):
        self.pipeline = pipeline
        self.horizons = horizons or [5, 10, 15, 20]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        val_metrics: Dict[int, Dict] = None
    ) -> Dict[int, EvaluationResult]:
        """
        Run complete test set evaluation.

        Args:
            test_df: Test DataFrame with features and labels
            val_metrics: Validation metrics for comparison

        Returns:
            Evaluation results per horizon
        """
        results = {}

        for horizon in self.horizons:
            print(f"Evaluating H{horizon}...")

            # Get predictions
            preds = self.pipeline.predict(test_df, horizon)

            # Get true labels
            label_col = f"label_h{horizon}"
            y_true = test_df[label_col].values

            # Classification metrics
            class_metrics = self._compute_classification_metrics(
                y_true, preds["predictions"], preds["probabilities"]
            )

            # Trading metrics
            trading_metrics = self._compute_trading_metrics(
                test_df, y_true, preds["predictions"], preds["probabilities"]
            )

            # Regime breakdown
            regime_perf = self._compute_regime_performance(
                test_df, y_true, preds["predictions"]
            )

            # Validation comparison
            val_test_gap = {}
            if val_metrics and horizon in val_metrics:
                val_test_gap = {
                    "sharpe_gap": val_metrics[horizon]["sharpe"] - trading_metrics["sharpe_ratio"],
                    "f1_gap": val_metrics[horizon]["f1"] - class_metrics["macro_f1"],
                    "sharpe_gap_pct": (
                        (val_metrics[horizon]["sharpe"] - trading_metrics["sharpe_ratio"]) /
                        val_metrics[horizon]["sharpe"] * 100
                    )
                }

            results[horizon] = EvaluationResult(
                horizon=horizon,
                symbol=None,
                **class_metrics,
                **trading_metrics,
                regime_performance=regime_perf,
                val_metrics=val_metrics.get(horizon, {}) if val_metrics else {},
                val_test_gap=val_test_gap
            )

        return results

    def evaluate_per_symbol(
        self,
        test_df: pd.DataFrame,
        symbols: List[str] = None
    ) -> Dict[str, Dict[int, EvaluationResult]]:
        """Evaluate separately for each symbol."""
        if symbols is None:
            symbols = test_df["symbol"].unique().tolist()

        results = {}
        for symbol in symbols:
            symbol_df = test_df[test_df["symbol"] == symbol]
            results[symbol] = self.evaluate(symbol_df)

        return results

    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Compute classification metrics."""
        labels = [-1, 0, 1]
        target_names = ["short", "neutral", "long"]

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels),
            "per_class_f1": {
                name: f1_score(y_true, y_pred, average=None, labels=labels)[i]
                for i, name in enumerate(target_names)
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, target_names=target_names
            )
        }

    def _compute_trading_metrics(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Compute trading performance metrics."""
        # Simulate trading based on predictions
        returns = df["return_1"].values  # Next bar return

        # Position: prediction directly (-1, 0, 1)
        positions = y_pred.astype(float)

        # Strategy returns
        strategy_returns = positions[:-1] * returns[1:]  # Lag by 1

        # Remove NaN
        strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

        # Sharpe ratio (annualized, assuming 5-min bars)
        bars_per_year = 252 * 78  # 78 5-min bars per trading day
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe = (
                strategy_returns.mean() / strategy_returns.std() *
                np.sqrt(bars_per_year)
            )
        else:
            sharpe = 0.0

        # Cumulative returns
        cumulative = (1 + strategy_returns).cumprod()

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]

        win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0

        # Profit factor
        if len(losing_trades) > 0 and losing_trades.sum() != 0:
            profit_factor = winning_trades.sum() / abs(losing_trades.sum())
        else:
            profit_factor = float('inf') if len(winning_trades) > 0 else 0

        # Annualized return
        if len(cumulative) > 0:
            total_return = cumulative[-1] - 1
            n_years = len(strategy_returns) / bars_per_year
            annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        else:
            annualized_return = 0

        return {
            "sharpe_ratio": sharpe,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": int((positions != 0).sum()),
            "avg_trade_return": strategy_returns.mean() if len(strategy_returns) > 0 else 0
        }

    def _compute_regime_performance(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute performance by market regime."""
        regimes = {}

        # Volatility regimes
        if "volatility_regime" in df.columns:
            for regime in df["volatility_regime"].unique():
                mask = df["volatility_regime"] == regime
                if mask.sum() > 100:  # Minimum samples
                    regimes[f"vol_{regime}"] = {
                        "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                        "f1": f1_score(y_true[mask], y_pred[mask], average="macro"),
                        "n_samples": int(mask.sum())
                    }

        # Trend regimes
        if "trend_regime" in df.columns:
            for regime in df["trend_regime"].unique():
                mask = df["trend_regime"] == regime
                if mask.sum() > 100:
                    regimes[f"trend_{regime}"] = {
                        "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                        "f1": f1_score(y_true[mask], y_pred[mask], average="macro"),
                        "n_samples": int(mask.sum())
                    }

        return regimes
```

---

## Generalization Analysis

### Validation vs Test Comparison

```python
def analyze_generalization(
    val_results: Dict[int, Dict],
    test_results: Dict[int, EvaluationResult]
) -> pd.DataFrame:
    """
    Compare validation and test performance to assess generalization.

    Acceptable gap: < 15%
    Concerning gap: > 20%
    """
    comparison = []

    for horizon in test_results:
        test = test_results[horizon]
        val = val_results.get(horizon, {})

        if not val:
            continue

        comparison.append({
            "horizon": horizon,
            "val_sharpe": val.get("sharpe", 0),
            "test_sharpe": test.sharpe_ratio,
            "sharpe_gap": val.get("sharpe", 0) - test.sharpe_ratio,
            "sharpe_gap_pct": (
                (val.get("sharpe", 0) - test.sharpe_ratio) /
                val.get("sharpe", 1) * 100
            ),
            "val_f1": val.get("f1", 0),
            "test_f1": test.macro_f1,
            "f1_gap": val.get("f1", 0) - test.macro_f1,
            "f1_gap_pct": (
                (val.get("f1", 0) - test.macro_f1) /
                val.get("f1", 1) * 100
            ),
            "generalization_grade": grade_generalization(
                val.get("sharpe", 0) - test.sharpe_ratio,
                val.get("sharpe", 1)
            )
        })

    return pd.DataFrame(comparison)


def grade_generalization(gap: float, val_value: float) -> str:
    """Grade generalization based on val-test gap."""
    if val_value == 0:
        return "Unknown"

    gap_pct = abs(gap / val_value) * 100

    if gap_pct < 5:
        return "Excellent"
    elif gap_pct < 10:
        return "Good"
    elif gap_pct < 15:
        return "Acceptable"
    elif gap_pct < 25:
        return "Concerning"
    else:
        return "Poor (likely overfitting)"
```

### Overfitting Diagnosis

```python
def diagnose_overfitting(
    val_results: Dict,
    test_results: Dict,
    threshold: float = 0.15
) -> Dict:
    """
    Diagnose potential overfitting issues.

    Returns diagnosis with actionable recommendations.
    """
    issues = []
    recommendations = []

    for horizon in test_results:
        test = test_results[horizon]
        val = val_results.get(horizon, {})

        if not val:
            continue

        # Sharpe gap
        sharpe_gap_pct = (val.get("sharpe", 0) - test.sharpe_ratio) / val.get("sharpe", 1)
        if sharpe_gap_pct > threshold:
            issues.append(f"H{horizon}: Sharpe gap {sharpe_gap_pct:.1%} exceeds threshold")

        # F1 gap
        f1_gap_pct = (val.get("f1", 0) - test.macro_f1) / val.get("f1", 1)
        if f1_gap_pct > threshold:
            issues.append(f"H{horizon}: F1 gap {f1_gap_pct:.1%} exceeds threshold")

    if issues:
        recommendations = [
            "Increase regularization in base models",
            "Reduce model complexity (fewer features, shallower trees)",
            "Increase purge/embargo gaps",
            "Use simpler meta-learner",
            "Collect more training data",
            "Consider using best single model instead of ensemble"
        ]

    return {
        "has_overfitting": len(issues) > 0,
        "issues": issues,
        "recommendations": recommendations
    }
```

---

## Drift Detection and Monitoring

### Distribution Shift Detection

```python
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np


class DriftDetector:
    """
    Detect distribution shift between training and production data.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        significance_level: float = 0.05
    ):
        self.reference = reference_data
        self.feature_columns = feature_columns
        self.significance_level = significance_level

        # Compute reference statistics
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute reference statistics for each feature."""
        stats = {}
        for col in self.feature_columns:
            stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "quantiles": df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            }
        return stats

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        method: str = "ks"  # ks, psi, or chi2
    ) -> Dict:
        """
        Detect drift between reference and current data.

        Args:
            current_data: New data to check
            method: Detection method (ks=Kolmogorov-Smirnov, psi=PSI)

        Returns:
            Drift detection results
        """
        results = {
            "drift_detected": False,
            "drifted_features": [],
            "feature_scores": {}
        }

        for col in self.feature_columns:
            if method == "ks":
                score, is_drift = self._ks_test(col, current_data)
            elif method == "psi":
                score, is_drift = self._psi_test(col, current_data)
            else:
                raise ValueError(f"Unknown method: {method}")

            results["feature_scores"][col] = score

            if is_drift:
                results["drifted_features"].append(col)
                results["drift_detected"] = True

        results["drift_severity"] = self._compute_severity(results)

        return results

    def _ks_test(self, col: str, current: pd.DataFrame) -> Tuple[float, bool]:
        """Kolmogorov-Smirnov test for drift."""
        ref_values = self.reference[col].dropna().values
        cur_values = current[col].dropna().values

        statistic, p_value = ks_2samp(ref_values, cur_values)

        return statistic, p_value < self.significance_level

    def _psi_test(self, col: str, current: pd.DataFrame) -> Tuple[float, bool]:
        """Population Stability Index for drift."""
        ref_values = self.reference[col].dropna().values
        cur_values = current[col].dropna().values

        # Create bins from reference distribution
        n_bins = 10
        _, bin_edges = np.histogram(ref_values, bins=n_bins)

        # Compute PSI
        ref_pct = np.histogram(ref_values, bins=bin_edges)[0] / len(ref_values)
        cur_pct = np.histogram(cur_values, bins=bin_edges)[0] / len(cur_values)

        # Add small value to avoid log(0)
        ref_pct = np.clip(ref_pct, 0.0001, None)
        cur_pct = np.clip(cur_pct, 0.0001, None)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        # PSI thresholds: <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
        return psi, psi > 0.2

    def _compute_severity(self, results: Dict) -> str:
        """Compute overall drift severity."""
        n_drifted = len(results["drifted_features"])
        n_total = len(self.feature_columns)
        pct_drifted = n_drifted / n_total

        if pct_drifted == 0:
            return "None"
        elif pct_drifted < 0.1:
            return "Low"
        elif pct_drifted < 0.3:
            return "Moderate"
        elif pct_drifted < 0.5:
            return "High"
        else:
            return "Critical"
```

### Performance Monitoring

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass
class MonitoringConfig:
    """Configuration for production monitoring."""
    check_interval_bars: int = 288      # Check every day (288 5-min bars)
    lookback_bars: int = 1440           # 5 days lookback
    sharpe_threshold: float = 0.0       # Alert if Sharpe drops below
    drawdown_threshold: float = -0.10   # Alert if drawdown exceeds 10%
    drift_check_interval: int = 1440    # Check drift every 5 days


class PerformanceMonitor:
    """
    Monitor model performance in production.
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        config: MonitoringConfig,
        alert_callback: Callable = None
    ):
        self.pipeline = pipeline
        self.config = config
        self.alert_callback = alert_callback

        self.performance_history = []
        self.drift_history = []
        self.alerts = []

    def check_performance(
        self,
        recent_data: pd.DataFrame,
        true_labels: np.ndarray = None
    ) -> Dict:
        """
        Check recent performance and generate alerts.

        Call periodically (e.g., daily) with recent data.
        """
        timestamp = datetime.now()

        # Generate predictions
        preds = self.pipeline.predict(recent_data, horizon=20)

        # If we have labels, compute metrics
        if true_labels is not None:
            returns = recent_data["return_1"].values
            positions = preds["predictions"].astype(float)
            strategy_returns = positions[:-1] * returns[1:]

            # Rolling Sharpe
            bars_per_year = 252 * 78
            rolling_sharpe = (
                strategy_returns.mean() / strategy_returns.std() *
                np.sqrt(bars_per_year)
            ) if strategy_returns.std() > 0 else 0

            # Cumulative return
            cumulative = (1 + strategy_returns).cumprod()
            current_drawdown = (cumulative[-1] / cumulative.max()) - 1

            metrics = {
                "timestamp": timestamp,
                "rolling_sharpe": rolling_sharpe,
                "current_drawdown": current_drawdown,
                "n_samples": len(recent_data),
                "avg_confidence": preds["confidence"].mean()
            }

            self.performance_history.append(metrics)

            # Check for alerts
            alerts = self._check_alerts(metrics)
            if alerts:
                self.alerts.extend(alerts)
                if self.alert_callback:
                    self.alert_callback(alerts)

            return {
                "metrics": metrics,
                "alerts": alerts
            }

        return {"metrics": None, "alerts": []}

    def _check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for alert conditions."""
        alerts = []

        if metrics["rolling_sharpe"] < self.config.sharpe_threshold:
            alerts.append({
                "type": "low_sharpe",
                "severity": "warning",
                "message": f"Rolling Sharpe {metrics['rolling_sharpe']:.2f} below threshold",
                "timestamp": metrics["timestamp"]
            })

        if metrics["current_drawdown"] < self.config.drawdown_threshold:
            alerts.append({
                "type": "high_drawdown",
                "severity": "critical",
                "message": f"Drawdown {metrics['current_drawdown']:.1%} exceeds threshold",
                "timestamp": metrics["timestamp"]
            })

        if metrics["avg_confidence"] < 0.4:
            alerts.append({
                "type": "low_confidence",
                "severity": "info",
                "message": f"Average prediction confidence {metrics['avg_confidence']:.2f} is low",
                "timestamp": metrics["timestamp"]
            })

        return alerts

    def get_performance_report(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        return pd.DataFrame(self.performance_history)
```

---

## Output Structure

### Directory Layout

```
outputs/phase5/
|
+-- test_evaluation/
|   +-- results_h5.json              # Evaluation results per horizon
|   +-- results_h10.json
|   +-- results_h15.json
|   +-- results_h20.json
|   +-- summary.json                 # Overall summary
|   +-- confusion_matrices.png       # Confusion matrices visualization
|   +-- per_symbol/
|       +-- MES_results.json
|       +-- MGC_results.json
|
+-- generalization/
|   +-- val_vs_test.csv              # Validation vs test comparison
|   +-- generalization_report.html   # Interactive report
|   +-- overfitting_diagnosis.json   # Overfitting analysis
|
+-- verification/
|   +-- lookahead_test.json          # Lookahead verification results
|   +-- integrity_check.json         # Test set integrity verification
|
+-- deployment/
|   +-- inference_pipeline/          # Serialized pipeline
|       +-- config.json
|       +-- scaler.pkl
|       +-- feature_columns.json
|       +-- models/
|       +-- ensemble/
|       +-- manifest.json
|
+-- monitoring/
|   +-- drift_detector_config.json   # Drift detection setup
|   +-- reference_stats.json         # Reference distribution stats
|   +-- monitoring_config.json       # Monitoring thresholds

reports/phase5/
|
+-- test_performance.html            # Interactive performance report
+-- tearsheet.html                   # Trading performance tearsheet
+-- final_summary.pdf                # Executive summary
```

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Test Sharpe (H20) | > 0.50 | Trading metrics |
| Test F1 (H20) | > 0.40 | Classification metrics |
| Val-Test Sharpe gap | < 15% | Generalization analysis |
| Val-Test F1 gap | < 15% | Generalization analysis |
| No lookahead | 0 violations | Lookahead verification |
| Pipeline runs | Error-free | Integration test |
| Inference latency | < 100ms | Timing benchmark |

### Per-Horizon Targets

| Horizon | Min Sharpe | Min F1 | Max Val-Test Gap |
|---------|------------|--------|------------------|
| H5 | 0.30 | 0.35 | 20% |
| H10 | 0.40 | 0.38 | 18% |
| H15 | 0.45 | 0.40 | 15% |
| H20 | 0.50 | 0.42 | 15% |

---

## Decision Framework

### If Targets Met

```
1. Document final system architecture
2. Serialize inference pipeline
3. Set up monitoring infrastructure
4. Deploy to paper trading
5. Monitor for 2-4 weeks
6. Deploy to production with position limits
```

### If Targets Not Met

```
1. Document actual performance honestly
2. Identify specific failure modes:
   - Which horizons underperform?
   - Which symbols underperform?
   - Which regimes cause failures?
3. Return to Phase 2-4 for refinement:
   - Add/remove base models
   - Adjust feature sets
   - Tune regularization
4. DO NOT iterate on test set
5. Use fresh test period if needed
```

### Iteration Rules

```python
def should_iterate(test_results: Dict, targets: Dict) -> Dict:
    """
    Determine if iteration is warranted and provide guidance.

    IMPORTANT: Never iterate based on test set results.
    """
    failures = []
    for horizon, result in test_results.items():
        if result.sharpe_ratio < targets[horizon]["min_sharpe"]:
            failures.append(f"H{horizon} Sharpe {result.sharpe_ratio:.2f} < target")
        if result.macro_f1 < targets[horizon]["min_f1"]:
            failures.append(f"H{horizon} F1 {result.macro_f1:.2f} < target")

    if not failures:
        return {
            "decision": "deploy",
            "reason": "All targets met",
            "next_steps": ["Set up monitoring", "Paper trade for 2 weeks", "Deploy"]
        }

    return {
        "decision": "investigate",
        "reason": f"{len(failures)} targets not met",
        "failures": failures,
        "next_steps": [
            "Analyze failure modes on validation set",
            "Adjust hyperparameters using validation set only",
            "Reserve new test period",
            "Re-run Phase 5 with new test period"
        ],
        "warning": "DO NOT iterate using test set results!"
    }
```

---

## Usage Examples

**NOTE: These scripts do not currently exist. Phase 5 requires Phase 2-4 to be completed first.**

```bash
# PLANNED (not yet implemented):
# python scripts/evaluate_test.py --test-data data/splits/scaled/test_scaled.parquet

# CURRENT STATUS:
# Phase 5 is PLANNED and cannot be run until Phase 2, 3, and 4 are complete.
#
# Implementation order:
# 1. Complete Phase 2: Train base models (XGBoost, LightGBM, LSTM, etc.)
# 2. Complete Phase 3: Generate out-of-fold predictions via cross-validation
# 3. Complete Phase 4: Train meta-learner on OOF predictions
# 4. Implement Phase 5: Evaluate on hold-out test set
# 5. Deploy to production

# Current implementation:
# Only Phase 1 data pipeline is implemented:
./pipeline run --symbols MES,MGC

# Test data is available at:
# data/splits/scaled/test_scaled.parquet
```

---

## Post-Deployment Checklist

```markdown
## Deployment Checklist

### Pre-Deployment
- [ ] Test set evaluation passed all targets
- [ ] Lookahead verification passed (0 violations)
- [ ] Generalization gap < 15% for all horizons
- [ ] Pipeline serialized and tested
- [ ] Drift detector configured with reference data
- [ ] Monitoring alerts configured

### Paper Trading (2-4 weeks)
- [ ] Pipeline deployed to paper trading environment
- [ ] Daily performance monitoring active
- [ ] Drift detection running
- [ ] No critical alerts triggered
- [ ] Performance matches test set expectations (+/- 20%)

### Production Deployment
- [ ] Position size limits configured
- [ ] Stop-loss mechanisms in place
- [ ] Fallback to simpler model if ensemble fails
- [ ] Alerting to trading desk active
- [ ] Daily performance reports automated
- [ ] Weekly drift reports scheduled

### Ongoing Monitoring
- [ ] Weekly performance review
- [ ] Monthly drift analysis
- [ ] Quarterly model refresh evaluation
- [ ] Annual full retrain consideration
```

---

## Next Steps After Phase 5

**If successful:**
1. Begin paper trading with production pipeline
2. Monitor performance and drift for 2-4 weeks
3. Deploy to production with conservative position limits
4. Gradually increase position sizes as confidence builds
5. Schedule periodic model refresh (quarterly or event-driven)

**Future phases (not documented):**
- Phase 6: Production monitoring and maintenance
- Phase 7: Continuous learning and model refresh
- Phase 8: Multi-asset expansion
