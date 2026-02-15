# Ensemble & Special Training Modes — Inference Plan

**Date:** 2026-02-15
**Scope:** How ensembles, walk-forward, regime-aware, and meta-labeling bundles work at inference time
**Input:** Phase 1 audit findings (Q4), training-output-flow audit, source code analysis

---

## 1. Standard Ensemble Inference

### Current State

`EnsembleBundle` (src/inference/ensemble_bundle.py) already has:
- `predict(base_predictions)` — takes pre-computed base model probability dicts, stacks them, feeds meta-learner
- `predict_from_base_features(X)` — loads base ModelBundles, runs each, then feeds meta-learner
- `_stack_predictions()` — handles both aligned (same-length) and misaligned (OOFAligner) cases
- `_simple_stack()` — concatenates probabilities + derived features (confidence, agreement)

### How `predict_from_raw()` Should Work

```
EnsembleBundle.predict_from_raw(raw_ohlcv_df)
  ├── FOR each base_bundle in self._base_bundles:
  │     ├── base_bundle.predict_from_raw(raw_ohlcv_df)  ← uses each bundle's own preprocessing + adapter
  │     └── Collect PredictionResult.class_probabilities
  ├── Align predictions (if different lengths due to sequence trimming)
  ├── Stack into meta-learner input via _stack_predictions()
  └── meta_learner.predict(stacked) → final PredictionResult
```

**Key insight:** Each base ModelBundle already handles its own adapter routing (once the universal inference pipeline from Plan #1 is implemented). EnsembleBundle just orchestrates and aggregates.

### Fix: Relative vs Absolute Path Issue

**Problem:** `base_bundles.json` stores absolute paths (e.g., `/home/user/experiments/run_001/bundles/xgboost_h20`). Moving the bundle directory breaks all references.

**Fix:** Store relative paths from ensemble bundle root.

```python
# In EnsembleBundle.save():
# Instead of: [str(p) for p in self.base_bundle_paths]
# Use:        [str(p.relative_to(path.parent)) for p in self.base_bundle_paths]

# In EnsembleBundle.load():
# Instead of: [Path(p) for p in data["paths"]]
# Use:        [path.parent / p for p in data["paths"]]
# With fallback: if not resolved.exists(), try as absolute path
```

**Files to change:**
- `src/inference/ensemble_bundle.py` — `save()` (line ~443-452) and `load()` (line ~539-543)

### New Method: `predict_from_raw()`

Add to `EnsembleBundle`:

```python
def predict_from_raw(
    self,
    raw_df: pd.DataFrame,
    calibrate: bool = True,
) -> PredictionResult:
    """End-to-end ensemble prediction from raw OHLCV data."""
    self._ensure_base_bundles_loaded()

    base_predictions: dict[str, np.ndarray] = {}
    for model_name, bundle in self._base_bundles.items():
        output = bundle.predict_from_raw(raw_df, calibrate=False)
        base_predictions[model_name] = output.class_probabilities

    return self.predict(base_predictions, calibrate=calibrate)
```

---

## 2. Walk-Forward Mode Inference

### What Training Produces

`_train_walk_forward()` (unified_orchestrator.py:1188) uses `WalkForwardTrainer` which produces:
- Multiple model versions per window (e.g., xgboost trained on data[0:1000], data[0:2000], data[0:3000])
- Aggregated metrics per model (mean_f1, mean_accuracy across windows)
- Currently stores only aggregated `ModelTrainingResult` — **individual window models are not preserved**

### Inference Strategy: Latest Window Only (MVP)

For MVP, use the model trained on the largest/latest window:

```
WalkForwardBundle/
  ├── manifest.json
  ├── metadata.json              # includes walk_forward_config
  ├── windows/
  │     ├── window_config.json   # window boundaries, dates
  │     └── latest/              # ModelBundle for last (largest) window
  │           ├── model/
  │           ├── scaler.pkl
  │           └── features.json
  └── aggregated_metrics.json    # metrics across all windows
```

**Prediction flow:**
```
WalkForwardBundle.predict_from_raw(raw_df)
  └── self.latest_bundle.predict_from_raw(raw_df)  # delegates to latest window's ModelBundle
```

### Future Enhancement: Window Ensemble

Later, support weighted averaging across window models:
- Weight recent windows higher (exponential decay)
- Or use all window models as a mini-ensemble with simple averaging
- This is a natural extension but not MVP-critical

### Changes Required

1. **`WalkForwardTrainer`** — Preserve per-window trainer objects (currently discarded)
2. **`UnifiedTrainingOrchestrator._train_walk_forward()`** — Store window trainers in `ModelTrainingResult`
3. **New: `WalkForwardBundle`** — Thin wrapper around ModelBundle with window metadata
4. **`BundleBuilder`** — Add `build_walk_forward_bundle()` method

---

## 3. Regime-Aware Mode Inference

### What Training Produces

`_train_regime_aware()` (unified_orchestrator.py:1286) produces:
- Per-regime models: e.g., `xgboost_h20_low_vol`, `xgboost_h20_high_vol`, `xgboost_h20_trending`
- Each stored as `ModelTrainingResult` with regime name in key
- `RegimeAwareTrainer` instance stored as `self._regime_trainer` (in-memory only, not bundled)
- Regime detection config from `PipelineConfig`: method, lookback, n_regimes, thresholds

### Inference Strategy: Detect Regime → Route to Correct Model

```
RegimeBundle.predict_from_raw(raw_df)
  ├── RegimeDetector.detect(raw_df)  → current_regime (e.g., "high_vol")
  ├── Select model: self.regime_models[current_regime]
  └── selected_model.predict_from_raw(raw_df) → PredictionResult
```

### Bundle Format

```
RegimeBundle/
  ├── manifest.json
  ├── metadata.json
  ├── regime_config.json          # detection method, thresholds, lookback
  │     {
  │       "method": "volatility_percentile",
  │       "n_regimes": 2,
  │       "lookback": 60,
  │       "volatility_window": 20,
  │       "adx_threshold": 25,
  │       "min_samples": 100
  │     }
  ├── regime_models/
  │     ├── low_vol/              # ModelBundle for low volatility regime
  │     ├── high_vol/             # ModelBundle for high volatility regime
  │     └── trending/             # ModelBundle for trending regime (if 3 regimes)
  └── fallback_model/             # Optional: model for unknown/transition regimes
```

### Regime Detection at Inference

The regime detector needs to run on recent OHLCV history (not just the prediction point):

```python
class RegimeDetector:
    """Detects current market regime from recent price data."""

    def detect(self, recent_ohlcv: pd.DataFrame) -> str:
        """
        Args:
            recent_ohlcv: Last N bars (N = lookback from config)
        Returns:
            Regime name string matching bundle keys
        """
        if self.method == "volatility_percentile":
            vol = recent_ohlcv["close"].pct_change().rolling(self.vol_window).std().iloc[-1]
            return "high_vol" if vol > self.threshold else "low_vol"
        elif self.method == "trend_adx":
            # Compute ADX on recent data
            ...
```

**Key decision:** The `RegimeDetector` should be serializable and saved in the bundle so the exact same detection logic used in training is replayed at inference.

### Changes Required

1. **New: `RegimeBundle`** class in `src/inference/regime_bundle.py`
2. **New: `RegimeDetector`** — Serializable regime detection (extract from `RegimeAwareTrainer`)
3. **`BundleBuilder`** — Add `build_regime_bundle()` method
4. **`UnifiedTrainingOrchestrator._train_regime_aware()`** — Serialize regime detection config alongside models

---

## 4. Meta-Labeling Mode Inference

### What Training Produces

`_train_meta_labeling_for_horizon()` (unified_orchestrator.py:1461) produces:
- **Primary trainer** (e.g., XGBoost): Direction prediction (+1, -1, 0)
- **Meta-model** (e.g., LogisticRegression): P(primary is correct) ∈ [0, 1]
- **Threshold** (default 0.5): Minimum confidence to take a trade
- Both stored in `self._trained_models` with keys like `meta_labeling_h20_primary`, `meta_labeling_h20_meta`
- Meta-model saved as sklearn pickle with metadata dict

### Inference Flow

```
MetaLabelingBundle.predict_from_raw(raw_df)
  ├── primary_bundle.predict_from_raw(raw_df)
  │     → PredictionResult with directions (-1, 0, +1) and probabilities
  ├── meta_bundle.predict(features)
  │     → meta_probabilities (P(primary_correct))
  ├── Apply threshold: take_trade = meta_prob >= threshold
  └── Return MetaLabelingPrediction:
        - directions: primary model class predictions
        - meta_probabilities: confidence scores
        - positions: direction * meta_prob * take_trade_mask
        - trade_mask: boolean array of which trades to take
```

### Bundle Format

```
MetaLabelingBundle/
  ├── manifest.json
  ├── metadata.json
  │     {
  │       "primary_model": "xgboost",
  │       "meta_model": "logistic",
  │       "threshold": 0.5,
  │       "horizon": 20,
  │       "training_metrics": { ... }
  │     }
  ├── primary_model/              # Full ModelBundle for primary (direction)
  │     ├── model/
  │     ├── scaler.pkl
  │     ├── features.json
  │     └── preprocessing_graph.json
  ├── meta_model/                 # Meta-model artifacts
  │     └── model.pkl             # sklearn model (logistic/RF/etc.)
  └── meta_config.json            # threshold, feature mapping
```

### Result Type

```python
@dataclass
class MetaLabelingPrediction:
    """Result from meta-labeling inference."""
    directions: np.ndarray       # Primary model class predictions
    meta_probabilities: np.ndarray  # P(primary correct)
    positions: np.ndarray        # direction * meta_prob (0 where below threshold)
    trade_mask: np.ndarray       # Boolean: which samples pass threshold
    threshold: float             # Threshold used
```

### Changes Required

1. **New: `MetaLabelingBundle`** class in `src/inference/meta_labeling_bundle.py`
2. **`BundleBuilder`** — Add `build_meta_labeling_bundle()` method
3. **`UnifiedTrainingOrchestrator._train_meta_labeling_for_horizon()`** — Ensure both models have proper trainer interfaces for extraction

---

## 5. Bundle Format Summary Per Mode

| Mode | Bundle Class | Key Contents | Prediction Method |
|------|-------------|-------------|------------------|
| **Standard** | `ModelBundle` | model, scaler, features, preprocessing_graph | `predict_from_raw(df)` |
| **Ensemble** | `EnsembleBundle` | meta-learner, base bundle refs, alignment config | `predict_from_raw(df)` (NEW) |
| **Walk-Forward** | `WalkForwardBundle` (NEW) | latest window ModelBundle, window config | `predict_from_raw(df)` via latest |
| **Regime-Aware** | `RegimeBundle` (NEW) | per-regime ModelBundles, regime detector config | `predict_from_raw(df)` via detected regime |
| **Meta-Labeling** | `MetaLabelingBundle` (NEW) | primary ModelBundle, meta sklearn model, threshold | `predict_from_raw(df)` → positions |

### Common Interface

All bundle types should implement:

```python
class InferenceBundle(Protocol):
    """Common interface for all bundle types."""

    def predict(self, X: np.ndarray, calibrate: bool = True) -> PredictionResult: ...
    def predict_from_raw(self, raw_df: pd.DataFrame) -> PredictionResult: ...
    def save(self, path: Path, overwrite: bool = False) -> Path: ...
    @classmethod
    def load(cls, path: Path) -> InferenceBundle: ...
    def validate(self) -> dict[str, Any]: ...
```

---

## 6. Type Alignment: EnsembleService Result vs BundleBuilder

### The Problem

Two type chains exist for ensemble results:

**Chain A (Training → Orchestrator):**
```
EnsembleService.build_ensemble(EnsembleRequest)
  → EnsembleServiceResult (has: aligned_oof, stacking_dataset, meta_learner, ensemble_metrics)
```

**Chain B (BundleBuilder expects):**
```
BundleBuilder.build_ensemble_bundle(EnsembleResult)
  → expects EnsembleResult from src/models/ensemble/orchestrator.py
  → has: ensemble_name, meta_learner_name, base_model_names, metrics, stacking_dataset, aligned_oof
```

**The gap:** `EnsembleServiceResult` ≠ `EnsembleResult`. The orchestrator converts `EnsembleServiceResult` to `ModelTrainingResult` (which loses stacking info), and neither is what `BundleBuilder.build_ensemble_bundle()` expects.

### Current Flow (Broken)

```
EnsembleService → EnsembleServiceResult
                      ↓ (in _build_ensemble)
              ModelTrainingResult (loses stacking info, aligned_oof)
                      ↓
              TrainingRunResult.ensemble_result
                      ↓ (in MLFactory._create_bundle)
              BundleBuilder.build_ensemble_bundle() ← expects EnsembleResult, gets ModelTrainingResult ❌
```

### Fix: Bridge EnsembleServiceResult → EnsembleResult

Add a conversion method:

```python
# In EnsembleService or as standalone function
def to_ensemble_result(
    service_result: EnsembleServiceResult,
    config: PipelineConfig,
) -> EnsembleResult:
    """Convert EnsembleServiceResult to EnsembleResult for BundleBuilder."""
    return EnsembleResult(
        ensemble_name=f"{config.meta_learner}_ensemble",
        meta_learner_name=config.meta_learner,
        base_model_names=service_result.aligned_oof.model_names if service_result.aligned_oof else [],
        metrics=service_result.ensemble_metrics,
        stacking_dataset=service_result.stacking_dataset,
        aligned_oof=service_result.aligned_oof,
        training_time_seconds=service_result.training_time_seconds,
        n_base_models=len(service_result.aligned_oof.model_names) if service_result.aligned_oof else 0,
        coverage=service_result.aligned_oof.coverage if service_result.aligned_oof else 0.0,
    )
```

### Updated Flow

```
EnsembleService → EnsembleServiceResult
                      ↓ to_ensemble_result()
              EnsembleResult (preserved with full stacking info)
                      ↓
              TrainingRunResult.ensemble_result (keep as EnsembleResult, not ModelTrainingResult)
                      ↓
              BundleBuilder.build_ensemble_bundle(ensemble_result) ✅
```

### Alternative: Unify the Types

Longer-term, consolidate `EnsembleServiceResult` and `EnsembleResult` into a single type. The `EnsembleResult` in `src/models/ensemble/orchestrator.py` is the richer type; `EnsembleServiceResult` could be removed and `EnsembleService` could return `EnsembleResult` directly.

**Recommendation:** Do the bridge conversion for MVP (low-risk), then consolidate types in cleanup phase.

### Files to Change

- `src/models/training/services/ensemble_service.py` — Add `to_ensemble_result()` conversion
- `src/models/training/unified_orchestrator.py` — Store `EnsembleResult` in `TrainingRunResult` instead of `ModelTrainingResult`
- `src/models/training/unified_orchestrator.py` — `TrainingRunResult.ensemble_result` type annotation: `EnsembleResult | None`
- `src/inference/builder.py` — Verify `build_ensemble_bundle()` works with converted result

---

## 7. Concrete Changes — File-by-File

### Existing Files to Modify

| File | Change | Priority |
|------|--------|----------|
| **src/inference/ensemble_bundle.py** | 1. Fix relative paths in save/load (lines 443, 539). 2. Add `predict_from_raw()` method. 3. Return `PredictionResult` from `predict()` instead of raw meta-learner output. | HIGH |
| **src/inference/builder.py** | 1. Add `build_walk_forward_bundle()`. 2. Add `build_regime_bundle()`. 3. Add `build_meta_labeling_bundle()`. 4. Fix `build_ensemble_bundle()` to accept converted `EnsembleResult`. | HIGH |
| **src/models/training/unified_orchestrator.py** | 1. Change `_build_ensemble()` to return `EnsembleResult` (not `ModelTrainingResult`). 2. Change `TrainingRunResult.ensemble_result` type to `EnsembleResult`. 3. In `_train_walk_forward()`, preserve per-window trainers. 4. In `_train_regime_aware()`, serialize regime detection config. 5. In `_train_meta_labeling_for_horizon()`, ensure both models are properly extractable. | HIGH |
| **src/models/training/services/ensemble_service.py** | Add `to_ensemble_result()` conversion method. | MEDIUM |
| **src/models/ensemble/orchestrator.py** | No changes needed — `EnsembleResult` is already correct. | — |

### New Files to Create

| File | Purpose | Complexity |
|------|---------|------------|
| **src/inference/walk_forward_bundle.py** | `WalkForwardBundle` class wrapping latest-window ModelBundle | LOW |
| **src/inference/regime_bundle.py** | `RegimeBundle` class with regime detection + per-regime ModelBundles | MEDIUM |
| **src/inference/meta_labeling_bundle.py** | `MetaLabelingBundle` with primary + meta model inference flow | MEDIUM |
| **src/inference/regime_detector.py** | Serializable `RegimeDetector` extracted from RegimeAwareTrainer | MEDIUM |

### Execution Order

```
Phase A (Foundation):
  1. Fix EnsembleBundle relative paths              [LOW risk, immediate value]
  2. Add EnsembleBundle.predict_from_raw()           [LOW risk, immediate value]
  3. Type alignment: EnsembleServiceResult → EnsembleResult bridge  [MEDIUM risk]

Phase B (Special Modes — parallel):
  4. MetaLabelingBundle (simplest new bundle)        [self-contained]
  5. WalkForwardBundle (thin wrapper)                [self-contained]
  6. RegimeBundle + RegimeDetector (most complex)    [self-contained]

Phase C (Integration):
  7. BundleBuilder methods for each mode
  8. Orchestrator changes to produce correct result types
  9. Update inference/__init__.py exports
```

### Dependencies

- Items 4-6 depend on the universal inference pipeline (Plan #1) for `predict_from_raw()` to work on base ModelBundles
- Item 3 (type alignment) is independent and can be done immediately
- Items 1-2 (EnsembleBundle fixes) are independent and can be done immediately

---

## Summary

The ensemble inference path is ~80% built. The main gaps are:
1. **EnsembleBundle** needs `predict_from_raw()` and relative paths (fixes, not new architecture)
2. **Type mismatch** between `EnsembleServiceResult` and `EnsembleResult` needs a bridge converter
3. **Three new bundle types** (WalkForward, Regime, MetaLabeling) need to be created, but they follow the established `ModelBundle` pattern and are straightforward
4. **RegimeDetector** is the only genuinely new inference component — it must be serializable and reproduce training-time regime detection exactly
