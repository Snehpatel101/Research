# Deployable Artifact Implementation Plan -- ML Factory

## Executive Summary

ML Factory's training pipeline produces models for 12 core families (3 boosting, 7 neural/CNN, 2 transformer) but only 3/12 can go from raw OHLCV bars to prediction in a single call. This plan closes that gap with 24 tasks across 4 phases (P0-A through P0-D), producing a `deploy/` directory with per-horizon artifacts, a JSON manifest, and validation reports. The critical path runs: foundation types/protocols (P0-A) -> adapter routing for 3D/4D models + double-scaling fix (P0-B) -> deploy packaging with artifact selection (P0-C) -> notebook inference demo (P0-D). Total estimated effort: 1,066-1,425 LOC across 3 new files and 7 modified files. After implementation, `artifact.predict_from_raw(raw_bars_df)` works for all 12 model families, ensembles included, with zero caller-side tensor shaping.

---

## Workstream Table

| Phase | Description | Tasks | New Files | Modified Files | LOC Estimate | Critical Path? |
|-------|-------------|-------|-----------|----------------|--------------|----------------|
| P0-A | Foundation: protocols, types, metadata, calibrator fix, scaling fix | 6 | 1 (`src/core/protocols.py`) | 3 (`types.py`, `bundle.py`, `builder.py`, `unified_orchestrator.py`) | 191-300 | YES |
| P0-B | Core Inference: adapter routing, 3D windowing, 4D MTF, ensemble predict_from_raw | 6 | 1 (`src/inference/errors.py`) | 2 (`bundle.py`, `ensemble_bundle.py`) | 220-320 | YES |
| P0-C | Deploy Packaging: manifest, selector, validator, factory Phase 5, config | 7 | 1 (`src/inference/deploy.py`) | 3 (`factory.py`, `__init__.py`, `experiment.py`) | 420-550 | YES |
| P0-D | Notebook Integration: inference demo, export, validation display, Drive persistence | 5 | 0 | 1 (`ml_factory_colab.ipynb`) | 235-255 | NO (terminal) |
| **TOTAL** | | **24** | **3** | **7 unique** | **1,066-1,425** | |

---

## Dependency Graph

```
                                 P0-A: FOUNDATION
  ========================================================================
  |                                                                      |
  |  BATCH 1 (all parallel, no deps):                                   |
  |                                                                      |
  |  P0-A-1 ──┐  P0-A-2    P0-A-3 ──┐  P0-A-4 ──┐  P0-A-5    P0-B-6  |
  |  protocols │  ScalingS  Metadata │  calibratr │  skip_scl  errors   |
  |            │           ┌────────┘            │                      |
  |            │           │                     │                      |
  |  BATCH 2:  ▼           ▼                     ▼                      |
  |         P0-A-6 ◄── (A-1 + A-3 + A-4)                               |
  |         protocol-aware BundleBuilder                                 |
  |                                                                      |
  |         P0-B-2 ◄── (A-3)          P0-B-3 ◄── (A-3)   P0-B-5       |
  |         _build_3d                  _build_4d           rel paths     |
  |                                                                      |
  ========================================================================
                          │                │
                          ▼                ▼
                                 P0-B: CORE INFERENCE
  ========================================================================
  |                                                                      |
  |  BATCH 3:                                                            |
  |         P0-B-1 ◄── (A-3 + A-5 + B-2 + B-3)                        |
  |         _apply_adapter routing                                       |
  |                │                                                     |
  |                │    P0-C-1 (DeployManifest, independent)             |
  |                │                                                     |
  |  BATCH 4:      ▼                                                     |
  |         P0-B-4 ◄── (B-1)                                            |
  |         EnsembleBundle.predict_from_raw                              |
  |                                                                      |
  |         P0-C-3 ◄── (C-1)   P0-C-4 ◄── (C-1)   P0-C-5 ◄── (C-1)  |
  |         selector            validator            loader              |
  |                                                                      |
  ========================================================================
                          │
                          ▼
                                 P0-C: DEPLOY PACKAGING
  ========================================================================
  |                                                                      |
  |  BATCH 5:                                                            |
  |         P0-C-2 ◄── (C-1 + C-3 + C-4)                               |
  |         factory.py Phase 5                                           |
  |                │                                                     |
  |         P0-C-6 ◄── (C-1 + C-3 + C-4 + C-5)                        |
  |         __init__.py exports                                          |
  |                │                                                     |
  |         P0-C-7 ◄── (C-2)                                            |
  |         config toggle                                                |
  |                                                                      |
  ========================================================================
                          │
                          ▼
                                 P0-D: NOTEBOOK INTEGRATION
  ========================================================================
  |                                                                      |
  |  BATCH 6:                                                            |
  |         P0-D-4 ◄── (C-7)                                            |
  |         config additions (Cell 2+5)                                  |
  |                                                                      |
  |  BATCH 7 (all parallel):                                             |
  |         P0-D-1 (inference demo, Cell 8)                              |
  |         P0-D-2 (deploy export, Cell 9)                               |
  |         P0-D-3 (validation cell, Cell 10)                            |
  |         P0-D-5 (Drive persistence, Cell 11)                          |
  |                                                                      |
  ========================================================================
```

**Critical path length: 7 sequential batches.**

---

## Risk Matrix

| Risk ID | Description | Likelihood | Impact | Severity | Mitigation |
|---------|------------|------------|--------|----------|------------|
| R1 | Double-scaling regression: `skip_scaling` reverts to `False` in future edits, causing features to be scaled twice | Medium | High | **HIGH** | Inline comment at fix site; integration test IT-1 checks single scaling; `ScalingSource` enum documents pattern |
| R2 | 4D model MTF data availability: user provides 5min data instead of 1min at inference time | High | Medium | **HIGH** | `_build_4d_input()` validates DatetimeIndex + OHLCV columns + min row counts; metadata stores `primary_timeframe`; descriptive `ValueError` messages |
| R3 | EnsembleBundle backward compat: absolute paths in old bundles break on relocation | Medium | High | **HIGH** | `load()` checks `Path.is_absolute()` -- absolute used as-is, relative resolved against parent; `save()` uses `try/except` for cross-root edge cases |
| R4 | Calibrator None propagation: calibrator lost during orchestrator result conversion | High | Medium | **HIGH** | Add explicit `calibrator` field to orchestrator `ModelTrainingResult`; copy via `getattr(result, "calibrator", None)` in both sequential and parallel training paths |
| R6 | Feature column drift between training and inference | Medium | High | **HIGH** | BundleMetadata stores `feature_names` backup; `_apply_adapter()` uses stored column order; `validate()` checks feature count |
| R5 | Notebook fails in Colab: version mismatch, OOM, dependency conflicts | Medium | Medium | **MEDIUM** | All cells wrapped in try/except; predict demo uses bounded `tail(n_bars)`; zip only `deploy/`; Drive persistence as alternative |
| R7 | Deploy selector picks wrong artifact due to metric key mismatch | Low | Medium | **MEDIUM** | Same `val_f1` key used consistently; falls back to best single model; manifest records metric snapshot |
| R8 | PreprocessingGraph hardcodes `source_timeframe="1min"`, `target_timeframe="5min"` | Medium | Medium | **MEDIUM** | P0-A-3 adds `scaler_type`/`primary_timeframe` from contract; 4D models bypass PreprocessingGraph; full fix deferred to P1 |
| R9 | Sliding window off-by-one in `_build_3d_input()` | Low | High | **MEDIUM** | Implementation mirrors `SequenceAdapter._build_sequences()` using same `sliding_window_view()`; IT-2 validates shapes |
| R10 | `build_ensemble_bundle()` output format not loadable by `EnsembleBundle.load()` (G6, deferred) | High | Medium | **MEDIUM** | Deploy selector falls back to best single model if ensemble bundle not loadable; warning logged |

---

## Validation Matrix

| Test ID | Type | What It Verifies | Phase Dep | Pass Criteria |
|---------|------|-----------------|-----------|---------------|
| SM-A1 | Smoke | `protocols.py` imports | P0-A-1 | `from src.core.protocols import TrainerProtocol, InferenceBundle` succeeds; 1 definition each |
| SM-A2 | Smoke | `ScalingSource` enum | P0-A-2 | `ScalingSource.BUNDLE.value == "bundle"`; 1 definition |
| SM-A3 | Smoke | BundleMetadata v1.3.0 | P0-A-3 | Version == "1.3.0"; round-trip serialization; v1.2.0 backward compat loads |
| SM-A4 | Smoke | Calibrator field | P0-A-4 | `ModelTrainingResult.calibrator` exists, defaults to None |
| SM-A5 | Smoke | Double-scaling fix | P0-A-5 | `skip_scaling=True` in `preprocess()` method |
| SM-A6 | Smoke | Protocol-aware builder | P0-A-6 | BundleBuilder imports; protocol check + duck-typing fallback |
| SM-B1 | Smoke | Adapter routing | P0-B-1 | `_apply_adapter()` method exists; routes by metadata flags |
| SM-B2 | Smoke | 3D sliding window | P0-B-2 | Shapes: (1,60,50), (61,60,50); ValueError on short input |
| SM-B3 | Smoke | 4D MTF generation | P0-B-3 | Correct shapes; OHLCV aggregation; DatetimeIndex required |
| SM-B4 | Smoke | Ensemble predict_from_raw | P0-B-4 | Method exists with `raw_df` and `calibrate` params |
| SM-B5 | Smoke | Relative paths | P0-B-5 | New saves relative; old absolute loads |
| SM-B6 | Smoke | Inference errors | P0-B-6 | Import succeeds; error message includes all diagnostic fields |
| SM-C1 | Smoke | DeployManifest | P0-C-1 | Round-trip save/load; pure JSON |
| SM-C2 | Smoke | Factory Phase 5 | P0-C-2 | `ExperimentResult.deploy_path` field exists |
| SM-C3 | Smoke | Artifact selector | P0-C-3 | `select_deploy_artifact` importable |
| SM-C4 | Smoke | Validation report | P0-C-4 | `validate_deploy_artifact` importable |
| SM-C5 | Smoke | Deploy loader | P0-C-5 | `load_deploy_artifact` importable |
| SM-C6 | Smoke | __init__ exports | P0-C-6 | All 5 deploy names importable from `src.inference` |
| SM-C7 | Smoke | Config toggle | P0-C-7 | `BundlingSection.deploy_artifact` defaults `True` |
| SM-D1 | Smoke | Inference demo cell | P0-D-1 | Cell 8 exists with "INFERENCE DEMO" title + predict_from_raw |
| SM-D2 | Smoke | Export cell | P0-D-2 | Cell 9 exists with "EXPORT DEPLOY ARTIFACT" |
| SM-D3 | Smoke | Validation cell | P0-D-3 | Cell 10 exists with "VALIDATION REPORT" |
| SM-D4 | Smoke | Config additions | P0-D-4 | `BUNDLING_ENABLED` + `DEPLOY_ARTIFACT` in Cell 2 |
| SM-D5 | Smoke | Drive persistence | P0-D-5 | Cell 11 with "GOOGLE DRIVE" + `load_deploy_artifact` |
| IT-1 | Integration | Tabular roundtrip (XGBoost) | P0-A+B | `predict_from_raw(raw_df)` -> PredictionResult; single scaling; calibrator present |
| IT-2 | Integration | Sequence roundtrip (LSTM) | P0-A+B | 3D adapter route; correct window shapes; ValueError on insufficient data |
| IT-3 | Integration | Transformer roundtrip (PatchTST) | P0-A+B | 4D adapter route; correct OHLCV aggregation; DatetimeIndex required |
| IT-4 | Integration | Ensemble roundtrip | P0-A+B | `predict_from_raw()` orchestrates base models with `calibrate=False`; relative paths |
| IT-5 | Integration | Deploy roundtrip | P0-A+B+C | Factory.run() -> deploy/ -> load_deploy_artifact() -> predict_from_raw() |
| IT-6 | Integration | Old bundle compat | P0-A | v1.2.0 bundle loads; predict(X_preshaped) works; defaults populated |

---

## Phase P0-A: Foundation

---

### P0-A-1: Create `src/core/protocols.py`

| Field | Value |
|-------|-------|
| Files | `src/core/protocols.py` (NEW) |
| Effort | S (40-60 LOC) |
| Depends On | None |
| Blocks | P0-A-6 |

**API Signature:**

```python
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd

@runtime_checkable
class TrainerProtocol(Protocol):
    @property
    def model(self) -> Any: ...
    @property
    def scaler(self) -> Any | None: ...
    @property
    def feature_columns(self) -> list[str]: ...
    @property
    def calibrator(self) -> Any | None: ...

@runtime_checkable
class InferenceBundle(Protocol):
    def predict_from_raw(self, raw_df: pd.DataFrame, calibrate: bool = True) -> Any: ...
    def predict(self, X: pd.DataFrame | np.ndarray, calibrate: bool = True) -> Any: ...
    def validate(self) -> dict[str, Any]: ...
    def save(self, path: str | Any, overwrite: bool = False) -> Any: ...
    @classmethod
    def load(cls, path: str | Any) -> Any: ...
```

**Acceptance Criteria:**
- [ ] `python -c "from src.core.protocols import TrainerProtocol, InferenceBundle; print('OK')"` succeeds
- [ ] `grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l` returns 1
- [ ] `grep -r "class InferenceBundle" src/ --include="*.py" | wc -l` returns 1
- [ ] `ruff check src/core/protocols.py` passes
- [ ] File includes `from __future__ import annotations`

**Verification:**
```bash
python -c "from src.core.protocols import TrainerProtocol, InferenceBundle; print('OK')" && \
ruff check src/core/protocols.py
```

---

### P0-A-2: Add `ScalingSource` Enum to `src/core/types.py`

| Field | Value |
|-------|-------|
| Files | `src/core/types.py` (MODIFY) |
| Effort | S (10-15 LOC) |
| Depends On | None |
| Blocks | None in P0 (used by future UIP) |

**API Signature:**

```python
class ScalingSource(str, Enum):
    """Controls which component applies feature scaling during inference."""
    BUNDLE = "bundle"
    PREPROCESSING = "preprocessing"
    NONE = "none"
```

Insert after `LabelingMethod` enum (~L226).

**Acceptance Criteria:**
- [ ] `python -c "from src.core.types import ScalingSource; print(ScalingSource.BUNDLE.value)"` prints `bundle`
- [ ] `grep -r "class ScalingSource" src/ --include="*.py" | wc -l` returns 1
- [ ] `ruff check src/core/types.py` passes

**Verification:**
```bash
python -c "from src.core.types import ScalingSource; print(ScalingSource.BUNDLE.value)" && \
ruff check src/core/types.py
```

---

### P0-A-3: Extend BundleMetadata + Bump BUNDLE_VERSION to 1.3.0

| Field | Value |
|-------|-------|
| Files | `src/inference/bundle.py` (MODIFY) |
| Effort | M (50-80 LOC) |
| Depends On | None |
| Blocks | P0-B-1, P0-B-2, P0-B-3 |

**API Signature -- new fields on `BundleMetadata`:**

```python
scaling_source: str = "bundle"
primary_timeframe: str = "5min"
mtf_timeframes: list[str] = field(default_factory=list)
feature_names: list[str] = field(default_factory=list)
arch_version: str = "0.0"
label_mapping: dict[int, str] = field(default_factory=dict)
scaler_type: str = "robust"
```

Also: `BUNDLE_VERSION = "1.3.0"` (line 54, currently `"1.2.0"`).

Update `to_dict()` to serialize all new fields. Update `from_dict()` to use `.get()` with safe defaults for all new fields.

**Acceptance Criteria:**
- [ ] `BUNDLE_VERSION` reads `"1.3.0"`
- [ ] All 7 new fields present with correct defaults
- [ ] `from_dict()` uses `.get()` -- v1.2.0 metadata JSON loads without error
- [ ] `to_dict()` includes all 7 new fields
- [ ] Round-trip: `BundleMetadata.from_dict(m.to_dict())` produces equivalent object
- [ ] `ruff check src/inference/bundle.py` passes

**Verification:**
```bash
python -c "
from src.inference.bundle import BundleMetadata, BUNDLE_VERSION
assert BUNDLE_VERSION == '1.3.0'
old = {'version':'1.2.0','created_at':'old','model_name':'xgb','model_family':'boosting','n_features':50,'input_rank':2}
m = BundleMetadata.from_dict(old)
assert m.primary_timeframe == '5min'
assert m.scaling_source == 'bundle'
print('OK')
" && ruff check src/inference/bundle.py
```

---

### P0-A-4: Fix Calibrator Transfer

| Field | Value |
|-------|-------|
| Files | `src/models/training/unified_orchestrator.py` (MODIFY), `src/inference/builder.py` (MODIFY) |
| Effort | M (30-50 LOC) |
| Depends On | None |
| Blocks | P0-A-6 |

**API Signature -- new fields on orchestrator `ModelTrainingResult` (L78-111):**

```python
calibrator: Any | None = None
calibration_metrics: Any | None = None
```

**Changes:**
1. Add `calibrator` + `calibration_metrics` fields to orchestrator's `ModelTrainingResult`
2. In `_train_single_model()` (L912-920): `calibrator=getattr(result, "calibrator", None)`
3. In `_train_boosting_parallel()` (L799-807): same pattern
4. In `builder.py _extract_calibrator()`: accept optional `model_result` param, check `model_result.calibrator` first, then fall back to duck-typing

**Acceptance Criteria:**
- [ ] `ModelTrainingResult(model_name="test", horizon=20)` works (calibrator defaults None)
- [ ] `_train_single_model()` copies calibrator from service result
- [ ] `_extract_calibrator(trainer, model_result=result)` checks `model_result.calibrator` first
- [ ] `ruff check src/models/training/unified_orchestrator.py src/inference/builder.py` passes

**Verification:**
```bash
python -c "
from src.models.training.unified_orchestrator import ModelTrainingResult
r = ModelTrainingResult(model_name='test', horizon=20)
assert r.calibrator is None
print('OK')
" && ruff check src/models/training/unified_orchestrator.py src/inference/builder.py
```

---

### P0-A-5: Fix Double-Scaling Bug

| Field | Value |
|-------|-------|
| Files | `src/inference/bundle.py` (MODIFY) |
| Effort | S (1-5 LOC) |
| Depends On | None |
| Blocks | P0-B-1 |

**Change:** In `preprocess()` at L1038-1042, change `skip_scaling=False` to `skip_scaling=True`.

```python
# BEFORE:
skip_scaling=False,

# AFTER:
skip_scaling=True,   # Bundle's own scaler applies in predict(); avoid double-scaling
```

**Acceptance Criteria:**
- [ ] `self.preprocessing_graph.transform()` called with `skip_scaling=True` in `preprocess()`
- [ ] Comment explains rationale
- [ ] `ruff check src/inference/bundle.py` passes

**Verification:**
```bash
grep -n "skip_scaling=True" src/inference/bundle.py | head -5
```

---

### P0-A-6: Make BundleBuilder Protocol-Aware

| Field | Value |
|-------|-------|
| Files | `src/inference/builder.py` (MODIFY) |
| Effort | M (60-90 LOC) |
| Depends On | P0-A-1, P0-A-3, P0-A-4 |
| Blocks | None |

**Changes:**
1. Update `_extract_model()`, `_extract_scaler()`, `_extract_feature_columns()`, `_extract_calibrator()` to check `isinstance(trainer, TrainerProtocol)` first, fall back to duck-typing
2. Populate new BundleMetadata fields from `get_model_contract()` when available

**API Signature (example for _extract_model):**

```python
def _extract_model(self, trainer: Any) -> Any | None:
    from src.core.protocols import TrainerProtocol
    if isinstance(trainer, TrainerProtocol):
        return trainer.model
    # Legacy duck-typing fallback (unchanged)
    for attr in ["model", "_model", "estimator", "_estimator"]:
        model = getattr(trainer, attr, None)
        if model is not None:
            return model
    ...
```

**Acceptance Criteria:**
- [ ] Each extraction method checks `isinstance(trainer, TrainerProtocol)` first
- [ ] Legacy duck-typing fallback chains remain intact
- [ ] New metadata fields populated from contract when available
- [ ] `ruff check src/inference/builder.py` passes

**Verification:**
```bash
python -c "from src.inference.builder import BundleBuilder; print('OK')" && \
ruff check src/inference/builder.py
```

---

## Phase P0-B: Core Inference

---

### P0-B-1: Add `_apply_adapter()` Routing to ModelBundle

| Field | Value |
|-------|-------|
| Files | `src/inference/bundle.py` (MODIFY) |
| Effort | M (40-60 LOC) |
| Depends On | P0-A-3, P0-A-5, P0-B-2, P0-B-3 |
| Blocks | P0-B-4 |

**API Signature:**

```python
def _apply_adapter(
    self,
    features_2d: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Route 2D preprocessed features through the appropriate adapter."""
    if self.metadata.requires_4d:
        return self._build_4d_input(raw_df)
    elif self.metadata.requires_sequences:
        return self._build_3d_input(features_2d)
    else:
        return features_2d.values.astype(np.float32)
```

Updated `predict_from_raw()`:

```python
def predict_from_raw(self, raw_df, calibrate=True, skip_cleaning=False):
    if self.metadata.requires_4d:
        adapted = self._build_4d_input(raw_df, skip_cleaning=skip_cleaning)
        return self.predict(adapted, calibrate=calibrate)
    features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)
    adapted = self._apply_adapter(features, raw_df=raw_df)
    return self.predict(adapted, calibrate=calibrate)
```

**Acceptance Criteria:**
- [ ] `predict_from_raw()` works for XGBoost (2D), LSTM (3D), PatchTST (4D) bundles
- [ ] `predict(pre_shaped_X)` continues to work unchanged
- [ ] Signature of `predict_from_raw()` unchanged
- [ ] `ruff check src/inference/bundle.py` passes

**Verification:**
```bash
python -c "
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_apply_adapter')
print('OK')
" && ruff check src/inference/bundle.py
```

---

### P0-B-2: Add `_build_3d_input()` (Sliding Window)

| Field | Value |
|-------|-------|
| Files | `src/inference/bundle.py` (MODIFY) |
| Effort | S (25-35 LOC) |
| Depends On | P0-A-3 |
| Blocks | P0-B-1 |

**API Signature:**

```python
def _build_3d_input(self, features_2d: pd.DataFrame) -> np.ndarray:
    """Convert 2D features to 3D sliding-window tensor (n_seq, seq_len, n_feat)."""
    seq_len = self.metadata.sequence_length or 60
    values = features_2d.values.astype(np.float32)
    n_rows, n_feat = values.shape
    if n_rows < seq_len:
        raise ValueError(f"Insufficient data: got {n_rows} rows, need {seq_len}")
    if n_rows == seq_len:
        return values.reshape(1, seq_len, n_feat)
    windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
    return windows.transpose(0, 2, 1).copy()
```

**Acceptance Criteria:**
- [ ] 60 rows, 50 features, seq_len=60 -> shape `(1, 60, 50)`
- [ ] 120 rows, 50 features, seq_len=60 -> shape `(61, 60, 50)`
- [ ] 30 rows, seq_len=60 -> `ValueError` with descriptive message
- [ ] Output dtype is `float32`
- [ ] `ruff check src/inference/bundle.py` passes

**Verification:**
```bash
python -c "
import numpy as np
values = np.random.randn(120, 50).astype(np.float32)
windows = np.lib.stride_tricks.sliding_window_view(values, 60, axis=0)
result = windows.transpose(0, 2, 1).copy()
assert result.shape == (61, 60, 50)
print('OK')
"
```

---

### P0-B-3: Add `_build_4d_input()` (MTF Generation)

| Field | Value |
|-------|-------|
| Files | `src/inference/bundle.py` (MODIFY) |
| Effort | L (80-120 LOC) |
| Depends On | P0-A-3 |
| Blocks | P0-B-1 |

**API Signature:**

```python
def _build_4d_input(
    self,
    raw_df: pd.DataFrame,
    skip_cleaning: bool = False,
) -> np.ndarray:
    """Build 4D multi-timeframe tensor (n_seq, n_tf, seq_len, n_feat) from raw OHLCV."""
```

Logic: reads `metadata.primary_timeframe`, `metadata.mtf_timeframes`, `metadata.sequence_length`. Resamples raw OHLCV to each timeframe using standard aggregation (open=first, high=max, low=min, close=last, volume=sum). Builds sliding windows per timeframe. Aligns by taking last `min_windows` from each. Stacks into 4D tensor.

**Acceptance Criteria:**
- [ ] 200 rows 1min OHLCV, primary="1min", mtf=["5min","15min"], seq_len=60 -> shape `(n, 3, 60, 5)`
- [ ] 30 rows -> `ValueError`
- [ ] Missing DatetimeIndex -> `ValueError`
- [ ] Missing OHLCV columns -> `ValueError`
- [ ] Output dtype is `float32`
- [ ] `ruff check src/inference/bundle.py` passes

**Verification:**
```bash
python -c "
import numpy as np, pandas as pd
dates = pd.date_range('2026-01-01', periods=200, freq='1min')
df = pd.DataFrame({'open':np.random.randn(200),'high':np.random.randn(200),
    'low':np.random.randn(200),'close':np.random.randn(200),
    'volume':np.abs(np.random.randn(200))*1000}, index=dates)
r5 = df.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
assert len(r5) > 0
print('OK')
"
```

---

### P0-B-4: Add `predict_from_raw()` to EnsembleBundle

| Field | Value |
|-------|-------|
| Files | `src/inference/ensemble_bundle.py` (MODIFY) |
| Effort | S (30-40 LOC) |
| Depends On | P0-B-1 |
| Blocks | None |

**API Signature:**

```python
def predict_from_raw(
    self,
    raw_df: pd.DataFrame,
    calibrate: bool = True,
    skip_cleaning: bool = False,
) -> Any:
    """End-to-end ensemble prediction from raw OHLCV data."""
    self._ensure_base_bundles_loaded()
    base_predictions: dict[str, np.ndarray] = {}
    for model_name, bundle in self._base_bundles.items():
        output = bundle.predict_from_raw(raw_df, calibrate=False, skip_cleaning=skip_cleaning)
        base_predictions[model_name] = output.class_probabilities
    return self.predict(base_predictions, calibrate=calibrate)
```

**Acceptance Criteria:**
- [ ] Method exists with `(raw_df, calibrate=True, skip_cleaning=False)` signature
- [ ] Base models called with `calibrate=False`
- [ ] Works with heterogeneous ensembles (XGBoost + LSTM)
- [ ] Raises `ValueError` if no base bundles or no meta-learner
- [ ] `ruff check src/inference/ensemble_bundle.py` passes

**Verification:**
```bash
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
import inspect
assert hasattr(EnsembleBundle, 'predict_from_raw')
sig = inspect.signature(EnsembleBundle.predict_from_raw)
assert 'raw_df' in sig.parameters
print('OK')
"
```

---

### P0-B-5: Fix EnsembleBundle Relative Paths

| Field | Value |
|-------|-------|
| Files | `src/inference/ensemble_bundle.py` (MODIFY) |
| Effort | S (20-30 LOC) |
| Depends On | None |
| Blocks | None |

**Changes:**
- `save()` at L442-452: convert `base_bundle_paths` to relative paths via `Path.relative_to(path.parent)`, with `try/except ValueError` fallback to absolute
- `load()` at L538-543: check `Path.is_absolute()`; if relative, resolve via `(path.parent / p).resolve()`

**Acceptance Criteria:**
- [ ] Newly saved bundles write relative paths in `base_bundles.json`
- [ ] Old bundles with absolute paths still load correctly
- [ ] `ruff check src/inference/ensemble_bundle.py` passes

**Verification:**
```bash
python -c "
from pathlib import Path
p = Path('/a/b/ensemble'); base = Path('/a/b/xgboost_h20')
rel = base.relative_to(p.parent)
assert (p.parent / rel).resolve() == base.resolve()
print('OK')
"
```

---

### P0-B-6: Create `src/inference/errors.py`

| Field | Value |
|-------|-------|
| Files | `src/inference/errors.py` (NEW) |
| Effort | S (25-35 LOC) |
| Depends On | None |
| Blocks | None |

**API Signature:**

```python
from __future__ import annotations

class InferenceError(Exception):
    """Base class for inference-related errors."""
    pass

class InferenceShapeMismatchError(InferenceError):
    def __init__(self, expected_shape, actual_shape, model_name="", hint=""):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.model_name = model_name
        self.hint = hint
        msg = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        if model_name: msg = f"[{model_name}] {msg}"
        if hint: msg = f"{msg}. {hint}"
        super().__init__(msg)
```

**Acceptance Criteria:**
- [ ] `from src.inference.errors import InferenceShapeMismatchError` succeeds
- [ ] Error message includes all four fields
- [ ] `ruff check src/inference/errors.py` passes

**Verification:**
```bash
python -c "
from src.inference.errors import InferenceShapeMismatchError
e = InferenceShapeMismatchError((1,60,50),(100,50),model_name='lstm',hint='Use windowing')
assert 'lstm' in str(e) and '(1, 60, 50)' in str(e)
print('OK')
"
```

---

## Phase P0-C: Deploy Packaging

---

### P0-C-1: Create `src/inference/deploy.py` with DeployManifest

| Field | Value |
|-------|-------|
| Files | `src/inference/deploy.py` (NEW) |
| Effort | M (100-130 LOC) |
| Depends On | None |
| Blocks | P0-C-2, P0-C-3, P0-C-4, P0-C-5, P0-C-6 |

**API Signature:**

```python
@dataclass
class HorizonArtifactEntry:
    horizon: int
    artifact_type: str          # "model" or "ensemble"
    model_key: str              # e.g. "xgboost_h20"
    model_family: str
    bundle_path: str            # Relative path from deploy/
    feature_count: int = 0
    sequence_length: int = 0
    requires_sequences: bool = False
    requires_4d: bool = False
    scaling_source: str = "bundle"
    metrics: dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    validation_path: str = ""

@dataclass
class DeployManifest:
    run_id: str
    created_at: str
    horizons: list[int]
    selected_artifacts: dict[str, HorizonArtifactEntry]
    runtime_profile: str = "native"
    total_models_trained: int = 0
    best_model_overall: str = ""
    ensemble_available: bool = False
    compatibility: dict[str, str] = field(default_factory=lambda: {"min_python": "3.10", "bundle_version": "1.3.0"})

    def to_dict(self) -> dict[str, Any]: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> DeployManifest: ...
```

**Acceptance Criteria:**
- [ ] Round-trip: `DeployManifest.load(path)` after `manifest.save(path)` produces equivalent object
- [ ] `manifest.json` loadable with `json.load()` -- no src/ imports needed
- [ ] `ruff check src/inference/deploy.py` passes

**Verification:**
```bash
python -c "
from src.inference.deploy import DeployManifest, HorizonArtifactEntry
import tempfile, json; from pathlib import Path
entry = HorizonArtifactEntry(horizon=20, artifact_type='model', model_key='xgboost_h20', model_family='boosting', bundle_path='h20/artifact')
m = DeployManifest(run_id='test', created_at='2026-01-01', horizons=[20], selected_artifacts={'h20': entry})
with tempfile.TemporaryDirectory() as d:
    m.save(Path(d)); m2 = DeployManifest.load(Path(d))
    assert m2.run_id == 'test'; print('OK')
"
```

---

### P0-C-2: Factory Phase 5 Integration

| Field | Value |
|-------|-------|
| Files | `src/factory.py` (MODIFY) |
| Effort | L (80-100 LOC) |
| Depends On | P0-C-1, P0-C-3, P0-C-4 |
| Blocks | P0-C-7, P0-D-1, P0-D-2, P0-D-3, P0-D-5 |

**Changes:**
1. Add `deploy_path: Path | None = None` field to `ExperimentResult`
2. Add `_create_deploy()` private method that builds `deploy/h{horizon}/artifact/`, selects best artifact, validates, writes `manifest.json`
3. Insert Phase 5 call after Phase 4 in `run()`: `deploy_path = self._create_deploy(training_result, bundle_path)`
4. Entire Phase 5 wrapped in `try/except` -- failure returns None, does not break run
5. Update phase numbering from `N/4` to `N/5`

**Acceptance Criteria:**
- [ ] `ExperimentResult.deploy_path` field exists, defaults None
- [ ] After `factory.run()` with bundling enabled: `result.deploy_path` points to `deploy/`
- [ ] `deploy/manifest.json` exists and is valid JSON
- [ ] `deploy/h{horizon}/artifact/` contains selected bundle
- [ ] Without bundling: `deploy_path` is None
- [ ] Deploy failure does not break factory run
- [ ] `ruff check src/factory.py` passes

**Verification:**
```bash
python -c "
from src.factory import ExperimentResult
r = ExperimentResult(success=True)
assert hasattr(r, 'deploy_path') and r.deploy_path is None
print('OK')
" && ruff check src/factory.py
```

---

### P0-C-3: Artifact Selector Logic

| Field | Value |
|-------|-------|
| Files | `src/inference/deploy.py` (MODIFY) |
| Effort | L (100-140 LOC) |
| Depends On | P0-C-1 |
| Blocks | P0-C-2 |

**API Signature:**

```python
def select_deploy_artifact(
    training_result: Any,
    bundle_path: Path,
    horizon: int,
    artifact_dir: Path,
    selection_metric: str = "val_f1",
    min_base_models_for_ensemble: int = 2,
) -> HorizonArtifactEntry | None:
```

**Selection policy:** Ensemble if exists, has >= min_base_models, and score >= best single model by `selection_metric`. Otherwise best single model. Copies selected bundle to `artifact_dir` via `shutil.copytree`. Returns `HorizonArtifactEntry` or None.

**Acceptance Criteria:**
- [ ] Correct selection for: 3 models no ensemble, ensemble > best, ensemble < best, 1 model only
- [ ] Copies correct bundle directory
- [ ] Returns None if no results for horizon
- [ ] `ruff check src/inference/deploy.py` passes

---

### P0-C-4: Validation Report Generation

| Field | Value |
|-------|-------|
| Files | `src/inference/deploy.py` (MODIFY) |
| Effort | M (80-100 LOC) |
| Depends On | P0-C-1 |
| Blocks | P0-C-2 |

**API Signature:**

```python
def validate_deploy_artifact(
    artifact_dir: Path,
    artifact_type: str = "model",
    sample_rows: int = 100,
) -> dict[str, Any]:
    """Returns {passed: bool, checks: [...], timing_seconds: float, ...}"""
```

Performs 5 checks: directory exists, manifest.json exists, bundle loads, validate() runs, smoke predict on synthetic data.

**Acceptance Criteria:**
- [ ] Valid bundle -> `{"passed": true}`
- [ ] Missing directory -> `{"passed": false}`
- [ ] Corrupt bundle -> `{"passed": false}` with `bundle_loads` failed
- [ ] `ruff check src/inference/deploy.py` passes

---

### P0-C-5: Deploy Helper (`load_deploy_artifact`)

| Field | Value |
|-------|-------|
| Files | `src/inference/deploy.py` (MODIFY) |
| Effort | S (40-50 LOC) |
| Depends On | P0-C-1 |
| Blocks | P0-D-1 |

**API Signature:**

```python
def load_deploy_artifact(
    deploy_dir: str | Path,
    horizon: int | None = None,
) -> Any:  # Returns ModelBundle or EnsembleBundle
```

Loads manifest, resolves horizon (auto-select if only one), loads appropriate bundle type.

**Acceptance Criteria:**
- [ ] Single horizon auto-selects without specifying
- [ ] Multiple horizons + no specification -> `ValueError`
- [ ] Returns correct bundle type based on manifest
- [ ] `ruff check src/inference/deploy.py` passes

---

### P0-C-6: Update `src/inference/__init__.py` Exports

| Field | Value |
|-------|-------|
| Files | `src/inference/__init__.py` (MODIFY) |
| Effort | S (10-15 LOC) |
| Depends On | P0-C-1, P0-C-3, P0-C-4, P0-C-5 |
| Blocks | None |

Add imports and `__all__` entries for: `DeployManifest`, `HorizonArtifactEntry`, `load_deploy_artifact`, `select_deploy_artifact`, `validate_deploy_artifact`.

**Acceptance Criteria:**
- [ ] `from src.inference import DeployManifest, load_deploy_artifact` succeeds
- [ ] `ruff check src/inference/__init__.py` passes

---

### P0-C-7: Add `deploy_artifact` Toggle to BundlingSection

| Field | Value |
|-------|-------|
| Files | `src/config/experiment.py` (MODIFY) |
| Effort | S (10-15 LOC) |
| Depends On | P0-C-2 |
| Blocks | P0-D-4 |

Add `deploy_artifact: bool = True` to `BundlingSection`. Update `from_dict()` and `to_dict()`.

**Acceptance Criteria:**
- [ ] `BundlingSection().deploy_artifact is True`
- [ ] `deploy_artifact=False` skips deploy
- [ ] Existing configs without key load with `True` default
- [ ] `ruff check src/config/experiment.py` passes

---

## Phase P0-D: Notebook Integration

All changes target `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb`.

---

### P0-D-1: Inference Demo Cell (Cell 8)

| Field | Value |
|-------|-------|
| Files | `notebooks/ml_factory_colab.ipynb` (new cell after cell 7) |
| Effort | L (~100 LOC) |
| Depends On | P0-C-2, P0-C-1 |

Loads deploy manifest, displays per-horizon artifact details, loads ModelBundle, runs `predict_from_raw()` demo on last N bars of `raw_data`. Falls back to `bundles/` listing if no `deploy/` exists. All operations wrapped in try/except.

**Acceptance Criteria:**
- [ ] Cell 8 titled "INFERENCE DEMO"
- [ ] Shows manifest info + per-horizon details
- [ ] Falls back to bundle listing without deploy/
- [ ] No errors when bundles/deploy/raw_data missing

---

### P0-D-2: Deploy Export Cell (Cell 9)

| Field | Value |
|-------|-------|
| Files | `notebooks/ml_factory_colab.ipynb` (new cell after cell 8) |
| Effort | S (35-40 LOC) |
| Depends On | P0-C-2 |

Zips `deploy/` directory only (not full output). Auto-download in Colab, prints path locally.

**Acceptance Criteria:**
- [ ] Cell 9 titled "EXPORT DEPLOY ARTIFACT"
- [ ] `.zip` contains only deploy/ contents
- [ ] Colab auto-download; local path printed

---

### P0-D-3: Validation Cell (Cell 10)

| Field | Value |
|-------|-------|
| Files | `notebooks/ml_factory_colab.ipynb` (new cell after cell 9) |
| Effort | S (45-50 LOC) |
| Depends On | P0-C-2, P0-C-4 |

Reads `manifest.json` and per-horizon `validation.json`. Displays per-check pass/fail + overall summary.

**Acceptance Criteria:**
- [ ] Cell 10 titled "VALIDATION REPORT"
- [ ] Per-check pass/fail with messages
- [ ] Overall "ALL PASSED" / "SOME CHECKS FAILED"

---

### P0-D-4: Config Additions (Cell 2 + Cell 5 Modification)

| Field | Value |
|-------|-------|
| Files | `notebooks/ml_factory_colab.ipynb` (modify cells 2 and 5) |
| Effort | S (10-15 LOC) |
| Depends On | P0-C-7 |

Add `BUNDLING_ENABLED = True` and `DEPLOY_ARTIFACT = True` to Cell 2. Update Cell 5 to import `BundlingSection` and pass to `ExperimentConfig`.

**Acceptance Criteria:**
- [ ] `BUNDLING_ENABLED` and `DEPLOY_ARTIFACT` in Cell 2
- [ ] `BundlingSection` import and usage in Cell 5

---

### P0-D-5: Drive Persistence Cell (Cell 11)

| Field | Value |
|-------|-------|
| Files | `notebooks/ml_factory_colab.ipynb` (new cell after cell 10) |
| Effort | S (45-50 LOC) |
| Depends On | P0-C-2, P0-D-1 |

Saves `deploy/` to Google Drive in Colab. Prints local path otherwise. Shows reload instructions.

**Acceptance Criteria:**
- [ ] Cell 11 titled "SAVE DEPLOY ARTIFACT TO GOOGLE DRIVE"
- [ ] Colab: copies to Drive, prints path + reload instructions
- [ ] Local: prints local deploy path

---

## Rollout Strategy

### Feature Flags

| Feature | Config Path | Default | Effect When False |
|---------|------------|---------|-------------------|
| `deploy_artifact` | `BundlingSection.deploy_artifact` | `True` | Skips Phase 5; no deploy/ created |
| `create_bundle` | `BundlingSection.create_bundle` | `True` | Skips Phase 4 AND Phase 5 |

### Migration Order

| Rollout Phase | Tasks | Risk | User-Visible Change |
|--------------|-------|------|---------------------|
| 1. Foundation | P0-A (Batch 1-2) | Very low | None. All additive or bug fixes |
| 2. Core Inference | P0-B (Batch 3-4) | Low | `predict_from_raw()` works for all 12 models |
| 3. Deploy Packaging | P0-C (Batch 5-6) | Low | Factory creates `deploy/` directory |
| 4. Notebook | P0-D (Batch 7) | Low | 4 new cells in notebook |

### Compatibility Window

| Component | Status After P0 | Deprecate After |
|-----------|----------------|-----------------|
| `InferencePipeline` | Active, NOT deprecated | P2 (after UIP built) |
| `InferenceOrchestrator` | Active, NOT deprecated | P2 (after deploy proven) |
| `ModelBundle.predict(X)` | Active, NEVER deprecated | Never |
| `predict_from_base_features()` | Active, NOT deprecated | P1 |
| Duck-typing in BundleBuilder | Active as fallback | P2 (after all trainers use protocol) |

### Success Metrics (Quantifiable)

| # | Criterion | Target |
|---|-----------|--------|
| 1 | Models pass `predict_from_raw()` | **12/12** |
| 2 | Notebook end-to-end in Colab | **12/12 cells pass** |
| 3 | Old v1.2.0 bundles load | **Yes** (safe defaults) |
| 4 | `ruff check src/` | **0 errors** |
| 5 | `black --check src/` | **0 reformats** |
| 6 | Deploy directory created | `deploy/` with `manifest.json` |
| 7 | Manifest is pure JSON | Loadable with `json.load()` only |
| 8 | Per-horizon validation | All `"passed": true` |
| 9 | Ensemble `predict_from_raw()` | Works for homogeneous + heterogeneous |
| 10 | Calibrator reaches bundles | `has_calibrator == True` when enabled |
| 11 | No double scaling | Single scaling verified |
| 12 | CRITICAL gaps closed | **5/5** (G1-G5) |
| 13 | Smoke tests pass | **24/24** |
| 14 | Integration tests pass | **6/6** |

---

## Strict Execution Order Checklist

```
BATCH 1 — Foundation (parallel, no dependencies)
 1. [ ] P0-A-1: Create src/core/protocols.py (TrainerProtocol + InferenceBundle)
 2. [ ] P0-A-2: Add ScalingSource enum to src/core/types.py
 3. [ ] P0-A-3: Extend BundleMetadata + bump BUNDLE_VERSION to 1.3.0
 4. [ ] P0-A-4: Fix calibrator transfer (orchestrator + builder)
 5. [ ] P0-A-5: Fix double-scaling bug (skip_scaling=True)
 6. [ ] P0-B-6: Create src/inference/errors.py

--- CHECKPOINT: Run P0-A smoke tests (SM-A1 through SM-A6, SM-B6) ---
--- Run: ruff check src/core/protocols.py src/core/types.py src/inference/bundle.py ---
--- Run: ruff check src/inference/builder.py src/models/training/unified_orchestrator.py ---

BATCH 2 — Adapters + Protocol Builder (depends on Batch 1)
 7. [ ] P0-A-6: Make BundleBuilder protocol-aware with legacy fallback
 8. [ ] P0-B-2: Add _build_3d_input() sliding window to ModelBundle
 9. [ ] P0-B-3: Add _build_4d_input() MTF generation to ModelBundle
10. [ ] P0-B-5: Fix EnsembleBundle relative paths (save/load)

--- CHECKPOINT: Run P0-B foundation smoke tests (SM-B2, SM-B3, SM-B5) ---
--- Verify: sliding window shapes, 4D resampling logic ---

BATCH 3 — Adapter Integration + Deploy Dataclass (depends on Batch 2)
11. [ ] P0-B-1: Add _apply_adapter() routing to ModelBundle + update predict_from_raw()
12. [ ] P0-C-1: Create src/inference/deploy.py (DeployManifest + HorizonArtifactEntry)

--- CHECKPOINT: Run SM-B1, SM-C1 ---
--- Verify: predict_from_raw() works for XGBoost (2D), LSTM (3D), PatchTST (4D) ---

BATCH 4 — Ensemble + Deploy Functions (depends on Batch 3)
13. [ ] P0-B-4: Add predict_from_raw() to EnsembleBundle
14. [ ] P0-C-3: Add select_deploy_artifact() to deploy.py
15. [ ] P0-C-4: Add validate_deploy_artifact() to deploy.py
16. [ ] P0-C-5: Add load_deploy_artifact() to deploy.py

--- CHECKPOINT: Run SM-B4, SM-C3, SM-C4, SM-C5 ---
--- Run integration tests IT-1, IT-2, IT-3, IT-4 ---

BATCH 5 — Factory + Config Integration (depends on Batch 4)
17. [ ] P0-C-2: Add Phase 5 (deploy packaging) to factory.py
18. [ ] P0-C-6: Update src/inference/__init__.py exports
19. [ ] P0-C-7: Add deploy_artifact toggle to BundlingSection

--- CHECKPOINT: Run SM-C2, SM-C6, SM-C7 ---
--- Run integration test IT-5 (deploy roundtrip) ---
--- Run: ruff check src/ && black --check src/ ---

BATCH 6 — Notebook Config (depends on Batch 5)
20. [ ] P0-D-4: Add BUNDLING_ENABLED + DEPLOY_ARTIFACT to notebook Cell 2+5

--- CHECKPOINT: Run SM-D4 ---

BATCH 7 — Notebook Cells (depends on Batch 5-6, all parallel)
21. [ ] P0-D-1: Add inference demo cell (Cell 8)
22. [ ] P0-D-2: Add deploy export cell (Cell 9)
23. [ ] P0-D-3: Add validation display cell (Cell 10)
24. [ ] P0-D-5: Add Drive persistence cell (Cell 11)

--- CHECKPOINT: Run SM-D1, SM-D2, SM-D3, SM-D5 ---
--- Run integration test IT-6 (old bundle compat) ---
--- Final: ruff check src/ && black --check src/ ---
--- Final: Run full notebook end-to-end in Colab ---
```

---

## Appendix: Gap Coverage

| Task | Gaps Closed | Gap IDs |
|------|-------------|---------|
| P0-A-1 | Protocol file, trainer protocol, inference bundle protocol | G10 |
| P0-A-2 | ScalingSource enum | G11 |
| P0-A-3 | BundleMetadata fields + version bump | G14, G20 |
| P0-A-4 | Calibrator transfer | G3, G21 (partial) |
| P0-A-5 | Double-scaling bug | G2 |
| P0-A-6 | Protocol-aware extraction, contract-based metadata | G17 (partial) |
| P0-B-1 | Universal adapter routing for 12/12 models | G1 |
| P0-B-2 | 3D sliding window for 7 sequence models | G16, G25 (partial) |
| P0-B-3 | 4D multi-timeframe preprocessing | G15 |
| P0-B-4 | EnsembleBundle.predict_from_raw | G4, G19 (partial) |
| P0-B-5 | Relative base bundle paths | G9 |
| P0-B-6 | Domain-specific inference errors | G22 |
| P0-C-1 | Deploy manifest dataclass | G5 (partial) |
| P0-C-2 | Deploy directory + factory integration | G5, G7 |
| P0-C-3 | Per-horizon artifact selection | G5 (partial), G8 (partial) |
| P0-C-4 | Validation/smoke test reports | G23 |
| P0-C-5 | User-facing deploy loader | G5 (partial) |
| P0-C-6 | Public API exports | (infrastructure) |
| P0-C-7 | Config toggle for deploy | (infrastructure) |
| P0-D-1 | Notebook inference demo | G13 (partial) |
| P0-D-2 | Notebook deploy export | G13 (partial) |
| P0-D-3 | Notebook validation display | G13 (partial) |
| P0-D-4 | Notebook config additions | G13 (partial) |
| P0-D-5 | Notebook Drive persistence | G13 (partial) |

### Coverage Summary

| Severity | Total | Fully Covered | Partially Covered | Deferred |
|----------|-------|--------------|-------------------|----------|
| CRITICAL (G1-G5) | 5 | **5** | 0 | 0 |
| HIGH (G6-G16) | 11 | 7 | 2 | 2 |
| MEDIUM (G17-G25) | 9 | 4 | 3 | 2 |
| LOW (G26-G29) | 4 | 0 | 0 | 4 |
| **TOTAL** | **29** | **16** | **5** | **8** |

**Deferred gaps (not in P0 scope):**
- G6 (build_ensemble_bundle format): Requires rewriting BundleBuilder ensemble output -> P1
- G12 (UniversalInferencePipeline): Large new component -> Phase 3B-2
- G18 (preprocessing config from training): Quality improvement -> P1
- G24 (meta-learner loading robustness): Edge case -> P1
- G25 partial (nbeats RAW feature mode): Windowing works; raw features deferred -> P1
- G26-G29 (cleanup): Low severity -> P1+

---

## Appendix: File Change Manifest

### New Files (3)

| File | Phase | Contents | LOC |
|------|-------|----------|-----|
| `/home/jake/Desktop/Research/src/core/protocols.py` | P0-A-1 | `TrainerProtocol`, `InferenceBundle` protocols | 40-60 |
| `/home/jake/Desktop/Research/src/inference/errors.py` | P0-B-6 | `InferenceError`, `InferenceShapeMismatchError` | 25-35 |
| `/home/jake/Desktop/Research/src/inference/deploy.py` | P0-C-1/3/4/5 | `DeployManifest`, `HorizonArtifactEntry`, `select_deploy_artifact()`, `validate_deploy_artifact()`, `load_deploy_artifact()` | 320-470 |

### Modified Files (7)

| File | Phases | Changes | LOC Delta |
|------|--------|---------|-----------|
| `/home/jake/Desktop/Research/src/core/types.py` | P0-A-2 | Add `ScalingSource` enum after `LabelingMethod` | +10-15 |
| `/home/jake/Desktop/Research/src/inference/bundle.py` | P0-A-3, P0-A-5, P0-B-1, P0-B-2, P0-B-3 | Bump BUNDLE_VERSION to 1.3.0; add 7 metadata fields + to_dict/from_dict; fix skip_scaling; add `_apply_adapter()`, `_build_3d_input()`, `_build_4d_input()`; update `predict_from_raw()` | +195-300 |
| `/home/jake/Desktop/Research/src/inference/builder.py` | P0-A-4, P0-A-6 | Protocol-aware extraction methods; calibrator from model_result; contract-based metadata population | +60-90 |
| `/home/jake/Desktop/Research/src/models/training/unified_orchestrator.py` | P0-A-4 | Add `calibrator`/`calibration_metrics` fields to ModelTrainingResult; copy calibrator in `_train_single_model()` and `_train_boosting_parallel()` | +10-20 |
| `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` | P0-B-4, P0-B-5 | Add `predict_from_raw()` method; relative paths in save/load | +50-70 |
| `/home/jake/Desktop/Research/src/factory.py` | P0-C-2 | Add `deploy_path` to ExperimentResult; add `_create_deploy()` method; Phase 5 in run(); phase renumbering | +80-100 |
| `/home/jake/Desktop/Research/src/config/experiment.py` | P0-C-7 | Add `deploy_artifact: bool = True` to BundlingSection; from_dict/to_dict | +5-10 |
| `/home/jake/Desktop/Research/src/inference/__init__.py` | P0-C-6 | Add deploy module imports + __all__ entries | +10-15 |
| `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb` | P0-D-1/2/3/4/5 | 4 new cells (8-11); modify cells 2 and 5 with config additions | +235-255 |

### Conflict Resolutions

| Conflict | Sources | Resolution |
|----------|---------|------------|
| ScalingSource location | MASTER-PLAN says `universal_pipeline.py`; CLAUDE.md says `types.py` | **Define in `src/core/types.py`**, import elsewhere. CLAUDE.md rule takes precedence. Warning W-1 agrees. |
| InferenceBundle location | MASTER-PLAN says `src/inference/`; CLAUDE.md says `src/core/protocols.py` | **Define in `src/core/protocols.py`**. CLAUDE.md rule takes precedence. Warning W-2 agrees. |
| Adapter in ModelBundle vs UIP | UNIFIED-ROADMAP says both; MASTER-PLAN says ModelBundle | **Methods on ModelBundle** (`_apply_adapter`, `_build_3d_input`, `_build_4d_input`). UIP can delegate to these when built. Avoids two implementations. |
| Ensemble per-horizon vs cross-horizon | Deploy plan requires per-horizon; orchestrator builds one ensemble across all horizons | **Known limitation**: ensemble artifact covers `horizons[0]` only. Per-horizon ensembles require orchestrator refactor (deferred to P1). Deploy selector assigns ensemble to its training horizon. |
| skip_scaling in predict_from_raw | Plan says `True`; code says `False` | **Fix to `True`**. This is the double-scaling bug (G2). Code matches plan after fix. |
| BundleBuilder ensemble format vs EnsembleBundle.load() | BundleBuilder saves custom files; EnsembleBundle expects different format | **Deferred to P1** (G6). Deploy selector falls back to best single model if ensemble bundle not loadable. |
| Model count 12 vs 23 | CLAUDE.md says 12; MODEL_CONTRACTS has 23 | **12 core** prediction models (notebook); 11 additional are classical/ensemble/meta-learner. P0 targets the 12 core models. |
| TFT classification | CLAUDE.md says Transformer; MODEL_CONTRACTS says neural with SEQUENCE_3D | **Functionally correct for routing**: TFT uses 3D input (sequence adapter), not 4D. Contract's `input_rank=SEQUENCE_3D` governs routing. |

---

*This is the definitive planning document for the ML Factory deployable-artifact implementation. All claims map to specific file paths. All API changes show exact signatures. No code has been modified.*

*Generated: 2026-02-15 by Agent 10/10 (Final Consolidator)*
*Source documents: Reports 01-09 in `/home/jake/Desktop/Research/.audit/deploy-plan/`*
