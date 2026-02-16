# 08 - Detailed Task Breakdown with Acceptance Criteria

**Date:** 2026-02-15
**Agent:** 8/10 (Task Breakdown Specialist)
**Purpose:** Comprehensive task breakdown for all 24 tasks across P0-A through P0-D, with acceptance criteria, dependency mapping, execution batches, and gap coverage matrix.

---

## Table of Contents

1. [P0-A: Foundation (6 tasks)](#p0-a-foundation)
2. [P0-B: Core Inference (6 tasks)](#p0-b-core-inference)
3. [P0-C: Deploy Packaging (7 tasks)](#p0-c-deploy-packaging)
4. [P0-D: Notebook Integration (5 tasks)](#p0-d-notebook-integration)
5. [Summary Tables](#summary-tables)
6. [Dependency Graph](#dependency-graph)
7. [Execution Batches](#execution-batches)
8. [Gap Coverage Matrix](#gap-coverage-matrix)

---

# P0-A: Foundation

---

### Task P0-A-1: Create `src/core/protocols.py` with TrainerProtocol and InferenceBundle

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/core/protocols.py` (NEW) |
| Depends On | None |
| Blocks | P0-A-6 |
| Effort | S |
| LOC Estimate | 40-60 |

**What to do:** Create a new file `src/core/protocols.py` containing two `@runtime_checkable` Protocol classes. `TrainerProtocol` defines the interface trainers must expose for bundle extraction (model, scaler, feature_columns, calibrator properties). `InferenceBundle` defines the interface for deployable artifacts (predict_from_raw, predict, validate, save, load).

**Acceptance Criteria:**
- [ ] File `src/core/protocols.py` exists and includes `from __future__ import annotations`
- [ ] `TrainerProtocol` is decorated with `@runtime_checkable` and defines `model`, `scaler`, `feature_columns`, `calibrator` as Protocol properties
- [ ] `InferenceBundle` is decorated with `@runtime_checkable` and defines `predict_from_raw()`, `predict()`, `validate()`, `save()`, `load()` methods
- [ ] Import succeeds: `from src.core.protocols import TrainerProtocol, InferenceBundle`
- [ ] Single definition: `grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l` returns 1
- [ ] Single definition: `grep -r "class InferenceBundle" src/ --include="*.py" | wc -l` returns 1
- [ ] Linting passes: `ruff check src/core/protocols.py` returns 0 errors

**Verification Command:**
```bash
python -c "from src.core.protocols import TrainerProtocol, InferenceBundle; print('OK')" && \
ruff check src/core/protocols.py && \
grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l && \
grep -r "class InferenceBundle" src/ --include="*.py" | wc -l
```

**Rollback:** Delete `src/core/protocols.py`. No other files modified.

---

### Task P0-A-2: Add `ScalingSource` Enum to `src/core/types.py`

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/core/types.py` (MODIFY) |
| Depends On | None |
| Blocks | None (used by future UIP in Phase 3B-2, included here as foundation type per CLAUDE.md) |
| Effort | S |
| LOC Estimate | 10-15 |

**What to do:** Add a `ScalingSource(str, Enum)` class to `src/core/types.py` after the `LabelingMethod` enum. It has three values: `BUNDLE = "bundle"`, `PREPROCESSING = "preprocessing"`, `NONE = "none"`. Include a docstring explaining that exactly one scaling source must be active per prediction path to prevent double-scaling.

**Acceptance Criteria:**
- [ ] `ScalingSource` enum exists in `src/core/types.py` with values BUNDLE, PREPROCESSING, NONE
- [ ] Import succeeds: `from src.core.types import ScalingSource`
- [ ] `ScalingSource.BUNDLE.value` returns `"bundle"`
- [ ] Single definition: `grep -r "class ScalingSource" src/ --include="*.py" | wc -l` returns 1
- [ ] Linting passes: `ruff check src/core/types.py` returns 0 errors
- [ ] Existing imports from `src/core/types.py` remain unaffected

**Verification Command:**
```bash
python -c "from src.core.types import ScalingSource; print(ScalingSource.BUNDLE.value)" && \
ruff check src/core/types.py && \
grep -r "class ScalingSource" src/ --include="*.py" | wc -l
```

**Rollback:** Remove the `ScalingSource` class from `src/core/types.py`. No other files modified.

---

### Task P0-A-3: Extend BundleMetadata with New Fields and Bump BUNDLE_VERSION

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/inference/bundle.py` (MODIFY) |
| Depends On | None |
| Blocks | P0-B-1, P0-B-2, P0-B-3 |
| Effort | M |
| LOC Estimate | 50-80 |

**What to do:** Add 7 new fields to the `BundleMetadata` dataclass: `scaling_source` (str, default "bundle"), `primary_timeframe` (str, default "5min"), `mtf_timeframes` (list[str], default []), `feature_names` (list[str], default []), `arch_version` (str, default "0.0"), `label_mapping` (dict[int,str], default {}), `scaler_type` (str, default "robust"). Update `to_dict()` and `from_dict()` to serialize/deserialize all new fields with safe defaults via `.get()`. Bump `BUNDLE_VERSION` from `"1.2.0"` to `"1.3.0"`.

**Acceptance Criteria:**
- [ ] `BUNDLE_VERSION` constant reads `"1.3.0"`
- [ ] All 7 new fields present on `BundleMetadata` dataclass with correct defaults
- [ ] `from_dict()` uses `.get()` with safe defaults for all new fields -- existing v1.2.0 metadata JSON loads without error
- [ ] `to_dict()` includes all 7 new fields in output
- [ ] Round-trip: `BundleMetadata.from_dict(metadata.to_dict())` produces equivalent object
- [ ] Linting passes: `ruff check src/inference/bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference.bundle import BundleMetadata, BUNDLE_VERSION
assert BUNDLE_VERSION == '1.3.0', f'Expected 1.3.0, got {BUNDLE_VERSION}'
m = BundleMetadata(version='1.3.0', created_at='now', model_name='test', model_family='boosting', n_features=10, input_rank=2)
d = m.to_dict()
assert 'primary_timeframe' in d
assert 'mtf_timeframes' in d
assert 'scaling_source' in d
m2 = BundleMetadata.from_dict(d)
assert m2.primary_timeframe == m.primary_timeframe
print('OK')
" && ruff check src/inference/bundle.py
```

**Rollback:** Revert `BundleMetadata` fields and `BUNDLE_VERSION` in `src/inference/bundle.py` to previous state. Git diff of the file shows exactly what to restore.

---

### Task P0-A-4: Fix Calibrator Transfer (Orchestrator -> ModelTrainingResult -> BundleBuilder)

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/models/training/unified_orchestrator.py` (MODIFY), `src/inference/builder.py` (MODIFY) |
| Depends On | None |
| Blocks | P0-A-6 |
| Effort | M |
| LOC Estimate | 30-50 |

**What to do:** Add `calibrator: Any | None = None` and `calibration_metrics: Any | None = None` fields to the orchestrator's `ModelTrainingResult` dataclass (L78-111). Update `_train_single_model()` at L912-920 and `_train_boosting_parallel()` at L799-807 to copy `calibrator` via `getattr(result, "calibrator", None)`. In `builder.py`, update `_extract_calibrator()` to accept an optional `model_result` parameter and check `model_result.calibrator` before falling back to duck-typing on the trainer.

**Acceptance Criteria:**
- [ ] Orchestrator `ModelTrainingResult` dataclass has `calibrator` and `calibration_metrics` fields defaulting to `None`
- [ ] `_train_single_model()` copies `calibrator` from service result to `ModelTrainingResult` via `getattr(result, "calibrator", None)`
- [ ] `_train_boosting_parallel()` also copies `calibrator` similarly
- [ ] `BundleBuilder._extract_calibrator()` accepts optional `model_result` param and checks it first
- [ ] Backward compat: `ModelTrainingResult(model_name="test", horizon=20)` still works without `calibrator` argument
- [ ] Linting passes: `ruff check src/models/training/unified_orchestrator.py src/inference/builder.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.models.training.unified_orchestrator import ModelTrainingResult
r = ModelTrainingResult(model_name='test', horizon=20)
assert r.calibrator is None
assert hasattr(r, 'calibration_metrics')
print('OK')
" && ruff check src/models/training/unified_orchestrator.py src/inference/builder.py
```

**Rollback:** Remove `calibrator` and `calibration_metrics` fields from orchestrator's `ModelTrainingResult`. Revert `_extract_calibrator()` in builder.py. Revert conversion code in `_train_single_model()` and `_train_boosting_parallel()`.

---

### Task P0-A-5: Fix Double-Scaling Bug in `ModelBundle.predict_from_raw()`

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/inference/bundle.py` (MODIFY) |
| Depends On | None |
| Blocks | P0-B-1 |
| Effort | S |
| LOC Estimate | 1-5 |

**What to do:** In `ModelBundle.preprocess()` at L1038-1042, change `skip_scaling=False` to `skip_scaling=True` in the `self.preprocessing_graph.transform()` call. Add a comment explaining that the bundle's own scaler applies in `predict()` so preprocessing must skip scaling to avoid double-scaling.

**Acceptance Criteria:**
- [ ] `preprocess()` calls `self.preprocessing_graph.transform()` with `skip_scaling=True`
- [ ] Comment present explaining the rationale for skip_scaling=True
- [ ] For a bundle with both preprocessing graph scaler and bundle scaler, `predict_from_raw()` applies scaling exactly once (in `predict()`)
- [ ] Linting passes: `ruff check src/inference/bundle.py` returns 0 errors

**Verification Command:**
```bash
grep -n "skip_scaling=True" src/inference/bundle.py | head -5 && \
ruff check src/inference/bundle.py
```

**Rollback:** Change `skip_scaling=True` back to `skip_scaling=False` on the same line.

---

### Task P0-A-6: Make BundleBuilder Protocol-Aware with Legacy Fallback

| Field | Value |
|-------|-------|
| Phase | P0-A |
| Files | `src/inference/builder.py` (MODIFY) |
| Depends On | P0-A-1, P0-A-3, P0-A-4 |
| Blocks | None |
| Effort | M |
| LOC Estimate | 60-90 |

**What to do:** Update `_extract_model()`, `_extract_scaler()`, `_extract_feature_columns()`, and `_extract_calibrator()` in BundleBuilder to first check if the trainer satisfies `TrainerProtocol` via `isinstance()`, and if so use protocol properties directly. If not, fall back to the existing duck-typing chains (unchanged). Additionally, update `build_from_training_result()` to populate the new BundleMetadata fields (`primary_timeframe`, `mtf_timeframes`, `scaler_type`) from model contracts when available via `get_model_contract()`.

**Acceptance Criteria:**
- [ ] Each extraction method (`_extract_model`, `_extract_scaler`, `_extract_feature_columns`, `_extract_calibrator`) checks `isinstance(trainer, TrainerProtocol)` first
- [ ] Legacy duck-typing fallback chains remain intact for non-protocol trainers
- [ ] New BundleMetadata fields populated from contract when available (via `get_model_contract()`)
- [ ] For a trainer NOT implementing protocol: extraction behavior is unchanged
- [ ] Import `from src.core.protocols import TrainerProtocol` does not fail
- [ ] Linting passes: `ruff check src/inference/builder.py` returns 0 errors

**Verification Command:**
```bash
python -c "from src.inference.builder import BundleBuilder; print('OK')" && \
ruff check src/inference/builder.py
```

**Rollback:** Revert all extraction methods in `builder.py` to remove `TrainerProtocol` checks. Remove contract lookup code in `build_from_training_result()`.

---

# P0-B: Core Inference

---

### Task P0-B-1: Add `_apply_adapter()` Routing to ModelBundle

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/bundle.py` (MODIFY) |
| Depends On | P0-A-3, P0-A-5, P0-B-2, P0-B-3 |
| Blocks | P0-B-4 |
| Effort | M |
| LOC Estimate | 40-60 |

**What to do:** Add a `_apply_adapter()` method to `ModelBundle` that routes 2D preprocessed features through the appropriate adapter based on metadata flags (`requires_4d`, `requires_sequences`). Update `predict_from_raw()` to: (a) for 4D models, call `_build_4d_input()` directly with raw_df, bypassing PreprocessingGraph; (b) for 2D/3D models, call `preprocess()` then `_apply_adapter()` then `predict()`.

**Acceptance Criteria:**
- [ ] `_apply_adapter()` method exists on ModelBundle
- [ ] `_apply_adapter()` returns 2D ndarray for tabular models (requires_4d=False, requires_sequences=False)
- [ ] `_apply_adapter()` calls `_build_3d_input()` for sequence models (requires_sequences=True)
- [ ] `_apply_adapter()` calls `_build_4d_input()` for 4D models (requires_4d=True)
- [ ] `predict_from_raw()` signature unchanged: same parameters, same return type (`PredictionResult`)
- [ ] `predict_from_raw(raw_df)` works for XGBoost (2D tabular) bundles
- [ ] `predict_from_raw(raw_df)` works for LSTM (3D sequence) bundles with sufficient data rows
- [ ] `predict_from_raw(raw_df)` works for PatchTST (4D multi-TF) bundles with raw 1min OHLCV
- [ ] `predict(pre_shaped_X)` continues to work for all pre-existing use cases
- [ ] Linting passes: `ruff check src/inference/bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_apply_adapter'), 'Missing _apply_adapter method'
print('OK')
" && ruff check src/inference/bundle.py
```

**Rollback:** Remove `_apply_adapter()` method. Revert `predict_from_raw()` to its original implementation at L1056-1077.

---

### Task P0-B-2: Add `_build_3d_input()` for Sequence Models (Sliding Window)

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/bundle.py` (MODIFY) |
| Depends On | P0-A-3 |
| Blocks | P0-B-1 |
| Effort | S |
| LOC Estimate | 25-35 |

**What to do:** Add a `_build_3d_input()` private method to `ModelBundle` that converts a 2D feature DataFrame to a 3D sliding-window tensor for sequence models. Uses `numpy.lib.stride_tricks.sliding_window_view()` for efficiency. Reads `self.metadata.sequence_length` (default 60). Raises `ValueError` if input rows < sequence_length.

**Acceptance Criteria:**
- [ ] `_build_3d_input()` method exists on ModelBundle
- [ ] For 60-row DataFrame with 50 features and sequence_length=60: returns shape `(1, 60, 50)`
- [ ] For 120-row DataFrame with 50 features and sequence_length=60: returns shape `(61, 60, 50)`
- [ ] For 30-row DataFrame with sequence_length=60: raises `ValueError` with descriptive message including row count and required length
- [ ] Output dtype is `float32`
- [ ] Uses `np.lib.stride_tricks.sliding_window_view` (not manual loop)
- [ ] Linting passes: `ruff check src/inference/bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
import numpy as np
import pandas as pd
from src.inference.bundle import BundleMetadata

# Test sliding window shape calculation
seq_len = 60
n_feat = 50
n_rows = 120
values = np.random.randn(n_rows, n_feat).astype(np.float32)
windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
result = windows.transpose(0, 2, 1)
assert result.shape == (61, 60, 50), f'Got {result.shape}'
print('Sliding window logic OK')
" && ruff check src/inference/bundle.py
```

**Rollback:** Remove `_build_3d_input()` method from `ModelBundle` class.

---

### Task P0-B-3: Add `_build_4d_input()` for Transformer Models (MTF Generation)

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/bundle.py` (MODIFY) |
| Depends On | P0-A-3 |
| Blocks | P0-B-1 |
| Effort | L |
| LOC Estimate | 80-120 |

**What to do:** Add a `_build_4d_input()` private method to `ModelBundle` that builds 4D multi-timeframe tensors from raw OHLCV data for transformer models (PatchTST, iTransformer). Bypasses PreprocessingGraph entirely. Resamples raw OHLCV to each required timeframe using pandas `.resample()` with standard OHLCV aggregation (open=first, high=max, low=min, close=last, volume=sum). Builds sliding windows per timeframe and stacks into 4D tensor `(n_sequences, n_timeframes, sequence_length, n_features)`.

**Acceptance Criteria:**
- [ ] `_build_4d_input()` method exists on ModelBundle
- [ ] For raw 1min OHLCV DataFrame with 200 rows, primary_timeframe="1min", mtf_timeframes=["5min","15min"], sequence_length=60: returns shape `(n, 3, 60, 5)` where n > 0
- [ ] For raw 1min data with only 30 rows: raises `ValueError` with descriptive message
- [ ] Resampled higher-TF bars use correct OHLCV aggregation (first/max/min/last/sum)
- [ ] Requires DatetimeIndex on input DataFrame; raises `ValueError` if not present
- [ ] Requires at least 4 of OHLCV columns; raises `ValueError` if missing
- [ ] Output dtype is `float32`
- [ ] No forward-looking data in windows (each window uses only past data up to that point)
- [ ] Linting passes: `ruff check src/inference/bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
import numpy as np
import pandas as pd

# Test resampling logic
dates = pd.date_range('2026-01-01', periods=200, freq='1min')
df = pd.DataFrame({
    'open': np.random.randn(200),
    'high': np.random.randn(200),
    'low': np.random.randn(200),
    'close': np.random.randn(200),
    'volume': np.abs(np.random.randn(200)) * 1000,
}, index=dates)
resampled = df.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
assert len(resampled) > 0, 'Resampling produced no rows'
print(f'5min bars: {len(resampled)} from {len(df)} 1min bars')
print('Resampling logic OK')
" && ruff check src/inference/bundle.py
```

**Rollback:** Remove `_build_4d_input()` method from `ModelBundle` class.

---

### Task P0-B-4: Add `predict_from_raw()` to EnsembleBundle

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/ensemble_bundle.py` (MODIFY) |
| Depends On | P0-B-1 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 30-40 |

**What to do:** Add a `predict_from_raw()` method to `EnsembleBundle` that loads base bundles, calls `predict_from_raw()` on each base bundle with `calibrate=False` (so calibration happens only at ensemble level), stacks base prediction outputs, and runs the meta-learner via the existing `predict()` method. Raises `ValueError` if no base bundles are loaded or meta-learner is None.

**Acceptance Criteria:**
- [ ] `predict_from_raw()` method exists on EnsembleBundle with signature `(raw_df, calibrate=True, skip_cleaning=False)`
- [ ] Base model `predict_from_raw()` is called with `calibrate=False`
- [ ] Works with heterogeneous ensembles (e.g., XGBoost + LSTM) because each base bundle handles its own adapter routing
- [ ] Raises `ValueError` with descriptive message if no base bundles loaded
- [ ] Raises `ValueError` with descriptive message if meta-learner is None
- [ ] Satisfies the `InferenceBundle` protocol structurally
- [ ] Linting passes: `ruff check src/inference/ensemble_bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
assert hasattr(EnsembleBundle, 'predict_from_raw'), 'Missing predict_from_raw'
import inspect
sig = inspect.signature(EnsembleBundle.predict_from_raw)
assert 'raw_df' in sig.parameters
assert 'calibrate' in sig.parameters
print('OK')
" && ruff check src/inference/ensemble_bundle.py
```

**Rollback:** Remove the `predict_from_raw()` method from `EnsembleBundle` class.

---

### Task P0-B-5: Fix EnsembleBundle Relative Paths (Save/Load)

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/ensemble_bundle.py` (MODIFY) |
| Depends On | None |
| Blocks | None |
| Effort | S |
| LOC Estimate | 20-30 |

**What to do:** In `EnsembleBundle.save()` at L442-452, convert `base_bundle_paths` to relative paths (relative to the ensemble bundle's parent directory) before writing to `base_bundles.json`. In `EnsembleBundle.load()` at L538-543, resolve relative paths by joining with the ensemble bundle's parent directory. Use `Path.is_absolute()` to handle both old (absolute) and new (relative) path formats.

**Acceptance Criteria:**
- [ ] Newly saved ensemble bundles write relative paths in `base_bundles.json`
- [ ] Old ensemble bundles with absolute paths in `base_bundles.json` still load correctly
- [ ] `save()` uses `try/except ValueError` for `relative_to()` to handle edge cases (different drive roots)
- [ ] `load()` resolves relative paths via `(path.parent / p).resolve()`
- [ ] Linting passes: `ruff check src/inference/ensemble_bundle.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from pathlib import Path
# Test relative path resolution logic
ensemble_path = Path('/home/user/deploy/h20/ensemble')
base_path = Path('/home/user/deploy/h20/xgboost_h20')
try:
    rel = base_path.relative_to(ensemble_path.parent)
    resolved = (ensemble_path.parent / rel).resolve()
    assert resolved == base_path.resolve()
    print(f'Relative: {rel}')
    print('OK')
except ValueError:
    print('Cannot make relative (different root)')
" && ruff check src/inference/ensemble_bundle.py
```

**Rollback:** Revert `save()` and `load()` in `ensemble_bundle.py` to use absolute paths as before.

---

### Task P0-B-6: Create `src/inference/errors.py` with InferenceShapeMismatchError

| Field | Value |
|-------|-------|
| Phase | P0-B |
| Files | `src/inference/errors.py` (NEW) |
| Depends On | None |
| Blocks | None (can be adopted incrementally by P0-B-1/B-2/B-3) |
| Effort | S |
| LOC Estimate | 25-35 |

**What to do:** Create a new file `src/inference/errors.py` with `InferenceError` base class and `InferenceShapeMismatchError` subclass. The shape error stores `expected_shape`, `actual_shape`, `model_name`, and `hint` fields and produces a descriptive error message combining all of them.

**Acceptance Criteria:**
- [ ] File `src/inference/errors.py` exists with `from __future__ import annotations`
- [ ] `InferenceError` base class extends `Exception`
- [ ] `InferenceShapeMismatchError` extends `InferenceError` and stores expected_shape, actual_shape, model_name, hint
- [ ] Error message includes all four fields when provided
- [ ] Import succeeds: `from src.inference.errors import InferenceShapeMismatchError`
- [ ] Single definition: `grep -r "class InferenceShapeMismatchError" src/ --include="*.py" | wc -l` returns 1
- [ ] Linting passes: `ruff check src/inference/errors.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference.errors import InferenceShapeMismatchError, InferenceError
e = InferenceShapeMismatchError(
    expected_shape=(1, 60, 50),
    actual_shape=(100, 50),
    model_name='lstm',
    hint='Use _build_3d_input() for sequence models'
)
assert 'lstm' in str(e)
assert '(1, 60, 50)' in str(e)
assert '(100, 50)' in str(e)
assert issubclass(InferenceShapeMismatchError, InferenceError)
print('OK')
" && ruff check src/inference/errors.py
```

**Rollback:** Delete `src/inference/errors.py`. No other files modified.

---

# P0-C: Deploy Packaging

---

### Task P0-C-1: Add `DeployManifest` and `HorizonArtifactEntry` Dataclasses

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/inference/deploy.py` (NEW) |
| Depends On | None |
| Blocks | P0-C-2, P0-C-3, P0-C-4, P0-C-5, P0-C-6 |
| Effort | M |
| LOC Estimate | 100-130 |

**What to do:** Create a new file `src/inference/deploy.py` with two dataclasses: `HorizonArtifactEntry` (per-horizon artifact metadata with horizon, artifact_type, model_key, model_family, bundle_path, feature_count, sequence_length, requires_sequences, requires_4d, scaling_source, metrics, validation_passed, validation_path) and `DeployManifest` (run-level manifest with run_id, created_at, horizons, selected_artifacts, runtime_profile, compatibility). Include `to_dict()`, `save()`, `load()` on `DeployManifest`. All serialization uses pure JSON -- no src/ imports needed to load.

**Acceptance Criteria:**
- [ ] File `src/inference/deploy.py` exists with `from __future__ import annotations`
- [ ] `HorizonArtifactEntry` dataclass with all 13 fields as specified
- [ ] `DeployManifest` dataclass with `to_dict()`, `save()`, `load()` methods
- [ ] Round-trip: `DeployManifest.load(path)` after `manifest.save(path)` produces equivalent object
- [ ] `manifest.json` is valid JSON loadable with `json.load()` without any src/ imports
- [ ] Import succeeds: `from src.inference.deploy import DeployManifest, HorizonArtifactEntry`
- [ ] Linting passes: `ruff check src/inference/deploy.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference.deploy import DeployManifest, HorizonArtifactEntry
import tempfile, json
from pathlib import Path

entry = HorizonArtifactEntry(horizon=20, artifact_type='model', model_key='xgboost_h20', model_family='boosting', bundle_path='h20/artifact')
m = DeployManifest(run_id='test', created_at='2026-01-01', horizons=[20], selected_artifacts={'h20': entry})
with tempfile.TemporaryDirectory() as d:
    m.save(Path(d))
    with open(Path(d) / 'manifest.json') as f:
        raw = json.load(f)
    assert 'run_id' in raw
    m2 = DeployManifest.load(Path(d))
    assert m2.run_id == 'test'
    print('OK')
" && ruff check src/inference/deploy.py
```

**Rollback:** Delete `src/inference/deploy.py`. No other files modified at this point.

---

### Task P0-C-2: Deploy Directory Creation in factory.py (Phase 5)

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/factory.py` (MODIFY) |
| Depends On | P0-C-1, P0-C-3, P0-C-4 |
| Blocks | P0-C-7, P0-D-1, P0-D-2, P0-D-3, P0-D-5 |
| Effort | L |
| LOC Estimate | 80-100 |

**What to do:** Add Phase 5 (Deploy Packaging) to `factory.run()` after Phase 4 (Bundling). Add `deploy_path: Path | None = None` field to `ExperimentResult`. Create new `_create_deploy()` private method that: builds per-horizon directories `deploy/h{horizon}/artifact/`, selects best artifact per horizon via `select_deploy_artifact()`, validates via `validate_deploy_artifact()`, writes `validation.json` per horizon, writes `manifest.json`. The entire phase is wrapped in try/except so deploy failure does not break the factory run. Update phase numbering in log messages from `N/4` to `N/5`. Add `import json` if not present.

**Acceptance Criteria:**
- [ ] `ExperimentResult` has `deploy_path: Path | None = None` field
- [ ] `_create_deploy()` method exists on the factory class
- [ ] After successful `factory.run()` with bundling enabled, `result.deploy_path` is a `Path` to `deploy/`
- [ ] `deploy/manifest.json` exists and is valid JSON
- [ ] `deploy/h{horizon}/artifact/` contains the selected bundle for each horizon
- [ ] `deploy/h{horizon}/validation.json` exists for each horizon
- [ ] Without bundling enabled (`create_bundle=False`), `deploy_path` is `None`
- [ ] If deploy packaging fails, factory run still succeeds (deploy_path is None)
- [ ] Phase labels in logs read `[Phase N/5]` instead of `[Phase N/4]`
- [ ] Linting passes: `ruff check src/factory.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.factory import ExperimentResult
r = ExperimentResult(success=True)
assert hasattr(r, 'deploy_path')
assert r.deploy_path is None
print('OK')
" && ruff check src/factory.py
```

**Rollback:** Remove `_create_deploy()` method, `deploy_path` field from `ExperimentResult`, Phase 5 call from `run()`, and revert phase numbering to `N/4`.

---

### Task P0-C-3: Artifact Selector Logic

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/inference/deploy.py` (MODIFY -- add to file from P0-C-1) |
| Depends On | P0-C-1 |
| Blocks | P0-C-2 |
| Effort | L |
| LOC Estimate | 100-140 |

**What to do:** Add `select_deploy_artifact()` function to `src/inference/deploy.py`. Implements the selection policy: (1) collect all single-model results for the given horizon from training_result, (2) sort by selection_metric descending, (3) if ensemble exists and has >= min_base_models and scores >= best single model, select ensemble, (4) otherwise select best single model. Copies the selected bundle directory to the artifact_dir. Returns `HorizonArtifactEntry` or None if no candidates found.

**Acceptance Criteria:**
- [ ] Function `select_deploy_artifact()` exists in `src/inference/deploy.py`
- [ ] With 3 models trained for horizon 20, no ensemble: selects model with highest val_f1
- [ ] With 3 models + ensemble where ensemble > best single: selects ensemble
- [ ] With 3 models + ensemble where ensemble < best single: selects best single model
- [ ] With ensemble but only 1 model trained: selects single model (min_base_models=2 default)
- [ ] Copies correct bundle directory to artifact_dir via `shutil.copytree`
- [ ] Returns `None` if no model results for given horizon
- [ ] All result access uses `getattr()` and `.get()` defensively
- [ ] Linting passes: `ruff check src/inference/deploy.py` returns 0 errors

**Verification Command:**
```bash
python -c "from src.inference.deploy import select_deploy_artifact; print('OK')" && \
ruff check src/inference/deploy.py
```

**Rollback:** Remove `select_deploy_artifact()` function from `src/inference/deploy.py`.

---

### Task P0-C-4: Validation Report Generation

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/inference/deploy.py` (MODIFY -- add to same file) |
| Depends On | P0-C-1 |
| Blocks | P0-C-2 |
| Effort | M |
| LOC Estimate | 80-100 |

**What to do:** Add `validate_deploy_artifact()` and helper `_validation_result()` functions to `src/inference/deploy.py`. Performs 5 checks: (1) directory exists, (2) manifest.json exists, (3) bundle loads via ModelBundle.load() or EnsembleBundle.load(), (4) bundle.validate() runs, (5) smoke predict on synthetic data with appropriate shape. Returns dict with `passed`, `checks`, `timing_seconds`, `artifact_type`, `artifact_dir`.

**Acceptance Criteria:**
- [ ] Function `validate_deploy_artifact()` exists in `src/inference/deploy.py`
- [ ] For valid ModelBundle directory: returns `{"passed": true, "checks": [...]}`
- [ ] For missing directory: returns `{"passed": false}` with `directory_exists` check failed
- [ ] For corrupt/invalid bundle: returns `{"passed": false}` with `bundle_loads` check failed
- [ ] Smoke predict uses synthetic ndarray of correct shape based on metadata (2D/3D/4D)
- [ ] All bundle operations wrapped in try/except for graceful failure
- [ ] Output is valid JSON-serializable dict
- [ ] Linting passes: `ruff check src/inference/deploy.py` returns 0 errors

**Verification Command:**
```bash
python -c "from src.inference.deploy import validate_deploy_artifact; print('OK')" && \
ruff check src/inference/deploy.py
```

**Rollback:** Remove `validate_deploy_artifact()` and `_validation_result()` from `src/inference/deploy.py`.

---

### Task P0-C-5: Deploy Helper Function (`load_deploy_artifact`)

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/inference/deploy.py` (MODIFY -- add to same file) |
| Depends On | P0-C-1 |
| Blocks | P0-D-1 |
| Effort | S |
| LOC Estimate | 40-50 |

**What to do:** Add `load_deploy_artifact()` function to `src/inference/deploy.py`. This is the primary user-facing function: takes a deploy_dir path and optional horizon, loads manifest.json, resolves the artifact path, and returns a loaded ModelBundle or EnsembleBundle. If horizon is None and only one horizon exists, auto-selects it. Raises FileNotFoundError or ValueError with clear messages.

**Acceptance Criteria:**
- [ ] Function `load_deploy_artifact()` exists in `src/inference/deploy.py`
- [ ] With single horizon, `load_deploy_artifact("path/to/deploy")` works without specifying horizon
- [ ] With multiple horizons and no horizon specified: raises `ValueError` listing available horizons
- [ ] With nonexistent directory: raises `FileNotFoundError`
- [ ] With unknown horizon: raises `ValueError` listing available horizons
- [ ] Returns `ModelBundle` or `EnsembleBundle` depending on artifact_type in manifest
- [ ] Linting passes: `ruff check src/inference/deploy.py` returns 0 errors

**Verification Command:**
```bash
python -c "from src.inference.deploy import load_deploy_artifact; print('OK')" && \
ruff check src/inference/deploy.py
```

**Rollback:** Remove `load_deploy_artifact()` from `src/inference/deploy.py`.

---

### Task P0-C-6: Update `src/inference/__init__.py` Exports

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/inference/__init__.py` (MODIFY) |
| Depends On | P0-C-1, P0-C-3, P0-C-4, P0-C-5 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 10-15 |

**What to do:** Add imports and `__all__` entries for `DeployManifest`, `HorizonArtifactEntry`, `load_deploy_artifact`, `select_deploy_artifact`, and `validate_deploy_artifact` from `src.inference.deploy` in `src/inference/__init__.py`.

**Acceptance Criteria:**
- [ ] All 5 names importable via `from src.inference import DeployManifest, load_deploy_artifact, ...`
- [ ] All 5 names present in `__all__` list
- [ ] Existing exports unchanged
- [ ] Linting passes: `ruff check src/inference/__init__.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.inference import DeployManifest, HorizonArtifactEntry, load_deploy_artifact, select_deploy_artifact, validate_deploy_artifact
print('OK')
" && ruff check src/inference/__init__.py
```

**Rollback:** Remove the added import block and `__all__` entries from `src/inference/__init__.py`.

---

### Task P0-C-7: Add `deploy_artifact` Toggle to BundlingSection

| Field | Value |
|-------|-------|
| Phase | P0-C |
| Files | `src/config/experiment.py` (MODIFY) |
| Depends On | P0-C-2 |
| Blocks | P0-D-4 |
| Effort | S |
| LOC Estimate | 10-15 |

**What to do:** Add `deploy_artifact: bool = True` field to the `BundlingSection` dataclass in `src/config/experiment.py`. Update `from_dict()` to parse it via `.get("deploy_artifact", True)` and `to_dict()` to serialize it. Update `factory.py` Phase 5 gate to check `self.config.bundling.deploy_artifact`.

**Acceptance Criteria:**
- [ ] `BundlingSection.deploy_artifact` field exists with default `True`
- [ ] `from_dict()` parses `deploy_artifact` with `.get("deploy_artifact", True)` default
- [ ] `to_dict()` includes `deploy_artifact` in output
- [ ] Existing config JSON without `deploy_artifact` key loads successfully (defaults to True)
- [ ] Setting `deploy_artifact=False` in config skips deploy directory creation
- [ ] Linting passes: `ruff check src/config/experiment.py` returns 0 errors

**Verification Command:**
```bash
python -c "
from src.config.experiment import BundlingSection
b = BundlingSection()
assert b.deploy_artifact is True
print('OK')
" && ruff check src/config/experiment.py
```

**Rollback:** Remove `deploy_artifact` field from `BundlingSection`. Revert `from_dict()` and `to_dict()` changes. Remove the config gate in `factory.py`.

---

# P0-D: Notebook Integration

---

### Task P0-D-1: Inference Demo Cell (Cell 8)

| Field | Value |
|-------|-------|
| Phase | P0-D |
| Files | `notebooks/ml_factory_colab.ipynb` (MODIFY -- new cell after cell 7) |
| Depends On | P0-C-2, P0-C-1 |
| Blocks | None |
| Effort | L |
| LOC Estimate | ~100 |

**What to do:** Add Cell 8 to the notebook after cell 7. Cell loads the deploy manifest, displays per-horizon artifact details (type, model, family, metrics, validation status), loads each ModelBundle, and runs a `predict_from_raw()` demo on the last N bars of raw_data. Falls back to listing bundles/ directory if no deploy/ exists. All operations wrapped in try/except for safety.

**Acceptance Criteria:**
- [ ] Cell 8 exists in notebook with title "INFERENCE DEMO"
- [ ] After successful factory run with deploy: shows manifest info + per-horizon artifact details
- [ ] After successful factory run without deploy: falls back to bundle listing from bundles/ directory
- [ ] `predict_from_raw()` demo runs for tabular models (may fail gracefully for neural until P0-B completes)
- [ ] No cell execution errors even when bundles/deploy/raw_data are missing
- [ ] Cell checks `result` exists and is successful before proceeding
- [ ] Works for both model and ensemble artifacts

**Verification Command:**
```bash
python -c "
import json
nb_path = 'notebooks/ml_factory_colab.ipynb'
with open(nb_path) as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 9, f'Expected >= 9 cells, got {len(cells)}'
cell8_src = ''.join(cells[8]['source'])
assert 'INFERENCE DEMO' in cell8_src, 'Cell 8 missing INFERENCE DEMO title'
print('OK')
"
```

**Rollback:** Remove cell 8 from the notebook JSON (delete the cell entry from the cells array).

---

### Task P0-D-2: Deploy Export Cell (Cell 9)

| Field | Value |
|-------|-------|
| Phase | P0-D |
| Files | `notebooks/ml_factory_colab.ipynb` (MODIFY -- new cell after cell 8) |
| Depends On | P0-C-2 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 35-40 |

**What to do:** Add Cell 9 that zips the `deploy/` directory (not full output), prints the archive path and size, and triggers auto-download in Colab via `google.colab.files.download()`. Falls back gracefully if no deploy directory exists.

**Acceptance Criteria:**
- [ ] Cell 9 exists in notebook with title "EXPORT DEPLOY ARTIFACT"
- [ ] Produces a `.zip` containing only the deploy/ directory contents (not full output)
- [ ] In Colab: triggers auto-download via `google.colab.files.download()`
- [ ] Locally: prints the archive path
- [ ] If no deploy/ exists: prints helpful message about `deploy_artifact=True`
- [ ] Prints archive size in MB

**Verification Command:**
```bash
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 10, f'Expected >= 10 cells, got {len(cells)}'
cell9_src = ''.join(cells[9]['source'])
assert 'EXPORT DEPLOY ARTIFACT' in cell9_src, 'Cell 9 missing title'
assert 'make_archive' in cell9_src or 'shutil' in cell9_src, 'Cell 9 missing archive logic'
print('OK')
"
```

**Rollback:** Remove cell 9 from the notebook JSON.

---

### Task P0-D-3: Validation Cell (Cell 10)

| Field | Value |
|-------|-------|
| Phase | P0-D |
| Files | `notebooks/ml_factory_colab.ipynb` (MODIFY -- new cell after cell 9) |
| Depends On | P0-C-2, P0-C-4 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 45-50 |

**What to do:** Add Cell 10 that reads `deploy/manifest.json` and per-horizon `validation.json` files. Displays per-check pass/fail with messages and an overall pass/fail summary. Handles missing validation files gracefully.

**Acceptance Criteria:**
- [ ] Cell 10 exists in notebook with title "VALIDATION REPORT"
- [ ] Displays per-horizon validation results from validation.json
- [ ] Shows per-check pass/fail with messages (truncated to 80 chars)
- [ ] Shows overall ALL PASSED / SOME CHECKS FAILED summary
- [ ] Handles missing validation files gracefully (prints "Validation file not found")
- [ ] No cell execution errors when deploy directory is missing

**Verification Command:**
```bash
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 11, f'Expected >= 11 cells, got {len(cells)}'
cell10_src = ''.join(cells[10]['source'])
assert 'VALIDATION REPORT' in cell10_src, 'Cell 10 missing title'
print('OK')
"
```

**Rollback:** Remove cell 10 from the notebook JSON.

---

### Task P0-D-4: Config Additions (Cell 2 + Cell 5 Modification)

| Field | Value |
|-------|-------|
| Phase | P0-D |
| Files | `notebooks/ml_factory_colab.ipynb` (MODIFY -- cells 2 and 5) |
| Depends On | P0-C-7 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 10-15 |

**What to do:** Add `BUNDLING_ENABLED = True` and `DEPLOY_ARTIFACT = True` config toggles to Cell 2. Update Cell 5 to import `BundlingSection` from `src.config.experiment` and pass `BundlingSection(create_bundle=BUNDLING_ENABLED, deploy_artifact=DEPLOY_ARTIFACT)` into the `ExperimentConfig` constructor.

**Acceptance Criteria:**
- [ ] Cell 2 contains `BUNDLING_ENABLED = True` and `DEPLOY_ARTIFACT = True` variables
- [ ] Cell 5 imports `BundlingSection` from `src.config.experiment`
- [ ] Cell 5 passes `bundling=BundlingSection(...)` to `ExperimentConfig`
- [ ] Default config produces deploy/ directory
- [ ] Setting `BUNDLING_ENABLED = False` skips both bundling and deploy
- [ ] Setting `DEPLOY_ARTIFACT = False` creates bundles but skips deploy
- [ ] All existing config toggles still work unchanged

**Verification Command:**
```bash
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
cell2_src = ''.join(cells[2]['source'])
assert 'BUNDLING_ENABLED' in cell2_src, 'Cell 2 missing BUNDLING_ENABLED'
assert 'DEPLOY_ARTIFACT' in cell2_src, 'Cell 2 missing DEPLOY_ARTIFACT'
print('OK')
"
```

**Rollback:** Remove `BUNDLING_ENABLED` and `DEPLOY_ARTIFACT` from Cell 2. Remove `BundlingSection` import and `bundling=` parameter from Cell 5.

---

### Task P0-D-5: Drive Persistence Cell (Cell 11)

| Field | Value |
|-------|-------|
| Phase | P0-D |
| Files | `notebooks/ml_factory_colab.ipynb` (MODIFY -- new cell after cell 10) |
| Depends On | P0-C-2, P0-D-1 |
| Blocks | None |
| Effort | S |
| LOC Estimate | 45-50 |

**What to do:** Add Cell 11 that saves the deploy/ directory to Google Drive (in Colab) or prints the local path (locally). Mounts Drive if not already mounted, copies deploy/ to `MyDrive/ml_factory_results/{EXPERIMENT_NAME}/deploy/`, and prints reload instructions using `load_deploy_artifact()`.

**Acceptance Criteria:**
- [ ] Cell 11 exists in notebook with title "SAVE DEPLOY ARTIFACT TO GOOGLE DRIVE"
- [ ] In Colab with Drive mounted: copies deploy/ to Drive, prints destination path and size
- [ ] In Colab without Drive: prints helpful mount instructions
- [ ] Locally: prints local deploy path
- [ ] Shows reload instructions with `load_deploy_artifact()` call
- [ ] Handles missing deploy directory gracefully

**Verification Command:**
```bash
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 12, f'Expected >= 12 cells, got {len(cells)}'
cell11_src = ''.join(cells[11]['source'])
assert 'GOOGLE DRIVE' in cell11_src, 'Cell 11 missing GOOGLE DRIVE title'
assert 'load_deploy_artifact' in cell11_src, 'Cell 11 missing reload instructions'
print('OK')
"
```

**Rollback:** Remove cell 11 from the notebook JSON.

---

# Summary Tables

---

## Workstream Table

| Phase | Tasks | Total LOC | New Files | Modified Files | Critical Path? |
|-------|-------|-----------|-----------|----------------|----------------|
| P0-A (Foundation) | 6 | 191-300 | 1 (`protocols.py`) | 3 (`types.py`, `bundle.py`, `builder.py`, `unified_orchestrator.py`) | YES -- unblocks P0-B |
| P0-B (Core Inference) | 6 | 220-320 | 1 (`errors.py`) | 2 (`bundle.py`, `ensemble_bundle.py`) | YES -- unblocks P0-C/D for full model coverage |
| P0-C (Deploy Packaging) | 7 | 420-550 | 1 (`deploy.py`) | 3 (`factory.py`, `__init__.py`, `experiment.py`) | YES -- unblocks P0-D |
| P0-D (Notebook Integration) | 5 | 235-255 | 0 | 1 (`ml_factory_colab.ipynb`) | NO -- terminal phase |
| **TOTAL** | **24** | **1066-1425** | **3** | **7 unique** | |

---

## Dependency Graph (ASCII)

```
                           P0-A: FOUNDATION
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  P0-A-1 ─────────────────────────────────────┐                  │
  │  (protocols.py)                               │                  │
  │                                               │                  │
  │  P0-A-2 ─────────(no downstream in P0)        │                  │
  │  (ScalingSource)                              │                  │
  │                                               ▼                  │
  │  P0-A-3 ─────────────────────────────────► P0-A-6               │
  │  (BundleMetadata)                          (protocol-aware       │
  │       │                                     BundleBuilder)       │
  │       │                                       ▲                  │
  │  P0-A-4 ─────────────────────────────────────┘                  │
  │  (calibrator fix)                                                │
  │                                                                  │
  │  P0-A-5 ──────────(blocks P0-B-1)                               │
  │  (double-scaling)                                                │
  └──────────┬───────────┬───────────────────────────────────────────┘
             │           │
             ▼           ▼
                           P0-B: CORE INFERENCE
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  P0-B-2 ─────────────────────────────────┐                      │
  │  (_build_3d)                              │                      │
  │       ▲ (needs A-3)                       │                      │
  │                                           ▼                      │
  │  P0-B-3 ─────────────────────────────► P0-B-1                   │
  │  (_build_4d)                           (_apply_adapter)          │
  │       ▲ (needs A-3)                       │                      │
  │                                           │ (needs A-5)          │
  │  P0-B-5 ──────────(independent)           │                      │
  │  (relative paths)                         ▼                      │
  │                                        P0-B-4                    │
  │  P0-B-6 ──────────(independent)        (EnsembleBundle           │
  │  (errors.py)                            .predict_from_raw)       │
  └──────────────────────────────────────────┬───────────────────────┘
                                             │
                                             ▼
                           P0-C: DEPLOY PACKAGING
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  P0-C-1 ──────────────────────┬──────────┬──────────┐           │
  │  (DeployManifest)             │          │          │           │
  │                               ▼          ▼          ▼           │
  │                           P0-C-3     P0-C-4     P0-C-5          │
  │                           (selector) (validator) (loader)       │
  │                               │          │                      │
  │                               ▼          ▼                      │
  │                           P0-C-2 ◄───────┘                      │
  │                           (factory Phase 5)                      │
  │                               │                                  │
  │                               ├──────► P0-C-7 (config toggle)   │
  │                               │                                  │
  │                           P0-C-6 (exports)                       │
  │                           (needs C-1,C-3,C-4,C-5)               │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
                           P0-D: NOTEBOOK INTEGRATION
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  P0-D-4 ──────── (needs C-7)                                    │
  │  (config cells)                                                  │
  │                                                                  │
  │  P0-D-1 ──────── (needs C-2, C-1)                               │
  │  (inference demo)                                                │
  │                                                                  │
  │  P0-D-2 ──────── (needs C-2)                                    │
  │  (export cell)                                                   │
  │                                                                  │
  │  P0-D-3 ──────── (needs C-2, C-4)                               │
  │  (validation cell)                                               │
  │                                                                  │
  │  P0-D-5 ──────── (needs C-2, D-1)                               │
  │  (Drive cell)                                                    │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Execution Batches

### Minimum Sequential Batches: 7

```
BATCH 1 [6 tasks, parallel, no dependencies]
  ├── P0-A-1  Create protocols.py                    (S, ~50 LOC)
  ├── P0-A-2  Add ScalingSource enum                 (S, ~12 LOC)
  ├── P0-A-3  Extend BundleMetadata                  (M, ~65 LOC)
  ├── P0-A-4  Fix calibrator transfer                (M, ~40 LOC)
  ├── P0-A-5  Fix double-scaling bug                 (S, ~3 LOC)
  └── P0-B-6  Create errors.py                       (S, ~30 LOC)
     Total: ~200 LOC

BATCH 2 [4 tasks, parallel, depends on Batch 1]
  ├── P0-A-6  Protocol-aware BundleBuilder           (M, ~75 LOC)  [needs A-1, A-3, A-4]
  ├── P0-B-2  _build_3d_input sliding window         (S, ~30 LOC)  [needs A-3]
  ├── P0-B-3  _build_4d_input MTF generation         (L, ~100 LOC) [needs A-3]
  └── P0-B-5  Fix relative paths in EnsembleBundle   (S, ~25 LOC)  [independent]
     Total: ~230 LOC

BATCH 3 [2 tasks, sequential dependency]
  ├── P0-B-1  _apply_adapter routing                 (M, ~50 LOC)  [needs A-3, A-5, B-2, B-3]
  └── P0-C-1  Create DeployManifest dataclasses      (M, ~115 LOC) [independent, can parallel]
     Total: ~165 LOC

BATCH 4 [2 tasks, depends on Batch 3]
  ├── P0-B-4  EnsembleBundle.predict_from_raw        (S, ~35 LOC)  [needs B-1]
  ├── P0-C-3  Artifact selector logic                (L, ~120 LOC) [needs C-1]
  ├── P0-C-4  Validation report generation           (M, ~90 LOC)  [needs C-1]
  └── P0-C-5  Deploy helper function                 (S, ~45 LOC)  [needs C-1]
     Total: ~290 LOC

BATCH 5 [3 tasks, depends on Batch 4]
  ├── P0-C-2  Factory Phase 5 integration            (L, ~90 LOC)  [needs C-1, C-3, C-4]
  ├── P0-C-6  __init__.py exports                    (S, ~12 LOC)  [needs C-1, C-3, C-4, C-5]
  └── P0-C-7  BundlingSection config toggle           (S, ~12 LOC)  [needs C-2]
     Total: ~114 LOC

BATCH 6 [1 task, depends on Batch 5]
  └── P0-D-4  Notebook config additions              (S, ~12 LOC)  [needs C-7]
     Total: ~12 LOC

BATCH 7 [4 tasks, parallel, depends on Batch 5-6]
  ├── P0-D-1  Inference demo cell                    (L, ~100 LOC) [needs C-2]
  ├── P0-D-2  Deploy export cell                     (S, ~38 LOC)  [needs C-2]
  ├── P0-D-3  Validation cell                        (S, ~48 LOC)  [needs C-2, C-4]
  └── P0-D-5  Drive persistence cell                 (S, ~48 LOC)  [needs C-2]
     Total: ~234 LOC
```

### Critical Path (longest sequential chain):

```
P0-A-3 (Batch 1) -> P0-B-2/B-3 (Batch 2) -> P0-B-1 (Batch 3) -> P0-B-4 (Batch 4)
                                              P0-C-1 (Batch 3) -> P0-C-3/C-4 (Batch 4) -> P0-C-2 (Batch 5) -> P0-C-7 (Batch 5) -> P0-D-4 (Batch 6) -> P0-D-1 (Batch 7)
```

**Critical path length: 7 sequential batches.**

---

## Gap Coverage Matrix

| Task | Gaps Closed | Gap IDs |
|------|-------------|---------|
| P0-A-1 | Protocols file, trainer protocol, inference bundle protocol | G10 |
| P0-A-2 | ScalingSource enum | G11 |
| P0-A-3 | BundleMetadata fields + version bump | G14, G20 |
| P0-A-4 | Calibrator transfer through orchestrator to bundle | G3, G21 (partial) |
| P0-A-5 | Double-scaling bug fix | G2 |
| P0-A-6 | Protocol-aware bundle extraction, feature spec pass-through | G17 (partial) |
| P0-B-1 | Adapter routing for all 12 models in predict_from_raw | G1 |
| P0-B-2 | 3D sliding window for 7 sequence models | G16, G25 (partial - nbeats windowing) |
| P0-B-3 | 4D multi-timeframe preprocessing for transformers | G15 |
| P0-B-4 | EnsembleBundle.predict_from_raw | G4, G19 (partial) |
| P0-B-5 | Relative base bundle paths for portability | G9 |
| P0-B-6 | Domain-specific inference error type | G22 |
| P0-C-1 | Deploy manifest dataclass | G5 (partial) |
| P0-C-2 | Deploy directory structure + factory integration | G5, G7 |
| P0-C-3 | Per-horizon artifact selection | G5 (partial), G8 (partial - single ensemble for horizons[0]) |
| P0-C-4 | Validation/smoke test reports | G23 |
| P0-C-5 | User-facing deploy loader | G5 (partial) |
| P0-C-6 | Public API exports | (infrastructure) |
| P0-C-7 | Config toggle for deploy | (infrastructure) |
| P0-D-1 | Notebook inference demo | G13 (partial) |
| P0-D-2 | Notebook deploy export | G13 (partial) |
| P0-D-3 | Notebook validation display | G13 (partial) |
| P0-D-4 | Notebook config additions | G13 (partial) |
| P0-D-5 | Notebook Drive persistence | G13 (partial) |

### Gap Coverage Summary

| Gap | Severity | Covered By | Status |
|-----|----------|------------|--------|
| G1 | CRITICAL | P0-B-1 | Fully covered |
| G2 | CRITICAL | P0-A-5 | Fully covered |
| G3 | CRITICAL | P0-A-4 | Fully covered |
| G4 | CRITICAL | P0-B-4 | Fully covered |
| G5 | CRITICAL | P0-C-1, C-2, C-3, C-5 | Fully covered |
| G6 | HIGH | Not in P0 scope | Deferred (requires rewrite of build_ensemble_bundle) |
| G7 | HIGH | P0-C-2 | Partially covered (factory calls deploy, not build_all) |
| G8 | HIGH | P0-C-3 | Partially covered (ensemble covers horizons[0] only; documented limitation) |
| G9 | HIGH | P0-B-5 | Fully covered |
| G10 | HIGH | P0-A-1 | Fully covered |
| G11 | HIGH | P0-A-2 | Fully covered |
| G12 | HIGH | Not in P0 scope | Deferred (UIP is Phase 3B-2) |
| G13 | HIGH | P0-D-1 through P0-D-5 | Fully covered |
| G14 | HIGH | P0-A-3 | Fully covered |
| G15 | HIGH | P0-B-3 | Fully covered |
| G16 | HIGH | P0-B-2 | Fully covered |
| G17 | MEDIUM | P0-A-6 | Partially covered (contract-based metadata, not full feature_specs) |
| G18 | MEDIUM | Not in P0 scope | Deferred (preprocessing config from training) |
| G19 | MEDIUM | P0-B-4 | Partially covered (predict_from_raw routes per base model) |
| G20 | MEDIUM | P0-A-3 | Fully covered |
| G21 | MEDIUM | P0-A-4 | Partially covered (calibrator field added, dual class still exists) |
| G22 | MEDIUM | P0-B-6 | Fully covered |
| G23 | MEDIUM | P0-C-4 | Fully covered |
| G24 | MEDIUM | Not in P0 scope | Deferred |
| G25 | MEDIUM | P0-B-2 | Partially covered (windowing works; raw feature mode deferred) |
| G26 | LOW | Not in P0 scope | Deferred (enum relocation) |
| G27 | LOW | Not in P0 scope | Deferred (dead import cleanup) |
| G28 | LOW | Not in P0 scope | Deferred (tar deprecation) |
| G29 | LOW | Not in P0 scope | Deferred (pickle safety) |

### Coverage Statistics

- **CRITICAL gaps (5):** 5/5 fully covered (100%)
- **HIGH gaps (10):** 7/10 fully covered, 1 partial, 2 deferred (80% addressed)
- **MEDIUM gaps (9):** 4/9 fully covered, 3 partial, 2 deferred (78% addressed)
- **LOW gaps (4):** 0/4 covered (deferred to post-P0 phases)
- **Overall:** 16/29 fully covered, 4 partially covered, 9 deferred (69% fully, 83% addressed)

---

## Deferred Gaps (Not in P0 Scope)

| Gap | Why Deferred | Recommended Phase |
|-----|-------------|-------------------|
| G6 (build_ensemble_bundle format) | Requires rewriting BundleBuilder.build_ensemble_bundle() to use EnsembleBundle.from_ensemble_result() | P1 |
| G12 (UniversalInferencePipeline) | Large new component, not blocking P0 deploy objective | Phase 3B-2 |
| G18 (preprocessing config from training) | Quality improvement, not blocking | P1 |
| G24 (meta-learner loading robustness) | Edge case, not blocking | P1 |
| G25 (nbeats RAW feature mode) | Requires dedicated feature path; windowing alone enables basic inference | P1 |
| G26-G29 (cleanup items) | Low severity, no functional impact | P1 or later |

---

*This document is a planning artifact. No code has been modified.*
