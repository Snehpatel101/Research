# 09 - Validation Plan, Risk Register, and Rollout Strategy

**Date:** 2026-02-15
**Agent:** 9/10 (Validation & Risk Analyst)
**Purpose:** Comprehensive validation plan with smoke tests, integration tests, and risk mitigations for the P0-A through P0-D deployable artifact implementation.

---

## References

- Reports 01-08 in `.audit/deploy-plan/`
- `CLAUDE.md` verification commands and linting requirements
- Gap table from `05-gap-analysis.md` (29 gaps, G1-G29)
- Task breakdown from `08-task-breakdown.md` (24 tasks, 7 execution batches)

---

# Part 1: Validation Plan

---

## 1.1 Smoke Tests (Per Phase)

### P0-A Smoke Tests (Foundation)

```bash
# P0-A-1: protocols.py exists and imports
python -c "from src.core.protocols import TrainerProtocol, InferenceBundle; print('P0-A-1 OK')"
grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l  # expect: 1
grep -r "class InferenceBundle" src/ --include="*.py" | wc -l  # expect: 1

# P0-A-2: ScalingSource enum
python -c "from src.core.types import ScalingSource; assert ScalingSource.BUNDLE.value == 'bundle'; print('P0-A-2 OK')"
grep -r "class ScalingSource" src/ --include="*.py" | wc -l  # expect: 1

# P0-A-3: BundleMetadata extended + version bump
python -c "
from src.inference.bundle import BundleMetadata, BUNDLE_VERSION
assert BUNDLE_VERSION == '1.3.0', f'Version mismatch: {BUNDLE_VERSION}'
m = BundleMetadata(version='1.3.0', created_at='now', model_name='test', model_family='boosting', n_features=10, input_rank=2)
d = m.to_dict()
assert 'primary_timeframe' in d, 'Missing primary_timeframe'
assert 'mtf_timeframes' in d, 'Missing mtf_timeframes'
assert 'scaling_source' in d, 'Missing scaling_source'
assert 'scaler_type' in d, 'Missing scaler_type'
m2 = BundleMetadata.from_dict(d)
assert m2.primary_timeframe == m.primary_timeframe
print('P0-A-3 OK')
"

# P0-A-3 backward compat: v1.2.0 metadata still loads
python -c "
from src.inference.bundle import BundleMetadata
old_meta = {'version': '1.2.0', 'created_at': 'old', 'model_name': 'xgb', 'model_family': 'boosting', 'n_features': 50, 'input_rank': 2}
m = BundleMetadata.from_dict(old_meta)
assert m.primary_timeframe == '5min', 'Default primary_timeframe wrong'
assert m.scaling_source == 'bundle', 'Default scaling_source wrong'
print('P0-A-3 backward compat OK')
"

# P0-A-4: Calibrator field on ModelTrainingResult
python -c "
from src.models.training.unified_orchestrator import ModelTrainingResult
r = ModelTrainingResult(model_name='test', horizon=20)
assert r.calibrator is None, 'Default calibrator should be None'
assert hasattr(r, 'calibration_metrics'), 'Missing calibration_metrics field'
print('P0-A-4 OK')
"

# P0-A-5: Double-scaling fix
grep -n "skip_scaling=True" src/inference/bundle.py | head -5
# expect: at least one match in the preprocess() method

# P0-A-6: BundleBuilder imports protocol
python -c "from src.inference.builder import BundleBuilder; print('P0-A-6 OK')"

# P0-A linting (all modified files)
ruff check src/core/protocols.py src/core/types.py src/inference/bundle.py src/inference/builder.py src/models/training/unified_orchestrator.py
```

### P0-B Smoke Tests (Core Inference)

```bash
# P0-B-1: _apply_adapter method exists
python -c "
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_apply_adapter'), 'Missing _apply_adapter'
print('P0-B-1 OK')
"

# P0-B-2: _build_3d_input method exists + sliding window logic
python -c "
import numpy as np
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_build_3d_input'), 'Missing _build_3d_input'

# Verify sliding window math independently
seq_len = 60
values = np.random.randn(120, 50).astype(np.float32)
windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
result = windows.transpose(0, 2, 1).copy()
assert result.shape == (61, 60, 50), f'Bad shape: {result.shape}'
print('P0-B-2 OK')
"

# P0-B-3: _build_4d_input method exists + resampling logic
python -c "
import numpy as np
import pandas as pd
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_build_4d_input'), 'Missing _build_4d_input'

# Verify resampling logic independently
dates = pd.date_range('2026-01-01', periods=200, freq='1min')
df = pd.DataFrame({
    'open': np.random.randn(200), 'high': np.random.randn(200),
    'low': np.random.randn(200), 'close': np.random.randn(200),
    'volume': np.abs(np.random.randn(200)) * 1000,
}, index=dates)
r5 = df.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
assert len(r5) > 0, 'Resampling produced no rows'
print('P0-B-3 OK')
"

# P0-B-4: EnsembleBundle.predict_from_raw exists
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
import inspect
assert hasattr(EnsembleBundle, 'predict_from_raw'), 'Missing predict_from_raw'
sig = inspect.signature(EnsembleBundle.predict_from_raw)
assert 'raw_df' in sig.parameters
assert 'calibrate' in sig.parameters
print('P0-B-4 OK')
"

# P0-B-5: Relative path logic
python -c "
from pathlib import Path
p = Path('/a/b/c/ensemble')
base = Path('/a/b/c/xgboost_h20')
rel = base.relative_to(p.parent)
resolved = (p.parent / rel).resolve()
assert resolved == base.resolve()
print('P0-B-5 OK')
"

# P0-B-6: errors.py
python -c "
from src.inference.errors import InferenceShapeMismatchError, InferenceError
e = InferenceShapeMismatchError(expected_shape=(1,60,50), actual_shape=(100,50), model_name='lstm', hint='Use windowing')
assert 'lstm' in str(e)
assert '(1, 60, 50)' in str(e)
assert issubclass(InferenceShapeMismatchError, InferenceError)
print('P0-B-6 OK')
"

# P0-B linting
ruff check src/inference/bundle.py src/inference/ensemble_bundle.py src/inference/errors.py
```

### P0-C Smoke Tests (Deploy Packaging)

```bash
# P0-C-1: DeployManifest round-trip
python -c "
from src.inference.deploy import DeployManifest, HorizonArtifactEntry
import tempfile, json
from pathlib import Path

entry = HorizonArtifactEntry(horizon=20, artifact_type='model', model_key='xgboost_h20', model_family='boosting', bundle_path='h20/artifact')
m = DeployManifest(run_id='test_run', created_at='2026-01-01T00:00:00', horizons=[20], selected_artifacts={'h20': entry})
with tempfile.TemporaryDirectory() as d:
    m.save(Path(d))
    with open(Path(d) / 'manifest.json') as f:
        raw = json.load(f)
    assert 'run_id' in raw
    assert raw['selected_artifacts']['h20']['model_key'] == 'xgboost_h20'
    m2 = DeployManifest.load(Path(d))
    assert m2.run_id == 'test_run'
    assert m2.selected_artifacts['h20'].horizon == 20
print('P0-C-1 OK')
"

# P0-C-2: ExperimentResult has deploy_path
python -c "
from src.factory import ExperimentResult
r = ExperimentResult(success=True)
assert hasattr(r, 'deploy_path'), 'Missing deploy_path'
assert r.deploy_path is None, 'Default should be None'
print('P0-C-2 OK')
"

# P0-C-3: Artifact selector importable
python -c "from src.inference.deploy import select_deploy_artifact; print('P0-C-3 OK')"

# P0-C-4: Validation report importable
python -c "from src.inference.deploy import validate_deploy_artifact; print('P0-C-4 OK')"

# P0-C-5: Deploy loader importable
python -c "from src.inference.deploy import load_deploy_artifact; print('P0-C-5 OK')"

# P0-C-6: Exports from __init__.py
python -c "
from src.inference import DeployManifest, HorizonArtifactEntry, load_deploy_artifact, select_deploy_artifact, validate_deploy_artifact
print('P0-C-6 OK')
"

# P0-C-7: BundlingSection toggle
python -c "
from src.config.experiment import BundlingSection
b = BundlingSection()
assert b.deploy_artifact is True, 'Default should be True'
print('P0-C-7 OK')
"

# P0-C linting
ruff check src/inference/deploy.py src/factory.py src/inference/__init__.py src/config/experiment.py
```

### P0-D Smoke Tests (Notebook Integration)

```bash
# P0-D-1: Cell 8 exists with INFERENCE DEMO
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 9, f'Expected >= 9 cells, got {len(cells)}'
cell8_src = ''.join(cells[8]['source'])
assert 'INFERENCE DEMO' in cell8_src, 'Cell 8 missing INFERENCE DEMO'
assert 'predict_from_raw' in cell8_src, 'Cell 8 missing predict_from_raw'
print('P0-D-1 OK')
"

# P0-D-2: Cell 9 exists with EXPORT
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 10, f'Expected >= 10 cells, got {len(cells)}'
cell9_src = ''.join(cells[9]['source'])
assert 'EXPORT DEPLOY ARTIFACT' in cell9_src, 'Cell 9 missing title'
assert 'make_archive' in cell9_src or 'shutil' in cell9_src
print('P0-D-2 OK')
"

# P0-D-3: Cell 10 exists with VALIDATION REPORT
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 11, f'Expected >= 11 cells, got {len(cells)}'
cell10_src = ''.join(cells[10]['source'])
assert 'VALIDATION REPORT' in cell10_src, 'Cell 10 missing title'
print('P0-D-3 OK')
"

# P0-D-4: Config additions in Cell 2
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cell2_src = ''.join(nb['cells'][2]['source'])
assert 'BUNDLING_ENABLED' in cell2_src, 'Cell 2 missing BUNDLING_ENABLED'
assert 'DEPLOY_ARTIFACT' in cell2_src, 'Cell 2 missing DEPLOY_ARTIFACT'
print('P0-D-4 OK')
"

# P0-D-5: Cell 11 Drive persistence
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
assert len(cells) >= 12, f'Expected >= 12 cells, got {len(cells)}'
cell11_src = ''.join(cells[11]['source'])
assert 'GOOGLE DRIVE' in cell11_src, 'Cell 11 missing GOOGLE DRIVE'
assert 'load_deploy_artifact' in cell11_src, 'Cell 11 missing reload instructions'
print('P0-D-5 OK')
"
```

### Full Post-Implementation Sweep

```bash
# Master linting check (must pass with 0 errors)
ruff check src/
black --check src/

# Import verification (from CLAUDE.md)
python -c "from src.core.types import DataRank, ModelFamily; print('core types OK')"
python -c "from src.core.contracts import get_model_contract; print('contracts OK')"
python -c "from src.data.adapters import get_adapter; print('adapters OK')"

# Single definition checks
grep -r "class DataRank" src/ --include="*.py" | wc -l   # expect: 1
grep -r "class ModelFamily" src/ --include="*.py" | wc -l # expect: 1
grep -r "class ScalingSource" src/ --include="*.py" | wc -l # expect: 1
grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l # expect: 1

# Dead import checks
grep -r "from src\.coordination" src/ --include="*.py" | wc -l  # expect: 0
grep -r "from src\.feature_selection" src/ --include="*.py" | wc -l  # expect: 0
```

---

## 1.2 Integration Tests

### IT-1: Tabular Roundtrip (XGBoost)

**Objective:** Verify the full predict_from_raw path for a 2D tabular model.

**Setup:**
- Train XGBoost on sample data with `auto_calibrate=True`
- Bundle via BundleBuilder
- Load the bundle via `ModelBundle.load()`

**Steps:**
1. Call `bundle.predict_from_raw(raw_ohlcv_df)` with 200 rows of OHLCV data
2. Verify return type is `PredictionResult`
3. Verify `result.predictions` is a 1D array with length > 0
4. Verify `result.class_probabilities` is not None (calibrator was enabled)
5. Verify no scaling was applied twice (compare feature values at preprocess() output with manual single-scaled values)

**Pass Criteria:**
- `predict_from_raw()` returns `PredictionResult` without error
- Predictions shape matches expected output dimensions
- `bundle.metadata.scaling_source == "bundle"`
- `bundle.metadata.has_calibrator == True`

---

### IT-2: Sequence Roundtrip (LSTM)

**Objective:** Verify predict_from_raw for a 3D sequence model including sliding window construction.

**Setup:**
- Train LSTM with `sequence_length=60` on sample data
- Bundle and load

**Steps:**
1. Call `bundle.predict_from_raw(raw_ohlcv_df)` with 200 rows of raw OHLCV
2. Verify `_build_3d_input()` was used (metadata.requires_sequences == True)
3. Verify intermediate 3D tensor shape is `(n, 60, n_features)` where n = 200 - 60 + 1 = 141 (approximately, depending on feature engineering output rows)
4. Verify `PredictionResult` returned
5. Test edge case: provide exactly `sequence_length` rows -- should return shape `(1, ...)`
6. Test error case: provide fewer than `sequence_length` rows -- should raise `ValueError`

**Pass Criteria:**
- Successful `PredictionResult` for sufficient data
- `ValueError` with descriptive message for insufficient data
- Output prediction count matches expected window count

---

### IT-3: Transformer Roundtrip (PatchTST)

**Objective:** Verify predict_from_raw for a 4D multi-timeframe model, bypassing PreprocessingGraph.

**Setup:**
- Train PatchTST with `primary_timeframe="1min"`, `mtf_timeframes=["5min", "15min"]`, `sequence_length=60`
- Bundle and load

**Steps:**
1. Call `bundle.predict_from_raw(raw_1min_ohlcv_df)` with 500 rows of 1-minute OHLCV data (DatetimeIndex required)
2. Verify `_build_4d_input()` was used (metadata.requires_4d == True)
3. Verify intermediate 4D tensor shape is `(n, 3, 60, 5)` -- 3 timeframes, 60 sequence length, 5 OHLCV features
4. Verify resampled 5min bars use correct OHLCV aggregation
5. Verify `PredictionResult` returned
6. Test error case: DataFrame without DatetimeIndex -- should raise `ValueError`
7. Test error case: DataFrame missing OHLCV columns -- should raise `ValueError`

**Pass Criteria:**
- Successful `PredictionResult` with 4D-routed data
- Correct OHLCV aggregation (open=first, high=max, low=min, close=last, volume=sum)
- Proper error messages for invalid inputs

---

### IT-4: Ensemble Roundtrip

**Objective:** Verify `EnsembleBundle.predict_from_raw()` orchestrates base model predictions through the meta-learner.

**Setup:**
- Train ensemble with at least 2 base models (e.g., XGBoost + LSTM)
- Build ensemble bundle with relative base paths
- Load the ensemble bundle

**Steps:**
1. Call `ensemble_bundle.predict_from_raw(raw_ohlcv_df)` with 200 rows
2. Verify each base bundle's `predict_from_raw()` is called with `calibrate=False`
3. Verify meta-learner receives stacked base predictions
4. Verify final `PredictionResult` returned
5. Verify base bundle paths in `base_bundles.json` are relative (not absolute)
6. Test error case: load without meta-learner -- should raise `ValueError`
7. Test error case: load without base bundles -- should raise `ValueError`

**Pass Criteria:**
- Ensemble produces valid predictions from raw OHLCV
- Heterogeneous base models (2D + 3D) handled correctly
- Base paths are portable (relative)

---

### IT-5: Deploy Roundtrip

**Objective:** Verify the full factory-to-deploy-to-inference pipeline.

**Setup:**
- Run `Factory.run()` with `create_bundle=True`, `deploy_artifact=True`

**Steps:**
1. Verify `result.deploy_path` is not None and points to a valid directory
2. Verify `deploy/manifest.json` exists and is valid JSON
3. Verify `deploy/h{horizon}/artifact/` exists for each configured horizon
4. Verify `deploy/h{horizon}/validation.json` exists with `passed: true`
5. Call `load_deploy_artifact(result.deploy_path, horizon=h)` -- returns ModelBundle or EnsembleBundle
6. Call `artifact.predict_from_raw(raw_ohlcv_df)` -- returns `PredictionResult`
7. Verify manifest `selected_artifacts` has correct model_key and metrics snapshot

**Pass Criteria:**
- Deploy directory structure matches specification
- `load_deploy_artifact()` returns working bundle
- `predict_from_raw()` produces valid predictions on the loaded artifact
- Manifest is pure JSON (loadable without src/ imports)

---

### IT-6: Old Bundle Compatibility

**Objective:** Verify that existing v1.2.0 bundles continue to load and work after the code changes.

**Setup:**
- Locate or create a bundle saved with BUNDLE_VERSION = "1.2.0" (before P0-A-3 changes)

**Steps:**
1. Load the v1.2.0 bundle via `ModelBundle.load(old_bundle_path)`
2. Verify `BundleMetadata.from_dict()` populates missing new fields with safe defaults:
   - `primary_timeframe` defaults to `"5min"`
   - `mtf_timeframes` defaults to `[]`
   - `scaling_source` defaults to `"bundle"`
   - `scaler_type` defaults to `"robust"`
3. Call `bundle.predict(pre_shaped_X)` with pre-shaped 2D input -- verify it still works
4. Verify `bundle.metadata.version` reads the original `"1.2.0"` (not overwritten)

**Pass Criteria:**
- Old bundles load without any error
- Default values are sensible and do not break inference
- `predict(X_preshaped)` works identically to before

---

## 1.3 Notebook E2E Tests

### Notebook Execution Checklist

Execute the notebook cells sequentially in a fresh Colab runtime. Verify each step:

| Step | Cell | What to Verify |
|------|------|----------------|
| 1 | Cell 0 (markdown) | Title renders correctly |
| 2 | Cell 1 (setup) | Dependencies install without error; `IN_COLAB` detected correctly |
| 3 | Cell 2 (config) | `BUNDLING_ENABLED = True` and `DEPLOY_ARTIFACT = True` variables present; no NameError |
| 4 | Cell 3 (validation) | Data validation passes |
| 5 | Cell 4 (data load) | Data loads; `raw_data` variable exists with > 200 rows |
| 6 | Cell 5 (training) | Training completes; `result.success == True`; `result.deploy_path` is not None |
| 7 | Cell 6 (results) | Metrics display; no errors |
| 8 | Cell 7 (save) | Original save/download logic works |
| 9 | Cell 8 (inference demo) | Manifest loaded; per-horizon details displayed; `predict_from_raw()` demo runs for at least the boosting model |
| 10 | Cell 9 (export) | `.zip` file created; size displayed; download triggered (Colab) or path printed (local) |
| 11 | Cell 10 (validation) | Per-horizon validation results displayed; overall PASS/FAIL shown |
| 12 | Cell 11 (Drive) | Drive persistence works (Colab) or local path printed |

### Error Scenarios to Test in Notebook

1. **No raw_data variable:** Cell 8 should skip predict demo gracefully
2. **BUNDLING_ENABLED = False:** Cells 8-11 should print "No bundles/deploy directory found"
3. **DEPLOY_ARTIFACT = False:** Cell 8 should fall back to bundle listing; Cells 9-11 should indicate no deploy directory
4. **Training fails (result.success = False):** All new cells should print "No successful result" and exit cleanly
5. **Insufficient raw data (< sequence_length rows):** Cell 8 predict demo should catch ValueError and print message

---

## 1.4 Artifact Load/Predict Checks

### Per-Model predict_from_raw Verification

For each of the 12 core model types, verify `predict_from_raw()` works after the full P0 implementation:

| # | Model | Adapter Path | Min Raw Rows | Expected Input Shape | Verification Command |
|---|-------|-------------|-------------|---------------------|---------------------|
| 1 | xgboost | 2D passthrough | 50 | `(n, features)` | `bundle.predict_from_raw(raw_df_200)` |
| 2 | lightgbm | 2D passthrough | 50 | `(n, features)` | `bundle.predict_from_raw(raw_df_200)` |
| 3 | catboost | 2D passthrough | 50 | `(n, features)` | `bundle.predict_from_raw(raw_df_200)` |
| 4 | lstm | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 5 | gru | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 6 | tcn | `_build_3d_input` | 240 | `(n, 120, feat)` | `bundle.predict_from_raw(raw_df_300)` |
| 7 | inceptiontime | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 8 | resnet1d | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 9 | tft | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 10 | nbeats | `_build_3d_input` | 120 | `(n, 60, feat)` | `bundle.predict_from_raw(raw_df_200)` |
| 11 | patchtst | `_build_4d_input` | 500 (1min) | `(n, 3, 60, 5)` | `bundle.predict_from_raw(raw_1min_df_500)` |
| 12 | itransformer | `_build_4d_input` | 500 (1min) | `(n, 3, 60, 5)` | `bundle.predict_from_raw(raw_1min_df_500)` |

### Ensemble predict_from_raw Verification

| # | Ensemble Type | Base Models | Verification |
|---|--------------|-------------|--------------|
| 1 | Homogeneous (all boosting) | xgboost + lightgbm + catboost | `ensemble.predict_from_raw(raw_df_200)` |
| 2 | Heterogeneous (mixed) | xgboost + lstm | `ensemble.predict_from_raw(raw_df_200)` |
| 3 | Full ensemble | All 12 models | `ensemble.predict_from_raw(raw_1min_df_500)` |

### Deploy Manifest Load Verification

```bash
# Verify manifest loads with pure JSON (no src/ imports)
python -c "
import json
from pathlib import Path

deploy_dir = Path('experiments/latest/deploy')  # adjust path
if deploy_dir.exists():
    with open(deploy_dir / 'manifest.json') as f:
        m = json.load(f)
    print(f'Run: {m[\"run_id\"]}')
    print(f'Horizons: {m[\"horizons\"]}')
    for k, v in m['selected_artifacts'].items():
        print(f'  {k}: {v[\"artifact_type\"]} ({v[\"model_key\"]})')
    print('Manifest loads with pure JSON: OK')
else:
    print('No deploy directory found (expected if not yet run)')
"
```

---

## 1.5 ONNX Fallback Validation

### What ONNX Support Means

ONNX support is **optional** and **never blocks native inference**. The deploy manifest includes a `runtime_profile` field (default: `"native"`) that indicates the inference runtime. ONNX is a future enhancement, not part of P0.

### Current State

- No ONNX export is implemented in P0
- `runtime_profile` defaults to `"native"` in `DeployManifest`
- The `InferenceBundle` protocol does not require ONNX methods

### When ONNX Is Available (Future)

To verify ONNX export works when eventually implemented:

```bash
# Future verification (not part of P0)
python -c "
# This will fail until ONNX support is added -- expected
try:
    from src.inference.onnx_export import export_to_onnx
    bundle = ModelBundle.load('path/to/bundle')
    onnx_path = export_to_onnx(bundle, 'output.onnx')
    print(f'ONNX exported: {onnx_path}')
except ImportError:
    print('ONNX export not yet implemented (expected in P0)')
"
```

### Validation Rule

- ONNX export should NEVER be a prerequisite for `predict_from_raw()` to work
- If ONNX export fails, native inference must remain fully functional
- The `runtime_profile` field in the manifest is informational only

---

## 1.6 Validation Matrix

| Test ID | What It Verifies | Phase Dep | Model Types | Pass Criteria |
|---------|-----------------|-----------|-------------|---------------|
| SM-A1 | protocols.py imports | P0-A-1 | N/A | Import succeeds; exactly 1 definition each |
| SM-A2 | ScalingSource enum | P0-A-2 | N/A | Import succeeds; `.BUNDLE.value == "bundle"` |
| SM-A3 | BundleMetadata v1.3.0 | P0-A-3 | All | Version == "1.3.0"; round-trip serialization; v1.2.0 backward compat |
| SM-A4 | Calibrator transfer | P0-A-4 | All | `ModelTrainingResult.calibrator` field exists; defaults to None |
| SM-A5 | Double-scaling fix | P0-A-5 | All | `skip_scaling=True` in preprocess() |
| SM-A6 | Protocol-aware builder | P0-A-6 | All | BundleBuilder imports OK; protocol check then duck-typing fallback |
| SM-B1 | Adapter routing | P0-B-1 | All 12 | `_apply_adapter()` method exists; routes by metadata flags |
| SM-B2 | 3D sliding window | P0-B-2 | 7 neural | Correct shapes: (1,60,50), (61,60,50); ValueError on short input |
| SM-B3 | 4D MTF generation | P0-B-3 | patchtst, itransformer | Correct shapes; OHLCV aggregation; DatetimeIndex required |
| SM-B4 | Ensemble predict_from_raw | P0-B-4 | Ensemble | Method exists; base models called with calibrate=False |
| SM-B5 | Relative paths | P0-B-5 | Ensemble | New saves use relative paths; old absolute paths still load |
| SM-B6 | Inference errors | P0-B-6 | All | Import succeeds; error message includes all diagnostic fields |
| SM-C1 | DeployManifest | P0-C-1 | N/A | Round-trip save/load; pure JSON; all fields serialized |
| SM-C2 | Factory Phase 5 | P0-C-2 | N/A | deploy_path on ExperimentResult; deploy/ directory created |
| SM-C3 | Artifact selector | P0-C-3 | N/A | Correct selection logic; ensemble beats single when score >= |
| SM-C4 | Validation report | P0-C-4 | All | 5 checks; passed/failed status; timing |
| SM-C5 | Deploy loader | P0-C-5 | All | Loads correct bundle type; auto-selects single horizon |
| SM-C6 | __init__ exports | P0-C-6 | N/A | All 5 names importable from src.inference |
| SM-C7 | Config toggle | P0-C-7 | N/A | deploy_artifact defaults True; False skips deploy |
| SM-D1 | Inference demo cell | P0-D-1 | All | Cell 8 in notebook; shows manifest; predict demo runs |
| SM-D2 | Export cell | P0-D-2 | N/A | Cell 9; zip created; Colab download works |
| SM-D3 | Validation cell | P0-D-3 | N/A | Cell 10; per-check results displayed |
| SM-D4 | Config additions | P0-D-4 | N/A | BUNDLING_ENABLED + DEPLOY_ARTIFACT in Cell 2 |
| SM-D5 | Drive persistence | P0-D-5 | N/A | Cell 11; Drive save or local path |
| IT-1 | Tabular roundtrip | P0-A+B | xgboost | predict_from_raw -> PredictionResult; single scaling |
| IT-2 | Sequence roundtrip | P0-A+B | lstm | predict_from_raw -> 3D adapter -> PredictionResult |
| IT-3 | Transformer roundtrip | P0-A+B | patchtst | predict_from_raw -> 4D adapter -> PredictionResult |
| IT-4 | Ensemble roundtrip | P0-A+B | ensemble | predict_from_raw -> base bundles -> meta-learner -> PredictionResult |
| IT-5 | Deploy roundtrip | P0-A+B+C | all | Factory.run() -> deploy/ -> load_deploy_artifact() -> predict_from_raw() |
| IT-6 | Old bundle compat | P0-A | all | v1.2.0 bundle loads; predict(X_preshaped) works |

---

# Part 2: Risk Register

---

## Risk Matrix

| Risk ID | Description | Likelihood | Impact | Severity | Mitigation | Owner Phase |
|---------|------------|------------|--------|----------|------------|-------------|
| R1 | Double-scaling regression: preprocess() `skip_scaling` reverts to False in future edits, causing features to be scaled twice | Medium | High | **HIGH** | Add comment explaining rationale; add `assert skip_scaling is True` style check in integration test; ScalingSource enum documents single-source-of-truth pattern | P0-A-5 |
| R2 | 4D model MTF data availability at inference: user provides 5min data instead of 1min, or insufficient bars for resampling to higher TFs | High | Medium | **HIGH** | `_build_4d_input()` validates DatetimeIndex, checks OHLCV columns, raises descriptive ValueError with minimum bar counts; metadata stores `primary_timeframe` so user knows what resolution is needed | P0-B-3 |
| R3 | EnsembleBundle backward compat: paths stored as raw strings (relative by default); old bundles with absolute paths break when moved to new machine or directory | Medium | High | **HIGH** | `load()` uses `Path.is_absolute()` check -- absolute paths used as-is, relative paths resolved against ensemble parent dir; both old and new formats work; save() uses `try/except ValueError` for cross-root edge cases | P0-B-5 |
| R4 | Calibrator None propagation: calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path, so bundles have `calibrator=None` even when calibration was enabled | High (currently always happens) | Medium | **HIGH** | Add explicit `calibrator` field to orchestrator's `ModelTrainingResult`; copy via `getattr(result, "calibrator", None)` in both `_train_single_model()` and `_train_boosting_parallel()`; BundleBuilder checks `model_result.calibrator` before duck-typing fallback | P0-A-4 |
| R5 | Notebook fails in Colab: torch version mismatch, CUDA OOM on 4D models, dependency conflicts, or memory pressure from large deploy zips | Medium | Medium | **MEDIUM** | Notebook cells wrapped in try/except; predict demo uses last N bars (not full dataset); zip only deploy/ (not full output); Drive persistence as alternative to download; Cell 8 gracefully skips models that fail | P0-D |
| R6 | Feature column drift between training and inference: feature engineering produces different column order or count at inference time vs training time | Medium | High | **HIGH** | BundleMetadata stores `feature_names` list; `_apply_adapter()` uses metadata column list to reorder if needed; PreprocessingGraph serializes the feature pipeline; `validate()` checks feature count matches | P0-A-3, P0-B-1 |
| R7 | Deploy selector picks wrong artifact: ensemble score appears higher due to metric calculation difference, or metric key mismatch between training and selection | Low | Medium | **MEDIUM** | Selection uses exact same metric key (`val_f1`) from `result.metrics`; fallback to best single model if ensemble score comparison fails; selection decision logged with metric values; manifest records the metric snapshot for auditability | P0-C-3 |
| R8 | PreprocessingGraph config hardcoding: `source_timeframe="1min"`, `target_timeframe="5min"`, `scaler_type="robust"` in BundleBuilder may not match actual training config | Medium | Medium | **MEDIUM** | P0-A-3 adds `scaler_type` and `primary_timeframe` to BundleMetadata from contract; P0-A-6 populates these from `get_model_contract()` at bundle creation time; full PreprocessingGraph config derivation deferred to P1 (G18) | P0-A-3, P0-A-6 |
| R9 | Sliding window off-by-one: `_build_3d_input()` produces windows that are misaligned with what the model was trained on (e.g., inclusive vs exclusive boundary) | Low | High | **MEDIUM** | Implementation mirrors `SequenceAdapter._build_sequences()` using the same `np.lib.stride_tricks.sliding_window_view()` call; IT-2 validates output shape matches expected `(n_rows - seq_len + 1, seq_len, n_feat)` | P0-B-2 |
| R10 | Ensemble build format mismatch: `build_ensemble_bundle()` output not loadable by `EnsembleBundle.load()` (G6, deferred) | High | Medium | **MEDIUM** | P0 does not fix G6 (deferred to P1); deploy selector falls back to best single model if ensemble bundle is not loadable; warning logged; manifest records artifact_type accurately | P0-C-3 |
| R11 | Race condition in Phase 5: deploy packaging reads bundles/ directory while bundling is still writing (unlikely in sequential factory, but possible in future parallel mode) | Very Low | Medium | **LOW** | Factory runs phases sequentially; Phase 5 only starts after Phase 4 checkpoint; directory operations use `exist_ok=True`; entire Phase 5 wrapped in try/except | P0-C-2 |
| R12 | 4D resampling alignment: different timeframes produce different numbers of windows; `min_windows` alignment truncates data, potentially losing significant history at coarser TFs | Low | Medium | **LOW** | Alignment takes the last `min_windows` from each TF (most recent data); this matches the training adapter behavior; documented in `_build_4d_input()` docstring | P0-B-3 |
| R13 | BundleMetadata field name collision: new field names (e.g., `scaling_source`) could conflict with existing dynamic attributes or subclass fields | Very Low | Medium | **LOW** | Field names are specific and unlikely to collide; `from_dict()` uses `.get()` which silently returns default on missing key; no subclassing of BundleMetadata exists in codebase | P0-A-3 |

---

## Risk Mitigations (Detailed)

### R1: Double-Scaling Regression

**When applied:** During P0-A-5 implementation and in ongoing code review.

**Specific mitigation:**
1. Change `skip_scaling=False` to `skip_scaling=True` at bundle.py L1041
2. Add inline comment: `# Bundle's own scaler applies in predict(); avoid double-scaling (see G2)`
3. Integration test IT-1 verifies single scaling by comparing feature magnitudes before and after
4. `ScalingSource` enum (P0-A-2) provides vocabulary for future documentation and enforcement
5. Code review checklist item: "Any change to preprocess() or predict() scaling logic must verify single-source scaling"

### R2: 4D Model MTF Data Availability

**When applied:** During P0-B-3 implementation.

**Specific mitigation:**
1. `_build_4d_input()` validates input at three levels:
   - DatetimeIndex present (raises `ValueError: 4D models require a DatetimeIndex`)
   - OHLCV columns present -- at least 4 of 5 (raises `ValueError: Need at least: open, high, low, close`)
   - Sufficient rows per timeframe after resampling (raises `ValueError: Insufficient data for timeframe '{tf}'`)
2. BundleMetadata stores `primary_timeframe` so users know what resolution to provide
3. Notebook Cell 8 catches ValueError and prints: "This is expected if P0-B is not yet implemented for this model type"
4. `bundle.validate()` includes metadata check for timeframe requirements

### R3: EnsembleBundle Backward Compat (Absolute Paths)

**When applied:** During P0-B-5 implementation.

**Specific mitigation:**
1. `save()`: Convert paths to relative via `Path.relative_to(path.parent)`, with `try/except ValueError` fallback to absolute for cross-root paths
2. `load()`: Check `Path.is_absolute()` on each loaded path; if absolute, use as-is; if relative, resolve via `(path.parent / p).resolve()`
3. Both old (absolute) and new (relative) formats work in the same load() method
4. Integration test IT-4 step 5 verifies `base_bundles.json` contains relative paths after save

### R4: Calibrator None Propagation

**When applied:** During P0-A-4 implementation.

**Specific mitigation:**
1. Add `calibrator: Any | None = None` field to orchestrator's `ModelTrainingResult` (L78-111)
2. Update `_train_single_model()` at L912-920: `calibrator=getattr(result, "calibrator", None)`
3. Update `_train_boosting_parallel()` at L799-807 with same pattern
4. Update `BundleBuilder._extract_calibrator()` to accept `model_result` parameter and check it first
5. Smoke test SM-A4 verifies the field exists
6. Integration test IT-1 verifies `has_calibrator == True` when `auto_calibrate=True` in training config

### R5: Notebook Fails in Colab

**When applied:** During P0-D implementation.

**Specific mitigation:**
1. Every notebook cell starts with `if "result" not in dir() or result is None or not result.success: print("No successful result.")`
2. Predict demo uses `raw_data.tail(n_bars)` -- bounded memory usage
3. Cell 9 zips only `deploy/` (not full output) -- smaller download
4. Cell 11 provides Drive persistence as alternative to download
5. All bundle loading and prediction wrapped in `try/except Exception`
6. Cell 8 prints "(This is expected if P0-B is not yet implemented for this model type)" on predict failure

### R6: Feature Column Drift

**When applied:** During P0-A-3 and P0-B-1 implementation.

**Specific mitigation:**
1. BundleMetadata v1.3.0 adds `feature_names: list[str]` field
2. `feature_columns` on ModelBundle remains the authoritative list (feature_names is backup)
3. `_apply_adapter()` for 2D models converts DataFrame to ndarray using the stored column order
4. `validate()` checks `len(feature_columns) == metadata.n_features`
5. For 4D models, feature drift is not applicable (raw OHLCV columns are fixed: open, high, low, close, volume)

### R7: Deploy Selector Picks Wrong Artifact

**When applied:** During P0-C-3 implementation.

**Specific mitigation:**
1. Selection uses `val_f1` consistently (same key used in both training metrics and selection)
2. If ensemble `val_f1` >= best single `val_f1`, ensemble wins; otherwise best single model wins
3. If ensemble bundle directory not found on disk despite ensemble result existing, falls back to best single model with warning
4. Manifest records the exact metric values used for the selection decision
5. Selection decision is logged: `f"h{horizon}: {entry.artifact_type} ({entry.model_key}) -> {artifact_dir}"`

### R8: PreprocessingGraph Config Hardcoding

**When applied:** During P0-A-3 and P0-A-6 implementation.

**Specific mitigation:**
1. P0-A-3 adds `scaler_type` and `primary_timeframe` to BundleMetadata
2. P0-A-6 populates these from `get_model_contract()` at bundle creation time
3. For 4D models, `_build_4d_input()` bypasses PreprocessingGraph entirely, avoiding the hardcoded config
4. For 2D/3D models, the preprocessing graph's hardcoded `scaler_type="robust"` matches the most common training config
5. Full fix (G18: derive preprocessing config from training) is deferred to P1 -- documented as known limitation

---

# Part 3: Rollout Strategy

---

## Feature Flags / Migration Order

### Toggleable Features

| Feature | Config Path | Default | Effect When False |
|---------|------------|---------|-------------------|
| `deploy_artifact` | `BundlingSection.deploy_artifact` | `True` | Skips Phase 5 entirely; no deploy/ directory created; factory run still succeeds |
| `create_bundle` | `BundlingSection.create_bundle` | `True` | Skips both Phase 4 (bundling) and Phase 5 (deploy); no bundles or deploy artifacts |

### Recommended Rollout Sequence

**Phase 1: Foundation (Batch 1-2, no user-visible changes)**

Deploy P0-A tasks. These are purely additive (new file, new enum, new fields with safe defaults, bug fixes). No existing behavior changes for users who do not use `predict_from_raw()` on neural models.

Risk: Very low. All changes are backward compatible.

Rollout: Merge directly. Run smoke tests SM-A1 through SM-A6.

**Phase 2: Core Inference (Batch 3-4, enables new capability)**

Deploy P0-B tasks. `predict_from_raw()` now works for all 12 models. `EnsembleBundle.predict_from_raw()` is new. Relative paths in ensemble bundles.

Risk: Low. New methods/paths are additive. Existing `predict(X_preshaped)` is unchanged.

Rollout: Merge directly. Run integration tests IT-1 through IT-4.

**Phase 3: Deploy Packaging (Batch 5-6, new Phase 5 in factory)**

Deploy P0-C tasks. Factory now creates `deploy/` directory after bundling. New `deploy_artifact` config toggle. New `load_deploy_artifact()` function.

Risk: Low. Phase 5 is fail-safe (try/except). Existing factory runs work unchanged.

Rollout: Merge directly. `deploy_artifact` defaults to `True` but can be disabled. Run IT-5.

**Phase 4: Notebook Integration (Batch 7, user-facing cells)**

Deploy P0-D tasks. Four new notebook cells + config additions.

Risk: Low. All cells are defensive (check result exists, catch exceptions).

Rollout: Merge directly. Run notebook E2E checklist.

---

## Compatibility Window

### Old Paths That Remain Functional

| Component | Current Path | New Path | Coexistence |
|-----------|-------------|----------|-------------|
| `InferencePipeline` | `src.inference.pipeline` | No replacement in P0 | Remains fully functional; not deprecated in P0 |
| `InferenceOrchestrator` | `src.inference.orchestrator` | `load_deploy_artifact()` | Both work; orchestrator is the established path; deploy is the new recommended path |
| `ModelBundle.predict(X)` | Pre-shaped input | `predict_from_raw(raw_df)` | Both work; predict() is unchanged |
| `BundleBuilder.build_from_training_result()` | Individual bundles only | Same (no change) | Unchanged |

### Deprecation Timeline

| Component | Status After P0 | Can Remove After |
|-----------|----------------|-----------------|
| `InferencePipeline` | Active, not deprecated | P2 (after UIP is built and validated) |
| `InferenceOrchestrator` | Active, not deprecated | P2 (after deploy path is proven in production) |
| `ModelBundle.predict(X)` | Active, will never be deprecated | Never (it is the core predict method) |
| `EnsembleBundle.predict_from_base_features()` | Active, not deprecated | P1 (after predict_from_raw is validated for ensembles) |
| Duck-typing extraction in BundleBuilder | Active as fallback | P2 (after all trainers implement TrainerProtocol) |

### Recommended Removal Timeline

- **P0 (current):** No removals. All old paths remain functional.
- **P1 (next):** Fix G6 (build_ensemble_bundle format), fix G18 (preprocessing config), fix G24 (meta-learner loading). Consider deprecation warnings on InferencePipeline.
- **P2 (future):** Build UniversalInferencePipeline (G12). Deprecate InferencePipeline and InferenceOrchestrator with removal warnings. Remove duck-typing fallbacks from BundleBuilder after verifying all trainers implement TrainerProtocol.
- **P3 (cleanup):** Remove deprecated components. Clean up G26-G29 (low-severity gaps).

---

## Success Metrics

### Quantifiable "Done" Criteria

| # | Criterion | Measurement | Target |
|---|-----------|-------------|--------|
| 1 | Models pass predict_from_raw smoke test | Count of models where `predict_from_raw()` returns `PredictionResult` | **12/12** |
| 2 | Notebook runs end-to-end with deploy output | Cell-by-cell execution in fresh Colab runtime completes without error | All 12 cells pass |
| 3 | Old bundles load without error | Load a v1.2.0 bundle after code changes | BundleMetadata.from_dict() succeeds with safe defaults |
| 4 | Ruff check passes | `ruff check src/` | **0 errors** |
| 5 | Black formatting passes | `black --check src/` | **0 reformats needed** |
| 6 | Deploy directory created | `result.deploy_path` is not None after `Factory.run()` | deploy/ exists with manifest.json |
| 7 | Manifest is pure JSON | Load manifest.json with `json.load()` and no src/ imports | Loads successfully |
| 8 | Per-horizon validation passes | `validation.json` for each horizon has `"passed": true` | All horizons pass |
| 9 | Ensemble predict_from_raw works | `EnsembleBundle.predict_from_raw(raw_df)` returns PredictionResult | Works for homogeneous and heterogeneous ensembles |
| 10 | Calibrator reaches bundles | Bundle saved with `auto_calibrate=True` has `metadata.has_calibrator == True` | Calibrator not None in bundle |
| 11 | No double scaling | Feature values after `predict_from_raw()` match single-scaled values | Manual comparison confirms single scaling |
| 12 | All CRITICAL gaps closed | G1-G5 from gap analysis | 5/5 CRITICAL gaps resolved |
| 13 | All smoke tests pass | Run all SM-* commands | **24/24** smoke tests pass |
| 14 | Integration tests pass | Run IT-1 through IT-6 | **6/6** integration tests pass |

### Acceptance Gate

The P0 implementation is considered **complete** when:

1. All 24 smoke tests pass (SM-A1 through SM-D5)
2. All 6 integration tests pass (IT-1 through IT-6)
3. `ruff check src/` returns 0 errors
4. `black --check src/` returns 0 reformats
5. 12/12 models pass `predict_from_raw()` (may be verified incrementally per model family)
6. Notebook executes end-to-end in Colab without cell errors
7. All 5 CRITICAL gaps (G1-G5) are closed per the gap coverage matrix

---

*This document is a planning artifact. No code has been modified.*
