# ML Factory - Remaining Tasks

**Last Updated:** 2026-01-24
**Status:** Phases 0-6 Complete | All Models Implemented

---

## Completed (see COMPLETION.md)

| Phase | Impact |
|-------|--------|
| Phase 0 | -5,336 lines (deduplication) |
| Phase 1 | +616 lines (contract enforcement) |
| Phase 2 | +958 lines (4D infrastructure) |
| Phase 3 | +2,298 lines (5D Optuna) |
| Phase 4 | +50 lines (validation wiring) |
| Phase 5 | +1,281 lines (MLFactory + ExperimentConfig) |
| Phase 6 | +3,690 lines (6 advanced neural models) |

---

## Completed: Advanced Models (Phase 6)

| Model | Status |
|-------|--------|
| InceptionTime | ✅ |
| 1D ResNet | ✅ |
| PatchTST | ✅ |
| iTransformer | ✅ |
| TFT | ✅ |
| N-BEATS | ✅ |

---

### 3D Models (Sequence Adapter)

#### InceptionTime ✅
**Location:** `src/models/neural/inceptiontime_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~500 lines)

#### 1D ResNet ✅
**Location:** `src/models/neural/resnet1d_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~550 lines)

---

### 4D Models (MultiStream Adapter)

#### PatchTST ✅
**Location:** `src/models/neural/patchtst_model.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`
**Status:** Implemented (~480 lines)

#### iTransformer ✅
**Location:** `src/models/neural/itransformer_model.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`
**Status:** Implemented (~620 lines)

### 3D Models (Sequence Adapter) - Continued

#### TFT (Temporal Fusion Transformer) ✅
**Location:** `src/models/neural/tft_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~780 lines)

#### N-BEATS ✅
**Location:** `src/models/neural/nbeats_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~760 lines)

---

---

## Phase 7: Production Hardening

**Status:** NOT STARTED
**Priority:** HIGH

### 7A: Make Validation Blocking by Default

| Task | File | Line | Action |
|------|------|------|--------|
| ⬜ | `src/validation/leakage_detection.py` | 121 | Change `raise_on_leakage: bool = False` → `True` |
| ⬜ | `src/validation/lookahead_audit.py` | 212 | Change `raise_on_lookahead: bool = False` → `True` |
| ⬜ | `src/models/training/unified_orchestrator.py` | 338 | Change `strict_validation` default → `True` |
| ⬜ | `src/models/training/trainer.py` | 393 | Add `_run_pre_training_validation()` call in `run()` |

**Verification:**
```bash
# Should raise on leakage by default
python -c "from src.validation.leakage_detection import check_feature_label_correlation; import inspect; sig=inspect.signature(check_feature_label_correlation); print('OK' if sig.parameters['raise_on_leakage'].default else 'FAIL')"
```

### 7B: Add Inter-Stage Schema Validation

| Task | File | Action |
|------|------|--------|
| ⬜ | `src/data/pipeline/schemas.py` | CREATE: Define input/output schemas for each stage |
| ⬜ | `src/data/pipeline/runner.py:186-214` | Add schema validation after each stage completes |
| ⬜ | `src/data/pipeline/stages/features/engineer.py:405-409` | Add minimum row validation after NaN cleaning |

**Schema Example:**
```python
STAGE_SCHEMAS = {
    "data_cleaning": {
        "required_columns": ["datetime", "open", "high", "low", "close", "volume"],
        "min_rows": 1000,
    },
    "feature_engineering": {
        "max_nan_ratio": 0.01,
    },
}
```

### 7C: Consistent Adapter Error Handling

| Task | File | Line | Action |
|------|------|------|--------|
| ⬜ | `src/data/adapters/sequence.py` | 170-174 | Change from `logger.warning()` to `raise ValidationError()` |
| ⬜ | `src/data/adapters/base.py` | 222 | Change `model_contract: ModelContract | None = None` → required |

**Verification:**
```bash
# SequenceAdapter should raise on validation failure
python -c "from src.data.adapters.sequence import SequenceAdapter; print('Check source for raise')"
```

### 7D: Feature Manifest System

| Task | File | Action |
|------|------|--------|
| ⬜ | `src/data/pipeline/stages/features/engineer.py` | Output feature manifest JSON with column names |
| ⬜ | `src/data/adapters/base.py:261-296` | Load manifest instead of auto-detecting via prefix matching |
| ⬜ | `src/data/pipeline/feature_manifest.py` | CREATE: `FeatureManifest` dataclass |

---

## Phase 8: Code Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM

### 8A: Extract Common Utilities

| Task | Action | Files to Update |
|------|--------|-----------------|
| ⬜ | CREATE `src/core/utils/math_utils.py` with `safe_divide()`, `sma()` | 5+ pipeline stage files |
| ⬜ | CREATE `src/core/utils/device_utils.py` with `check_cuda_available()` | xgboost, lightgbm, catboost, neural |
| ⬜ | CREATE `src/models/common/class_weights.py` with `compute_balanced_weights()` | xgboost, lightgbm, catboost, base_rnn |

**Files with `_safe_divide` to update:**
- `src/data/pipeline/stages/features/microstructure.py:25`
- `src/data/pipeline/stages/features/momentum.py:18`
- `src/data/pipeline/stages/features/moving_averages.py:22`
- `src/data/pipeline/stages/features/price_features.py:31`
- `src/data/pipeline/stages/features/volume.py:19`

**Files with `_check_cuda_available` to update:**
- `src/models/boosting/catboost_model.py:89`
- `src/models/boosting/lightgbm_model.py:92`
- `src/models/boosting/xgboost_model.py:95`
- `src/models/ensemble/xgboost_meta.py:67`
- `src/models/neural/base_rnn.py:156`

### 8B: Consolidate Exceptions

| Task | File | Action |
|------|------|--------|
| ⬜ | `src/core/exceptions.py` | CREATE: Unified exception hierarchy |
| ⬜ | `src/models/config/exceptions.py:10` | RENAME: `ConfigValidationError` → `ModelConfigValidationError` |
| ⬜ | `src/data/store/*.py` | Consolidate store errors to use common base |

**Exception Hierarchy:**
```python
# src/core/exceptions.py
class MLFactoryError(Exception): ...
class ValidationError(MLFactoryError): ...
class ConfigError(MLFactoryError): ...
class ContractViolation(MLFactoryError): ...
class DataError(MLFactoryError): ...
```

### 8C: Extract Magic Numbers

| Task | Action |
|------|--------|
| ⬜ | CREATE `src/config/constants/default_periods.py` with `RSI_PERIOD=14`, `ATR_PERIOD=14`, `BB_PERIOD=20`, `DEFAULT_LOOKBACK=100` |
| ⬜ | CREATE `src/config/constants/thresholds.py` with `MIN_SIGNAL_RATIO=0.10`, `DEFAULT_WEIGHT=0.5`, `VOL_RATIO_MIN=0.5`, `VOL_RATIO_MAX=2.0` |
| ⬜ | Update `src/data/pipeline/config/labeling_config.py` to use constants |
| ⬜ | Update `src/data/pipeline/config/regime_config.py` to use constants |
| ⬜ | Update `src/data/pipeline/config/features.py` to use constants |

### 8D: Deprecation Cleanup

| Task | File | Line | Action |
|------|------|------|--------|
| ⬜ | `src/models/boosting/catboost_model.py` | 23 | Change `PredictionOutput` → `PredictionResult` |
| ⬜ | `src/models/classical/random_forest.py` | - | Change `PredictionOutput` → `PredictionResult` |
| ⬜ | `src/models/base.py` | 456 | Remove deprecated `PredictionOutput` alias after migration |

**Verification:**
```bash
# Should find 0 uses of PredictionOutput
grep -r "PredictionOutput" src/ --include="*.py" | wc -l
```

---

## Phase 9: Directory Cleanup

**Status:** NOT STARTED
**Priority:** LOW

### 9A: Delete Empty Directories

```bash
# Execute after verifying no imports
rm -rf src/contracts
rm -rf src/ml_pipeline
rm -rf src/adapters
rm -rf src/common
rm -rf src/monitoring
rm -rf src/feature_store
rm -rf src/utils
rm -rf src/cross_validation
rm -rf src/evaluation
rm -rf src/backtesting
rm -rf src/pipeline
rm -rf src/features
```

**Pre-check:** Verify no imports reference these directories:
```bash
# All should return 0
grep -r "from src\.contracts" src/ --include="*.py" | wc -l
grep -r "from src\.ml_pipeline" src/ --include="*.py" | wc -l
grep -r "from src\.adapters" src/ --include="*.py" | wc -l
# ... etc
```

### 9B: Delete Deprecated Shims

| Task | File | Pre-requisite |
|------|------|---------------|
| ⬜ | Fix imports from `src.training` | Update 3 files that import from it |
| ⬜ | Fix imports from `src.pipeline_config` | Update 3 files that import from it |
| ⬜ | DELETE `src/training/__init__.py` | After fixing imports |
| ⬜ | DELETE `src/pipeline_config.py` | After fixing imports |
| ⬜ | DELETE `src/data/pipeline/stages/datasets/adapters/__init__.py` | Verify no imports |

**Files importing from deprecated paths:**
- `src/models/training/regime_detector.py` → imports from `src.training`
- `src/models/training/modes/__init__.py` → imports from `src.training`
- `src/orchestrator.py` → imports from `src.pipeline_config`
- `src/config/smart_config.py` → imports from `src.pipeline_config`
- `src/cli/status_commands.py` → imports from `src.pipeline_config`

### 9C: Update Documentation

| Task | File | Section | Action |
|------|------|---------|--------|
| ⬜ | `CLAUDE.md` | "Model Support" (line ~89) | Update to show all 12 models working |
| ⬜ | `CLAUDE.md` | "Project Structure" (line ~140) | Remove `src/training/` or clarify deprecation |
| ⬜ | `CLAUDE.md` | "Current Status" | Update Phase 0-6 complete status |

---

## Phase 10: Refactor Complex Functions

**Status:** NOT STARTED
**Priority:** LOW (HIGH RISK)

### 10A: Split stacking.py:fit()

**Location:** `src/models/ensemble/stacking.py:187`
**Current:** 400+ lines, 30 ifs, 30 loops

| Task | Extract To | Lines |
|------|-----------|-------|
| ⬜ | `_validate_ensemble_config()` | ~30 lines |
| ⬜ | `_prepare_heterogeneous_data()` | ~50 lines |
| ⬜ | `_cache_sequence_data()` | ~40 lines |
| ⬜ | `_check_memory_usage()` | ~20 lines |
| ⬜ | `_train_final_base_models()` | ~80 lines |
| ⬜ | `_compute_diversity_metrics()` | ~30 lines |

**Target:** Reduce `fit()` to ~50 lines of orchestration

### 10B: Split _pre_training_validation()

**Location:** `src/models/training/unified_orchestrator.py:299`
**Current:** 150+ lines, 24 ifs, 11 loops

| Task | Extract To | Lines |
|------|-----------|-------|
| ⬜ | `_validate_data_contract()` | ~30 lines |
| ⬜ | `_detect_leakage()` | ~40 lines |
| ⬜ | `_audit_lookahead()` | ~30 lines |
| ⬜ | `_validate_columns()` | ~20 lines |

**Target:** Reduce `_pre_training_validation()` to ~20 lines of delegation

---

## Phase 11: Deferred Items (Low Priority)

| Task | Description | Notes |
|------|-------------|-------|
| 5C | Unified deployment bundle (tar.gz format) | Needs bundle spec |
| 4C | Ensemble diversity analysis integration | Wire DiversityAnalyzer |
| 4D | Deflated Sharpe Ratio post-Optuna validation | Add DSR gate |
| 4E | Bootstrap CIs in financial reports | Wire BootstrapCI |
| 4F | Auto calibration in orchestrator | Wire CalibrationManager |
| 4G | Bet sizing connection to backtest | Wire BetSizer |
| - | MTF ablation flag | Add `mtf.enabled` config |

---

*For completed phase details, see COMPLETION.md*
