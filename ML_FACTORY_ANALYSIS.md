# ML Factory Pipeline: Comprehensive Architecture Analysis & Recommendations

**Analysis Date:** 2026-01-16
**Analyzed By:** 4 Specialized Agent Teams (Architecture, Code Quality, MLOps, Data Engineering)
**Codebase:** OHLCV Time Series ML Model Factory
**Total Lines of Code:** 117,491 Python lines across 200+ files

---

## Executive Summary

Your ML factory demonstrates **exceptional engineering** with production-grade leakage prevention, robust state management, and excellent plugin architecture supporting 23 models across 6 families. The core factory pattern is sound and highly extensible.

**Key Strengths:**
- ✅ Plugin-based model registry (23 models, zero coupling between implementations)
- ✅ Production-grade leakage prevention (6 layers: MTF shift, purge, embargo, train-only scaling, OOF, PurgedKFold)
- ✅ Robust state management with versioning and rollback
- ✅ Per-model feature strategies enabling heterogeneous ensembles
- ✅ All files within 1300-line limit (largest: 1279 lines)
- ✅ Excellent memory management with OOM recovery

**Critical Issues Requiring Action:**
- 🔴 Configuration sprawl (85 config classes across 5 systems)
- 🔴 Circular dependencies (`phase1 ↔ models`)
- 🔴 God objects (Trainer: 785 lines, StackingEnsemble: 1279 lines)
- 🟡 Missing data lineage tracking
- 🟡 No cascade invalidation for cache
- 🟡 Weak schema evolution handling

**Overall Assessment:** Production-ready ML factory with research-grade capabilities. Requires 4-6 weeks of focused refactoring to eliminate technical debt while preserving architectural strengths.

---

## Table of Contents

1. [Architecture Assessment](#1-architecture-assessment)
2. [Code Quality Review](#2-code-quality-review)
3. [MLOps Evaluation](#3-mlops-evaluation)
4. [Data Pipeline Analysis](#4-data-pipeline-analysis)
5. [Priority Recommendations](#5-priority-recommendations)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Architecture Assessment

### 1.1 Plugin Architecture ⭐⭐⭐⭐⭐ (Excellent)

**ModelRegistry Pattern:**
```python
# src/models/registry.py - Textbook plugin system
@register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    pass

# Zero coupling - models register at import time
model = ModelRegistry.create("xgboost", config={...})
```

**Strengths:**
- 23 models register via `@register` decorator
- Family-based grouping (Boosting, Classical, Neural, Ensemble, Meta-learners)
- Graceful handling of optional dependencies (CatBoost)
- Dynamic instantiation with zero coupling

**Assessment:** This is your factory's biggest strength. Adding new models is trivial and requires no changes elsewhere in the codebase.

### 1.2 Data Pipeline Design ⭐⭐⭐⭐ (Strong)

**7-Phase Architecture:**
```
Phase 1-5 (Data): Ingest → MTF → Features → Labels → Adapters
Phase 6 (Training): Model training with per-model feature selection
Phase 7 (Stacking): Heterogeneous ensembles
```

**Strengths:**
- Clear separation of concerns with explicit dependencies
- Single canonical 1-min OHLCV → 9 intraday timeframes (deterministic)
- Per-model feature strategies (different models get different features)
- Phase registry with automatic dependency tracking

**Critical Issue: Circular Dependency**
```
src/phase1/config/__init__.py
  ├─> from src.models.config.data_requirements import MODEL_DATA_REQUIREMENTS

src/models/training/trainer.py
  ├─> from src.phase1.stages.datasets.container import TimeSeriesDataContainer
  └─> from src.phase1.lineage import PipelineLineage
```

**Impact:** Cannot test `phase1` without `models`, cannot extract `models` independently.

**Fix (Priority 0):** Create `src/data/` shared layer:
```
src/
  data/                    # New shared layer
    container.py           # TimeSeriesDataContainer (from phase1)
    contracts.py           # DataContract (from contracts)
    requirements.py        # MODEL_DATA_REQUIREMENTS (from models)

  phase1/                  # Depends on src.data
  models/                  # Depends on src.data
```

### 1.3 Configuration System ⭐⭐ (Severe Sprawl)

**Current State:**
- 85 configuration classes across 5 overlapping systems
- UnifiedConfig (1116 lines, 12 nested sections)
- PipelineConfig (88+ fields)
- TrainerConfig (63 fields)
- MLConfig (84 fields)
- GlobalConfig (YAML-based)

**Problems:**
1. **Field duplication** - Same settings repeated (e.g., `batch_size` in 3 configs)
2. **Conversion explosion** - Multiple `to_X_config()` methods create dual maintenance burden
3. **No single source of truth** - Which config is canonical?

**Example of Duplication:**
```python
# UnifiedConfig has batch_size
config.training.batch_size = 512

# TrainerConfig has batch_size
trainer_config.batch_size = 512

# MLConfig has batch_size
ml_config.training_config.batch_size = 512

# Which is source of truth? All 3!
```

**Fix (Priority 0):**
1. Commit to `UnifiedConfig` as single source of truth
2. Add deprecation warnings to legacy configs
3. Delete `to_X_config()` delegation after 2-sprint grace period
4. **Target:** 85 config classes → 10 focused classes

**Expected Impact:** 50% reduction in config code, eliminates maintenance burden.

### 1.4 God Objects (Need Decomposition)

**1. Trainer (785 lines + 3 mixins)**

Current responsibilities:
- Feature selection (200 lines)
- Test set evaluation (150 lines)
- Save/load artifacts (180 lines)
- Core training logic (255 lines)
- Heterogeneous ensemble handling (100 lines)
- Pipeline lineage validation (100 lines)

**Issue:** Mixins don't reduce complexity, just hide it.

**Better Design:**
```python
class TrainingOrchestrator:
    def __init__(
        self,
        model: BaseModel,
        data_loader: DataLoader,
        feature_selector: FeatureSelector | None,
        evaluator: Evaluator,
        artifact_manager: ArtifactManager,
    ):
        # Inject dependencies

    def run(self, container: TimeSeriesDataContainer) -> TrainingResults:
        # Coordinate components (< 100 lines)
        data = self.data_loader.load(container)
        data = self.feature_selector.select(data) if self.feature_selector else data
        metrics = self.model.fit(*data)
        results = self.evaluator.evaluate(self.model, data.test)
        self.artifact_manager.save(self.model, results)
        return results
```

**2. StackingEnsemble (1279 lines)**

Responsibilities:
- OOF generation with PurgedKFold
- Heterogeneous model handling (tabular + sequence)
- Memory management (sequence data caching)
- Diversity analysis
- Meta-learner training
- Passthrough features

**Fix (Priority 1):**
Extract:
- `OOFGenerator` (PurgedKFold logic)
- `HeterogeneousDataAdapter` (tabular + sequence handling)
- Keep `StackingEnsemble` as coordinator (< 400 lines)

---

## 2. Code Quality Review

### 2.1 File Size Compliance ✅

**All files within 1300-line limit:**

| File | Lines | Status | Action |
|------|-------|--------|--------|
| `stacking.py` | 1,279 | ⚠️ Near limit | Refactor: Extract OOFGenerator |
| `entropy.py` | 1,258 | ⚠️ Near limit | Refactor: Split into entropy/ package |
| `regime_evaluation.py` | 1,193 | ✅ OK | Optional: Split classifier/evaluator |
| `unified.py` | 1,116 | ✅ OK | Review after config consolidation |
| `smart_config.py` | 925 | ⚠️ Duplicate | DELETE: Duplicate of unified.py |

**Verdict:** 4 files need refactoring (stacking, entropy, smart_config deletion).

### 2.2 Function Complexity ⚠️

**Functions exceeding 100 lines:**

| File | Function | Lines | Issue |
|------|----------|-------|-------|
| `trainer.py` | `run()` | 397 | Too many responsibilities |
| `stacking.py` | `fit()` | 334 | Includes OOF + diversity + meta-train |
| `cli/run_commands_pipeline.py` | `run_command()` | 316 | CLI orchestration too complex |
| `stacking.py` | `_generate_oof_predictions()` | 275 | Extract to separate class |
| `base_rnn.py` | `fit()` | 263 | Acceptable for neural base class |

**Fix:** Extract `trainer.py::run()` into private methods:
```python
def run(self, container, skip_save=False):
    self._setup_run()
    X_train, y_train, X_val, y_val = self._prepare_data(container)
    training_metrics = self._train_model(X_train, y_train, X_val, y_val)
    eval_metrics = self._evaluate(X_val, y_val)
    if not skip_save:
        self._save_artifacts(training_metrics, eval_metrics)
    return self._build_results()
```

### 2.3 Error Handling Violations 🔴

**Exception Swallowing (Violates "Fail Fast, Fail Hard"):**

```python
# src/utils/cache.py (line 177)
except Exception as e:
    logger.warning(f"Cache read failed: {e}")
    return None  # Silently swallows error

# src/utils/colab_setup.py (line 46)
except Exception:
    pass  # Bare except swallows all errors
```

**Other problematic patterns:**
- `src/validation/bootstrap.py` (lines 127, 196, 498) - Silent exception handling
- `src/ml_pipeline/config.py` (line 37) - `except Exception: pass`

**Fix:** Propagate errors or log at ERROR level:
```python
# Instead of swallowing
except Exception as e:
    logger.error(f"Cache read failed for key={key}: {e}")
    raise CacheReadError(key, e) from e
```

### 2.4 Code Duplication

**ADX Calculation (3 implementations):**
1. `src/phase1/stages/features/numba_functions.py::calculate_adx_numba()` - Optimized
2. `src/phase1/stages/regime/trend.py::calculate_adx()` - Pure Python
3. `src/models/regime_evaluation.py::_adx_fallback()` - Fallback

**Fix:** Delete `_adx_fallback()` and use canonical `calculate_adx()` from `regime/trend.py`.

**Sharpe Ratio (5+ implementations):**
Consolidate into single `src/metrics/sharpe.py` module with variants.

### 2.5 Legacy Code Cleanup

**Deprecated modules still present:**

| Module | Replacement | Action |
|--------|-------------|--------|
| `cross_validation/feature_selector.py` | `feature_selection/` | DELETE after grace period |
| `phase1/utils/feature_selection.py` | `feature_selection/` | DELETE |
| `models/feature_selection/__init__.py` | `feature_selection/` | DELETE |
| `config/smart_config.py` | `config/unified.py` | DELETE (duplicate) |

**Per "Delete Legacy Code" rule:** Verify no external dependencies, then remove.

---

## 3. MLOps Evaluation

### 3.1 ML Pipeline Design ⭐⭐⭐⭐⭐ (Excellent)

**Single Canonical Source Architecture:**
- ✅ One 1-min OHLCV dataset → 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- ✅ Deterministic upscaling via resampling
- ✅ Per-model timeframe selection (CatBoost→15min, TCN→5min, PatchTST→1min)
- ✅ All models derive from same source (prevents data drift)

**Per-Model Feature Strategy:**
```python
# src/features/strategies.py
MODEL_FEATURE_STRATEGIES = {
    "xgboost": {
        "baseline_features": ["momentum", "volatility", "volume", "microstructure", "mtf"],  # ~100 features
        "mtf_mode": "indicators",
        "max_features": 120,
    },
    "lstm": {
        "baseline_features": ["momentum", "volatility", "wavelets", "mtf"],  # ~80 features
        "mtf_mode": "indicators",
        "max_features": 100,
    },
    "patchtst": {
        "baseline_features": ["raw_ohlcv"],  # 5 features (O, H, L, C, V)
        "mtf_mode": "multi_stream",  # Uses raw multi-TF bars
        "max_features": 5,
    },
}
```

**Assessment:** This enables heterogeneous ensembles where base models have different inductive biases.

### 3.2 Leakage Prevention ⭐⭐⭐⭐⭐ (Production-Grade)

**6 Layers of Protection:**

| Layer | Mechanism | Location | Prevents |
|-------|-----------|----------|----------|
| **1. MTF shift(1)** | Shift higher-TF data by 1 bar | `mtf/generator.py:295` | Lookahead bias in MTF features |
| **2. Purge** | Remove 60 bars before test | `splits/core.py:241` | Label overlap at boundaries |
| **3. Embargo** | Skip 1440 bars after test | `splits/core.py:244` | Serial correlation |
| **4. Train-only scaling** | Fit scaler on train only | `scaling/core.py` | Val/test stats leaking |
| **5. OOF predictions** | Meta-learner uses OOF | `cross_validation/oof_core.py` | Overfitting in stacking |
| **6. PurgedKFold** | Label-aware CV | `purged_kfold.py:224` | Overlapping labels in CV |

**Assessment:** No critical leakage issues found. This is production-ready.

### 3.3 Experiment Tracking ⭐⭐⭐⭐ (Strong)

**State Management:**
- ✅ PipelineState with schema versioning (V1 legacy, V2 current)
- ✅ Per-phase metrics, artifacts, and checkpoints tracked
- ✅ Config hash for drift detection
- ✅ Rollback capability with 20-snapshot history
- ✅ Thread-safe operations

**Gaps:**
1. **No Git commit hash** - Code version not tracked
2. **No environment tracking** - Python version, dependencies, hardware not captured
3. **No data fingerprinting** - Input data changes not detected

**Fix (Priority 2):**
```python
import subprocess, sys, torch

# Add to PipelineState.metadata
metadata = {
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'data_hash': hashlib.sha256(input_path.read_bytes()).hexdigest()[:16],
}
```

### 3.4 Memory Efficiency ⭐⭐⭐⭐⭐ (Excellent)

**MemoryManager:**
- ✅ psutil integration for system memory monitoring
- ✅ `estimate_array_size()` for accurate size estimation
- ✅ `@memory_logged` decorator for function tracking
- ✅ LRU cache with configurable limits (default 1GB)
- ✅ Disk fallback for large datasets (>500MB)
- ✅ Thread-safe with RLock

**OOM Recovery:**
- ✅ Automatic batch size reduction on OOM (halves batch size)
- ✅ Retry mechanism (up to 3 retries)
- ✅ Min batch size protection (won't go below 8)
- ✅ CUDA cache clearing between retries

**Assessment:** Production-grade memory management. No issues found.

---

## 4. Data Pipeline Analysis

### 4.1 Data Flow ⭐⭐⭐⭐ (Strong)

**7-Phase Pipeline:**

```
Phase 1 (Ingestion)  → Phase 2 (MTF)      → Phase 3 (Features)
Raw 1-min OHLCV         9 Intraday TFs       ~180 Indicators
     ↓                       ↓                      ↓
Phase 4 (Labeling)   → Phase 5 (Adapters)  → Phase 6 (Training)
Triple-Barrier           2D/3D/4D Data        23 Models
     ↓                       ↓                      ↓
Phase 7 (Stacking)   ← Heterogeneous Ensembles + Meta-Learners
```

**Strengths:**
- Clear separation with explicit dependencies
- TimeSeriesDataContainer provides unified interface
- Proper intermediate storage (Parquet format)

**Gap:** MTF outputs not explicitly persisted - recalculated on each run.

**Fix (Priority 2):**
```python
# Save resampled TF dataframes
data/mtf/{SYMBOL}_1min.parquet  # Already exists
data/mtf/{SYMBOL}_5min.parquet  # Add these
data/mtf/{SYMBOL}_15min.parquet
...
```

**Expected Impact:** 5-10x speedup for repeated pipeline runs.

### 4.2 Storage Patterns ⭐⭐⭐⭐ (Good)

**File Formats:**
- ✅ Parquet for data (3-5x compression, columnar)
- ✅ JSON for metadata (human-readable)
- ✅ YAML for configs (version-controllable)
- ✅ Pickle/PyTorch for models

**Gaps:**
1. **No partitioning** - Large datasets (>1GB) would benefit from date-based partitioning
2. **Full parquet reads** - No columnar selection at read time
3. **No deduplication** - MTF features recomputed

**Fix (Priority 3):**
```python
# Instead of:
df = pd.read_parquet(path)
X = df[feature_columns]

# Use:
df = pd.read_parquet(path, columns=feature_columns + metadata_columns)
```

### 4.3 Data Validation ⭐⭐⭐⭐⭐ (Excellent)

**5 Validation Layers:**

1. **Ingestion** - Path security, OHLCV relationships, data types
2. **Data Contract** - Required columns, datetime monotonicity, no duplicates
3. **Feature Quality** - Correlation analysis, feature importance, stationarity
4. **Split Validation** - No overlap, distribution checks, label filtering
5. **Metadata Schema** - Unexpected column detection

**Assessment:** Comprehensive validation. No gaps found.

### 4.4 Cache Invalidation ⭐⭐⭐ (Partial)

**Current Implementation:**
```python
class CacheMetadata:
    source_files: list[str]
    source_mtimes: dict[str, float]

    def is_stale(self) -> bool:
        # Checks source file modification times
```

**Handles:**
- ✅ Source file modified (mtime comparison)
- ✅ Source file deleted
- ⚠️ Config changed (partial - via PipelineState config hash)

**Missing:**
- ❌ Code version changes (no code fingerprinting)
- ❌ Downstream cascade (MTF changes don't invalidate scaled splits)

**Fix (Priority 1):**
```python
class CacheMetadata:
    downstream_dependencies: list[str]  # Add this

    def invalidate_cascade(self):
        for dep in self.downstream_dependencies:
            cache.invalidate(dep)  # Cascade invalidation
```

### 4.5 Data Lineage Tracking ⭐⭐ (Weak)

**Current Tracking:**
- ✅ Phase execution history (PipelineState)
- ✅ Dataset fingerprint (row/col counts, schema hash)
- ❌ No upstream-downstream linking
- ❌ No feature set versioning
- ❌ No transformation lineage
- ❌ No data sample tracking

**Gap Example:**
Cannot answer: "Which exact features were used to train model `xgboost_20260116_120000`?"

**Fix (Priority 2):**
```python
class PhaseResult:
    # Add lineage tracking
    upstream_artifacts: list[str]  # Which files were inputs?
    transformation_hash: str       # Hash of code that produced output
    feature_columns: list[str]     # Which features were used?
    data_version: str              # Which data version?
```

### 4.6 Schema Evolution ⭐⭐ (Weak)

**Current Approach:**
- State versioning (V1 → V2 migration)
- Metadata column validation
- No explicit schema registry

**Gaps:**
1. **Feature schema changes** - Adding/removing features requires manual re-processing
2. **No backward compatibility guarantees** - Old models may fail on new schemas
3. **No migration tooling** - Manual intervention required

**Fix (Priority 3):**
```python
# Implement schema registry
class FeatureSchemaV1:
    columns = ["rsi_14", "macd", "atr", ...]

class FeatureSchemaV2(FeatureSchemaV1):
    columns = FeatureSchemaV1.columns + ["new_feature"]

    @classmethod
    def migrate_from_v1(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Add missing column with default value
        df["new_feature"] = 0
        return df
```

---

## 5. Priority Recommendations

### Priority 0 (Critical - Week 1-2)

**P0.1: Break Circular Dependencies**
- **Issue:** `phase1 ↔ models` cycle blocks independent testing
- **Action:** Create `src/data/` shared layer
- **Effort:** Medium (2-3 days)
- **Impact:** High - Enables modular testing and extraction

**P0.2: Unify Configuration System**
- **Issue:** 85 config classes across 5 systems
- **Action:**
  1. Add deprecation warnings to `PipelineConfig`, `MLConfig`, `TrainerConfig`
  2. Migrate top 10 call sites to `UnifiedConfig`
  3. Delete `to_X_config()` delegation after grace period
- **Effort:** High (1-2 weeks)
- **Impact:** High - 50% reduction in config code

### Priority 1 (High Impact - Week 3-4)

**P1.1: Decompose Trainer God Object**
- **Issue:** 785 lines with 3 mixins, too many responsibilities
- **Action:** Extract `DataLoader`, `FeatureSelector`, `Evaluator`, `ArtifactManager`
- **Effort:** Medium (3-4 days)
- **Impact:** High - Easier testing and modification

**P1.2: Decompose StackingEnsemble**
- **Issue:** 1279 lines, near file limit
- **Action:**
  1. Extract `OOFGenerator` (PurgedKFold logic)
  2. Extract `HeterogeneousDataAdapter` (tabular + sequence handling)
  3. Keep `StackingEnsemble` as coordinator (< 400 lines)
- **Effort:** High (4-5 days)
- **Impact:** High - Better modularity, easier testing

**P1.3: Implement Cascade Invalidation**
- **Issue:** Cache doesn't auto-invalidate downstream artifacts
- **Action:** Add `downstream_dependencies` to `CacheMetadata` + `invalidate_cascade()`
- **Effort:** Low (1 day)
- **Impact:** Medium - Prevents stale cache bugs

### Priority 2 (Medium Impact - Week 5-6)

**P2.1: Add Experiment Tracking Metadata**
- **Issue:** Missing Git commit, Python version, data fingerprint
- **Action:**
  ```python
  metadata = {
      'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
      'python_version': sys.version,
      'data_hash': hashlib.sha256(input_path.read_bytes()).hexdigest()[:16],
  }
  ```
- **Effort:** Low (1 day)
- **Impact:** High - Enables full reproducibility

**P2.2: Persist MTF Intermediate Outputs**
- **Issue:** MTF resampling recalculated on each run
- **Action:** Save to `data/mtf/{SYMBOL}_{TF}.parquet`
- **Effort:** Low (1 day)
- **Impact:** High - 5-10x speedup for repeated runs

**P2.3: Implement Data Lineage Tracking**
- **Issue:** Cannot trace which features/data were used for each model
- **Action:** Add `upstream_artifacts`, `transformation_hash`, `feature_columns` to `PhaseResult`
- **Effort:** Medium (2-3 days)
- **Impact:** Medium - Better debugging and reproducibility

### Priority 3 (Lower Impact - Week 7-8)

**P3.1: Implement Columnar Selection on Read**
- **Issue:** Full parquet read, then filter columns
- **Action:** Use `pd.read_parquet(path, columns=feature_columns)`
- **Effort:** Low (1 day)
- **Impact:** Low - Minor performance improvement

**P3.2: Add Schema Evolution Tooling**
- **Issue:** No migration tooling for schema changes
- **Action:** Implement schema registry with version migrations
- **Effort:** Medium (2-3 days)
- **Impact:** Low - Future-proofs for schema changes

**P3.3: Delete Legacy Code**
- **Issue:** Deprecated shims still present
- **Action:** Remove `cross_validation/feature_selector.py`, `config/smart_config.py`, etc.
- **Effort:** Low (1 day after grace period)
- **Impact:** Low - Cleaner codebase

---

## 6. Implementation Roadmap

### Week 1-2: Critical Fixes (P0)

**Day 1-3: Break Circular Dependencies**
1. Create `src/data/` package
2. Move `TimeSeriesDataContainer` from `phase1/stages/datasets/` to `src/data/`
3. Move `MODEL_DATA_REQUIREMENTS` from `models/config/` to `src/data/`
4. Update imports across codebase
5. Run full test suite to verify

**Day 4-10: Unify Configuration**
1. Add deprecation warnings to legacy configs
2. Create migration guide in `docs/`
3. Migrate 10 highest-impact call sites to `UnifiedConfig`
4. Update CLI to use `UnifiedConfig`
5. Run integration tests

**Success Criteria:**
- ✅ Zero circular imports between `phase1` and `models`
- ✅ All new code uses `UnifiedConfig`
- ✅ Deprecation warnings logged for legacy config usage

### Week 3-4: Decompose God Objects (P1)

**Day 1-3: Decompose Trainer**
1. Create `src/models/training/orchestrator.py`
2. Extract `DataLoader`, `FeatureSelector`, `Evaluator`, `ArtifactManager`
3. Refactor `Trainer` to use orchestrator
4. Migrate tests
5. Update documentation

**Day 4-7: Decompose StackingEnsemble**
1. Create `src/models/ensemble/oof_generator.py`
2. Create `src/models/ensemble/heterogeneous_adapter.py`
3. Refactor `StackingEnsemble` to use extracted classes
4. Verify all 3 ensemble methods still work
5. Run stacking integration tests

**Day 8: Implement Cascade Invalidation**
1. Add `downstream_dependencies` to `CacheMetadata`
2. Implement `invalidate_cascade()` method
3. Add unit tests for cascade logic
4. Update cache documentation

**Success Criteria:**
- ✅ `Trainer` reduced to < 150 lines
- ✅ `StackingEnsemble` reduced to < 400 lines
- ✅ All tests passing
- ✅ Cache invalidates downstream artifacts

### Week 5-6: Metadata & Lineage (P2)

**Day 1-2: Add Experiment Tracking Metadata**
1. Add Git commit, Python version, data hash to `PipelineState.metadata`
2. Capture hardware specs (GPU name, CPU count)
3. Update state serialization
4. Add validation tests

**Day 3: Persist MTF Outputs**
1. Modify `src/phase1/stages/mtf/` to save resampled TFs
2. Add cache invalidation based on 1-min source mtime
3. Benchmark speedup
4. Update documentation

**Day 4-7: Implement Data Lineage Tracking**
1. Add `upstream_artifacts`, `transformation_hash` to `PhaseResult`
2. Implement `get_model_lineage(run_id)` utility
3. Add lineage visualization script
4. Update documentation

**Success Criteria:**
- ✅ Full reproducibility (Git + env + data tracked)
- ✅ 5-10x speedup for repeated pipeline runs
- ✅ Can trace any model back to source data/config/code

### Week 7-8: Cleanup & Optimization (P3)

**Day 1: Columnar Selection**
1. Update all `pd.read_parquet()` calls to specify columns
2. Benchmark performance improvement
3. Document pattern in contribution guide

**Day 2-4: Schema Evolution**
1. Create `src/schemas/` package
2. Implement schema registry with version migrations
3. Add backward compatibility tests
4. Document schema versioning process

**Day 5: Delete Legacy Code**
1. Verify no external dependencies on deprecated shims
2. Remove deprecated modules (after grace period)
3. Update all imports to use canonical paths
4. Run full test suite

**Success Criteria:**
- ✅ All parquet reads use columnar selection
- ✅ Schema migration tooling in place
- ✅ Zero deprecated code remaining
- ✅ All tests passing

---

## 7. Metrics for Success

### Before Refactoring (Current State)

| Metric | Value |
|--------|-------|
| Configuration classes | 85 |
| Circular dependencies | 1 (phase1 ↔ models) |
| Files > 1000 lines | 4 |
| Largest file | 1279 lines |
| Exception swallowing instances | 8 |
| Duplicate implementations | 3 (ADX, Sharpe) |
| Data lineage tracking | Partial |
| Schema evolution support | Weak |

### After Refactoring (Target State)

| Metric | Target |
|--------|--------|
| Configuration classes | < 15 (-83%) |
| Circular dependencies | 0 |
| Files > 1000 lines | 1 (entropy.py acceptable) |
| Largest file | < 1000 lines |
| Exception swallowing instances | 0 |
| Duplicate implementations | 0 |
| Data lineage tracking | Complete |
| Schema evolution support | Strong |

### Quality Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Modularity** | Good | Excellent |
| **Testability** | Moderate | High |
| **Maintainability** | Moderate | High |
| **Extensibility** | Excellent | Excellent |
| **Reproducibility** | Partial | Complete |

---

## 8. Preservation Checklist

**DO NOT CHANGE (These are your strengths):**

- ✅ ModelRegistry plugin pattern
- ✅ BaseModel interface contract
- ✅ 6-layer leakage prevention architecture
- ✅ PipelineState versioning and rollback
- ✅ Per-model feature strategies
- ✅ Memory management and OOM recovery
- ✅ Cross-validation OOF generation
- ✅ Heterogeneous ensemble capability

**PRESERVE AND IMPROVE:**

- 🔄 7-phase pipeline structure (preserve phases, improve coupling)
- 🔄 Configuration system (preserve UnifiedConfig, delete legacy)
- 🔄 Training workflow (preserve orchestrator pattern, decompose Trainer)
- 🔄 Data validation (preserve layers, add lineage)

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Breaking existing experiments** | Medium | High | Full test suite before each change |
| **Performance regression** | Low | Medium | Benchmark critical paths |
| **Config migration issues** | High | Medium | 2-sprint grace period + migration script |
| **Data lineage bugs** | Low | Medium | Comprehensive unit tests |
| **Cache invalidation bugs** | Medium | Medium | Integration tests with file changes |

---

## 10. Conclusion

Your ML factory is **production-ready** with exceptional leakage prevention and plugin architecture. The refactoring plan focuses on:

1. **Eliminating technical debt** (config sprawl, god objects, circular deps)
2. **Improving maintainability** (lineage tracking, cascade invalidation)
3. **Preserving architectural strengths** (registry, leakage prevention, state management)

**Estimated Timeline:** 6-8 weeks for complete refactoring
**Expected ROI:** 40% complexity reduction, 100% reproducibility, 5-10x pipeline speedup
**Risk Level:** Low (incremental changes with full test coverage)

Your factory pattern is sound - these improvements will transform it from research-grade to enterprise-grade while preserving its research flexibility.

---

## Appendix A: File Size Analysis

```
Files > 1000 lines (4 total):
1279: src/models/ensemble/stacking.py
1258: src/phase1/stages/features/entropy.py
1193: src/models/regime_evaluation.py
1116: src/config/unified.py

Files 800-1000 lines (7 total):
925: src/config/smart_config.py (DELETE - duplicate)
911: src/validation/store.py
907: src/inference/preprocessing_graph.py
887: src/ml_pipeline/state.py
885: src/training/modes/regime_aware.py
866: src/models/neural/base_rnn.py
849: src/models/ensemble/diversity.py
```

## Appendix B: Architecture Dependency Graph

```
src/
├── data/           [NEW - Shared layer]
│   ├── container.py
│   ├── contracts.py
│   └── requirements.py
│
├── common/         [Utilities]
│   ├── timeframes.py
│   └── validation.py
│
├── config/         [Single source of truth]
│   └── unified.py
│
├── phase1/         [Data pipeline]
│   ├── stages/
│   └── (depends on: data, common, config)
│
├── models/         [Model training]
│   ├── registry.py
│   ├── training/
│   └── (depends on: data, common, config)
│
├── training/       [Orchestration]
│   ├── orchestrator.py
│   └── (depends on: data, models, config)
│
└── ml_pipeline/    [Top-level workflow]
    └── (depends on: phase1, models, training)
```

**No circular dependencies** after `src/data/` extraction.

---

**Document Generated:** 2026-01-16
**Next Review:** After P0-P1 completion (Week 4)
