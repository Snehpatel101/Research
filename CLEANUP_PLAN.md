# Cleanup Plan: ML Factory

**Status:** Phase 23 In Progress
**Last Updated:** 2026-01-28

---

## Completed Phases Summary

| Phase | Description | Impact | Date |
|-------|-------------|--------|------|
| 0 | Deduplication | -5,336 lines removed | 2026-01-23 |
| 1 | Contract Enforcement | +616 lines, 7 new exceptions | 2026-01-23 |
| 2 | 4D Infrastructure | +958 lines, raw MTF store | 2026-01-24 |
| 3 | 5-Dimension Optuna | +2,298 lines, FeatureSpec artifact | 2026-01-24 |
| 4 | Validation Integration | +50 lines, leakage/lookahead wiring | 2026-01-24 |
| 5 | Unified Entry Point | +1,281 lines, MLFactory class | 2026-01-24 |
| 6 | Advanced Models | +3,690 lines, 6 new models | 2026-01-24 |
| 7-10 | Production Hardening | +1,525 lines, schemas, manifests | 2026-01-24 |
| 12 | Trading Profitability | +5,780 lines, Sharpe optimization, circuit breakers | 2026-01-24 |
| 12.5 | Code Quality Pass | Ruff 210→93, StageName enum | 2026-01-25 |
| 13 | Performance Optimization | MTF cache, batch inference | 2026-01-25 |
| 14 | Data Quality Hardening | Dynamic purge, NaN monitoring | 2026-01-25 |
| 15-18 | Ensemble & Resilience | +2,230 lines, meta-selection, checkpoints | 2026-01-25 |
| 19 | Comprehensive Optimization | +750 lines, 34 new features | 2026-01-25 |
| 20 | Performance & Quality Polish | -851 lines, 50-500x speedup | 2026-01-25 |
| 21 | ML Pipeline Review Fixes | 10 tasks, 3 disproven | 2026-01-27 |
| 22 | OPTIMIZE_FOR Metric Wiring | 7 changes, scoring.py | 2026-01-27 |

**Net Impact:** ~+12,010 lines | 196 features | 13 models | See **COMPLETION.md** for details.

---

## Phase 23: Critical Bugfixes, Validation & Performance (IN PROGRESS)

**Status:** IN PROGRESS | 2026-01-28
**Priority:** CRITICAL → HIGH → MEDIUM → LOW
**Source:** Runtime errors from Colab notebook + PERFORMANCE_FIXES.md analysis

---

### Phase 23 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ML FACTORY PIPELINE FLOW                                 │
│                                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────────────────┐  │
│  │  Raw     │───▶│ Feature  │───▶│  Labeling  │───▶│  Adapter Transform      │  │
│  │  OHLCV   │    │  Eng.    │    │  Stage     │    │  (2D/3D/4D)             │  │
│  └──────────┘    └────┬─────┘    └─────┬──────┘    └───────────┬─────────────┘  │
│                       │                │                        │                │
│                       │ 🔴 23C         │                        │ 🟡 23B-1      │
│                       │ PERFORMANCE    │                        │ VALIDATION    │
│                       │ (DataFrame     │                        │ TIMING        │
│                       │  fragmentation)│                        │               │
│                       ▼                ▼                        ▼               │
│                  ┌─────────────────────────────────────────────────────────┐    │
│                  │           Pre-Training Validation                        │    │
│                  │           (unified_orchestrator.py:501)                  │    │
│                  │                                                          │    │
│                  │   🔴 23A: Label in features → 100% accuracy → LEAKAGE   │    │
│                  │   🟡 23B-2: 218 features > model max (200/120/10)       │    │
│                  └───────────────────────────────────┬─────────────────────┘    │
│                                                      │                          │
│                                                      ▼                          │
│                                              ┌────────────┐                     │
│                                              │  Training  │                     │
│                                              │  (BLOCKED) │                     │
│                                              └────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────────┘

LEGEND:
  🔴 CRITICAL - Phase 23A (Label Leakage)
  🟡 HIGH     - Phase 23B (Validation Timing + Feature Count)
  🟠 MEDIUM   - Phase 23C (Performance Anti-Patterns)
  ⚪ LOW      - Phase 23D (Config Gaps - Deferred)
```

---

### Root Cause Analysis

| Issue Category | Root Cause | Symptom | Impact |
|----------------|------------|---------|--------|
| **Label Leakage** | `base.py:339` excludes `label_*` prefix but not bare `label` | 100% train accuracy | Models memorize labels, useless in production |
| **Validation Timing** | Validation at L501 runs on raw 2D before adapter at L579 | Rank mismatch errors for 3D/4D models | Cannot train TCN, PatchTST |
| **Feature Count** | Pipeline generates 218 features, no auto-selection | Contract violations: LightGBM(200), TCN(120), PatchTST(10) | Training blocked |
| **DataFrame Fragmentation** | 83+ individual `df[col] = ...` assignments | "DataFrame is highly fragmented" warnings | 5-20x slower feature generation |
| **sqrt Warning** | Negative values passed to np.sqrt | "invalid value encountered in sqrt" | NaN features, garbage predictions |
| **fillna Deprecation** | `fillna(method="bfill")` deprecated | FutureWarning logged | Will break in pandas 3.0 |

---

### Dependencies Between Fixes

```
Phase 23A (Label Leakage) ──────────────────────────────────────────┐
                                                                     │
Phase 23B-1 (Validation Timing) ────┬───▶ Phase 23B-2 (Feature     │
                                    │      Selection) ─────────────▶ │
                                    │                                │
                                    │                                ▼
                                    │                     TRAINING CAN START
                                    │
Phase 23C (Performance) ────────────┴─── Independent, can run in parallel
```

**Execution Order:**
1. **23A FIRST** - Without this fix, all trained models are garbage
2. **23B-1** - Must fix validation timing before feature selection helps
3. **23B-2** - Auto-select features to pass contract validation
4. **23C** - Performance can be addressed any time (no blocking)
5. **23D** - Deferred to Phase 24

---

## Phase 23A: Critical Label Leakage (PRIORITY: CRITICAL)

### The Bug

**Location:** `src/data/adapters/base.py:339-347`

```python
# CURRENT CODE (BUGGY)
exclude_prefixes = (
    "label_",           # ← Excludes "label_h5", "label_h15", etc.
    "sample_weight_",
    "regime_",
    ...
)
exclude_exact = {
    "open", "high", "low", "close", "volume",
    "bar_index", "session_id",
    # ← MISSING: "label"
}
```

### Why It Causes 100% Training Accuracy

When the label column is included as a feature:
- X = [feature_1, feature_2, ..., label] ← Label IS a feature
- y = label ← Label IS the target
- Model learns: f(X) = X[:, -1] ← Just read the label column!
- Result: Training accuracy 100%, production accuracy random

This is **catastrophic data leakage** - the model memorizes the answer from the input.

### The Exact Fix

Add `"label"` to the `exclude_exact` set in `base.py:339-347`.

---

## Phase 23B: Validation Timing & Feature Selection (PRIORITY: HIGH)

### Current vs Correct Validation Flow

```
CURRENT FLOW (BROKEN):
  df (raw 2D)
       │
       ▼
  _pre_training_validation(df)  ← L501: Validates 2D data
       │  ❌ FAILS: "Data rank mismatch: model expects 3D, data is 2D"
       ▼
  prepare(df, model_name)       ← L579: Adapter transforms (never reached)

CORRECT FLOW (FIXED):
  df (raw 2D)
       │
       ▼
  [1] Auto-select features based on model contract
       │
       ▼
  prepare(df, model_name)       ← Adapter transforms 2D→3D/4D
       │
       ▼
  _pre_training_validation(prepared_data)  ← Validate AFTER
       │
       ▼
  Training
```

### Contract Violation Details

| Model | Expected Rank | Max Features | Actual | Gap |
|-------|---------------|--------------|--------|-----|
| LightGBM | 2D | 200 | 218 | -18 |
| TCN | 3D | 120 | 218 | -98 |
| PatchTST | 4D | 10 | 218 | -208 |

### Fix Options

**Option 1 (Recommended):** Move validation after adapter transformation
**Option 2:** Skip rank validation on raw data, only check feature count

---

## Phase 23C: Feature Engineering Performance (PRIORITY: MEDIUM)

### Affected Files (from PERFORMANCE_FIXES.md)

| File | Priority | Individual Assigns | pd.Series() Wraps | Key Fix |
|------|----------|-------------------|-------------------|---------|
| `temporal.py` | **P0** | 13 | 0 | `.apply()` → `np.select()` |
| `momentum.py` | **P0** | 14 | 6 | Batch `pd.concat()` |
| `volatility.py` | **P0** | 20 | 4 | Batch `pd.concat()` |
| `volume.py` | **P1** | 18 | 1 | Remove temp columns |
| `microstructure.py` | **P1** | 17 | 1 | Replace loop with concat |
| `entropy.py` | **P2** | 1 | 5 | `np.concatenate()` shift |
| `wavelets.py` | **P2** | 10 | 10 | Batch loop outputs |
| `trend.py` | **P2** | 6 | 5 | Batch assignments |
| `regime.py` | **P3** | 5 | 0 | Minor batching |
| `price_features.py` | **P3** | 4 | 0 | Minor batching |

**Total: 83 individual assignments, 38 pd.Series() wrappers**

### Expected Speedup

| Fix | Current | Fixed | Speedup |
|-----|---------|-------|---------|
| `.apply()` → `np.select()` | 100ms/call | 1ms/call | **100x** |
| Loop → `pd.concat()` | O(n*k) copies | O(1) copies | **5-20x** |
| `pd.Series().shift()` → `np.concatenate()` | 2 allocs | 1 alloc | **2-5x** |

**Combined potential: 2-5x overall feature generation speedup**

### Architecture Pattern: FeatureBuilder

```python
class FeatureBuilder:
    def __init__(self, index): self.cols = {}
    def add(self, name, values): self.cols[name] = values; return self
    def to_frame(self): return pd.DataFrame(self.cols, index=self.index)
    def concat_to(self, df): return pd.concat([df, self.to_frame()], axis=1)
```

---

## Phase 23D: Config Gaps (PRIORITY: LOW - DEFERRED TO PHASE 24)

**Status:** DEFERRED | Will be addressed in Phase 24
**Reason:** System is functional without these; 23A-C fixes are blocking

---

### 23D Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION DEPLOYMENT GAPS                               │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  ExperimentConfig  │    │  BundleConfig   │    │  InferenceConfig │    │  Monitoring     │  │
│  │                │    │                │    │                │    │                │  │
│  │  ⚪ MTF mode   │    │  ⚪ Versioning  │    │  ⚪ Streaming  │    │  ⚪ Drift      │  │
│  │  ⚪ Feature    │    │  ⚪ Registry    │    │  ⚪ A/B test   │    │  ⚪ Alerts     │  │
│  │    selection  │    │  ⚪ Rollback    │    │  ⚪ Canary     │    │  ⚪ Retrain    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                  │
│  CURRENT STATE: All configs exist but lack production deployment features       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Gap Analysis

| Gap | Current State | Missing | Location | Priority |
|-----|---------------|---------|----------|----------|
| **MTF Mode** | MTFConfig has `mode` field | No ExperimentConfig exposure | `src/config/experiment.py` | LOW |
| **Feature Selection** | FeatureConfig has `selection_*` | No per-model override | `src/config/experiment.py` | LOW |
| **Compatibility Matrix** | MODEL_CONTRACTS has all data | No visual documentation | `docs/` | LOW |
| **Bundle Versioning** | BundleConfig has `version` | No version strategy/registry | `src/inference/` | MEDIUM |
| **A/B Testing** | None | Traffic split config | `src/config/inference.py` | MEDIUM |
| **Drift Detection** | None | Threshold/alert config | `src/config/monitoring.py` | MEDIUM |
| **Streaming Inference** | InferenceConfig has `mode` | Buffer management incomplete | `src/inference/` | LOW |

---

### Task 23D-1: MTF Mode in ExperimentConfig

**Current:** MTFConfig exists with full mode options, but ExperimentConfig doesn't expose it clearly.

```python
# src/config/data.py (EXISTS)
class MTFConfig:
    mode: str = "indicators"  # 'none', 'indicators', 'bars', 'both', 'multi_stream'
    timeframes: list[str] = ["5min", "15min", "1h"]

# src/config/experiment.py (MISSING clear exposure)
class ExperimentConfig:
    # mtf_mode not directly accessible
```

**Fix:** Add `mtf_mode` parameter to `ExperimentConfig.__init__()` that maps to `MTFConfig.mode`.

---

### Task 23D-2: Per-Model Feature Selection Override

**Current:** FeatureConfig has global selection settings.

```python
# src/config/data.py (EXISTS)
class FeatureConfig:
    selection_enabled: bool = True
    selection_method: str = "mda"
    selection_n_features: int = 50
```

**Missing:** Per-model overrides (e.g., PatchTST needs 10 features, LightGBM can use 200).

**Fix:** Add `model_feature_overrides: dict[str, int]` to ExperimentConfig.

---

### Task 23D-3: Bundle Registry & Versioning

**Current:** BundleConfig has version field but no registry.

```python
# src/inference/bundle.py (EXISTS)
@dataclass
class BundleConfig:
    version: str = "1.0.0"  # Just a string, no strategy
```

**Missing:**
- Version increment strategy (semver, timestamp, hash)
- Bundle registry (catalog of deployed bundles)
- Rollback support (link to previous version)

**Fix:** Create `src/inference/registry.py` with:
```python
class BundleRegistry:
    def register(bundle: ModelBundle) -> str: ...
    def get(bundle_id: str) -> ModelBundle: ...
    def rollback(bundle_id: str) -> ModelBundle: ...
    def list_versions(model_name: str) -> list[str]: ...
```

---

### Task 23D-4: A/B Testing Configuration

**Current:** No A/B testing support.

**Missing:** Traffic split configuration for comparing models.

**Fix:** Add to `src/config/inference.py`:
```python
@dataclass
class ABTestConfig:
    enabled: bool = False
    control_bundle_id: str = ""
    treatment_bundle_id: str = ""
    traffic_split: float = 0.5  # % to treatment
    metric: str = "sharpe_ratio"
    min_samples: int = 1000
    significance_level: float = 0.05
```

---

### Task 23D-5: Drift Detection Configuration

**Current:** No drift monitoring.

**Missing:** Feature/prediction drift thresholds and alerts.

**Fix:** Create `src/config/monitoring.py`:
```python
@dataclass
class DriftConfig:
    enabled: bool = True
    feature_drift_threshold: float = 0.1  # PSI threshold
    prediction_drift_threshold: float = 0.15
    check_interval_hours: int = 24
    alert_channels: list[str] = ["log"]  # log, email, slack
    auto_retrain_trigger: bool = False
```

---

### Task 23D-6: Streaming Inference Buffer

**Current:** InferenceConfig has `mode="streaming"` but implementation incomplete.

```python
# src/config/inference.py (EXISTS)
class InferenceConfig:
    mode: str = "single"  # single, batch, streaming
```

**Missing:** Buffer management for streaming mode.

**Fix:** Add streaming config:
```python
@dataclass
class StreamingConfig:
    buffer_size: int = 1000
    flush_interval_seconds: float = 1.0
    max_latency_ms: float = 100.0
    backpressure_strategy: str = "drop"  # drop, block, sample
```

---

### Task 23D-7: Compatibility Matrix Documentation

**Current:** MODEL_CONTRACTS contains all compatibility data but not documented visually.

**Fix:** Generate `docs/COMPATIBILITY.md` from MODEL_CONTRACTS:

```markdown
## Model Compatibility Matrix

| Model | Adapter | Rank | MTF Mode | Feature Mode | Max Features |
|-------|---------|------|----------|--------------|--------------|
| xgboost | tabular | 2D | indicators | engineered | 200 |
| lightgbm | tabular | 2D | indicators | engineered | 200 |
| tcn | sequence | 3D | none | engineered | 120 |
| patchtst | multi_stream | 4D | multi_stream | raw | 10 |
...
```

---

### Why Deferred

| Gap | Deferral Reason |
|-----|-----------------|
| MTF Mode | Current default works for most cases |
| Feature Selection | Auto-selection in 23B-2 covers 80% |
| Bundle Registry | Manual versioning works for now |
| A/B Testing | Can do manually with separate runs |
| Drift Detection | External tools (MLflow, Evidently) available |
| Streaming | Batch mode sufficient for current use |
| Compatibility Matrix | Users can check MODEL_CONTRACTS |

**Blocking Issues (23A-C) take priority.** Config gaps don't prevent training.

---

## Validation Criteria

### Phase 23A
```bash
grep -n '"label"' src/data/adapters/base.py | grep exclude_exact
# Training accuracy should be 40-70%, NOT 100%
```

### Phase 23B
```bash
# Validation runs after adapter - no rank mismatch errors
# Feature count <= contract.max_features
```

### Phase 23C
```bash
# No PerformanceWarning in logs
# No "invalid value encountered in sqrt"
# No "fillna with method is deprecated"
```

---

## Lessons Learned

1. **Column exclusion must be exhaustive** - prefix matching misses bare column names
2. **Validation timing matters** - validate actual training data, not raw input
3. **Contract max_features are real limits** - models can't accept more features
4. **DataFrame fragmentation is real** - 83 individual assignments = 5-20x slowdown
5. **Runtime warnings predict bugs** - don't ignore sqrt/fillna warnings

---

## Summary

| Category | Priority | Impact | Effort | Tasks |
|----------|----------|--------|--------|-------|
| **23A: Label Leakage** | CRITICAL | Training unusable | 1 line | 1 |
| **23B: Validation** | HIGH | Training blocked | ~50 lines | 2 |
| **23C: Performance** | MEDIUM | 2-5x speedup | ~200 lines | 10 |
| **23D: Config** | LOW | Production features | ~500 lines | 7 (deferred) |

**Total Phase 23:** 13 active tasks + 7 deferred = 20 tasks

---

*See CLEANUP_TASKS.md for specific file:line tasks*
*See COMPLETION.md for implementation details after completion*
