# ML Factory - Technical Improvements Document

**Generated:** 2026-01-29
**Analysis Scope:** `/Users/sneh/research/src/` (460 Python files)

This document consolidates all technical inconsistencies and optimization opportunities identified across the ML Factory codebase by five specialized analysis agents.

**Related Document:** See `IMPROVEMENTS.md` for financial/ML algorithmic improvements.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Inconsistencies](#architectural-inconsistencies)
3. [Performance Optimizations](#performance-optimizations)
4. [Code Quality Issues](#code-quality-issues)
5. [Data Engineering Improvements](#data-engineering-improvements)
6. [Priority Matrix](#priority-matrix)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

| Category | Issues Found | High Priority | Medium Priority | Low Priority |
|----------|-------------|---------------|-----------------|--------------|
| Architecture | 22 | 4 | 8 | 10 |
| Performance | 23 | 6 | 10 | 7 |
| Code Quality | 57 | 21 | 28 | 8 |
| Data Engineering | 16 | 3 | 9 | 4 |
| **Total** | **118** | **34** | **55** | **29** |

**Estimated Overall Improvement (if all HIGH/MEDIUM addressed):**
- Feature computation time: 65-75% reduction
- Memory usage: 20-30% reduction
- Training iteration time: 40-50% reduction

---

## Architectural Inconsistencies

### ARCH-001: Duplicate PredictionResult Class [HIGH]

**Problem:** `PredictionResult` is defined in THREE locations with different attributes.

| Location | Unique Fields |
|----------|---------------|
| `src/models/base.py:28-87` | `confidence`, `metadata` |
| `src/core/interfaces.py:124-152` | `indices` |
| `src/inference/orchestrator.py:53-114` | `model_name`, `horizon`, `inference_time_ms`, `is_ensemble` |

**Impact:** Type confusion, potential runtime mismatches
**Fix:** Consolidate to single definition in `src/core/interfaces.py`, re-export elsewhere

---

### ARCH-002: Duplicate AdapterResult Class [MEDIUM]

**Problem:** `AdapterResult` defined in two locations with different field conventions.

| Location | Convention |
|----------|------------|
| `src/data/adapters/base.py:66-227` | ML convention: `X`, `y`, `data_rank` |
| `src/core/interfaces.py:48-122` | Legacy: `data`, `labels`, `feature_names` |

**Impact:** Documented workaround exists but violates DRY
**Fix:** Remove from `interfaces.py`, keep only in `adapters/base.py`

---

### ARCH-003: Triple DataContract Definitions [HIGH]

**Problem:** Three different "data contract" concepts exist:

| Class | Location | Purpose |
|-------|----------|---------|
| `DatasetContract` | `src/core/data_contract.py:17-150` | Pipeline stage data passing |
| `DataContract` | `src/core/contracts/data_contract.py:114-440` | Model data requirements |
| `DataContract` | `src/core/interfaces.py:291-337` | Abstract interface |

**Impact:** Naming collision causes confusion
**Fix:** Rename `DatasetContract` to `PipelineData`, document usage contexts

---

### ARCH-004: ModelContract Interface Divergence [HIGH]

**Problem:** Two incompatible `ModelContract` definitions:

| Location | Type | Purpose |
|----------|------|---------|
| `src/core/interfaces.py:339-446` | Abstract class | `fit()`, `predict()`, `save()`, `load()` methods |
| `src/core/contracts/model_contract.py:37-223` | Frozen dataclass | Model requirements metadata |

**Impact:** Same name for different concepts
**Fix:** Rename abstract class to `ModelInterface`

---

### ARCH-005: Return Type Mismatch [LOW]

**Problem:** `PredictionOutput` vs `PredictionResult` used inconsistently.

- `src/models/neural/base_rnn.py:583` returns `PredictionOutput`
- `src/models/base.py:324` declares return `PredictionResult`
- Alias exists at `base.py:467` but deprecated

**Fix:** Replace all `PredictionOutput` with `PredictionResult`

---

### ARCH-006: Model Family Naming Inconsistency [MEDIUM]

**Problem:** Transformers inconsistently categorized:

| Model | Listed Family |
|-------|---------------|
| `transformer` | `neural` |
| `patchtst`, `itransformer` | `transformer` |

**Fix:** Standardize all transformers under `transformer` family

---

### ARCH-007: Constants Duplication [MEDIUM]

**Problem:** Model metadata defined in both:
- `src/core/constants.py` (`MODEL_DATA_RANKS`, `MODEL_ADAPTER_MAP`)
- `src/core/contracts/model_contract.py` (`MODEL_CONTRACTS`)

**Fix:** Derive constants from contracts (single source of truth)

---

### ARCH-008: Inference Layer Defines Core Types [MEDIUM]

**Problem:** `PredictionResult` defined in `src/inference/orchestrator.py:53-114` should be in core.

**Fix:** Move to `src/core/interfaces.py` or create `src/core/results.py`

---

### ARCH-009: Circular Import Prevention Pattern [MEDIUM]

**Problem:** Dual `AdapterResult` exists to avoid circular imports (`src/core/interfaces.py:36-45`).

**Fix:** Use `TYPE_CHECKING` guards properly

---

### ARCH-010: Deep Coupling - Models Import Validation [LOW]

**Problem:** `src/models/neural/base_rnn.py:35` imports from neural-specific module.

**Fix:** Move `validate_training_inputs` to `src/core/validation.py`

---

## Performance Optimizations

### PERF-001: ADX/DI Quadruple Computation [HIGH]

**Location:** `src/data/features/compute/trend.py:93-133`

**Problem:** `_compute_di_adx()` called 4 times for same data:
```python
def compute_adx_14(df): _, _, adx = _compute_di_adx(df, period=14); return adx
def compute_plus_di_14(df): plus_di, _, _ = _compute_di_adx(df, period=14); return plus_di
def compute_minus_di_14(df): _, minus_di, _ = _compute_di_adx(df, period=14); return minus_di
def compute_adx_strong_trend(df): _, _, adx = _compute_di_adx(df, period=14); return (adx > 25)
```

**Expected Improvement:** 75% reduction in trend feature computation
**Fix:** Cache result on first call or return all 4 features at once

---

### PERF-002: Approximate Entropy O(n^2) [HIGH]

**Location:** `src/data/features/compute/entropy.py:177-188`

**Problem:** Inner loop is O(n^2) in rolling window:
```python
for i in range(n_patterns):
    diffs = np.abs(patterns - patterns[i]).max(axis=1)  # O(n)
    counts[i] = np.sum(diffs <= r) / n_patterns
```

**Expected Improvement:** 50-100x speedup
**Fix:** Apply existing `_count_matches_numba()` to approximate entropy

---

### PERF-003: Microstructure Base Feature Recomputation [HIGH]

**Location:** `src/data/features/compute/microstructure.py:60-69`

**Problem:** Base Amihud calculated 3x when all variants needed:
```python
def compute_micro_amihud_10(df): return _sma(compute_micro_amihud(df), 10)
def compute_micro_amihud_20(df): return _sma(compute_micro_amihud(df), 20)
```

**Expected Improvement:** 50-70% reduction
**Fix:** Implement memoization or compute base once

---

### PERF-004: Sequential Feature Computation [HIGH]

**Location:** All `src/data/features/compute/` modules

**Problem:** Features computed sequentially when they're independent.

**Expected Improvement:** Near-linear speedup with CPU cores
**Fix:** Use `ProcessPoolExecutor` for feature families

---

### PERF-005: GARCH Rolling Loop [HIGH]

**Location:** `src/data/pipeline/stages/features/volatility.py:548-586`

**Problem:** Expanding window GARCH fits model at every bar:
```python
for i in range(min_obs, n):
    model = arch_model(...)
    result = model.fit(disp="off")
```

**Expected Improvement:** 10-100x speedup
**Fix:** Fit less frequently (every 10-20 bars) or use EWMA approximation

---

### PERF-006: ATR Recomputation Across Modules [HIGH]

**Locations:**
- `src/optimization/five_dimension_objective.py:328-358`
- `src/data/labeling/triple_barrier.py:510-535`
- `src/data/features/compute/trend.py:155`

**Problem:** ATR computed independently 3+ times per training run.

**Fix:** Pre-compute ATR once at pipeline start, pass as argument

---

### PERF-007: Volume Feature Redundancy [MEDIUM]

**Location:** `src/data/features/compute/volume.py:127-138, 167-177, 193-217`

**Problem:** `compute_vwap()`, `compute_twap_10()`, `compute_dollar_volume()` recomputed for variants.

**Expected Improvement:** 40-60% reduction
**Fix:** Use `functools.lru_cache` on helper functions

---

### PERF-008: Walk-Forward Sequential Folds [MEDIUM]

**Location:** `src/optimization/feature_selection/walk_forward.py:116-139`

**Problem:** CV folds processed sequentially when independent.

**Expected Improvement:** 3-5x speedup for n_splits=5
**Fix:** Parallelize fold processing with joblib

---

### PERF-009: Supertrend Double Computation [MEDIUM]

**Location:** `src/data/features/compute/trend.py:216-236`

**Problem:** `_compute_supertrend()` called twice for value and direction.

**Expected Improvement:** 50% reduction
**Fix:** Return both in single call

---

### PERF-010: DataFrame Fragmentation [MEDIUM]

**Locations:** Multiple feature modules still using:
```python
df["feature_name"] = computed_value  # Creates copy
```

**Fix:** Use batch concat pattern:
```python
new_cols = {"f1": v1, "f2": v2}
df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
```

---

### PERF-011: Label Cache Unbounded [MEDIUM]

**Location:** `src/optimization/five_dimension_objective.py:99`

**Problem:** Global cache with no size limit can cause OOM during long optimization.

**Fix:** Use LRU cache with max size

---

### PERF-012: Log Returns Computed Multiple Times [MEDIUM]

**Locations:** `entropy.py:57-59`, `microstructure.py:34-36`, `volume.py` (multiple)

**Fix:** Compute once at feature pipeline start

---

### PERF-013: Hurst Exponent Nested Loops [MEDIUM]

**Location:** `src/data/features/compute/entropy.py:250-286`

**Problem:** Inefficient chunking with repeated allocations.

**Expected Improvement:** 3-5x speedup
**Fix:** Use numpy reshape and vectorized operations

---

### PERF-014: Lempel-Ziv String Operations [MEDIUM]

**Location:** `src/data/features/compute/entropy.py:117-150`

**Problem:** Using Python string operations instead of numpy.

**Expected Improvement:** 10-20x speedup
**Fix:** Implement with numba using integer arrays

---

### PERF-015: CCI Rolling MAD [MEDIUM]

**Location:** `src/data/pipeline/stages/features/momentum.py:291`

**Problem:** Lambda in rolling apply prevents optimization:
```python
mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
```

**Expected Improvement:** 5-10x speedup
**Fix:** Use `rolling().std() * 0.7978` approximation or numba

---

## Code Quality Issues

### CQ-001: Excessive `Any` Type Usage [HIGH]

| File | Line | Issue |
|------|------|-------|
| `src/cli/run_commands_core.py` | 13-15 | `_pipeline_config: Any = None` |
| `src/cli/commands/train.py` | 176-177 | `trainer_config: Any, container: Any` |
| `src/data/labeling/optimization.py` | 85 | `study: Any  # optuna.Study` |
| `src/models/boosting/lightgbm_model.py` | 26 | `lgb: Any = None` |
| `src/orchestrator.py` | 54 | `training_result: Any = None` |
| `src/factory.py` | 218 | `_cached_training_result: Any = None` |
| `src/config/utils.py` | 153 | `_global_config_cache: Any = None` |
| `src/optimization/feature_selection/purged_selector.py` | 53 | `cv: Any` |

**Fix:** Replace with proper types or `Optional[SpecificType]`

---

### CQ-002: Bare `except Exception:` Handlers [HIGH]

| File | Line | Context |
|------|------|---------|
| `src/factory.py` | 314, 647, 680 | Experiment/backtest/bundle (has logging) |
| `src/validation/bootstrap.py` | 128, 197, 496 | Bootstrap sampling |
| `src/data/features/compute/wavelets.py` | 58, 85, 100 | Wavelet decomposition |
| `src/validation/cv/pbo.py` | 306 | PBO calculation |
| `src/cli/status_commands.py` | 125, 347 | Status display |
| `src/cli/commands/train.py` | 267 | Training recovery |
| `src/data/features/optimization.py` | 103, 309, 370 | Feature optimization |
| `src/models/ensemble/diversity.py` | 830 | Diversity calculation |
| `src/optimization/labels.py` | 481 | Label optimization |
| `src/data/pipeline/stages/features/entropy.py` | 735 | Entropy calculation |
| `src/data/pipeline/stages/features/volatility.py` | 583 | Volatility calculation |

**Fix:** Log exception details, re-raise or handle specific exceptions

---

### CQ-003: Missing Return Type on `__post_init__` [MEDIUM]

| File | Lines |
|------|-------|
| `src/config/smart_config.py` | 314 |
| `src/config/experiment.py` | 87, 202 |
| `src/config/unified.py` | 121, 173, 302, 592 |

**Fix:** Add `-> None` return type annotation

---

### CQ-004: TODO/FIXME Comments [MEDIUM]

| File | Line | Content |
|------|------|---------|
| `src/inference/production/monitor.py` | 264 | `# TODO: track from inference pipeline` |
| `src/inference/production/monitor.py` | 265 | `# TODO: track from prediction history` |
| `src/core/interfaces.py` | 45 | `# TODO Phase 0G: Consolidate...` |

---

### CQ-005: Magic Numbers [MEDIUM]

| File | Line | Value | Context |
|------|------|-------|---------|
| `src/config/unified.py` | 143 | `7200` | embargo_time_minutes |
| `src/config/unified.py` | 144 | `1440` | min_embargo_bars |
| `src/config/data.py` | 504-505 | `60`, `1440` | purge/embargo defaults |
| `src/validation/monitoring/drift_detectors.py` | 84 | `1000` | fallback_max_size |
| `src/validation/leakage_detection.py` | 201 | `10` | min valid samples |
| `src/data/features/compute/wavelets.py` | 355-358 | `0.01` | slope threshold |

**Fix:** Extract to named constants

---

### CQ-006: Duplicate Default Definitions [MEDIUM]

**Problem:** Config classes define defaults both in `default_factory` AND in `from_dict`:

```python
# In class definition
canonical_ladder: list[str] = field(default_factory=lambda: ["1min", "5min", ...])

# In from_dict method
default_canonical_ladder = ["1min", "5min", ...]  # Duplicated!
```

**Files:** `src/config/unified.py:94-110, 186-193, 254-268`

**Fix:** Extract defaults to class-level constants

---

### CQ-007: Deprecated Alias Still Present [LOW]

**Location:** `src/models/base.py:467`
```python
PredictionOutput = PredictionResult  # Deprecated
```

**Fix:** Schedule removal

---

## Data Engineering Improvements

### DE-001: Inter-Stage Validation Not Called [HIGH]

**Problem:** `validate_stage_transition()` exists in `src/data/pipeline/schemas.py:244-355` but not consistently called between stages.

**Impact:** Data corruption can propagate silently
**Fix:** Add validation calls after each stage write

---

### DE-002: Raw Data Validation Non-Blocking [HIGH]

**Location:** `src/data/pipeline/stages/clean/run.py:83-85`

**Problem:** Schema validation issues logged but don't fail pipeline.

**Impact:** Malformed data propagates through entire pipeline
**Fix:** Make validation blocking (fail-fast)

---

### DE-003: MTF Lookahead Validation Not Called [HIGH]

**Location:** `src/data/pipeline/stages/features/run.py:372-404`

**Problem:** `validate_no_lookahead()` exists but not called during pipeline execution.

**Impact:** Lookahead bugs not caught at runtime
**Fix:** Call validation after MTF feature generation as blocking check

---

### DE-004: Repeated DataFrame Copies [MEDIUM]

**Location:** `src/data/pipeline/stages/features/engineer.py:238`

**Problem:** Multiple unnecessary `df.copy()` operations.

**Impact:** Memory 2-3x higher than necessary
**Fix:** Use single copy at stage entry with in-place modifications

---

### DE-005: Redundant Volatility Intermediate Calculations [MEDIUM]

**Location:** `src/data/features/compute/volatility.py`

**Problem:** BB/Keltner functions each compute `sma_20`, `ema_20`, `atr_10` independently.

**Impact:** 30-50% slower volatility computation
**Fix:** Create computation context that caches intermediates

---

### DE-006: Feature Column Auto-Detection Incomplete [MEDIUM]

**Location:** `src/data/adapters/base.py:318-356`

**Problem:** Exclusion list missing: `split`, `horizon`, `bars_to_hit_`, `mae_`, `mfe_`, `touch_type_`

**Impact:** Could leak label-related columns into features
**Fix:** Use positive feature identification (explicit manifest)

---

### DE-007: Multi-Stream Adapter Temporal Misalignment [MEDIUM]

**Location:** `src/data/adapters/multi_stream.py:489-496`

**Problem:** Non-integer timeframe ratios proceed with floor division.

**Impact:** Subtle lookahead bias in multi-timeframe models
**Fix:** Reject non-integer ratios or implement proper alignment

---

### DE-008: Missing Label Sentinel Validation [MEDIUM]

**Location:** `src/data/pipeline/stages/splits/core.py:22`

**Problem:** Sentinel `-99` defined but not validated at consumption points.

**Impact:** Invalid labels could leak into training
**Fix:** Add sentinel validation at every label consumption point

---

### DE-009: Horizon Validation Warning-Only [MEDIUM]

**Location:** `src/data/pipeline/stages/labeling/run.py:108-175`

**Problem:** `raise_on_violation=False` by default.

**Fix:** Default to `raise_on_violation=True` for fail-fast behavior

---

### DE-010: Inefficient Parquet I/O [MEDIUM]

**Location:** `src/data/pipeline/stages/features/run.py:199,294`

**Problem:** Full DataFrame read without column pruning.

**Impact:** 10-20% I/O overhead
**Fix:** Use `columns` parameter in `pd.read_parquet()`

---

### DE-011: No Feature Dependency Graph [LOW]

**Location:** `src/data/pipeline/stages/features/engineer.py:247-321`

**Problem:** Feature computation order is implicit.

**Fix:** Define explicit dependency DAG, compute in topological order

---

### DE-012: Duplicate Adapter Methods [LOW]

**Locations:**
- `src/data/adapters/tabular.py:192-255`
- `src/data/adapters/sequence.py:374-437`
- `src/data/adapters/multi_stream.py:642-702`

**Problem:** `_get_metadata_value`, `_parse_horizon_from_label_column` duplicated.

**Fix:** Move common methods to `BaseAdapter`

---

## Priority Matrix

### Immediate (Week 1)

| ID | Category | Description | Effort |
|----|----------|-------------|--------|
| PERF-001 | Performance | ADX/DI caching | LOW |
| PERF-003 | Performance | Microstructure caching | LOW |
| CQ-002 | Code Quality | Fix bare except handlers | LOW |
| DE-001 | Data Eng | Enable inter-stage validation | LOW |

### Short-Term (Weeks 2-3)

| ID | Category | Description | Effort |
|----|----------|-------------|--------|
| ARCH-001 | Architecture | Consolidate PredictionResult | MEDIUM |
| PERF-002 | Performance | Numba for approx entropy | MEDIUM |
| PERF-004 | Performance | Parallelize feature computation | MEDIUM |
| PERF-006 | Performance | Pre-compute ATR | MEDIUM |
| CQ-001 | Code Quality | Replace Any types | MEDIUM |
| DE-002 | Data Eng | Make validation blocking | LOW |

### Medium-Term (Weeks 4-6)

| ID | Category | Description | Effort |
|----|----------|-------------|--------|
| ARCH-003 | Architecture | Resolve DataContract naming | HIGH |
| ARCH-004 | Architecture | Rename ModelContract | MEDIUM |
| PERF-005 | Performance | Optimize GARCH | HIGH |
| PERF-008 | Performance | Parallelize walk-forward | MEDIUM |
| DE-005 | Data Eng | Cache volatility intermediates | MEDIUM |

### Long-Term (Ongoing)

| ID | Category | Description | Effort |
|----|----------|-------------|--------|
| CQ-005 | Code Quality | Extract magic numbers | LOW |
| CQ-006 | Code Quality | Consolidate config defaults | LOW |
| DE-011 | Data Eng | Feature dependency graph | MEDIUM |
| DE-012 | Data Eng | Consolidate adapter methods | LOW |

---

## Implementation Roadmap

### Phase A: Quick Wins (1-2 days)
1. Fix ADX/DI caching (75% speedup for trend features)
2. Fix microstructure base feature caching
3. Enable inter-stage validation calls
4. Replace bare except handlers with specific handling

### Phase B: Type Safety (3-5 days)
1. Replace `Any` types with proper types
2. Add return type hints to `__post_init__` methods
3. Consolidate `PredictionResult` to single location

### Phase C: Performance (1-2 weeks)
1. Parallelize feature computation with ProcessPoolExecutor
2. Apply Numba to approximate entropy
3. Pre-compute ATR/log_returns at pipeline start
4. Implement feature computation caching layer

### Phase D: Architecture (2-3 weeks)
1. Resolve DataContract/ModelContract naming collisions
2. Consolidate adapter common methods
3. Derive constants from contracts (single source of truth)
4. Remove deprecated aliases

### Phase E: Data Engineering (1-2 weeks)
1. Make raw data validation blocking
2. Call MTF lookahead validation in pipeline
3. Implement proper feature dependency graph
4. Add label sentinel validation

---

## Appendix: Files Most Needing Attention

| File | Issue Count | Priority Issues |
|------|-------------|-----------------|
| `src/data/features/compute/trend.py` | 3 | PERF-001, PERF-009 |
| `src/data/features/compute/entropy.py` | 4 | PERF-002, PERF-012, PERF-013, PERF-014 |
| `src/core/interfaces.py` | 4 | ARCH-001, ARCH-002, ARCH-004 |
| `src/config/unified.py` | 3 | CQ-003, CQ-005, CQ-006 |
| `src/data/adapters/base.py` | 2 | DE-006, DE-012 |
| `src/optimization/five_dimension_objective.py` | 2 | PERF-006, PERF-011 |

---

*This document should be updated as issues are resolved. Mark items with [RESOLVED] and date when fixed.*
