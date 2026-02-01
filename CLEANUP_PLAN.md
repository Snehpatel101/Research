# Cleanup Plan: ML Factory

**Status:** Phase 31 Complete, Phases 32-34 Planned
**Last Updated:** 2026-02-01

---

## Completed Phases (24-31)

See **COMPLETION.md** for full details on all completed phases.

| Phase | Description | Status | Completed |
|-------|-------------|--------|-----------|
| 24 | Feature Computation Caching (ADX/DI, microstructure, supertrend) | ✅ COMPLETE | 2026-01-29 |
| 25 | Data Validation Hardening (fail-fast validation) | ✅ COMPLETE | 2026-01-29 |
| 26 | Type Safety & Code Quality (Any types, return annotations) | ✅ COMPLETE | 2026-01-29 |
| 27 | Architecture Consolidation (class deduplication) | ✅ COMPLETE | 2026-01-29 |
| 28 | Compute Performance (numba, parallelization, GARCH, caching) | ✅ COMPLETE | 2026-01-30 |
| 29 | Memory Performance (cache bounds, log_returns consolidation) | ✅ COMPLETE | 2026-01-29 |
| 30 | Advanced Architecture (transformer family, derived constants, SMA/EMA/STD caching) | ✅ COMPLETE | 2026-01-30 |
| 31 | Code Polish (TODOs, constants, adapters, feature DAG) | ✅ COMPLETE | 2026-01-31 |

**Summary Impact:** 8 phases complete, 35+ files modified, ~400 lines net change, 50-100x speedup on key operations, fail-fast validation, single-definition principle enforced.

---

## Phase Summary

| Phase | Focus | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| 24-31 | See above | VARIOUS | 8 days | ✅ ALL COMPLETE - See COMPLETION.md |
| 32 | Critical Fixes | CRITICAL | 3-5 days | 📋 PLANNED (12 tasks - model families, data leakage, numerical) |
| 33 | Perf & Arch | HIGH | 4-6 days | 📋 PLANNED (11 tasks - evaluators, layer violations, optimizations) |
| 34 | Cleanup | MEDIUM | 2-3 days | 📋 PLANNED (11 tasks - orphaned files, MTF, fragmentation) |

---

## Phase 32: Critical Fixes

**Status:** NOT STARTED
**Priority:** CRITICAL
**Effort:** 3-5 days
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CRITICAL PRODUCTION ISSUES                               │
│                                                                                  │
│  MODEL FAMILY MISMATCHES (Contract vs Registration):                            │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  patchtst:       contract=transformer  →  registration=neural  │             │
│  │  itransformer:   contract=transformer  →  registration=neural  │             │
│  │  ridge_meta:     contract=meta_learner →  registration=ensemble│             │
│  │  mlp_meta:       contract=meta_learner →  registration=ensemble│             │
│  │  xgboost_meta:   contract=meta_learner →  registration=ensemble│             │
│  │  calibrated_meta: contract=meta_learner → registration=ensemble│             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  DATA LEAKAGE VIA train_test_split:                                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  optimization/features.py:320        - shuffle=True            │             │
│  │  optimization/hyperparameters.py:616 - shuffle=True            │             │
│  │  optimization/pipeline.py:401        - shuffle=True            │             │
│  │  cli/commands/train.py:583           - shuffle=True            │             │
│  │  → All leak future data into training via random splits        │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  NUMERICAL ISSUES (Gradient Explosion):                                         │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  liquidity.py:95    - Division by zero returns 1e10            │             │
│  │  mean_reversion.py:127 - Returns np.inf (gradient explosion)   │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File:Line | Priority | Description |
|------|-----------|----------|-------------|
| 32-1 | `neural/patchtst_model.py:240-243` | CRITICAL | Change registration from `neural` to `transformer` |
| 32-2 | `neural/itransformer_model.py:258-261` | CRITICAL | Change registration from `neural` to `transformer` |
| 32-3 | `ensemble/ridge_meta.py:26-29` | CRITICAL | Change registration from `ensemble` to `meta_learner` |
| 32-4 | `ensemble/mlp_meta.py:25-28` | CRITICAL | Change registration from `ensemble` to `meta_learner` |
| 32-5 | `ensemble/xgboost_meta.py:22-25` | CRITICAL | Change registration from `ensemble` to `meta_learner` |
| 32-6 | `ensemble/calibrated_meta.py:26-29` | CRITICAL | Change registration from `ensemble` to `meta_learner` |
| 32-7 | `optimization/features.py:320` | CRITICAL | Replace train_test_split with time-based split |
| 32-8 | `optimization/hyperparameters.py:616` | CRITICAL | Replace train_test_split with time-based split |
| 32-9 | `optimization/pipeline.py:401` | CRITICAL | Replace train_test_split with time-based split |
| 32-10 | `cli/commands/train.py:583` | CRITICAL | Replace train_test_split with time-based split |
| 32-11 | `features/compute/liquidity.py:95` | HIGH | Return 0 or np.nan instead of 1e10 |
| 32-12 | `features/compute/mean_reversion.py:127` | HIGH | Clip to large finite value instead of np.inf |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Model family mismatches | 6 models | 0 | `grep -r "register_model" src/models/` verify all match contracts |
| train_test_split usage | 4 locations | 0 | `grep -r "train_test_split.*shuffle" src/` returns 0 |
| Infinite values in features | 2 locations | 0 | Run feature pipeline, check for inf/1e10 values |
| **Data integrity** | VIOLATED | **ENFORCED** | All time-series splits preserve temporal order |

---

## Phase 33: Performance & Architecture

**Status:** NOT STARTED
**Priority:** HIGH
**Effort:** 4-6 days
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE & ARCHITECTURE GAPS                               │
│                                                                                  │
│  INCOMPLETE IMPLEMENTATIONS (NotImplementedError):                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  validation/evaluation/cpcv_pbo_evaluator.py:52                │             │
│  │  validation/evaluation/cv_evaluator.py:51                      │             │
│  │  validation/evaluation/walk_forward_evaluator.py:51            │             │
│  │  → Three evaluator classes not implemented                     │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  LAYER VIOLATIONS (Core → Data):                                                │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/container.py:673 - imports MultiResolution4DAdapter      │             │
│  │  core/container.py:739 - imports MultiStreamAdapter            │             │
│  │  → Core layer should not depend on data layer                  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  PERFORMANCE OPTIMIZATIONS REMAINING:                                            │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  CCI vectorization:       5-10x speedup                        │             │
│  │  Variance ratio:          10-20x speedup                       │             │
│  │  Order flow caching:      3-4x speedup                         │             │
│  │  Regime caching:          3x speedup                           │             │
│  │  Wavelet numba:           10-50x speedup                       │             │
│  │  Hurst O(n) algorithm:    5-10x speedup                        │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File:Line | Priority | Description | Speedup |
|------|-----------|----------|-------------|---------|
| 33-1 | `validation/evaluation/cpcv_pbo_evaluator.py:52` | HIGH | Implement CPCV-PBO evaluator | N/A |
| 33-2 | `validation/evaluation/cv_evaluator.py:51` | HIGH | Implement CV evaluator | N/A |
| 33-3 | `validation/evaluation/walk_forward_evaluator.py:51` | HIGH | Implement walk-forward evaluator | N/A |
| 33-4 | `core/container.py:673` | HIGH | Remove MultiResolution4DAdapter import | N/A |
| 33-5 | `core/container.py:739` | HIGH | Remove MultiStreamAdapter import | N/A |
| 33-6 | `features/compute/momentum.py:322-341` | MEDIUM | Vectorize CCI computation | 5-10x |
| 33-7 | `features/compute/mean_reversion.py:250-300` | MEDIUM | Vectorize variance ratio | 10-20x |
| 33-8 | `features/compute/order_flow.py:53-103` | MEDIUM | Add caching to order flow features | 3-4x |
| 33-9 | `features/compute/regime.py:53-86,120-135` | MEDIUM | Add caching to regime features | 3x |
| 33-10 | `features/compute/wavelets.py:62-88` | MEDIUM | Apply numba to wavelet transform | 10-50x |
| 33-11 | `features/compute/mean_reversion.py:156-200` | MEDIUM | Replace Hurst with O(n) algorithm | 5-10x |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| NotImplementedError count | 3 | 0 | Run all validation tests |
| Core → Data imports | 2 | 0 | `grep "from src.data" src/core/` returns 0 |
| CCI computation time | Baseline | 10-20% | Profile CCI features |
| Variance ratio time | Baseline | 5-10% | Profile mean reversion |
| Order flow time | Baseline | 25-30% | Profile with caching |
| Wavelet time | Baseline | 2-5% | Profile with numba |
| **Overall pipeline speedup** | 100% | **60-70%** | Full pipeline benchmark |

---

## Phase 34: Cleanup & Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM
**Effort:** 2-3 days
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ORPHANED FILES & DUPLICATES                               │
│                                                                                  │
│  ORPHANED FILES (0 imports):                                                     │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/features/__init__.py              - Empty placeholder    │             │
│  │  core/training/__init__.py              - Empty placeholder    │             │
│  │  core/types_pkg/__init__.py             - Unused re-export     │             │
│  │  data/store/lineage.py                  - Not integrated       │             │
│  │  data/store/versioning.py               - Not integrated       │             │
│  │  data/store/cache.py                    - Not integrated       │             │
│  │  pipeline/stages/features/cli.py        - Not connected        │             │
│  │  pipeline/stages/labeling/adaptive_barriers.py - Not used      │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  MTF TIMEFRAME INCONSISTENCIES:                                                  │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  core/constants.py:35       → ["5min", "15min", "60min"]      │             │
│  │  config/unified.py:270      → ["1min", "15min", "60min"]      │             │
│  │  adapters/multi_stream.py   → ["1min", "5min", "15min"]       │             │
│  │  → Three different defaults causing confusion                  │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  DATAFRAME FRAGMENTATION (Deferred from Phase 31):                               │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │  117 patterns of df['col'] = value causing fragmentation       │             │
│  │  → Needs systematic batch concat refactoring                   │             │
│  └────────────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Priority | Description |
|------|------|----------|-------------|
| 34-1 | `core/features/__init__.py` | LOW | Delete empty placeholder |
| 34-2 | `core/training/__init__.py` | LOW | Delete empty placeholder |
| 34-3 | `core/types_pkg/__init__.py` | LOW | Delete unused re-export |
| 34-4 | `data/store/lineage.py` | MEDIUM | Integrate or delete (~170 lines) |
| 34-5 | `data/store/versioning.py` | MEDIUM | Integrate or delete |
| 34-6 | `data/store/cache.py` | MEDIUM | Integrate or delete |
| 34-7 | `pipeline/stages/features/cli.py` | LOW | Delete unconnected CLI |
| 34-8 | `pipeline/stages/labeling/adaptive_barriers.py` | MEDIUM | Integrate or delete |
| 34-9 | `core/constants.py` | HIGH | Consolidate MTF defaults to single source |
| 34-10 | `config/unified.py` | HIGH | Import MTF defaults from constants |
| 34-11 | `features/compute/*.py` | MEDIUM | Systematic fragmentation refactoring (117 patterns) |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Empty placeholder files | 3 | 0 | Verify files deleted |
| Orphaned implementation files | 5 | 0 | Each file either integrated or deleted with justification |
| MTF default definitions | 3 | 1 | `grep -r "DEFAULT.*MTF.*TIMEFRAMES" src/` |
| Fragmentation patterns | 117 | 0 | Run pipeline, check for PerformanceWarning |
| **Code cleanliness** | Good | **Excellent** | No dead code, single source of truth

---

## Execution Order

```
Phase 24 (Quick Wins) ────────┐
                              │
Phase 25 (Validation) ────────┼───▶ Can run in parallel (different files)
                              │
Phase 26 (Type Safety) ───────┘

                              ▼

Phase 27 (Architecture) ──────▶ Depends on 26 (type changes first)

                              ▼

Phase 28 (Compute) ───────────┐
                              ├───▶ Can run in parallel
Phase 29 (Memory) ────────────┘

                              ▼

Phase 30 (Adv Architecture) ──▶ Depends on 27

                              ▼

Phase 31 (Polish) ────────────▶ Ongoing, can start anytime
```

---

## Validation Commands

### Phase 24
```bash
# Profile trend features before/after
python -c "
import time
from src.data.features.compute import trend
import pandas as pd
df = pd.DataFrame({'high': [100]*1000, 'low': [99]*1000, 'close': [99.5]*1000})
start = time.time()
trend.compute_adx_14(df)
trend.compute_plus_di_14(df)
trend.compute_minus_di_14(df)
trend.compute_adx_strong_trend(df)
print(f'Time: {time.time()-start:.3f}s')
"
```

### Phase 25
```bash
# Verify validation is called
grep -r "validate_stage_transition" src/data/pipeline/stages/*/run.py
```

### Phase 26
```bash
# Count Any types in module-level caches and function signatures
grep -rn ": Any" src/ --include="*.py" | grep -v "test" | grep -v "dict\[str, Any\]" | wc -l
# Should be 0 (legitimate kwargs with dict[str, Any] excluded)

# Count bare excepts
grep -rn "except Exception:" src/ --include="*.py" | wc -l
```

### Phase 27
```bash
# Count class definitions
grep -r "class PredictionResult" src/ | wc -l  # Should be 1
grep -r "class AdapterResult" src/ | wc -l     # Should be 2 (documented exception)
grep -r "class DataContract" src/ | wc -l      # Should be 1
grep -r "class ModelContract" src/ | wc -l     # Should be 1
grep -r "class ModelContractViolation" src/ | wc -l  # Should be 1

# Test imports
python -c "from src.core.interfaces import PredictionResult; print('OK')"
python -c "from src.models.base import PredictionResult; print('OK')"
python -c "from src.inference.orchestrator import PredictionResult; print('OK')"

# Run tests
pytest tests/ -v  # Should pass all 42 tests
```

---

*See CLEANUP_TASKS.md for detailed file:line instructions*
*See COMPLETION.md for implementation details after completion*
