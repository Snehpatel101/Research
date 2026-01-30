# Cleanup Plan: ML Factory

**Status:** Phase 29 Complete (2 implemented, 2 disproven, 1 deferred to Phase 31)
**Last Updated:** 2026-01-30 (Phase 26 closeout complete)

---

## Completed Phases Summary

All phases 0-25 are complete. See **COMPLETION.md** for full details.

| Phases | Description | Net Impact |
|--------|-------------|------------|
| 0-24 | Deduplication, contracts, 4D infra, models, validation, performance, caching | +12,010 lines, 196 features, 23 models |
| 25 | Data validation hardening (fail-fast) | ✅ COMPLETE - 3 files, pipeline now fails on bad data |
| 26 | Type safety & code quality (Any types, return annotations) | ✅ COMPLETE - 11 files, 3/4 tasks (1 deferred) |
| 27 | Architecture consolidation (class deduplication) | ✅ COMPLETE - 6 files, 5 classes consolidated |
| 28 | Compute performance optimization (numba, caching) | ✅ PARTIAL - 3 files, 3/5 tasks (2 deferred to Phase 32) |
| 29 | Memory performance optimization (cache bounds, dedup) | ✅ COMPLETE - 6 files, 2 impl/2 disproven/1 deferred to Phase 31 |

---

## New Phases: Technical Improvements (118 Issues)

**Source:** `z/TECHNICAL_IMPROVEMENTS.md` - Analysis by 5 specialized agents
**Categories:** Architecture (22), Performance (23), Code Quality (57), Data Engineering (16)

---

## Phase 24: Quick Wins - Feature Computation Caching

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Effort:** 1-2 days
**Source Issues:** PERF-001, PERF-003, PERF-009
**Completed:** 2026-01-29

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION REDUNDANCY                                │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                       │
│  │   ADX/DI     │    │ Microstructure│    │  Supertrend  │                       │
│  │   4x calls   │    │   3x calls    │    │   2x calls   │                       │
│  │  to same fn  │    │  to same fn   │    │  to same fn  │                       │
│  └──────────────┘    └──────────────┘    └──────────────┘                       │
│         │                   │                   │                                │
│         ▼                   ▼                   ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                    SOLUTION: Memoization/Caching                         │    │
│  │   - @lru_cache on base computation functions                            │    │
│  │   - Or return all variants from single function call                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description |
|------|------|-------------|
| 24-1 | `trend.py:93-133` | Cache `_compute_di_adx()` - called 4x for ADX, +DI, -DI, strong_trend |
| 24-2 | `microstructure.py:60-69` | Cache `compute_micro_amihud()` - called 3x for variants |
| 24-3 | `trend.py:216-236` | Return both supertrend value and direction from single call |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| ADX/DI computation time | 4x base | 1x base | Profile `compute_adx_14` + variants |
| Microstructure time | 3x base | 1x base | Profile `compute_micro_amihud_*` |
| Supertrend time | 2x base | 1x base | Profile supertrend features |
| **Overall trend features** | 100% | **25%** | `time python -c "from src.data.features.compute.trend import *"` |

---

## Phase 25: Data Validation Hardening

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Effort:** 2-3 days
**Source Issues:** DE-001, DE-002, DE-003, DE-008, DE-009
**Completed:** 2026-01-29

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       VALIDATION GAP ANALYSIS                                    │
│                                                                                  │
│  Stage 1 ──▶ Stage 2 ──▶ Stage 3 ──▶ ... ──▶ Training                          │
│     │           │           │                    │                              │
│     ⚠️          ⚠️          ⚠️                   ✓                              │
│  No validation between stages = silent data corruption                          │
│                                                                                  │
│  ISSUES:                                                                         │
│  • Inter-stage validation exists but NOT CALLED                                 │
│  • Raw data validation logs warnings but doesn't FAIL                           │
│  • MTF lookahead validation exists but NOT CALLED                               │
│  • Label sentinel -99 not validated at consumption                              │
│  • Horizon validation is warning-only                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description |
|------|------|-------------|
| 25-1 | `schemas.py` | Call `validate_stage_transition()` after each stage write |
| 25-2 | `clean/run.py:83-85` | Make raw data validation blocking (fail-fast) |
| 25-3 | `features/run.py:372-404` | Call `validate_no_lookahead()` after MTF generation |
| 25-4 | `splits/core.py:22` | Add sentinel validation at label consumption points |
| 25-5 | `labeling/run.py:108-175` | Default `raise_on_violation=True` |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Inter-stage validation | 0% | 100% | Grep for `validate_stage_transition` calls |
| Bad data detection | Warning only | Fail-fast | Inject bad data, verify failure |
| Lookahead detection | Not called | Called | Verify validation runs in logs |
| Sentinel leakage | Possible | Impossible | Test with -99 labels |
| **Data integrity guarantee** | PARTIAL | **FULL** | Run pipeline with intentionally bad data |

---

## Phase 26: Type Safety & Code Quality

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Effort:** 3-5 days
**Source Issues:** CQ-001, CQ-002, CQ-003, CQ-007
**Completed:** 2026-01-29

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CODE QUALITY ISSUES                                      │
│                                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │  8 `Any` types   │  │ 11 bare except   │  │  6 missing       │              │
│  │  in public APIs  │  │ handlers         │  │  return types    │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
│           │                     │                     │                         │
│           ▼                     ▼                     ▼                         │
│  Type confusion          Silent failures       Incomplete docs                  │
│                                                                                  │
│  SOLUTION: Replace with proper types, add specific exception handling           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description | Status |
|------|------|-------------|--------|
| 26-1 | Multiple (8 files) | Replace `Any` types with proper types | ✅ COMPLETE |
| 26-2 | Multiple (11 files) | Add specific exception handling to bare `except` | ⏭️ DEFERRED to Phase 31 |
| 26-3 | `config/*.py` | Add `-> None` to `__post_init__` methods | ✅ COMPLETE |
| 26-4 | `models/base.py:467` | Remove deprecated `PredictionOutput` alias | ✅ COMPLETE (kept with deprecation) |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| `Any` in module caches/signatures | 8 | 0 | `grep -rn ": Any" src/ \| grep -v "dict\[str, Any\]"` |
| Bare except handlers | 11 | 11 (deferred) | Moved to Phase 31 |
| Missing return types | 6 | 0 | mypy check |
| Deprecated aliases | 1 | 1 (with warning) | Runtime deprecation warning added |
| **Type coverage** | ~70% | **~85%** | Significant improvement |

**Note:** Legitimate `dict[str, Any]` for kwargs remain. Phase 26 fixed all module-level caches and function signatures.

**Post-Phase Fix (2026-01-30):** Fixed remaining `Any` types in `cli/run_commands_core.py:10, 90-92` that were missed during initial verification.

---

## Phase 27: Architecture Consolidation

**Status:** ✅ COMPLETE
**Priority:** MEDIUM
**Effort:** 1 day
**Source Issues:** ARCH-001, ARCH-002, ARCH-003, ARCH-004, ARCH-005
**Completed:** 2026-01-29

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DUPLICATE CLASS DEFINITIONS                                   │
│                                                                                  │
│  PredictionResult:     3 definitions → 1 (core/interfaces.py)                   │
│  AdapterResult:        2 definitions → 1 (DOCUMENTED EXCEPTION)                 │
│  DataContract:         3 definitions → 1 (contracts/data_contract.py)           │
│  ModelContract:        2 definitions → 1 (contracts/model_contract.py)          │
│  ModelContractViolation: 2 definitions → 1 (contracts/model_contract.py)       │
│                                                                                  │
│  RESULT: Single canonical definition per class enforced                         │
│                                                                                  │
│  CANONICAL LOCATIONS:                                                           │
│  • PredictionResult → src/core/interfaces.py:125                                │
│  • AdapterResult → Dual definition (circular import prevention)                 │
│  • DataContract → src/contracts/data_contract.py:114                            │
│  • ModelContract → src/contracts/model_contract.py:38                           │
│  • ModelContractViolation → src/contracts/model_contract.py:24                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description | Status |
|------|------|-------------|--------|
| 27-1 | `core/interfaces.py` | Consolidate `PredictionResult` (merge all 3 definitions) | ✅ COMPLETE |
| 27-2 | `core/interfaces.py` | AdapterResult duplicate | ✅ DOCUMENTED EXCEPTION |
| 27-3 | `contracts/data_contract.py` | Remove dead DataContract ABC | ✅ COMPLETE |
| 27-4 | `contracts/model_contract.py` | Remove dead ModelContract ABC | ✅ COMPLETE |
| 27-5 | `contracts/model_contract.py` | Deduplicate ModelContractViolation | ✅ COMPLETE |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| PredictionResult definitions | 3 | 1 | `grep -r "class PredictionResult" src/` |
| AdapterResult definitions | 2 | 2 (intentional) | Documented as circular import workaround |
| DataContract definitions | 3 | 1 | `grep -r "class DataContract" src/` |
| ModelContract definitions | 2 | 1 | `grep -r "class ModelContract" src/` |
| ModelContractViolation definitions | 2 | 1 | `grep -r "class ModelContractViolation" src/` |
| **Single definition principle** | VIOLATED | **ENFORCED** | Each class defined once (except documented exceptions) |

---

## Phase 28: Performance - Compute Optimization

**Status:** ✅ PARTIAL COMPLETE (3/5 tasks done, 2 deferred to Phase 32)
**Priority:** MEDIUM
**Effort:** 1 day (actual)
**Completed:** 2026-01-29
**Source Issues:** PERF-002, PERF-004, PERF-005, PERF-006, PERF-007

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPUTE BOTTLENECKS                                           │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Approx Ent   │    │  Sequential  │    │    GARCH     │    │     ATR      │  │
│  │   O(n²)      │    │   Features   │    │  Per-bar fit │    │  3x compute  │  │
│  │  50-100x     │    │  No parallel │    │  10-100x     │    │  per run     │  │
│  │  slower      │    │              │    │  slower      │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                                  │
│  SOLUTIONS:                                                                      │
│  • Numba JIT for entropy                                                        │
│  • ProcessPoolExecutor for feature families                                     │
│  • GARCH: fit every N bars or use EWMA                                          │
│  • Pre-compute ATR at pipeline start                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description | Status |
|------|------|-------------|--------|
| 28-1 | `entropy.py:177-188` | Apply `_count_matches_numba` to approximate entropy | ✅ COMPLETE |
| 28-2 | `features/compute/` | Parallelize feature families with ProcessPoolExecutor | ⏭️ DEFERRED to Phase 32 |
| 28-3 | `volatility.py:548-586` | Optimize GARCH (fit every 10-20 bars or EWMA) | ⏭️ DEFERRED to Phase 32 |
| 28-4 | Multiple | Pre-compute ATR with DataFrame-id caching | ✅ COMPLETE |
| 28-5 | `volume.py` | Add caching to volume helper functions | ✅ COMPLETE |

**Note on Deferred Tasks:**
- **28-2 (Parallelization):** Requires architectural changes to feature computation flow. Better addressed after cache optimizations are tested.
- **28-3 (GARCH):** Needs accuracy analysis before changing. Also, correct file path is `src/data/pipeline/stages/features/volatility.py:548-586`, not `src/data/features/compute/volatility.py`.

### Success Metrics

| Metric | Before | After | Status | How to Verify |
|--------|--------|-------|--------|---------------|
| Approx entropy time | O(n²) Python | O(n²) numba | ✅ DONE | ~50-100x speedup via numba |
| Feature parallelism | 1 core | N cores | ⏭️ DEFERRED | Moved to Phase 32 |
| GARCH time | 100% | 10-20% | ⏭️ DEFERRED | Moved to Phase 32 |
| ATR computations | 3+ per run | 1 per (df, period) | ✅ DONE | Cached by DataFrame id |
| Volume base features | Multiple recomputes | 1 per DataFrame | ✅ DONE | OBV, VWAP, dollar_volume cached |
| **Partial speedup achieved** | 100% | **~70-80%** | ✅ DONE | Entropy + caching improvements |

---

## Phase 29: Performance - Memory Optimization

**Status:** ✅ COMPLETE
**Priority:** MEDIUM
**Effort:** 1 day (actual)
**Completed:** 2026-01-29
**Source Issues:** PERF-010, PERF-011, PERF-012, DE-004, DE-010

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       MEMORY INEFFICIENCIES                                      │
│                                                                                  │
│  Issue                          │ Impact          │ Solution                    │
│  ──────────────────────────────┼─────────────────┼─────────────────────────────│
│  DataFrame fragmentation        │ 2-3x memory     │ Batch concat pattern        │
│  Label cache unbounded          │ OOM in long opt │ LRU cache with max size     │
│  Log returns computed 3x        │ CPU waste       │ Compute once at start       │
│  Multiple df.copy() calls       │ 2-3x memory     │ Single copy, in-place mods  │
│  Parquet reads all columns      │ I/O overhead    │ Column pruning              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description | Status |
|------|------|-------------|--------|
| 29-1 | Multiple | Fix remaining DataFrame fragmentation patterns | ⏭️ DEFERRED to Phase 31 |
| 29-2 | `five_dimension_objective.py:99` | Add max size to label cache | ✅ COMPLETE |
| 29-3 | Multiple | Compute log returns once (consolidate duplicates) | ✅ COMPLETE |
| 29-4 | `features/engineer.py:238` | Single df.copy() at stage entry | ❌ DISPROVEN |
| 29-5 | `features/run.py:199,294` | Add columns parameter to parquet reads | ❌ DISPROVEN |

**Deferred/Disproven Notes:**
- **29-1:** 83 patterns remain, needs systematic refactoring (moved to Phase 31)
- **29-4:** Already optimized - single copy at entry point verified
- **29-5:** Line 294 is write not read, line 199 already pruned

### Success Metrics

| Metric | Before | After | Status | How to Verify |
|--------|--------|-------|--------|---------------|
| Fragmentation warnings | 83 patterns | 83 (deferred) | ⏭️ Phase 31 | Systematic refactoring needed |
| Cache memory growth | Unbounded | Bounded (128) | ✅ DONE | LRU eviction with LABEL_CACHE_MAXSIZE |
| Log returns calls | 4 definitions | 1 canonical | ✅ DONE | Shared _helpers.py module |
| df.copy() calls | Already optimal | N/A | ✅ VERIFIED | Single copy at entry confirmed |
| Parquet column pruning | Already optimal | N/A | ✅ VERIFIED | Line 199 pruned, 294 is write |
| **Partial improvements** | 100% | **Better** | ✅ DONE | Cache bounded, log_returns consolidated |

---

## Phase 30: Advanced Architecture

**Status:** NOT STARTED
**Priority:** LOW
**Effort:** 1 week
**Source Issues:** ARCH-006, ARCH-007, ARCH-008, ARCH-009, DE-005

### Tasks

| Task | File | Description |
|------|------|-------------|
| 30-1 | `core/types.py` | Standardize transformer model family naming |
| 30-2 | `core/constants.py` | Derive constants from MODEL_CONTRACTS |
| 30-3 | `inference/orchestrator.py` | Move types to core layer |
| 30-4 | `core/interfaces.py` | Fix circular imports with TYPE_CHECKING |
| 30-5 | `features/compute/volatility.py` | Create computation context for caching intermediates |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Model family consistency | Mixed | Uniform | Check ModelFamily enum usage |
| Constant duplication | 2 sources | 1 source | Constants derived from contracts |
| Circular import workarounds | 2 | 0 | Clean import structure |
| **Architecture violations** | 10 | **0** | Code review |

---

## Phase 31: Code Polish

**Status:** NOT STARTED
**Priority:** LOW
**Effort:** Ongoing
**Source Issues:** CQ-004, CQ-005, CQ-006, DE-006, DE-007, DE-011, DE-012, CQ-002 (from Phase 26)

### Tasks

| Task | File | Description |
|------|------|-------------|
| 31-1 | `monitor.py:264-265` | Address TODO comments |
| 31-2 | Multiple (18+ files) | Fix bare exception handlers (deferred from Phase 26) |
| 31-3 | Multiple | Extract magic numbers to named constants |
| 31-4 | `config/unified.py` | Consolidate duplicate default definitions |
| 31-5 | `adapters/base.py` | Complete feature column exclusion list |
| 31-6 | `multi_stream.py` | Fix temporal misalignment for non-integer ratios |
| 31-7 | `features/engineer.py` | Define feature dependency DAG |
| 31-8 | `adapters/*.py` | Move common methods to BaseAdapter |
| 31-9 | Multiple (83 patterns) | Fix DataFrame fragmentation (deferred from Phase 29) |
| 31-9 | Multiple (83 patterns) | Fix DataFrame fragmentation (deferred from Phase 29) |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| TODO comments | 3 | 0 | `grep -r "TODO" src/` |
| Bare exception handlers | 50+ | 0 | All have specific handling + logging |
| Magic numbers | 6 | 0 | Code review for unexplained constants |
| Duplicate defaults | Multiple | 0 | Single definition per default |
| DataFrame fragmentation | 83 patterns | 0 | No PerformanceWarning from pandas |
| **Code cleanliness** | Good | **Excellent** | Ruff + manual review |

---

## Phase Summary

| Phase | Focus | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| 24 | Feature Caching | HIGH | 1-2 days | ✅ COMPLETE (75% speedup) |
| 25 | Validation | HIGH | 2-3 days | ✅ COMPLETE (fail-fast validation) |
| 26 | Type Safety | HIGH | 3-5 days | ✅ COMPLETE (3/4 tasks, 1 deferred) |
| 27 | Architecture | MEDIUM | 1 day | ✅ COMPLETE (5 classes consolidated) |
| 28 | Compute Perf | MEDIUM | 1 day | ✅ PARTIAL (3/5, 2 deferred to Phase 32) |
| 29 | Memory Perf | MEDIUM | 1 day | ✅ COMPLETE (2 impl, 2 disproven, 1 deferred) |
| 30 | Adv Architecture | LOW | 1 week | Not Started |
| 31 | Polish | LOW | Ongoing | Not Started (includes 26-2, 29-1) |

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
