# Cleanup Plan: ML Factory

**Status:** Phase 26 Complete, Phase 27 Ready to Start
**Last Updated:** 2026-01-29

---

## Completed Phases Summary

All phases 0-25 are complete. See **COMPLETION.md** for full details.

| Phases | Description | Net Impact |
|--------|-------------|------------|
| 0-24 | Deduplication, contracts, 4D infra, models, validation, performance, caching | +12,010 lines, 196 features, 23 models |
| 25 | Data validation hardening (fail-fast) | 3 files, pipeline now fails on bad data |
| 26 | Type safety & code quality (Any types, return annotations) | 11 files, 3/4 tasks (1 deferred) |

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
| `Any` in public APIs | 8 | 0 | `grep -r ": Any" src/ \| wc -l` |
| Bare except handlers | 11 | 11 (deferred) | Moved to Phase 31 |
| Missing return types | 6 | 0 | mypy check |
| Deprecated aliases | 1 | 1 (with warning) | Runtime deprecation warning added |
| **Type coverage** | ~70% | **~85%** | Significant improvement |

---

## Phase 27: Architecture Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM
**Effort:** 1 week
**Source Issues:** ARCH-001, ARCH-002, ARCH-003, ARCH-004, ARCH-005

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DUPLICATE CLASS DEFINITIONS                                   │
│                                                                                  │
│  PredictionResult:     3 definitions (models/base, core/interfaces, inference)  │
│  AdapterResult:        2 definitions (adapters/base, core/interfaces)           │
│  DataContract:         3 definitions (data_contract, contracts/, interfaces)    │
│  ModelContract:        2 definitions (interfaces=abstract, contracts/=dataclass)│
│                                                                                  │
│  SOLUTION: Single canonical definition per class, re-export where needed        │
│                                                                                  │
│  CANONICAL LOCATIONS:                                                           │
│  • PredictionResult → src/core/interfaces.py                                    │
│  • AdapterResult → src/data/adapters/base.py                                    │
│  • DataContract → rename DatasetContract to PipelineData                        │
│  • ModelContract (abstract) → rename to ModelInterface                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Description |
|------|------|-------------|
| 27-1 | `core/interfaces.py` | Consolidate `PredictionResult` (merge all 3 definitions) |
| 27-2 | `core/interfaces.py` | Remove duplicate `AdapterResult`, import from adapters |
| 27-3 | `core/data_contract.py` | Rename `DatasetContract` to `PipelineData` |
| 27-4 | `core/interfaces.py` | Rename abstract `ModelContract` to `ModelInterface` |
| 27-5 | `models/neural/*.py` | Replace `PredictionOutput` with `PredictionResult` |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| PredictionResult definitions | 3 | 1 | `grep -r "class PredictionResult" src/` |
| AdapterResult definitions | 2 | 1 | `grep -r "class AdapterResult" src/` |
| DataContract naming collisions | 3 | 0 | No same-name classes |
| ModelContract confusion | 2 concepts | Clear separation | Docs + naming |
| **Single definition principle** | VIOLATED | **ENFORCED** | Each class defined once |

---

## Phase 28: Performance - Compute Optimization

**Status:** NOT STARTED
**Priority:** MEDIUM
**Effort:** 1-2 weeks
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

| Task | File | Description |
|------|------|-------------|
| 28-1 | `entropy.py:177-188` | Apply `_count_matches_numba` to approximate entropy |
| 28-2 | `features/compute/` | Parallelize feature families with ProcessPoolExecutor |
| 28-3 | `volatility.py:548-586` | Optimize GARCH (fit every 10-20 bars or EWMA) |
| 28-4 | Multiple | Pre-compute ATR once at pipeline start |
| 28-5 | `volume.py` | Add `@lru_cache` to volume helper functions |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Approx entropy time | O(n²) | O(n) | Profile entropy features |
| Feature parallelism | 1 core | N cores | Check CPU usage during feature gen |
| GARCH time | 100% | 10-20% | Profile volatility features |
| ATR computations | 3+ per run | 1 per run | Grep for ATR compute calls |
| **Feature gen time** | 100% | **25-50%** | End-to-end feature benchmark |

---

## Phase 29: Performance - Memory Optimization

**Status:** NOT STARTED
**Priority:** MEDIUM
**Effort:** 3-5 days
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

| Task | File | Description |
|------|------|-------------|
| 29-1 | Multiple | Fix remaining DataFrame fragmentation patterns |
| 29-2 | `five_dimension_objective.py:99` | Add max size to label cache |
| 29-3 | Multiple | Compute log returns once at pipeline start |
| 29-4 | `features/engineer.py:238` | Single df.copy() at stage entry |
| 29-5 | `features/run.py:199,294` | Add columns parameter to parquet reads |

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| Fragmentation warnings | Some | 0 | Check logs for PerformanceWarning |
| Cache memory growth | Unbounded | Bounded | Monitor memory in long opt |
| Log returns calls | 3+ | 1 | Grep for log return computation |
| df.copy() calls | Multiple | 1 per stage | Grep for `.copy()` |
| **Peak memory** | 100% | **70-80%** | Memory profiler during training |

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

### Success Metrics

| Metric | Before | After | How to Verify |
|--------|--------|-------|---------------|
| TODO comments | 3 | 0 | `grep -r "TODO" src/` |
| Bare exception handlers | 50+ | 0 | All have specific handling + logging |
| Magic numbers | 6 | 0 | Code review for unexplained constants |
| Duplicate defaults | Multiple | 0 | Single definition per default |
| **Code cleanliness** | Good | **Excellent** | Ruff + manual review |

---

## Phase Summary

| Phase | Focus | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| 24 | Feature Caching | HIGH | 1-2 days | ✅ COMPLETE (75% speedup) |
| 25 | Validation | HIGH | 2-3 days | ✅ COMPLETE (fail-fast validation) |
| 26 | Type Safety | HIGH | 3-5 days | ✅ COMPLETE (3/4 tasks, 1 deferred) |
| 27 | Architecture | MEDIUM | 1 week | Ready to Start |
| 28 | Compute Perf | MEDIUM | 1-2 weeks | Not Started |
| 29 | Memory Perf | MEDIUM | 3-5 days | Not Started |
| 30 | Adv Architecture | LOW | 1 week | Not Started |
| 31 | Polish | LOW | Ongoing | Not Started (includes 26-2) |

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
# Count Any types
grep -rn ": Any" src/ --include="*.py" | grep -v "test" | wc -l

# Count bare excepts
grep -rn "except Exception:" src/ --include="*.py" | wc -l
```

### Phase 27
```bash
# Count class definitions
grep -r "class PredictionResult" src/ | wc -l  # Should be 1
grep -r "class AdapterResult" src/ | wc -l     # Should be 1
```

---

*See CLEANUP_TASKS.md for detailed file:line instructions*
*See COMPLETION.md for implementation details after completion*
