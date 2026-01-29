# Cleanup Plan: ML Factory

**Status:** Phase 23 In Progress
**Last Updated:** 2026-01-28

---

## Completed Phases

| Phase | Description | Impact | Date |
|-------|-------------|--------|------|
| 0 | Deduplication | -5,336 lines | 2026-01-23 |
| 1 | Contract Enforcement | +616 lines | 2026-01-23 |
| 2 | 4D Infrastructure | +958 lines | 2026-01-24 |
| 3 | 5-Dimension Optuna | +2,298 lines | 2026-01-24 |
| 4-10 | Validation, Entry Point, Models, Hardening | +6,556 lines | 2026-01-24 |
| 12-18 | Trading, Quality, Performance, Ensemble | +8,464 lines | 2026-01-25 |
| 19 | Comprehensive Optimization | +750 lines, 34 features | 2026-01-25 |
| 20 | Performance & Quality Polish | -851 lines, 50-100x speedup | 2026-01-25 |
| 21 | ML Pipeline Review Fixes | 10 tasks, 3 disproven | 2026-01-27 |
| 22 | OPTIMIZE_FOR Metric Wiring | 7 changes | 2026-01-27 |

**Net Impact:** ~+12,010 lines | See **COMPLETION.md** for details.

---

## Phase 23: Critical Bugfixes & Validation Fixes (IN PROGRESS)

**Status:** IN PROGRESS | 2026-01-28
**Priority:** CRITICAL + HIGH
**Source:** 4 parallel verification agents

### Overview

| Sub-Phase | Issue | Priority | Status |
|-----------|-------|----------|--------|
| 23A | Label column data leakage | CRITICAL | Pending |
| 23B | Validation timing + feature count | HIGH | Pending |
| 23C | Feature engineering performance | MEDIUM | Pending |
| 23D | Config gaps | LOW | Deferred |

---

### 23A: Critical Label Leakage Bugfix

**Issue:** `src/data/adapters/base.py:339-347` does NOT exclude "label" column from features.

**Impact:** ALL models train with label as feature = 100% training accuracy, catastrophic production failure.

**Fix:** Add `"label"` to `exclude_exact` set.

---

### 23B: Validation Timing & Feature Selection

**Issue 1:** `unified_orchestrator.py:501` validates raw 2D DataFrame before adapters transform to 3D/4D at line 579.

**Issue 2:** Pipeline produces 218 features, exceeds:
- LightGBM: max 200
- TCN: max 120
- PatchTST: max 10

**Fix:**
1. Move validation after adapter OR skip rank validation on raw data
2. Add auto feature selection before validation

---

### 23C: Feature Engineering Performance

| File | Line | Issue | Speedup |
|------|------|-------|---------|
| temporal.py | 64 | `.apply(get_session)` | 10-50x |
| microstructure.py | 589-591 | Loop assignment | 5-20x |
| volatility.py | 97-116 | Individual BB assignments | 2-5x |

**Fix:** Vectorize with `np.select()`, `pd.concat()`, `df.assign()`

---

### 23D: Config Gaps (Deferred to Phase 24)

- Bundle registry/versioning
- A/B testing configuration
- Drift detection config

---

## Deferred Backlog

| Task | Description |
|------|-------------|
| 5C | Unified deployment bundle |
| 4D | Deflated Sharpe Ratio |
| 4E | Bootstrap CIs |
| 4F | Auto calibration |

---

*See COMPLETION.md for implementation details*
