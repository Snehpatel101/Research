# ML Factory - Cleanup Tasks

**Status:** Phase 23 In Progress
**Last Updated:** 2026-01-28

---

## Completed Phases Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|------------------|
| 0-10 | 47/49 | Deduplication, contracts, 4D infra, Optuna, models |
| 12-18 | 76/80 | Trading, quality, performance, ensemble, resilience |
| 19 | 17/21 | 34 new features, vectorization, code quality |
| 20 | 9/15 | -851 lines, 50-100x speedup |
| 21 | 10/11 | ML pipeline fixes (3 disproven) |
| 22 | 7/7 | OPTIMIZE_FOR metric wiring |

**Net Impact:** ~+12,010 lines | See **COMPLETION.md** for details.

---

## Phase 23: Critical Bugfixes & Validation Fixes (IN PROGRESS)

**Status:** IN PROGRESS | 2026-01-28
**Tasks:** 0/6 (3 deferred)
**Source:** 4 parallel verification agents

---

### 23A: Critical Label Leakage Bugfix (PRIORITY 1)

#### 23A-1: Add "label" to exclude_exact Set
- [ ] File: `src/data/adapters/base.py:339-347`
- [ ] Issue: "label" column included as feature = 100% leakage
- [ ] Fix: Add `"label"` to exclude_exact set
- [ ] Validation: `grep -n '"label"' src/data/adapters/base.py`

---

### 23B: Validation Timing & Feature Selection (PRIORITY 2)

#### 23B-1: Move Validation After Adapter Transformation
- [ ] File: `src/models/training/unified_orchestrator.py:501,579`
- [ ] Issue: Line 501 validates 2D data, adapters transform at 579
- [ ] Fix: Move validation after adapter OR add skip_rank_validation

#### 23B-2: Add Auto Feature Selection Before Validation
- [ ] File: `src/models/training/unified_orchestrator.py:499`
- [ ] Issue: 218 features exceeds LightGBM(200), TCN(120), PatchTST(10)
- [ ] Fix: Auto-select top N features before validation

---

### 23C: Feature Engineering Performance (PRIORITY 3)

#### 23C-1: Vectorize temporal.py get_session
- [ ] File: `src/data/pipeline/stages/features/temporal.py:64`
- [ ] Issue: `.apply(get_session)` is 10-50x slower
- [ ] Fix: Replace with `np.select()`

#### 23C-2: Replace microstructure.py Loop
- [ ] File: `src/data/pipeline/stages/features/microstructure.py:589-591`
- [ ] Issue: Loop assignment is 5-20x slower
- [ ] Fix: Replace with `pd.concat()`

#### 23C-3: Batch Bollinger Band Assignments
- [ ] File: `src/data/pipeline/stages/features/volatility.py:97-116`
- [ ] Issue: Individual assignments 2-5x slower
- [ ] Fix: Single `df.assign()` call

---

### 23D: Config Gaps (DEFERRED TO PHASE 24)

- [ ] 23D-1: Bundle registry/versioning
- [ ] 23D-2: A/B testing configuration
- [ ] 23D-3: Drift detection config

---

## Deferred Backlog (Low Priority)

| Task | Description | Notes |
|------|-------------|-------|
| 5C | Unified deployment bundle | Needs spec |
| 4D | Deflated Sharpe Ratio | Post-Optuna gate |
| 4E | Bootstrap CIs | Wire BootstrapCI |
| 4F | Auto calibration | Wire CalibrationManager |

---

## Verification Commands

```bash
# Core imports
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"

# Phase 23 verification
grep -n "exclude_exact" src/data/adapters/base.py
grep -n "_pre_training_validation" src/models/training/unified_orchestrator.py
```

---

*See COMPLETION.md for implementation details*
