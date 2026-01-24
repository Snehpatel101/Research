# Cleanup Plan: ML Factory - Remaining Work

**Status:** Phases 0-6 Complete | All Models Implemented
**Last Updated:** 2026-01-24

---

## Completed Phases (see COMPLETION.md)

| Phase | Description | Lines Changed |
|-------|-------------|---------------|
| 0 | Deduplication | -5,336 |
| 1 | Contract Enforcement | +616 |
| 2 | 4D Infrastructure | +958 |
| 3 | 5-Dimension Optuna | +2,298 |
| 4 | Validation Integration | +50 |
| 5 | Unified Entry Point | +1,281 |
| 6 | Advanced Models | +3,690 |

---

## Remaining Work

---

## Phase 7: Production Hardening

**Status:** NOT STARTED
**Priority:** HIGH - Critical for production reliability
**Estimated Impact:** Makes validation blocking by default, adds pipeline robustness

### Rationale

Analysis revealed that validation is warning-only by default:
- `raise_on_leakage=False` in leakage_detection.py
- `raise_on_lookahead=False` in lookahead_audit.py
- `Trainer` class has NO validation hooks
- Pipeline stages lack inter-stage schema validation

### Architecture Changes

```
BEFORE:                              AFTER:
┌─────────────┐                     ┌─────────────┐
│   Trainer   │ ← No validation     │   Trainer   │ ← Pre-training validation
└─────────────┘                     └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │  Leakage    │ ← Blocking by default
                                    │  Lookahead  │
                                    └─────────────┘

Pipeline Stages:
Stage N → Stage N+1                 Stage N → Schema Check → Stage N+1
(implicit trust)                    (explicit validation)
```

### Tasks Overview

| Task | Description | Risk |
|------|-------------|------|
| 7A | Make validation blocking by default | LOW |
| 7B | Add inter-stage schema validation | MEDIUM |
| 7C | Consistent adapter error handling | LOW |
| 7D | Feature manifest system | MEDIUM |

---

## Phase 8: Code Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM - Reduces maintenance burden
**Estimated Impact:** -500 lines duplicate code, unified patterns

### Rationale

Analysis found significant duplication:
- 79 files have `to_dict` with inconsistent implementations
- Class weight calculation duplicated across 4+ model files
- `_safe_divide`, `_sma`, `_check_cuda_available` in 5+ files each
- 24 custom exception classes, many similar
- 150+ magic numbers without named constants

### Architecture Changes

```
BEFORE:                              AFTER:
src/models/boosting/                src/models/common/
  xgboost_model.py                    class_weights.py  ← Single source
    └─ _compute_class_weights()       device_utils.py
  lightgbm_model.py
    └─ _compute_class_weights()     src/core/utils/
  catboost_model.py                   math_utils.py     ← _safe_divide, _sma
    └─ _compute_class_weights()
                                    src/core/exceptions.py ← All exceptions
src/data/pipeline/stages/
  microstructure.py                 src/config/constants/
    └─ _safe_divide()                 default_periods.py  ← 14, 20, 100
  momentum.py                         thresholds.py       ← 0.10, 0.5
    └─ _safe_divide()
```

### Tasks Overview

| Task | Description | Risk |
|------|-------------|------|
| 8A | Extract common utilities | LOW |
| 8B | Consolidate exceptions | LOW |
| 8C | Extract magic numbers to constants | LOW |
| 8D | Deprecation cleanup (PredictionOutput) | LOW |

---

## Phase 9: Directory Cleanup

**Status:** NOT STARTED
**Priority:** LOW - Reduces cognitive overhead
**Estimated Impact:** -13 empty directories, updated docs

### Rationale

Analysis found 13 empty directories containing only `__pycache__`:
- Created during Phase 0 consolidation but not deleted
- Cause import confusion
- 7 deprecated shim files still exist

### Directories to Delete

```
src/
├── contracts/       ← EMPTY (consolidated to src/core/contracts/)
├── ml_pipeline/     ← EMPTY (consolidated to src/data/pipeline/)
├── adapters/        ← EMPTY (consolidated to src/data/adapters/)
├── common/          ← EMPTY (consolidated to src/core/common/)
├── monitoring/      ← EMPTY (consolidated to src/validation/monitoring/)
├── feature_store/   ← EMPTY (unused)
├── utils/           ← EMPTY (consolidated to src/core/utils/)
├── cross_validation/← EMPTY (consolidated to src/validation/cv/)
├── evaluation/      ← EMPTY (consolidated to src/models/evaluation/)
├── backtesting/     ← EMPTY (consolidated to src/inference/backtesting/)
├── pipeline/        ← EMPTY (consolidated to src/data/pipeline/)
├── features/        ← EMPTY (consolidated to src/data/features/)
└── training/        ← SHIM ONLY (re-exports from src/models/training/)
```

### Tasks Overview

| Task | Description | Risk |
|------|-------------|------|
| 9A | Delete empty directories | LOW |
| 9B | Delete deprecated shims | LOW |
| 9C | Update CLAUDE.md documentation | LOW |

---

## Phase 10: Refactor Complex Functions

**Status:** NOT STARTED
**Priority:** LOW - Improves testability and readability
**Estimated Impact:** Better test coverage, reduced cyclomatic complexity

### Rationale

Analysis found functions with extreme complexity:
- `stacking.py:fit()` - 400+ lines, 30 ifs, 30 loops
- `unified_orchestrator.py:_pre_training_validation()` - 150+ lines, 24 ifs

### Refactoring Strategy

```
BEFORE: stacking.py:fit() (400+ lines)
┌─────────────────────────────────────┐
│ 1. Config validation                │
│ 2. Heterogeneous ensemble detection │
│ 3. Input shape validation           │
│ 4. Sequence data caching            │
│ 5. Memory warnings                  │
│ 6. OOF prediction generation        │
│ 7. Meta-learner training            │
│ 8. Base model retraining            │
│ 9. Diversity analysis               │
│ 10. Metrics computation             │
└─────────────────────────────────────┘

AFTER: stacking.py:fit() (~50 lines)
┌─────────────────────────────────────┐
│ self._validate_ensemble_config()    │
│ self._prepare_heterogeneous_data()  │
│ self._cache_sequence_data()         │
│ self._generate_oof_predictions()    │
│ self._train_meta_learner()          │
│ self._train_final_base_models()     │
│ return self._compute_metrics()      │
└─────────────────────────────────────┘
```

### Tasks Overview

| Task | Description | Risk |
|------|-------------|------|
| 10A | Split stacking.py:fit() | HIGH |
| 10B | Split _pre_training_validation() | MEDIUM |

---

## Previously Deferred (Now Phase 11+)

| Item | Description | Original Phase | New Phase |
|------|-------------|----------------|-----------|
| 5C | Unified deployment bundle (tar.gz) | Phase 5 | Phase 11 |
| 4C | Ensemble diversity analysis integration | Phase 4 | Phase 11 |
| 4D | Deflated Sharpe Ratio post-Optuna | Phase 4 | Phase 11 |
| 4E | Bootstrap CIs in financial reports | Phase 4 | Phase 11 |
| 4F | Auto calibration in orchestrator | Phase 4 | Phase 11 |
| 4G | Bet sizing connection | Phase 4 | Phase 11 |
| - | MTF ablation flag | Phase 3 | Phase 11 |

---

## Execution Protocol

### Recommended Phase Order

```
Phase 7 (HIGH priority) → Phase 8 (MEDIUM) → Phase 9 (LOW) → Phase 10 (LOW, HIGH risk)
```

### Before Each Phase:

1. Read CLEANUP_TASKS.md for specific file:line targets
2. Run verification commands to confirm current state
3. Create feature branch: `git checkout -b phase-N-description`

### After Each Task:

```bash
# 1. Lint changed files
ruff check src/path/to/changed/file.py --fix
black src/path/to/changed/file.py

# 2. Run verification command from CLEANUP_TASKS.md
# 3. Mark task ✅ in CLEANUP_TASKS.md
```

### After Each Phase:

1. Update CLEANUP_TASKS.md - mark all tasks ✅
2. Update CLEANUP_PLAN.md - change status to COMPLETE
3. Add phase summary to COMPLETION.md
4. Commit with descriptive message

### Phase-Specific Notes

| Phase | Risk | Notes |
|-------|------|-------|
| 7 | LOW | Mostly default value changes |
| 8 | LOW | New files, minimal changes to existing |
| 9 | LOW | Deletions only after import verification |
| 10 | HIGH | Major refactoring, needs careful testing |

---

*For completed phase details, see COMPLETION.md*
