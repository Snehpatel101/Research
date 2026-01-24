# ML Factory Cleanup Tasks - Active Work

**Last Updated:** 2026-01-23
**Status:** ✅ Phase 0 Complete | Phase 1 Ready
**Phase 0 Impact:** -5,336 lines removed | 7 naming conflicts resolved

---

## Table of Contents
- [Completed: Phase 0](#completed-phase-0-deduplication)
- [Active: Phase 1](#phase-1-contract-enforcement)
- [Upcoming: Phases 2-5](#upcoming-phases)
- [False Positives](#false-positives)
- [Deferred Items](#deferred-items)
- [Verification Commands](#verification-commands)

---

## Completed: Phase 0 Deduplication

**Status:** ✅ COMPLETE (2026-01-23)
**Full Details:** `X ( IN PROGRESS DOCS) X/PHASE_0_COMPLETION.md`

| Task | Description | Lines | Status |
|------|-------------|-------|--------|
| 0A | DataRank → `src/core/types.py` | -15 | ✅ |
| 0B | ModelFamily + TRANSFORMER → `src/core/types.py` | -30 | ✅ |
| 0C | Delete `src/coordination/` | -1,166 | ✅ |
| 0D | Delete `src/feature_selection/` | -3,508 | ✅ |
| 0E | MultiResolution4DAdapter → `src/data/adapters/` | -617 | ✅ |
| 0F | AdapterResult compatibility properties | ±0 | ✅ |
| 0G | Rename validation DataContract → OHLCVValidationSchema | ±0 | ✅ |

**Key Results:**
- Single source of truth for all enums in `src/core/types.py`
- All duplicate directories eliminated
- Backward compatibility maintained via re-exports
- 3 parallel verification agents + Task Agent 7 APPROVED

---

## Phase 1: Contract Enforcement

**Status:** Ready to Begin
**Blocked By:** Phase 0 ✅
**Estimated Effort:** 2-3 days

### Task 1A: Make DataContract.validate_dataframe() Raise on Failure
**Priority:** CRITICAL

```python
# BEFORE (current broken pattern):
is_valid, issues = contract.validate_dataframe(df)
# Caller ignores is_valid, proceeds anyway

# AFTER (enforced pattern):
def validate_dataframe_strict(self, df: pd.DataFrame) -> None:
    is_valid, issues = self.validate_dataframe(df)
    if not is_valid:
        raise DataContractViolation(issues)
```

**Files:**
- `src/core/contracts/data_contract.py:195-224`

---

### Task 1B: Call ModelContract.validate_data_contract() in Adapter Load Path
**Priority:** CRITICAL

**Files:**
- `src/data/adapters/base.py` - Add validation before returning data

---

### Task 1C: Add Pre-Training Validation Hook
**Priority:** HIGH

Wire leakage detection and lookahead audit to block training when issues found.

**Files:**
- `src/training/orchestrator.py` - Add validation hook
- `src/validation/leakage_detection.py` - Add blocking mode
- `src/validation/lookahead_audit.py` - Add blocking mode

---

### Task 1D: Fix Chronological Splits Validation
**Priority:** MEDIUM

Validate chronological sorting BEFORE computing split indices.

**Files:**
- `src/data/pipeline/stages/splits/core.py`

---

## Upcoming Phases

| Phase | Description | Blocked By | Est. Effort |
|-------|-------------|------------|-------------|
| 2 | 4D Infrastructure (raw MTF store + adapter) | Phase 1 | 5-7 days |
| 3 | 5-Dimension Optuna | Phase 2 | 5-7 days |
| 4 | Validation Integration | Phase 1 | 3-4 days |
| 5 | Unified MLFactory Entry Point | Phase 3 | 3-4 days |

**Full details:** See `CLEANUP_PLAN.md`

---

## False Positives

Things that LOOK wrong but are OK:

| Item | Location | Why It's OK |
|------|----------|-------------|
| ModelFamilyDefaults | `feature_selection/config.py` | Different class (dataclass, not enum) |
| Multiple FeatureSelector | `data/features/`, `models/training/` | Different contexts, both valid |
| DatasetContract vs DataContract | `core/data_contract.py` vs `core/contracts/` | Intentionally different purposes |

---

## Deferred Items

| Item | Reason | Future Phase |
|------|--------|--------------|
| Scaler Consolidation | Requires pipeline integration understanding | Phase 2+ |
| Validator Pattern Standardization | Multiple valid patterns for different contexts | Phase 3+ |
| Pipeline Stage Refactoring | Large architectural change | Phase 4+ |

---

## Verification Commands

### Quick Health Check
```bash
# Core imports
python -c "from src.core.types import DataRank, ModelFamily; print('Types OK')"
python -c "from src.core.coordination import TimeframeCoordinator; print('Coordination OK')"
python -c "from src.optimization.feature_selection import FeatureSelectionResult; print('Feature Selection OK')"
python -c "from src.data.adapters import MultiResolution4DAdapter; print('Adapters OK')"

# Single definitions
grep -r "class DataRank" src/ --include="*.py" | wc -l  # Should be 1
grep -r "class ModelFamily" src/ --include="*.py" | wc -l  # Should be 1

# No dead imports
grep -r "from src\.coordination" src/ --include="*.py" | wc -l  # Should be 0
grep -r "from src\.feature_selection" src/ --include="*.py" | wc -l  # Should be 0
```

---

## Change Log

| Date | Phase | Impact | Notes |
|------|-------|--------|-------|
| 2026-01-23 | Phase 0 | -5,336 lines | 7 tasks complete, all verified |

---

*Next Phase: Contract Enforcement (Phase 1)*
*Full cleanup plan: `CLEANUP_PLAN.md`*
