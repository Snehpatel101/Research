# Critical Bugs and Technical Debt

## Overview

**Status 2026-01-23:** All 5 previously documented critical bugs are FIXED.

No new critical bugs identified. Focus is now on technical debt reduction (see below).

---

## ✅ FIXED: HMM Regime Detection - Lookahead Bias

**File**: `src/phase1/stages/regime/hmm.py` 
**Status**: FIXED on 2026-01-01

### Solution Applied
Added shift(1) to all HMM outputs in `detect_with_probabilities()` method:
```python
# Lines 452-457 - ANTI-LOOKAHEAD fix
regimes = regimes.shift(1)
prob_df = prob_df.shift(1)
```

This ensures regime at bar N only uses data from bars 0..N-1.

---

## ✅ FIXED: GA Optimization - Test Data Leakage

**File**: `src/phase1/stages/ga_optimize/optuna_optimizer.py`
**Status**: FIXED via safe_mode default

### Solution Applied
- `safe_mode = True` is now the default (run.py line 136)
- When safe_mode is enabled, optimization uses only the first 70% of data
- This prevents test data from influencing parameter optimization

---

## ✅ FIXED: Transaction Costs NOT in Labels

**File**: `src/phase1/stages/labeling/triple_barrier.py`
**Status**: FIXED

### Solution Applied
Transaction costs are now properly integrated:
- `apply_transaction_costs=True` by default
- `cost_in_atr` calculated from symbol-specific costs
- Upper barrier adjusted: `k_up_effective = k_up + cost_in_atr`
- WIN requires: `gross_profit >= (k_up + cost_in_atr) * ATR`

Example with k_up=2.0, cost_in_atr=0.15:
- Effective upper barrier: 2.15 ATR
- Only trades with 2.15+ ATR profit are labeled WIN

---

## ✅ FIXED: MTF/Regime Missing shift(1) at Output

**Files**: `src/phase1/stages/regime/*.py`, `src/phase1/stages/mtf/generator.py`
**Status**: FIXED

### Solution Already Applied
- `composite.py` lines 266-276: Applies shift(1) to all regime columns (volatility, trend, structure)
- `hmm.py` lines 452-457: Now applies shift(1) to HMM regimes and probabilities (fixed 2026-01-01)

---

## ✅ FIXED: LightGBM num_leaves/max_depth Constraint

**File**: `src/cross_validation/param_spaces.py`
**Status**: FIXED

### Solution Applied
- Validation function enforces `num_leaves <= 2^max_depth`
- Conservative default: num_leaves capped at 64
- Dynamic tuning validates constraint during hyperparameter search
- Clear error messages when constraint violated

---

## Validation Commands

After fixing, run:
```bash
# Full test suite
pytest tests/ -v

# Lookahead-specific tests
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v

# Leakage tests
pytest tests/phase_1_tests/stages/test_splits_leakage.py -v
```

**Expected**: Some performance reduction after fixes (indicates leakage was real).

---

## Technical Debt Issues (Identified 2026-01-23)

### HIGH Priority

| ID | Issue | Location | Impact | Effort |
|----|-------|----------|--------|--------|
| CFG-002 | 71+ duplicated _get_global_or_default() patterns | Multiple files | Maintenance nightmare | 2 days |
| SCALE-001 | Fragile string matching for feature columns | Scaling stage | Silent failures | 1 day |
| VAL-001 | Hardcoded validation thresholds (0.85 corr, 0.01 var) | Validation stage | Inflexibility | 0.5 days |

### MEDIUM Priority

| ID | Issue | Location | Impact | Effort |
|----|-------|----------|--------|--------|
| CFG-005 | No cross-config validation | Config system | Silent conflicts | 1 day |
| LIN-001 | Lineage queries O(n) | LineageTracker | Performance at scale | 0.5 days |
| PIPE-008 | Single-symbol enforcement | Pipeline core | Blocks batch processing | 2-3 days |

### LOW Priority

| ID | Issue | Location | Impact | Effort |
|----|-------|----------|--------|--------|
| CORE-001 | Missing core.py for 10/12 stages | Pipeline stages | Inconsistent patterns | 5-7 days |
| PIT-001 | DateTime column name assumptions | FeatureStore | Brittleness | 0.5 days |
| CONC-001 | No concurrent access testing | Data store | Production risk | 1 day |
| ML-001 | Asymmetry bonus duplicated | fitness.py, optuna_optimizer.py | Inconsistency risk | 0.5 days |

### Recommendations

1. **Immediate:** CFG-002 migration should be first priority - create centralized `get_config_value()` pattern
2. **Short-term:** VAL-001 and SCALE-001 are quick wins with high impact
3. **Medium-term:** CFG-005 CompositeValidator prevents config conflicts

---

**Last Updated:** 2026-01-23 (Phase 2 Agent #4 - SERENA Memories Updater)
