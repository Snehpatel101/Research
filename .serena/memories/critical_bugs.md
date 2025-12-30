# Critical Bugs - MUST FIX BEFORE PRODUCTION

## Overview

There are **3 CRITICAL** and **2 HIGH** severity bugs that must be fixed before any model can be trusted for live trading.

---

## CRITICAL #1: HMM Regime Detection - Lookahead Bias

**File**: `src/phase1/stages/regime/hmm.py` lines 329-354
**Impact**: SEVERE - Models trained with HMM regime features will have artificially inflated performance

### Problem
When `expanding=True` (the DEFAULT), the HMM is trained on the ENTIRE dataset including future data. States are then assigned retroactively.

```python
# PROBLEMATIC CODE
if self.expanding:
    window_obs = observations[:n_samples]  # ALL observations including future!
    self._fitted_model, raw_states, raw_probs = fit_gaussian_hmm(window_obs, ...)
```

### Fix Options
1. **Disable expanding mode** - Use rolling mode only for production
2. **Fix expanding mode** - Train incrementally using only past data at each point
3. **Add warning** - Mark expanding mode as research-only

---

## CRITICAL #2: GA Optimization - Test Data Leakage

**File**: `src/phase1/stages/ga_optimize/optuna_optimizer.py`
**Impact**: SEVERE - Test set performance will be optimistically biased

### Problem
Stage 5 (GA optimization) runs BEFORE Stage 7 (train/test splits). The barrier parameter optimization uses data that will later become the test set.

```
Current Order (WRONG):
Stage 4: Initial Labeling (ALL data)
Stage 5: GA Optimization (ALL data) ← LEAKS TEST DATA
Stage 7: Train/Val/Test Splits
```

### Fix
Use only pre-split training data (first 70%) for optimization, OR reorder pipeline to split BEFORE optimization.

---

## CRITICAL #3: Transaction Costs NOT in Labels

**File**: `src/phase1/stages/labeling/triple_barrier.py`
**Impact**: SEVERE - Models learn gross profits, trades pay net

### Problem
Triple-barrier labels show GROSS profit targets without deducting transaction costs.

```python
# PROBLEMATIC CODE
upper_barrier = entry_price + k_up * entry_atr  # GROSS target (no costs!)
```

### Example
- Barrier: 2.0 ATR profit target
- Transaction cost: 0.3 ATR (commission + slippage)
- Model labels 1.8 ATR as WIN, but it's actually a LOSS after costs

### Fix
Adjust barriers to account for round-trip costs:
```python
cost_in_atr = cost_in_price / entry_atr
upper_barrier = entry_price + (k_up - cost_in_atr) * entry_atr
```

---

## HIGH #1: MTF/Regime Missing shift(1) at Output

**Files**: `src/phase1/stages/regime/*.py`, `src/phase1/stages/mtf/generator.py`
**Impact**: Mild lookahead bias when regime features included

### Problem
Internal calculations use shift(1), but FINAL regime output columns are NOT shifted.

### Fix
Add final shift(1) to all regime outputs:
```python
df['volatility_regime'] = volatility_regime_values.shift(1)
```

---

## HIGH #2: LightGBM num_leaves/max_depth Constraint

**File**: `src/cross_validation/param_spaces.py`
**Impact**: Invalid hyperparameter combinations

### Problem
Search allows `num_leaves` to exceed `2^max_depth` (invalid).

```python
# PROBLEMATIC
"max_depth": {"low": 3, "high": 10},
"num_leaves": {"low": 20, "high": 100}  # 100 > 2^3=8!
```

### Fix
Constrain num_leaves or add dynamic constraint.

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
