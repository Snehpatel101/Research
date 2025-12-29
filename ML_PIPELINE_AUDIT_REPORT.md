# ML Pipeline Comprehensive Audit Report

**Date:** 2025-12-29
**Audited By:** 20 Specialized AI Agents
**Project:** TOPSTEPX ML Model Factory
**Total Tests in Suite:** 2,060

---

## Executive Summary

This document presents the findings from a comprehensive audit of the ML pipeline using 20 specialized agents. The audit covered all aspects of the pipeline including lookahead bias, label leakage, cross-validation, model implementations, and end-to-end integration.

**Overall Assessment:** The pipeline is well-architected with 3 critical issues requiring immediate attention.

| Category | Count |
|----------|-------|
| Critical Issues | 3 |
| High Severity | 2 |
| Moderate Issues | 3 |
| Well-Implemented | 14 |

---

## Critical Issues (Must Fix Immediately)

### Issue #1: HMM Regime Detection - SEVERE LOOKAHEAD BIAS

**Agent:** 12 (Regime Detection)
**Severity:** CRITICAL
**Location:** `src/phase1/stages/regime/hmm.py` lines 329-354

**Problem:**
When `expanding=True` (the DEFAULT setting), the HMM is trained on the ENTIRE dataset including future data, then states are assigned to ALL bars retroactively. This means bar[0] gets a regime label that was determined using data from bar[N].

**Current Code (Problematic):**
```python
if self.expanding:
    window_obs = observations[:n_samples]  # ALL observations including future!
    self._fitted_model, raw_states, raw_probs = fit_gaussian_hmm(
        window_obs,  # Trains on ENTIRE dataset
        ...
    )
    states = ordered_states  # Applied to ALL bars retroactively
```

**Fix Options:**

**Option A - Disable Expanding Mode (Simplest):**
```python
# In hmm.py config or usage
expanding = False  # Use rolling mode only for production
```

**Option B - Fix Expanding Mode to Use Only Past Data:**
```python
if self.expanding:
    # Train incrementally, only using past data at each point
    for i in range(min_samples, n_samples):
        window_obs = observations[:i]  # Only data up to current bar
        model, _, _ = fit_gaussian_hmm(window_obs, ...)
        # Predict only current bar using model trained on past
        current_window = observations[max(0, i-lookback):i+1]
        states[i] = model.predict(current_window)[-1]
```

**Option C - Add Warning and Research-Only Flag:**
```python
if self.expanding:
    logger.warning(
        "EXPANDING MODE: HMM trained on full dataset. "
        "This mode has LOOKAHEAD BIAS and should only be used for research/analysis, "
        "NOT for live trading features. Use expanding=False for production."
    )
```

**Impact if Unfixed:**
Models trained with HMM regime features will have artificially inflated performance that will not generalize to live trading.

---

### Issue #2: GA Optimization - TEST DATA LEAKAGE

**Agent:** 13 (GA/Optuna Optimization)
**Severity:** CRITICAL
**Location:** `src/phase1/stages/ga_optimize/optuna_optimizer.py`

**Problem:**
Stage 5 (GA optimization) runs BEFORE Stage 7 (train/test splits). The barrier parameter optimization uses data that will later become the test set, creating label leakage.

**Pipeline Order (Current):**
```
Stage 4: Initial Labeling (ALL data)
Stage 5: GA Optimization (ALL data) ← PROBLEM: Uses future test data
Stage 6: Final Labels (ALL data)
Stage 7: Train/Val/Test Splits (70/15/15)
```

**Current Code (Problematic):**
```python
# In optuna_optimizer.py - run_optuna_optimization()
df_subset = get_contiguous_subset(df, subset_fraction, seed=seed)
# This 30% subset could include what will become test data!

# Validation on full dataset
logger.info("Validating on full dataset...")
# Explicitly validates on ALL data including future test set
```

**Fix - Use Only Pre-Split Training Data:**
```python
def run_optuna_optimization_safe(
    df: pd.DataFrame,
    horizon: int,
    train_ratio: float = 0.70,  # Match Phase 1 train ratio
    ...
) -> Tuple[Dict[str, Any], ConvergenceRecord]:
    """Optimize barrier parameters using ONLY training data portion."""

    # Only use first 70% for optimization (will become train set)
    n = len(df)
    train_end = int(n * train_ratio)
    df_train = df.iloc[:train_end].copy()

    logger.info(f"Optimizing on training portion only: {train_end:,} / {n:,} samples")

    # Optimize only on training portion
    results, record = run_optuna_optimization(df_train, horizon, ...)
    return results, record
```

**Alternative Fix - Move GA After Splits:**
```
Stage 7: Train/Val/Test Splits FIRST
Stage 5: GA Optimization (train data only)
Stage 6: Final Labels (apply optimized params to all)
```

**Impact if Unfixed:**
Test set performance will be optimistically biased because the labeling parameters were tuned to look good on that exact test data.

---

### Issue #3: Transaction Costs NOT Applied to Labels

**Agent:** 14 (Transaction Costs)
**Severity:** CRITICAL
**Location:** `src/phase1/stages/labeling/triple_barrier.py`

**Problem:**
Triple-barrier labels show GROSS profit targets. The upper barrier at `entry + k_up * ATR` doesn't deduct transaction costs. Models learn to hit gross profit targets, but real trading incurs round-trip costs (commission + slippage).

**Current Code (Problematic):**
```python
# In triple_barrier_numba()
upper_barrier = entry_price + k_up * entry_atr  # GROSS target
lower_barrier = entry_price - k_down * entry_atr
```

**Example Impact:**
- Barrier: 2.0 ATR profit target
- Transaction cost: 0.3 ATR (commission + slippage)
- Model learns: Hit 2.0 ATR = WIN
- Reality: 2.0 ATR - 0.3 ATR = 1.7 ATR net profit
- A trade hitting 1.8 ATR is labeled WIN but actually loses money after costs!

**Fix - Adjust Barriers by Transaction Cost:**
```python
def triple_barrier_numba_with_costs(
    close, high, low, atr,
    k_up, k_down, max_bars,
    cost_ticks, tick_value  # New parameters
):
    """Triple barrier with transaction cost adjustment."""

    for i in range(n - max_bars):
        entry_price = close[i]
        entry_atr = atr[i]

        # Convert cost to ATR units
        cost_in_price = cost_ticks * tick_value
        cost_in_atr = cost_in_price / entry_atr

        # Adjust barriers to account for round-trip costs
        # Option A: Make upper barrier harder to hit (recommended)
        upper_barrier = entry_price + (k_up - cost_in_atr) * entry_atr
        lower_barrier = entry_price - k_down * entry_atr

        # Option B: Adjust both barriers symmetrically
        # upper_barrier = entry_price + (k_up - cost_in_atr/2) * entry_atr
        # lower_barrier = entry_price - (k_down + cost_in_atr/2) * entry_atr
```

**Transaction Costs (Already Defined):**
```python
# From barriers_config.py
TRANSACTION_COSTS = {
    'MES': 0.5,  # ticks round-trip commission
    'MGC': 0.3,
}
SLIPPAGE_TICKS = {
    'MES': {'low_vol': 0.5, 'high_vol': 1.0},  # per fill
    'MGC': {'low_vol': 0.75, 'high_vol': 1.5},
}
# Total round-trip MES low_vol: 0.5 + 2*0.5 = 1.5 ticks = $1.875
```

**Impact if Unfixed:**
Models will appear profitable in backtests but lose money in live trading due to transaction costs eating into smaller wins and turning marginal wins into losses.

---

## High Severity Issues (Should Fix)

### Issue #4: MTF/Regime Missing shift(1) at Output

**Agent:** 1 (Lookahead Bias Detection)
**Severity:** HIGH
**Locations:**
- `src/phase1/stages/mtf/generator.py` lines 244-291
- `src/phase1/stages/regime/volatility.py` lines 176-181
- `src/phase1/stages/regime/trend.py`
- `src/phase1/stages/regime/structure.py`

**Problem:**
While internal calculations correctly use shift(1), the FINAL regime output columns are NOT shifted before being used as features. This causes lookahead bias when regime features are included in the model.

**Current Code (Problematic):**
```python
# Regime detection computes values correctly internally, but output is not shifted
df['volatility_regime'] = volatility_regime_values  # NOT SHIFTED!
df['trend_regime'] = trend_regime_values  # NOT SHIFTED!
```

**Fix - Add Final shift(1) to All Regime Outputs:**
```python
# In each regime detector's output assignment
df['volatility_regime'] = volatility_regime_values.shift(1)
df['trend_regime'] = trend_regime_values.shift(1)
df['structure_regime'] = structure_regime_values.shift(1)
df['composite_regime'] = composite_regime_values.shift(1)

# Or add a centralized shift in the composite detector
def detect_all(self, df: pd.DataFrame) -> CompositeRegimeResult:
    # ... compute all regimes ...

    # ANTI-LOOKAHEAD: Shift all regime outputs by 1 bar
    for col in regime_columns:
        df[col] = df[col].shift(1)
```

---

### Issue #5: LightGBM num_leaves/max_depth Constraint

**Agent:** 9 (Boosting Models)
**Severity:** HIGH
**Location:** `src/cross_validation/param_spaces.py`

**Problem:**
The hyperparameter search space allows invalid combinations where `num_leaves` exceeds `2^max_depth`.

**Current Code (Problematic):**
```python
"lightgbm": {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "num_leaves": {"type": "int", "low": 20, "high": 100},  # 100 > 2^3=8!
}
```

**Fix Options:**

**Option A - Constrain num_leaves Range:**
```python
"lightgbm": {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "num_leaves": {"type": "int", "low": 8, "high": 64},  # max=2^6
}
```

**Option B - Add Constraint in Tuning Code:**
```python
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves = 2 ** max_depth
    num_leaves = trial.suggest_int("num_leaves", 8, min(max_leaves, 128))
```

---

## Moderate Issues (Recommended Fixes)

### Issue #6: Configuration Inconsistency

**Agent:** 17 (Configuration System)
**Locations:**
- `src/common/horizon_config.py` - defines 8 horizons
- `src/phase1/config/features.py` - defines 4 horizons

**Fix:** Consolidate to single source of truth:
```python
# In src/common/horizon_config.py (canonical source)
SUPPORTED_HORIZONS = [1, 5, 10, 15, 20, 30, 60, 120]  # All supported
ACTIVE_HORIZONS = [5, 10, 15, 20]  # Default active set

# In features.py - import from canonical source
from src.common.horizon_config import ACTIVE_HORIZONS as LABEL_HORIZONS
```

---

### Issue #7: Neural Bidirectional Warning Missing

**Agent:** 10 (Neural Models)
**Location:** `src/models/neural/base_rnn.py`

**Problem:**
No warning when `bidirectional=True` is enabled. Bidirectional RNNs can capture non-causal patterns within the sequence window.

**Fix:**
```python
# In BaseRNNModel.fit() or __init__
if self._config.get("bidirectional", False):
    logger.warning(
        "BIDIRECTIONAL RNN ENABLED: The backward pass sees 'future' positions "
        "within each sequence window. While not technically lookahead (data is "
        "within the observed window), this can capture patterns that won't "
        "generalize to real-time inference. Consider bidirectional=False for "
        "production trading models."
    )
```

---

### Issue #8: Missing Label Distribution Tests

**Agent:** 18 (Test Suite)
**Location:** `tests/phase_1_tests/stages/labeling/`

**Recommendation:** Add tests to validate label distribution:
```python
def test_label_distribution_balanced():
    """Labels should be approximately balanced after triple barrier."""
    result = labeler.compute_labels(df, horizon=20)
    valid_labels = result.labels[result.labels != -99]
    counts = pd.Series(valid_labels).value_counts(normalize=True)

    # Each class should be between 20% and 50%
    assert 0.20 < counts.get(-1, 0) < 0.50, "Short labels out of range"
    assert 0.20 < counts.get(0, 0) < 0.50, "Neutral labels out of range"
    assert 0.20 < counts.get(1, 0) < 0.50, "Long labels out of range"
```

---

## Well-Implemented Components (No Fixes Needed)

The following components passed their audits with no issues:

| Component | Agent | Key Findings |
|-----------|-------|--------------|
| **Triple-Barrier Labeling** | 2 | Proper shift(1) on all volatility inputs, correct barrier math |
| **Purge/Embargo Splits** | 3 | 60 bars purge (3x max_horizon), 1440 bars embargo (~5 days) |
| **Scaling Pipeline** | 4 | Fit on train only, `validate_no_leakage()` check built-in |
| **PurgedKFold CV** | 5 | Follows López de Prado methodology, label-aware purging |
| **Walk-Forward Validation** | 6 | Correct temporal ordering, gap/embargo handling |
| **OOF Generation** | 7 | No leakage, proper fold handling, coverage tracking |
| **Model Registry** | 8 | 100% compliance across all 13 models |
| **Boosting Models** | 9 | Proper early stopping, sample weights, GPU support |
| **Neural Models** | 10 | Causal TCN convolutions, proper sequence alignment |
| **Ensemble Models** | 11 | Proper OOF for stacking, time-based split for blending |
| **Quality Weighting** | 15 | Correctly applied to training only |
| **Data Ingestion** | 16 | Comprehensive OHLCV validation |
| **Test Suite** | 18 | 2,060 tests, explicit lookahead/leakage tests |
| **Pipeline Integration** | 19 | Correct DAG, state management, artifact tracking |

---

## Fix Priority Order

### Phase 1: Critical Fixes (Week 1)

| Priority | Issue | File | Effort |
|----------|-------|------|--------|
| P0 | HMM expanding mode lookahead | `regime/hmm.py` | 2-4 hours |
| P0 | GA optimization test leakage | `ga_optimize/optuna_optimizer.py` | 4-8 hours |
| P0 | Transaction costs in labels | `labeling/triple_barrier.py` | 4-6 hours |

### Phase 2: High Severity (Week 2)

| Priority | Issue | File | Effort |
|----------|-------|------|--------|
| P1 | Regime output shift(1) | `regime/*.py` | 2-3 hours |
| P1 | LightGBM num_leaves constraint | `param_spaces.py` | 1 hour |

### Phase 3: Moderate Improvements (Week 3)

| Priority | Issue | File | Effort |
|----------|-------|------|--------|
| P2 | Configuration consolidation | `horizon_config.py`, `features.py` | 1 hour |
| P2 | Bidirectional warning | `neural/base_rnn.py` | 30 mins |
| P2 | Label distribution tests | `tests/` | 2 hours |

---

## Validation After Fixes

After implementing fixes, run these validation steps:

### 1. Run Full Test Suite
```bash
pytest tests/ -v --tb=short
```

### 2. Run Lookahead Invariance Tests
```bash
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v
pytest tests/validation/test_lookahead.py -v
```

### 3. Run Split Leakage Tests
```bash
pytest tests/phase_1_tests/stages/test_splits_leakage.py -v
```

### 4. Verify Pipeline End-to-End
```bash
./pipeline run --symbols MES --dry-run  # Validate config
./pipeline run --symbols MES            # Full run
```

### 5. Compare Before/After Metrics
```bash
# Run CV before and after fixes
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# If Sharpe drops significantly after fixes, the original metrics
# were inflated by the bugs (expected and correct behavior)
```

---

## Appendix: Agent Summary Table

| Agent # | Focus Area | Verdict | Critical Issues |
|---------|------------|---------|-----------------|
| 1 | Lookahead Bias - Features | PASS (minor) | MTF/regime shift missing |
| 2 | Triple-Barrier Labeling | PASS | None |
| 3 | Purge/Embargo Splits | PASS | None |
| 4 | Scaling Pipeline | PASS | None |
| 5 | PurgedKFold CV | PASS | None |
| 6 | Walk-Forward Validation | PASS | None |
| 7 | OOF Generation | PASS | None |
| 8 | Model Registry | PASS | None |
| 9 | Boosting Models | PASS (minor) | num_leaves constraint |
| 10 | Neural Models | PASS (minor) | bidirectional warning |
| 11 | Ensemble Models | PASS | None |
| 12 | Regime Detection | FAIL | HMM expanding lookahead |
| 13 | GA/Optuna Optimization | FAIL | Test data leakage |
| 14 | Transaction Costs | FAIL | Costs not in labels |
| 15 | Quality Weighting | PASS | None |
| 16 | Data Ingestion | PASS | None |
| 17 | Configuration System | PASS (minor) | Horizon inconsistency |
| 18 | Test Suite | PASS | None |
| 19 | Pipeline Integration | PASS | None |

---

## Conclusion

The ML pipeline is fundamentally well-designed with proper attention to time-series specific concerns like purge/embargo, lookahead prevention, and model standardization. The **3 critical issues** are common pitfalls in ML for trading:

1. **HMM lookahead** - Expanding window training on future data
2. **GA optimization timing** - Parameter tuning before train/test split
3. **Transaction costs** - Labels show gross, trades pay net

After fixing these issues, the pipeline will produce **realistic, unbiased performance estimates** that should generalize to live trading.

**Estimated Total Fix Time:** 15-25 hours for all critical and high severity issues.
