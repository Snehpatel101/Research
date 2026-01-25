# ML Factory - Cleanup Tasks

**Status:** Phases 0-18 Complete | Phase 19 Planned
**Last Updated:** 2026-01-25

---

## Completed Tasks Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|------------------|
| 0 | 7/7 | Deduplication (-5,336 lines) |
| 1 | 7/7 | Contract enforcement, blocking validation |
| 2 | 7/7 | 4D infrastructure, MultiStreamAdapter |
| 3 | 6/6 | 5-dimension Optuna, FeatureSpec |
| 4 | 2/7 | Leakage/lookahead wiring (rest deferred) |
| 5 | 4/5 | MLFactory, ExperimentConfig |
| 6 | 6/6 | 6 advanced models (InceptionTime, TFT, etc.) |
| 7 | 4/4 | Production hardening, schemas |
| 8 | 4/4 | Utils consolidation, exceptions |
| 9 | 2/2 | Directory cleanup (-12 dirs) |
| 10 | 1/2 | Partial refactor |
| 12 | 37/39 | Trading profitability, tests, circuit breakers |
| 12.5 | 8/8 | Ruff 210→93, StageName enum |
| 13 | 7/7 | MTF cache, batch inference |
| 14 | 7/7 | Dynamic purge, mandatory shift, NaN monitoring |
| 15 | 5/5 | Market hours, bet sizing integration |
| 16 | 5/5 | Diversity objective, meta-learner selection, second-level stacking |
| 17 | 5/5 | Checkpointing, timeout, retry, circuit breakers |
| 18 | 2/3 | DataContractViolation consolidation |

See **COMPLETION.md** for detailed implementation records.

---

## Phase 19: Comprehensive Optimization (NEW)

**Status:** PLANNED | 6-Agent Analysis Complete
**Estimated Impact:** +0.35-0.65 Sharpe, 2-4x additional speedup

---

### Phase 19A: ML Pipeline Enhancements

**Priority:** HIGH | **Estimated Sharpe Impact:** +0.30-0.55

#### 19A-1: Order Flow Imbalance Indicators
- [ ] Create `src/data/features/compute/order_flow.py`
- [ ] Implement `compute_order_imbalance(df)` - estimate buy/sell volume from OHLC
- [ ] Implement `compute_net_order_flow(df)` - cumulative order imbalance
- [ ] Implement `compute_buy_sell_pressure(df)` - pressure index
- [ ] Add 6-8 features to BASE_FEATURE_SETS
- [ ] Wire into feature engineering pipeline

**Code Pattern:**
```python
def compute_order_imbalance(df):
    """Estimate buy/sell volume from high/low intra-bar movement."""
    buy_vol = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'])
    sell_vol = df['volume'] - buy_vol
    return buy_vol / (buy_vol + sell_vol + 1e-10)
```

#### 19A-2: Liquidity Dry-Up Detectors
- [ ] Create `src/data/features/compute/liquidity.py`
- [ ] Implement `estimate_bid_ask_spread(df)` - spread from OHLC
- [ ] Implement `liquidity_regime(df, window=20)` - high/normal/low classification
- [ ] Implement `slippage_predictor(df)` - expected slippage estimate
- [ ] Add 4-6 features to BASE_FEATURE_SETS

**Code Pattern:**
```python
def estimate_bid_ask_spread(df):
    """Estimate spread from OHLC."""
    return (df['high'] - df['low']) / df['close']
```

#### 19A-3: Mean-Reversion Metrics
- [ ] Create `src/data/features/compute/mean_reversion.py`
- [ ] Implement `ornstein_uhlenbeck_halflife(prices, window=60)`
- [ ] Implement `mean_reversion_zscore(prices, window=20)`
- [ ] Implement `variance_ratio_statistic(prices, lags=[2,4,8,16])`
- [ ] Implement `hurst_exponent(prices, max_lag=20)`
- [ ] Add 8-10 features to BASE_FEATURE_SETS

#### 19A-4: Optimize MTF Timeframes
- [ ] Update `src/core/constants.py`: `DEFAULT_MTF_TIMEFRAMES = ["1min", "5min", "15min", "30min", "60min"]`
- [ ] Update `src/config/training.py`: mtf_timeframes default
- [ ] Remove redundant 10min, 20min, 25min, 45min from default set
- [ ] Verify -35% compute reduction

#### 19A-5: Enhanced Labeling with Gap Risk
- [ ] Modify `src/data/labeling/triple_barrier.py`
- [ ] Add gap detection: `gap_detected = prices.diff().abs() > prices * 0.02`
- [ ] Handle gap events with immediate label assignment
- [ ] Add gap_risk quality penalty to sample weights

---

### Phase 19B: Performance Optimization

**Priority:** HIGH | **Estimated Speedup:** 2-4x additional

#### 19B-1: Vectorize Correlation Loops
- [ ] File: `src/optimization/feature_selection/filtering.py:176-195`
- [ ] Replace nested for loop with vectorized numpy:
```python
# Before (O(n²)):
for i, col1 in enumerate(feature_cols):
    for j, col2 in enumerate(feature_cols):
        if corr_val >= threshold: union(col1, col2)

# After (vectorized):
mask = np.abs(np.triu(corr_matrix.values, k=1)) >= threshold
for (i, j) in zip(*np.where(mask)):
    union(feature_cols[i], feature_cols[j])
```
- [ ] Verify 3-5x speedup for 200+ features

#### 19B-2: Remove DataFrame Copies in Scaling
- [ ] File: `src/data/pipeline/stages/scaling/run.py:138-140`
- [ ] Change `.copy()` to view:
```python
# Before:
train_df = df.iloc[train_indices].copy()
# After:
train_df = df.iloc[train_indices]  # Scaler creates new arrays internally
```
- [ ] Verify 1.5-2x speedup, -1.5GB memory

#### 19B-3: Remove Cache Copy on Hit
- [ ] File: `src/data/store/raw_mtf_store.py:140`
- [ ] Change `return entry.df.copy()` to return view or copy-on-write
- [ ] Consider adding immutable flag to cache entries
- [ ] Verify 1.2-1.5x speedup for Optuna

#### 19B-4: Optimize Concat+Sort Patterns
- [ ] File: `src/data/pipeline/stages/splits/run.py:111`
- [ ] Avoid double memory allocation from concat + sort_values + reset_index
- [ ] Use `sort_index()` instead of `sort_values("datetime")` where possible
- [ ] Verify 1.3-1.8x speedup

#### 19B-5: Pre-compute Correlations in Ensemble
- [ ] File: `src/optimization/ensemble_objective.py:80-97`
- [ ] Move correlation computation outside loop:
```python
# Before (in loop):
for i in range(n_existing):
    corr = np.corrcoef(new_predictions, existing_predictions[:, i])[0, 1]

# After (vectorized):
all_preds = np.hstack([new_predictions[:, None], existing_predictions])
corr_matrix = np.corrcoef(all_preds.T)
correlations = np.abs(corr_matrix[0, 1:])
```
- [ ] Verify 1.5-2x speedup for ensemble optimization

---

### Phase 19C: Architecture Cleanup

**Priority:** MEDIUM

#### 19C-1: Move Misplaced Core Utilities
- [ ] Move `src/core/utils/notebook.py` → `src/models/utils/notebook.py`
- [ ] Move `src/core/utils/colab_setup.py` → `src/models/utils/colab.py`
- [ ] Move `src/core/utils/device_utils.py` → `src/models/device/utils.py`
- [ ] Update all imports (expect ~5-10 files)
- [ ] Verify no circular imports introduced

#### 19C-2: Delete Duplicate Exception File
- [ ] Verify no imports: `grep -r "from src.models.config.exceptions import" src/`
- [ ] Delete `src/models/config/exceptions.py`
- [ ] Verify tests still pass

#### 19C-3: Consolidate ConfigValidationError
- [ ] Add `ConfigValidationError` to `src/core/exceptions.py` (if not present)
- [ ] Update `src/config/validators.py` to import from core
- [ ] Remove local definition

#### 19C-4: Remove Deprecated Orchestrator
- [ ] Verify no direct imports of `src/orchestrator.py` except lazy export
- [ ] Update `src/__init__.py` to remove MLPipeline export
- [ ] Delete `src/orchestrator.py`
- [ ] Update CLI if needed

---

### Phase 19D: Code Quality

**Priority:** LOW

#### 19D-1: Ruff Auto-Fixes
```bash
# Run in order:
ruff check src/ --select E402,I001 --fix  # Import ordering (14)
ruff check src/ --select UP038 --fix       # isinstance syntax (29)
ruff check src/ --select SIM102,SIM108 --fix  # Nested if (30)
ruff check src/ --select E721 --fix        # Type comparisons (5)
```
- [ ] Verify 77 violations fixed
- [ ] Run tests after each batch

#### 19D-2: Fix B904 Exception Chaining
- [ ] File: `src/config/utils.py` - 7 violations
- [ ] File: `src/config/validators.py` - 5 violations
- [ ] Add `from err` or `from None` to all `raise` in `except` blocks
- [ ] Verify 19 violations fixed

#### 19D-3: Add Missing Type Hints
- [ ] `src/models/training/unified_orchestrator.py:1176` - add return type to `_train_meta_labeling_for_horizon()`
- [ ] `src/optimization/five_dimension_objective.py:367` - add `Callable[[optuna.Trial], float]` return type
- [ ] `src/models/ensemble/stacking.py:213` - add type coverage to `fit()`
- [ ] `src/models/ensemble/diversity.py` - add return types to public methods
- [ ] `src/inference/bundle.py` - replace `Any` with specific types

#### 19D-4: Investigate Orphaned File
- [ ] Check `src/pipeline_cli.py` - is it used or legacy?
- [ ] If legacy: delete
- [ ] If used: document purpose

---

## Verification Commands

```bash
# Core imports
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Phase 19A (new features)
python -c "from src.data.features.compute.order_flow import compute_order_imbalance; print('OK')"
python -c "from src.data.features.compute.liquidity import estimate_bid_ask_spread; print('OK')"
python -c "from src.data.features.compute.mean_reversion import ornstein_uhlenbeck_halflife; print('OK')"

# Tests
pytest tests/ -v  # 42+ tests passing

# Linting
ruff check src/  # Target: <50 violations
```

---

## Remaining: Phase 11 Deferred Items

Low priority backlog items:

| Task | Description | File Location | Notes |
|------|-------------|---------------|-------|
| 5C | Unified deployment bundle | `src/inference/bundle.py` | Needs spec |
| 4D | Deflated Sharpe Ratio | `src/validation/` | Post-Optuna gate |
| 4E | Bootstrap CIs | `src/validation/metrics/` | Wire BootstrapCI |
| 4F | Auto calibration | `src/models/training/` | Wire CalibrationManager |
| - | MTF ablation flag | `src/config/` | Add `mtf.enabled` |

**Priority:** Low - System is production-ready without these

---

## Verification Commands

```bash
# Core imports
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Phase 16 (Ensemble)
python -c "from src.optimization import diversity_aware_objective; print('OK')"
python -c "from src.models.ensemble import SecondLevelStacker; print('OK')"

# Phase 17 (Resilience)
python -c "from src.core.resilience import timeout, CircuitBreaker, retry; print('OK')"
python -c "from src.core.checkpoint import PipelineCheckpointManager; print('OK')"

# Tests
pytest tests/ -v  # 42 tests passing

# Linting
ruff check src/  # Should pass on new files
```

---

*For full implementation details, see COMPLETION.md*
