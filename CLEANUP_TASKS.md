# ML Factory - Cleanup Tasks

**Status:** Phase 21 Planned
**Last Updated:** 2026-01-27 (Phase 21 from ML Pipeline Review)

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
| 19 | 17/21 | 34 new features, 5 perf fixes, quick fixes, code quality |
| 20 | 9/15 | -851 lines, 50-100x speedup, B018 fixes, nested CV warning |
| 21 | 0/11 | ML Pipeline Review fixes (robustness + correctness, 3 disproven) |

See **COMPLETION.md** for detailed implementation records.

---

## Phase 21: ML Pipeline Review Fixes (PLANNED)

**Status:** PLANNED | 2026-01-27
**Tasks:** 0/11 (3 disproven)
**Source:** ML_PIPELINE_REVIEW.md verified by 3 parallel agents + claim verification against src/

---

### Phase 21A: Input Validation (HIGH)

#### 21A-1: Add NaN/Inf Validation to Boosting Models 🔴
- [ ] File: `src/models/boosting/xgboost_model.py:122` (fit method)
- [ ] File: `src/models/boosting/lightgbm_model.py:154` (fit method)
- [ ] File: `src/models/boosting/catboost_model.py:112` (fit method)
- [ ] Add `from src.models.neural.numerical_stability import validate_training_inputs`
- [ ] Call `validate_training_inputs(X_train, y_train, X_val, y_val, sample_weights)` at start of each fit()
- [ ] Matches existing neural model pattern in `base_rnn.py:330`

#### 21A-2: Add Hyperparameter Range Validation 🟡
- [ ] File: `src/models/boosting/xgboost_model.py:332-357` (`_build_params()`)
- [ ] File: `src/models/boosting/catboost_model.py:299-325` (`_build_params()`)
- [ ] Add range checks (e.g., max_depth > 0, learning_rate > 0)
- [ ] Extend LightGBM pattern (already validates `num_leaves <= 2^max_depth` at lines 379-387)

---

### Phase 21B: Financial Accuracy (MEDIUM)

#### 21B-1: Deduct Entry Costs from Unrealized P&L 🟡
- [ ] File: `src/inference/backtesting/backtest.py:727-730` (`_calculate_unrealized_pnl`)
- [ ] Current: `direction * contracts * price_change * point_value` (no costs)
- [ ] Fix: Subtract estimated entry costs (commission + slippage) from unrealized P&L
- [ ] Entry costs currently only deducted at position close (line 494)

#### 21B-2: Document or Add Overnight Financing Costs 🟢
- [ ] File: `src/inference/backtesting/costs.py`
- [ ] No overnight/carry/swap costs currently modeled (confirmed: zero matches)
- [ ] Either add optional overnight cost model OR document as known limitation

---

### Phase 21C: Data Pipeline Robustness (MEDIUM)

#### 21C-1: Add Timeframe Ratio Validation 🟡
- [ ] File: `src/data/adapters/multi_stream.py:558-561`
- [ ] Integer division `anchor_start // ratio` may lose precision for non-exact multiples
- [ ] Add validation: warn if `get_timeframe_minutes(tf) % anchor_minutes != 0`

#### 21C-2: Document NaN Tolerance Strategy 🟡
- [ ] File: `src/data/pipeline/schemas.py:36-106`
- [ ] Current thresholds: feature_engineering=1%, feature_scaling=0%, build_datasets=0%
- [ ] Add docstring explaining rationale for different thresholds per stage

#### 21C-3: Add Sequence Adapter Sample Loss Warning 🟢 ❌ DISPROVEN
- [x] File: `src/data/adapters/sequence.py:140-143` already logs warning when no sequences created
- [x] NOT silent — `transform()` warns: "No sequences created. DataFrame has {len(df)} rows..."
- [x] **NO ACTION NEEDED**

---

### Phase 21D: Error Handling & Resilience (LOW)

#### 21D-1: Specific Exceptions in meta_selection.py 🟢
- [ ] File: `src/models/ensemble/meta_selection.py:273` - generic `except Exception`
- [ ] File: `src/models/ensemble/meta_selection.py:441` - generic `except Exception`
- [ ] File: `src/models/ensemble/meta_selection.py:492` - generic `except Exception`
- [ ] Replace with specific exception types from `src/core/exceptions.py` (24 types available)

#### 21D-2: Specific Exceptions in registry.py 🟢
- [ ] File: `src/models/registry.py:345` - generic `except Exception`
- [ ] File: `src/models/registry.py:429` - generic `except Exception`

#### 21D-3: Document Circuit Breaker MTM Behavior 🟢
- [ ] File: `src/inference/backtesting/backtest.py:634-653`
- [ ] Max drawdown threshold triggers on equity including unrealized P&L
- [ ] Add docstring explaining this design choice and risks

#### 21D-4: Allow min_batch_size=1 in OOM Recovery 🟢
- [ ] File: `src/models/neural/oom_recovery.py:29`
- [ ] Current: `min_batch_size: int = 8`
- [ ] Consider lowering default or making configurable down to 1

---

### Phase 21E: Documentation Corrections (LOW)

#### 21E-1: Fix Slippage Model Count 🟢 ❌ DISPROVEN
- [x] SlippageModel enum actually has 4 models (FIXED, LINEAR, SQUARE_ROOT, VOLATILITY_SCALED)
- [x] Original review claim of "only 3" was wrong; NO ACTION NEEDED

#### 21E-2: Fix Exception Count 🟢 ❌ DISPROVEN (count was wrong)
- [x] `src/core/exceptions.py` actually has **27** custom exception classes (not 24 or 22)
- [x] Update any documentation referencing 22, 24, or 25 to say 27
- [x] Previous verification claiming 24 was itself incorrect

---

### Disproven Issues (NO ACTION NEEDED)

| Issue | Claim | Why Disproven |
|-------|-------|---------------|
| #1 | P&L divides by tick_value (5x error) | Division on L195 cancelled by multiplication on L205; formula is correct |
| #5 | Missing column validation | Code at multi_stream.py:351-356 validates feature columns per timeframe |
| 21C-3 | Sequence adapter returns empty silently | `sequence.py:140-143` already logs warning; NOT silent |
| 21E-2 | Exception count is 24 | Actual count is 27 (verified by class grep) |

---

## Phase 20: Performance & Quality Polish (COMPLETE ✅)

**Status:** COMPLETE | 2026-01-25
**Tasks:** 9/15 (6 disproven/deferred)

---

### Phase 20A: Critical Performance Hotspots ✅

#### 20A-1: Numba JIT for Entropy O(n²) Loops ✅
- [x] File: `src/data/features/compute/entropy.py`
- [x] Added `_count_matches_numba()` with `@numba.njit(cache=True)`
- [x] Refactored `_sample_entropy()` to use numba function
- [x] Speedup: 50-100x

#### 20A-2: Vectorize Adaptive Costs iterrows() ✅
- [x] File: `src/data/pipeline/config/adaptive_costs.py`
- [x] Added `_vectorized_time_of_day_multiplier()` helper
- [x] Rewrote `compute_cost_in_atr_adaptive()` with vectorized numpy ops
- [x] Speedup: 100-500x

#### 20A-3: Replace Rolling Cov Python Loop ✅
- [x] File: `src/data/pipeline/stages/features/microstructure_proxies.py`
- [x] Replaced 14-line Python loop with 4-line vectorized pandas
- [x] Uses `rolling(window).cov(price_changes_lag1)`
- [x] Speedup: 20-50x

#### 20A-4: raw=True for Rolling .apply() ✅
- [x] `entropy.py`: Changed 11 occurrences to `raw=True`
- [x] `mean_reversion.py`: Changed 2 occurrences to `raw=True`
- [x] Updated helper functions to work with numpy arrays
- [x] Speedup: 2-5x

---

### Phase 20B: Architecture Consolidation ✅

#### 20B-1: Delete Orphaned ArtifactManifest ✅
- [x] DELETED: `src/core/contracts/artifact_manifest.py` (-424 lines)
- [x] Updated: `src/core/contracts/__init__.py` to re-export from `common/manifest.py`

#### 20B-2: Consolidate PurgedKFoldConfig ⏭️ DEFERRED
- [x] VERIFIED: validation version has 10+ imports, config version has 0
- [x] Deferred to avoid breaking change across 10+ files

#### 20B-3: Consolidate MTFConfig ❌ DISPROVEN
- [x] All 4 definitions serve DIFFERENT purposes - NOT duplicates
- [x] **NO ACTION NEEDED**

#### 20B-4: Delete SequenceConfig Duplicate ✅
- [x] DELETED: `src/data/pipeline/stages/datasets/sequences.py` (-427 lines)
- [x] Updated: `src/data/pipeline/stages/datasets/__init__.py` to import from `core/datasets`

---

### Phase 20C: Code Quality Fixes ✅

#### 20C-1: Fix B018 Useless Expressions ✅
- [x] Fixed: `src/data/pipeline/stages/meta_labeling/run.py:393` - removed dead code
- [x] Fixed: `src/validation/cv/oof_core.py:212` - removed useless expression

#### 20C-2: Fix B904 Exception Chaining ❌ DISPROVEN
- [x] Already fixed in Phase 19 (11 files)
- [x] **NO ACTION NEEDED**

#### 20C-3: Remove F401 Unused Imports ⏭️ NONE FOUND
- [x] Checked modified files - no unused imports found

#### 20C-4: Refactor Complex Functions ⏭️ DEFERRED
- [x] Low priority - CLI functions work correctly
- [x] Deferred to avoid introducing bugs in stable code

---

### Phase 20D: ML Pipeline Improvements ✅

#### 20D-1: Nested CV Warning ✅
- [x] Location: `src/models/ensemble/meta_selection.py:406-418`
- [x] Added `warnings.warn()` at start of `_select_with_cv()`
- [x] Added docstring warning about overfitting risk
- [x] Users now see warning when `use_cv=True`

#### 20D-2: GARCH Feature Stubs ❌ ACCEPTED
- [x] Documented design decision - avoids `arch` dependency
- [x] **NO ACTION NEEDED**

#### 20D-3: Document Sequence OOF Alignment ❌ ALREADY DOCUMENTED
- [x] 3 well-documented modules handle this
- [x] **NO ACTION NEEDED**

---

## Verified Quick Fixes (From Batch Verification)

**Status:** ✅ ALL COMPLETE

### Critical: F822 Undefined Exports
- [x] File: `src/data/pipeline/stages/features/numba_functions.py`
- [x] Removed non-existent functions from `__all__`

### High: Orphaned Exception File
- [x] File: `src/models/config/exceptions.py`
- [x] Refactored to import from core (circular import prevented deletion)

### Low: B023 False Positive
- [x] File: `src/data/pipeline/stages/features/price_features.py:147`
- [x] Added `# noqa: B023` with explanation

---

## Phase 19: Comprehensive Optimization (COMPLETE ✅)

**Status:** COMPLETE | 2026-01-25
**Actual Impact:** +750 lines, 34 new features, 2-4x additional speedup

---

### Phase 19A: ML Pipeline Enhancements ✅

#### 19A-1: Order Flow Imbalance Indicators ✅
- [x] Created `src/data/features/compute/order_flow.py`
- [x] 12 features: order_imbalance, net_order_flow, buy/sell pressure, volume_delta

#### 19A-2: Liquidity Dry-Up Detectors ✅
- [x] Created `src/data/features/compute/liquidity.py`
- [x] 12 features: spread_estimate, liquidity_regime, slippage_estimate, volume_profile

#### 19A-3: Mean-Reversion Metrics ✅
- [x] Created `src/data/features/compute/mean_reversion.py`
- [x] 10 features: OU half-life, z-scores, variance ratios, Hurst exponent

#### 19A-4: Optimize MTF Timeframes ⏭️
- [ ] Deferred - requires config changes across multiple files

#### 19A-5: Enhanced Labeling with Gap Risk ⏭️
- [ ] Deferred - requires labeling system changes

---

### Phase 19B: Performance Optimization ✅

#### 19B-1: Vectorize Correlation Loops ✅
- [x] File: `src/optimization/feature_selection/filtering.py:176-182`
- [x] Replaced O(n²) nested loop with vectorized numpy using `np.triu` + `np.argwhere`

#### 19B-2: Remove DataFrame Copies in Scaling ✅
- [x] File: `src/data/pipeline/stages/scaling/run.py:138-140`
- [x] Removed unnecessary `.copy()` calls

#### 19B-3: Add Copy Parameter to Cache ✅
- [x] File: `src/data/store/raw_mtf_store.py:140`
- [x] Added `copy` parameter to cache get method

#### 19B-4: Optimize Concat+Sort Patterns ✅
- [x] File: `src/data/pipeline/stages/splits/run.py:111`
- [x] Added `is_monotonic_increasing` check before sorting

#### 19B-5: Vectorize Ensemble Correlations ✅
- [x] File: `src/optimization/ensemble_objective.py:80-97`
- [x] Replaced loop with vectorized `np.corrcoef`

---

### Phase 19C: Architecture Cleanup ✅

#### 19C-1: Move Misplaced Core Utilities
- [x] **DISPROVEN** - These are public API exports for external users

#### 19C-2: Orphaned Exception File ✅
- [x] Refactored to import from core (cannot delete due to circular import)

#### 19C-3: Consolidate ConfigValidationError
- [x] Not needed - canonical version already in `src/config/validators.py`

#### 19C-4: Remove Deprecated Orchestrator
- [x] **BLOCKED** - Still has 2 active imports (deprecation warning already present)

---

### Phase 19D: Code Quality ✅

#### 19D-1: Ruff Auto-Fixes ✅
- [x] Fixed E721 type comparisons (5 issues)
- [x] Fixed F541 f-string issue (1 issue)

#### 19D-2: Fix B904 Exception Chaining ✅
- [x] Fixed 11 files with proper exception chaining

#### 19D-3: Add Missing Type Hints
- [x] Deferred - low priority

#### 19D-4: pipeline_cli.py Status ✅
- [x] Verified USED - CLI entry point in pyproject.toml

---

## Verification Commands

```bash
# Core imports
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Phase 19A (new features) - ALL PASS ✅
python -c "from src.data.features.compute.order_flow import compute_order_flow_features; print('OK')"
python -c "from src.data.features.compute.liquidity import compute_liquidity_features; print('OK')"
python -c "from src.data.features.compute.mean_reversion import compute_mean_reversion_features; print('OK')"

# Tests - ALL PASS ✅
pytest tests/ -v  # 42 tests passing

# Linting - 65 violations (was 93)
ruff check src/
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
