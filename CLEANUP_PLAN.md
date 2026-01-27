# Cleanup Plan: ML Factory

**Status:** Phase 21 Planned
**Last Updated:** 2026-01-27 (Phase 21 from ML Pipeline Review)

---

## Completed Phases

All major phases complete. See **COMPLETION.md** for full implementation details.

| Phase | Description | Impact | Date |
|-------|-------------|--------|------|
| 0 | Deduplication | -5,336 lines | 2026-01-23 |
| 1 | Contract Enforcement | +616 lines | 2026-01-23 |
| 2 | 4D Infrastructure | +958 lines | 2026-01-24 |
| 3 | 5-Dimension Optuna | +2,298 lines | 2026-01-24 |
| 4 | Validation Integration | +50 lines | 2026-01-24 |
| 5 | Unified Entry Point | +1,281 lines | 2026-01-24 |
| 6 | Advanced Models | +3,690 lines | 2026-01-24 |
| 7 | Production Hardening | +850 lines | 2026-01-24 |
| 8 | Code Consolidation | +650 lines | 2026-01-24 |
| 9 | Directory Cleanup | -12 dirs | 2026-01-24 |
| 10 | Refactor (partial) | +25 lines | 2026-01-24 |
| 12 | Trading Profitability | +5,780 lines | 2026-01-24 |
| 12.5 | Code Quality Pass | Ruff 210→93 | 2026-01-25 |
| 13 | Performance Optimization | +504 lines | 2026-01-25 |
| 14 | Data Quality Hardening | +450 lines | 2026-01-25 |
| 15 | Backtesting Realism | 5 tasks | 2026-01-25 |
| 16 | Ensemble Optimization | +1,480 lines | 2026-01-25 |
| 17 | Architecture Resilience | +750 lines | 2026-01-25 |
| 18 | Code Cleanup | 2/3 tasks | 2026-01-25 |
| 19 | Comprehensive Optimization | +750 lines, 34 features | 2026-01-25 |
| 20 | Performance & Quality Polish | -851 lines, 50-100x speedup | 2026-01-25 |
| 21 | ML Pipeline Review Fixes | Robustness + correctness (3 disproven) | PLANNED |

**Net Impact:** ~+11,900 lines of production infrastructure

---

## Phase 20: Performance & Quality Polish (COMPLETE ✅)

**Status:** COMPLETE | 2026-01-25
**Actual Impact:** -851 lines, 50-100x speedup on critical paths

### Phase 20A: Critical Performance Hotspots ✅

| Task | Description | Speedup | Status |
|------|-------------|---------|--------|
| 20A-1 | Numba JIT for entropy.py O(n²) loops | 50-100x | ✅ Complete |
| 20A-2 | Vectorize adaptive_costs.py iterrows() | 100-500x | ✅ Complete |
| 20A-3 | Replace rolling cov Python loop | 20-50x | ✅ Complete |
| 20A-4 | raw=True for rolling .apply() patterns | 2-5x | ✅ Complete |

### Phase 20B: Architecture Consolidation ✅

| Task | Description | Status |
|------|-------------|--------|
| 20B-1 | Delete orphaned `contracts/artifact_manifest.py` (-424 lines) | ✅ Complete |
| 20B-2 | PurgedKFoldConfig consolidation | ⏭️ Deferred (breaking change) |
| 20B-3 | MTFConfig 4 definitions | ❌ DISPROVEN (different purposes) |
| 20B-4 | Delete SequenceConfig duplicate (-427 lines) | ✅ Complete |

### Phase 20C: Code Quality Fixes ✅

| Task | Description | Status |
|------|-------------|--------|
| 20C-1 | Fix B018 useless expressions (2 bugs) | ✅ Complete |
| 20C-2 | Fix B904 exception chaining | ❌ DISPROVEN (fixed in Phase 19) |
| 20C-3 | Remove F401 unused imports | ⏭️ None found |
| 20C-4 | Refactor complex functions | ⏭️ Deferred (low priority) |

### Phase 20D: ML Pipeline Improvements ✅

| Task | Description | Status |
|------|-------------|--------|
| 20D-1 | Add nested CV warning in meta_selection.py | ✅ Complete |
| 20D-2 | GARCH stubs | ❌ ACCEPTED (documented design decision) |
| 20D-3 | Sequence OOF alignment | ❌ ALREADY DOCUMENTED |

---

## Phase 21: ML Pipeline Review Fixes (PLANNED)

**Status:** PLANNED | 2026-01-27
**Source:** ML_PIPELINE_REVIEW.md (verified by 3 parallel agents against src/)
**Scope:** Fix verified issues from comprehensive pipeline review

### Review Verification Summary

| Issue | Review Claim | Agent Verdict |
|-------|-------------|---------------|
| #1 P&L bug (financial_report.py) | 5x understatement | **FALSE** - division/multiplication cancels out |
| #2 Boosting NaN validation | Missing in all 3 | **TRUE** - confirmed |
| #3 Unrealized P&L costs | Overstates equity | **TRUE** - confirmed |
| #4 Multi-stream integer division | Temporal misalignment | **TRUE** - confirmed |
| #5 Feature column validation | Missing cross-TF check | **FALSE** - validation exists at L351-356 |
| #6 Hyperparameter validation | Incomplete | **PARTIALLY TRUE** - XGB/Cat lack, LGB has partial |
| #7 NaN tolerance inconsistency | 1% vs 0% thresholds | **TRUE** - confirmed |
| #8 Sequence adapter sample loss | Silent empty return | **FALSE** - sequence.py:140-143 already logs warning |
| #9 Circuit breaker MTM equity | Unrealized included | **TRUE** - confirmed |
| #10 OOM min_batch_size=8 | Too high | **TRUE** - confirmed |
| #11 No overnight costs | Missing carry/swap | **TRUE** - confirmed |
| #12 Generic except Exception | Mixed patterns | **TRUE** - confirmed at 5 locations |
| Strength: 4 slippage models | 4 models claimed | **TRUE** - 4 in enum (FIXED, LINEAR, SQUARE_ROOT, VOLATILITY_SCALED) |
| Strength: 22 exceptions | 22 types claimed | **MINOR ERROR** - actually 27 |

### Phase 21A: Input Validation (HIGH)

| Task | Description | Priority |
|------|-------------|----------|
| 21A-1 | Add NaN/Inf validation to boosting model fit() methods | 🔴 HIGH |
| 21A-2 | Add hyperparameter range validation to XGBoost/CatBoost `_build_params()` | 🟡 MEDIUM |

### Phase 21B: Financial Accuracy (MEDIUM)

| Task | Description | Priority |
|------|-------------|----------|
| 21B-1 | Deduct entry costs from unrealized P&L in backtest.py | 🟡 MEDIUM |
| 21B-2 | Document or add overnight financing costs | 🟢 LOW |

### Phase 21C: Data Pipeline Robustness (MEDIUM)

| Task | Description | Priority |
|------|-------------|----------|
| 21C-1 | Add timeframe ratio validation in multi_stream.py | 🟡 MEDIUM |
| 21C-2 | Document NaN tolerance strategy across pipeline stages | 🟡 MEDIUM |
| 21C-3 | ~~Add warning log for sequence adapter sample loss >10%~~ ❌ DISPROVEN (already warns at sequence.py:140-143) | ❌ N/A |

### Phase 21D: Error Handling & Resilience (LOW)

| Task | Description | Priority |
|------|-------------|----------|
| 21D-1 | Replace generic `except Exception` in meta_selection.py (3 locations) | 🟢 LOW |
| 21D-2 | Replace generic `except Exception` in `src/models/registry.py` (lines 345, 429) | 🟢 LOW |
| 21D-3 | Document circuit breaker MTM equity behavior | 🟢 LOW |
| 21D-4 | Allow min_batch_size=1 in OOM recovery config | 🟢 LOW |

### Phase 21E: Documentation Corrections (LOW)

| Task | Description | Priority |
|------|-------------|----------|
| 21E-1 | ~~Fix slippage model count~~ ❌ DISPROVEN (enum has 4, claim was wrong) | ❌ N/A |
| 21E-2 | ~~Fix exception count~~ ❌ DISPROVEN (actual count is 27, not 24 or 22) | ❌ N/A |

### Validation Criteria

```bash
# 21A-1: Boosting validation
grep -rn "validate_training_inputs" src/models/boosting/  # Should find matches after fix

# 21B-1: Unrealized P&L
grep -n "entry_cost\|entry_commission" src/inference/backtesting/backtest.py  # In unrealized calc

# 21C-1: Timeframe ratio
grep -n "not exact multiple\|ratio.*validation" src/data/adapters/multi_stream.py
```

### Disproven Issues (NO ACTION NEEDED)

- **Issue #1:** P&L calculation in financial_report.py - division by tick_value is cancelled by multiplication; formula is correct despite being ugly
- **Issue #5:** Feature column validation - code at multi_stream.py:351-356 already validates columns per timeframe
- **Issue #8:** Sequence adapter sample loss - `sequence.py:140-143` already logs warning; NOT silent
- **21E-1:** SlippageModel enum has 4 models (FIXED, LINEAR, SQUARE_ROOT, VOLATILITY_SCALED), not 3 as review claimed
- **21E-2:** Exception count is 27 (not 24 or 22) - verified by class grep of `core/exceptions.py`

---

## Verified Quick Fixes (From Batch Verification)

**Status:** ✅ ALL FIXED (Phase 19)

| Priority | Item | Location | Status |
|----------|------|----------|--------|
| 🔴 Critical | F822 undefined exports | `numba_functions.py` | ✅ Fixed |
| 🟠 High | Orphaned exceptions file | `models/config/exceptions.py` | ✅ Refactored (circular import) |
| ⚪ Low | B023 false positive | `price_features.py:147` | ✅ Added noqa |

**Disproven Claims (Confirmed NOT bugs):**
- B023 loop variable closure - False positive, lambda executed immediately
- notebook.py/colab_setup.py dead - Re-exported for external users
- orchestrator.py deleted - Still has 2 active imports (deprecation warning added)

---

## Phase 19: Comprehensive Optimization (COMPLETE ✅)

**Status:** COMPLETE | 2026-01-25
**Actual Impact:** +750 lines, 34 new features, 2-4x additional speedup

### Phase 19A: ML Pipeline Enhancements ✅

| Task | Description | Status |
|------|-------------|--------|
| 19A-1 | Order Flow Imbalance indicators (12 features) | ✅ Complete |
| 19A-2 | Liquidity Dry-Up detectors (12 features) | ✅ Complete |
| 19A-3 | Mean-Reversion metrics (10 features) | ✅ Complete |
| 19A-4 | MTF optimization | ⏭️ Deferred |
| 19A-5 | Gap risk handling | ⏭️ Deferred |

**New Files Created:**
- `src/data/features/compute/order_flow.py` (12 features)
- `src/data/features/compute/liquidity.py` (12 features)
- `src/data/features/compute/mean_reversion.py` (10 features)

### Phase 19B: Performance Optimization ✅

| Task | Description | Status |
|------|-------------|--------|
| 19B-1 | Vectorize O(n²) correlation loops | ✅ Fixed |
| 19B-2 | Remove DataFrame copies in scaling | ✅ Fixed |
| 19B-3 | Add copy parameter to cache | ✅ Fixed |
| 19B-4 | Optimize concat+sort patterns | ✅ Fixed |
| 19B-5 | Vectorize ensemble correlations | ✅ Fixed |

### Phase 19C: Architecture Cleanup ✅

| Task | Description | Status |
|------|-------------|--------|
| 19C-1 | Move misplaced utilities | ❌ DISPROVEN (public API) |
| 19C-2 | Delete/refactor exceptions.py | ✅ Refactored (circular import) |
| 19C-3 | Consolidate ConfigValidationError | ⏭️ Not needed |
| 19C-4 | Remove orchestrator.py | ⏸️ BLOCKED (2 active imports) |

### Phase 19D: Code Quality ✅

| Task | Description | Status |
|------|-------------|--------|
| 19D-1 | Ruff auto-fixes | ✅ E721, F541 fixed |
| 19D-2 | B904 exception chaining | ✅ 11 files fixed |
| 19D-3 | Type hints | ⏭️ Deferred |
| 19D-4 | pipeline_cli.py status | ✅ Verified USED (CLI entry point)

---

## Remaining Work: Phase 11 Deferred Items (Low Priority)

These are backlog items that were intentionally deferred:

| Task | Description | Notes |
|------|-------------|-------|
| 5C | Unified deployment bundle (tar.gz) | Needs bundle spec |
| 4D | Deflated Sharpe Ratio post-Optuna | Add DSR gate |
| 4E | Bootstrap CIs in financial reports | Wire BootstrapCI |
| 4F | Auto calibration in orchestrator | Wire CalibrationManager |
| - | MTF ablation flag | Add `mtf.enabled` config |

**Priority:** Low - System is production-ready without these

---

## Quick Reference: What's Implemented

### Core Guarantees
- ✅ Sharpe ratio optimization (not F1 score)
- ✅ No data leakage (purge/embargo in all CV)
- ✅ No lookahead bias (mandatory shift, audit)
- ✅ Reproducible (same config = same output)

### Trading Infrastructure
- ✅ Circuit breakers (drawdown, daily loss, consecutive losses)
- ✅ R-multiple tracking for every trade
- ✅ Realistic costs (volatility-scaled slippage)
- ✅ Market hours filtering

### Performance
- ✅ 10-50x speedup (caching, parallel, GPU, batch inference)
- ✅ 2-4x additional speedup (Phase 19 vectorization)
- ✅ 42 tests passing
- ✅ MLflow auto-enabled

### New Features (Phase 19)
- ✅ 34 new ML features (order flow, liquidity, mean-reversion)
- ✅ 196 total features (was 162)
- ✅ Ruff violations: 65 (was 93)

### Resilience
- ✅ Pipeline checkpointing (resume from failures)
- ✅ Timeout protection
- ✅ Retry with exponential backoff
- ✅ Circuit breaker pattern

---

*For implementation details, see COMPLETION.md*
