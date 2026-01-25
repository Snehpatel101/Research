# Cleanup Plan: ML Factory

**Status:** Phases 0-18 Complete | Phase 19 Planned
**Last Updated:** 2026-01-25

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

**Net Impact:** ~+12,000 lines of production infrastructure

---

## Phase 19: Comprehensive Optimization (NEW)

**Status:** PLANNED | Analysis Complete
**Estimated Impact:** +0.35-0.65 Sharpe, 2-4x additional speedup

### Overview

6-agent comprehensive analysis identified 8 prioritized improvements across ML pipeline, performance, and code quality.

### Phase 19A: ML Pipeline Enhancements (HIGH PRIORITY)

**Estimated Sharpe Impact:** +0.30-0.55

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 19A-1 | Add Order Flow Imbalance indicators (6-8 features) | +0.08-0.15 Sharpe | 1 day |
| 19A-2 | Add Liquidity Dry-Up detectors (4-6 features) | +0.05-0.10 Sharpe | 1 day |
| 19A-3 | Add Mean-Reversion metrics (8-10 features) | +0.08-0.15 Sharpe | 1 day |
| 19A-4 | Optimize MTF from 9→5 timeframes | +0.05-0.15 Sharpe, -35% compute | 4 hours |
| 19A-5 | Enhanced labeling with gap risk handling | +0.05-0.10 Sharpe | 1-2 days |

**New Files:**
- `src/data/features/compute/order_flow.py` - Order imbalance, buy/sell pressure
- `src/data/features/compute/liquidity.py` - Spread estimates, liquidity regime
- `src/data/features/compute/mean_reversion.py` - OU half-life, z-scores, variance ratio

### Phase 19B: Performance Optimization (HIGH PRIORITY)

**Estimated Speedup:** 2-4x additional (on top of existing 10-50x)

| Task | Description | Impact | Location |
|------|-------------|--------|----------|
| 19B-1 | Vectorize O(n²) correlation loops | 3-5x for feature selection | `filtering.py:176-195` |
| 19B-2 | Remove DataFrame copies in scaling | 1.5-2x, -1.5GB memory | `scaling/run.py:138-140` |
| 19B-3 | Remove cache copy on hit | 1.2-1.5x for Optuna | `raw_mtf_store.py:140` |
| 19B-4 | Optimize concat+sort patterns | 1.3-1.8x for splits | `splits/run.py:111` |
| 19B-5 | Pre-compute correlations in ensemble | 1.5-2x for ensemble opt | `ensemble_objective.py:80-97` |

### Phase 19C: Architecture Cleanup (MEDIUM PRIORITY)

| Task | Description | Location |
|------|-------------|----------|
| 19C-1 | Move misplaced utilities from core to models | `core/utils/notebook.py`, `colab_setup.py`, `device_utils.py` |
| 19C-2 | Delete duplicate exception file | `models/config/exceptions.py` |
| 19C-3 | Consolidate ConfigValidationError to core | `config/validators.py` → `core/exceptions.py` |
| 19C-4 | Remove deprecated orchestrator.py | `src/orchestrator.py` |

### Phase 19D: Code Quality (LOW PRIORITY)

| Task | Description | Count |
|------|-------------|-------|
| 19D-1 | Ruff auto-fixes (UP038, SIM102, E402) | 77 violations |
| 19D-2 | Fix B904 exception chaining | 19 violations |
| 19D-3 | Add missing type hints to orchestrators | 5 large files |
| 19D-4 | Investigate orphaned pipeline_cli.py | 1 file |

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
- ✅ 42 tests passing
- ✅ MLflow auto-enabled

### Resilience
- ✅ Pipeline checkpointing (resume from failures)
- ✅ Timeout protection
- ✅ Retry with exponential backoff
- ✅ Circuit breaker pattern

---

*For implementation details, see COMPLETION.md*
