# Cleanup Plan: ML Factory

**Status:** Phases 0-12 Complete | Production Ready
**Last Updated:** 2026-01-24 (Phase 12 Complete)

---

## Completed Phases

See COMPLETION.md for full details.

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
| **12** | **Trading Profitability** | **+5,780 lines** | **2026-01-24** |

---

## Phase 12: Trading Profitability & Live Deployment ✅ COMPLETE

**Completion Date:** 2026-01-24
**Status:** 37/39 tasks completed (95%)
**Impact:** +5,780 lines, 57 files modified, 10 new files created

### Resolution Summary

All critical findings from 10-agent analysis have been addressed:

- ✅ **FIXED:** Optuna now optimizes Sharpe ratio instead of F1 score
- ✅ **FIXED:** R-multiple tracking implemented for every trade
- ✅ **FIXED:** Test suite created (42 tests, 981 lines)
- ✅ **FIXED:** FeatureStore integrated into pipeline (30-120s speedup)
- ✅ **FIXED:** MLflow auto-enabled by default
- ✅ **FIXED:** 3 circuit breakers implemented
- ✅ **FIXED:** ProductionMonitor with drift detection active

---

## Phase 12: Trading Profitability & Live Deployment (COMPLETE - P0)

**Goal:** Fix optimization metric, add live trading safeguards, enable deployment infrastructure

### Overview

| Section | Focus | Tasks | Impact |
|---------|-------|-------|--------|
| 12A | Trading Profitability | 8 | CRITICAL - Optimize for profit, not F1 |
| 12B | Live Trading Safeguards | 7 | CRITICAL - Circuit breakers, R-tracking |
| 12C | Deployment Infrastructure | 6 | HIGH - API, monitoring, MLflow |
| 12D | Pipeline Performance | 7 | HIGH - 2-5x faster training |
| 12E | Testing Infrastructure | 5 | CRITICAL - First unit tests |
| 12F | Architecture Cleanup | 6 | MEDIUM - Duplicates, exceptions |

---

## Phase 12A: Trading Profitability (CRITICAL - P0)

**Goal:** Optimize for profit (not F1), fix cost models, ensure realistic backtests

**CRITICAL FINDING:** Optuna optimizes F1 score (classification metric) instead of Sharpe ratio or profit factor (trading metrics). High F1 ≠ profitable trading!

| Task | Description | File:Line | Impact |
|------|-------------|-----------|--------|
| 12A-1 | **P&L-Based Optuna Objective** | `src/optimization/five_dimension_objective.py:437-444` | Use Sharpe/profit instead of F1 |
| 12A-2 | Volatility-Scaled Slippage | `src/inference/backtesting/costs.py:337` | More realistic slippage model |
| 12A-3 | Market Hours Filtering | NEW: `src/inference/backtesting/execution.py` | Only trade liquid hours (CME calendar) |
| 12A-4 | Volume-Relative Position Limits | `src/inference/backtesting/backtest.py` | Limit to 1% of market volume |
| 12A-5 | Adverse Selection Bias | `src/inference/backtesting/execution.py` | Fill price worse by 0.5-1 tick |
| 12A-6 | Integrate Bet Sizing (4G) | `src/inference/backtesting/backtest.py` | Use meta-labeling confidence |
| 12A-7 | Ensemble Diversity Penalty | NEW: `src/optimization/ensemble_objective.py` | Optimize = 0.7×acc + 0.3×diversity |
| 12A-8 | Ensemble Feature Orthogonality | `src/optimization/five_dimension_objective.py` | <30% feature overlap between models |

**Most Critical Fix (12A-1):**
```python
# src/optimization/five_dimension_objective.py:437-444
# BEFORE (WRONG):
def default_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")

# AFTER (CORRECT):
def default_metric(predictions, actuals, prices, labels):
    backtest = run_quick_backtest(predictions, prices, labels, costs=realistic_costs)
    return backtest.metrics.sharpe_ratio  # Or profit_factor, Sortino, etc.
```

**Validation:**
```bash
# Compare F1-optimized vs Sharpe-optimized models
python -m src optimize --metric f1_score --output exp_f1/
python -m src optimize --metric sharpe_ratio --output exp_sharpe/
python -m src compare-experiments exp_f1/ exp_sharpe/ --live-trading-sim
```

---

## Phase 12B: Live Trading Safeguards (CRITICAL - P0)

**Goal:** Add circuit breakers, R-multiple tracking, enforce stop losses

**CRITICAL GAPS:** No drawdown limits, no R-based analysis, stop losses not enforced

| Task | Description | File:Line | Impact |
|------|-------------|-----------|--------|
| 12B-1 | **Drawdown Circuit Breaker** | `src/inference/backtesting/backtest.py:585` | Halt at -10% max drawdown |
| 12B-2 | **R-Multiple Tracking** | `src/inference/backtesting/equity_curve.py:25-90` | Add initial_risk_1r, r_multiple to Trade |
| 12B-3 | **Enforce Stop Losses** | `src/inference/backtesting/backtest.py:392-417` | Calculate ATR-based stops on entry |
| 12B-4 | Daily Loss Limits | NEW: `src/inference/backtesting/risk_controls.py` | Halt at -2% daily loss |
| 12B-5 | R-Based Expectancy | NEW: `src/inference/backtesting/r_analysis.py` | E = (Win%×AvgWinR) - (Loss%×AvgLossR) |
| 12B-6 | Monte Carlo Stress Testing | NEW: `src/inference/backtesting/monte_carlo.py` | 95th percentile worst drawdown |
| 12B-7 | Portfolio Risk Aggregation | NEW: `src/inference/backtesting/portfolio_risk.py` | Max leverage, correlation limits |

**Critical Implementation (12B-1):**
```python
# src/inference/backtesting/backtest.py:585 (main trading loop)
current_drawdown = self.get_drawdowns()[-1] if self._equity_history else 0
if abs(current_drawdown) > self.config.max_drawdown_threshold:
    logger.critical(f"CIRCUIT BREAKER: Drawdown {current_drawdown:.2%} exceeds {self.config.max_drawdown_threshold:.2%}")
    self._halt_trading = True
    self._liquidate_all_positions()
    break
```

**Validation:**
```bash
# Test circuit breaker triggers
python -m src backtest --max-drawdown-threshold 0.10 --test-circuit-breaker
# Verify R-multiple calculation
python -m src analyze-r-distribution --trades backtest_output/trades.csv
```

---

## Phase 12C: Deployment Infrastructure (HIGH - P1)

**Goal:** Production API, monitoring integration, auto-enable MLflow

**FINDING:** FastAPI server exists (server.py) but no monitoring integration. MLflow exists but defaults to "local" JSON, not MLflow server.

| Task | Description | File:Line | Impact |
|------|-------------|-----------|--------|
| 12C-1 | **Enable MLflow by Default** | `src/config/training.py:398` | Change tracking_backend: "local" → "mlflow" |
| 12C-2 | Integrate Drift Monitoring | `src/inference/pipeline.py` | Call FeatureDriftMonitor in predict() |
| 12C-3 | Add ProductionMonitor | NEW: `src/inference/monitor.py` | Unified health checks (drift, freshness, perf) |
| 12C-4 | Slack/PagerDuty Alerts | NEW: `src/validation/monitoring/connectors/slack.py` | Send drift alerts to Slack |
| 12C-5 | Feature Distribution Validation | `src/inference/bundle.py:706` | Compare inference to training distribution |
| 12C-6 | Prometheus Metrics Export | `src/inference/server.py` | Add /metrics endpoint for Grafana |

**Quick Win (12C-1):**
```python
# src/config/training.py:398
tracking_enabled: bool = True
tracking_backend: str = "mlflow"  # CHANGE from "local"
```

**Validation:**
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlruns.db --port 5000
# Train model (should auto-log to MLflow)
python -m src train --symbol MES --models xgboost
# Check MLflow UI
open http://localhost:5000
```

---

## Phase 12D: Pipeline Performance (HIGH - P1)

**Goal:** 5-50x speedup by using existing infrastructure + integrating FeatureStore

**FINDING:** ParallelTrainingService exists but unused. FeatureStore fully implemented but not wired up to pipeline.

| Task | Description | File:Line | Impact |
|------|-------------|-----------|--------|
| 12D-1 | **Enable Parallel Training** | `src/models/training/unified_orchestrator.py:80` | Use ParallelTrainingService (5-10x) |
| 12D-2 | **Integrate FeatureStore** | `src/data/pipeline/stages/features/run.py:123` | Cache features (saves 30-120s/symbol) |
| 12D-3 | Parallelize Optuna Trials | `src/optimization/five_dimension_objective.py:42` | Add n_jobs=-1 (8-12x) |
| 12D-4 | Enable GPU for Boosting | `src/models/boosting/xgboost_model.py` | tree_method='gpu_hist' (10-20x) |
| 12D-5 | Numba Parallel Labeling | `src/data/labeling/triple_barrier.py:232` | @nb.jit(parallel=True) (4-8x) |
| 12D-6 | Cache MTF Upsampled Data | `src/data/pipeline/stages/mtf/__init__.py` | Memoize with mtime invalidation |
| 12D-7 | Stage Timeout Protection | `src/data/pipeline/stages/*/run.py` | Prevent indefinite hangs |

**Biggest Win (12D-2) - Wire FeatureStore:**
```python
# src/data/pipeline/stages/features/run.py:123
from src.data.store import FeatureStore, compute_file_checksum, compute_config_hash

store = FeatureStore(cache_dir=config.feature_cache_dir)
cache_key = f"{symbol}_{tf}_{config_hash}"

if store.has_features(symbol=symbol, feature_set=cache_key):
    df_features = store.get_features(symbol=symbol, feature_set=cache_key)
    logger.info(f"✓ CACHE HIT: {symbol}@{tf}")
else:
    df_features, _ = engineer.engineer_features(df, symbol)
    store.put_features(df_features, symbol=symbol, feature_set=cache_key, ...)
```

**Validation:**
```bash
# Benchmark before/after
time python -m src pipeline --stages features --no-cache
time python -m src pipeline --stages features  # With cache (should be 10x faster on 2nd run)
```

---

## Phase 12E: Testing Infrastructure (CRITICAL - P0)

**Goal:** Create first unit tests for critical paths (0 tests currently!)

**CRITICAL GAP:** 448 Python files, 3,787 functions, **ZERO unit tests**

| Task | Description | New Files | Impact |
|------|-------------|-----------|--------|
| 12E-1 | **Test Cross-Validation** | `tests/test_validation.py` | Verify purge/embargo applied correctly |
| 12E-2 | **Test Adapters** | `tests/test_adapters.py` | Verify output shapes match contracts |
| 12E-3 | **Test Leakage Detection** | `tests/test_leakage.py` | Verify known leakage caught |
| 12E-4 | **Test Backtester** | `tests/test_backtest.py` | Verify costs, circuit breakers |
| 12E-5 | Setup pytest + fixtures | `tests/conftest.py` | Test infrastructure |

**Minimum Test Coverage (12E-1):**
```python
# tests/test_validation.py
def test_purged_kfold_no_overlap():
    """Verify train/test labels don't overlap within horizon."""
    df = create_labeled_data(n=1000, horizon=20)
    cv = PurgedKFold(n_splits=5, purge_bars=60, embargo_bars=100)

    for train_idx, test_idx in cv.split(df):
        train_labels = df.iloc[train_idx].index
        test_labels = df.iloc[test_idx].index

        # Check no label overlap (train label at t shouldn't predict test at t+1..t+20)
        for test_t in test_labels:
            assert all(abs(train_t - test_t) > 20 for train_t in train_labels[-100:])
```

**Validation:**
```bash
# Run tests
pytest tests/ -v
# Check coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Phase 12F: Architecture Cleanup (MEDIUM - P2)

**Goal:** Consolidate duplicates, unify exception hierarchy, remove dead code

**FINDINGS:**
- 3 ValidationError definitions
- 10+ exceptions not inheriting from MLFactoryError
- orchestrator.py deprecated but still exported
- 27 bare exception handlers

| Task | Description | File:Line | Impact |
|------|-------------|-----------|--------|
| 12F-1 | Unify Exception Hierarchy | `src/core/exceptions.py` | All exceptions inherit from MLFactoryError |
| 12F-2 | Remove Duplicate ValidationError | `src/core/validation.py:40` | Delete, use src.core.exceptions.ValidationError |
| 12F-3 | Delete orchestrator.py | `src/orchestrator.py` | Remove 376 lines (deprecated) |
| 12F-4 | Fix Bare Exception Handlers | 27 locations | Replace except Exception with specific types |
| 12F-5 | Consolidate OHLCV Validators | NEW: `src/core/ohlcv_validation.py` | Single validator with strictness levels |
| 12F-6 | Extract Trading Constants | `src/models/metrics.py` | Move 12 magic numbers to constants |

**Critical Fix (12F-1) - Unify Exceptions:**
```python
# Update 10+ exceptions to inherit from MLFactoryError:
# src/validation/leakage_detection.py:31
class LeakageDetectedError(MLFactoryError):  # Was: Exception
    pass

# src/data/store/store.py:65
class FeatureStoreError(MLFactoryError):  # Was: Exception
    pass

# src/validation/lookahead_audit.py:26
class LookaheadBiasError(MLFactoryError):  # Was: Exception
    pass

# ... 7 more files
```

**Validation:**
```bash
# Verify all exceptions caught by MLFactoryError
python -c "from src.core.exceptions import MLFactoryError; from src.validation.leakage_detection import LeakageDetectedError; assert issubclass(LeakageDetectedError, MLFactoryError)"
```

---

## Priority Execution Order

**Week 1 (CRITICAL):**
1. 12A-1: Fix Optuna to use Sharpe instead of F1 (1 day)
2. 12B-1: Add drawdown circuit breaker (1 day)
3. 12B-2: Implement R-multiple tracking (1 day)
4. 12E-1,2: First unit tests (CV + adapters) (2 days)

**Week 2 (HIGH):**
5. 12D-1: Enable parallel training (5 min!)
6. 12D-2: Wire FeatureStore (1 day)
7. 12C-1: Enable MLflow by default (5 min!)
8. 12B-3: Enforce stop losses (1 day)
9. 12E-3,4,5: More tests (backtest, leakage) (2 days)

**Week 3-4 (MEDIUM):**
10. 12C-2,3,4: Production monitoring integration (3 days)
11. 12F-1,2,3: Architecture cleanup (3 days)
12. 12A-3,4,5: Backtesting realism (2 days)
13. 12B-6,7: Monte Carlo + portfolio risk (2 days)

**Total Estimated Effort: 3-4 weeks**

---

## Success Metrics

| Metric | Before | After Target |
|--------|--------|--------------|
| **Optuna Objective** | F1 score | Sharpe ratio |
| **Max Drawdown Protection** | None | -10% circuit breaker |
| **R-Multiple Tracking** | None | Full distribution analysis |
| **Unit Tests** | 0 | 50+ critical path tests |
| **Training Time** | 16h | 1.5-2h (10x faster) |
| **Feature Cache Hit Rate** | 0% (no cache) | 80%+ on repeat runs |
| **MLflow Adoption** | Manual | Auto-enabled |
| **Stop Loss Enforcement** | 0% | 100% of trades |
| **Backtest Realism** | Optimistic | Matches live ±20% |

---

## Remaining Work from Original Phase 11-18

These phases are now superseded by Phase 12 above. Original deferred items:

---

## Phase 13: Performance Optimization (HIGH - P1)

**Goal:** 10-50x speedup in training/inference for live trading latency requirements

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 13A | Enable Parallel Model Training | Use existing ParallelTrainingService by default | 5-10x speedup | 5 min |
| 13B | Parallelize Optuna Trials | Add `n_jobs=-1` to study.optimize() | 8-12x speedup | 10 min |
| 13C | Enable GPU for Boosting Models | Set `tree_method='gpu_hist'` for XGB/LGBM/Cat | 10-20x speedup | 20 min |
| 13D | Parallelize Feature Engineering | Joblib across symbols in feature stage | 5x speedup | 30 min |
| 13E | Numba Parallel Labeling | Add `@nb.jit(parallel=True)` + `nb.prange` | 4-8x speedup | 1 hour |
| 13F | Cache MTF Upsampled Data | Memoize upsampling with mtime invalidation | Save 5-10 min/run | 2 hours |
| 13G | Batch Inference for Ensembles | Batch predictions across base models | 10x faster inference | 3 hours |

**Files:**
- MODIFY: `src/models/training/unified_orchestrator.py:80` (ParallelTrainingService default)
- MODIFY: `src/optimization/five_dimension_objective.py:42` (n_jobs=-1)
- MODIFY: `src/models/boosting/xgboost_model.py` (gpu_hist default)
- MODIFY: `src/data/pipeline/stages/features/run.py:123` (joblib.Parallel)
- MODIFY: `src/data/labeling/triple_barrier.py:232` (parallel=True, prange)
- MODIFY: `src/data/pipeline/stages/mtf/__init__.py` (add caching)
- MODIFY: `src/inference/batch.py` (batch ensemble)

**Validation:**
```bash
# Training time before/after (target: 16h → 1.5h)
time python -m src train --symbol MES --models xgboost,lightgbm,catboost
# Inference latency (target: <50ms p99)
python -m src benchmark-inference --bundle experiments/run_001/
```

---

## Phase 14: Data Quality Hardening (HIGH - P1)

**Goal:** Eliminate silent data quality failures and leakage risks

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 14A | Dynamic Purge Bars Calculation | purge_bars = max(horizons) * 3 (auto-computed) | Prevent label leakage | 1 hour |
| 14B | Make MTF shift(1) Mandatory | Remove apply_shift parameter, always shift | Prevent lookahead | 30 min |
| 14C | Automatic Lookahead Audit | Integrate LookaheadAuditor into pipeline validation | Catch new feature bugs | 2 hours |
| 14D | Per-Feature NaN Monitoring | Fail-fast if feature >10% NaN before warmup | Prevent silent data loss | 2 hours |
| 14E | Label Alignment Validation | Verify labels match sequence timestamps | Prevent off-by-one | 1 hour |
| 14F | Inter-Stage Schema Validation | Validate schema between pipeline stages | Catch corruption early | 2 hours |
| 14G | Feature Manifest with Params | Store computation params per feature | Reproducibility | 2 hours |

**Files:**
- MODIFY: `src/validation/cv/purged_kfold.py:39` (dynamic purge_bars)
- MODIFY: `src/data/features/compute/mtf.py:115` (remove apply_shift param)
- MODIFY: `src/data/pipeline/stages/validation/run.py` (add lookahead audit)
- MODIFY: `src/data/pipeline/stages/features/*.py` (NaN monitoring)
- MODIFY: `src/data/adapters/base.py:143` (label alignment check)
- MODIFY: `src/data/pipeline/stages/*/run.py` (schema validation)
- MODIFY: `src/data/pipeline/feature_manifest.py:42` (add params field)

**Validation:**
```bash
# Should fail if purge_bars < max(horizons)*3
python -m src validate-config --config invalid_purge.yaml
# Lookahead audit (should detect corruption)
python -m src audit-lookahead --features data/canonical/engineered/
```

---

## Phase 15: Backtesting Realism (HIGH - P1)

**Goal:** Realistic backtests that match live trading performance

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 15A | Market Hours Filtering | Only trade during CME liquid hours (NY session) | Prevent unrealistic fills | 2 hours |
| 15B | Volume-Relative Position Limits | Max position = 1% of rolling volume | Prevent market impact | 1 hour |
| 15C | Adverse Selection Bias | Fill price worse by 0.5-1 tick when predicting moves | Realistic slippage | 2 hours |
| 15D | Volatility-Scaled Slippage | Replace FixedSlippage with VolatilityScaledSlippage | Better cost modeling | 1 hour |
| 15E | Integrate Bet Sizing (4G) | Use meta-labeling confidence for position sizing | Better risk mgmt | 3 hours |

**Files:**
- NEW: `src/inference/backtesting/execution.py` (market hours, adverse selection)
- MODIFY: `src/inference/backtesting/backtest.py:176` (integrate execution model)
- MODIFY: `src/inference/backtesting/backtest.py:237` (VolatilityScaledSlippage default)
- MODIFY: `src/inference/backtesting/costs.py:337` (volume-relative limits)
- MODIFY: `src/inference/backtesting/backtest.py` (bet sizing integration)

**Validation:**
```bash
# Compare backtest vs paper trading (should match within 20%)
python -m src backtest --realistic-execution
python -m src compare-backtest-vs-live --experiment run_001
```

---

## Phase 16: Ensemble Optimization (MEDIUM - P2)

**Goal:** Maximize ensemble diversity and profitability

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 16A | Diversity-Aware Selection | Optuna objective = 0.7*acc + 0.3*diversity | Better ensemble gains | 4 hours |
| 16B | Ensemble-Aware Features | Constrain feature overlap <30% between models | Orthogonal predictions | 3 hours |
| 16C | Auto Meta-Learner Selection | Optuna selects meta-learner type | Optimal ensemble | 1 hour |
| 16D | Second-Level Stacking | Meta-meta-learner for 12-model system | Max performance | 6 hours |
| 16E | Diversity Analysis Integration (4C) | Wire DiversityAnalyzer to orchestrator | Visibility | 2 hours |

**Files:**
- NEW: `src/optimization/ensemble_objective.py` (diversity penalty)
- MODIFY: `src/optimization/five_dimension_objective.py:450` (feature constraints)
- MODIFY: `src/models/ensemble/orchestrator.py` (auto meta-learner)
- NEW: `src/models/ensemble/second_level.py` (meta-meta-learner)
- MODIFY: `src/models/training/unified_orchestrator.py` (wire DiversityAnalyzer)

**Validation:**
```bash
# Diversity should be >0.3 (Q-statistic)
python -m src evaluate-diversity --experiment run_001
# Ensemble Sharpe should beat best base model by 20%+
python -m src compare-ensemble-vs-base --experiment run_001
```

---

## Phase 17: Architecture Resilience (MEDIUM - P2)

**Goal:** Production-grade error handling and recovery

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 17A | State Checkpointing in MLFactory | Resume from failures, save intermediate results | No wasted compute | 2-3 days |
| 17B | Timeout Protection | Max time per Optuna trial/model/stage | Prevent hung processes | 2 days |
| 17C | Circuit Breakers | Isolate model failures, continue with others | Fault tolerance | 2 days |
| 17D | Retry Logic with Backoff | Auto-retry transient failures (GPU OOM, network) | Resilience | 2-3 days |
| 17E | Unify Exception Hierarchy | Single MLFactoryError base class | Consistent errors | 3-4 days |

**Files:**
- MODIFY: `src/factory.py:184` (add checkpointing)
- MODIFY: `src/models/training/unified_orchestrator.py:257` (circuit breaker)
- MODIFY: `src/optimization/five_dimension_objective.py` (timeout decorator)
- NEW: `src/core/resilience.py` (retry decorators)
- MODIFY: `src/core/exceptions.py` (unify hierarchy)

**Validation:**
```bash
# Kill process mid-training, should resume
python -m src train --checkpoint-every 10-trials
kill -9 $PID
python -m src resume --checkpoint last
```

---

## Phase 18: Code Cleanup (LOW - P3)

**Goal:** Minimal duplication, clean architecture

| Task | Description | Impact | Effort |
|------|-------------|--------|--------|
| 18A | Consolidate DataContract | Remove duplicate in core/data_contract.py | Clean imports | 1-2 hours |
| 18B | Resolve AdapterResult (Phase 0G) | Single definition in adapters/base.py | Clean architecture | 2-3 hours |
| 18C | Refactor Large Files | Split unified_orchestrator.py if needed | Maintainability | 1 day (optional) |

**Files:**
- DELETE: `src/core/data_contract.py` (keep contracts/data_contract.py)
- MODIFY: `src/core/interfaces.py:45` (remove AdapterResult)
- OPTIONAL: Split `src/models/training/unified_orchestrator.py` (1679 lines)

---

## Phase 11: Deferred Items (Low Priority)

| Task | Description | Notes |
|------|-------------|-------|
| 5C | Unified deployment bundle (tar.gz) | Needs bundle spec |
| 4D | Deflated Sharpe Ratio post-Optuna | Add DSR gate |
| 4E | Bootstrap CIs in financial reports | Wire BootstrapCI |
| 4F | Auto calibration in orchestrator | Wire CalibrationManager |
| - | MTF ablation flag | Add `mtf.enabled` config |

---

## Priority Summary

**Week 1-2 (CRITICAL):** Phase 12 (Live Trading Infrastructure)
**Week 3 (HIGH):** Phase 13A-C (Quick performance wins: parallel training, GPU)
**Week 4-5 (HIGH):** Phase 14 (Data quality hardening)
**Week 6-7 (HIGH):** Phase 15 (Backtesting realism)
**Month 2:** Phases 16-17 (Ensemble optimization, resilience)
**Backlog:** Phase 18 (Code cleanup)

**Estimated Total:** 8-12 weeks for full production hardening

---

*For completed phase details, see COMPLETION.md*
