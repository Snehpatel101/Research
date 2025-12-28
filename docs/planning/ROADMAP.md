# ROADMAP: OHLCV ML Trading Pipeline

**Version:** 1.0.0
**Date:** 2025-12-28
**Status:** Ready for Development

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1: Critical Foundation (Week 1-2)](#phase-1-critical-foundation-week-1-2)
3. [Phase 2: Production Safety (Week 3-4)](#phase-2-production-safety-week-3-4)
4. [Phase 3: Performance Upgrades (Week 5-8)](#phase-3-performance-upgrades-week-5-8)
5. [Expected Outcomes](#expected-outcomes)
6. [Timeline Summary](#timeline-summary)

---

## Executive Summary

This implementation plan addresses **12 critical issues** identified in the OHLCV ML trading pipeline. The current pipeline has a solid foundation with 12 implemented models across 4 families, 150+ features, and proper purged k-fold CV. However, several gaps prevent production deployment:

**P0 Issues (Must Fix):**
- No probability calibration (30-50% position sizing errors)
- CV loads pre-scaled data globally (leaking future statistics)
- Phase 5 (Inference) not implemented

**P1 Issues (Production Safety):**
- No online drift detection
- Label-aware purging not wired
- No walk-forward evaluation
- Sequence model CV uses wrong data structure

**Expected Impact After Implementation:**
| Metric | Before | After |
|--------|--------|-------|
| Calibration (Brier) | 0.25-0.30 | < 0.15 |
| CV Metric Inflation | +15-25% | < 5% |
| Drift Detection | None | Real-time |
| Deployment Ready | No | Yes |

**Timeline:** 8 weeks for full implementation

---

## Phase 1: Critical Foundation (Week 1-2)

**Goal:** Fix critical issues that invalidate current results or block deployment.

### 1.1 Probability Calibration

**Problem:** Tree models (XGBoost, LightGBM, CatBoost) output miscalibrated probabilities. This causes:
- Position sizing based on wrong confidence
- Ensemble stacking learns from distorted inputs
- Threshold-based trading decisions are biased

**Solution:** Implement isotonic/Platt calibration with Brier/ECE metrics.

**Files to Create:**
```
src/models/calibration/
    __init__.py
    calibrator.py         # CalibratedPredictor class
    metrics.py            # Brier score, ECE, reliability curves
```

**Files to Modify:**
- `src/models/trainer.py` - Add calibration step after training
- `src/cross_validation/oof_generator.py` - Calibrate OOF predictions
- `src/models/base.py` - Add `calibrator` attribute to PredictionOutput

**Detailed Spec:** [specs/probability_calibration.md](specs/probability_calibration.md)

**Acceptance Criteria:**
- Brier score < 0.15 (down from 0.25-0.30)
- ECE < 0.05
- Reliability curves show diagonal alignment

---

### 1.2 Fold-Aware Scaling in CV

**Problem:** `oof_generator.py` loads pre-scaled data from `TimeSeriesDataContainer.get_sklearn_arrays("train")`. The scaler was fit on the entire training set, so each fold's validation data has been transformed using statistics from "future" samples in other folds.

**Current Code (Problem):**
```python
# src/cross_validation/oof_generator.py line 197-198
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
# X is already scaled globally - WRONG
```

**Solution:** Scale within each fold using only that fold's training data.

**Files to Create:**
```
src/cross_validation/fold_scaling.py   # FoldAwareScaler class
```

**Files to Modify:**
- `src/cross_validation/oof_generator.py` - Add per-fold scaling
- `src/cross_validation/cv_runner.py` - Pass unscaled data option

**Detailed Spec:** [specs/cv_improvements.md](specs/cv_improvements.md#fold-aware-scaling)

**Acceptance Criteria:**
- CV metrics drop 5-15% (expected - removes optimism)
- No scaling parameters leak between folds
- Fold validation uses only training fold statistics

---

### 1.3 Lookahead Audit for MTF Features

**Problem:** Multi-timeframe features (e.g., 15min, 1H aggregates) may inadvertently peek into future data if resampling isn't done with `closed='left', label='left'`.

**Solution:** Automated lookahead verification suite.

**Files to Create:**
```
src/validation/lookahead_audit.py      # LookaheadAuditor class
tests/validation/test_lookahead.py     # Automated tests
```

**Files to Modify:**
- `src/phase1/stages/mtf/generator.py` - Document/verify closed/label params
- Pipeline reporting - Add lookahead verification results

**Detailed Spec:** [specs/cv_validation_methods.md](specs/cv_validation_methods.md#lookahead-audit)

**Acceptance Criteria:**
- 0 lookahead violations detected
- All MTF resampling uses `closed='left', label='left'`
- Corruption test passes (future corruption doesn't affect features)

---

### 1.4 Fast Ensemble Baseline

**Problem:** No quick-deploy ensemble option for production testing.

**Solution:** Implement low-latency boosting ensemble (XGBoost + LightGBM + CatBoost) with voting.

**Files to Modify:**
- `src/models/ensemble/voting.py` - Optimize for inference speed
- Add benchmark script

**Acceptance Criteria:**
- Inference latency < 5ms for 1000 samples
- Memory < 500MB total
- No GPU required

---

## Phase 2: Production Safety (Week 3-4)

**Goal:** Add production monitoring and improve CV reliability.

### 2.1 Online Drift Detection

**Problem:** Models decay silently in production without alerts. Current pipeline only has offline PSI checks in `scaled_validation`.

**Solution:** Implement real-time drift monitoring using ADWIN for performance drift and PSI for feature drift.

**Files to Create:**
```
src/monitoring/
    __init__.py
    drift_detector.py     # DriftDetector with ADWIN, PSI
    performance_monitor.py # PerformanceMonitor with alerting
    alert_handlers.py     # Slack, email, logging handlers
```

**Dependencies:** `river` (for ADWIN)

**Detailed Spec:** [specs/drift_detection.md](specs/drift_detection.md)

**Acceptance Criteria:**
- ADWIN detects performance drift within 100 samples of shift
- PSI alerts when feature drift > 0.2
- Configurable alert thresholds and handlers

---

### 2.2 Label-Aware Purging

**Problem:** `PurgedKFold` supports `label_end_times` but it's never passed. Triple-barrier labels have variable resolution times (stored in `bars_to_hit`), so fixed purge bars is a coarse approximation.

**Current Code (Unused):**
```python
# src/cross_validation/purged_kfold.py line 207-212
if label_end_times is not None and has_datetime_index:
    test_start_time = X.index[test_start]
    for i in range(purge_start):
        if label_end_times.iloc[i] >= test_start_time:
            train_mask[i] = False
```

**Solution:** Compute and persist `label_end_time` during labeling, wire it through to CV.

**Files to Modify:**
- `src/phase1/stages/labeling/triple_barrier.py` - Compute `label_end_time`
- `src/phase1/stages/datasets/container.py` - Expose `label_end_times`
- `src/cross_validation/cv_runner.py` - Pass to PurgedKFold
- `src/cross_validation/oof_generator.py` - Pass to PurgedKFold

**Detailed Spec:** [specs/cv_improvements.md](specs/cv_improvements.md#label-aware-purging)

**Acceptance Criteria:**
- `label_end_time_h{horizon}` column persisted in parquet
- CV uses label-aware purging for overlapping events
- Validation set contains no samples whose labels depend on future test data

---

### 2.3 Walk-Forward Evaluation

**Problem:** Purged k-fold averages hide temporal degradation. A model may perform well in 2020 folds but fail in 2024.

**Solution:** Add rolling-origin walk-forward evaluator alongside k-fold.

**Files to Create:**
```
src/cross_validation/walk_forward.py   # WalkForwardEvaluator class
scripts/run_walk_forward.py            # CLI entrypoint
```

**Detailed Spec:** [specs/cv_validation_methods.md](specs/cv_validation_methods.md#walk-forward-evaluation)

**Acceptance Criteria:**
- Reports per-window metrics (Sharpe, F1, accuracy)
- Shows temporal degradation curve
- Supports expanding and rolling windows

---

### 2.4 CPCV (Combinatorial Purged CV)

**Problem:** Standard purged k-fold tests a single path through time. With hyperparameter tuning across many trials, winner selection may be overfitting to that specific path.

**Solution:** Implement CPCV that tests C(n,k) combinations to estimate probability of backtest overfitting (PBO).

**Files to Create:**
```
src/cross_validation/cpcv.py           # CombinatorialPurgedCV class
src/cross_validation/pbo.py            # ProbabilityOfBacktestOverfitting
```

**Reference:** Bailey et al. (2014) "The Probability of Backtest Overfitting"

**Detailed Spec:** [specs/cv_cpcv.md](specs/cv_cpcv.md)

**Acceptance Criteria:**
- PBO estimate computed after hyperparameter tuning
- Warning when PBO > 0.5
- Block deployment when PBO > 0.8

---

### 2.5 Sequence Model CV Fix

**Problem:** `cv_runner.py` uses `container.get_sklearn_arrays("train")` for all models, but LSTM/GRU/TCN need 3D sequences from `get_pytorch_sequences()`.

**Current Code (Wrong):**
```python
# src/cross_validation/cv_runner.py line 327
X, y, weights = container.get_sklearn_arrays("train", return_df=True)
# This returns 2D arrays, but neural models need 3D sequences
```

**Solution:** Add model-type-aware data loading in CV.

**Files to Modify:**
- `src/cross_validation/cv_runner.py` - Check `model.requires_sequences`
- `src/cross_validation/oof_generator.py` - Handle sequence data

**Detailed Spec:** [specs/cv_improvements.md](specs/cv_improvements.md#sequence-model-cv-fix)

**Acceptance Criteria:**
- LSTM/GRU/TCN CV uses correct sequence structure
- Sequences don't cross symbol boundaries
- Temporal ordering preserved

---

## Phase 3: Performance Upgrades (Week 5-8)

**Goal:** Complete the pipeline with deployment capability and advanced features.

### 3.1 Inference Pipeline (Phase 5)

**Problem:** No production deployment capability. Phase 5 is documented in `docs/phases/PHASE_5.md` but `src/inference/` doesn't exist.

**Solution:** Implement complete inference pipeline with serialization, monitoring, and deployment profiles.

**Files to Create:**
```
src/inference/
    __init__.py
    pipeline.py           # InferencePipeline class
    serializer.py         # PipelineSerializer (scaler + features + models + calibrator)
    server.py             # FastAPI inference server (optional)
    batch.py              # BatchInference for offline scoring
scripts/
    serve_model.py        # Start inference server
    batch_inference.py    # Run batch predictions
```

**Detailed Spec:** [specs/inference_pipeline.md](specs/inference_pipeline.md)

**Acceptance Criteria:**
- Single artifact bundle with everything needed for inference
- Inference latency < 100ms for ensemble
- Lookahead verification passes
- Drift monitoring integrated

---

### 3.2 Regime-Adaptive Models

**Problem:** Single model for all market regimes misses 20-30% Sharpe improvement from regime-specific strategies.

**Solution:** HMM-based regime detection with specialist models per regime.

**Files to Create:**
```
src/regime/
    __init__.py
    detector.py           # RegimeDetector with HMM (3 states: bull/bear/neutral)
    specialist.py         # RegimeSpecialistModel wrapper
    ensemble.py           # RegimeAdaptiveEnsemble
```

**Dependencies:** `hmmlearn`

**Acceptance Criteria:**
- HMM converges to 3 interpretable regimes
- Specialist models train on regime-specific data
- Ensemble routes predictions through active regime

---

### 3.3 Conformal Prediction

**Problem:** No uncertainty quantification for position sizing. Model may be confident but wrong.

**Solution:** Implement conformal prediction for prediction intervals.

**Files to Create:**
```
src/models/conformal.py  # ConformalPredictor class
```

**Reference:** Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"

**Acceptance Criteria:**
- 90% prediction intervals contain true value 90% of time
- Coverage calibrated across regimes
- Position sizing based on interval width

---

### 3.4 Alternative Bar Types

**Problem:** Time bars have poor statistical properties (varying volume, activity). Dollar bars sample by fixed notional value, yielding better i.i.d. properties.

**Solution:** Implement dollar/volume bar generation.

**Files to Create:**
```
src/phase1/stages/ingest/alternative_bars.py   # DollarBarGenerator, VolumeBarGenerator
```

**Reference:** Lopez de Prado (2018) "Advances in Financial Machine Learning" Chapter 2

**Acceptance Criteria:**
- Dollar bars have lower autocorrelation than time bars
- Returns closer to normal distribution
- Pipeline supports bar type selection via config

---

## Expected Outcomes

### Metrics Before/After

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| **CV F1 (XGBoost, H20)** | 0.52 | 0.45-0.48 | Expected drop from removing leakage |
| **Brier Score** | 0.25-0.30 | < 0.15 | Calibration improvement |
| **ECE** | 0.10-0.15 | < 0.05 | Calibration improvement |
| **Val-Test Gap** | 15-25% | < 10% | Better generalization |
| **PBO** | Unknown | < 0.5 | Quantified overfitting risk |
| **Drift Detection** | None | < 100 samples | Real-time monitoring |

### Production Readiness Checklist

After implementation:

- [ ] Probability calibration active (Brier < 0.15, ECE < 0.05)
- [ ] CV uses fold-aware scaling
- [ ] Label-aware purging enabled
- [ ] Lookahead audit passes (0 violations)
- [ ] Walk-forward evaluation shows stable performance
- [ ] PBO < 0.5 for selected model
- [ ] Inference pipeline serialized
- [ ] Drift monitoring active
- [ ] Test set evaluation complete

---

## Timeline Summary

### Week 1-2: Critical Foundation (Phase 1)

**Week 1:**
- Day 1-2: Implement probability calibration
- Day 3-4: Implement fold-aware scaling
- Day 5: Integrate calibration into trainer and OOF

**Week 2:**
- Day 1-2: Implement lookahead audit
- Day 3-4: Run MTF feature verification
- Day 5: Fast ensemble baseline benchmark

### Week 3-4: Production Safety (Phase 2)

**Week 3:**
- Day 1-2: Implement drift detection (ADWIN, PSI)
- Day 3-4: Implement label-aware purging
- Day 5: Wire label_end_times through pipeline

**Week 4:**
- Day 1-2: Implement walk-forward evaluation
- Day 3-4: Implement CPCV and PBO
- Day 5: Fix sequence model CV

### Week 5-8: Performance Upgrades (Phase 3)

**Week 5-6:**
- Implement inference pipeline
- Serialization and deployment artifacts
- Lookahead verification in production

**Week 7:**
- Regime-adaptive models (optional)
- Conformal prediction (optional)

**Week 8:**
- Alternative bar types (optional)
- Final integration and testing
- Documentation updates

---

## Cross-References

For detailed implementation specifications, see:
- [GAPS_ANALYSIS.md](GAPS_ANALYSIS.md) - Detailed gap analysis and current state
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration steps, breaking changes, risk mitigation
- [specs/probability_calibration.md](specs/probability_calibration.md) - Calibration implementation
- [specs/cv_improvements.md](specs/cv_improvements.md) - CV core improvements
- [specs/cv_validation_methods.md](specs/cv_validation_methods.md) - Advanced CV methods
- [specs/cv_cpcv.md](specs/cv_cpcv.md) - CPCV and PBO implementation
- [specs/drift_detection.md](specs/drift_detection.md) - Drift monitoring
- [specs/inference_pipeline.md](specs/inference_pipeline.md) - Production deployment

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial roadmap extracted from IMPLEMENTATION_PLAN.md |

**Next Review:** After Phase 1 completion (Week 2)

---

*This roadmap provides a high-level overview of the implementation plan. See individual spec documents for detailed code examples and technical specifications.*
