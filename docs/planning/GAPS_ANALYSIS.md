# GAPS ANALYSIS: OHLCV ML Trading Pipeline

**Version:** 1.0.0
**Date:** 2025-12-28
**Status:** Analysis Complete

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Critical Gaps (P0)](#critical-gaps-p0)
3. [Production Safety Gaps (P1)](#production-safety-gaps-p1)
4. [Performance Enhancement Gaps (P2-P3)](#performance-enhancement-gaps-p2-p3)
5. [Priority Matrix](#priority-matrix)
6. [Impact Assessment](#impact-assessment)

---

## Current State Assessment

### Pipeline Strengths

1. **Solid Model Factory Architecture**
   - 12 models registered: XGBoost, LightGBM, CatBoost, LSTM, GRU, TCN, Transformer, RF, Logistic, SVM, Voting, Stacking, Blending
   - Plugin-based registry with `@register()` decorator
   - Unified `BaseModel` interface (`src/models/base.py`)
   - `TrainingMetrics` and `PredictionOutput` standardized containers

2. **Comprehensive Data Pipeline**
   - 15 stages: ingest -> clean -> sessions -> regime -> mtf -> features -> labeling -> ga_optimize -> final_labels -> splits -> scaling -> scaled_validation -> datasets -> validation -> reporting
   - 150+ features (momentum, volatility, trend, wavelets, microstructure)
   - Triple-barrier labeling with Optuna optimization
   - `TimeSeriesDataContainer` for unified data access

3. **Cross-Validation Foundation**
   - `PurgedKFold` with configurable purge (60 bars) and embargo (1440 bars)
   - `ModelAwareCV` adapts splits per model family
   - `OOFGenerator` creates stacking datasets
   - Walk-forward feature selection exists

4. **Code Quality**
   - Type hints throughout
   - Dataclasses for configuration
   - Comprehensive docstrings
   - Logging infrastructure

---

## Critical Gaps (P0)

These gaps **must** be fixed before any production deployment. They either invalidate current results or completely block deployment.

### Gap 1: No Probability Calibration

**Location:** `src/models/trainer.py`, `src/cross_validation/oof_generator.py`

**Problem:**
Tree-based models (XGBoost, LightGBM, CatBoost) are notoriously miscalibrated. They output probability-like scores, but these don't reflect true likelihoods:

```python
# Current output (uncalibrated)
model.predict_proba([[...]])
# Output: [0.15, 0.35, 0.50] for [SHORT, NEUTRAL, LONG]
# But true accuracy at 0.50 confidence may only be 42%!
```

**Impact:**
- **Position Sizing Errors:** If the model says 50% confidence but true accuracy is 42%, position sizes will be too large, leading to increased risk
- **Ensemble Stacking Distortion:** Stacking meta-learners learn from miscalibrated inputs, compounding the error
- **Threshold Selection Invalid:** Trading strategies that filter on confidence (e.g., "only trade if confidence > 0.6") use wrong thresholds

**Evidence:**
```python
# Typical uncalibrated Brier score: 0.25-0.30
# Expected calibrated Brier score: < 0.15
# ECE (Expected Calibration Error): 0.10-0.15 (should be < 0.05)
```

**Required Fix:**
Implement isotonic regression or Platt scaling to map raw model outputs to calibrated probabilities.

**Priority:** P0 - Blocks accurate position sizing
**Effort:** 2-3 days
**Reference:** [specs/probability_calibration.md](specs/probability_calibration.md)

---

### Gap 2: CV Leakage from Global Scaling

**Location:** `src/cross_validation/oof_generator.py` line 197-198

**Problem:**
The OOF generator loads pre-scaled data from `TimeSeriesDataContainer.get_sklearn_arrays("train")`. The scaler was fit on the **entire** training set during Phase 1:

```python
# src/phase1/stages/scaling/scaler.py (Phase 1)
scaler.fit(X_train)  # X_train = all 70% training data

# Later in CV (Phase 3)
# src/cross_validation/oof_generator.py line 197
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
# But X is already scaled using statistics from ALL training data,
# including samples in val_idx!
```

**Impact:**
- **Optimistic CV Metrics:** Validation folds have been scaled using statistics that include "future" data from other folds
- **Inflated Performance:** F1 scores may be 5-15% higher than true out-of-sample performance
- **Incorrect Model Selection:** Hyperparameter tuning may select overly complex models that won't generalize

**Evidence:**
```python
# Typical observation:
# CV F1: 0.52
# Test F1: 0.43 (17% drop)
# Expected after fix:
# CV F1: 0.46-0.48 (5-10% lower, but honest)
# Test F1: 0.43-0.45 (smaller gap)
```

**Required Fix:**
Implement fold-aware scaling where each CV fold's scaler is fit only on that fold's training data:

```python
for train_idx, val_idx in cv.split(X):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
    X_val_scaled = scaler.transform(X.iloc[val_idx])  # No leakage
```

**Priority:** P0 - Invalidates current CV results
**Effort:** 2 days
**Reference:** [specs/cv_improvements.md](specs/cv_improvements.md#fold-aware-scaling)

---

### Gap 3: Phase 5 (Inference) Not Implemented

**Location:** Missing `src/inference/`

**Problem:**
The pipeline can train models and evaluate them, but there's no way to:
- Serialize a complete pipeline for deployment
- Load a saved pipeline and make predictions on new data
- Ensure production inference matches training preprocessing
- Monitor inference performance

**Impact:**
- **No Deployment Path:** Cannot move models to production
- **Preprocessing Drift Risk:** Manual reproduction of feature engineering in production may introduce bugs
- **No Production Monitoring:** Cannot detect model decay in real-time

**Required Components:**
1. `InferencePipeline` class that bundles:
   - Feature scaler
   - Feature column list
   - Trained models
   - Calibrators
   - Ensemble meta-learner
2. Serialization/deserialization
3. Prediction API
4. Drift monitoring integration

**Priority:** P0 - Blocks production deployment
**Effort:** 5-7 days
**Reference:** [specs/inference_pipeline.md](specs/inference_pipeline.md)

---

## Production Safety Gaps (P1)

These gaps don't invalidate current results but are **required** for safe production operation.

### Gap 4: No Online Drift Detection

**Location:** Missing `src/monitoring/`

**Problem:**
The pipeline has offline PSI checks during Phase 1 validation, but no runtime monitoring. In production:
- Market regimes change (e.g., low vol -> high vol)
- Feature distributions shift (e.g., new trading patterns)
- Model performance degrades silently

**Impact:**
- **Silent Failure:** Model may stop working without any alert
- **Capital Loss:** Continued trading with a broken model
- **Delayed Response:** May not notice degradation for days/weeks

**Current State:**
```python
# Phase 1 has static PSI checks
# src/phase1/stages/scaled_validation/validator.py
psi = compute_psi(train_features, val_features)
# But this is one-time, offline, and doesn't help in production
```

**Required Fix:**
Implement real-time drift detection using:
1. **ADWIN** (ADaptive WINdowing) for performance drift
2. **PSI** (Population Stability Index) for feature drift
3. **Alert System** for Slack/email/logging

**Priority:** P1 - Required for production safety
**Effort:** 3-4 days
**Reference:** [specs/drift_detection.md](specs/drift_detection.md)

---

### Gap 5: Label-Aware Purging Dead Code

**Location:** `src/phase1/stages/labeling/triple_barrier.py`, `src/cross_validation/purged_kfold.py`

**Problem:**
`PurgedKFold` has code to support label-aware purging, but it's never used:

```python
# src/cross_validation/purged_kfold.py line 207-212
if label_end_times is not None and has_datetime_index:
    test_start_time = X.index[test_start]
    for i in range(purge_start):
        if label_end_times.iloc[i] >= test_start_time:
            train_mask[i] = False
```

Triple-barrier labels have variable resolution times (some hit barriers at 5 bars, some timeout at 20 bars), but the pipeline uses a fixed purge of 60 bars for all samples.

**Impact:**
- **Suboptimal Purging:** Over-purging (removing more training data than necessary) or under-purging (potential leakage)
- **Reduced Training Data:** Fixed purge may be too conservative for labels that resolved quickly

**Current State:**
```python
# src/phase1/stages/labeling/triple_barrier.py computes bars_to_hit
# but never converts this to label_end_time
result.metadata['bars_to_hit'] = bars_to_hit  # Stored but unused
```

**Required Fix:**
1. Compute `label_end_time` during labeling
2. Persist in parquet alongside labels
3. Wire through to CV splits

**Priority:** P1 - Improves CV accuracy
**Effort:** 1-2 days
**Reference:** [specs/cv_improvements.md](specs/cv_improvements.md#label-aware-purging)

---

### Gap 6: No Walk-Forward Evaluation

**Location:** Missing `src/cross_validation/walk_forward.py`

**Problem:**
Purged k-fold averages performance across randomly shuffled time periods. This hides temporal degradation:

```
K-Fold CV:
  Fold 1: [2020, 2022, 2024] train | [2021] test -> F1: 0.50
  Fold 2: [2020, 2021, 2024] train | [2022] test -> F1: 0.48
  Fold 3: [2020, 2021, 2022] train | [2023] test -> F1: 0.45
  Fold 4: [2021, 2022, 2023] train | [2024] test -> F1: 0.38
  Mean: 0.45 (hides the 2024 degradation!)

Walk-Forward:
  Window 1: [2020-2021] train | [2022-Q1] test -> F1: 0.50
  Window 2: [2020-2022] train | [2022-Q2] test -> F1: 0.48
  Window 3: [2020-2022] train | [2023-Q1] test -> F1: 0.45
  Window 4: [2020-2023] train | [2023-Q2] test -> F1: 0.42
  Window 5: [2020-2023] train | [2024-Q1] test -> F1: 0.38
  Clear degradation trend!
```

**Impact:**
- **Hidden Model Decay:** May not notice that recent performance is much worse
- **Overconfident Deployment:** Average metrics look good but current performance is poor

**Required Fix:**
Implement rolling-origin walk-forward evaluation that:
1. Expands or rolls training window forward in time
2. Always tests on future data
3. Reports per-window metrics to detect trends

**Priority:** P1 - Detects temporal issues
**Effort:** 2-3 days
**Reference:** [specs/cv_validation_methods.md](specs/cv_validation_methods.md#walk-forward-evaluation)

---

### Gap 7: Sequence Model CV Uses Wrong Data Structure

**Location:** `src/cross_validation/cv_runner.py` line 327

**Problem:**
All models use the same data loading path:

```python
# src/cross_validation/cv_runner.py line 327
X, y, weights = container.get_sklearn_arrays("train", return_df=True)
# Returns 2D array: (n_samples, n_features)

# But LSTM/GRU/TCN need 3D sequences: (n_samples, seq_len, n_features)
# Currently broken for sequence models in CV!
```

**Impact:**
- **Invalid CV for Neural Models:** LSTM/GRU/TCN CV results are unreliable
- **Can't Compare Models:** Can't fairly compare boosting vs neural since CV is broken for neural

**Current Workaround:**
Neural models work in standalone training (Phase 2) because `trainer.py` calls `get_pytorch_sequences()`, but this isn't wired into CV.

**Required Fix:**
Model-aware data loading in CV:

```python
if model.requires_sequences:
    X, y = create_sequences(X_df, y_series, seq_len=60)
else:
    X, y = X_df.values, y_series.values
```

**Priority:** P1 - Breaks neural model CV
**Effort:** 2 days
**Reference:** [specs/cv_improvements.md](specs/cv_improvements.md#sequence-model-cv-fix)

---

## Performance Enhancement Gaps (P2-P3)

These gaps are **optional** but provide significant performance improvements or risk management benefits.

### Gap 8: MTF Lookahead Unverified (P2)

**Location:** `src/phase1/stages/mtf/generator.py`

**Problem:**
Multi-timeframe features aggregate 5-min bars into 15-min, 1H, 4H, daily. If resampling parameters are wrong, these may peek into the future:

```python
# WRONG (lookahead bias):
df.resample('15T', closed='right', label='right').mean()

# CORRECT (no lookahead):
df.resample('15T', closed='left', label='left').mean()
```

**Impact:**
- **Subtle Data Leakage:** May give model future information
- **Optimistic Metrics:** Features that shouldn't exist yet

**Required Fix:**
Automated lookahead audit using corruption testing (see [specs/cv_validation_methods.md](specs/cv_validation_methods.md#lookahead-audit)).

**Priority:** P2 - Potential leakage
**Effort:** 2-3 days

---

### Gap 9: No CPCV (P2)

**Location:** Missing `src/cross_validation/cpcv.py`

**Problem:**
Standard k-fold tests a single temporal path. With hyperparameter tuning, we may select a model that works well on **that specific path** but fails on others.

**Solution:**
Combinatorial Purged Cross-Validation tests C(n,k) combinations of time groups to estimate Probability of Backtest Overfitting (PBO).

**Priority:** P2 - Strengthens hyperparameter selection
**Effort:** 3 days
**Reference:** [specs/cv_cpcv.md](specs/cv_cpcv.md)

---

### Gap 10: No Regime-Adaptive Models (P2)

**Location:** Missing `src/regime/`

**Problem:**
Single model for all market conditions. Research shows regime-specific models can improve Sharpe by 20-30%.

**Solution:**
HMM-based regime detection with specialist models per regime.

**Priority:** P2 - Performance opportunity
**Effort:** 5-7 days

---

### Gap 11: No Conformal Prediction (P3)

**Location:** Missing `src/models/conformal.py`

**Problem:**
No uncertainty quantification. Can't size positions based on prediction reliability.

**Solution:**
Conformal prediction provides prediction intervals calibrated to contain true value with specified probability.

**Priority:** P3 - Advanced risk management
**Effort:** 3-4 days

---

### Gap 12: Time Bars Only (P3)

**Location:** Missing `src/phase1/stages/ingest/alternative_bars.py`

**Problem:**
Time bars have poor statistical properties (heteroscedastic, autocorrelated). Dollar/volume bars are more i.i.d.

**Solution:**
Implement alternative bar types (dollar, volume, tick).

**Priority:** P3 - Data quality improvement
**Effort:** 3-4 days

---

## Priority Matrix

| Gap | Priority | Impact | Effort | Status |
|-----|----------|--------|--------|--------|
| **1. Probability Calibration** | P0 | 30-50% position sizing errors | 2-3 days | Not Started |
| **2. CV Leakage (Scaling)** | P0 | 15-25% metric inflation | 2 days | Not Started |
| **3. Phase 5 (Inference)** | P0 | Blocks deployment | 5-7 days | Not Started |
| **4. Drift Detection** | P1 | Silent model decay | 3-4 days | Not Started |
| **5. Label-Aware Purging** | P1 | Suboptimal purging | 1-2 days | Not Started |
| **6. Walk-Forward Eval** | P1 | Misses temporal degradation | 2-3 days | Not Started |
| **7. Sequence Model CV** | P1 | Invalid neural CV | 2 days | Not Started |
| **8. MTF Lookahead** | P2 | Potential leakage | 2-3 days | Not Started |
| **9. CPCV** | P2 | Weak hyperparameter robustness | 3 days | Not Started |
| **10. Regime-Adaptive** | P2 | 20-30% Sharpe opportunity | 5-7 days | Not Started |
| **11. Conformal Prediction** | P3 | No uncertainty quantification | 3-4 days | Not Started |
| **12. Alternative Bars** | P3 | Suboptimal sampling | 3-4 days | Not Started |

---

## Impact Assessment

### Quantitative Impact Estimates

**P0 Gaps (Must Fix):**
- **Calibration:** 30-50% reduction in position sizing errors
- **CV Scaling:** 5-15% reduction in CV metric optimism (metrics will drop but be honest)
- **Inference:** Enables production deployment (binary: yes/no)

**P1 Gaps (Production Safety):**
- **Drift Detection:** Alerts within 100 samples of distribution shift
- **Label-Aware Purging:** 2-5% improvement in training data utilization
- **Walk-Forward:** Detects temporal degradation (informational, not performance)
- **Sequence CV:** Enables fair neural model comparison

**P2-P3 Gaps (Optional):**
- **CPCV:** Reduces overfitting risk (PBO < 0.5)
- **Regime-Adaptive:** 20-30% Sharpe improvement (research estimate)
- **Conformal:** Better position sizing under uncertainty
- **Alternative Bars:** 5-10% improvement in signal-to-noise

### Deployment Blockers

Before production deployment, **must complete:**
1. ✅ Gap 1: Probability Calibration
2. ✅ Gap 2: CV Scaling Fix
3. ✅ Gap 3: Phase 5 Inference
4. ✅ Gap 4: Drift Detection
5. ✅ Gap 5: Label-Aware Purging (recommended)

**Optional but strongly recommended:**
- Gap 6: Walk-Forward Evaluation (to verify temporal stability)
- Gap 7: Sequence Model CV (if using neural models)

---

## Cross-References

For implementation details, see:
- [ROADMAP.md](ROADMAP.md) - Implementation timeline
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Migration steps and risk mitigation
- Individual spec documents in [specs/](specs/) directory

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial gaps analysis extracted from IMPLEMENTATION_PLAN.md |

**Next Review:** After Phase 1 completion (Week 2)
