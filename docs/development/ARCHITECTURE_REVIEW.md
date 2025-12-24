# Architecture Review: ML Model Factory

**Reviewer:** Software Architecture Expert
**Date:** 2024-12-24
**Scope:** Phases 1-5 of Ensemble Price Prediction Pipeline
**Verdict:** Solid foundation with addressable gaps

---

## Executive Summary

The ML Factory architecture is **fundamentally sound** and follows modern best practices for ML pipelines. The plugin-based model registry, strict leakage prevention, and modular design are excellent choices. However, there are architectural gaps that should be addressed before Phase 2 implementation begins.

**Overall Assessment: 8/10 - Well-designed with minor gaps**

---

## What's Good About This Approach

### 1. Plugin-Based Model Registry (Excellent)

The decorator-based registration pattern is the right choice:

```python
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    ...
```

**Why this works:**
- Zero-friction model addition (just implement interface + add decorator)
- Auto-discovery eliminates manual wiring
- Fail-fast validation at registration time catches contract violations early
- Factory pattern (`ModelRegistry.create()`) centralizes instantiation logic

**Verdict:** This is industry-standard (similar to TensorFlow's registry, HuggingFace's model hub). Keep it.

### 2. Leakage Prevention (Rigorous)

The 4-layer defense system is comprehensive:

| Layer | Implementation | Status |
|-------|----------------|--------|
| Config validation | `PURGE_BARS >= max_bars` enforced at import | Correct |
| Temporal splits | Purge (60 bars) + Embargo (1440 bars) | Correct |
| Train-only scaling | `scaler.fit(train)` then `transform(val/test)` | Correct |
| Cross-asset validation | Length mismatch detection | Correct |

**Sentinel labels (-99)** for invalid samples is a nice touch - prevents models from learning spurious patterns at data boundaries.

### 3. Phase 1 Output Contract

The Phase 1 output structure is clean:

```
data/splits/scaled/
  train_scaled.parquet  (87,094 x 126)
  val_scaled.parquet    (18,591 x 126)
  test_scaled.parquet   (18,592 x 126)
```

**What's right:**
- Pre-scaled data (train statistics applied to all splits)
- Parquet format (efficient, preserves types)
- Clear separation of splits
- Labels embedded in same files (no join needed)

### 4. Triple-Barrier Labeling with GA Optimization

The labeling approach is sophisticated:

- **Symbol-specific barriers:** MES asymmetric (k_down > k_up) to counter equity drift; MGC symmetric for mean-reversion
- **GA optimization:** DEAP-based search with Sharpe + drawdown fitness
- **Quality weighting:** Sample weights based on signal quality tiers

This is exactly how quantitative firms approach labeling - the asymmetric barriers for MES is a nice insight about equity index drift.

### 5. Modular Package Structure

The refactoring of large files into packages is correct:

```
feature_scaler/
  core.py (195 lines)
  scalers.py (125 lines)
  scaler.py (546 lines)
  validators.py (366 lines)
```

**Adherence to 650-line limit** is unusual but enforces good decomposition. This is a reasonable constraint.

---

## What's Missing or Could Be Improved

### 1. Cross-Validation Strategy Has a Gap

**Phase 3 describes purged k-fold CV, but there's a subtle issue:**

The current approach:
1. Train on fold train data
2. Predict on fold test data
3. Aggregate all fold predictions for stacking

**The gap:** Walk-forward validation is more realistic for time series:

```
Fold 1: Train [2015-2017], Test [2018-Q1]
Fold 2: Train [2015-2018-Q1], Test [2018-Q2]  (expanding window)
Fold 3: Train [2015-2018-Q2], Test [2018-Q3]
...
```

**Current approach (blocked k-fold)** risks:
- Testing on older data than training (unrealistic)
- Regime shifts between non-adjacent folds

**Recommendation:** Add walk-forward CV as an option alongside purged k-fold. The docs mention it as "alternative" but it should be the primary method for final validation.

### 2. Meta-Learner is Too Simple

Phase 4 recommends Logistic Regression for stacking:

```python
meta_lr = LogisticRegression(multi_class='multinomial', C=1.0)
```

**The gap:** This ignores the time-varying nature of model strengths.

**Better approach: Regime-Conditional Meta-Learner**

```python
# Instead of flat stacking:
ensemble_prob = meta_lr.predict_proba(model_probs)

# Use regime-aware stacking:
class RegimeAwareMeta:
    def __init__(self):
        self.high_vol_meta = LogisticRegression()
        self.low_vol_meta = LogisticRegression()

    def predict(self, X, regime):
        if regime == 'high_vol':
            return self.high_vol_meta.predict_proba(X)
        else:
            return self.low_vol_meta.predict_proba(X)
```

Phase 3's regime analysis already identifies that TFT excels in high volatility. The meta-learner should leverage this.

**Recommendation:** Add `vol_regime` and `trend_regime` as features to meta-learner, or train separate meta-learners per regime.

### 3. No Calibration Step

Model probability outputs are used raw:

```python
probs_1 = softmax(logits_1, dim=-1)
# Directly used for stacking
```

**The gap:** Neural network probabilities are often miscalibrated. A 70% probability might actually mean 55% accuracy.

**Solution:** Add calibration after base model training:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)
```

**Recommendation:** Add Phase 2.5 - Probability Calibration before stacking.

### 4. Missing Uncertainty Quantification

The architecture produces point predictions:

```python
predictions = {
    '5': {'class': 2, 'signal': +1}  # Just class, no uncertainty
}
```

**The gap:** Trading systems need uncertainty estimates:
- Low confidence predictions should have smaller position sizes
- Ensemble disagreement indicates regime uncertainty

**Solution:** Add prediction intervals or entropy-based confidence:

```python
def predict_with_confidence(self, X):
    probs = self.predict_proba(X)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    confidence = 1 - (entropy / np.log(3))  # Normalized
    return {
        'class': np.argmax(probs, axis=1),
        'confidence': confidence,
        'probs': probs
    }
```

### 5. No Regime Detection Model

Regime analysis happens in Phase 3 after training, but regime detection should be a first-class model:

**Current:**
- Volatility regime: ATR percentile thresholds (fixed)
- Trend regime: SMA crossover (lagging)

**Better:**
- Train a dedicated regime classifier (Hidden Markov Model or simple LSTM)
- Use regime as explicit input feature to all models
- Enable regime-conditional trading rules

**Recommendation:** Add Stage 3.5 - Regime Model Training.

### 6. Stacking Architecture is Shallow

Phase 4 describes single-level stacking:

```
Base Models [3] -> Meta-Learner -> Final Prediction
```

**The gap:** Research shows multi-level stacking often improves performance:

```
Level 0: Base Models (N-HiTS, TFT, PatchTST)
Level 1: Family Meta-Learners (boosting-meta, timeseries-meta)
Level 2: Final Meta-Learner
```

However, for 3 base models, single-level stacking is probably sufficient. The complexity isn't worth it until you have 6+ models.

**Recommendation:** Keep single-level for now; extend if adding more model families.

### 7. No Online Learning / Model Update Strategy

The architecture is static:

```
Train once -> Deploy -> Never update
```

**The gap:** Financial regimes change. Models decay.

**Missing pieces:**
- Model staleness detection (performance monitoring)
- Incremental update strategy (when to retrain)
- A/B testing framework (new model vs. production)

**Recommendation:** Add Phase 7 - Model Lifecycle Management.

---

## Data Flow Analysis: Phase 1 -> Phase 2

### Current Contract

```
Phase 1 Output:
  data/splits/scaled/train_scaled.parquet
    Columns: [datetime, symbol, features..., label_h5, label_h20, sample_weight_h5, ...]
    Rows: 87,094

Phase 2 Input:
  TimeSeriesDataset(
    train_path='data/splits/scaled/train_scaled.parquet',
    horizon=5,
    sequence_length=60
  )
```

**Gap:** The `TimeSeriesDataset` class is described but not implemented:

```python
# Described in ROADMAP.md but not in src/
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, label_col, lookback, indices):
        ...
```

**Recommendation:** Implement `TimeSeriesDataset` before starting Phase 2 model training.

### Sequence Length Mismatch Risk

Phase 2 models use different sequence lengths:

| Model | Recommended seq_len |
|-------|---------------------|
| N-HiTS | 128 |
| TFT | 128 |
| PatchTST | 128 |
| XGBoost | 1 (no sequence) |

But Phase 1 purge/embargo is based on `max_bars=60`.

**Potential issue:** If `seq_len=128 > PURGE_BARS=60`, there's a risk of feature leakage.

**Recommendation:** Verify that `PURGE_BARS >= max(max_bars, seq_len)`. Update config:

```python
PURGE_BARS = max(60, 128)  # = 128
```

---

## Ensemble Approach: Is Stacking Right?

### Stacking is Appropriate Here

For this use case, stacking is the right choice:

1. **Diverse base learners:** MLP (N-HiTS), Transformer (TFT, PatchTST), potentially boosting
2. **Different inductive biases:** Local patterns (N-HiTS) vs. long-range dependencies (PatchTST)
3. **Complementary errors:** Phase 3 shows 60-70% prediction correlation (not too high, not too low)

### Alternative: Boosted Meta-Learner

If stacking doesn't beat simple averaging, consider:

```python
# Instead of Logistic Regression meta-learner:
from xgboost import XGBClassifier

meta_xgb = XGBClassifier(
    n_estimators=50,
    max_depth=3,  # Keep shallow
    learning_rate=0.1
)
```

XGBoost can capture non-linear interactions between model predictions that LogReg misses.

### Alternative: Confidence-Weighted Voting

Simpler than stacking, often works well:

```python
def confidence_weighted_vote(model_probs, model_confidences):
    weights = model_confidences / model_confidences.sum()
    weighted_probs = sum(p * w for p, w in zip(model_probs, weights))
    return np.argmax(weighted_probs)
```

**Recommendation:** Implement confidence-weighted voting as baseline to compare against stacking.

---

## Specific Recommendations

### High Priority (Before Phase 2)

1. **Implement TimeSeriesDataset**
   - Zero-leakage windowing
   - Symbol isolation
   - Support variable sequence lengths
   - Estimated effort: 5 hours

2. **Update PURGE_BARS to account for sequence length**
   ```python
   PURGE_BARS = max(60, 128)  # Whichever is larger
   ```

3. **Add probability calibration pipeline**
   - Isotonic regression or Platt scaling
   - Validate on held-out calibration set
   - Estimated effort: 3 hours

### Medium Priority (During Phase 2-4)

4. **Regime-aware meta-learner**
   - Add `vol_regime`, `trend_regime` as meta-features
   - Consider separate meta-learners per regime
   - Estimated effort: 4 hours

5. **Add prediction confidence output**
   - Entropy-based confidence scores
   - Ensemble disagreement metric
   - Estimated effort: 2 hours

6. **Walk-forward CV option**
   - Expanding window validation
   - More realistic than blocked k-fold
   - Estimated effort: 6 hours

### Lower Priority (Phase 6+)

7. **Model lifecycle management**
   - Performance monitoring
   - Staleness detection
   - Retraining triggers

8. **Dedicated regime model**
   - HMM or LSTM for regime classification
   - Regime as explicit model input

---

## Summary Table

| Aspect | Assessment | Score |
|--------|------------|-------|
| Plugin architecture | Excellent - industry standard | 9/10 |
| Leakage prevention | Rigorous - 4-layer defense | 9/10 |
| Data flow (Phase 1 -> 2) | Good, but seq_len gap | 7/10 |
| Cross-validation | Adequate, prefer walk-forward | 7/10 |
| Ensemble strategy | Correct choice (stacking) | 8/10 |
| Meta-learner design | Too simple, needs regime awareness | 6/10 |
| Calibration | Missing | 4/10 |
| Uncertainty quantification | Missing | 4/10 |
| Model lifecycle | Missing | 3/10 |

---

## Final Verdict

**Is this the right direction? YES.**

The architecture is solid. The plugin-based model registry, rigorous leakage prevention, and modular design are exactly right. The gaps identified are refinements, not fundamental flaws.

**Critical path items before Phase 2:**
1. Implement `TimeSeriesDataset`
2. Update `PURGE_BARS >= 128`
3. Add probability calibration

**Recommended additions during Phase 2-4:**
1. Regime-aware meta-learner
2. Prediction confidence scores
3. Walk-forward CV

**The architecture will support production deployment** once these gaps are addressed. The author clearly understands ML pipeline pitfalls (leakage, regime shifts, transaction costs) and has designed accordingly.

---

**Reviewed by:** Architecture Expert
**Confidence:** High
**Next action:** Proceed with Phase 2 after implementing critical path items
