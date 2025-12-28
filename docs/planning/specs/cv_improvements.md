# Cross-Validation Core Improvements Specification

**Version:** 1.0.0
**Date:** 2025-12-28
**Priority:** P0-P1 (Critical)

---

## Overview

This specification covers three critical fixes to the cross-validation system:

1. **Fold-Aware Scaling** (P0) - Prevents data leakage from global scaling
2. **Label-Aware Purging** (P1) - Uses actual label resolution times for purging
3. **Sequence Model CV Fix** (P1) - Enables proper CV for LSTM/GRU/TCN models

### Combined Impact

| Metric | Before | After |
|--------|--------|-------|
| CV Metric Optimism | +15-25% | < 5% |
| Training Data Utilization | Suboptimal (fixed purge) | Optimal (label-aware) |
| Neural Model CV | Broken | Working |

---

## 1. Fold-Aware Scaling

### Problem Statement

**Location:** `src/cross_validation/oof_generator.py` line 197-198

The current CV loads pre-scaled data where the scaler was fit on the entire training set. Each fold's validation data has been scaled using statistics that include "future" samples from other folds.

```python
# Current (WRONG)
# Phase 1: Fit scaler on ALL training data
scaler.fit(X_train)  # X_train = entire 70%

# Phase 3 CV: Load pre-scaled data
X, y, weights = container.get_sklearn_arrays("train")  # Already scaled
for train_idx, val_idx in cv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # X_val was scaled using statistics from samples in train_idx!
```

**Impact:**
- CV metrics inflated by 5-15%
- Optimistic model selection
- Val-test gap of 15-25%

### Solution

Scale within each fold using only that fold's training data:

```python
for train_idx, val_idx in cv.split(X):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
    X_val_scaled = scaler.transform(X.iloc[val_idx])
```

### Implementation

**File: `src/cross_validation/fold_scaling.py`**

See full implementation in the appendix or original IMPLEMENTATION_PLAN.md section 6.2.

Key class: `FoldAwareScaler`

**Integration into `src/cross_validation/oof_generator.py`:**

Replace lines 193-198:

```python
from src.cross_validation.fold_scaling import FoldAwareScaler

for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
    # Extract fold data
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Apply fold-aware scaling
    scaler = FoldAwareScaler(method="robust")
    X_train, X_val = scaler.fit_transform_fold(
        X_train_raw.values, X_val_raw.values
    )
```

### Testing

```python
# Test that scaling doesn't leak
def test_no_leakage():
    X = np.random.randn(1000, 10)
    train_idx = np.arange(700)
    val_idx = np.arange(700, 1000)
    
    scaler = FoldAwareScaler()
    X_train, X_val = scaler.fit_transform_fold(
        X[train_idx], X[val_idx]
    )
    
    # Scaler should only have seen training data
    assert scaler._scaler.center_.shape == (10,)
    # Verify statistics computed only from training
    expected_center = np.median(X[train_idx], axis=0)
    np.testing.assert_array_almost_equal(
        scaler._scaler.center_, expected_center
    )
```

### Acceptance Criteria

- [ ] CV F1 drops 5-15% (expected - removes optimism)
- [ ] Val-test gap decreases to < 10%
- [ ] Fold scalers fit only on fold training data
- [ ] Test passes: no statistics leak between folds

---

## 2. Label-Aware Purging

### Problem Statement

**Location:** `src/phase1/stages/labeling/triple_barrier.py`, `src/cross_validation/purged_kfold.py`

Triple-barrier labels have variable resolution times (some hit barriers at 5 bars, some timeout at 20 bars). Current pipeline uses fixed purge of 60 bars for all samples, which is suboptimal.

**Impact:**
- Over-purging for labels that resolved quickly (removes more training data than needed)
- Potential under-purging for labels that took full timeout

### Solution

Compute `label_end_time` during labeling and use it for precise purging:

```python
label_end_time = label_start_time + (bars_to_hit * bar_interval)
```

### Implementation

**Add to `src/phase1/stages/labeling/triple_barrier.py`:**

```python
def compute_label_end_times(
    df: pd.DataFrame,
    bars_to_hit: np.ndarray,
    bar_interval_minutes: int = 5,
) -> pd.Series:
    """
    Compute when each label's outcome is known.
    
    For triple-barrier labels, the outcome is known when a barrier is hit
    or timeout occurs. This is essential for proper CV purging.
    """
    datetime_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df['datetime']
    
    label_end_times = datetime_col + pd.to_timedelta(
        bars_to_hit * bar_interval_minutes, unit='m'
    )
    
    return pd.Series(label_end_times, index=df.index, name='label_end_time')
```

**Modify `TripleBarrierLabeler.compute_labels()`:**

Add after computing bars_to_hit:

```python
# Compute label end times for CV purging
label_end_times = compute_label_end_times(
    df, bars_to_hit, bar_interval_minutes=5
)
result.metadata['label_end_times'] = label_end_times.values
```

**Wire through to CV:**

Modify `src/cross_validation/oof_generator.py`:

```python
# Get label end times if available
label_end_times = kwargs.get('label_end_times', None)

for fold_idx, (train_idx, val_idx) in enumerate(
    self.cv.split(X, y, label_end_times=label_end_times)
):
    ...
```

### Testing

```python
def test_label_end_times():
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=100, freq='5T'),
        'close': np.random.randn(100),
    }).set_index('datetime')
    
    bars_to_hit = np.array([5, 10, 20] * 33 + [5])
    
    end_times = compute_label_end_times(df, bars_to_hit, bar_interval_minutes=5)
    
    # Verify first sample
    expected_end = df.index[0] + pd.Timedelta(minutes=5*5)
    assert end_times.iloc[0] == expected_end
```

### Acceptance Criteria

- [ ] `label_end_time_h{horizon}` column in parquet files
- [ ] CV uses label-aware purging
- [ ] Training data utilization improves by 2-5%
- [ ] No leakage: validation labels don't depend on test data

---

## 3. Sequence Model CV Fix

### Problem Statement

**Location:** `src/cross_validation/cv_runner.py` line 327

All models use the same data loading:

```python
X, y, weights = container.get_sklearn_arrays("train", return_df=True)
# Returns 2D: (n_samples, n_features)
```

But LSTM/GRU/TCN need 3D sequences: `(n_samples, seq_len, n_features)`.

**Impact:**
- LSTM/GRU/TCN CV is currently broken
- Can't fairly compare boosting vs neural models

### Solution

Model-aware data loading in CV:

```python
if model.requires_sequences:
    X, y = create_sequences(X_df, y_series, seq_len=60)
else:
    X, y = X_df.values, y_series.values
```

### Implementation

**Modify `src/cross_validation/cv_runner.py`:**

Replace `_run_single_cv` method (line 317):

```python
def _run_single_cv(
    self,
    container: "TimeSeriesDataContainer",
    model_name: str,
    horizon: int,
) -> CVResult:
    """Run CV for single model/horizon combination."""
    # Get model info to determine data requirements
    model_info = ModelRegistry.get_model_info(model_name)
    requires_sequences = model_info.get("requires_sequences", False)
    
    # Get appropriate data format
    X_df, y_series, weights = container.get_sklearn_arrays("train", return_df=True)
    
    if requires_sequences:
        seq_len = self.sequence_length or 60
        logger.info(f"Using sequence data with seq_len={seq_len}")
        use_sequence_cv = True
    else:
        use_sequence_cv = False
    
    # Generate OOF predictions
    if use_sequence_cv:
        oof_pred = self._generate_sequence_oof(
            X_df, y_series, weights, cv_splits, model_name, seq_len, horizon
        )
    else:
        oof_generator = OOFGenerator(self.cv)
        oof_predictions = oof_generator.generate_oof_predictions(...)
        oof_pred = oof_predictions[model_name]
```

**Add new method `_generate_sequence_oof`:**

```python
def _generate_sequence_oof(
    self,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    weights: pd.Series,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    seq_len: int,
    horizon: int,
) -> OOFPrediction:
    """Generate OOF predictions for sequence models."""
    from src.phase1.stages.datasets.sequences import create_sequences
    
    n_samples = len(X_df)
    n_classes = 3
    
    oof_probs = np.full((n_samples, n_classes), np.nan)
    oof_preds = np.full(n_samples, np.nan)
    fold_info = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Build sequences for this fold
        X_train_seq, y_train_seq = create_sequences(
            X_df.iloc[train_idx].values,
            y_series.iloc[train_idx].values,
            seq_len=seq_len,
        )
        X_val_seq, y_val_seq = create_sequences(
            X_df.iloc[val_idx].values,
            y_series.iloc[val_idx].values,
            seq_len=seq_len,
        )
        
        # Train and predict
        model = ModelRegistry.create(model_name)
        model.fit(
            X_train=X_train_seq,
            y_train=y_train_seq,
            X_val=X_val_seq,
            y_val=y_val_seq,
        )
        predictions = model.predict(X_val_seq)
        
        # Map predictions back to original indices
        valid_val_idx = val_idx[seq_len - 1:]
        for i, orig_idx in enumerate(valid_val_idx):
            if i < len(predictions.class_predictions):
                oof_probs[orig_idx] = predictions.class_probabilities[i]
                oof_preds[orig_idx] = predictions.class_predictions[i]
    
    # Build OOF result
    oof_df = pd.DataFrame({
        "datetime": X_df.index,
        f"{model_name}_prob_short": oof_probs[:, 0],
        f"{model_name}_prob_neutral": oof_probs[:, 1],
        f"{model_name}_prob_long": oof_probs[:, 2],
        f"{model_name}_pred": oof_preds,
    })
    
    coverage = float((~np.isnan(oof_preds)).mean())
    
    return OOFPrediction(
        model_name=model_name,
        predictions=oof_df,
        fold_info=fold_info,
        coverage=coverage,
    )
```

### Testing

```bash
# Test LSTM CV
python scripts/run_cv.py --models lstm,gru --horizons 20 --n-splits 5

# Verify:
# - Completes without errors
# - F1 scores are reasonable (0.35-0.45)
# - Sequences don't cross fold boundaries
```

### Acceptance Criteria

- [ ] LSTM/GRU/TCN CV runs without errors
- [ ] Sequences built per fold (no cross-fold leakage)
- [ ] Coverage > 80% (some samples lost due to seq_len)
- [ ] Temporal ordering preserved
- [ ] F1 scores comparable to standalone training

---

## Cross-References

- [ROADMAP.md](../ROADMAP.md) - Implementation timeline
- [GAPS_ANALYSIS.md](../GAPS_ANALYSIS.md) - Detailed gap analysis
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Step-by-step migration

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | ML Engineering | Initial CV improvements spec from IMPLEMENTATION_PLAN.md |
