# Workflow Best Practices

This document provides critical guidelines for maintaining rigorous machine learning methodology throughout the model development lifecycle.

---

## Table of Contents

1. [Test Set Discipline](#test-set-discipline)
2. [Phase 3→4 Integration](#phase-34-integration)
3. [Preventing Data Leakage](#preventing-data-leakage)
4. [Model Iteration Best Practices](#model-iteration-best-practices)
5. [Cross-Validation Guidelines](#cross-validation-guidelines)

---

## Test Set Discipline

### The Golden Rule

**The test set is for final evaluation ONLY. It is your one-shot generalization estimate.**

### Why This Matters

Every time you evaluate on the test set and make decisions based on those results, you leak information from the test set into your model selection process. This is **overfitting to the test set**, and it invalidates your generalization estimate.

### When to Look at Test Set Results

Look at test set results **ONLY** when:

✅ All hyperparameter tuning is complete
✅ All feature engineering iterations are done
✅ All model architecture experiments are finished
✅ You're ready to make a final decision about deployment
✅ You won't iterate further based on test results

### When NOT to Look at Test Set Results

❌ During hyperparameter tuning
❌ During feature selection
❌ While comparing model architectures
❌ When debugging unexpected behavior
❌ When trying to improve performance

**Instead:** Use the validation set for all development and iteration.

### What to Do if Test Results Are Disappointing

If your test set results are worse than validation results:

1. **DO NOT** tune parameters and re-evaluate on test
2. **DO NOT** add features and re-evaluate on test
3. **DO NOT** try different models and re-evaluate on test

**Instead:**

- Accept the results as your true generalization estimate
- Analyze the gap between validation and test (distribution shift? overfitting?)
- Move on to your next experiment with lessons learned
- Consider using cross-validation for more robust estimates in future experiments

### Default Behavior

By default, `train_model.py` **evaluates on the test set** with clear warnings:

```bash
# Default: Test set evaluation enabled (with warnings)
python scripts/train_model.py --model xgboost --horizon 20

# Disable test set evaluation during development
python scripts/train_model.py --model xgboost --horizon 20 --no-evaluate-test
```

### Recommended Workflow

```
Development Phase (iterate freely):
  python scripts/train_model.py --model xgboost --horizon 20 --no-evaluate-test
  → Evaluate on validation set only
  → Iterate on hyperparameters, features, model architecture
  → Compare models based on validation metrics

Final Evaluation (one-shot, no iteration):
  python scripts/train_model.py --model xgboost --horizon 20 --evaluate-test
  → Evaluate on test set (default behavior)
  → Report final generalization estimate
  → Make deployment decision
  → DO NOT iterate further
```

---

## Phase 3→4 Integration

### Overview

Phase 3 (Cross-Validation) generates leakage-safe out-of-fold (OOF) predictions for stacking ensembles. Phase 4 (Ensemble Training) uses these OOF predictions to train meta-learners.

### Why Use Phase 3 Data for Stacking?

**Problem:** Naive stacking causes data leakage:
1. Train base models on full training set
2. Generate predictions on same training set
3. Train meta-learner on these predictions
4. **Leakage:** Meta-learner sees predictions from models that saw the same data

**Solution:** Use OOF predictions from cross-validation:
1. Train base models with K-fold CV (Phase 3)
2. Generate OOF predictions (each sample predicted by models that didn't train on it)
3. Train meta-learner on OOF predictions (Phase 4)
4. **No leakage:** Meta-learner sees predictions from models that never saw those samples

### Recommended Workflow: Phase 3→4

#### Step 1: Generate OOF Predictions (Phase 3)

```bash
# Run cross-validation with multiple models
python scripts/run_cv.py \
  --models xgboost,lightgbm,catboost,lstm \
  --horizons 20 \
  --n-splits 5 \
  --generate-stacking-data

# This creates: data/stacking/{cv_run_id}/stacking_dataset_h20.parquet
```

**Output:**
- OOF predictions from all base models
- Metadata: model names, CV configuration, label_end_times
- Leakage-safe: Each prediction is from a fold that didn't train on that sample

#### Step 2: Train Stacking Ensemble (Phase 4)

```bash
# Use Phase 3 OOF predictions to train meta-learner
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --stacking-data {cv_run_id}  # From Phase 3

# Example with actual run ID:
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --stacking-data 20251228_143025_789456_a3f9
```

**What happens:**
- Loads OOF predictions from Phase 3
- Trains meta-learner on these predictions
- **No fresh OOF generation** (already done in Phase 3)
- Fast training (only meta-learner, not base models)

### Alternative: Train Ensemble from Scratch (Phase 4 only)

If you want to train an ensemble **without** Phase 3 data:

```bash
# Voting ensemble (no OOF needed, simple averaging)
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20

# Stacking ensemble (generates OOF internally)
python scripts/train_model.py \
  --model stacking \
  --base-models xgboost,lightgbm,lstm \
  --horizon 20
```

**Trade-offs:**
- ✅ Simpler workflow (single command)
- ❌ Slower (trains base models + generates OOF + trains meta-learner)
- ❌ Less control over CV configuration
- ❌ Can't reuse OOF predictions for multiple meta-learners

### When to Generate Fresh OOF Predictions

**Almost never.** Once you have OOF predictions from Phase 3, reuse them for:

- Training different meta-learner types (logistic, random forest, XGBoost)
- Experimenting with meta-learner hyperparameters
- Comparing stacking vs. blending

**Only regenerate OOF when:**
- Base model configurations change
- Feature engineering changes
- Data splits change (different train/val/test cutoffs)

### Benefits of Phase 3→4 Workflow

1. **Leakage Prevention:** OOF predictions ensure no training data leakage
2. **Efficiency:** Generate OOF once, reuse for multiple meta-learners
3. **Reproducibility:** CV configuration stored with OOF predictions
4. **Quality Control:** Metadata tracks which models and folds generated predictions

---

## Preventing Data Leakage

### What is Data Leakage?

Data leakage occurs when information from the test/validation set influences model training, either directly or indirectly. This leads to overly optimistic performance estimates that don't generalize.

### Common Leakage Sources in Time Series

#### 1. Overlapping Labels (Time Leakage)

**Problem:**
```
Sample 1: Bar at t=100, label horizon=20 → Label ends at t=120
Sample 2: Bar at t=110, label horizon=20 → Label ends at t=130
```
If Sample 1 is in training and Sample 2 is in validation, the model can learn the label for Sample 2 indirectly through Sample 1's label.

**Solution:** Use `PurgedKFold` with `label_end_times`:
```python
from src.cross_validation.purged_kfold import PurgedKFold

# Purge samples with overlapping label periods
cv = PurgedKFold(
    n_splits=5,
    purge_bars=60,  # 3x max horizon
    embargo_bars=1440,  # ~5 days
)

# Automatically removes overlapping samples
for train_idx, val_idx in cv.split(X, y, label_end_times):
    # train_idx and val_idx have no overlapping label periods
    ...
```

**How it works:**
- For each validation sample at time `t`, purge training samples with `label_end_time > t - purge_bars`
- This ensures no training label overlaps with validation samples

#### 2. Embargo Period (Serial Correlation)

**Problem:**
Time series data has serial correlation. Training on `t-1` helps predict `t` even if labels don't overlap.

**Solution:** Add an embargo period after each validation fold:
```python
cv = PurgedKFold(
    n_splits=5,
    purge_bars=60,
    embargo_bars=1440,  # Gap between train and validation
)
```

**Recommended embargo:**
- **Minimum:** 3x label horizon (prevents direct label overlap)
- **Better:** 1440 bars (~5 days at 5-min bars) for serial correlation
- **Conservative:** 2880 bars (~10 days) for very noisy data

#### 3. Feature Scaling Leakage

**Problem:**
```python
# WRONG: Fit scaler on train+val, apply to both
scaler = RobustScaler().fit(X_all)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
# Leakage: Validation statistics leaked into training scaling
```

**Solution:** Fit scaler on training set ONLY:
```python
# CORRECT: Fit scaler on train only
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Apply training scaler
X_test_scaled = scaler.transform(X_test)  # Apply training scaler
```

**Pipeline does this correctly:** See `src/phase1/stages/scaling/`

#### 4. Feature Selection Leakage

**Problem:**
```python
# WRONG: Select features on full dataset
selected_features = select_features(X_all, y_all)
X_train = X_train[selected_features]
X_val = X_val[selected_features]
# Leakage: Validation set influenced feature selection
```

**Solution:** Select features using cross-validation:
```python
# CORRECT: Walk-forward feature selection
from src.cross_validation.feature_selector import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(cv=PurgedKFold(...))
selected_features = selector.fit(X_train, y_train, label_end_times_train)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
```

#### 5. Ensemble Leakage (OOF Predictions)

**Problem:**
```python
# WRONG: Train base models on full training set, predict same set
model1.fit(X_train, y_train)
train_preds1 = model1.predict(X_train)  # Leakage!

meta_learner.fit(train_preds1, y_train)
# Meta-learner sees predictions from models that trained on this data
```

**Solution:** Use out-of-fold (OOF) predictions:
```python
# CORRECT: Generate OOF predictions via cross-validation
from src.cross_validation.oof_generator import OOFGenerator

oof_gen = OOFGenerator(cv=PurgedKFold(...))
train_preds_oof = oof_gen.generate_oof(model1, X_train, y_train, label_end_times)

meta_learner.fit(train_preds_oof, y_train)
# Meta-learner sees predictions from models that never saw this data
```

**Phase 3 handles this automatically:** See Phase 3→4 Integration section.

### Leakage Checklist

Before training a model, verify:

- [ ] Train/val/test splits are time-based (no random shuffling)
- [ ] Purge and embargo applied to prevent overlapping labels
- [ ] Feature scaling fit on training set only
- [ ] Feature selection uses cross-validation or training set only
- [ ] Ensemble models use OOF predictions
- [ ] No test set evaluation during development
- [ ] No future data in features (e.g., no shifted features without lag)

---

## Model Iteration Best Practices

### Development Workflow

```
1. Baseline Model (validation only)
   → python scripts/train_model.py --model xgboost --horizon 20 --no-evaluate-test
   → Evaluate on validation set
   → Record baseline metrics

2. Feature Engineering (validation only)
   → Add new features to pipeline
   → Retrain model
   → Compare validation metrics to baseline
   → Iterate until satisfied

3. Hyperparameter Tuning (validation only)
   → Use cross-validation: python scripts/run_cv.py --tune
   → Find best hyperparameters based on CV results
   → Retrain with best params
   → Evaluate on validation set

4. Model Selection (validation only)
   → Train multiple model types
   → Compare validation metrics
   → Select best model(s)

5. Final Evaluation (test set, ONE TIME)
   → python scripts/train_model.py --model best_model --horizon 20 --evaluate-test
   → Report test metrics as final generalization estimate
   → DO NOT iterate further
```

### Metrics to Track During Development

**Validation metrics** (iterate on these):
- Macro F1 (overall performance)
- Per-class F1 (long/short/neutral balance)
- Position win rate (trading accuracy)
- Long/short accuracy (directional edge)
- Position Sharpe (simplified risk-adjusted return)

**Cross-validation metrics** (for hyperparameter tuning):
- Mean CV F1 ± std
- Consistency across folds
- OOF predictions quality

**Test metrics** (report once, DO NOT iterate):
- Final generalization estimate
- True out-of-sample performance

### Common Pitfalls

❌ **Pitfall:** Tune hyperparameters on validation set, then report test results
✅ **Solution:** Use cross-validation for tuning, validation for model selection, test for final report

❌ **Pitfall:** Try 10 models, pick the one with best test F1
✅ **Solution:** Pick model based on validation/CV, then evaluate test ONCE

❌ **Pitfall:** Test results are bad → tweak features → test again
✅ **Solution:** Accept test results, start new experiment from scratch

---

## Cross-Validation Guidelines

### When to Use Cross-Validation

Use CV for:
- **Hyperparameter tuning:** Find optimal params without overfitting
- **Model comparison:** Compare model families with robust estimates
- **Feature selection:** Select features based on cross-validated importance
- **Ensemble stacking:** Generate OOF predictions for meta-learners

### PurgedKFold Configuration

```python
from src.cross_validation.purged_kfold import PurgedKFold

cv = PurgedKFold(
    n_splits=5,           # 5 folds is standard
    purge_bars=60,        # 3x max horizon (prevents label overlap)
    embargo_bars=1440,    # ~5 days (prevents serial correlation)
)
```

**Recommended settings:**
- **n_splits:** 5 (standard), 10 (more robust but slower)
- **purge_bars:** 3× max label horizon (e.g., 60 for horizon=20)
- **embargo_bars:** 1440 (~5 days at 5-min bars), 2880 for conservative

### Running Cross-Validation

```bash
# Basic CV (single model, single horizon)
python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --n-splits 5

# Multi-model, multi-horizon CV
python scripts/run_cv.py \
  --models xgboost,lightgbm,catboost,lstm \
  --horizons 5,10,15,20 \
  --n-splits 5

# CV with hyperparameter tuning (Optuna)
python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --n-splits 5 \
  --tune \
  --n-trials 50

# CV with OOF generation for stacking
python scripts/run_cv.py \
  --models xgboost,lightgbm,catboost \
  --horizons 20 \
  --n-splits 5 \
  --generate-stacking-data
```

### Interpreting CV Results

**Good CV performance:**
- Low variance across folds (consistent)
- Mean CV F1 close to validation F1 (no overfitting)
- OOF predictions align with true labels

**Warning signs:**
- High variance across folds (model unstable)
- CV F1 >> validation F1 (overfitting to CV splits)
- CV F1 << validation F1 (distribution shift in validation set)

### Walk-Forward Validation (Advanced)

For production-like evaluation, use walk-forward validation:

```bash
python scripts/run_walk_forward.py \
  --model xgboost \
  --horizon 20 \
  --n-splits 5 \
  --min-train-samples 10000
```

**Benefits:**
- Mimics production deployment (train on past, predict future)
- Detects regime changes and distribution shift
- More realistic performance estimates

---

## Summary

### The Three Commandments

1. **Thou shalt not iterate on test set results**
   - Test set is one-shot, final evaluation only
   - Use validation/CV for all development

2. **Thou shalt prevent data leakage**
   - Use PurgedKFold for time series
   - Fit scalers on training set only
   - Use OOF predictions for ensembles

3. **Thou shalt use proper cross-validation**
   - PurgedKFold with purge + embargo
   - CV for hyperparameter tuning
   - OOF predictions for stacking

### Quick Reference

| Task | Command | Split to Evaluate |
|------|---------|-------------------|
| Development | `train_model.py --no-evaluate-test` | Validation only |
| Hyperparameter tuning | `run_cv.py --tune` | Cross-validation |
| Model selection | `train_model.py --no-evaluate-test` | Validation only |
| Final evaluation | `train_model.py --evaluate-test` | Test (ONE TIME) |
| Ensemble stacking | `run_cv.py --generate-stacking-data` then `train_model.py --stacking-data {cv_run_id}` | OOF → Meta-learner |

### Further Reading

- **Cross-Validation:** See `docs/implementation/PHASE_3.md`
- **Ensemble Methods:** See `docs/implementation/PHASE_4.md`
- **Data Leakage:** See "Advances in Financial Machine Learning" (López de Prado, 2018), Chapter 7
- **Test Set Discipline:** See "Pattern Recognition and Machine Learning" (Bishop, 2006), Chapter 1.3

---

*Last Updated: 2025-12-28*
