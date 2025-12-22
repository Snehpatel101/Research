# Phase 4 – Train the Ensemble Meta-Learner (Stacking)

## Overview

Phase 4 trains a meta-learner that combines the three base models (N-HiTS, TFT, PatchTST) into a final ensemble. The meta-learner learns optimal weights for each model's predictions based on their strengths and weaknesses.

**Phase 4 Goal:** Create a stacked ensemble that outperforms any individual base model by intelligently weighting their predictions.

**Think of Phase 4 as:** "Teaching a 'manager quant' how to listen to three analysts and make the final trading call."

---

## Objectives

### Primary Goals
- Train a meta-learner (stacking model) on out-of-sample predictions from Phase 3
- Learn model-specific weights and interaction patterns
- Achieve ensemble performance that beats best single model
- Generate final ensemble predictions on validation set
- Measure improvement from stacking vs simple averaging

### Success Criteria
- Ensemble Sharpe > max(individual model Sharpe) by at least 0.1
- Ensemble F1 ≥ best individual model F1
- Meta-learner does not overfit (simple, interpretable model)
- Final predictions ready for Phase 5 (test set evaluation)

---

## Prerequisites

### Required Inputs from Phase 3
- `data/stacking/oos_predictions.parquet`
  - Contains: model predictions (9 prob columns per horizon) + true labels
  - All predictions are out-of-sample (CV-generated)

### Infrastructure Requirements
- CPU sufficient (meta-learner is lightweight)
- RAM: 16-32 GB
- Libraries: scikit-learn, xgboost (optional), pandas, numpy

---

## Meta-Learner Design

### Why Stacking Works

**Intuition:**
- Each base model has strengths and weaknesses
- N-HiTS: fast, stable, good baseline
- TFT: strong on heterogeneous features, interpretable
- PatchTST: excellent long-range, best on 20-bar

**Stacking Goal:**
- Learn: "When should I trust N-HiTS vs TFT vs PatchTST?"
- Example: "In high vol, trust TFT more. In sideways markets, trust N-HiTS."

**Meta-learner learns weights:**
```
final_prob = w_nhits * P_nhits + w_tft * P_tft + w_patchtst * P_patchtst
```

Or more complex interactions if using non-linear meta-model.

### Meta-Learner Options

#### Option 1: Logistic Regression (Recommended)

**Pros:**
- Simple, interpretable
- Fast to train
- Learns linear weights per model
- Regularization prevents overfitting

**Structure:**
```
Inputs: [P_nhits_class0, P_nhits_class1, P_nhits_class2,
         P_tft_class0, P_tft_class1, P_tft_class2,
         P_patchtst_class0, P_patchtst_class1, P_patchtst_class2]
         → 9 features per horizon

Output: Final class prediction {0, 1, 2}

Logistic Regression (multinomial) with L2 regularization
```

**When to use:**
- First pass, want interpretability
- Base models already strong (just need smart weighting)

#### Option 2: Gradient Boosted Trees (XGBoost/LightGBM)

**Pros:**
- Can learn non-linear interactions
- Can incorporate additional features (regime flags, etc.)
- Often gives small boost over logistic regression

**Cons:**
- More complex, less interpretable
- Easier to overfit

**When to use:**
- Logistic regression not giving improvement
- Want to include regime features or other meta-features

#### Option 3: Simple Averaging (Baseline)

**No learning, just average:**
```
final_prob = (P_nhits + P_tft + P_patchtst) / 3
```

**Use as benchmark:**
- Stacking should beat this to justify complexity

---

## Stacking Dataset Structure

### Load OOS Predictions

```python
import pandas as pd
import numpy as np

# Load stacking data from Phase 3
stacking_df = pd.read_parquet('data/stacking/oos_predictions.parquet')

print(f"Stacking dataset shape: {stacking_df.shape}")
print(stacking_df.head())
```

**Expected columns:**
```
sample_idx, timestamp, symbol
nhits_1_p0, nhits_1_p1, nhits_1_p2  (3 probs for 1-bar horizon)
nhits_5_p0, nhits_5_p1, nhits_5_p2  (3 probs for 5-bar horizon)
nhits_20_p0, nhits_20_p1, nhits_20_p2
tft_1_p0, tft_1_p1, tft_1_p2
tft_5_p0, tft_5_p1, tft_5_p2
tft_20_p0, tft_20_p1, tft_20_p2
patchtst_1_p0, patchtst_1_p1, patchtst_1_p2
patchtst_5_p0, patchtst_5_p1, patchtst_5_p2
patchtst_20_p0, patchtst_20_p1, patchtst_20_p2
label_1, label_5, label_20  (true labels)
```

### Feature Engineering for Meta-Learner

**Option A: Raw Probabilities Only (simplest)**
```python
# For 5-bar horizon example
X_meta = stacking_df[['nhits_5_p0', 'nhits_5_p1', 'nhits_5_p2',
                       'tft_5_p0', 'tft_5_p1', 'tft_5_p2',
                       'patchtst_5_p0', 'patchtst_5_p1', 'patchtst_5_p2']].values
y_meta = stacking_df['label_5'].values
```

**Option B: Add Derived Features**
```python
# Confidence features
stacking_df['nhits_5_confidence'] = stacking_df[['nhits_5_p0', 'nhits_5_p1', 'nhits_5_p2']].max(axis=1)
stacking_df['tft_5_confidence'] = stacking_df[['tft_5_p0', 'tft_5_p1', 'tft_5_p2']].max(axis=1)
stacking_df['patchtst_5_confidence'] = stacking_df[['patchtst_5_p0', 'patchtst_5_p1', 'patchtst_5_p2']].max(axis=1)

# Agreement features
stacking_df['nhits_pred'] = stacking_df[['nhits_5_p0', 'nhits_5_p1', 'nhits_5_p2']].idxmax(axis=1).str[-1].astype(int)
stacking_df['tft_pred'] = stacking_df[['tft_5_p0', 'tft_5_p1', 'tft_5_p2']].idxmax(axis=1).str[-1].astype(int)
stacking_df['patchtst_pred'] = stacking_df[['patchtst_5_p0', 'patchtst_5_p1', 'patchtst_5_p2']].idxmax(axis=1).str[-1].astype(int)

stacking_df['model_agreement'] = (stacking_df['nhits_pred'] == stacking_df['tft_pred']).astype(int) + \
                                  (stacking_df['tft_pred'] == stacking_df['patchtst_pred']).astype(int)

# Add to features
X_meta = stacking_df[['nhits_5_p0', 'nhits_5_p1', 'nhits_5_p2',
                       'tft_5_p0', 'tft_5_p1', 'tft_5_p2',
                       'patchtst_5_p0', 'patchtst_5_p1', 'patchtst_5_p2',
                       'nhits_5_confidence', 'tft_5_confidence', 'patchtst_5_confidence',
                       'model_agreement']].values
```

**Option C: Add Regime Features** (if Phase 3 regime analysis showed strong patterns)
```python
# Merge regime info from original val dataset
# e.g., vol_regime, trend_regime

X_meta = np.hstack([
    X_meta,
    stacking_df[['vol_regime', 'trend_regime']].values
])
```

**Recommendation:** Start with Option A (raw probs only), add derived features if needed.

---

## Training Meta-Learner

### Split Stacking Data

**Important:** Use a holdout from stacking data to validate meta-learner.

```python
from sklearn.model_selection import train_test_split

# Split 80/20 for meta-train / meta-val
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
    X_meta, y_meta, test_size=0.2, shuffle=False  # Time-ordered split
)

print(f"Meta-train: {len(X_meta_train)}, Meta-val: {len(X_meta_val)}")
```

**Note:** Shuffle=False maintains time order (conservative). Can shuffle if comfortable.

### Option 1: Logistic Regression Meta-Learner

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Train logistic regression
meta_lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,  # Regularization strength (inverse of lambda)
    max_iter=500,
    random_state=42
)

meta_lr.fit(X_meta_train, y_meta_train)

# Predict on meta-val
meta_val_preds = meta_lr.predict(X_meta_val)
meta_val_probs = meta_lr.predict_proba(X_meta_val)

# Metrics
acc = accuracy_score(y_meta_val, meta_val_preds)
f1 = f1_score(y_meta_val, meta_val_preds, average='macro')
logloss = log_loss(y_meta_val, meta_val_probs)

print(f"Meta-Learner Validation:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 (macro): {f1:.4f}")
print(f"  Log Loss: {logloss:.4f}")

# Compare to best single model on same val set
# (use Phase 3 results)
```

**Interpret Coefficients:**
```python
# For each class, print feature importances (coefficients)
feature_names = ['nhits_p0', 'nhits_p1', 'nhits_p2',
                 'tft_p0', 'tft_p1', 'tft_p2',
                 'patchtst_p0', 'patchtst_p1', 'patchtst_p2']

for class_idx in range(3):
    print(f"\nClass {class_idx} coefficients:")
    coefs = meta_lr.coef_[class_idx]
    for name, coef in zip(feature_names, coefs):
        print(f"  {name}: {coef:.4f}")
```

**Example output:**
```
Class 2 (+1, long signal) coefficients:
  nhits_p2: 0.85
  tft_p2: 1.20
  patchtst_p2: 1.10
  ...
```
→ For long signals, TFT and PatchTST get more weight than N-HiTS.

### Option 2: XGBoost Meta-Learner

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

# Train XGBoost
meta_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=3,  # Keep shallow to avoid overfitting
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

meta_xgb.fit(
    X_meta_train, y_meta_train,
    eval_set=[(X_meta_val, y_meta_val)],
    early_stopping_rounds=10,
    verbose=False
)

# Predict
meta_val_preds = meta_xgb.predict(X_meta_val)
meta_val_probs = meta_xgb.predict_proba(X_meta_val)

# Metrics
acc = accuracy_score(y_meta_val, meta_val_preds)
f1 = f1_score(y_meta_val, meta_val_preds, average='macro')

print(f"XGBoost Meta-Learner Validation:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1: {f1:.4f}")

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(meta_xgb, max_num_features=10)
plt.tight_layout()
plt.savefig('reports/phase4_xgb_feature_importance.png')
```

### Baseline: Simple Averaging

```python
# Average model probabilities
avg_probs = (X_meta_val[:, 0:3] + X_meta_val[:, 3:6] + X_meta_val[:, 6:9]) / 3
avg_preds = np.argmax(avg_probs, axis=1)

# Metrics
acc_avg = accuracy_score(y_meta_val, avg_preds)
f1_avg = f1_score(y_meta_val, avg_preds, average='macro')

print(f"Simple Averaging Baseline:")
print(f"  Accuracy: {acc_avg:.4f}")
print(f"  F1: {f1_avg:.4f}")
```

**Comparison:**
| Method | Accuracy | F1 | Sharpe |
|--------|----------|-----|--------|
| Best Single Model (PatchTST) | 0.565 | 0.43 | 0.62 |
| Simple Averaging | 0.571 | 0.44 | 0.65 |
| Logistic Regression | 0.578 | 0.46 | 0.68 |
| XGBoost | 0.582 | 0.47 | 0.70 |

**Expectation:** Stacking should beat simple averaging by 1-3% in F1 and 0.05-0.10 in Sharpe.

---

## Training Separate Meta-Learners Per Horizon

**Strategy:** Train one meta-learner for each horizon (1-bar, 5-bar, 20-bar).

**Rationale:**
- Different horizons may need different model weightings
- 1-bar: fast, noisy → may prefer N-HiTS
- 20-bar: long-range → may prefer PatchTST

```python
meta_learners = {}

for horizon in ['1', '5', '20']:
    print(f"\nTraining meta-learner for {horizon}-bar horizon...")
    
    # Extract features for this horizon
    feature_cols = [f'nhits_{horizon}_p0', f'nhits_{horizon}_p1', f'nhits_{horizon}_p2',
                    f'tft_{horizon}_p0', f'tft_{horizon}_p1', f'tft_{horizon}_p2',
                    f'patchtst_{horizon}_p0', f'patchtst_{horizon}_p1', f'patchtst_{horizon}_p2']
    
    X_h = stacking_df[feature_cols].values
    y_h = stacking_df[f'label_{horizon}'].values
    
    # Split
    X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
        X_h, y_h, test_size=0.2, shuffle=False
    )
    
    # Train logistic regression
    meta_h = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=500)
    meta_h.fit(X_h_train, y_h_train)
    
    # Evaluate
    preds_h = meta_h.predict(X_h_val)
    acc_h = accuracy_score(y_h_val, preds_h)
    f1_h = f1_score(y_h_val, preds_h, average='macro')
    
    print(f"  {horizon}-bar: Acc={acc_h:.4f}, F1={f1_h:.4f}")
    
    # Save
    meta_learners[horizon] = meta_h
    
    import joblib
    joblib.dump(meta_h, f'models/meta_learner_{horizon}bar.pkl')
```

**Output:**
```
Training meta-learner for 1-bar horizon...
  1-bar: Acc=0.552, F1=0.40

Training meta-learner for 5-bar horizon...
  5-bar: Acc=0.578, F1=0.46

Training meta-learner for 20-bar horizon...
  20-bar: Acc=0.591, F1=0.49
```

---

## Generate Final Ensemble Predictions

### On Validation Set

```python
# Load meta-learners
meta_1 = joblib.load('models/meta_learner_1bar.pkl')
meta_5 = joblib.load('models/meta_learner_5bar.pkl')
meta_20 = joblib.load('models/meta_learner_20bar.pkl')

# Full stacking dataset (or just validation portion)
X_1 = stacking_df[[f'nhits_1_p{i}' for i in range(3)] +
                   [f'tft_1_p{i}' for i in range(3)] +
                   [f'patchtst_1_p{i}' for i in range(3)]].values

X_5 = stacking_df[[f'nhits_5_p{i}' for i in range(3)] +
                   [f'tft_5_p{i}' for i in range(3)] +
                   [f'patchtst_5_p{i}' for i in range(3)]].values

X_20 = stacking_df[[f'nhits_20_p{i}' for i in range(3)] +
                    [f'tft_20_p{i}' for i in range(3)] +
                    [f'patchtst_20_p{i}' for i in range(3)]].values

# Predict
ensemble_probs_1 = meta_1.predict_proba(X_1)
ensemble_probs_5 = meta_5.predict_proba(X_5)
ensemble_probs_20 = meta_20.predict_proba(X_20)

# Save
np.save('predictions/ensemble_val_probs_1.npy', ensemble_probs_1)
np.save('predictions/ensemble_val_probs_5.npy', ensemble_probs_5)
np.save('predictions/ensemble_val_probs_20.npy', ensemble_probs_20)

print("Ensemble predictions saved.")
```

### Backtest Ensemble

```python
import vectorbt as vbt

# Load validation price data
val_df = pd.read_parquet('data/final/combined_final_labeled.parquet')
val_indices = np.load('config/splits/val_indices.npy')
val_df = val_df.iloc[val_indices].reset_index(drop=True)

# Use 5-bar ensemble predictions
pred_classes_5 = np.argmax(ensemble_probs_5, axis=1) - 1  # {0,1,2} → {-1,0,+1}

entries_long = (pred_classes_5 == 1)
entries_short = (pred_classes_5 == -1)

pf = vbt.Portfolio.from_signals(
    close=val_df['close'].values,
    entries=entries_long,
    exits=entries_short,
    freq='5T',
)

sharpe_ensemble = pf.sharpe_ratio()
dd_ensemble = pf.max_drawdown()
n_trades = pf.trades.count()
win_rate = pf.trades.win_rate()

print(f"\nEnsemble 5-bar Validation Backtest:")
print(f"  Sharpe: {sharpe_ensemble:.3f}")
print(f"  Max DD: {dd_ensemble:.3%}")
print(f"  Trades: {n_trades}")
print(f"  Win Rate: {win_rate:.3%}")

# Compare to best single model
print(f"\nBest Single Model (PatchTST) 5-bar:")
print(f"  Sharpe: 0.620")  # From Phase 3
print(f"  Improvement: {sharpe_ensemble - 0.620:.3f}")
```

**Expected Result:**
- Ensemble Sharpe: 0.68-0.72
- Improvement over best single: +0.05 to +0.10

**Save tearsheet:**
```python
fig = pf.plot()
fig.write_html('reports/phase4_ensemble_5bar_tearsheet.html')
```

---

## Meta-Learner Analysis and Interpretation

### Model Weights Analysis

**For Logistic Regression:**
```python
def analyze_meta_weights(meta_model, horizon, feature_names):
    """
    Analyze learned weights for each class.
    """
    print(f"\n{horizon}-bar Meta-Learner Weights:\n")
    
    for class_idx, class_name in enumerate(['Short (-1)', 'Neutral (0)', 'Long (+1)']):
        print(f"{class_name}:")
        coefs = meta_model.coef_[class_idx]
        
        # Group by model
        nhits_weight = coefs[0:3].sum()
        tft_weight = coefs[3:6].sum()
        patchtst_weight = coefs[6:9].sum()
        
        total = nhits_weight + tft_weight + patchtst_weight
        
        print(f"  N-HiTS:   {nhits_weight/total:.2%}")
        print(f"  TFT:      {tft_weight/total:.2%}")
        print(f"  PatchTST: {patchtst_weight/total:.2%}")
        print()

analyze_meta_weights(meta_5, '5', feature_names)
```

**Example Output:**
```
5-bar Meta-Learner Weights:

Short (-1):
  N-HiTS:   28%
  TFT:      36%
  PatchTST: 36%

Neutral (0):
  N-HiTS:   35%
  TFT:      32%
  PatchTST: 33%

Long (+1):
  N-HiTS:   25%
  TFT:      40%
  PatchTST: 35%
```

**Interpretation:**
- For long signals, TFT gets highest weight (40%)
- N-HiTS contributes least to long signals (25%)
- Neutral class has balanced weights (all ~33%)

### Agreement vs Disagreement Analysis

**When do models agree vs disagree?**

```python
def analyze_agreement(stacking_df, horizon='5'):
    """
    Analyze when models agree or disagree and performance in each case.
    """
    # Predicted classes
    nhits_pred = stacking_df[[f'nhits_{horizon}_p{i}' for i in range(3)]].idxmax(axis=1).str[-1].astype(int)
    tft_pred = stacking_df[[f'tft_{horizon}_p{i}' for i in range(3)]].idxmax(axis=1).str[-1].astype(int)
    patchtst_pred = stacking_df[[f'patchtst_{horizon}_p{i}' for i in range(3)]].idxmax(axis=1).str[-1].astype(int)
    
    # Agreement count
    agreement = (nhits_pred == tft_pred).astype(int) + \
                (tft_pred == patchtst_pred).astype(int) + \
                (nhits_pred == patchtst_pred).astype(int)
    
    # 3 = all agree, 0 = all disagree
    stacking_df['agreement_score'] = agreement
    
    # Analyze accuracy by agreement level
    true_labels = stacking_df[f'label_{horizon}']
    
    for score in [0, 1, 2, 3]:
        mask = (agreement == score)
        if mask.sum() == 0:
            continue
        
        # Majority vote
        votes = np.vstack([nhits_pred[mask], tft_pred[mask], patchtst_pred[mask]]).T
        majority = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=votes)
        
        acc = accuracy_score(true_labels[mask], majority)
        print(f"Agreement {score}/3: n={mask.sum()}, Accuracy={acc:.3f}")

analyze_agreement(stacking_df, horizon='5')
```

**Example Output:**
```
Agreement 0/3: n=1234, Accuracy=0.412
Agreement 1/3: n=8567, Accuracy=0.523
Agreement 2/3: n=15432, Accuracy=0.591
Agreement 3/3: n=4321, Accuracy=0.672
```

**Insight:** When all models agree (3/3), accuracy is much higher (67%). Meta-learner can weight these predictions more heavily.

---

## Improvement Validation

### Compare Ensemble vs Single Models

```python
def compare_ensemble_to_singles(horizon='5'):
    """
    Compare ensemble to each single model on validation set.
    """
    # Load single model predictions
    nhits_probs = np.load(f'predictions/nhits_val_probs_{horizon}.npy')
    tft_probs = np.load(f'predictions/tft_val_probs_{horizon}.npy')
    patchtst_probs = np.load(f'predictions/patchtst_val_probs_{horizon}.npy')
    ensemble_probs = np.load(f'predictions/ensemble_val_probs_{horizon}.npy')
    
    # True labels
    labels = stacking_df[f'label_{horizon}'].values
    
    # Metrics
    models = {
        'N-HiTS': nhits_probs,
        'TFT': tft_probs,
        'PatchTST': patchtst_probs,
        'Ensemble': ensemble_probs,
    }
    
    results = []
    for name, probs in models.items():
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        
        # Backtest
        pred_signal = preds - 1  # {0,1,2} → {-1,0,+1}
        entries_long = (pred_signal == 1)
        entries_short = (pred_signal == -1)
        
        pf = vbt.Portfolio.from_signals(
            close=val_df['close'].values,
            entries=entries_long,
            exits=entries_short,
            freq='5T',
        )
        
        sharpe = pf.sharpe_ratio()
        dd = pf.max_drawdown()
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'F1': f1,
            'Sharpe': sharpe,
            'Max DD': dd,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{horizon}-bar Horizon Comparison:")
    print(results_df.to_string(index=False))
    
    return results_df

comparison_5bar = compare_ensemble_to_singles(horizon='5')
comparison_5bar.to_csv('reports/phase4_ensemble_comparison_5bar.csv', index=False)
```

**Expected Output:**
```
5-bar Horizon Comparison:
     Model  Accuracy    F1  Sharpe  Max DD
    N-HiTS     0.552  0.42   0.568   0.112
       TFT     0.571  0.44   0.619   0.095
  PatchTST     0.585  0.46   0.652   0.087
  Ensemble     0.591  0.47   0.694   0.081
```

**Success:** Ensemble beats all single models on all metrics.

---

## Phase 4 Deliverables

### Primary Outputs

1. **Trained Meta-Learners**
   - `models/meta_learner_1bar.pkl`
   - `models/meta_learner_5bar.pkl`
   - `models/meta_learner_20bar.pkl`
   - Saved as scikit-learn models (joblib)

2. **Ensemble Predictions (Validation Set)**
   - `predictions/ensemble_val_probs_1.npy`
   - `predictions/ensemble_val_probs_5.npy`
   - `predictions/ensemble_val_probs_20.npy`

3. **Performance Reports**
   - `reports/phase4_meta_weights_analysis.txt`
   - `reports/phase4_ensemble_comparison_1bar.csv`
   - `reports/phase4_ensemble_comparison_5bar.csv`
   - `reports/phase4_ensemble_comparison_20bar.csv`

4. **Backtest Tearsheets**
   - `reports/phase4_ensemble_5bar_tearsheet.html`

5. **Summary Report**
   - `reports/PHASE4_summary.md`
     - Improvement summary (vs single models)
     - Meta-learner weights interpretation
     - Agreement analysis
     - Recommendations for Phase 5

### Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Ensemble vs Best Single (Sharpe) | +0.05 | ___ |
| Ensemble vs Best Single (F1) | +0.01 | ___ |
| Meta-learner overfitting | None (meta-val ≈ meta-train) | ___ |
| Meta-learner interpretability | Clear weights per model | ___ |

---

## Implementation Checklist

### Pre-Phase Tasks
- [ ] Verify Phase 3 stacking dataset exists and is valid
- [ ] Set up meta-learner directories

### Data Preparation
- [ ] Load stacking dataset (`oos_predictions.parquet`)
- [ ] Verify all predictions are present (no NaN)
- [ ] Split stacking data (meta-train / meta-val)

### Baseline: Simple Averaging
- [ ] Implement simple averaging of model probabilities
- [ ] Compute metrics (Acc, F1, Sharpe) on meta-val
- [ ] Save as baseline to beat

### Train Meta-Learner (Logistic Regression)
- [ ] Train LR meta-learner for 1-bar horizon
- [ ] Train LR meta-learner for 5-bar horizon
- [ ] Train LR meta-learner for 20-bar horizon
- [ ] Evaluate each on meta-val set
- [ ] Save trained models (joblib)

### Optional: Try XGBoost Meta-Learner
- [ ] Train XGBoost for each horizon
- [ ] Compare to LR performance
- [ ] Choose best meta-learner type per horizon

### Generate Ensemble Predictions
- [ ] Apply meta-learners to full stacking dataset
- [ ] Generate ensemble probabilities for all horizons
- [ ] Save predictions to `predictions/ensemble_val_probs_*.npy`

### Backtesting
- [ ] Run vectorbt backtest on ensemble 5-bar predictions
- [ ] Compute Sharpe, DD, trade count, win rate
- [ ] Save tearsheet HTML

### Analysis
- [ ] Analyze meta-learner weights (which model gets most weight per class)
- [ ] Agreement analysis (performance when models agree vs disagree)
- [ ] Compare ensemble to each single model (table)
- [ ] Verify ensemble beats all singles on key metrics

### Reporting
- [ ] Create comparison tables (CSV)
- [ ] Write Phase 4 summary report
- [ ] Document meta-learner weights and interpretation

### Post-Phase Tasks
- [ ] Archive meta-training logs
- [ ] Prepare ensemble for Phase 5 (test set evaluation)

---

## Notes and Considerations

### Meta-Learner Complexity

**Keep it simple:**
- Logistic Regression is usually sufficient
- More complex models (XGBoost, NN) risk overfitting
- Stacking improvement is typically 5-10%, not 50%

**When to increase complexity:**
- LR not beating simple averaging
- Clear non-linear interactions in data
- Have additional meta-features (regime flags, etc.)

### Overfitting Risk

**Signs of overfitting:**
- Meta-train metrics much better than meta-val
- Ensemble performs worse on new data than singles

**Prevention:**
- Regularization (L2 penalty in LR)
- Keep meta-model simple
- Use out-of-sample predictions from CV (already done in Phase 3)

### Feature Selection for Meta-Learner

**Start minimal:** Raw probabilities only (9 features per horizon)

**Add if needed:**
- Confidence scores (max prob per model)
- Agreement indicators
- Regime flags (vol_regime, trend_regime)

**Don't add:**
- Original OHLCV or indicators (meta-learner should trust base models)
- Too many engineered features (overfitting risk)

### Interpreting Weights

**Logistic Regression Coefficients:**
- Positive coef → model's prediction for that class increases final prob
- Higher absolute value → more influence
- Can compare across models to see relative importance

**Example:**
- If TFT's coef for class 2 (+1) is 1.5 and N-HiTS's is 0.8:
  - TFT is trusted more for long signals

### When Stacking Doesn't Help

**Possible reasons:**
1. Base models too similar (high correlation)
   - Solution: Add more diverse models or drop redundant ones
2. One model dominates all others
   - Solution: Meta-learner might just copy best model (still okay)
3. Meta-learner overfitting
   - Solution: Simplify, increase regularization

**Fallback:**
- If stacking doesn't beat best single model → use best single model
- Document why and move to Phase 5

---

## Success Criteria Summary

Phase 4 is complete and successful when:

1. ✅ Meta-learners trained for all three horizons
2. ✅ Ensemble predictions generated on validation set
3. ✅ Ensemble Sharpe > best single model Sharpe by ≥ 0.05
4. ✅ Ensemble F1 ≥ best single model F1
5. ✅ Meta-learner weights are interpretable
6. ✅ Agreement analysis completed
7. ✅ No meta-learner overfitting (meta-train ≈ meta-val)
8. ✅ Comprehensive comparison report generated

**Proceed to Phase 5** (Full Integration & Final Test) only after all criteria are met.

---

**End of Phase 4 Specification**
