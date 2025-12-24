# Phase 3 – Cross-Validation and Out-of-Sample Predictions

## Overview

Phase 3 applies rigorous time-series cross-validation to generate truly out-of-sample predictions from all three base models. These predictions become the training data for the Phase 4 ensemble meta-learner.

**Phase 3 Goal:** Obtain unbiased performance estimates and collect out-of-sample predictions for stacking.

**Think of Phase 3 as:** "Stress-testing each model on data it's never seen, building the evidence that justifies combining them."

---

## Objectives

### Primary Goals
- Implement purged k-fold cross-validation with embargo periods
- Train each base model on multiple CV folds
- Generate out-of-sample predictions for entire validation dataset
- Measure per-fold performance stability across market regimes
- Build stacking training dataset (model predictions + true labels)
- Identify model strengths by regime, horizon, and market condition

### Success Criteria
- Complete 3-5 fold CV for all three models
- All predictions are strictly out-of-sample (no lookahead or leakage)
- Per-fold Sharpe variance < 0.5 (models are relatively stable across folds)
- Stacking dataset covers 100% of validation samples
- Clear documentation of which samples belong to which fold
- Regime-specific performance analysis completed

---

## Prerequisites

### Required Inputs
- Trained model architectures from Phase 2 (configs, not necessarily weights)
- Training and validation indices from Phase 1
- Full labeled dataset: `data/final/combined_final_labeled.parquet`
- GA-optimized barrier parameters

### Infrastructure Requirements
- Same as Phase 2 (GPU for training)
- Additional storage: ~5-10 GB for CV fold models and predictions

---

## Purged K-Fold Cross-Validation

### Why Purged CV for Financial Data

**Standard K-Fold Problems:**
- Random shuffling breaks temporal order
- Train samples can "see into future" of test samples via label lookahead
- Overlapping label horizons create leakage

**Purged CV Solution:**
1. **Split by time blocks** (not random)
2. **Purge** training samples whose labels overlap with test period
3. **Embargo** additional buffer zone around test folds
4. **Walk-forward** or blocked validation structure

### CV Fold Configuration

#### Recommended Setup: 5-Fold Purged CV

**Time-Based Splits:**
Assuming validation set spans years 2018-2021 (~3-4 years):

| Fold | Test Period | Approx Duration |
|------|-------------|-----------------|
| 1 | 2018 Q1-Q2 | 6 months |
| 2 | 2018 Q3-Q4 | 6 months |
| 3 | 2019 Q1-Q3 | 9 months |
| 4 | 2020 Q1-Q2 | 6 months |
| 5 | 2020 Q3 - 2021 | 15 months |

**Each fold:**
- Train: All validation data EXCEPT test period (with purging/embargo)
- Test: Designated test period
- Models trained from scratch each fold

**Alternative: 3-Fold for Speed**
- Larger test periods (~1 year each)
- Fewer total training runs (3× instead of 5×)
- Still sufficient for stable estimates

### Purging Logic

**For each CV fold boundary:**

```python
def purge_train_samples(train_df, test_start, test_end, max_horizon_bars=20, bar_duration='5T'):
    """
    Remove training samples whose label horizon overlaps with test period.
    
    Args:
        train_df: Training data for this fold
        test_start: Start timestamp of test fold
        test_end: End timestamp of test fold
        max_horizon_bars: Maximum forecast horizon (20 bars)
        bar_duration: Duration of one bar ('5T' for 5 minutes)
    
    Returns:
        Purged training DataFrame
    """
    # Convert bar horizon to time duration
    horizon_duration = pd.Timedelta(bar_duration) * max_horizon_bars
    
    # Calculate purge threshold: test_start minus horizon
    purge_threshold = test_start - horizon_duration
    
    # Remove samples after purge threshold
    purged_train = train_df[train_df['timestamp'] < purge_threshold]
    
    return purged_train
```

**Example:**
- Test fold: 2020-01-01 to 2020-06-30
- Max horizon: 20 bars × 5 min = 100 minutes
- Purge threshold: 2020-01-01 00:00 - 100 min = 2019-12-31 22:20
- Remove all training samples after 2019-12-31 22:20

### Embargo Period

**Add buffer zone after purging:**

```python
def add_embargo(train_df, test_start, embargo_duration='1D'):
    """
    Add embargo period around test fold.
    
    Args:
        train_df: Already-purged training data
        test_start: Start of test fold
        embargo_duration: Additional buffer ('1D' = 1 day)
    
    Returns:
        Embargoed training DataFrame
    """
    embargo_threshold = test_start - pd.Timedelta(embargo_duration)
    embargoed_train = train_df[train_df['timestamp'] < embargo_threshold]
    return embargoed_train
```

**Purpose:**
- Prevents subtle correlation between end-of-train and start-of-test
- Reduces regime persistence effects

**Typical embargo:** 1 day (288 bars for 5-min data)

### CV Fold Implementation

```python
def create_purged_cv_folds(val_df, n_folds=5, max_horizon_bars=20, embargo_duration='1D'):
    """
    Create purged K-fold splits for time-series CV.
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    val_df = val_df.sort_values('timestamp').reset_index(drop=True)
    total_samples = len(val_df)
    
    fold_size = total_samples // n_folds
    folds = []
    
    for fold_idx in range(n_folds):
        # Define test period for this fold
        test_start_idx = fold_idx * fold_size
        test_end_idx = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else total_samples
        
        test_indices = list(range(test_start_idx, test_end_idx))
        test_start_time = val_df.iloc[test_start_idx]['timestamp']
        test_end_time = val_df.iloc[test_end_idx - 1]['timestamp']
        
        # Training: all OTHER indices
        train_indices = list(range(0, test_start_idx)) + list(range(test_end_idx, total_samples))
        train_df = val_df.iloc[train_indices]
        
        # Apply purging
        train_df_purged = purge_train_samples(
            train_df, 
            test_start_time, 
            test_end_time, 
            max_horizon_bars=max_horizon_bars
        )
        
        # Apply embargo
        train_df_embargoed = add_embargo(
            train_df_purged,
            test_start_time,
            embargo_duration=embargo_duration
        )
        
        # Get final train indices
        final_train_indices = train_df_embargoed.index.tolist()
        
        folds.append({
            'fold': fold_idx,
            'train_indices': final_train_indices,
            'test_indices': test_indices,
            'test_start': test_start_time,
            'test_end': test_end_time,
            'n_train': len(final_train_indices),
            'n_test': len(test_indices),
        })
        
        print(f"Fold {fold_idx}: Train={len(final_train_indices)}, Test={len(test_indices)}")
    
    return folds
```

**Save fold configuration:**
```python
import json
with open('config/cv_folds.json', 'w') as f:
    json.dump(folds, f, indent=2, default=str)
```

---

## Model Training Per Fold

### Training Loop Structure

**For each model (N-HiTS, TFT, PatchTST) × each fold:**

```python
def train_model_on_fold(model_class, model_config, train_indices, test_indices, fold_idx):
    """
    Train model on one CV fold and generate predictions.
    
    Returns:
        test_predictions: Probabilities [N_test, 3] per horizon
        test_metrics: Dictionary of performance metrics
    """
    # Create datasets for this fold
    train_dataset = create_dataset(train_indices, model_type=model_class)
    test_dataset = create_dataset(test_indices, model_type=model_class)
    
    # Initialize fresh model
    model = model_class(model_config)
    
    # Train (same as Phase 2 but on fold data)
    trainer = create_trainer(max_epochs=50, patience=7)
    trainer.fit(model, train_dataset)
    
    # Generate test predictions
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            out = model(batch['X'])
            probs_1 = F.softmax(out['logits_1'], dim=-1)
            probs_5 = F.softmax(out['logits_5'], dim=-1)
            probs_20 = F.softmax(out['logits_20'], dim=-1)
            test_preds.append((probs_1, probs_5, probs_20))
    
    # Stack predictions
    test_probs_1 = torch.cat([p[0] for p in test_preds]).cpu().numpy()
    test_probs_5 = torch.cat([p[1] for p in test_preds]).cpu().numpy()
    test_probs_20 = torch.cat([p[2] for p in test_preds]).cpu().numpy()
    
    # Compute metrics
    test_labels = get_test_labels(test_indices)
    metrics = compute_metrics(test_probs_5, test_labels)
    
    # Save fold model (optional, for later inspection)
    save_model(model, f'models/{model_class.__name__}_fold{fold_idx}.pth')
    
    return {
        'probs_1': test_probs_1,
        'probs_5': test_probs_5,
        'probs_20': test_probs_20,
        'metrics': metrics,
    }
```

### Parallel CV Execution

**Option 1: Sequential (safer, less complexity)**
```python
all_results = {}

for model_name in ['NHiTS', 'TFT', 'PatchTST']:
    model_results = []
    for fold_idx, fold in enumerate(cv_folds):
        print(f"Training {model_name} on Fold {fold_idx}...")
        result = train_model_on_fold(
            model_class=get_model_class(model_name),
            model_config=get_model_config(model_name),
            train_indices=fold['train_indices'],
            test_indices=fold['test_indices'],
            fold_idx=fold_idx,
        )
        model_results.append(result)
    
    all_results[model_name] = model_results
```

**Option 2: Parallel (faster with multiple GPUs)**
```python
from joblib import Parallel, delayed

def train_wrapper(model_name, fold_idx, fold, gpu_id):
    set_gpu(gpu_id)
    return train_model_on_fold(...)

results = Parallel(n_jobs=3)(
    delayed(train_wrapper)(model_name, fold_idx, fold, gpu_id)
    for gpu_id, model_name in enumerate(['NHiTS', 'TFT', 'PatchTST'])
    for fold_idx, fold in enumerate(cv_folds)
)
```

---

## Assembling Out-of-Sample Predictions

### Objective

**Goal:** Create a single prediction matrix for entire validation set where every prediction is out-of-sample.

**Structure:**
```
For each validation sample i:
    - Determine which fold it was in (as test sample)
    - Use that fold's model predictions
    - Store: [nhits_probs, tft_probs, patchtst_probs, true_label]
```

### Aggregation Process

```python
def aggregate_cv_predictions(cv_results, val_df, n_folds):
    """
    Combine per-fold predictions into single out-of-sample prediction matrix.
    
    Returns:
        oos_predictions: Dict with keys for each model/horizon
        oos_labels: True labels aligned with predictions
    """
    n_samples = len(val_df)
    n_models = 3  # N-HiTS, TFT, PatchTST
    n_horizons = 3  # 1, 5, 20 bars
    n_classes = 3  # -1, 0, +1
    
    # Initialize storage
    oos_preds = {
        'nhits_1': np.zeros((n_samples, n_classes)),
        'nhits_5': np.zeros((n_samples, n_classes)),
        'nhits_20': np.zeros((n_samples, n_classes)),
        'tft_1': np.zeros((n_samples, n_classes)),
        'tft_5': np.zeros((n_samples, n_classes)),
        'tft_20': np.zeros((n_samples, n_classes)),
        'patchtst_1': np.zeros((n_samples, n_classes)),
        'patchtst_5': np.zeros((n_samples, n_classes)),
        'patchtst_20': np.zeros((n_samples, n_classes)),
    }
    
    oos_labels = {
        'label_1': np.zeros(n_samples, dtype=int),
        'label_5': np.zeros(n_samples, dtype=int),
        'label_20': np.zeros(n_samples, dtype=int),
    }
    
    # Fill in predictions from each fold
    for fold_idx in range(n_folds):
        test_indices = cv_folds[fold_idx]['test_indices']
        
        # N-HiTS predictions
        oos_preds['nhits_1'][test_indices] = cv_results['NHiTS'][fold_idx]['probs_1']
        oos_preds['nhits_5'][test_indices] = cv_results['NHiTS'][fold_idx]['probs_5']
        oos_preds['nhits_20'][test_indices] = cv_results['NHiTS'][fold_idx]['probs_20']
        
        # TFT predictions
        oos_preds['tft_1'][test_indices] = cv_results['TFT'][fold_idx]['probs_1']
        oos_preds['tft_5'][test_indices] = cv_results['TFT'][fold_idx]['probs_5']
        oos_preds['tft_20'][test_indices] = cv_results['TFT'][fold_idx]['probs_20']
        
        # PatchTST predictions
        oos_preds['patchtst_1'][test_indices] = cv_results['PatchTST'][fold_idx]['probs_1']
        oos_preds['patchtst_5'][test_indices] = cv_results['PatchTST'][fold_idx]['probs_5']
        oos_preds['patchtst_20'][test_indices] = cv_results['PatchTST'][fold_idx]['probs_20']
        
        # True labels
        oos_labels['label_1'][test_indices] = val_df.iloc[test_indices]['label_1'].values + 1
        oos_labels['label_5'][test_indices] = val_df.iloc[test_indices]['label_5'].values + 1
        oos_labels['label_20'][test_indices] = val_df.iloc[test_indices]['label_20'].values + 1
    
    return oos_preds, oos_labels
```

### Save Stacking Dataset

```python
# Create stacking input DataFrame
stacking_data = pd.DataFrame({
    # Sample metadata
    'sample_idx': np.arange(len(val_df)),
    'timestamp': val_df['timestamp'].values,
    'symbol': val_df['symbol'].values,
    
    # Model predictions (9 columns of probabilities per horizon)
    'nhits_1_p0': oos_preds['nhits_1'][:, 0],
    'nhits_1_p1': oos_preds['nhits_1'][:, 1],
    'nhits_1_p2': oos_preds['nhits_1'][:, 2],
    # ... repeat for all model/horizon combinations
    
    # True labels
    'label_1': oos_labels['label_1'],
    'label_5': oos_labels['label_5'],
    'label_20': oos_labels['label_20'],
})

stacking_data.to_parquet('data/stacking/oos_predictions.parquet')
print(f"Stacking dataset saved: {len(stacking_data)} samples")
```

---

## Per-Fold Performance Analysis

### Metrics to Track

**For each (model, fold, horizon) combination:**

1. **Classification Metrics**
   - Accuracy
   - Macro F1
   - Per-class Precision/Recall
   - Confusion Matrix

2. **Trading Metrics**
   - Sharpe Ratio
   - Max Drawdown
   - Number of Trades
   - Win Rate
   - Profit Factor

### Stability Analysis

**Goal:** Measure consistency across folds.

**Coefficient of Variation (CV):**
```python
def compute_stability(fold_metrics, metric_name='sharpe'):
    """
    Measure performance stability across folds.
    
    Lower CV = more stable model.
    """
    values = [fold[metric_name] for fold in fold_metrics]
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val != 0 else float('inf')
    return cv

# Example
nhits_sharpe_cv = compute_stability(all_results['NHiTS'], 'sharpe')
print(f"N-HiTS Sharpe CV: {nhits_sharpe_cv:.3f}")  # Target: < 0.5
```

**Report:**
| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | CV |
|-------|--------|--------|--------|--------|--------|--------|------|-----|-----|
| N-HiTS | Sharpe | 0.52 | 0.48 | 0.61 | 0.44 | 0.58 | 0.53 | 0.07 | 0.13 |
| TFT | Sharpe | 0.59 | 0.55 | 0.68 | 0.50 | 0.64 | 0.59 | 0.07 | 0.12 |
| PatchTST | Sharpe | 0.62 | 0.58 | 0.71 | 0.54 | 0.67 | 0.62 | 0.07 | 0.11 |

**Interpretation:**
- CV < 0.2: Good stability
- CV 0.2-0.5: Acceptable
- CV > 0.5: Model is regime-sensitive, may need regime adaptation

---

## Regime-Specific Performance

### Objective

**Question:** Do models perform differently in different market conditions?

**Regimes to analyze:**
1. **Volatility:** Low / Medium / High (from Phase 1 vol_regime)
2. **Trend:** Uptrend / Downtrend / Sideways
3. **Time period:** 2018 vs 2019 vs 2020 vs 2021

### Analysis Process

```python
def analyze_performance_by_regime(oos_preds, oos_labels, val_df):
    """
    Break down model performance by regime.
    """
    results = []
    
    for regime_type in ['vol_regime', 'trend_regime']:
        regime_values = val_df[regime_type].values
        
        for model_name in ['nhits', 'tft', 'patchtst']:
            for horizon in ['5']:  # Focus on 5-bar for example
                preds = oos_preds[f'{model_name}_{horizon}']
                labels = oos_labels[f'label_{horizon}']
                
                for regime_val in np.unique(regime_values):
                    mask = (regime_values == regime_val)
                    
                    if mask.sum() < 100:  # Skip small regimes
                        continue
                    
                    regime_preds = preds[mask]
                    regime_labels = labels[mask]
                    
                    # Compute metrics
                    acc = accuracy_score(regime_labels, np.argmax(regime_preds, axis=1))
                    f1 = f1_score(regime_labels, np.argmax(regime_preds, axis=1), average='macro')
                    
                    # Backtest
                    regime_closes = val_df.loc[mask, 'close'].values
                    sharpe, dd = quick_backtest(regime_preds, regime_labels, regime_closes)
                    
                    results.append({
                        'model': model_name,
                        'horizon': horizon,
                        'regime_type': regime_type,
                        'regime_value': regime_val,
                        'n_samples': mask.sum(),
                        'accuracy': acc,
                        'f1': f1,
                        'sharpe': sharpe,
                        'max_dd': dd,
                    })
    
    return pd.DataFrame(results)

regime_analysis = analyze_performance_by_regime(oos_preds, oos_labels, val_df)
regime_analysis.to_csv('reports/phase3_regime_analysis.csv')
```

**Example Insights:**
- "TFT excels in high volatility (Sharpe 0.72 vs 0.54 in low vol)"
- "PatchTST performs better in trending markets (F1 0.48 vs 0.38 in sideways)"
- "N-HiTS is most consistent across regimes (low variance)"

**Use in Phase 4:**
- Meta-learner can use regime features to weight models dynamically
- E.g., "In high vol, trust TFT more; in low vol, trust N-HiTS"

---

## Prediction Correlation Analysis

### Objective

**Goal:** Measure how similar/different model predictions are.

**Why it matters:**
- High correlation → models see markets similarly (less ensemble value)
- Low correlation → models are complementary (good for ensembling)

### Correlation Matrix

```python
def compute_prediction_correlations(oos_preds, horizon='5'):
    """
    Measure correlation between model predictions for a given horizon.
    """
    # Extract predicted classes
    nhits_class = np.argmax(oos_preds[f'nhits_{horizon}'], axis=1)
    tft_class = np.argmax(oos_preds[f'tft_{horizon}'], axis=1)
    patchtst_class = np.argmax(oos_preds[f'patchtst_{horizon}'], axis=1)
    
    # Stack into matrix
    pred_matrix = np.stack([nhits_class, tft_class, patchtst_class], axis=1)
    
    # Compute correlation
    corr = np.corrcoef(pred_matrix.T)
    
    # Create DataFrame
    corr_df = pd.DataFrame(
        corr,
        columns=['N-HiTS', 'TFT', 'PatchTST'],
        index=['N-HiTS', 'TFT', 'PatchTST']
    )
    
    return corr_df

corr_5bar = compute_prediction_correlations(oos_preds, horizon='5')
print("5-bar Horizon Prediction Correlations:")
print(corr_5bar)
```

**Example output:**
```
           N-HiTS    TFT  PatchTST
N-HiTS      1.00   0.68      0.62
TFT         0.68   1.00      0.71
PatchTST    0.62   0.71      1.00
```

**Interpretation:**
- Correlations 0.6-0.7: Moderate agreement, good for ensembling
- Correlation < 0.5: Very complementary (excellent)
- Correlation > 0.8: Models are too similar (diminishing returns)

**Visualize:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(corr_5bar, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Model Prediction Correlations (5-bar Horizon)')
plt.savefig('reports/phase3_pred_correlation.png')
```

---

## Phase 3 Deliverables

### Primary Outputs

1. **CV Fold Configuration**
   - `config/cv_folds.json`
   - Documents train/test indices per fold, purge/embargo settings

2. **Per-Fold Models (optional)**
   - `models/{model}_fold{i}.pth` (if saving for later inspection)
   - Can be deleted after Phase 3 to save space

3. **Out-of-Sample Predictions (Stacking Dataset)**
   - `data/stacking/oos_predictions.parquet`
   - Contains all model predictions + true labels
   - **Critical input for Phase 4**

4. **Performance Reports**
   - `reports/phase3_cv_metrics.csv`
     - Per-fold metrics for all models/horizons
   - `reports/phase3_stability_analysis.csv`
     - CV statistics per model
   - `reports/phase3_regime_analysis.csv`
     - Performance breakdown by market regime

5. **Correlation Analysis**
   - `reports/phase3_pred_correlation.png` (heatmaps per horizon)
   - `reports/phase3_correlation_summary.txt`

6. **Comprehensive Summary**
   - `reports/PHASE3_summary.md`
     - Key findings
     - Model rankings per horizon
     - Regime insights
     - Recommendations for Phase 4

### Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| CV folds completed | 5 | ___ |
| % of val set with OOS predictions | 100% | ___ |
| Mean fold Sharpe (best model) | >0.5 | ___ |
| Sharpe CV (best model) | <0.5 | ___ |
| Pred correlation (avg) | 0.5-0.7 | ___ |

---

## Implementation Checklist

### Pre-Phase Tasks
- [ ] Verify Phase 2 model configs and training procedures are solid
- [ ] Load full validation dataset from Phase 1
- [ ] Set up CV fold directories and logging

### Fold Creation
- [ ] Implement purged k-fold splitting function
- [ ] Define purge threshold (max_horizon_bars)
- [ ] Define embargo period (e.g., 1 day)
- [ ] Generate 3-5 CV folds on validation set
- [ ] Save fold configuration to JSON
- [ ] Verify no train/test overlap per fold

### Model Training Per Fold
- [ ] For each model (N-HiTS, TFT, PatchTST):
  - [ ] Train on Fold 1 train set
  - [ ] Generate predictions on Fold 1 test set
  - [ ] Compute and log fold metrics
  - [ ] Repeat for Folds 2-5
- [ ] Save per-fold models (optional) or discard to save space
- [ ] Log training time per fold

### Aggregation
- [ ] Collect predictions from all folds
- [ ] Verify coverage: every val sample has predictions
- [ ] Check for NaN or invalid predictions
- [ ] Assemble stacking dataset (9 pred columns + 3 label columns)
- [ ] Save to `data/stacking/oos_predictions.parquet`

### Analysis
- [ ] Compute per-fold metrics (Acc, F1, Sharpe, DD)
- [ ] Calculate stability metrics (mean, std, CV per model)
- [ ] Generate metrics summary table
- [ ] Perform regime-specific analysis
- [ ] Compute prediction correlation matrix
- [ ] Create correlation heatmaps
- [ ] Identify best models per horizon and regime

### Reporting
- [ ] Write Phase 3 summary report
- [ ] Include key findings and insights
- [ ] Document any issues or anomalies encountered
- [ ] Recommend meta-learner features based on regime analysis

### Post-Phase Tasks
- [ ] Archive or delete per-fold models if space constrained
- [ ] Verify stacking dataset is ready for Phase 4
- [ ] Update project documentation

---

## Notes and Considerations

### Computational Cost

**Training Load:**
- 3 models × 5 folds = 15 training runs
- Each model: 2-12 hours (N-HiTS fast, TFT slow)
- **Total Phase 3 time: 30-150 hours**

**Mitigation:**
- Use 3 folds instead of 5 (faster, still valid)
- Parallelize across multiple GPUs
- Reduce max_epochs per fold (use early stopping aggressively)

### Overfitting to Folds

**Concern:** If you tweak models based on CV results, you overfit to validation set.

**Solution:**
- Treat Phase 3 CV as "final validation"
- Don't go back and retrain Phase 2 models based on these results
- Any hyperparameter tuning should happen in Phase 2 (on train set)

### Handling Fold Imbalance

**Issue:** Different folds may have different label distributions (regime-dependent).

**Examples:**
- Fold covering 2020 COVID period: high volatility, many ±1 labels
- Fold covering 2019: low volatility, many 0 labels

**Impact:**
- Metrics vary across folds
- Models may perform very differently

**Handling:**
- Report per-fold results separately
- Use regime analysis to explain variance
- Meta-learner (Phase 4) can learn these nuances

### Walk-Forward as Alternative

**Purged K-Fold (current):**
- Pros: More data per fold, tests different periods
- Cons: Training on "past + future" of test fold (though purged)

**Walk-Forward (alternative):**
- Fold 1: Train on 2018, test on 2019 Q1
- Fold 2: Train on 2018-2019 Q1, test on 2019 Q2
- Fold 3: Train on 2018-2019 Q2, test on 2019 Q3
- ... (expanding window)

**Pros:**
- More realistic (always training on past only)
- Mimics live trading deployment

**Cons:**
- Early folds have less training data
- Computationally heavier (each fold trains on more data)

**Recommendation:**
- Start with purged k-fold (faster, industry standard)
- Use walk-forward in Phase 6 (refinement) if needed

---

## Success Criteria Summary

Phase 3 is complete and successful when:

1. ✅ CV folds created with proper purging and embargo
2. ✅ All three models trained on all folds (9-15 training runs)
3. ✅ Out-of-sample predictions generated for 100% of validation set
4. ✅ Stacking dataset saved and validated
5. ✅ Per-fold performance metrics logged
6. ✅ Stability analysis shows CV < 0.5 for Sharpe per model
7. ✅ Regime analysis completed
8. ✅ Prediction correlation analysis completed
9. ✅ Comprehensive Phase 3 report generated

**Proceed to Phase 4** (Meta-Learner Training) only after all criteria are met.

---

**End of Phase 3 Specification**
