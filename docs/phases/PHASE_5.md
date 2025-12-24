# Phase 5 – Full Integration and Final Test Set Evaluation

## Overview

Phase 5 integrates all components into a unified pipeline and evaluates the complete ensemble system on the final hold-out test set. This is the "moment of truth" – the true out-of-sample performance measure that has never been seen by any model or meta-learner.

**Phase 5 Goal:** Validate that the complete system (base models + ensemble) generalizes to truly unseen data and meets performance targets.

**Think of Phase 5 as:** "Taking the final exam after all the practice tests."

---

## Objectives

### Primary Goals
- Build complete inference pipeline (data → features → base models → meta-learner → final predictions)
- Run pipeline on final test set (completely untouched until now)
- Generate true out-of-sample performance metrics
- Verify no data leakage or lookahead bias in pipeline
- Document final system performance
- Identify any failure modes or unexpected behavior

### Success Criteria
- Test set Sharpe > 0.4 (for 5-bar horizon)
- Test set F1 > 0.35 (for 5-bar horizon)
- Performance gap between validation and test < 15%
- No evidence of data leakage
- Pipeline runs end-to-end without errors
- Final predictions aligned with timestamps (no misalignment)

---

## Prerequisites

### Required Inputs
- Trained base models from Phase 2:
  - `models/nhits_final.pth`
  - `models/tft_final.pth`
  - `models/patchtst_final.pth`
- Trained meta-learners from Phase 4:
  - `models/meta_learner_1bar.pkl`
  - `models/meta_learner_5bar.pkl`
  - `models/meta_learner_20bar.pkl`
- Feature scalers:
  - `models/nhits_scaler.pkl`
- Test set indices:
  - `config/splits/test_indices.npy`
- Full labeled dataset:
  - `data/final/combined_final_labeled.parquet`

---

## Complete Inference Pipeline

### Pipeline Architecture

```
Input: Raw OHLCV bar at time t
    ↓
Step 1: Feature Engineering
    - Compute indicators (RSI, ATR, MAs, etc.)
    - Compute regime flags
    - Create rolling window (last L bars)
    ↓
Step 2: Feature Scaling
    - Apply StandardScaler (fitted on train set)
    ↓
Step 3: Base Model Predictions
    - N-HiTS → probs_nhits [3]
    - TFT → probs_tft [3]
    - PatchTST → probs_patchtst [3]
    ↓
Step 4: Stack Features
    - Concatenate: [probs_nhits, probs_tft, probs_patchtst] → [9]
    ↓
Step 5: Meta-Learner Prediction
    - Apply trained meta-learner per horizon
    - Output: final_probs [3] per horizon
    ↓
Output: Final ensemble prediction {−1, 0, +1} per horizon
```

### Implementation

#### Pipeline Class

```python
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class EnsembleForecastPipeline:
    def __init__(self, model_dir='models/', scaler_path='models/nhits_scaler.pkl'):
        """
        Complete inference pipeline from OHLCV to ensemble prediction.
        """
        # Load scalers
        self.scaler = joblib.load(scaler_path)
        
        # Load base models
        self.nhits = self.load_nhits(f'{model_dir}/nhits_final.pth')
        self.tft = self.load_tft(f'{model_dir}/tft_final.pth')
        self.patchtst = self.load_patchtst(f'{model_dir}/patchtst_final.pth')
        
        # Load meta-learners
        self.meta_1 = joblib.load(f'{model_dir}/meta_learner_1bar.pkl')
        self.meta_5 = joblib.load(f'{model_dir}/meta_learner_5bar.pkl')
        self.meta_20 = joblib.load(f'{model_dir}/meta_learner_20bar.pkl')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nhits.to(self.device).eval()
        self.tft.to(self.device).eval()
        self.patchtst.to(self.device).eval()
    
    def load_nhits(self, path):
        checkpoint = torch.load(path)
        model = NHiTSClassifier(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def load_tft(self, path):
        checkpoint = torch.load(path)
        model = TFTClassifier(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def load_patchtst(self, path):
        checkpoint = torch.load(path)
        model = PatchTSTClassifier(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def compute_features(self, df, lookback=128):
        """
        Compute all required features for a given DataFrame.
        Assumes df has OHLCV columns.
        """
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Add more indicators as per Phase 1...
        # (full feature engineering from Phase 1)
        
        return df
    
    def create_sequence(self, df, idx, lookback=128, feature_cols=None):
        """
        Create a sequence window for a given index.
        """
        if idx < lookback:
            raise ValueError(f"Index {idx} too early, need at least {lookback} bars")
        
        seq = df.iloc[idx - lookback : idx][feature_cols].values
        return seq
    
    def predict(self, df, idx, lookback=128, feature_cols=None):
        """
        Make ensemble prediction for a single bar.
        
        Args:
            df: DataFrame with all features computed
            idx: Index of bar to predict
            lookback: Sequence length
            feature_cols: List of feature column names
        
        Returns:
            predictions: Dict with keys '1', '5', '20'
                         Each contains: {'probs': [3], 'class': int}
        """
        # Create sequence
        seq = self.create_sequence(df, idx, lookback, feature_cols)
        
        # Scale features
        seq_scaled = self.scaler.transform(seq)
        
        # Convert to tensor
        X = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)  # [1, L, F]
        
        # Base model predictions
        with torch.no_grad():
            # N-HiTS
            nhits_out = self.nhits(X)
            nhits_probs_1 = torch.softmax(nhits_out['logits_1'], dim=-1).cpu().numpy()[0]
            nhits_probs_5 = torch.softmax(nhits_out['logits_5'], dim=-1).cpu().numpy()[0]
            nhits_probs_20 = torch.softmax(nhits_out['logits_20'], dim=-1).cpu().numpy()[0]
            
            # TFT
            tft_out = self.tft(X)
            tft_probs_1 = torch.softmax(tft_out['logits_1'], dim=-1).cpu().numpy()[0]
            tft_probs_5 = torch.softmax(tft_out['logits_5'], dim=-1).cpu().numpy()[0]
            tft_probs_20 = torch.softmax(tft_out['logits_20'], dim=-1).cpu().numpy()[0]
            
            # PatchTST
            patchtst_out = self.patchtst(X)
            patchtst_probs_1 = torch.softmax(patchtst_out['logits_1'], dim=-1).cpu().numpy()[0]
            patchtst_probs_5 = torch.softmax(patchtst_out['logits_5'], dim=-1).cpu().numpy()[0]
            patchtst_probs_20 = torch.softmax(patchtst_out['logits_20'], dim=-1).cpu().numpy()[0]
        
        # Stack for meta-learner
        meta_input_1 = np.concatenate([nhits_probs_1, tft_probs_1, patchtst_probs_1]).reshape(1, -1)
        meta_input_5 = np.concatenate([nhits_probs_5, tft_probs_5, patchtst_probs_5]).reshape(1, -1)
        meta_input_20 = np.concatenate([nhits_probs_20, tft_probs_20, patchtst_probs_20]).reshape(1, -1)
        
        # Meta-learner predictions
        ensemble_probs_1 = self.meta_1.predict_proba(meta_input_1)[0]
        ensemble_probs_5 = self.meta_5.predict_proba(meta_input_5)[0]
        ensemble_probs_20 = self.meta_20.predict_proba(meta_input_20)[0]
        
        # Final predictions
        predictions = {
            '1': {
                'probs': ensemble_probs_1,
                'class': int(np.argmax(ensemble_probs_1)),
                'signal': int(np.argmax(ensemble_probs_1)) - 1,  # {0,1,2} → {-1,0,+1}
            },
            '5': {
                'probs': ensemble_probs_5,
                'class': int(np.argmax(ensemble_probs_5)),
                'signal': int(np.argmax(ensemble_probs_5)) - 1,
            },
            '20': {
                'probs': ensemble_probs_20,
                'class': int(np.argmax(ensemble_probs_20)),
                'signal': int(np.argmax(ensemble_probs_20)) - 1,
            },
        }
        
        return predictions
    
    def predict_batch(self, df, indices, lookback=128, feature_cols=None):
        """
        Make predictions for a batch of indices.
        """
        all_predictions = {
            '1': {'probs': [], 'signals': []},
            '5': {'probs': [], 'signals': []},
            '20': {'probs': [], 'signals': []},
        }
        
        for idx in indices:
            try:
                pred = self.predict(df, idx, lookback, feature_cols)
                for h in ['1', '5', '20']:
                    all_predictions[h]['probs'].append(pred[h]['probs'])
                    all_predictions[h]['signals'].append(pred[h]['signal'])
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                # Append neutral prediction
                for h in ['1', '5', '20']:
                    all_predictions[h]['probs'].append(np.array([0.33, 0.34, 0.33]))
                    all_predictions[h]['signals'].append(0)
        
        # Convert to arrays
        for h in ['1', '5', '20']:
            all_predictions[h]['probs'] = np.array(all_predictions[h]['probs'])
            all_predictions[h]['signals'] = np.array(all_predictions[h]['signals'])
        
        return all_predictions
```

---

## Test Set Evaluation

### Load Test Data

```python
# Load full dataset
full_df = pd.read_parquet('data/final/combined_final_labeled.parquet')

# Load test indices
test_indices = np.load('config/splits/test_indices.npy')

# Extract test set
test_df = full_df.iloc[test_indices].reset_index(drop=True)

print(f"Test set size: {len(test_df)} samples")
print(f"Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
```

### Run Pipeline on Test Set

```python
# Initialize pipeline
pipeline = EnsembleForecastPipeline()

# Define feature columns (same as used in training)
feature_cols = [col for col in test_df.columns if col.startswith('feature_')]

# Get valid indices (need lookback bars)
lookback = 128
valid_indices = list(range(lookback, len(test_df)))

print(f"Generating predictions for {len(valid_indices)} test samples...")

# Generate predictions
test_predictions = pipeline.predict_batch(
    test_df,
    valid_indices,
    lookback=lookback,
    feature_cols=feature_cols
)

print("Predictions complete.")

# Save predictions
for h in ['1', '5', '20']:
    np.save(f'predictions/test_ensemble_probs_{h}.npy', test_predictions[h]['probs'])
    np.save(f'predictions/test_ensemble_signals_{h}.npy', test_predictions[h]['signals'])
```

### Sanity Check: No Lookahead

**Critical verification:**

```python
def verify_no_lookahead(pipeline, test_df, sample_idx=1000):
    """
    Verify that prediction at time t only uses data up to t.
    """
    # Make prediction at sample_idx
    pred_t = pipeline.predict(test_df, sample_idx, lookback=128, feature_cols=feature_cols)
    
    # Modify future data (should not affect prediction)
    test_df_modified = test_df.copy()
    test_df_modified.iloc[sample_idx+1:sample_idx+100, :] = 999  # Corrupt future
    
    # Remake prediction
    pred_t_modified = pipeline.predict(test_df_modified, sample_idx, lookback=128, feature_cols=feature_cols)
    
    # Compare
    for h in ['1', '5', '20']:
        assert np.allclose(pred_t[h]['probs'], pred_t_modified[h]['probs']), \
            f"Lookahead detected! Predictions changed when future data modified."
    
    print("✓ No lookahead bias detected")

verify_no_lookahead(pipeline, test_df)
```

---

## Test Set Performance Metrics

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_classification(predictions, true_labels, horizon='5'):
    """
    Compute classification metrics on test set.
    """
    pred_signals = predictions[horizon]['signals']
    pred_classes = pred_signals + 1  # {-1,0,+1} → {0,1,2}
    
    # Align with valid indices (skip first 'lookback' bars)
    true_labels = true_labels[lookback:]
    
    # Metrics
    acc = accuracy_score(true_labels, pred_classes)
    f1 = f1_score(true_labels, pred_classes, average='macro')
    
    # Per-class metrics
    report = classification_report(true_labels, pred_classes, target_names=['Short', 'Neutral', 'Long'])
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_classes)
    
    print(f"\n{horizon}-bar Test Set Classification Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print("\nPer-Class Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Short', 'Neutral', 'Long'],
                yticklabels=['Short', 'Neutral', 'Long'])
    plt.title(f'{horizon}-bar Confusion Matrix (Test Set)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'reports/phase5_confusion_matrix_{horizon}bar.png')
    
    return {'accuracy': acc, 'f1': f1, 'confusion_matrix': cm}

# Evaluate all horizons
test_labels_1 = test_df['label_1'].values + 1
test_labels_5 = test_df['label_5'].values + 1
test_labels_20 = test_df['label_20'].values + 1

metrics_1 = evaluate_classification(test_predictions, test_labels_1, horizon='1')
metrics_5 = evaluate_classification(test_predictions, test_labels_5, horizon='5')
metrics_20 = evaluate_classification(test_predictions, test_labels_20, horizon='20')
```

### Trading Metrics

```python
import vectorbt as vbt

def evaluate_trading(predictions, test_df, horizon='5'):
    """
    Backtest ensemble predictions on test set.
    """
    # Get signals (already aligned)
    signals = predictions[horizon]['signals']  # {-1, 0, +1}
    
    # Get corresponding prices (skip first 'lookback' bars)
    prices = test_df['close'].iloc[lookback:].values
    timestamps = test_df['timestamp'].iloc[lookback:].values
    
    # Create entry/exit signals
    entries_long = (signals == 1)
    entries_short = (signals == -1)
    
    # Vectorbt portfolio
    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries_long,
        exits=entries_short,
        freq='5T',  # 5-minute bars
        init_cash=100000,
    )
    
    # Metrics
    sharpe = pf.sharpe_ratio()
    max_dd = pf.max_drawdown()
    total_return = pf.total_return()
    n_trades = pf.trades.count()
    win_rate = pf.trades.win_rate()
    profit_factor = pf.trades.profit_factor()
    
    print(f"\n{horizon}-bar Test Set Trading Metrics:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Max Drawdown: {max_dd:.3%}")
    print(f"  Total Return: {total_return:.3%}")
    print(f"  Number of Trades: {n_trades}")
    print(f"  Win Rate: {win_rate:.3%}")
    print(f"  Profit Factor: {profit_factor:.3f}")
    
    # Generate tearsheet
    fig = pf.plot()
    fig.write_html(f'reports/phase5_test_tearsheet_{horizon}bar.html')
    
    return {
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_return': total_return,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
    }

# Evaluate all horizons
trading_1 = evaluate_trading(test_predictions, test_df, horizon='1')
trading_5 = evaluate_trading(test_predictions, test_df, horizon='5')
trading_20 = evaluate_trading(test_predictions, test_df, horizon='20')
```

---

## Validation vs Test Comparison

### Performance Degradation Check

```python
def compare_val_vs_test(val_metrics, test_metrics, horizon='5'):
    """
    Compare validation and test performance to check generalization.
    """
    comparison = {
        'Metric': ['Accuracy', 'F1', 'Sharpe', 'Max DD'],
        'Validation': [
            val_metrics['accuracy'],
            val_metrics['f1'],
            val_metrics['sharpe'],
            val_metrics['max_dd'],
        ],
        'Test': [
            test_metrics['accuracy'],
            test_metrics['f1'],
            test_metrics['sharpe'],
            test_metrics['max_dd'],
        ],
    }
    
    df_comp = pd.DataFrame(comparison)
    df_comp['Difference'] = df_comp['Test'] - df_comp['Validation']
    df_comp['% Change'] = (df_comp['Difference'] / df_comp['Validation']) * 100
    
    print(f"\n{horizon}-bar: Validation vs Test Comparison")
    print(df_comp.to_string(index=False))
    
    # Alert if large degradation
    if abs(df_comp.loc[df_comp['Metric'] == 'Sharpe', '% Change'].values[0]) > 15:
        print("⚠️  WARNING: Sharpe degraded >15% from validation to test!")
    
    return df_comp

# Load validation metrics from Phase 4
val_metrics_5 = {
    'accuracy': 0.591,
    'f1': 0.47,
    'sharpe': 0.694,
    'max_dd': 0.081,
}

comparison_5 = compare_val_vs_test(val_metrics_5, {**metrics_5, **trading_5}, horizon='5')
comparison_5.to_csv('reports/phase5_val_vs_test_5bar.csv', index=False)
```

**Expected Output:**
```
5-bar: Validation vs Test Comparison
    Metric  Validation    Test  Difference  % Change
  Accuracy       0.591   0.578      -0.013     -2.20
        F1       0.470   0.452      -0.018     -3.83
    Sharpe       0.694   0.671      -0.023     -3.31
    Max DD       0.081   0.089       0.008      9.88
```

**Interpretation:**
- Small degradation (<5%) is expected and acceptable
- Large degradation (>15%) suggests overfitting in validation phase

---

## Per-Symbol Analysis

### MES vs MGC Performance

```python
def analyze_per_symbol(predictions, test_df, horizon='5'):
    """
    Break down performance by symbol (MES vs MGC).
    """
    signals = predictions[horizon]['signals']
    labels = (test_df['label_' + horizon].iloc[lookback:].values + 1) - 1  # Back to {-1,0,+1}
    symbols = test_df['symbol'].iloc[lookback:].values
    prices = test_df['close'].iloc[lookback:].values
    
    results = []
    for sym in ['MES', 'MGC']:
        mask = (symbols == sym)
        
        sym_signals = signals[mask]
        sym_labels = labels[mask]
        sym_prices = prices[mask]
        
        # Classification
        acc = accuracy_score(sym_labels + 1, sym_signals + 1)
        f1 = f1_score(sym_labels + 1, sym_signals + 1, average='macro')
        
        # Trading
        entries_long = (sym_signals == 1)
        entries_short = (sym_signals == -1)
        
        pf = vbt.Portfolio.from_signals(
            close=sym_prices,
            entries=entries_long,
            exits=entries_short,
            freq='5T',
        )
        
        sharpe = pf.sharpe_ratio()
        max_dd = pf.max_drawdown()
        
        results.append({
            'Symbol': sym,
            'N_samples': mask.sum(),
            'Accuracy': acc,
            'F1': f1,
            'Sharpe': sharpe,
            'Max DD': max_dd,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{horizon}-bar Per-Symbol Analysis:")
    print(results_df.to_string(index=False))
    
    return results_df

symbol_analysis_5 = analyze_per_symbol(test_predictions, test_df, horizon='5')
symbol_analysis_5.to_csv('reports/phase5_per_symbol_5bar.csv', index=False)
```

---

## Phase 5 Deliverables

### Primary Outputs

1. **Integrated Pipeline Code**
   - `src/ensemble_pipeline.py` (EnsembleForecastPipeline class)
   - Fully functional end-to-end inference

2. **Test Set Predictions**
   - `predictions/test_ensemble_probs_1.npy`
   - `predictions/test_ensemble_probs_5.npy`
   - `predictions/test_ensemble_probs_20.npy`
   - `predictions/test_ensemble_signals_1.npy`
   - `predictions/test_ensemble_signals_5.npy`
   - `predictions/test_ensemble_signals_20.npy`

3. **Test Set Metrics**
   - `reports/phase5_test_metrics.json`
     - Classification and trading metrics per horizon
   - `reports/phase5_confusion_matrix_{1,5,20}bar.png`

4. **Backtest Tearsheets**
   - `reports/phase5_test_tearsheet_1bar.html`
   - `reports/phase5_test_tearsheet_5bar.html`
   - `reports/phase5_test_tearsheet_20bar.html`

5. **Comparison Reports**
   - `reports/phase5_val_vs_test_{1,5,20}bar.csv`
   - `reports/phase5_per_symbol_{1,5,20}bar.csv`

6. **Summary Report**
   - `reports/PHASE5_summary.md`
     - Final test results
     - Generalization analysis
     - Per-symbol breakdown
     - System readiness assessment

### Quality Metrics

| Metric (5-bar) | Target | Actual |
|---------------|--------|--------|
| Test Sharpe | >0.40 | ___ |
| Test F1 | >0.35 | ___ |
| Val-Test Sharpe gap | <15% | ___ |
| Val-Test F1 gap | <15% | ___ |
| No lookahead detected | Yes | ___ |
| Pipeline runs error-free | Yes | ___ |

---

## Implementation Checklist

### Pre-Phase Tasks
- [ ] Verify all Phase 2-4 outputs are available
- [ ] Load test set indices
- [ ] Confirm test set has never been used for training/tuning

### Pipeline Development
- [ ] Implement `EnsembleForecastPipeline` class
- [ ] Load all trained models and meta-learners
- [ ] Implement feature engineering function
- [ ] Implement sequence creation
- [ ] Implement single-sample prediction
- [ ] Implement batch prediction

### Pipeline Testing
- [ ] Test pipeline on small sample (10 bars)
- [ ] Verify outputs match expected format
- [ ] Run lookahead sanity check
- [ ] Test pipeline on different symbols (MES, MGC)

### Test Set Evaluation
- [ ] Load test set data
- [ ] Run pipeline on full test set
- [ ] Save predictions to files
- [ ] Verify no errors or warnings

### Metrics Computation
- [ ] Compute classification metrics (Acc, F1, confusion matrix)
- [ ] Compute trading metrics (Sharpe, DD, trades, win rate)
- [ ] Repeat for all horizons (1, 5, 20 bars)

### Comparison Analysis
- [ ] Load validation metrics from Phase 4
- [ ] Compare val vs test performance
- [ ] Check for large degradation (>15%)
- [ ] Analyze per-symbol performance (MES vs MGC)

### Visualization
- [ ] Generate confusion matrices (heatmaps)
- [ ] Generate backtest tearsheets (vectorbt)
- [ ] Create comparison charts (val vs test)

### Reporting
- [ ] Compile all test metrics into summary table
- [ ] Write Phase 5 summary report
- [ ] Document any unexpected findings or issues
- [ ] Assess system readiness for deployment

### Post-Phase Tasks
- [ ] Archive test predictions
- [ ] Update project README with final results
- [ ] Prepare for Phase 6 (refinement if needed)

---

## Notes and Considerations

### Test Set Integrity

**Critical:**
- Test set must be completely untouched until Phase 5
- No hyperparameter tuning based on test results
- No "peeking" at test labels during development

**If test performance is poor:**
- Document honestly, do not iterate on test set
- Analysis and refinement happens in Phase 6 on validation set

### Generalization Gap

**Expected:**
- Small degradation (3-10%) from validation to test is normal
- Different time periods, possibly different regimes

**Concerning:**
- >15% degradation suggests overfitting
- Test metrics wildly different from validation

**Actions if gap is large:**
- Review Phase 2-4 for overfitting signs
- Check if test period has unusual market conditions
- Phase 6 may focus on robustness improvements

### Pipeline Performance

**Inference Speed:**
- For real-time use, pipeline should run in <100ms per bar
- Profile bottlenecks (likely model inference)

**Memory Usage:**
- Monitor RAM usage for large batches
- Optimize if needed for deployment

### Symbol-Specific Behavior

**MES vs MGC:**
- May perform differently due to different dynamics
- MES: equity index, more trend-following
- MGC: commodity, more mean-reverting

**Action if one symbol underperforms:**
- Consider separate models per symbol (Phase 6)
- Or adjust meta-learner to be symbol-aware

---

## Success Criteria Summary

Phase 5 is complete and successful when:

1. ✅ Complete pipeline implemented and tested
2. ✅ Pipeline runs on test set without errors
3. ✅ No lookahead bias detected
4. ✅ Test set Sharpe (5-bar) > 0.40
5. ✅ Test set F1 (5-bar) > 0.35
6. ✅ Val-Test performance gap < 15%
7. ✅ All metrics computed and documented
8. ✅ Backtest tearsheets generated
9. ✅ Per-symbol analysis completed
10. ✅ System ready for deployment (or Phase 6 refinement)

**Proceed to Phase 6** (Iterative Refinement) if improvements are needed, or to **Phase 7** (Deployment) if test results meet all targets.

---

**End of Phase 5 Specification**
