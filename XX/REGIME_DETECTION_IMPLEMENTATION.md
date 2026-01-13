# Regime Detection Implementation Summary

## Overview

Successfully added **regime detection** and **regime-aware evaluation** to `/home/jake/Desktop/Research/notebooks/ML_Pipeline.ipynb` based on 2025 financial ML best practices.

## Research Context

Recent research shows that ML models trained on one market regime (bull) often fail in others (sideways/bearish). Hidden Markov Models (HMMs) effectively capture regime transitions by modeling market regimes as hidden states. State Street research identified 4 distinct market regimes using 23 performance/uncertainty datasets.

## Implementation Details

### Changes Made

#### 1. NotebookConfig Updates (Cell 4)

Added regime detection fields to the `NotebookConfig` dataclass:

```python
# Regime detection
regime_labels: Optional[np.ndarray] = None
regime_probabilities: Optional[np.ndarray] = None
n_regimes: int = 4
regime_names: List[str] = field(default_factory=lambda: ["low_vol", "normal", "high_vol", "crisis"])
```

These fields store:
- `regime_labels`: Array of regime labels aligned with the data (same length as train+val+test combined)
- `regime_probabilities`: HMM state probabilities for each bar
- `n_regimes`: Number of regimes detected (default 4 per research)
- `regime_names`: Human-readable regime names

#### 2. Cell 3.4: Detect Market Regimes (HMM)

New cell inserted after Cell 3.3 (Verify Processed Data) that:

**Configuration:**
- Uses 4 regimes (per State Street 2025 research)
- HMM-based detection (from `src.phase1.stages.regime.HMMRegimeDetector`)
- Volatility-based regime classification
- Expanding window with periodic retraining

**Process:**
1. Loads processed data from `config.splits_dir` (train+val+test combined)
2. Detects regimes using `HMMRegimeDetector`:
   - `n_states=4`: 4 hidden states
   - `lookback=252`: ~1 trading year lookback
   - `input_type="returns"`: Use log returns as HMM input
   - `expanding=True`: Expanding window (no lookahead bias)
   - `retrain_interval=50`: Retrain every 50 bars for efficiency
3. Stores regime labels in `config.regime_labels`
4. Visualizes regime transitions with:
   - Price chart with regime background shading
   - Regime state transitions over time
5. Calculates regime statistics:
   - Distribution (% time in each regime)
   - Average returns per regime (annualized)
   - Average volatility per regime (annualized)
   - Regime transitions and average duration

**Regime Classification:**
- **State 0 (low_vol)**: Low volatility regime - Green background
- **State 1 (normal)**: Normal volatility regime - Blue background
- **State 2 (high_vol)**: High volatility regime - Orange background
- **State 3 (crisis)**: Crisis/extreme volatility regime - Red background

States are automatically ordered by volatility (low to high) by the HMM detector.

**Output:**
- `config.regime_labels`: Array of regime labels (aligned with data)
- `config.regime_probabilities`: State probabilities for each bar
- Visualization showing regime transitions over time
- Regime statistics printed to console

#### 3. Cell 4.1 Updates: Store Validation Predictions

Updated the model training cell to store validation predictions:

```python
config.training_results[model_name] = {
    'metrics': result.metrics,
    'time': model_time,
    'run_id': result.run_id,
    'model_path': str(result.model_path),
    'val_predictions': result.val_predictions,  # ← Added
    'config': trainer_config.__dict__,
}
```

This enables regime-aware analysis by providing access to the raw validation predictions for each model.

#### 4. Cell 4.2 Updates: Training Summary with Per-Regime Performance

Updated the Training Summary cell to show per-regime performance breakdown:

```python
# PER-REGIME PERFORMANCE BREAKDOWN
if hasattr(config, 'regime_labels') and config.regime_labels is not None:
    print("\n" + "=" * 70)
    print(" PERFORMANCE BY MARKET REGIME")
    print("=" * 70)

    # For each successful model
    for model_name in successful['Model'].values:
        val_preds = config.training_results[model_name]['val_predictions']

        # Calculate accuracy and F1 per regime
        for regime in unique_regimes:
            regime_mask = (val_regimes == regime)
            n_samples = regime_mask.sum()

            if n_samples >= 10:
                regime_acc = accuracy_score(y_val[regime_mask], val_preds[regime_mask])
                regime_f1 = f1_score(y_val[regime_mask], val_preds[regime_mask], average='macro')
                print(f"  {regime}: Acc={regime_acc:.4f}, F1={regime_f1:.4f} ({n_samples} samples)")
```

**Output:**
- Shows accuracy and F1 score for each model in each regime
- Identifies which models are regime-agnostic (consistent across regimes)
- Highlights models that specialize in specific regimes

#### 5. Cell 4.3: Regime Performance Heatmap

New cell inserted after Cell 4.2 (Training Summary) that creates a comprehensive regime performance visualization:

**Features:**
- Heatmap showing Models (rows) × Regimes (columns)
- Cell values = Macro F1 score in that regime
- Color scale: Red (poor) → Yellow (medium) → Green (excellent)
- Sample counts shown in column headers

**Analysis Provided:**
1. **Most Regime-Agnostic Model**: Model with lowest std dev across regimes
2. **Best Specialist per Regime**: Best performing model in each regime
3. **Best Overall Model**: Highest average F1 across all regimes

**Output:**
- Visual heatmap (uses seaborn)
- Insights printed to console
- Regime performance stored in `config.training_results[model]['regime_performance']`

**Use Cases:**
- Identify models that perform consistently across all regimes
- Identify models that specialize in specific market conditions
- Inform regime-switching ensemble strategies
- Validate model robustness

## Key Implementation Notes

### Lookahead Prevention

The HMM detector uses an **expanding window** with proper anti-lookahead measures:

1. **Expanding Mode**: Trains on data from start up to current bar (no future data)
2. **Periodic Retraining**: Retrains every 50 bars for efficiency (configurable)
3. **Shift by 1**: All regime predictions are shifted by 1 bar to ensure no lookahead

This ensures that the regime at bar N only uses data from bars 0..N-1.

### Data Requirements

The regime detection requires:
- `close` price column in the processed data
- Sufficient data for HMM fitting (min 252 bars recommended)
- Train/val/test data from `config.splits_dir`

If `close` column is missing, the cell will attempt to find close-related columns or skip gracefully.

### Performance Considerations

- HMM fitting is computationally intensive
- Expanding mode with `retrain_interval=50` balances accuracy and speed
- For faster execution, increase `retrain_interval` (e.g., 100)
- For maximum accuracy, set `retrain_interval=1` (retrain every bar)

### Regime Naming Convention

The 4 regimes are named based on volatility ordering:
- **low_vol**: Lowest volatility state (trending, stable)
- **normal**: Medium-low volatility state
- **high_vol**: Medium-high volatility state
- **crisis**: Highest volatility state (extreme moves, uncertainty)

These names are derived from the HMM state ordering by volatility.

## File Structure

### Modified Files

1. `/home/jake/Desktop/Research/notebooks/ML_Pipeline.ipynb` (28 cells, was 26)
   - Cell 4: Updated NotebookConfig
   - Cell 11: **NEW** - Detect Market Regimes (HMM)
   - Cell 14: Updated Train Models (stores val_predictions)
   - Cell 15: Updated Training Summary (per-regime performance)
   - Cell 16: **NEW** - Regime Performance Heatmap

### Backup

- `/home/jake/Desktop/Research/notebooks/ML_Pipeline_backup.ipynb` (original)

### Scripts

- `/home/jake/Desktop/Research/notebooks/add_regime_detection.py` (implementation script)

## Usage Workflow

### In the Notebook

1. **Run Section 3**: Data pipeline and processing
2. **Run Cell 3.4**: Detect market regimes (HMM)
   - View regime transitions visualization
   - Review regime statistics
3. **Run Cell 4.1**: Train models
4. **Run Cell 4.2**: View training summary
   - Now includes per-regime performance breakdown
5. **Run Cell 4.3**: View regime performance heatmap
   - Identify regime-agnostic vs specialist models

### Expected Output

#### Cell 3.4 Output

```
======================================================================
 REGIME DETECTION (HMM)
======================================================================

  Method: HMM
  Regime Type: volatility
  Number of Regimes: 4

  Total samples: 150,000

  Running HMM regime detection...

  Regime Distribution:
    low_vol: 45,000 bars (30.0%)
    normal: 52,500 bars (35.0%)
    high_vol: 37,500 bars (25.0%)
    crisis: 15,000 bars (10.0%)

  Regime Characteristics:
    low_vol:
      Avg Return: +12.50% (annualized)
      Avg Volatility: 8.20% (annualized)
    normal:
      Avg Return: +8.30% (annualized)
      Avg Volatility: 15.40% (annualized)
    high_vol:
      Avg Return: -2.10% (annualized)
      Avg Volatility: 28.70% (annualized)
    crisis:
      Avg Return: -15.80% (annualized)
      Avg Volatility: 45.20% (annualized)

  Regime Transitions: 342
  Avg Regime Duration: 438.6 bars

  Regime labels stored in config.regime_labels
======================================================================
```

#### Cell 4.2 Additional Output

```
======================================================================
 PERFORMANCE BY MARKET REGIME
======================================================================

xgboost:
  low_vol     : Acc=0.5823, F1=0.5612 (6,750 samples)
  normal      : Acc=0.5691, F1=0.5498 (7,875 samples)
  high_vol    : Acc=0.5234, F1=0.5089 (5,625 samples)
  crisis      : Acc=0.4892, F1=0.4723 (2,250 samples)

lstm:
  low_vol     : Acc=0.5945, F1=0.5734 (6,750 samples)
  normal      : Acc=0.5812, F1=0.5623 (7,875 samples)
  high_vol    : Acc=0.5567, F1=0.5401 (5,625 samples)
  crisis      : Acc=0.5123, F1=0.4967 (2,250 samples)
```

#### Cell 4.3 Output

```
======================================================================
 REGIME PERFORMANCE HEATMAP
======================================================================

  Regimes in validation set: ['low_vol', 'normal', 'high_vol', 'crisis']
  Validation samples: 22,500

  Key Insights:
    Most Regime-Agnostic: lstm (std=0.0345)

    Best Model per Regime:
      low_vol: lstm (F1=0.5734)
      normal: lstm (F1=0.5623)
      high_vol: lstm (F1=0.5401)
      crisis: lstm (F1=0.4967)

    Best Overall (avg across regimes): lstm (F1=0.5431)

  Regime performance stored in config.training_results[model]['regime_performance']
======================================================================
```

## Dependencies

The implementation uses existing codebase components:

- `src.phase1.stages.regime.HMMRegimeDetector`: HMM regime detection
- `hmmlearn`: Python library for HMM (required)
- `sklearn.metrics`: For per-regime metrics
- `matplotlib`, `seaborn`: For visualizations

### Installation

If `hmmlearn` is not installed:

```bash
pip install hmmlearn
```

## Next Steps

### Potential Extensions

1. **Regime-Switching Ensemble**
   - Train separate models per regime
   - Route predictions based on detected regime
   - Implement in Cell 6.1 (Ensemble Training)

2. **Adaptive Parameters**
   - Adjust triple-barrier parameters per regime
   - Use tighter barriers in low_vol, wider in high_vol
   - Modify `src/phase1/stages/labeling/`

3. **Regime-Aware Walk-Forward**
   - Split WF periods by regime
   - Test model degradation across regime transitions
   - Update Cell 3.5 (Walk-Forward Backtest)

4. **Feature Importance per Regime**
   - Calculate SHAP values per regime
   - Identify regime-specific features
   - Add to Cell 4.3 or create new cell

5. **Regime Transition Prediction**
   - Train model to predict upcoming regime changes
   - Use as early warning signal
   - New cell in Section 4

## Validation

All changes have been validated:

- ✓ Notebook is valid JSON (28 cells total)
- ✓ Cell 11 (3.4 Detect Market Regimes) created successfully
- ✓ Cell 16 (4.3 Regime Performance Heatmap) created successfully
- ✓ NotebookConfig updated with regime fields
- ✓ Cell 14 (4.1) stores val_predictions
- ✓ Cell 15 (4.2) shows per-regime performance
- ✓ Backup saved to `notebooks/ML_Pipeline_backup.ipynb`

## Summary

The notebook now has **comprehensive regime detection and regime-aware evaluation**:

1. **Automatic regime detection** using HMM with 4 states (low_vol, normal, high_vol, crisis)
2. **Per-regime performance metrics** showing how each model performs in different market conditions
3. **Regime performance heatmap** visualizing model × regime performance
4. **Regime statistics** including distribution, returns, volatility, and transitions
5. **Regime transition visualization** showing market state changes over time

This enables users to:
- Understand which models are robust across regimes vs specialists
- Identify market conditions where models fail
- Build regime-switching ensembles
- Validate model robustness before deployment

All implementation follows 2025 financial ML best practices with proper anti-lookahead measures and expanding window HMM fitting.
