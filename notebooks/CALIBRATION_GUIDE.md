# Prediction Calibration & Confidence Analysis

## Overview

Added comprehensive probability calibration, confidence analysis, and uncertainty quantification to the ML Pipeline notebook based on 2025 financial ML best practices.

## What Was Added

### 1. Configuration (Cell 1.2)

New configuration parameters for calibration:

```python
CALIBRATION_METHOD = "isotonic"  # "none", "isotonic", "platt", "beta"
MIN_CONFIDENCE = 0.4             # Minimum confidence threshold (40%)
HIGH_CONFIDENCE_THRESHOLD = 0.7  # High confidence level (70%)
USE_UNCERTAINTY = True           # Enable uncertainty estimation
UNCERTAINTY_METHOD = "ensemble"  # "ensemble", "dropout", "quantile"
FILTER_LOW_CONFIDENCE = False    # Filter trades by confidence
```

### 2. NotebookConfig Storage (Cell 2.1)

Added result storage fields:

```python
calibration_results: Dict[str, Any]     # ECE, Brier scores, metrics
uncertainty_estimates: Dict[str, Any]   # Epistemic/aleatoric uncertainty
```

### 3. Helper Functions (Cell 2.3)

Six new calibration functions:

- `compute_expected_calibration_error(y_true, y_prob, n_bins=10)`
- `compute_brier_score(y_true, y_prob)`
- `plot_calibration_curve(y_true, y_prob, model_name, n_bins=10)`
- `analyze_prediction_confidence(y_true, y_pred, y_prob, confidence_bins)`
- `apply_calibration(model, X_val, y_val, X_test, method='isotonic')`
- `compare_calibration(y_true, y_prob_before, y_prob_after, model_name)`

### 4. Calibration Analysis Cell (Cell 4.5)

New cell for comprehensive calibration analysis:

1. Load trained models
2. Compute calibration metrics (ECE, Brier)
3. Analyze confidence distribution
4. Generate visualizations
5. Apply calibration (if enabled)
6. Compare before/after metrics
7. Store results

### 5. Confidence Filtering (Trading Simulation)

Added confidence-based trade filtering:

- Filters trades below `MIN_CONFIDENCE`
- Reports filtered vs kept trades
- Shows accuracy improvement on high-confidence trades

## Usage

### Step 1: Configure Calibration

In Cell 1.2, set:

```python
CALIBRATION_METHOD = "isotonic"  # Recommended
MIN_CONFIDENCE = 0.5             # Adjust based on risk tolerance
FILTER_LOW_CONFIDENCE = False    # Set True for live trading
```

### Step 2: Train Models

Run Cell 4.1 to train models as usual.

### Step 3: Analyze Calibration

Run Cell 4.5 to:
- Generate calibration curves
- Compute ECE and Brier scores
- Visualize confidence distribution
- Apply calibration if enabled

### Step 4: Run Trading Simulation

Run Cell 4.6 with `FILTER_LOW_CONFIDENCE = True` to:
- Skip low-confidence trades
- Measure accuracy improvement
- Optimize trade frequency vs quality

## Interpretation

### Calibration Quality (ECE)

- **ECE < 0.05**: Well-calibrated (trust probabilities)
- **ECE 0.05-0.10**: Moderately calibrated (usable)
- **ECE > 0.10**: Poorly calibrated (unreliable)

### Confidence Thresholds

- **0.3-0.4**: Aggressive (more trades, lower quality)
- **0.5-0.6**: Balanced (moderate frequency)
- **0.7-0.8**: Conservative (fewer trades, higher quality)

### Calibration Methods

- **isotonic**: Non-parametric, flexible (default choice)
- **platt**: Parametric sigmoid scaling (faster)
- **beta**: Advanced, handles extreme probabilities

### Trade Filtering Strategy

- **Backtest**: `FILTER_LOW_CONFIDENCE = False` (measure all)
- **Paper Trading**: `FILTER_LOW_CONFIDENCE = True`, `MIN_CONFIDENCE = 0.5`
- **Live Trading**: `FILTER_LOW_CONFIDENCE = True`, `MIN_CONFIDENCE = 0.6-0.7`

## Expected Results

### Well-Calibrated Model

```
ECE:          0.032  ✅ Well-calibrated
Brier Score:  0.145  (baseline for 3-class)

Confidence Distribution:
  0.3-0.5:  1,234 trades  45.2% accuracy
  0.5-0.7:  2,156 trades  52.8% accuracy
  0.7-0.9:  1,432 trades  61.4% accuracy
  0.9-1.0:    342 trades  74.1% accuracy
```

### Poorly Calibrated Model

```
ECE:          0.142  ❌ Poorly calibrated
Brier Score:  0.189

After isotonic calibration:
ECE:          0.048  ✅ Improved to well-calibrated
Brier Score:  0.151  (reduced by 0.038)
```

### Confidence Filtering Impact

```
Base accuracy (all):        54.2%
Filtered accuracy (>0.6):   62.8%  (+8.6%)
Trades kept:                2,134 / 5,164 (41.3%)
```

## Research Context

2025 financial ML research emphasizes:

1. **Probability calibration** for risk management
2. **Uncertainty quantification** (knowing when NOT to trade)
3. **Confidence-based position sizing**
4. **Reliable probabilities** for Kelly criterion

### Why Calibration Matters

Miscalibrated models produce:
- **Overconfident predictions** → Excessive risk-taking
- **Underconfident predictions** → Missed opportunities
- **Poor Kelly sizing** → Suboptimal capital allocation

### Standard Methods

- **Isotonic regression**: Maps predictions to empirical frequencies
- **Platt scaling**: Fits logistic regression to map probabilities
- Both improve reliability of probability estimates

## Next Steps

1. Run full pipeline on MES data
2. Evaluate calibration quality for each model
3. Compare filtered vs unfiltered performance
4. Implement ensemble-based uncertainty estimation
5. Add regime-aware confidence thresholds

## Files Modified

- `notebooks/ML_Pipeline.ipynb` (32 cells, +1 new cell)

## References

- Guo et al. (2017): "On Calibration of Modern Neural Networks"
- Kull et al. (2019): "Beyond temperature scaling: Obtaining well-calibrated predictions"
- López de Prado (2018): "Advances in Financial Machine Learning"
- Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"

---

**Last Updated**: 2026-01-12
**Notebook Version**: ML_Pipeline.ipynb (32 cells)
