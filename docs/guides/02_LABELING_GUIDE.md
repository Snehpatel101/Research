# Labeling Pipeline - Quick Start Guide

## Overview

This pipeline creates production-ready labels for financial time series using:
1. Triple-barrier method with ATR-based dynamic barriers
2. Genetic algorithm optimization for parameter tuning
3. Quality-based sample weighting

## Installation

```bash
pip install -r requirements_labeling.txt
```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run all stages (4, 5, 6) sequentially
python src/run_labeling_pipeline.py
```

Expected runtime: ~10-15 minutes for 2 symbols with 1 year of 5-minute data.

### Option 2: Run Stages Individually

```bash
# Stage 4: Initial labeling with default parameters
python src/stages/stage4_labeling.py

# Stage 5: GA optimization (finds best parameters)
python src/stages/stage5_ga_optimize.py

# Stage 6: Apply optimized labels with quality scores
python src/stages/stage6_final_labels.py
```

## Output Files

After running the pipeline:

```
data/
├── labels/
│   ├── MES_labels_init.parquet       # Stage 4 output
│   └── MGC_labels_init.parquet
├── final/
│   ├── MES_final_labeled.parquet     # Stage 6 output (use this!)
│   └── MGC_final_labeled.parquet

config/
└── ga_results/
    ├── MES_ga_h1_best.json           # Optimized parameters
    ├── MES_ga_h5_best.json
    ├── MES_ga_h20_best.json
    └── optimization_summary.json      # All results

results/
├── ga_plots/
│   ├── MES_ga_h1_convergence.png     # GA convergence plots
│   ├── MES_ga_h5_convergence.png
│   └── MES_ga_h20_convergence.png
└── labeling_report.md                # Summary report
```

## Using the Labels

### Load Final Labeled Data

```python
import pandas as pd
import numpy as np

# Load final labeled data
df = pd.read_parquet('data/final/MES_final_labeled.parquet')

print(f"Total samples: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
```

### Extract Features and Labels

```python
# Define feature columns (exclude metadata and labels)
exclude_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
label_cols = [c for c in df.columns if c.startswith(('label_', 'bars_to_hit_', 
                                                       'mae_', 'mfe_', 'touch_type_',
                                                       'quality_', 'sample_weight_'))]
feature_cols = [c for c in df.columns if c not in exclude_cols + label_cols]

print(f"Features: {len(feature_cols)}")

# Extract for horizon 5
X = df[feature_cols].values
y = df['label_h5'].values
weights = df['sample_weight_h5'].values

print(f"X shape: {X.shape}")
print(f"y distribution: {np.bincount(y + 1)}")  # [-1, 0, 1] -> [0, 1, 2]
print(f"Weight distribution: {np.bincount(weights.astype(int))}")
```

### Train with Sample Weights

```python
from sklearn.ensemble import RandomForestClassifier

# Train model with quality-based sample weights
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y, sample_weight=weights)

# Higher quality samples have more influence on training
```

### Understanding Labels

```python
# Examine label statistics by horizon
for horizon in [1, 5, 20]:
    labels = df[f'label_h{horizon}']
    quality = df[f'quality_h{horizon}']
    bars = df[f'bars_to_hit_h{horizon}']
    
    print(f"\n=== Horizon {horizon} ===")
    print(f"Long/Win:    {(labels == 1).sum():6d} ({(labels == 1).mean()*100:5.1f}%)")
    print(f"Short/Loss:  {(labels == -1).sum():6d} ({(labels == -1).mean()*100:5.1f}%)")
    print(f"Neutral:     {(labels == 0).sum():6d} ({(labels == 0).mean()*100:5.1f}%)")
    print(f"Avg bars:    {bars[bars > 0].mean():.1f}")
    print(f"Avg quality: {quality.mean():.3f}")
```

## Advanced Usage

### Custom Parameters for Stage 4

```python
from stages.stage4_labeling import process_symbol_labeling
from pathlib import Path

custom_params = {
    'k_up': 2.5,      # More aggressive profit target
    'k_down': 1.5,    # Tighter stop loss
    'atr_column': 'atr_14'
}

df = process_symbol_labeling(
    input_path=Path('data/features/MES_5m_features.parquet'),
    output_path=Path('data/labels/MES_custom.parquet'),
    symbol='MES',
    horizons=[1, 5, 20],
    default_params=custom_params
)
```

### Custom GA Settings for Stage 5

```python
from stages.stage5_ga_optimize import process_symbol_ga

results = process_symbol_ga(
    symbol='MES',
    horizons=[5],           # Only optimize horizon 5
    population_size=100,    # Larger population
    generations=50          # More generations
)

print(f"Best parameters: {results[5]}")
```

### Manual Labeling (No GA)

```python
from stages.stage6_final_labels import apply_optimized_labels
import pandas as pd

df = pd.read_parquet('data/features/MES_5m_features.parquet')

# Manually specify parameters (skip GA optimization)
best_params = {
    'k_up': 2.0,
    'k_down': 1.0,
    'max_bars': 15
}

df = apply_optimized_labels(df, horizon=5, best_params=best_params)
```

## Performance Tips

### Speed Up GA Optimization

1. **Use smaller data subset**:
```python
# In stage5_ga_optimize.py, adjust subset_fraction
run_ga_optimization(df, horizon=5, subset_fraction=0.1)  # Use 10% instead of 20%
```

2. **Reduce population/generations**:
```python
process_symbol_ga('MES', population_size=30, generations=20)
```

3. **Run in parallel** (for multiple symbols):
```python
from multiprocessing import Pool

def optimize_symbol(symbol):
    return process_symbol_ga(symbol)

with Pool(2) as p:
    results = p.map(optimize_symbol, ['MES', 'MGC'])
```

### Memory Optimization

For large datasets (>1M rows):
```python
# Process in chunks
chunk_size = 100000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    # Process chunk...
```

## Troubleshooting

### Issue: GA fitness not improving

**Solution**: Check data quality
```python
# Ensure ATR values are valid
df['atr_14'].describe()  # Should have reasonable values
df['atr_14'].isna().sum()  # Should be 0
```

### Issue: All labels are neutral (0)

**Cause**: Barriers too wide or max_bars too small

**Solution**: Adjust parameters
```python
# Try smaller k_up/k_down or larger max_bars
best_params = {
    'k_up': 1.5,      # Smaller (easier to hit)
    'k_down': 1.0,
    'max_bars': 50    # Larger (more time)
}
```

### Issue: Labels too imbalanced

**Cause**: Parameter mismatch

**Solution**: Run GA optimization (it optimizes for balance)
```python
python src/stages/stage5_ga_optimize.py
```

## Validation

Check label quality:
```python
df = pd.read_parquet('data/final/MES_final_labeled.parquet')

# Check balance
labels = df['label_h5']
print("Balance check:")
print(f"  Long:    {(labels == 1).mean()*100:.1f}%  [target: 30-40%]")
print(f"  Short:   {(labels == -1).mean()*100:.1f}%  [target: 30-40%]")
print(f"  Neutral: {(labels == 0).mean()*100:.1f}%  [target: 20-40%]")

# Check speed
bars = df['bars_to_hit_h5']
print(f"\nSpeed check:")
print(f"  Avg bars: {bars[bars > 0].mean():.1f}  [target: 5-15 for h=5]")

# Check quality
quality = df['quality_h5']
print(f"\nQuality check:")
print(f"  Mean: {quality.mean():.3f}")
print(f"  Std:  {quality.std():.3f}")
```

## Next Steps

1. **Train models**: Use labeled data with sample weights
2. **Backtest**: Validate label quality in trading simulation
3. **Iterate**: Adjust GA fitness function if needed

For more details, see:
- `src/stages/README.md` - Detailed documentation
- `results/labeling_report.md` - Generated after Stage 6
- `config/ga_results/optimization_summary.json` - Best parameters

---

*Happy labeling!*
