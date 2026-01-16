# Jupyter Notebooks for ML Model Factory

This directory contains Jupyter notebooks for exploratory analysis, experimentation, and production monitoring.

## Directory Structure

```
notebooks/
├── 01_exploration/      # Exploratory Data Analysis
├── 02_labeling/         # Label quality analysis
├── 03_modeling/         # Model training experiments
├── 04_production/       # Production monitoring
└── templates/           # Parameterized templates (Papermill)
```

## Quick Start

### 1. Install Required Tools

```bash
pip install jupyter papermill nbconvert jupytext
```

### 2. Create Experiment Notebook

See `ANALYSIS_AND_RECOMMENDATIONS.md` (Section: Template Notebook) for complete code.

Copy the experiment template code and save as `templates/experiment_template.ipynb`.

### 3. Run Parameterized Experiment

```bash
papermill templates/experiment_template.ipynb \
  results/MES_xgboost_h20.ipynb \
  -p symbol MES \
  -p horizon 20 \
  -p model_name xgboost
```

### 4. Version Control with Jupytext

```bash
# Pair notebooks with .py files for git
jupytext --set-formats ipynb,py:percent **/*.ipynb

# Auto-sync on save
jupytext --sync **/*.ipynb
```

## Notebook Templates

### `experiment_template.ipynb` (Parameterized)

**Parameters:**
- `symbol`: Contract symbol (MES, MGC, etc.)
- `horizon`: Label horizon (5, 10, 15, 20)
- `model_name`: Model to train (xgboost, lstm, etc.)
- `use_microstructure`: Enable 2024 microstructure features
- `cross_validate`: Run cross-validation

**Outputs:**
- Feature distributions
- Correlation analysis
- Model training results
- Cross-validation scores
- Feature importance
- Experiment summary CSV/JSON

**Usage:**
```bash
# Single experiment
papermill templates/experiment_template.ipynb output.ipynb \
  -p symbol MES -p model_name xgboost -p horizon 20

# Batch experiments
for model in xgboost lightgbm catboost; do
  papermill templates/experiment_template.ipynb \
    results/MES_${model}_h20.ipynb \
    -p symbol MES -p model_name ${model} -p horizon 20
done
```

## Best Practices

### 1. Notebook Organization

- **One task per notebook:** Keep notebooks focused
- **Clear markdown cells:** Document assumptions and findings
- **Cell execution order:** Run top-to-bottom (no hidden state)
- **Output cells:** Clear before committing to git

### 2. Reproducibility

- **Random seeds:** Set `random_seed=42` consistently
- **Version pinning:** Document library versions
- **Data versioning:** Use DVC for large datasets
- **Parameter tracking:** Use Papermill for experiments

### 3. Code Quality

- **Imports at top:** All imports in first code cell
- **Helper functions:** Extract to `src/` modules
- **Magic commands:** Use sparingly, document why needed
- **Cell length:** Keep cells under 20 lines when possible

### 4. Version Control

- **Jupytext pairing:** All notebooks paired with `.py` files
- **Git diffs:** Review `.py` files, not `.ipynb`
- **Clear outputs:** Use `jupyter nbconvert --clear-output` before commit
- **`.gitignore`:** Exclude large output files

## Example Workflows

### Exploratory Data Analysis

```python
# 01_exploration/feature_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet('data/processed/MES_5min/features.parquet')

# Check for 2024 microstructure features
micro_2024 = [c for c in df.columns if 'edge_spread' in c or 'vpin' in c or 'time_to' in c]
print(f"2024 features: {micro_2024}")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feat in enumerate(micro_2024[:4]):
    ax = axes[i // 2, i % 2]
    df[feat].hist(bins=50, ax=ax)
    ax.set_title(feat)
plt.tight_layout()
```

### Model Comparison

```python
# 03_modeling/model_comparison.ipynb
import json
from pathlib import Path

models = ['xgboost', 'lightgbm', 'catboost', 'lstm', 'tcn']
results = {}

for model in models:
    result_path = Path(f'experiments/{model}_h20/results.json')
    if result_path.exists():
        with open(result_path) as f:
            results[model] = json.load(f)

# Compare F1 scores
import pandas as pd
df_results = pd.DataFrame({
    model: {
        'val_f1': res['evaluation_metrics']['val_f1'],
        'val_sharpe': res['evaluation_metrics'].get('val_sharpe', None)
    }
    for model, res in results.items()
}).T

print(df_results.sort_values('val_f1', ascending=False))
```

### Production Monitoring

```python
# 04_production/model_monitoring.ipynb
import pandas as pd
from datetime import datetime, timedelta

# Load recent predictions
pred_path = 'experiments/production/latest_predictions.parquet'
df_pred = pd.read_parquet(pred_path)

# Check prediction drift
recent_7d = df_pred[df_pred.index > datetime.now() - timedelta(days=7)]
recent_30d = df_pred[df_pred.index > datetime.now() - timedelta(days=30)]

print(f"7-day pred mean: {recent_7d['prediction'].mean():.3f}")
print(f"30-day pred mean: {recent_30d['prediction'].mean():.3f}")
print(f"Drift: {abs(recent_7d['prediction'].mean() - recent_30d['prediction'].mean()):.3f}")
```

## DVC Integration

Track notebook outputs and large datasets:

```yaml
# dvc.yaml
stages:
  exploration:
    cmd: jupyter nbconvert --execute --to notebook 01_exploration/feature_analysis.ipynb
    deps:
      - data/processed/MES_5min/features.parquet
      - 01_exploration/feature_analysis.ipynb
    outs:
      - 01_exploration/feature_analysis.nbconvert.ipynb

  experiment:
    cmd: papermill templates/experiment_template.ipynb experiments/${symbol}_${model}_h${horizon}.ipynb -p symbol ${symbol} -p model_name ${model} -p horizon ${horizon}
    params:
      - symbol
      - model
      - horizon
    deps:
      - data/splits/scaled/
      - templates/experiment_template.ipynb
    outs:
      - experiments/${symbol}_${model}_h${horizon}.ipynb
```

Run with:
```bash
dvc repro exploration
dvc exp run -S symbol=MES -S model=xgboost -S horizon=20
```

## Troubleshooting

**Q: Notebook kernel dies?**  
A: Check memory usage. Large datasets may need chunking or Dask.

**Q: Can't import from `src/`?**  
A: Add to sys.path: `sys.path.insert(0, '..')`

**Q: Papermill parameters not working?**  
A: Tag parameter cell with "parameters" (Cell → Cell Tags in Jupyter)

**Q: Jupytext sync issues?**  
A: Run `jupytext --sync notebook.ipynb` manually

---

**For complete notebook templates, see:** `ANALYSIS_AND_RECOMMENDATIONS.md`
