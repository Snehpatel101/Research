# Unified ML Pipeline - Quick Start Guide

**TL;DR:** One interface for everything. Data → Training → Evaluation in ONE call.

---

## 🚀 5-Minute Quickstart

### Option 1: Python API

```python
from src.ml_pipeline import MLPipeline, MLConfig

# Create config
config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost", "lstm"],
    build_ensemble=True,
)

# Run full pipeline
pipeline = MLPipeline(config)
results = pipeline.run()  # Data + Training + Evaluation

# Done! Models saved to experiments/runs/{run_id}/
```

### Option 2: CLI

```bash
# Full pipeline in one command
ml run --symbol MES --models xgboost lstm --build-ensemble

# Results saved to experiments/runs/{run_id}/
```

---

## 📋 What You Get

**After running the pipeline:**

```
experiments/runs/{run_id}/
├── models/                      # Trained models
│   ├── xgboost_h20.pkl
│   └── lstm_h20.pkl
├── results/                     # Performance metrics
│   ├── xgboost_h20_results.json
│   └── lstm_h20_results.json
├── pipeline_state.json          # Checkpoint file
└── config.yaml                  # Configuration used
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Experiment

```python
# Train single model
config = MLConfig(symbol="MES", models=["xgboost"])
pipeline = MLPipeline(config)
results = pipeline.run()
```

### Use Case 2: Compare Multiple Models

```python
# Train and compare
config = MLConfig(
    symbol="MES",
    models=["xgboost", "lightgbm", "lstm"],
    build_ensemble=True,
)
pipeline = MLPipeline(config)
results = pipeline.run()

# Best model automatically identified
print(f"Best: {results['evaluation']['best_model']}")
```

### Use Case 3: Heterogeneous Ensemble

```python
from src.ml_pipeline import ModelConfig

config = MLConfig(
    symbol="MES",
    models=[
        ModelConfig(name="xgboost", timeframe="15min"),
        ModelConfig(name="lstm", timeframe="5min"),
        ModelConfig(name="patchtst", timeframe="1min"),
    ],
    build_ensemble=True,
    meta_learner="ridge_meta",
)

pipeline = MLPipeline(config)
results = pipeline.run()
```

### Use Case 4: Resume After Failure

```python
# First run (interrupted)
config = MLConfig(symbol="MES", models=["xgboost"])
pipeline = MLPipeline(config)
try:
    results = pipeline.run()
except Exception:
    pass  # Checkpoint saved automatically

# Resume later
pipeline = MLPipeline.from_checkpoint(run_id="20260116_120000")
results = pipeline.resume()  # Continues from checkpoint
```

---

## 🎨 Available Models

**23 models across 6 families (22 if CatBoost unavailable):**

| Family | Models |
|--------|--------|
| **Boosting** | `xgboost`, `lightgbm`, `catboost` |
| **Classical** | `random_forest`, `logistic`, `svm` |
| **Neural** | `lstm`, `gru`, `tcn`, `transformer` |
| **Advanced** | `patchtst`, `itransformer`, `tft`, `nbeats`, `inceptiontime`, `resnet1d` |
| **Ensemble** | `voting`, `stacking`, `blending` |
| **Meta** | `ridge_meta`, `mlp_meta`, `calibrated_meta`, `xgboost_meta` |

**Usage:**
```python
# Single model
config = MLConfig(models=["xgboost"])

# Multiple models
config = MLConfig(models=["xgboost", "lstm", "patchtst"])

# All tabular models
config = MLConfig(models=["xgboost", "lightgbm", "catboost", "random_forest"])
```

---

## ⚙️ Configuration Options

### Basic Options

```python
config = MLConfig(
    # Data
    symbol="MES",              # Contract: MES, MGC, ES, GC
    horizons=[5, 10, 15, 20],  # Label horizons
    start_date="2020-01-01",   # Optional (uses all data if not set)
    end_date="2024-12-31",     # Optional
    
    # Models
    models=["xgboost", "lstm"],  # Model list
    build_ensemble=True,          # Build ensemble from models
    
    # Training
    training_mode="standard",  # standard (others: walk_forward, regime_aware)
    
    # Evaluation
    evaluation_method="cv",    # cv, walk_forward, cpcv_pbo
    cv_splits=5,               # Number of CV folds
)
```

### Advanced Options

```python
from src.ml_pipeline import ModelConfig

config = MLConfig(
    symbol="MES",
    horizons=[20],
    
    # Per-model configuration
    models=[
        ModelConfig(
            name="xgboost",
            timeframe="15min",           # Train on 15min bars
            optimize_features=True,      # Optuna feature selection
            feature_opt_trials=30,       # Optuna trials
        ),
        ModelConfig(
            name="lstm",
            timeframe="5min",            # Train on 5min bars
            sequence_length=60,          # 60-bar lookback
            optimize_features=True,
        ),
    ],
    
    # Ensemble
    build_ensemble=True,
    ensemble_method="stacking",     # voting/stacking/blending
    meta_learner="ridge_meta",      # ridge_meta/mlp_meta/xgboost_meta
)
```

---

## 🔧 CLI Reference

### Commands

```bash
# Full pipeline
ml run --symbol MES --models xgboost lstm --build-ensemble

# Data only (Phase 1-5)
ml data --symbol MES --horizons 20

# Training only (Phase 6)
ml train --models xgboost --training-mode standard

# Evaluation only (Phase 7)
ml evaluate --models xgboost --evaluation-method cv

# Resume from checkpoint
ml resume --run-id 20260116_120000

# Check status
ml status --run-id 20260116_120000
```

### Using YAML Config

```bash
# Create config file
cat > my_config.yaml << EOF
symbol: MES
horizons: [20]
models:
  - xgboost
  - lstm
build_ensemble: true
training_mode: standard
EOF

# Run with config
ml run --config my_config.yaml
```

---

## 🐛 Troubleshooting

### Issue 1: NotImplementedError for Advanced Modes

**Error:**
```
NotImplementedError: Walk-forward training requires extraction from scripts/run_walk_forward.py
```

**Solution:** Use existing script as workaround:
```bash
python scripts/run_walk_forward.py --model xgboost --symbol MES
```

**Why:** Advanced training modes use placeholder implementations. Standard mode (90% of use cases) fully works.

### Issue 2: No Data Found

**Error:**
```
FileNotFoundError: No data found at data/raw/MES_1m.parquet
```

**Solution:** Ensure raw data exists:
```bash
ls data/raw/
# Should show: MES_1m.parquet (or MES_1m.csv)
```

### Issue 3: Model Not Found

**Error:**
```
ValueError: Model 'my_model' not registered
```

**Solution:** List available models:
```bash
ml run --list-models
# Or in Python:
from src.models import ModelRegistry
print(ModelRegistry.list_all())
```

---

## 📚 More Information

**Documentation:**
- `docs/UNIFIED_PIPELINE_ARCHITECTURE.md` - Complete architecture
- `docs/UNIFIED_PIPELINE_FINAL_SUMMARY.md` - Full completion summary
- `docs/MODEL_FAMILY_COMPATIBILITY.md` - Model compatibility details
- `examples/unified_pipeline_basic.py` - More Python examples
- `notebooks/unified_training_colab.ipynb` - Jupyter notebook examples

**Tests:**
```bash
# Run tests
cd /home/jake/Desktop/Research
python -m pytest tests/test_unified_pipeline.py -v
```

---

## 💡 Tips

### Tip 1: Start Simple

Start with one model, then add more:
```python
# Start here
config = MLConfig(symbol="MES", models=["xgboost"])

# Then expand
config = MLConfig(symbol="MES", models=["xgboost", "lightgbm", "lstm"])

# Then ensemble
config = MLConfig(
    symbol="MES",
    models=["xgboost", "lightgbm", "lstm"],
    build_ensemble=True,
)
```

### Tip 2: Use Phase Control

Run phases separately for debugging:
```python
pipeline = MLPipeline(config)

# Phase 1: Data
data_results = pipeline.run_data()
print(f"Features: {data_results['n_features']}")

# Phase 2: Training
training_results = pipeline.run_training()
print(f"Model path: {training_results['model_path']}")

# Phase 3: Evaluation
eval_results = pipeline.run_evaluation()
print(f"F1 score: {eval_results['val_f1']}")
```

### Tip 3: Save Your Configs

```python
# Save config for reproducibility
config = MLConfig(symbol="MES", models=["xgboost"])
config.to_yaml("experiments/my_experiment.yaml")

# Load later
config = MLConfig.from_yaml("experiments/my_experiment.yaml")
pipeline = MLPipeline(config)
```

---

## 🎯 Quick Command Reference

```bash
# Train single model
ml run --symbol MES --models xgboost

# Train multiple models
ml run --symbol MES --models xgboost lightgbm lstm

# Train with ensemble
ml run --symbol MES --models xgboost lstm --build-ensemble

# Multiple horizons
ml run --symbol MES --horizons 5,10,15,20 --models xgboost

# Resume from checkpoint
ml resume --run-id 20260116_120000

# Check status
ml status --run-id 20260116_120000

# List models
ml run --list-models
```

---

## 🚦 Status Check

**Is the pipeline working?**
```bash
# Run tests
python -m pytest tests/test_unified_pipeline.py -v

# Should show: 23 tests passed ✅
```

**Quick smoke test:**
```python
from src.ml_pipeline import MLPipeline, MLConfig

# This should work without errors
config = MLConfig(symbol="MES", models=["xgboost"])
pipeline = MLPipeline(config)
print("✅ Pipeline initialized successfully!")
```

---

## 🎉 You're Ready!

**Next steps:**
1. Prepare your data (`data/raw/MES_1m.parquet`)
2. Run your first experiment
3. Check results in `experiments/runs/{run_id}/`
4. Iterate and improve!

**Need help?** Check the documentation in `docs/` directory.

---

*Last Updated: 2026-01-16*
