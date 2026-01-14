# Quick Reference

Command cheatsheet and FAQ for the ML Model Factory.

**Last Updated:** 2026-01-08

---

## Table of Contents

1. [Data Pipeline (Phase 1)](#data-pipeline-phase-1)
2. [Model Training (Phase 2)](#model-training-phase-2)
3. [Cross-Validation (Phase 3)](#cross-validation-phase-3)
4. [Ensembles (Phase 4)](#ensembles-phase-4)
5. [Inference & Serving (Phase 5)](#inference--serving-phase-5)
6. [Notebook Workflow](#notebook-workflow)
7. [Key Configuration Files](#key-configuration-files)
8. [Common File Paths](#common-file-paths)
9. [FAQ](#faq)
10. [Troubleshooting Quick Links](#troubleshooting-quick-links)

---

## Data Pipeline (Phase 1)

### Run Full Pipeline

```bash
# Process single contract
./pipeline run --symbols SI

# Process with custom data directory
./pipeline run --symbols MES --data-dir /path/to/data

# Dry run (validate only)
./pipeline run --symbols MGC --dry-run
```

### Check Pipeline Status

```bash
# Show run status
./pipeline status <run_id>

# Watch logs
tail -f runs/<run_id>/logs/pipeline.log

# List recent runs
ls -la runs/
```

### Pipeline Outputs

| Output | Location |
|--------|----------|
| Run metadata | `runs/<run_id>/` |
| Processed data | `data/splits/scaled/*.parquet` |
| Features | `data/features/{symbol}_features.parquet` |
| Logs | `runs/<run_id>/logs/` |

---

## Model Training (Phase 2)

### List Available Models

```bash
python scripts/train_model.py --list-models
```

**Output:** 23 models across 4 families (22 if CatBoost unavailable)

| Family | Models |
|--------|--------|
| Tabular (6) | `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm` |
| Neural (10) | `lstm`, `gru`, `tcn`, `transformer`, `inceptiontime`, `resnet1d`, `nbeats`, `patchtst`, `itransformer`, `tft` |
| Ensemble (3) | `voting`, `stacking`, `blending` |
| Meta-Learners (4) | `ridge_meta`, `mlp_meta`, `calibrated_meta`, `xgboost_meta` |

### Train Single Model

```bash
# Boosting models (fast, no GPU required)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lightgbm --horizon 20
python scripts/train_model.py --model catboost --horizon 20

# Neural sequence models (GPU recommended)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60
python scripts/train_model.py --model gru --horizon 20 --seq-len 60
python scripts/train_model.py --model tcn --horizon 20 --seq-len 120
python scripts/train_model.py --model transformer --horizon 20 --seq-len 60

# Advanced neural models (GPU recommended)
python scripts/train_model.py --model inceptiontime --horizon 20 --seq-len 60
python scripts/train_model.py --model resnet1d --horizon 20 --seq-len 60
python scripts/train_model.py --model nbeats --horizon 20 --seq-len 60
python scripts/train_model.py --model patchtst --horizon 20 --seq-len 60
python scripts/train_model.py --model itransformer --horizon 20 --seq-len 60
python scripts/train_model.py --model tft --horizon 20 --seq-len 60

# Classical tabular models (CPU only)
python scripts/train_model.py --model random_forest --horizon 20
python scripts/train_model.py --model logistic --horizon 20
python scripts/train_model.py --model svm --horizon 20
```

### Train All Models

```bash
# Train all models for a horizon
python scripts/train_model.py --model all --horizon 20
```

### Training Outputs

| Output | Location |
|--------|----------|
| Trained models | `experiments/runs/<run_id>/models/` |
| Performance reports | `experiments/runs/<run_id>/reports/` |
| Training logs | `experiments/runs/<run_id>/logs/` |

---

## Cross-Validation (Phase 3)

### Run Cross-Validation

```bash
# Single model, single horizon
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Multiple models and horizons
python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,15,20

# All models
python scripts/run_cv.py --models all --horizons 20

# With hyperparameter tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 50
```

### Walk-Forward Validation

```bash
python scripts/run_walk_forward.py --model xgboost --horizon 20 --n-splits 5
```

### CPCV-PBO Validation

```bash
python scripts/run_cpcv_pbo.py --model xgboost --horizon 20 --n-paths 100
```

### CV Outputs

| Output | Location |
|--------|----------|
| CV results | `data/stacking/cv_results.json` |
| Tuned params | `data/stacking/tuned_params/*.json` |
| OOF datasets | `data/stacking/stacking/*.parquet` |

---

## Ensembles (Phase 4)

### Voting Ensemble

```bash
# Tabular models only
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence models only
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60

# Custom weights
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"weights": [0.6, 0.4]}'
```

### Stacking Ensemble

```bash
# With logistic meta-learner (default)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,random_forest

# With XGBoost meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost \
  --config '{"meta_learner_name": "xgboost"}'

# Fewer folds (faster)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60 \
  --config '{"n_folds": 3}'
```

### Blending Ensemble

```bash
# Standard blending
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Larger holdout
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"holdout_fraction": 0.3}'
```

### Heterogeneous Stacking (Phase 7)

```bash
# Cross-family stacking with meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm,patchtst --meta-learner ridge_meta

# Meta-learner options: ridge_meta, mlp_meta, calibrated_meta, xgboost_meta
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,gru,tft --meta-learner xgboost_meta
```

### Compatibility Rules

| Valid Ensembles | Notes |
|-----------------|-------|
| `xgboost + lightgbm + catboost` | Same-family (tabular) |
| `lstm + gru + tcn` | Same-family (neural sequence) |
| `xgboost + lstm + patchtst` | Heterogeneous (requires meta-learner) |
| `catboost + tcn + tft` | Heterogeneous (requires meta-learner) |

---

## Inference & Serving (Phase 5)

### Serve Model

```bash
python scripts/serve_model.py --bundle /path/to/bundle --port 8080
```

### Batch Inference

```bash
python scripts/batch_inference.py \
  --bundle /path/to/bundle \
  --input /path/to/input.parquet \
  --output /path/to/preds.parquet
```

---

## Notebook Workflow

### Open Notebook

```bash
# Local Jupyter
jupyter lab
# Navigate to notebooks/ML_Pipeline.ipynb

# Google Colab
# Upload notebooks/ML_Pipeline.ipynb to Colab
```

### Minimal Configuration

```python
# Cell 1: Configuration
SYMBOL = "SI"
TRAIN_XGBOOST = True
TRAINING_HORIZON = 20

# Run All Cells: Ctrl+F9 (Colab) / Cmd+F9 (Mac)
```

### Common Workflows

| Workflow | Time | Config |
|----------|------|--------|
| Quick Boosting Comparison | 30 min | `TRAIN_XGBOOST/LIGHTGBM/CATBOOST = True` |
| Neural Training | 60 min | `TRAIN_LSTM = True` (GPU required) |
| Full Pipeline + Ensemble | 2 hours | All models + `TRAIN_STACKING = True` |

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config/pipeline.yaml` | Pipeline configuration |
| `config/features.yaml` | Feature engineering settings |
| `config/labeling.yaml` | Triple-barrier parameters |
| `config/ensembles.yaml` | Ensemble defaults |
| `config/models/*.yaml` | Per-model configurations |

### Model Config Example

```yaml
# config/models/xgboost.yaml
model:
  name: xgboost
  family: boosting

defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1
```

---

## Common File Paths

| Path | Purpose |
|------|---------|
| `data/raw/` | Raw OHLCV data (e.g., `MES_1m.parquet`) |
| `data/splits/scaled/` | Processed train/val/test splits |
| `data/features/` | Engineered features |
| `data/stacking/` | CV results and OOF datasets |
| `experiments/runs/<run_id>/` | Training runs and artifacts |
| `runs/<run_id>/` | Pipeline run metadata |
| `config/` | Configuration files |
| `notebooks/` | Jupyter notebooks |

---

## FAQ

### Q: How do I add a new model?

**A:** Implement `BaseModel` interface and use `@register` decorator:

```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, ...):
        pass
    def predict(self, X):
        pass
    def save(self, path):
        pass
    @classmethod
    def load(cls, path):
        pass
```

**See:** `docs/guides/MODEL_INTEGRATION.md`

---

### Q: How do I train on a different symbol?

**A:** Run the pipeline with the new symbol:

```bash
./pipeline run --symbols MGC
python scripts/train_model.py --model xgboost --horizon 20
```

---

### Q: Why can't I mix tabular and neural models in ensembles?

**A:** Tabular models use 2D input `(N, F)`, neural models use 3D input `(N, T, F)`. Ensembles require consistent input shapes.

**See:** `docs/guides/META_LEARNER_STACKING.md`

---

### Q: How do I reduce GPU memory usage?

**A:** Reduce batch size or sequence length:

```python
# In config or CLI
BATCH_SIZE = 128      # Default: 256
SEQUENCE_LENGTH = 30  # Default: 60
```

---

### Q: How do I skip Phase 1 (data pipeline)?

**A:** Set `RUN_DATA_PIPELINE = False` in notebook config, or ensure `data/splits/scaled/` contains processed data before training.

---

### Q: What horizons are supported?

**A:** Default horizons are 5, 10, 15, 20 bars. Configure in `config/labeling.yaml`:

```yaml
horizons:
  - 5
  - 10
  - 15
  - 20
```

---

### Q: How do I enable GPU for boosting models?

**A:** Boosting models auto-detect GPU. To force GPU:

```yaml
# config/models/xgboost.yaml
defaults:
  tree_method: hist  # Enable GPU
  device: cuda
```

---

### Q: How do I export trained models?

**A:** Models are saved automatically to `experiments/runs/<run_id>/models/`. For notebook export, use Section 7.

---

### Q: What features are available?

**A:** ~180 features including:
- Momentum: RSI, MACD, Stochastic, ROC
- Trend: ADX, MACD Signal, Moving Averages
- Volatility: ATR, Bollinger Bands
- Wavelets: Db4/Haar decomposition (3 levels)
- Microstructure: Spread proxies, order flow

**See:** `docs/guides/FEATURE_ENGINEERING.md`

---

### Q: How do I tune hyperparameters?

**A:** Use Optuna via CV:

```bash
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100
```

**See:** `docs/guides/HYPERPARAMETER_TUNING.md`

---

## Troubleshooting Quick Links

| Issue | Doc |
|-------|-----|
| GPU not detected | `docs/guides/NOTEBOOK_SETUP.md#issue-gpu-not-detected` |
| Out of memory | `docs/guides/NOTEBOOK_SETUP.md#issue-out-of-memory` |
| Data file not found | `docs/guides/NOTEBOOK_SETUP.md#issue-data-file-not-found` |
| MTF issues | `docs/troubleshooting/MTF_TROUBLESHOOTING.md` |
| Import errors | `docs/guides/NOTEBOOK_SETUP.md#issue-package-import-errors` |
| Ensemble compatibility | `docs/guides/META_LEARNER_STACKING.md` |
| Session timeout (Colab) | `docs/guides/NOTEBOOK_SETUP.md#issue-session-timeout` |

---

## See Also

- **Architecture:** `docs/ARCHITECTURE.md`
- **Implementation:** `docs/implementation/PHASE_*.md`
- **Guides:** `docs/guides/`
- **Models Reference:** `docs/reference/MODELS.md`
