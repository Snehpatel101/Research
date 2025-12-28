# ML Pipeline Notebook Documentation

Complete reference for the ML Model Factory notebook (`notebooks/ML_Pipeline.ipynb`).

---

## Quick Start

```
1. Open notebooks/ML_Pipeline.ipynb in Colab or Jupyter
2. Configure Section 1 (symbol, models, horizons)
3. Run All Cells (Ctrl+F9 / Cmd+F9)
4. Export trained models from Section 7
```

**Minimum Configuration:**
```python
SYMBOL = "SI"           # Your contract symbol
TRAIN_XGBOOST = True    # Enable at least one model
TRAINING_HORIZON = 20   # Prediction horizon (bars forward)
```

---

## Documentation Structure

| Document | Purpose | Lines |
|----------|---------|-------|
| [CONFIGURATION.md](CONFIGURATION.md) | All 45+ config parameters, model selection | ~380 |
| [CELL_REFERENCE.md](CELL_REFERENCE.md) | Cell-by-cell documentation for all 7 sections | ~450 |
| [COLAB_SETUP.md](COLAB_SETUP.md) | Google Colab setup, GPU config, data mounting | ~280 |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors, validation failures, solutions | ~180 |

---

## Pipeline Overview

**File:** `notebooks/ML_Pipeline.ipynb`
**Models:** 13 (Boosting, Neural, Classical, Ensemble)
**Features:** 150+ technical indicators

### Pipeline Phases

1. **Configuration** - Master settings panel
2. **Environment Setup** - Auto-detects Colab vs Local
3. **Phase 1: Data Pipeline** - Clean → Features → Labels → Splits → Scale
4. **Phase 2: Model Training** - Train any of 13 model types
5. **Phase 3: Cross-Validation** - Purged K-fold with optional tuning
6. **Phase 4: Ensemble** - Combine models intelligently
7. **Results & Export** - Professional packages with ONNX

---

## Model Support (13 Models)

| Family | Models | Count | GPU Required |
|--------|--------|-------|--------------|
| Boosting | XGBoost, LightGBM, CatBoost | 3 | No |
| Neural | LSTM, GRU, TCN, Transformer | 4 | Yes |
| Classical | Random Forest, Logistic Regression, SVM | 3 | No |
| Ensemble | Voting, Stacking, Blending | 3 | Inherited |

**Training Time (H20 on GPU):**
- Boosting: 1-2 min per model
- Neural: 8-15 min per model
- Classical: 2-5 min per model
- Ensemble: Sum of base models + meta-learner

---

## Common Workflows

### Workflow 1: Quick Model Comparison (30 min)
```
1. Enable: TRAIN_XGBOOST, TRAIN_LIGHTGBM, TRAIN_CATBOOST
2. Run All Cells
3. Check Section 4.2 for comparison table
4. Best model highlighted automatically
```

### Workflow 2: Neural Network Training (60 min)
```
1. Enable GPU runtime (Colab: Runtime > Change runtime type > GPU)
2. Enable: TRAIN_LSTM or TRAIN_TRANSFORMER
3. Adjust SEQUENCE_LENGTH, BATCH_SIZE for GPU memory
4. Run All Cells
5. Check Section 4.3 for learning curves
6. Check Section 4.4 for attention visualization (transformer)
```

### Workflow 3: Hyperparameter Tuning (2-3 hours)
```
1. Train base models first (Workflow 1)
2. Set: RUN_CROSS_VALIDATION = True
3. Set: CV_TUNE_HYPERPARAMS = True
4. Set: CV_N_TRIALS = 50
5. Run Section 5 cells
6. Check Section 5.2 for best params
7. Retrain models with optimized params
```

### Workflow 4: Production Ensemble (90 min)
```
1. Train diverse base models (boosting + neural)
2. Enable: TRAIN_STACKING = True
3. Configure STACKING_BASE_MODELS with trained models
4. Run Section 6 cells
5. Check diversity analysis (Section 6.2)
6. Export best ensemble (Section 7.2)
```

---

## Data Flow Architecture

```
RAW DATA (1-min) → PHASE 1: DATA PIPELINE → Scaled Datasets
                     ├── Clean (resample 1m→5m)
                     ├── Features (150+ indicators)
                     ├── Labels (triple-barrier)
                     ├── Splits (70/15/15 + purge/embargo)
                     └── Scale (train-only robust scaling)
                            ↓
                    PHASE 2: MODEL TRAINING
                     ├── Boosting (3 models)
                     ├── Neural (4 models)
                     ├── Classical (3 models)
                     └── Results dict
                            ↓
                    PHASE 3: CROSS-VALIDATION (optional)
                     ├── PurgedKFold (5 splits)
                     └── Optuna tuning
                            ↓
                    PHASE 4: ENSEMBLE (optional)
                     ├── Voting
                     ├── Stacking
                     └── Blending
                            ↓
                    PHASE 5: EXPORT
                     └── Complete package with models, metrics, viz
```

---

## Key Safeguards

| Safeguard | Implementation | Prevents |
|-----------|----------------|----------|
| Train-only scaling | Scaler fit only on train split | Data leakage from val/test |
| Purge (60 bars) | Gap at split boundaries | Label leakage |
| Embargo (1440 bars) | Post-split buffer (~5 days) | Serial correlation leakage |
| Random seeds | Python, NumPy, PyTorch seeded | Non-reproducible results |
| Class weights | Balanced weighting | Majority class bias |
| Sample weights | Quality-based weighting | Low-quality sample influence |
| Early stopping | Monitor val loss | Overfitting |
| PurgedKFold | Time-aware CV splits | Lookahead bias in CV |

---

## Next Steps

1. **Configuration:** See [CONFIGURATION.md](CONFIGURATION.md) for all parameters
2. **Cell Reference:** See [CELL_REFERENCE.md](CELL_REFERENCE.md) for detailed cell documentation
3. **Colab Setup:** See [COLAB_SETUP.md](COLAB_SETUP.md) for Google Colab instructions
4. **Troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

---

**Last Updated:** 2025-12-28
**Notebook Version:** 2.0
