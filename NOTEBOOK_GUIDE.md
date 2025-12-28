# ML Pipeline Notebook Guide

Complete guide for `notebooks/ML_Pipeline.ipynb` - a unified ML factory for OHLCV time series prediction.

---

## Quick Start

```
1. Open notebooks/ML_Pipeline.ipynb in Colab or Jupyter
2. Configure Section 1 (symbol, models, horizons)
3. Run All Cells
4. Export trained models from Section 7
```

---

## Data Flow Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML PIPELINE DATA FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

    RAW DATA                        PHASE 1                         PHASE 2
    ────────                        ───────                         ───────

  ┌──────────┐     ┌─────────────────────────────────────┐     ┌─────────────┐
  │  CSV/    │     │         DATA PIPELINE               │     │   MODEL     │
  │ Parquet  │────▶│                                     │────▶│  TRAINING   │
  │  OHLCV   │     │  Clean → Features → Labels → Split  │     │             │
  └──────────┘     └─────────────────────────────────────┘     └──────┬──────┘
       │                          │                                   │
       │                          ▼                                   │
       │           ┌─────────────────────────────┐                    │
       │           │   data/splits/scaled/       │                    │
       │           │   ├── train_h20.parquet     │                    │
       │           │   ├── val_h20.parquet       │                    │
       │           │   └── test_h20.parquet      │                    │
       │           └─────────────────────────────┘                    │
       │                                                              │
       │                                                              ▼
       │                                                    ┌─────────────────┐
       │              PHASE 3 (Optional)                    │ TRAINING_RESULTS│
       │              ──────────────────                    │  {model: {...}} │
       │                                                    └────────┬────────┘
       │           ┌─────────────────────────────┐                   │
       │           │    CROSS-VALIDATION         │                   │
       │           │    ─────────────────        │                   │
       │           │  PurgedKFold (5 splits)     │◀──────────────────┤
       │           │  Optuna Hyperparameter Tune │                   │
       │           │  Out-of-Fold Predictions    │                   │
       │           └─────────────────────────────┘                   │
       │                                                              │
       │                                                              ▼
       │              PHASE 4 (Optional)                    ┌─────────────────┐
       │              ──────────────────                    │ ENSEMBLE_RESULTS│
       │                                                    │  {voting: {...}}│
       │           ┌─────────────────────────────┐          └────────┬────────┘
       │           │    ENSEMBLE TRAINING        │                   │
       │           │    ─────────────────        │◀──────────────────┘
       │           │  Voting (weighted avg)      │
       │           │  Stacking (meta-learner)    │
       │           │  Blending (holdout-based)   │
       │           └─────────────────────────────┘
       │                          │
       │                          ▼
       │           ┌─────────────────────────────┐
       │           │         EXPORT              │
       │           │    experiments/export/      │
       │           │    ├── model.joblib         │
       │           │    ├── model.onnx           │
       │           │    ├── config.json          │
       │           │    └── manifest.json        │
       └──────────▶└─────────────────────────────┘
```

---

## Notebook Structure

| Section | Cells | Purpose |
|---------|-------|---------|
| **1. Configuration** | 2 | Master config panel - all settings |
| **2. Environment** | 4-8 | Colab/local setup, GPU, memory |
| **3. Data Pipeline** | 10-12 | Load raw data, run Phase 1, verify |
| **4. Model Training** | 14-18 | Train models, compare, visualize |
| **5. Cross-Validation** | 20-21 | Optional CV and hyperparameter tuning |
| **6. Ensemble** | 23-24 | Optional ensemble training |
| **7. Export** | 26-27 | Summary and model export |

---

## Configuration Reference

### Data Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SYMBOL` | `"SI"` | Contract symbol (MES, MGC, SI, etc.) |
| `DATE_RANGE` | `"2019-2024"` | Data date range filter |
| `HORIZONS` | `"5,10,15,20"` | Prediction horizons (bars forward) |
| `TRAIN_RATIO` | `0.70` | Training set proportion |
| `VAL_RATIO` | `0.15` | Validation set proportion |
| `TEST_RATIO` | `0.15` | Test set proportion |
| `PURGE_BARS` | `60` | Gap between train/val to prevent leakage |
| `EMBARGO_BARS` | `1440` | Post-test embargo (~5 days at 5min) |

### Model Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_XGBOOST` | `True` | XGBoost gradient boosting |
| `TRAIN_LIGHTGBM` | `True` | LightGBM gradient boosting |
| `TRAIN_CATBOOST` | `True` | CatBoost gradient boosting |
| `TRAIN_RANDOM_FOREST` | `False` | Random Forest classifier |
| `TRAIN_LOGISTIC` | `False` | Logistic Regression |
| `TRAIN_SVM` | `False` | Support Vector Machine |
| `TRAIN_LSTM` | `False` | LSTM neural network |
| `TRAIN_GRU` | `False` | GRU neural network |
| `TRAIN_TCN` | `False` | Temporal Convolutional Network |
| `TRAIN_TRANSFORMER` | `False` | Transformer with attention |

### Neural Network Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQUENCE_LENGTH` | `60` | Input sequence length (RNN/TCN) |
| `BATCH_SIZE` | `256` | Training batch size |
| `MAX_EPOCHS` | `50` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `10` | Epochs without improvement before stop |

### Transformer Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRANSFORMER_SEQUENCE_LENGTH` | `128` | Transformer input length |
| `TRANSFORMER_N_HEADS` | `8` | Attention heads |
| `TRANSFORMER_N_LAYERS` | `3` | Transformer layers |
| `TRANSFORMER_D_MODEL` | `256` | Model dimension |

### Boosting Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ESTIMATORS` | `500` | Number of trees |
| `BOOSTING_EARLY_STOPPING` | `50` | Early stopping rounds |

### Ensemble Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_VOTING` | `False` | Train voting ensemble |
| `TRAIN_STACKING` | `False` | Train stacking ensemble |
| `TRAIN_BLENDING` | `False` | Train blending ensemble |
| `VOTING_BASE_MODELS` | `"xgboost,lightgbm,catboost"` | Models for voting |
| `STACKING_BASE_MODELS` | `"xgboost,lightgbm,lstm"` | Models for stacking |
| `STACKING_META_LEARNER` | `"logistic"` | Meta-learner type |
| `BLENDING_HOLDOUT_RATIO` | `0.2` | Holdout for blending |

### Class Balancing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_CLASS_WEIGHTS` | `True` | Balance classes (Long/Neutral/Short) |
| `USE_SAMPLE_WEIGHTS` | `True` | Use pipeline quality weights |

### Cross-Validation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_CROSS_VALIDATION` | `False` | Enable CV |
| `CV_N_SPLITS` | `5` | Number of CV folds |
| `CV_TUNE_HYPERPARAMS` | `False` | Run Optuna tuning |
| `CV_N_TRIALS` | `20` | Optuna trials per model |
| `CV_USE_PRESCALED` | `True` | Use pre-scaled data (faster) |

### Reproducibility

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | `42` | Random seed (0 = random) |
| `SAFE_MODE` | `False` | Extra validation checks |

---

## Model Families (13 Models)

### Boosting (Fast, Interpretable)
- **XGBoost** - Industry standard, handles missing values
- **LightGBM** - Fastest training, low memory
- **CatBoost** - Best for categorical features

### Neural Networks (Sequential Patterns)
- **LSTM** - Long short-term memory
- **GRU** - Gated recurrent unit (faster than LSTM)
- **TCN** - Temporal convolutional network
- **Transformer** - Self-attention mechanism

### Classical ML (Robust Baselines)
- **Random Forest** - Ensemble of decision trees
- **Logistic Regression** - Linear classifier
- **SVM** - Support vector machine

### Ensembles (Combine Models)
- **Voting** - Weighted average of predictions
- **Stacking** - Meta-learner on OOF predictions
- **Blending** - Meta-learner on holdout predictions

---

## Feature Pipeline (150+ Features)

```
Raw OHLCV → Technical Indicators → Wavelet Decomposition → Microstructure
         → Multi-Timeframe (5min, 15min, 1hr, daily)
         → Triple-Barrier Labels → Quality Weights
```

### Feature Categories
- **Momentum**: RSI, MACD, Stochastic, ROC, Williams %R
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, VWAP, Volume Profile
- **Wavelets**: Multi-scale decomposition (4 levels)
- **Microstructure**: Bid-ask spread, order flow imbalance

---

## Typical Workflows

### Workflow 1: Quick Model Comparison
```
1. Set SYMBOL, enable TRAIN_XGBOOST/LIGHTGBM/CATBOOST
2. Run All
3. Check Section 4.2 (Compare Models)
```

### Workflow 2: Neural Network Training
```
1. Enable TRAIN_LSTM or TRAIN_TRANSFORMER
2. Ensure GPU is available (Section 2.3)
3. Adjust SEQUENCE_LENGTH, BATCH_SIZE
4. Run All
```

### Workflow 3: Hyperparameter Tuning
```
1. Set RUN_CROSS_VALIDATION = True
2. Set CV_TUNE_HYPERPARAMS = True
3. Set CV_N_TRIALS = 50 (or more)
4. Run All
5. Check Section 5.2 for best params
```

### Workflow 4: Production Ensemble
```
1. Train individual models first (Workflow 1)
2. Enable TRAIN_STACKING = True
3. Configure STACKING_BASE_MODELS
4. Run Section 6 cells
5. Export from Section 7
```

---

## Output Files

```
experiments/
├── runs/{timestamp}/          # Training run artifacts
│   ├── models/                # Saved model files
│   ├── metrics/               # Performance metrics
│   └── config.json            # Run configuration
│
└── export/                    # Production export
    ├── model.joblib           # Sklearn-compatible model
    ├── model.onnx             # ONNX format (optional)
    ├── config.json            # Model configuration
    ├── feature_names.json     # Required features
    └── manifest.json          # Export metadata
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data found" | Check `SYMBOL` matches filename in `data/raw/` |
| GPU not detected | Restart runtime, check Colab GPU setting |
| Out of memory | Reduce `BATCH_SIZE`, disable neural models |
| Poor accuracy | Enable `USE_CLASS_WEIGHTS`, try more `HORIZONS` |
| Slow training | Use boosting models only, reduce `N_ESTIMATORS` |

---

## Key Safeguards

- **No Data Leakage**: Purge/embargo between train/val/test
- **Reproducible**: Random seeds set for all libraries
- **Class Balanced**: Automatic class weight calculation
- **Quality Weighted**: Pipeline quality scores used in training
- **CV Scaling**: Optional per-fold scaling for strict validation

---

## Version Info

- **Notebook Version**: 7-agent reviewed (Dec 27, 2025)
- **Models**: 13 (3 boosting, 4 neural, 3 classical, 3 ensemble)
- **Features**: 150+ indicators
- **Supported Symbols**: Any futures contract with OHLCV data
