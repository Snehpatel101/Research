# ML Model Factory Documentation

## Overview

This is a modular ML Factory for OHLCV time series that builds model-ready datasets (Phase 1) and trains any model family via plugins (Phases 2-5). The factory is designed to produce production-ready trading models with proper backtesting, cross-validation, and deployment infrastructure.

---

## Phase Status

| Phase | Status | Description | Key Output | Doc |
|-------|--------|-------------|------------|-----|
| 1 | **COMPLETE** | Data prep + labeling | `TimeSeriesDataContainer` | [PHASE_1.md](phases/PHASE_1.md) |
| 2 | **COMPLETE** | Model factory (12 models) | Trained base models | [PHASE_2.md](phases/PHASE_2.md) |
| 3 | **COMPLETE** | Cross-validation | OOS predictions + stacking dataset | [PHASE_3.md](phases/PHASE_3.md) |
| 4 | **COMPLETE** | Ensemble stacking | Meta-learner + ensemble | [PHASE_4.md](phases/PHASE_4.md) |
| 5 | Planned | Test set evaluation | Final metrics + deployment | [PHASE_5.md](phases/PHASE_5.md) |

---

## Factory Pipeline Overview

```
Raw OHLCV Data
      |
      v
+------------------+
|  PHASE 1 (DONE)  |  Data Preparation
|  Ingest -> Clean |  150+ features, multi-timeframe
|  -> Features     |  Triple-barrier labels
|  -> Labels       |  Train/val/test with purge/embargo
|  -> Splits       |  Quality-weighted samples
|  -> Scale        |  TimeSeriesDataContainer
+------------------+
      |
      v
+------------------+
|  PHASE 2 (DONE)  |  Model Training (12 models)
|  Plugin Registry |  Boosting: XGBoost, LightGBM, CatBoost
|  -> Boosting     |  Neural: LSTM, GRU, TCN
|  -> Neural       |  Classical: Random Forest, Logistic, SVM
|  -> Classical    |  Ensemble: Voting, Stacking, Blending
|  -> Ensemble     |  Unified BaseModel interface
+------------------+
      |
      v
+------------------+
|  PHASE 3 (DONE)  |  Cross-Validation
|  Purged K-Fold   |  5-fold with purge/embargo
|  -> OOS Preds    |  Walk-forward feature selection
|  -> HP Tuning    |  Optuna hyperparameter tuning
|  -> Stacking DS  |  Stacking dataset generation
+------------------+
      |
      v
+------------------+
|  PHASE 4 (DONE)  |  Ensemble
|  Meta-Learner    |  Voting, Stacking, Blending
|  -> Voting       |  Soft/hard voting ensembles
|  -> Stacking     |  Meta-learner on OOF predictions
|  -> Blending     |  Holdout-based ensembles
+------------------+
      |
      v
+------------------+
|  PHASE 5         |  Deployment
|  Test Eval       |  Final out-of-sample eval
|  -> Serialization|  Pipeline serialization
|  -> Monitoring   |  Drift detection setup
|  -> Production   |  Paper trading -> Live
+------------------+
```

---

## Quick Navigation

| Goal | Document |
|------|----------|
| Understand Phase 1 (simple) | [phase1/README.md](phase1/README.md) |
| Phase 1 technical spec | [phases/PHASE_1.md](phases/PHASE_1.md) |
| Phase 2 model training | [phases/PHASE_2.md](phases/PHASE_2.md) |
| Phase 3 cross-validation | [phases/PHASE_3.md](phases/PHASE_3.md) |
| Phase 4 ensembles | [phases/PHASE_4.md](phases/PHASE_4.md) |
| Phase 5 deployment | [phases/PHASE_5.md](phases/PHASE_5.md) |
| Get started | [getting-started/QUICKSTART.md](getting-started/QUICKSTART.md) |
| CLI reference | [getting-started/PIPELINE_CLI.md](getting-started/PIPELINE_CLI.md) |
| Architecture | [reference/ARCHITECTURE.md](reference/ARCHITECTURE.md) |
| Feature catalog | [reference/FEATURES.md](reference/FEATURES.md) |
| **Colab/Jupyter setup** | [**COLAB_GUIDE.md**](COLAB_GUIDE.md) |
| **Interactive notebooks** | `../notebooks/` (4 notebooks) |

---

## Model Family Quick Reference

### Which Model Family to Use

| Use Case | Recommended | Reasoning |
|----------|-------------|-----------|
| Quick baseline | XGBoost | Fast, interpretable, strong default |
| Best single model | LightGBM or CatBoost | Often matches XGBoost, sometimes better |
| Temporal patterns | LSTM/GRU | Learns sequential dependencies |
| Long-range dependencies | Transformer | Attention captures distant patterns |
| Interpretability | Logistic Regression | Coefficients are weights |
| Production ensemble | XGBoost + LSTM + RF | Diverse model families |

### Model Family Requirements

| Family | Scaling | Sequences | Feature Set | GPU |
|--------|---------|-----------|-------------|-----|
| **Boosting** (XGBoost, LightGBM, CatBoost) | No | No | `boosting_optimal` | Optional |
| **Neural** (LSTM, GRU, TCN) | RobustScaler | Yes (60 steps) | `neural_optimal` | Recommended |
| **Classical** (Random Forest, Logistic, SVM) | Varies | No | `boosting_optimal` | No |
| **Ensemble** (Voting, Stacking, Blending) | Inherited | Inherited | Mixed | Inherited |

### Feature Sets (from Phase 1)

| Feature Set | Models | Features | Description |
|-------------|--------|----------|-------------|
| `boosting_optimal` | Tree-based | 80-120 | All useful features, correlation-pruned |
| `neural_optimal` | LSTM, GRU | 40-60 | Bounded/normalized features + wavelets |
| `transformer_raw` | Transformers | 10-15 | Minimal features, returns + temporal |
| `ensemble_base` | Mixed | 60-80 | Diverse for ensemble diversity |

---

## Quick Commands

### Phase 1 (Data Preparation)

```bash
# Run full Phase 1 pipeline
./pipeline run --symbols MES,MGC

# Check pipeline status
./pipeline status <run_id>

# Validate outputs
./pipeline validate --run-id <run_id>
```

### Phase 2 (Model Training) - COMPLETE

```bash
# Train single model
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60
python scripts/train_model.py --model random_forest --horizon 20

# Train multiple models
python scripts/train_model.py --model xgboost,lightgbm,lstm --horizon 5,10,15,20

# List available models (should show 12)
python scripts/train_model.py --list-models
```

### Phase 3 (Cross-Validation) - COMPLETE

```bash
# Run cross-validation
python scripts/run_cv.py --models xgboost,lstm --horizons 5,10,15,20 --n-splits 5

# Run with hyperparameter tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100

# Run all models
python scripts/run_cv.py --models all --horizons all
```

### Phase 4 (Ensemble) - COMPLETE

```bash
# Train voting ensemble
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,lstm --horizon 20

# Train stacking ensemble
python scripts/train_model.py --model stacking --base-models xgboost,lgbm,rf --horizon 20

# Train blending ensemble
python scripts/train_model.py --model blending --base-models lstm,gru,tcn --horizon 20
```

### Phase 5 (Deployment) - Planned

```bash
# Final test set evaluation
python scripts/evaluate_test.py --test-data data/splits/scaled/test_scaled.parquet

# Serialize for deployment
python scripts/serialize_pipeline.py --output-dir deployment/pipeline_v1

# Setup monitoring
python scripts/setup_monitoring.py --reference-data data/splits/scaled/train_scaled.parquet
```

---

## Key Configuration Parameters

### Phase 1 Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Horizons | [5, 10, 15, 20] bars | Short to medium-term trading |
| Timeframe | 5-minute bars | Balance of signal vs noise |
| Splits | 70/15/15 | Standard train/val/test |
| Purge | 60 bars | 3x max horizon (prevents leakage) |
| Embargo | 1440 bars | 5 trading days (breaks serial correlation) |
| MES barriers | 1.5:1 (up:down) | Long bias from research |
| MGC barriers | 1.0:1.0 | Symmetric (no directional bias) |
| Optimizer | Optuna TPE | Superior to grid/random search |

### Phase 2-5 Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CV folds | 5 | Standard for stable estimates |
| Optuna trials | 100 | Balance of search vs compute |
| Early stopping | 10-15 epochs | Prevent overfitting |
| Meta-learner | Logistic | Simple, interpretable, low overfit |
| Val-test gap threshold | 15% | Acceptable generalization |
| Drift detection | PSI > 0.2 | Standard threshold |

---

## Expected Performance Targets

### By Horizon (Test Set)

| Horizon | Min Sharpe | Min F1 | Max Drawdown | Notes |
|---------|------------|--------|--------------|-------|
| H5 | 0.30 | 0.35 | -25% | Most noise, hardest |
| H10 | 0.40 | 0.38 | -20% | Reasonable baseline |
| H15 | 0.45 | 0.40 | -18% | Good signal |
| H20 | 0.50 | 0.42 | -18% | Best expected performance |

### By Model Family (Validation)

| Model | H20 Sharpe | H20 F1 | Training Time |
|-------|------------|--------|---------------|
| XGBoost | 0.55-0.70 | 0.44-0.48 | 5-10 min |
| LightGBM | 0.55-0.70 | 0.44-0.48 | 3-8 min |
| LSTM | 0.50-0.65 | 0.42-0.46 | 30-60 min |
| Ensemble | 0.60-0.80 | 0.46-0.50 | (sum of base) |

---

## Data Flow Summary

### Phase 1 Output (Input to Phase 2)

```
data/splits/
|
+-- datasets/
|   +-- core_full/
|       +-- h5/
|       |   +-- train.parquet    # Training data for H5
|       |   +-- val.parquet
|       |   +-- test.parquet
|       +-- h10/
|       +-- h15/
|       +-- h20/
|
+-- scaled/
    +-- feature_scaler.pkl       # Trained scaler (fit on train only)
    +-- train_scaled.parquet
    +-- val_scaled.parquet
    +-- test_scaled.parquet
```

### Phase 3 Output (Input to Phase 4)

```
data/stacking/
|
+-- h5/
|   +-- stacking_dataset.parquet  # OOS predictions for ensemble training
+-- h10/
+-- h15/
+-- h20/
|
+-- tuned_params/
    +-- xgboost_h20.json          # Best hyperparameters
    +-- lstm_h20.json
```

### Phase 5 Output (Deployment)

```
deployment/pipeline_v1/
|
+-- config.json                   # Pipeline configuration
+-- scaler.pkl                    # Feature scaler
+-- feature_columns.json          # Feature list
+-- models/                       # Base models
|   +-- xgboost_h20.pkl
|   +-- lstm_h20.pkl
+-- ensemble/                     # Meta-learners
|   +-- meta_h20.pkl
+-- manifest.json                 # Version and metadata
```

---

## Research Foundation

### Key References

1. **Lopez de Prado (2018)** - "Advances in Financial Machine Learning"
   - Triple-barrier labeling
   - Purged k-fold cross-validation
   - MDA feature importance

2. **Lopez de Prado (2020)** - "Machine Learning for Asset Managers"
   - Sample weighting
   - Feature clustering
   - Hierarchical risk parity

3. **PatchTST (2023)** - "A Time Series is Worth 64 Words"
   - Transformer patching for efficiency
   - Channel-independence

4. **TimesFM (2024)** - Google foundation model
   - Zero-shot forecasting
   - Fine-tuning strategies

### Design Decisions

| Decision | Choice | Alternative | Why |
|----------|--------|-------------|-----|
| Labeling | Triple-barrier | Fixed threshold | Captures stop-loss behavior |
| Splits | Time-based | Random | Prevents future leakage |
| Embargo | 1440 bars | Purge only | Breaks serial correlation |
| Meta-learner | Logistic | XGBoost | Lower overfit risk |
| Feature selection | Walk-forward | Full data | Prevents lookahead bias |

---

## Directory Structure

```
docs/
  README.md                    # This file (hub document)
  phase1/
    README.md                  # Layman's explanation of Phase 1
  phases/
    PHASE_1.md                 # Technical spec (COMPLETE)
    PHASE_2.md                 # Model factory (PLANNED)
    PHASE_3.md                 # Cross-validation (PLANNED)
    PHASE_4.md                 # Ensemble (PLANNED)
    PHASE_5.md                 # Deployment (PLANNED)
  getting-started/
    QUICKSTART.md              # Getting started guide
    PIPELINE_CLI.md            # CLI reference
  reference/
    ARCHITECTURE.md            # System architecture
    FEATURES.md                # Feature catalog (150+ features)
    SLIPPAGE.md                # Transaction cost modeling
```

---

## Contributing

### Adding a New Model

1. Implement `BaseModel` interface (see PHASE_2.md)
2. Add `@ModelRegistry.register` decorator
3. Create config file in `config/models/`
4. Add tests
5. Document in PHASE_2.md

### Adding New Features

1. Add feature calculation in `src/phase1/stages/features/`
2. Update feature catalog in `docs/reference/FEATURES.md`
3. Assign to appropriate feature set
4. Add tests with known values

### Reporting Issues

1. Describe expected vs actual behavior
2. Include relevant configuration
3. Provide minimal reproduction steps
4. Attach relevant logs

---

## Changelog

### 2025-12-24

- Phase 1 marked COMPLETE
- Documentation expanded for Phases 2-5
- Added model family quick reference
- Added feature set guidance
- Added expected performance targets
