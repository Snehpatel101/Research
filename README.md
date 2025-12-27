# ML Model Factory for OHLCV Time Series

Modular ML pipeline that turns raw OHLCV bars into trained models with leakage-safe splits, cross-validation, and unified evaluation.

**Single-Contract Architecture:** This is a single-contract ML factory. Each contract (MES, MGC, etc.) is trained in complete isolation. No cross-symbol correlation or feature engineering.

```
[ Phase 1: Data ] → [ Phase 2: Models ] → [ Phase 3: CV ] → [ Phase 4: Ensemble ] → [ Phase 5: Prod ]
    COMPLETE           COMPLETE            COMPLETE           COMPLETE            PLANNED
```

## Quick Start

```bash
# Run Phase 1 pipeline (requires real data in data/raw/)
./pipeline run --symbols MGC

# Train a model (Phase 2)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30

# Run cross-validation (Phase 3)
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Train ensemble (Phase 4)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,lstm --horizon 20

# List available models
python scripts/train_model.py --list-models
```

**Colab Quick Start:**

See [COLAB_GUIDE.md](docs/COLAB_GUIDE.md) for detailed setup instructions or use the quickstart notebooks:

```bash
notebooks/01_quickstart.ipynb              # Complete pipeline walkthrough
notebooks/02_train_all_models.ipynb        # Train all 12 models
notebooks/03_cross_validation.ipynb        # CV and hyperparameter tuning
```

```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir(
    "data/splits/scaled",
    horizon=20,
)
X_train, y_train, w_train = container.get_sklearn_arrays("train")
```

## Available Models (12 Total)

| Family | Models | GPU Support | Status |
|--------|--------|-------------|--------|
| Boosting | XGBoost, LightGBM, CatBoost | Optional | Complete |
| Neural | LSTM, GRU, TCN | Required (CUDA) | Complete |
| Classical | Random Forest, Logistic, SVM | No | Complete |
| Ensemble | Voting, Stacking, Blending | No | Complete |

**All 12 models** implement the unified `BaseModel` interface and work with the same preprocessed datasets from Phase 1.

## Pipeline Stages

**Phase 1: Data Pipeline**
1. Ingest - Load and validate raw OHLCV
2. Clean - Resample 1min → 5min, handle gaps
3. Features - 150+ indicators (momentum, wavelets, microstructure)
4. Labels - Triple-barrier with symbol-specific asymmetric barriers
5. GA Optimize - Optuna parameter optimization
6. Final Labels - Apply optimized parameters
7. Splits - Train/val/test with purge (60) and embargo (1440)
8. Scaling - Train-only robust scaling
9. Datasets - Build TimeSeriesDataContainer
10. Validation - Feature correlation and quality checks

**Phase 2: Model Factory**
- Plugin-based model registry with `@register` decorator
- Unified `BaseModel` interface for all model types
- GPU-optimized training with mixed precision

**Phase 3: Cross-Validation**
- PurgedKFold with configurable purge/embargo
- Walk-forward feature selection (MDA/MDI)
- Out-of-fold predictions for stacking
- Optuna hyperparameter tuning

**Phase 4: Ensemble Models**
- Voting ensembles (soft/hard voting)
- Stacking with meta-learners
- Blending with holdout predictions
- Diversity analysis and model weighting

## Key Outputs

- Scaled splits: `data/splits/scaled/train_scaled.parquet`
- Trained models: `experiments/runs/<run_id>/`
- CV results: `experiments/cv/<run_id>/`

## Configuration

```python
# Single contract per pipeline run
SYMBOL = 'MES'                 # or 'MGC' - one contract at a time

HORIZONS = [5, 10, 15, 20]      # Label horizons
TRAIN/VAL/TEST = 70/15/15      # Split ratios
PURGE_BARS = 60                # Prevent label leakage
EMBARGO_BARS = 1440            # ~5 days for serial correlation
```

### Symbol Configuration

Each contract requires its own pipeline run:

```bash
# Train MES model
./pipeline run --symbols MES

# Train MGC model (separate run, separate model)
./pipeline run --symbols MGC
```

Data paths resolve from symbol: `data/raw/{symbol}_1m.parquet`

## Project Structure

```
├── src/
│   ├── models/           # Phase 2: Model factory (12 models)
│   │   ├── boosting/     # XGBoost, LightGBM, CatBoost
│   │   ├── neural/       # LSTM, GRU, TCN
│   │   ├── classical/    # Random Forest, Logistic, SVM
│   │   └── ensemble/     # Voting, Stacking, Blending
│   ├── cross_validation/ # Phase 3: CV system
│   ├── phase1/           # Phase 1: Data pipeline
│   └── cli/              # CLI entrypoints
├── scripts/              # Training scripts
├── config/models/        # Model YAML configs (12 configs)
├── notebooks/            # Jupyter/Colab notebooks (4 notebooks)
├── data/                 # Data artifacts
├── experiments/          # Training outputs
└── tests/                # Test suite (1592 passing, 13 skipped)
```

## Documentation

- `ARCHITECTURE_MAP.md` - Visual system architecture with all 12 models
- `CLAUDE.md` - Development guidelines and factory pattern
- `docs/phases/` - Phase specifications (Phases 1-5)
- `docs/COLAB_GUIDE.md` - Google Colab setup and GPU configuration
- `notebooks/` - Interactive Jupyter/Colab notebooks

## Notebooks

Four interactive notebooks for training and experimentation:

1. `01_quickstart.ipynb` - Complete pipeline walkthrough
2. `02_train_all_models.ipynb` - Train all 12 models across horizons
3. `03_cross_validation.ipynb` - CV and hyperparameter tuning
4. `Phase1_Pipeline_Colab.ipynb` - Phase 1 on Google Colab with GPU

## Tests

**Test Coverage: 1592 tests passing, 13 skipped**

```bash
# Run all tests
python -m pytest tests/ -v

# Run model tests only
python -m pytest tests/models tests/cross_validation -v

# Run specific model family
python -m pytest tests/models/test_classical_models.py -v
python -m pytest tests/models/test_ensemble_models.py -v
```
