# ML Model Factory for OHLCV Time Series

Modular ML pipeline that turns raw OHLCV bars into trained models with leakage-safe splits, cross-validation, and unified evaluation.

**Single-Contract Architecture:** Each contract (MES, MGC, SI, etc.) is trained in complete isolation. No cross-symbol correlation.

```
[ Phase 1: Data ] → [ Phase 2: Models ] → [ Phase 3: CV ] → [ Phase 4: Ensemble ]
    COMPLETE           COMPLETE            COMPLETE           COMPLETE
```

---

## Quick Start (Notebook)

The unified notebook is the recommended way to use this pipeline:

```
notebooks/ML_Pipeline.ipynb
```

1. Open in [Google Colab](https://colab.research.google.com/) or Jupyter
2. Configure Section 1 (symbol, models, horizons)
3. Run All Cells
4. Export trained models from Section 7

**Full notebook documentation:** [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)

---

## Quick Start (CLI)

```bash
# Run Phase 1 pipeline (requires data in data/raw/)
./pipeline run --symbols SI

# Train a model
python scripts/train_model.py --model xgboost --horizon 20

# Run cross-validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Train ensemble
python scripts/train_model.py --model voting --base-models xgboost,lightgbm --horizon 20

# List available models
python scripts/train_model.py --list-models
```

---

## Available Models (13 Total)

| Family | Models | GPU | Description |
|--------|--------|-----|-------------|
| **Boosting** | XGBoost, LightGBM, CatBoost | Optional | Fast, interpretable |
| **Neural** | LSTM, GRU, TCN, Transformer | Required | Sequential patterns |
| **Classical** | Random Forest, Logistic, SVM | No | Robust baselines |
| **Ensemble** | Voting, Stacking, Blending | No | Combine models |

All models implement the unified `BaseModel` interface.

---

## Data Flow

```
Raw OHLCV (.csv/.parquet)
       │
       ▼
┌─────────────────────────────────────────┐
│           PHASE 1: DATA PIPELINE        │
│  Clean → Features (150+) → Labels       │
│  → Split (70/15/15) → Scale             │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│          PHASE 2: MODEL TRAINING        │
│  Boosting │ Neural │ Classical          │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│     PHASE 3: CROSS-VALIDATION (opt)     │
│  PurgedKFold │ Optuna Tuning            │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│       PHASE 4: ENSEMBLE (opt)           │
│  Voting │ Stacking │ Blending           │
└─────────────────────────────────────────┘
       │
       ▼
   Exported Models (.joblib, .onnx)
```

---

## Key Features

- **No Data Leakage**: Purge (60 bars) + Embargo (1440 bars) between splits
- **Reproducible**: Random seeds for all libraries
- **Class Balanced**: Automatic class weight calculation
- **Quality Weighted**: Pipeline quality scores used in training
- **150+ Features**: Momentum, volatility, wavelets, microstructure
- **Multi-Timeframe**: 5min, 15min, 1hr, daily aggregations

---

## Project Structure

```
├── notebooks/
│   └── ML_Pipeline.ipynb    # Unified notebook (recommended)
├── src/
│   ├── models/              # 13 model implementations
│   ├── cross_validation/    # PurgedKFold, Optuna tuning
│   └── phase1/              # Data pipeline stages
├── data/
│   ├── raw/                 # Input: {symbol}_1m.csv
│   └── splits/scaled/       # Output: train/val/test parquet
├── experiments/             # Training outputs
└── config/models/           # Model YAML configs
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) | Complete notebook usage guide |
| [CLAUDE.md](CLAUDE.md) | Development guidelines |
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | Google Colab setup |

---

## Configuration

```python
SYMBOL = 'SI'              # Contract symbol
HORIZONS = [5, 10, 15, 20] # Prediction horizons (bars forward)
TRAIN/VAL/TEST = 70/15/15  # Split ratios
PURGE_BARS = 60            # Gap to prevent label leakage
EMBARGO_BARS = 1440        # ~5 days for serial correlation
RANDOM_SEED = 42           # Reproducibility
```

---

## Requirements

- Python 3.10+
- PyTorch (for neural models, GPU recommended)
- XGBoost, LightGBM, CatBoost
- scikit-learn, pandas, numpy

```bash
pip install -r requirements.txt
```
