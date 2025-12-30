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

**Full notebook documentation:** [docs/notebook/README.md](docs/notebook/README.md)

---

## Quick Start (CLI)

```bash
# Run Phase 1 pipeline (requires data in data/raw/)
./pipeline run --symbols SI

# Train a model
python scripts/train_model.py --model xgboost --horizon 20

# Run cross-validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Train ensemble (tabular-only or sequence-only)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

# List available models (should show 13)
python scripts/train_model.py --list-models
```

---

## Available Models (13 Implemented + 6 Planned = 19 Total)

| Family | Models | Input | GPU | Status |
|--------|--------|-------|-----|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D tabular | Optional | ✅ Complete |
| **Neural** (4) | LSTM, GRU, TCN, Transformer | 3D sequences | Required | ✅ Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D tabular | No | ✅ Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | Mixed | No | ✅ Complete |
| **CNN** (2) | InceptionTime, 1D ResNet | 3D sequences | Required | 📋 Planned |
| **Advanced Transformers** (3) | PatchTST, iTransformer, TFT | 3D sequences | Required | 📋 Planned |
| **MLP** (1) | N-BEATS | 3D sequences | Optional | 📋 Planned |

**Model Categories:**
- **Tabular (6):** Boosting + Classical → 2D input `(n_samples, n_features)`
- **Sequence (13):** Neural + CNN + Advanced + MLP → 3D input `(n_samples, seq_len, n_features)`

All models implement the unified `BaseModel` interface. See `docs/roadmaps/ADVANCED_MODELS_ROADMAP.md` for planned models.

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
   Saved training artifacts (see `experiments/runs/`)

Note: The notebook includes optional export helpers (e.g. ONNX) but the core trainer saves model-family artifacts under each run directory.
```

---

## Key Features

- **No Data Leakage**: Purge (60 bars) + Embargo (1440 bars) between splits
- **Seeded**: Configuration includes a `random_seed` (full determinism depends on backend)
- **Class Balanced**: Automatic class weight calculation
- **Quality Weighted**: Pipeline quality scores used in training
- **180+ Features**: ~150 base indicators + ~30 MTF indicators (all indicator-derived, from 5 of 9 intended timeframes)
- **Multi-Timeframe**: Indicator features from 5 timeframes (15min, 30min, 1h, 4h, daily) - **intended: 9-timeframe ladder** (1min→1h)

---

## ⚠️ Current MTF Limitations

**All models currently receive the same indicator-derived features (~180 total).**

The intended architecture (per `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`) includes **model-specific data strategies**:

| Strategy | Data Type | Model Families | Status |
|----------|-----------|----------------|--------|
| **Strategy 1: Single-TF** | One timeframe, no MTF | All 19 models (baselines) | ❌ Not implemented |
| **Strategy 2: MTF Indicators** | Indicator features from 9 timeframes | **Tabular (6):** Boosting + Classical | ⚠️ Partial (5 of 9 TFs, all models get this) |
| **Strategy 3: MTF Ingestion** | Raw OHLCV bars from 9 timeframes | **Sequence (13):** Neural + CNN + Advanced + MLP | ❌ Not implemented |

**Current Impact:**
- **Tabular models** (6: XGBoost, LightGBM, CatBoost, RF, Logistic, SVM) → Receive indicator features ✅ Appropriate, but only 5 of 9 timeframes
- **Sequence models** (7 implemented + 6 planned = 13) → Receive indicators when they should get raw multi-resolution OHLCV bars ⚠️

**Planned Sequence Models (6):** InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS

**See:** `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md` for detailed analysis

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

**📚 [Complete Documentation Index](docs/INDEX.md)** - Start here for comprehensive guides

### Quick Links

| Category | Document | Purpose |
|----------|----------|---------|
| **Getting Started** | [Quickstart Guide](docs/getting-started/QUICKSTART.md) | Step-by-step setup |
| **Getting Started** | [Quick Reference](docs/QUICK_REFERENCE.md) | Command cheatsheet |
| **Planning** | [Project Charter](docs/planning/PROJECT_CHARTER.md) | Project status: 13 implemented + 6 planned models |
| **Roadmaps** | [Advanced Models Roadmap](docs/roadmaps/ADVANCED_MODELS_ROADMAP.md) | 6 new models, 18 days implementation |
| **Roadmaps** | [MTF Implementation Roadmap](docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) | Multi-timeframe expansion |
| **Guides** | [Model Integration Guide](docs/guides/MODEL_INTEGRATION_GUIDE.md) | How to add new models |
| **Guides** | [Feature Engineering Guide](docs/guides/FEATURE_ENGINEERING_GUIDE.md) | Model-specific feature strategies |
| **Reference** | [Architecture](docs/reference/ARCHITECTURE.md) | System design patterns |
| **Notebook** | [Notebook README](docs/notebook/README.md) | Jupyter/Colab guide |

**5,617 lines** of implementation guides covering model integration, feature engineering, hyperparameter optimization, and infrastructure requirements

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
