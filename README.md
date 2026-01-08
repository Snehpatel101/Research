# ML Model Factory for OHLCV Time Series

Modular ML pipeline that turns raw OHLCV bars into trained models with leakage-safe splits, cross-validation, and unified evaluation.

**Single-Contract Architecture:** Each contract (MES, MGC, SI, etc.) is trained in complete isolation. No cross-symbol correlation.

```
[ Phase 1: Data ] -> [ Phase 6: Models ] -> [ Phase 7: Stacking ]
    COMPLETE            COMPLETE             COMPLETE
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

**Full notebook documentation:** [docs/guides/NOTEBOOK_SETUP.md](docs/guides/NOTEBOOK_SETUP.md)

---

## Quick Start (CLI)

```bash
# Run Phase 1 pipeline (requires data in data/raw/)
./pipeline run --symbols MES

# Train a model
python scripts/train_model.py --model xgboost --horizon 20

# Run cross-validation
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Train ensemble (same-family or heterogeneous)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

# Heterogeneous stacking (different model families)
python scripts/train_model.py --model stacking --base-models xgboost,lstm,patchtst --meta-learner ridge_meta --horizon 20

# List available models (should show 22)
python scripts/train_model.py --list-models
```

---

## Available Models (22 Implemented)

| Family | Models | Input | GPU | Status |
|--------|--------|-------|-----|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D tabular | Optional | Complete |
| **Neural** (4) | LSTM, GRU, TCN, Transformer | 3D sequences | Required | Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D tabular | No | Complete |
| **CNN** (2) | InceptionTime, ResNet1D | 3D sequences | Required | Complete |
| **Advanced Transformers** (3) | PatchTST, iTransformer, TFT | 4D multi-res | Required | Complete |
| **MLP** (1) | N-BEATS | 3D sequences | Optional | Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | OOF predictions | No | Complete |
| **Meta-Learners** (4) | Ridge, MLP, Calibrated, XGBoost | OOF predictions | No | Complete |

**Model Categories:**
- **Tabular (6):** Boosting + Classical -> 2D input `(n_samples, n_features)`
- **Sequence (7):** Neural + CNN + MLP -> 3D input `(n_samples, seq_len, n_features)`
- **Multi-Res (3):** Advanced Transformers -> 4D input `(n_samples, n_timeframes, seq_len, n_features)`
- **Ensemble (6):** Same-family or heterogeneous stacking

All models implement the unified `BaseModel` interface.

---

## Data Flow

```
Raw OHLCV (.csv/.parquet)
       |
       v
+---------------------------------------------+
|           PHASE 1: DATA PIPELINE            |
|  Ingest -> MTF (8 TFs) -> Features (180+)   |
|  -> Labels -> Split (70/15/15) -> Scale     |
+---------------------------------------------+
       |
       v
+---------------------------------------------+
|          PHASE 6: MODEL TRAINING            |
|  Boosting | Neural | Classical | Advanced   |
+---------------------------------------------+
       |
       v
+---------------------------------------------+
|     PHASE 7: HETEROGENEOUS STACKING         |
|  OOF Generation -> Meta-Learner Training    |
+---------------------------------------------+
       |
       v
   Saved training artifacts (experiments/runs/)
```

---

## Key Features

- **No Data Leakage**: Purge (60 bars) + Embargo (1440 bars) between splits
- **Multi-Timeframe**: 8 intraday timeframes (5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- **180+ Features**: Base indicators + MTF indicators + wavelets + microstructure
- **Heterogeneous Ensembles**: Different model families with different timeframes
- **Class Balanced**: Automatic class weight calculation
- **Quality Weighted**: Pipeline quality scores used in training

---

## Project Structure

```
+-- notebooks/
|   +-- ML_Pipeline.ipynb    # Unified notebook (recommended)
+-- src/
|   +-- models/              # 22 model implementations
|   +-- cross_validation/    # PurgedKFold, Optuna tuning
|   +-- phase1/              # Data pipeline stages (15 stages)
+-- data/
|   +-- raw/                 # Input: {symbol}_1m.csv
|   +-- splits/scaled/       # Output: train/val/test parquet
+-- experiments/             # Training outputs
+-- config/models/           # Model YAML configs
+-- docs/                    # Comprehensive documentation
```

---

## Documentation

**[Documentation Hub](docs/README.md)** - Start here for comprehensive guides

### Quick Links

| Category | Document | Purpose |
|----------|----------|---------|
| **Getting Started** | [Quick Reference](docs/QUICK_REFERENCE.md) | Command cheatsheet |
| **Getting Started** | [Notebook Setup](docs/guides/NOTEBOOK_SETUP.md) | Jupyter/Colab guide |
| **Planning** | [Project Charter](docs/planning/PROJECT_CHARTER.md) | Project scope and status |
| **Architecture** | [Architecture](docs/ARCHITECTURE.md) | System design patterns |
| **Implementation** | [Phase 7 Stacking](docs/implementation/PHASE_7_META_LEARNER_STACKING.md) | Heterogeneous ensemble |
| **Guides** | [Model Integration](docs/guides/MODEL_INTEGRATION.md) | How to add new models |

---

## Configuration

```python
SYMBOL = 'MES'             # Contract symbol
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
