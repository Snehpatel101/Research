# ML Model Factory for OHLCV Time Series

Ensemble ML pipeline for financial price prediction using triple-barrier labeling and a plugin-based model factory architecture.

## Overview

This project implements a **model factory** for training, evaluating, and comparing ML models on OHLCV bar data. The factory supports 22 models across 6 families with a unified pipeline architecture.

### Key Features

- **Plugin-Based Model Registry**: Add new model types without rewriting pipelines
- **22 Models Across 6 Families**: Boosting, Neural, Classical, CNN, Transformers, Ensemble
- **Triple-Barrier Labeling**: ATR-based dynamic barriers with Optuna optimization
- **Multi-Timeframe Support**: 9-timeframe ladder (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- **Leakage Prevention**: 5 layers including purge/embargo, train-only scaling, OOF for stacking
- **Single-Contract Architecture**: Complete isolation per futures contract

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd research

# Install with pip (requires Python 3.11+)
pip install -e .

# Or with optional dependencies
pip install -e ".[dev]"  # Development tools
pip install -e ".[all]"  # All optional dependencies
```

## Quick Start

### 1. Run Data Pipeline (Phase 1)

```bash
# Basic usage with single symbol
pipeline run --symbols MES --start 2020-01-01 --end 2024-12-31

# With preset configuration
pipeline run --preset day_trading --symbols MES

# Custom horizons and barriers
pipeline run --symbols MES --horizons 5,10,20 --k-up 1.5 --k-down 1.0
```

### 2. Train Models (Phase 2)

```bash
# Train boosting model
python scripts/train_model.py --model xgboost --horizon 20

# Train neural model
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Train ensemble
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20

# List available models
python scripts/train_model.py --list-models
```

### 3. Run Cross-Validation (Phase 3)

```bash
# Single model CV
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Multiple models with tuning
python scripts/run_cv.py --models all --horizons 5,10,15,20 --tune
```

## Model Families

| Family | Models | Data Format |
|--------|--------|-------------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 2D tabular |
| **Neural** | LSTM, GRU, TCN, Transformer | 3D sequences |
| **Classical** | Random Forest, Logistic, SVM | 2D tabular |
| **CNN** | InceptionTime, ResNet1D | 3D/4D sequences |
| **Advanced** | PatchTST, iTransformer, TFT, N-BEATS | 4D multi-res |
| **Ensemble** | Voting, Stacking, Blending + 4 Meta-learners | OOF predictions |

## Project Structure

```
research/
├── src/
│   ├── phase1/           # Data pipeline stages
│   │   └── stages/       # Ingest, clean, features, labeling, scaling
│   ├── models/           # Model factory
│   │   ├── registry.py   # Plugin registration
│   │   ├── boosting/     # XGBoost, LightGBM, CatBoost
│   │   ├── neural/       # LSTM, GRU, TCN, Transformer, etc.
│   │   └── ensemble/     # Voting, Stacking, Meta-learners
│   └── cross_validation/ # PurgedKFold, OOF generation
├── scripts/              # CLI entry points
├── docs/                 # Architecture and implementation docs
├── tests/                # Test suite
└── data/                 # Data directory (not tracked)
```

## Configuration

Key parameters can be set via CLI or config files:

```python
# Default parameters
LABEL_HORIZONS = [5, 10, 15, 20]
TRAIN/VAL/TEST = 70/15/15
PURGE_BARS = 60      # Prevents label leakage
EMBARGO_BARS = 1440  # ~5 days at 5-min
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Phase Implementation Guides](docs/phases/)
- [Advanced Models Roadmap](docs/roadmaps/ADVANCED_MODELS_ROADMAP.md)

## License

MIT License - see [pyproject.toml](pyproject.toml) for details.
