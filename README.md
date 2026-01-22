# ML Model Factory for OHLCV Time Series

Ensemble ML pipeline for financial price prediction using triple-barrier labeling and a plugin-based model factory architecture.

## Overview

A **model factory** for training, evaluating, and comparing ML models on OHLCV bar data. Supports **23 models** across 6 families with a unified pipeline architecture.

### Key Features

- **Plugin-Based Model Registry**: Add new model types without rewriting pipelines
- **23 Models Across 6 Families**: Boosting, Neural, Classical, Advanced Transformers, Ensemble, Meta-Learners
- **Triple-Barrier Labeling**: ATR-based dynamic barriers with Optuna optimization
- **Multi-Timeframe Support**: 9-timeframe ladder (1m → 1h)
- **Leakage Prevention**: Purge/embargo, train-only scaling, OOF for stacking
- **Single-Contract Architecture**: Complete isolation per futures contract

## Installation

```bash
pip install -e .
# Or with dev tools: pip install -e ".[dev]"
```

Requires Python 3.11+

### Developer Setup

Install development dependencies and pre-commit hooks:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (runs automatically on git commit)
pre-commit install

# Run pre-commit manually on all files
pre-commit run --all-files

# Run pre-commit on specific files
pre-commit run --files src/some_file.py
```

Pre-commit hooks will automatically:
- Format code with ruff
- Check for common issues with ruff linting
- Run type checking with mypy (src/ only)
- Run quick tests to catch breaking changes

## Quick Start

```bash
# Run data pipeline
pipeline run --symbols MES --start 2020-01-01 --end 2024-12-31

# Train model
python scripts/train_model.py --model xgboost --horizon 20

# List available models
python scripts/train_model.py --list-models
```

## Model Families

| Family | Models | Format |
|--------|--------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 2D |
| **Neural** | LSTM, GRU, TCN, Transformer | 3D |
| **Classical** | Random Forest, Logistic, SVM | 2D |
| **CNN** | InceptionTime, ResNet1D | 3D/4D |
| **Advanced** | PatchTST, iTransformer, TFT, N-BEATS | 4D |
| **Ensemble** | Voting, Stacking, Blending + 4 Meta-learners | OOF |

## Project Structure

```
src/
├── pipeline/          # Data pipeline (ingestion → features → labels → splits)
├── models/            # Model registry and implementations
├── training/          # Training infrastructure
├── adapters/          # Data format adapters (2D/3D/4D)
├── cross_validation/  # PurgedKFold, OOF generation
└── factory.py         # MLFactory entry point

config/                # Configuration files
data/                  # Raw and processed data
docs/                  # Documentation
```

## Configuration

All defaults in `config/global.yaml`:

```yaml
random_seed: 42
train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
purge_bars: 60
embargo_bars: 1440
label_horizons: [5, 10, 15, 20]
```

## Documentation

- [Documentation Hub](docs/README.md) - Complete index
- [Architecture](docs/ARCHITECTURE.md) - System design
- [CLAUDE.md](CLAUDE.md) - AI assistant context

## License

MIT License
