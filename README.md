# Universal ML Pipeline for OHLCV Time Series

**Train ANY model type on market data through a modular, phase-based architecture.**

```
[ Phase 1: Data ] → [ Phase 2: Models ] → [ Phase 3-4: Ensemble ] → [ Phase 5: Prod ] → [ Phase 6: UI ]
     ✅ DONE            IN DESIGN              PLANNED               PLANNED           FUTURE
```

## Quick Start

```bash
# Run Phase 1 pipeline
./pipeline run --symbols MES --stages all

# Use outputs for Phase 2
from src.stages.datasets import TimeSeriesDataContainer
container = TimeSeriesDataContainer.from_parquet_dir('data/splits/scaled', horizon=20)

# Get data in any format
X, y, w = container.get_sklearn_arrays('train')      # XGBoost/LightGBM
dataset = container.get_pytorch_sequences('train')   # LSTM/TCN/Transformer
nf_df = container.get_neuralforecast_df('train')     # N-HiTS/TFT/PatchTST
```

## Project Structure

```
.
├── phase1/              # Data preparation (COMPLETE)
│   ├── src/ → ../src    # Symlink to source
│   ├── tests/ → ../tests
│   └── README.md
│
├── src/                 # Phase 1 source code
│   ├── stages/          # Pipeline stages (ingest→validate)
│   ├── config/          # Configuration
│   └── pipeline/        # Orchestration
│
├── tests/               # Phase 1 tests
│   └── phase_1_tests/   # Unit & integration tests
│
├── data/                # All pipeline data
│   ├── raw/             # Source OHLCV data
│   ├── splits/scaled/   # Model-ready parquet files
│   └── ...
│
├── docs/                # Documentation
│   ├── phases/          # PHASE_1.md through PHASE_5.md
│   ├── getting-started/ # Quickstart, CLI reference
│   ├── reference/       # Architecture, features
│   └── archive/         # Historical docs
│
├── config/              # GA optimization results
└── notebooks/           # Jupyter notebooks
```

## Phase Overview

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| **1** | Data Pipeline | COMPLETE | Ingest → Clean → Features → Labels → Splits → Scale |
| **2** | Model Factory | IN DESIGN | Plugin architecture for ANY model (XGBoost, LSTM, TFT, etc.) |
| **3** | Cross-Validation | PLANNED | Purged k-fold, walk-forward, OOS predictions |
| **4** | Ensemble | PLANNED | Stacking meta-learner, regime-aware weighting |
| **5** | Production | PLANNED | Real-time inference, monitoring, A/B testing |
| **6** | Orchestrator | FUTURE | Central UI, model comparison dashboard |

## Supported Model Families (Phase 2)

| Family | Models | Status |
|--------|--------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost | Planned |
| **Time Series** | TCN, N-HiTS, TFT, PatchTST, TimesNet | Planned |
| **Neural** | LSTM, GRU, Transformer | Planned |
| **Foundation** | TimesFM 2.0, TimeLLM | Planned |
| **Classical** | RandomForest, SVM, LogisticRegression | Planned |

## Key Features

- **Model-Agnostic Data Serving** - One container, multiple output formats
- **Zero Leakage** - Purge (60 bars) + Embargo (288 bars) + train-only scaling
- **107 Technical Features** - Momentum, volatility, volume, multi-timeframe
- **GA-Optimized Labels** - Symbol-specific asymmetric barriers
- **Plugin Architecture** - Add new models with 4 methods + decorator

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Documentation hub |
| [docs/getting-started/QUICKSTART.md](docs/getting-started/QUICKSTART.md) | 15-min setup |
| [docs/phases/PHASE_1.md](docs/phases/PHASE_1.md) | Phase 1 specification |
| [docs/phases/PHASE_2.md](docs/phases/PHASE_2.md) | Phase 2 specification |
| [docs/reference/ARCHITECTURE.md](docs/reference/ARCHITECTURE.md) | System design |

## Engineering Principles

1. **Modularity** - No monoliths, clear phase separation
2. **650-line limit** - Forces good decomposition
3. **Fail fast** - Validate at boundaries
4. **Plugin architecture** - Add models without rewriting infrastructure
5. **Delete unused** - Git is the archive

---

**Phase 1:** COMPLETE (9.5/10) | **Next:** Build Phase 2 Model Factory
