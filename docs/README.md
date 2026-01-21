# ML Model Factory Documentation

## Overview

Documentation hub for the **ML Model Factory** - a unified pipeline for training ML models on OHLCV time series data.

**Key Principle:** One canonical 1-min OHLCV dataset → Model-specific adapters → Per-model training

---

## Quick Start

| Resource | Description |
|----------|-------------|
| [Main README](../README.md) | Project overview and quick commands |
| [CLAUDE.md](../CLAUDE.md) | Complete system documentation |
| [Quickstart Guide](guides/QUICKSTART.md) | 5-minute getting started |

---

## Guides

| Guide | Purpose |
|-------|---------|
| [Quickstart](guides/QUICKSTART.md) | Get started in 5 minutes |
| [Model Integration](guides/MODEL_INTEGRATION.md) | Adding new models |
| [Meta-Learner Stacking](guides/META_LEARNER_STACKING.md) | Heterogeneous ensemble training |
| [Feature Engineering](guides/FEATURE_ENGINEERING.md) | Feature strategies |
| [Hyperparameter Tuning](guides/HYPERPARAMETER_TUNING.md) | Optuna tuning |
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter/Colab configuration |

---

## Reference

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete system architecture |
| [Models Reference](reference/MODELS.md) | All 23 models |
| [Pipeline Stages](reference/PIPELINE_STAGES.md) | Data flow details |
| [Infrastructure](reference/INFRASTRUCTURE.md) | Hardware requirements |

---

## Implementation Details

| Doc | Purpose |
|-----|---------|
| [Phase 1: Ingestion](implementation/PHASE_1_INGESTION.md) | Data loading and validation |
| [Phase 2: MTF Upscaling](implementation/PHASE_2_MTF_UPSCALING.md) | Multi-timeframe resampling |
| [Phase 3: Features](implementation/PHASE_3_FEATURES.md) | Feature engineering |
| [Phase 4: Labeling](implementation/PHASE_4_LABELING.md) | Triple-barrier labeling |
| [Phase 5: Adapters](implementation/PHASE_5_ADAPTERS.md) | Model-family data preparation |
| [Phase 6: Training](implementation/PHASE_6_TRAINING.md) | Model training |
| [Phase 7: Stacking](implementation/PHASE_7_META_LEARNER_STACKING.md) | Meta-learner ensembles |
| [Unified Training System](implementation/UNIFIED_TRAINING_SYSTEM.md) | Build-a-bear interface |
| [Unified Pipeline Architecture](implementation/UNIFIED_PIPELINE_ARCHITECTURE.md) | Pipeline design |

---

## Troubleshooting

| Doc | Purpose |
|-----|---------|
| [MTF Troubleshooting](troubleshooting/MTF_TROUBLESHOOTING.md) | Multi-timeframe issues |

---

## Project Planning

| Doc | Purpose |
|-----|---------|
| [Project Charter](planning/PROJECT_CHARTER.md) | Goals, scope, status |

---

## Current Implementation

| Component | Count |
|-----------|-------|
| **Models** | 23 |
| **MTF Timeframes** | 9 (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h) |
| **Ensemble Methods** | 3 (Voting, Stacking, Blending) |
| **Meta-Learners** | 4 (Ridge, MLP, Calibrated, XGBoost) |
| **Features** | ~180 |

**Models by Family:**
- **Tabular (6):** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM
- **Neural (10):** LSTM, GRU, TCN, Transformer, InceptionTime, ResNet1D, N-BEATS, PatchTST, iTransformer, TFT
- **Ensemble (7):** Voting, Stacking, Blending + 4 Meta-Learners

---

## Key Paths

| Path | Purpose |
|------|---------|
| `src/pipeline/` | Data pipeline stages |
| `src/models/` | Model registry and implementations |
| `src/training/` | Training infrastructure |
| `data/raw/` | Raw OHLCV data |
| `config/` | Configuration files |

---

*Last Updated: 2026-01-21*
