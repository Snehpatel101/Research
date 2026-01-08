# TOPSTEPX ML Model Factory Documentation

---

## Overview

This is the documentation hub for the **ML Model Factory** - a single-pipeline architecture for training, evaluating, and deploying machine learning models on OHLCV time series data.

**Key Principle:** One canonical dataset -> Deterministic adapters -> Model-specific training

The factory processes one futures contract at a time through a unified 7-phase pipeline, producing trained models with standardized artifacts and performance reports.

---

## Quick Start

**New here?** Start with these:

| Resource | Description |
|----------|-------------|
| [Quick Reference](QUICK_REFERENCE.md) | Command cheatsheet for common tasks |
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter/Colab setup |

---

## Architecture

The factory implements a single-pipeline architecture with 7 implementation phases:

```
Raw OHLCV -> [MTF Upscaling] -> [Features] -> [Labels] -> [Adapters]
                                                              |
                                              +---------------+---------------+
                                              |               |               |
                                          Tabular(2D)    Sequence(3D)    Multi-Res(4D)
                                              |               |               |
                                          Boosting       Neural          Advanced
                                          Classical      CNN/MLP         Transformers
                                              |               |               |
                                              +-------+-------+---------------+
                                                      |
                                              [Ensembles] -> [Meta-Learners]
```

**Comprehensive Reference:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Implementation Phases

| Phase | Name | Description | Status | Doc |
|:-----:|------|-------------|:------:|-----|
| 1 | Ingestion | Load and validate raw OHLCV data | ✅ Complete | [PHASE_1_INGESTION.md](implementation/PHASE_1_INGESTION.md) |
| 2 | MTF Upscaling | Multi-timeframe resampling (9 TFs) | ✅ Complete | [PHASE_2_MTF_UPSCALING.md](implementation/PHASE_2_MTF_UPSCALING.md) |
| 3 | Features | 180+ indicator features | ✅ Complete | [PHASE_3_FEATURES.md](implementation/PHASE_3_FEATURES.md) |
| 4 | Labeling | Triple-barrier + Optuna optimization | ✅ Complete | [PHASE_4_LABELING.md](implementation/PHASE_4_LABELING.md) |
| 5 | Adapters | Model-family data adapters | ✅ Complete | [PHASE_5_ADAPTERS.md](implementation/PHASE_5_ADAPTERS.md) |
| 6 | Training | 23 models across 4 families | ✅ Complete | [PHASE_6_TRAINING.md](implementation/PHASE_6_TRAINING.md) |
| 7 | Stacking | Heterogeneous ensemble training | ✅ Complete | [PHASE_7_META_LEARNER_STACKING.md](implementation/PHASE_7_META_LEARNER_STACKING.md) |

---

## User Guides

### Getting Started

| Guide | Purpose |
|-------|---------|
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter and Colab setup |
| [Quick Reference](QUICK_REFERENCE.md) | Command cheatsheet |

### Core Guides

| Guide | Purpose |
|-------|---------|
| [Model Integration](guides/MODEL_INTEGRATION.md) | Adding new models |
| [Meta-Learner Stacking](guides/META_LEARNER_STACKING.md) | Heterogeneous ensemble training |
| [Feature Engineering](guides/FEATURE_ENGINEERING.md) | Feature strategies |
| [Hyperparameter Tuning](guides/HYPERPARAMETER_TUNING.md) | Optuna tuning |

### Infrastructure

| Guide | Purpose |
|-------|---------|
| [Infrastructure Requirements](reference/INFRASTRUCTURE.md) | Hardware requirements |

---

## Reference Documentation

### Core Reference

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete system architecture |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command cheatsheet |
| [Models Reference](reference/MODELS.md) | All 23 models (22 if CatBoost unavailable) |

### Technical Reference

| Doc | Purpose |
|-----|---------|
| [Pipeline Stages](reference/PIPELINE_STAGES.md) | Data flow details |

---

## Troubleshooting

| Doc | Purpose |
|-----|---------|
| [MTF Troubleshooting](troubleshooting/MTF_TROUBLESHOOTING.md) | MTF-specific issues |

---

## Project Planning

| Doc | Purpose |
|-----|---------|
| [Project Charter](planning/PROJECT_CHARTER.md) | Goals, scope, status |
| [Advanced Models Roadmap](implementation/ADVANCED_MODELS_ROADMAP.md) | 6 planned models |
| [MTF Implementation Roadmap](implementation/MTF_IMPLEMENTATION_ROADMAP.md) | 9-timeframe ladder |

---

## Analysis & Research

| Doc | Purpose |
|-----|---------|
| [Feature Engineering Reality](analysis/PHASE1_FEATURE_ENGINEERING_REALITY.md) | Current feature analysis |
| [Implementation Tasks](analysis/IMPLEMENTATION_TASKS.md) | Development tasks |

---

## Archive

Historical and legacy documentation is preserved in [archive/implementation/](archive/implementation/). These documents are for reference only and do not reflect the current implementation.

---

## Current Implementation Summary

| Component | Status | Count |
|-----------|--------|-------|
| **Models Implemented** | ✅ Complete | 23 of 23 (22 if CatBoost unavailable) |
| **MTF Timeframes** | ✅ Complete | 9 of 9 (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h) |
| **Ensemble Methods** | ✅ Complete | 3 (Voting, Stacking, Blending) |
| **Meta-Learners** | ✅ Complete | 4 (Ridge, MLP, Calibrated, XGBoost) |
| **Features** | ✅ Complete | ~180 |

**Models by Family (4 Families, 23 Models):**
- **Tabular (6):** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM
- **Neural (10):** LSTM, GRU, TCN, Transformer, InceptionTime, ResNet1D, N-BEATS, PatchTST, iTransformer, TFT
- **Ensemble (3):** Voting, Stacking, Blending
- **Meta-Learners (4):** Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta

---

## Pipeline Paths

| Path | Purpose |
|------|---------|
| `data/raw/` | Raw OHLCV data (e.g., `MES_1m.parquet`) |
| `data/splits/scaled/` | Processed train/val/test splits |
| `experiments/runs/{run_id}/` | Training artifacts and models |
| `config/models/` | Model configuration files |

---

*Last Updated: 2026-01-08*
