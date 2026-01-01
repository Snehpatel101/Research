# TOPSTEPX ML Model Factory Documentation

---

## Overview

This is the documentation hub for the **ML Model Factory** - a single-pipeline architecture for training, evaluating, and deploying machine learning models on OHLCV time series data.

**Key Principle:** One canonical dataset -> Deterministic adapters -> Model-specific training

The factory processes one futures contract at a time through a unified 8-phase pipeline, producing trained models with standardized artifacts and performance reports.

---

## Quick Start

**New here?** Start with these:

| Resource | Description |
|----------|-------------|
| [Quick Reference](QUICK_REFERENCE.md) | Command cheatsheet for common tasks |
| [Quickstart Guide](getting-started/QUICKSTART.md) | First pipeline run tutorial |
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter/Colab setup |

---

## Architecture

The factory implements a single-pipeline architecture with 8 implementation phases:

```
Raw OHLCV -> [MTF Upscaling] -> [Features] -> [Labels] -> [Adapters]
                                                              |
                                              +---------------+---------------+
                                              |               |               |
                                          Tabular(2D)    Sequence(3D)    Multi-Res(4D)
                                              |               |               |
                                          Boosting       Neural          Advanced
                                          Classical      TCN/Trans       (Planned)
                                              |               |
                                              +-------+-------+
                                                      |
                                              [Ensembles] -> [Meta-Learners]
```

**Comprehensive Reference:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Implementation Phases

| Phase | Name | Description | Status | Doc |
|:-----:|------|-------------|:------:|-----|
| 1 | Ingestion | Load and validate raw OHLCV data | Complete | [PHASE_1_INGESTION.md](implementation/PHASE_1_INGESTION.md) |
| 2 | MTF Upscaling | Multi-timeframe resampling | Partial (5/9 TFs) | [PHASE_2_MTF_UPSCALING.md](implementation/PHASE_2_MTF_UPSCALING.md) |
| 3 | Features | 180+ indicator features | Complete | [PHASE_3_FEATURES.md](implementation/PHASE_3_FEATURES.md) |
| 4 | Labeling | Triple-barrier + Optuna optimization | Complete | [PHASE_4_LABELING.md](implementation/PHASE_4_LABELING.md) |
| 5 | Adapters | Model-family data adapters | Complete | [PHASE_5_ADAPTERS.md](implementation/PHASE_5_ADAPTERS.md) |
| 6 | Training | 13 models across 4 families | Complete | [PHASE_6_TRAINING.md](implementation/PHASE_6_TRAINING.md) |
| 7 | Ensembles | Voting, Stacking, Blending | Complete | [PHASE_7_ENSEMBLES.md](implementation/PHASE_7_ENSEMBLES.md) |
| 8 | Meta-Learners | Regime-aware, adaptive | Planned | [PHASE_8_META_LEARNERS.md](implementation/PHASE_8_META_LEARNERS.md) |

---

## User Guides

### Getting Started

| Guide | Purpose |
|-------|---------|
| [Quickstart](getting-started/QUICKSTART.md) | First pipeline run |
| [Pipeline CLI](getting-started/PIPELINE_CLI.md) | CLI options reference |
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter and Colab setup |

### Core Guides

| Guide | Purpose |
|-------|---------|
| [Model Integration](guides/MODEL_INTEGRATION.md) | Adding new models |
| [Ensemble Configuration](guides/ENSEMBLE_CONFIGURATION.md) | Ensemble methods and configs |
| [Feature Engineering](guides/FEATURE_ENGINEERING.md) | Feature strategies |
| [Hyperparameter Tuning](guides/HYPERPARAMETER_TUNING.md) | Optuna tuning |

### Infrastructure

| Guide | Purpose |
|-------|---------|
| [Infrastructure Requirements](reference/INFRASTRUCTURE.md) | Hardware requirements |
| [Notebook Configuration](notebook/CONFIGURATION.md) | Notebook parameters |

---

## Reference Documentation

### Core Reference

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete system architecture |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command cheatsheet |
| [Models Reference](reference/MODELS.md) | All 19 models (13 implemented, 6 planned) |
| [Features Reference](reference/FEATURES.md) | 180+ feature catalog |

### Technical Reference

| Doc | Purpose |
|-----|---------|
| [Pipeline Stages](reference/PIPELINE_STAGES.md) | Data flow details |
| [Slippage](reference/SLIPPAGE.md) | Transaction cost modeling |

---

## Troubleshooting

| Doc | Purpose |
|-----|---------|
| [MTF Troubleshooting](troubleshooting/MTF_TROUBLESHOOTING.md) | MTF-specific issues |
| [Notebook Troubleshooting](notebook/TROUBLESHOOTING.md) | Colab/Jupyter issues |

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

Historical and legacy documentation is preserved in [archive/](archive/). These documents are for reference only and do not reflect the current implementation.

| Archive | Contents |
|---------|----------|
| [Phases](archive/phases/) | Legacy phase docs |
| [Reference](archive/reference/) | Outdated reference docs |
| [Research](archive/research/) | Historical research notes |
| [Roadmaps](archive/roadmaps/) | Completed or superseded roadmaps |

---

## Current Implementation Summary

| Component | Status | Count |
|-----------|--------|-------|
| **Models Implemented** | Complete | 13 of 19 |
| **MTF Timeframes** | Partial | 5 of 9 |
| **Ensemble Methods** | Complete | 3 (Voting, Stacking, Blending) |
| **Features** | Complete | ~180 |

**Models by Family:**
- Boosting (3): XGBoost, LightGBM, CatBoost
- Neural (4): LSTM, GRU, TCN, Transformer
- Classical (3): Random Forest, Logistic, SVM
- Ensemble (3): Voting, Stacking, Blending
- Planned (6): InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS

---

## Pipeline Paths

| Path | Purpose |
|------|---------|
| `data/raw/` | Raw OHLCV data (e.g., `MES_1m.parquet`) |
| `data/splits/scaled/` | Processed train/val/test splits |
| `experiments/runs/{run_id}/` | Training artifacts and models |
| `config/models/` | Model configuration files |

---

*Last Updated: 2026-01-01*
