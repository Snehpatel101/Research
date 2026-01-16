# ML Model Factory Documentation

## Overview

This is the documentation hub for the **ML Model Factory** - a single-pipeline architecture for training, evaluating, and deploying machine learning models on OHLCV time series data.

**Key Principle:** One canonical 1-min OHLCV dataset → Model-specific adapters → Per-model training

The factory processes one futures contract at a time through a unified 7-phase pipeline, producing trained models with standardized artifacts and performance reports.

---

## Quick Start

**New here?** Start with these:

| Resource | Description |
|----------|-------------|
| [Main README](../README.md) | Project overview and quick commands |
| [CLAUDE.md](../CLAUDE.md) | Complete system documentation |
| [Quickstart Guide](guides/QUICKSTART.md) | 5-minute getting started |
| [Notebook Setup](guides/NOTEBOOK_SETUP.md) | Jupyter and Colab setup |

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

## SNwH Implementation Plan (NEW)

**Unified Multi-Timeframe Model Factory** - Complete implementation plan to make every model work with every timeframe and enable heterogeneous ensembles by default.

| Doc | Purpose |
|-----|---------|
| [Z/README.md](../Z/README.md) | **Start here** - Overview and quick start |
| [Z/00_INDEX.md](../Z/00_INDEX.md) | Master index with roadmap |
| [Z/SNWH_ARCHITECTURE_SYNTHESIS.md](../Z/SNWH_ARCHITECTURE_SYNTHESIS.md) | Gap analysis, dependency graph |
| [Z/SNWH_IMPLEMENTATION_PHASE_0.md](../Z/SNWH_IMPLEMENTATION_PHASE_0.md) | Canonical Contracts |
| [Z/SNWH_IMPLEMENTATION_PHASE_1.md](../Z/SNWH_IMPLEMENTATION_PHASE_1.md) | Configuration Layer |
| [Z/SNWH_IMPLEMENTATION_PHASE_2.md](../Z/SNWH_IMPLEMENTATION_PHASE_2.md) | Adapter Architecture |
| [Z/SNWH_IMPLEMENTATION_PHASE_3.md](../Z/SNWH_IMPLEMENTATION_PHASE_3.md) | Timeframe Coordination |
| [Z/SNWH_IMPLEMENTATION_PHASE_4.md](../Z/SNWH_IMPLEMENTATION_PHASE_4.md) | OOF Integrity |
| [Z/SNWH_IMPLEMENTATION_PHASE_5.md](../Z/SNWH_IMPLEMENTATION_PHASE_5.md) | Feature Strategy |
| [Z/SNWH_TESTING_STRATEGY.md](../Z/SNWH_TESTING_STRATEGY.md) | Testing (28 files, 305 methods) |
| [Z/SNWH_IMPLEMENTATION_PHASE_6.md](../Z/SNWH_IMPLEMENTATION_PHASE_6.md) | Single Config System (89% code reduction) |

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
| [Advanced Models Roadmap](implementation/ADVANCED_MODELS_ROADMAP.md) | Neural model history |

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

## Archive

Historical and legacy documentation is preserved in [archive/](archive/README.md). These documents are for reference only and do not reflect the current implementation.

---

## Current Implementation Summary

| Component | Status | Count |
|-----------|--------|-------|
| **Models** | ✅ Complete | 23 (22 if CatBoost unavailable) |
| **MTF Timeframes** | ✅ Complete | 9 (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h) |
| **Ensemble Methods** | ✅ Complete | 3 (Voting, Stacking, Blending) |
| **Meta-Learners** | ✅ Complete | 4 (Ridge, MLP, Calibrated, XGBoost) |
| **Features** | ✅ Complete | ~180 |

**Models by Family:**

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

*Last Updated: 2026-01-16*
