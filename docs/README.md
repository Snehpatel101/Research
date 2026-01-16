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
| [Main README](../README.md) | Project overview and quick commands |
| [CLAUDE.md](../CLAUDE.md) | Instructions for AI assistants |
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

## Recent Improvements (2026-01-15)

| Doc | Purpose |
|-----|---------|
| [Config & P0 Improvements Summary](implementation/CONFIG_AND_P0_IMPROVEMENTS_SUMMARY.md) | Complete refactoring summary (13 tasks) |
| [Configuration Refactor](implementation/CONFIG_REFACTOR_SUMMARY_2026_01_15.md) | Centralized config system |
| [Lineage Tracking](implementation/LINEAGE_TRACKING_IMPLEMENTATION.md) | Dataset provenance validation |
| [Timestamp Alignment](implementation/TIMESTAMP_ALIGNMENT_IMPLEMENTATION.md) | Heterogeneous stacking validation |

## Project Planning

| Doc | Purpose |
|-----|---------|
| [Project Charter](planning/PROJECT_CHARTER.md) | Goals, scope, status |
| [Advanced Models Roadmap](implementation/ADVANCED_MODELS_ROADMAP.md) | Implementation history of 6 advanced neural models |

---

## Analysis & Research

| Doc | Purpose |
|-----|---------|
| [Feature Engineering Reality](analysis/PHASE1_FEATURE_ENGINEERING_REALITY.md) | Current feature analysis |
| [Implementation Tasks](analysis/IMPLEMENTATION_TASKS.md) | Development tasks |

---

## Archive

Historical and legacy documentation is preserved in [archive/implementation/](archive/implementation/README.md). These documents are for reference only and do not reflect the current implementation.

---

## Notes

Project notes and refactor summaries live in [notes/](notes/README.md).

---

## Current Implementation Summary

| Component | Status | Count |
|-----------|--------|-------|
| **Models Implemented** | ✅ Complete | 23 models (22 if CatBoost unavailable) |
| **MTF Stage 2** | ✅ Complete | 9 of 9 timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h) |
| **MTF Stages 3-6** | ✅ Complete | All stages support multi-TF via `--process-all-timeframes` |
| **Ensemble Methods** | ✅ Complete | 3 (Voting, Stacking, Blending) |
| **Meta-Learners** | ✅ Complete | 4 (Ridge, MLP, Calibrated, XGBoost) |
| **Features** | ✅ Complete | ~180 |

**Models by Family (6 Families, 23 Models):**

> **Note:** Run `python scripts/generate_model_inventory.py` to regenerate this list from the ModelRegistry.

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

*Last Updated: 2026-01-15*
