# Documentation Index

Complete guide to the ML Model Factory for OHLCV Time Series.

## 🎯 Start Here

| Document | Description |
|----------|-------------|
| [Project Charter](planning/PROJECT_CHARTER.md) | Project overview, goals, and current status (13 implemented + 6 planned models) |
| [Quick Reference](QUICK_REFERENCE.md) | Command cheatsheet for common operations |
| [Getting Started Guide](getting-started/QUICKSTART.md) | Step-by-step setup and first pipeline run |

## 📋 Planning & Architecture

### Project Planning
| Document | Description |
|----------|-------------|
| [Project Charter](planning/PROJECT_CHARTER.md) | Official project status: 13 models implemented, 6 planned |
| [Alignment Plan](planning/ALIGNMENT_PLAN.md) | Repository alignment strategy and goals |
| [Repository Organization Analysis](planning/REPO_ORGANIZATION_ANALYSIS.md) | Current vs target state analysis |

### Architecture Reference
| Document | Description |
|----------|-------------|
| [Architecture Overview](reference/ARCHITECTURE.md) | System architecture and design patterns |
| [Pipeline Flow](reference/PIPELINE_FLOW.md) | Detailed pipeline execution flow (Phase 1-5) |
| [Feature Reference](reference/FEATURES.md) | Complete feature catalog (150+ indicators) |

## 🛣️ Implementation Roadmaps

### Model Expansion
| Document | Effort | Description |
|----------|--------|-------------|
| [Advanced Models Roadmap](roadmaps/ADVANCED_MODELS_ROADMAP.md) | 18 days | Implementation guide for 6 advanced time-series models (InceptionTime, ResNet, PatchTST, iTransformer, TFT, N-BEATS) |
| [MTF Implementation Roadmap](roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) | 6 phases | Multi-timeframe feature expansion plan |

**Roadmap Summary:**
- **Model 14:** InceptionTime (multi-scale CNN) - 3 days
- **Model 15:** 1D ResNet (deep residual learning) - 2 days
- **Model 16:** PatchTST (patch-based transformer) - 4 days
- **Model 17:** iTransformer (inverted attention) - 3 days
- **Model 18:** TFT (temporal fusion + quantiles) - 5 days
- **Model 19:** N-BEATS (interpretable decomposition) - 1 day
- **Total:** 18 days (3.5 weeks @ 1 engineer, 2 weeks @ 2 engineers)

## 📖 Implementation Guides

### Core Guides
| Document | Lines | Description |
|----------|-------|-------------|
| [Model Integration Guide](guides/MODEL_INTEGRATION_GUIDE.md) | 1,251 | How to add new models to the factory (BaseModel interface, registration, testing) |
| [Feature Engineering Guide](guides/FEATURE_ENGINEERING_GUIDE.md) | 1,298 | Model-specific feature strategies (tabular: 150-200 features, sequence: 25-30 features) |
| [Hyperparameter Optimization Guide](guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) | 971 | GA for labels + Optuna for model hyperparameters |
| [Model Infrastructure Requirements](guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) | 1,080 | Hardware requirements, GPU memory estimation, training benchmarks |

### Specialized Topics
| Document | Description |
|----------|-------------|
| [Quantitative Trading Analysis](QUANTITATIVE_TRADING_ANALYSIS.md) | Triple-barrier labeling, risk metrics, backtesting |
| [Advanced Feature Selection](research/ADVANCED_FEATURE_SELECTION.md) | MDA, MDI, SHAP feature selection methods |
| [Best OHLCV Features](research/BEST_OHLCV_FEATURES.md) | Research-backed feature recommendations |
| [Best OHLCV Models 2025](research/BEST_OHLCV_MODELS_2025.md) | State-of-the-art model architectures for financial time series |

## 🏗️ Phase Documentation

| Phase | Document | Status | Description |
|-------|----------|--------|-------------|
| **Phase 1** | [Phase 1: Data Pipeline](phases/PHASE_1.md) | ✅ Complete | 14-stage data pipeline (ingest → datasets) |
| **Phase 2** | [Phase 2: Model Training](phases/PHASE_2.md) | ✅ Complete | 13 models across 4 families (boosting, neural, classical, ensemble) |
| **Phase 3** | [Phase 3: Cross-Validation](phases/PHASE_3.md) | ✅ Complete | Purged K-Fold, OOF generation, hyperparameter tuning |
| **Phase 4** | [Phase 4: Ensemble Methods](phases/PHASE_4.md) | ✅ Complete | Voting, stacking, blending ensembles |
| **Phase 5** | [Phase 5: Walk-Forward & CPCV](phases/PHASE_5.md) | ✅ Complete | Walk-forward validation, combinatorial purged cross-validation |

## 🚀 Getting Started

### For New Users
1. Read [Project Charter](planning/PROJECT_CHARTER.md) - Understand project goals and status
2. Read [Quick Reference](QUICK_REFERENCE.md) - Learn common commands
3. Follow [Quickstart Guide](getting-started/QUICKSTART.md) - Run your first pipeline
4. Review [Pipeline CLI](getting-started/PIPELINE_CLI.md) - Master the command-line interface

### For Model Developers
1. Read [Model Integration Guide](guides/MODEL_INTEGRATION_GUIDE.md) - Learn the BaseModel interface
2. Review [Advanced Models Roadmap](roadmaps/ADVANCED_MODELS_ROADMAP.md) - See planned implementations
3. Study [Feature Engineering Guide](guides/FEATURE_ENGINEERING_GUIDE.md) - Understand model-specific features
4. Check [Model Infrastructure Requirements](guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) - Plan hardware needs

### For Feature Engineers
1. Read [Feature Engineering Guide](guides/FEATURE_ENGINEERING_GUIDE.md) - Model-specific strategies
2. Review [Best OHLCV Features](research/BEST_OHLCV_FEATURES.md) - Research-backed features
3. Study [MTF Implementation Roadmap](roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) - Multi-timeframe expansion
4. Check [Advanced Feature Selection](research/ADVANCED_FEATURE_SELECTION.md) - Selection methods

## 🧪 Validation & Testing

| Document | Description |
|----------|-------------|
| [Validation Checklist](VALIDATION_CHECKLIST.md) | Pre-deployment validation steps |
| [Workflow Best Practices](WORKFLOW_BEST_PRACTICES.md) | Development workflow and testing patterns |

## 📊 Reports & Analysis

| Document | Description |
|----------|-------------|
| [ML Pipeline Audit Report](reports/ML_PIPELINE_AUDIT_REPORT.md) | Comprehensive pipeline analysis and findings |

## 🛠️ Technical Reference

### Pipeline Internals
| Document | Description |
|----------|-------------|
| [Pipeline Fixes](reference/PIPELINE_FIXES.md) | Known issues and fixes |
| [Slippage Modeling](reference/SLIPPAGE.md) | Transaction cost modeling |

### Notebook Development
| Document | Description |
|----------|-------------|
| [Notebook README](notebook/README.md) | Jupyter notebook development guide |
| [Colab Setup](notebook/COLAB_SETUP.md) | Google Colab configuration |
| [Cell Reference](notebook/CELL_REFERENCE.md) | Standard notebook cells |
| [Troubleshooting](notebook/TROUBLESHOOTING.md) | Common notebook issues |

## 📚 Research

| Document | Description |
|----------|-------------|
| [Best OHLCV Models 2025](research/BEST_OHLCV_MODELS_2025.md) | State-of-the-art model survey |
| [Feature Requirements by Model](research/FEATURE_REQUIREMENTS_BY_MODEL.md) | Model-specific feature needs |
| [Feature Selection Methods](research/FEATURE_SELECTION_METHODS.md) | MDA, MDI, SHAP comparison |

## 🗂️ Archive

Legacy documentation preserved for reference:
- [Archive Directory](archive/) - Outdated phase docs and references

---

## Quick Navigation

### By Role
- **Data Scientist:** [Quickstart](getting-started/QUICKSTART.md) → [Phase 1](phases/PHASE_1.md) → [Pipeline CLI](getting-started/PIPELINE_CLI.md)
- **ML Engineer:** [Model Integration](guides/MODEL_INTEGRATION_GUIDE.md) → [Advanced Models](roadmaps/ADVANCED_MODELS_ROADMAP.md) → [Phase 2](phases/PHASE_2.md)
- **Feature Engineer:** [Feature Engineering](guides/FEATURE_ENGINEERING_GUIDE.md) → [MTF Roadmap](roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) → [Best Features](research/BEST_OHLCV_FEATURES.md)
- **DevOps:** [Infrastructure Requirements](guides/MODEL_INFRASTRUCTURE_REQUIREMENTS.md) → [Architecture](reference/ARCHITECTURE.md) → [Validation](VALIDATION_CHECKLIST.md)

### By Task
- **Run Pipeline:** [Quickstart](getting-started/QUICKSTART.md) → [Pipeline CLI](getting-started/PIPELINE_CLI.md)
- **Train Model:** [Model Integration](guides/MODEL_INTEGRATION_GUIDE.md) → [Phase 2](phases/PHASE_2.md)
- **Add Features:** [Feature Engineering](guides/FEATURE_ENGINEERING_GUIDE.md) → [MTF Roadmap](roadmaps/MTF_IMPLEMENTATION_ROADMAP.md)
- **Optimize Hyperparameters:** [Hyperparameter Guide](guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) → [Phase 3](phases/PHASE_3.md)
- **Build Ensemble:** [Phase 4](phases/PHASE_4.md) → [Model Integration](guides/MODEL_INTEGRATION_GUIDE.md)
- **Add New Model:** [Advanced Models Roadmap](roadmaps/ADVANCED_MODELS_ROADMAP.md) → [Model Integration](guides/MODEL_INTEGRATION_GUIDE.md)

---

## Documentation Statistics

- **Total Guides:** 5,617 lines across 4 core implementation guides
- **Roadmaps:** 2 comprehensive roadmaps (18 days implementation timeline)
- **Phase Docs:** 5 phases (Phase 1-5 all complete)
- **Research Papers:** 5 research documents
- **Getting Started:** 2 quickstart guides
- **Reference Docs:** 5 technical references

---

*Last Updated: 2025-12-30*
*Documentation Version: 2.0*
