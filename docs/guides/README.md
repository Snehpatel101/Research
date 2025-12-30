# Implementation Guides

Comprehensive how-to guides for working with the ML model factory.

## 📚 Available Guides

### Core Implementation Guides

| Guide | Lines | Description |
|-------|-------|-------------|
| [Model Integration Guide](MODEL_INTEGRATION_GUIDE.md) | 1,251 | Complete guide for adding new models to the factory |
| [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md) | 1,298 | Model-specific feature engineering strategies |
| [Hyperparameter Optimization Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md) | 971 | GA for labels + Optuna for model hyperparameters |
| [Model Infrastructure Requirements](MODEL_INFRASTRUCTURE_REQUIREMENTS.md) | 1,080 | Hardware requirements and resource planning |

**Total:** 4,600 lines of detailed implementation guidance

---

## 🔧 Model Integration Guide

Learn how to add new models to the factory:

**Topics Covered:**
- ✅ BaseModel interface implementation
- ✅ Model registration with `@register` decorator
- ✅ Configuration management (YAML + dataclasses)
- ✅ Testing requirements and examples
- ✅ Integration checklists per model family
- ✅ Common pitfalls and solutions

**Who should read this:**
- ML engineers adding new models
- Researchers implementing custom architectures
- Contributors extending model families

---

## 🎨 Feature Engineering Guide

Model-specific feature engineering strategies:

**Topics Covered:**
- ✅ Tabular models: 150-200 MTF features recommended
- ✅ Sequence models: 25-30 minimal features recommended
- ✅ Feature selection methods (MDA, MDI, SHAP)
- ✅ Multi-timeframe alignment and leakage prevention
- ✅ Model family-specific feature requirements
- ✅ Code examples for each model type

**Who should read this:**
- Feature engineers designing model inputs
- Data scientists optimizing model performance
- Researchers exploring feature combinations

---

## 🎯 Hyperparameter Optimization Guide

Dual-stage optimization strategy:

**Topics Covered:**
- ✅ Stage 1: Genetic algorithms for label parameters
  - Triple-barrier thresholds
  - Horizon selection
  - Transaction cost-aware fitness
- ✅ Stage 2: Optuna for model hyperparameters
  - Search spaces for all 19 models
  - Purged K-Fold integration
  - Early stopping strategies
- ✅ Best practices and common patterns

**Who should read this:**
- ML engineers tuning model performance
- Researchers running hyperparameter sweeps
- Quants optimizing label definitions

---

## 🖥️ Model Infrastructure Requirements

Hardware planning and resource estimation:

**Topics Covered:**
- ✅ GPU memory estimation formulas
- ✅ CPU-only fallback strategies
- ✅ Training time benchmarks per model family
- ✅ Batch size optimization guidelines
- ✅ Production deployment configurations
- ✅ Cost analysis (cloud vs on-premise)

**Who should read this:**
- DevOps planning infrastructure
- ML engineers estimating resource needs
- Managers budgeting for compute resources

---

## 🔗 Quick Navigation

### By Model Family

**Boosting Models (XGBoost, LightGBM, CatBoost):**
1. [Model Integration Guide](MODEL_INTEGRATION_GUIDE.md#boosting-models) - Implementation patterns
2. [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md#tabular-models) - 150-200 features
3. [Hyperparameter Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md#boosting-search-spaces) - Search spaces
4. [Infrastructure Guide](MODEL_INFRASTRUCTURE_REQUIREMENTS.md#boosting-requirements) - CPU/GPU requirements

**Neural Models (LSTM, GRU, TCN, Transformer):**
1. [Model Integration Guide](MODEL_INTEGRATION_GUIDE.md#neural-models) - BaseRNNModel interface
2. [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md#sequence-models) - 25-30 minimal features
3. [Hyperparameter Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md#neural-search-spaces) - Architecture tuning
4. [Infrastructure Guide](MODEL_INFRASTRUCTURE_REQUIREMENTS.md#neural-requirements) - GPU memory formulas

**Classical Models (Random Forest, Logistic, SVM):**
1. [Model Integration Guide](MODEL_INTEGRATION_GUIDE.md#classical-models) - Scikit-learn integration
2. [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md#tabular-models) - Feature scaling
3. [Hyperparameter Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md#classical-search-spaces) - Tuning ranges
4. [Infrastructure Guide](MODEL_INFRASTRUCTURE_REQUIREMENTS.md#classical-requirements) - CPU-only specs

**Ensemble Models (Voting, Stacking, Blending):**
1. [Model Integration Guide](MODEL_INTEGRATION_GUIDE.md#ensemble-models) - Meta-learner patterns
2. [Feature Engineering Guide](FEATURE_ENGINEERING_GUIDE.md#ensemble-features) - OOF datasets
3. [Hyperparameter Guide](HYPERPARAMETER_OPTIMIZATION_GUIDE.md#ensemble-tuning) - Weight optimization
4. [Infrastructure Guide](MODEL_INFRASTRUCTURE_REQUIREMENTS.md#ensemble-requirements) - Combined resources

---

## 📖 Related Documentation

- [Advanced Models Roadmap](../roadmaps/ADVANCED_MODELS_ROADMAP.md) - 6 new models implementation plan
- [MTF Implementation Roadmap](../roadmaps/MTF_IMPLEMENTATION_ROADMAP.md) - Multi-timeframe expansion
- [Project Charter](../planning/PROJECT_CHARTER.md) - Current status: 13 implemented + 6 planned
- [Phase 2 Documentation](../phases/PHASE_2.md) - Model training details

---

*See [Documentation Index](../INDEX.md) for complete documentation overview*
