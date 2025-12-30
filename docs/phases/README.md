# Phase Documentation

Implementation documentation for all 5 completed phases.

## Phases Overview

| Phase | Status | Description | Duration | Output |
|-------|--------|-------------|----------|--------|
| [Phase 1](PHASE_1.md) | ✅ Complete | Data pipeline (14 stages) | 2-5 min | Standardized datasets |
| [Phase 2](PHASE_2.md) | ✅ Complete | Model training (13 models) | 10-90 min | Trained models |
| [Phase 3](PHASE_3.md) | ✅ Complete | Cross-validation (purged K-fold) | 1-6 hours | CV results, OOF predictions |
| [Phase 4](PHASE_4.md) | ✅ Complete | Ensemble methods (3 types) | 45-180 min | Ensemble models |
| [Phase 5](PHASE_5.md) | ✅ Complete | Walk-forward & CPCV | 6-12 hours | Robust validation |

## Phase 1: Data Pipeline

**Input:** Raw OHLCV data (1-minute bars)
**Output:** Standardized datasets for all models
**Runtime:** 2-5 minutes per symbol

**Key Stages:**
1. Ingest & validate
2. Clean & resample (1min → 5min)
3. Session filtering
4. Feature engineering (150+ indicators)
5. Regime detection
6. Multi-timeframe features
7. Triple-barrier labeling
8. GA optimization
9. Final labels
10. Train/val/test splits
11. Robust scaling
12. Dataset creation
13. Validation
14. Reporting

**Documentation:** [PHASE_1.md](PHASE_1.md)

## Phase 2: Model Training

**Input:** Standardized datasets from Phase 1
**Output:** Trained models + performance reports
**Runtime:** 10-90 minutes per model

**Implemented Models:**
- **Boosting (3):** XGBoost, LightGBM, CatBoost
- **Neural (4):** LSTM, GRU, TCN, Transformer
- **Classical (3):** Random Forest, Logistic, SVM
- **Ensemble (3):** Voting, Stacking, Blending

**Documentation:** [PHASE_2.md](PHASE_2.md)

## Phase 3: Cross-Validation

**Input:** Trained models from Phase 2
**Output:** CV results, OOF predictions, hyperparameter tuning
**Runtime:** 1-6 hours depending on n_splits and tuning

**Features:**
- Time-series aware CV (expanding window)
- Purged K-Fold (no leakage)
- OOF prediction generation
- Optuna hyperparameter tuning
- Walk-forward feature selection

**Documentation:** [PHASE_3.md](PHASE_3.md)

## Phase 4: Ensemble Methods

**Input:** OOF predictions from Phase 3
**Output:** Ensemble models combining multiple base models
**Runtime:** 45-180 minutes per ensemble

**Methods:**
- **Voting:** Probability averaging
- **Stacking:** Meta-learner on OOF predictions
- **Blending:** Meta-learner on holdout predictions

**Documentation:** [PHASE_4.md](PHASE_4.md)

## Phase 5: Walk-Forward & CPCV

**Input:** Models from Phase 2/4
**Output:** Robust out-of-sample performance estimates
**Runtime:** 6-12 hours for comprehensive validation

**Validation Methods:**
- Walk-forward validation (rolling retraining)
- Combinatorial Purged Cross-Validation (CPCV)
- Probability of Backtest Overfitting (PBO)

**Documentation:** [PHASE_5.md](PHASE_5.md)

## Navigation

### For New Users
1. Start with [Phase 1](PHASE_1.md) - Understand data pipeline
2. Read [Phase 2](PHASE_2.md) - Learn model training
3. Review [Quick Reference](../QUICK_REFERENCE.md) - Command cheatsheet

### For Developers
1. [Phase 1](PHASE_1.md) - Pipeline architecture
2. [Phase 2](PHASE_2.md) - Model implementation patterns
3. [Phase 3](PHASE_3.md) - CV and hyperparameter tuning
4. [Phase 4](PHASE_4.md) - Ensemble architecture

### For Researchers
1. [Phase 3](PHASE_3.md) - Robust validation methods
2. [Phase 5](PHASE_5.md) - Walk-forward and CPCV
3. [Quantitative Trading Analysis](../QUANTITATIVE_TRADING_ANALYSIS.md)

---

*Last Updated: 2025-12-30*
