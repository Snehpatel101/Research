# Development Roadmap

**Project:** Ensemble Price Prediction Pipeline
**Last Updated:** 2025-12-24

---

## Overview

This roadmap tracks the evolution from data ingestion to production trading system. Each phase builds modular infrastructure, not monolithic implementations.

---

## Phase 1: Dynamic ML Factory (COMPLETED)

**Status:** Production-ready (Score: 7.5/10)

**Goal:** Build data preparation factory that generates labeled training data for ANY model.

### Deliverables
- Modular pipeline runner with stage registry
- Configurable labeling (triple-barrier with GA-optimized barriers)
- Feature engineering (107 features)
- Purged time series splits (train/val/test with 60-bar purge, 288-bar embargo)
- Comprehensive validation and reporting

### Key Infrastructure
- `src/pipeline/` - DAG-based pipeline orchestration
- `src/pipeline/stages/` - Pluggable pipeline stages
- `src/stages/` - Core data processing logic
- `data/splits/scaled/` - Ready-to-use parquet files

### Outputs for Phase 2
- `train_scaled.parquet` (87,094 rows × 126 cols)
- `val_scaled.parquet` (18,591 rows × 126 cols)
- `test_scaled.parquet` (18,592 rows × 126 cols)
- Features: 107 engineered features
- Labels: `label_h5`, `label_h10`, `label_h15`, `label_h20`

---

## Phase 2: Model Factory (IN DESIGN)

**Status:** Architecture complete, ready for implementation

**Goal:** Build MODEL FACTORY infrastructure - a training system where adding a new model is just an interface + config, not a rewrite.

### Core Concept: Plugin Architecture

Models are plugins, not hardcoded implementations:

```python
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    def fit(...): ...
    def predict(...): ...
    def save(...): ...
    def load(...): ...

# Auto-discovery + factory instantiation
model = ModelRegistry.create("xgboost", config, horizon, features)
```

### Infrastructure to Build

#### Week 1: Core Factory
- [ ] `src/models/base.py` - BaseModel interface, ModelConfig, PredictionOutput
- [ ] `src/models/registry.py` - ModelRegistry with decorator registration
- [ ] `src/data/dataset.py` - TimeSeriesDataset with zero-leakage windowing
- [ ] Unit tests for validation logic

#### Week 2: First Model Family (Boosting)
- [ ] `src/models/boosting/xgboost.py` - XGBoost implementation
- [ ] `src/models/boosting/lightgbm.py` - LightGBM implementation
- [ ] `src/models/boosting/catboost.py` - CatBoost implementation
- [ ] End-to-end test with real Phase 1 data

#### Week 3: Training Infrastructure
- [ ] `src/training/trainer.py` - Trainer orchestration
- [ ] `src/training/evaluator.py` - ModelEvaluator (metrics + backtests)
- [ ] `src/training/callbacks.py` - Early stopping, checkpointing
- [ ] CLI scripts: `train_model.py`, `run_experiment.py`
- [ ] MLflow integration

#### Week 4: Example Models from Other Families
- [ ] `src/models/timeseries/nhits.py` - N-HiTS (time series specialist)
- [ ] `src/models/timeseries/tft.py` - TFT (optional)
- [ ] `src/models/neural/lstm.py` - LSTM baseline (optional)
- [ ] Run baseline experiments (all models, all horizons)

#### Week 5: Hyperparameter Tuning (Optional)
- [ ] `src/tuning/optuna_tuner.py` - Optuna integration
- [ ] `src/tuning/search_spaces.py` - Model-specific search spaces
- [ ] Run tuning experiments (50-100 trials per model)
- [ ] Lock in production configs

### Supported Model Families

The factory will support ANY model family that implements the 4-method interface:

| Family | Examples | Strengths | Training Time |
|--------|----------|-----------|---------------|
| **Boosting** | XGBoost, LightGBM, CatBoost | Fast, interpretable, strong baseline | 1-3 hours |
| **Time Series** | N-HiTS, TFT, PatchTST, TimesFM | Multi-horizon, sequential patterns | 8-24 hours |
| **Neural** | LSTM, GRU, Transformer | Flexible, complex patterns | 4-12 hours |
| **Classical** | RandomForest, SVM, LogisticRegression | Simple, fast baseline | <1 hour |

### Deliverables

**Core Infrastructure:**
- BaseModel abstract interface
- ModelRegistry plugin system
- TimeSeriesDataset (temporal windowing with zero leakage)
- Trainer orchestration
- ModelEvaluator (classification + trading metrics)

**Example Models (3+ families):**
- Boosting: XGBoost, LightGBM, CatBoost
- Time Series: N-HiTS, TFT (optional)
- Neural: LSTM (optional)

**Per-Model Artifacts:**
- `experiments/runs/{model}_{timestamp}/checkpoints/` - Model weights
- `experiments/runs/{model}_{timestamp}/predictions/` - val/test predictions.parquet
- `experiments/runs/{model}_{timestamp}/metrics/` - metrics.json
- `experiments/runs/{model}_{timestamp}/plots/` - Visualizations

**Comparative Analysis:**
- `experiments/model_comparison.md` - Side-by-side metrics
- Best model per horizon
- Prediction correlations (diversity analysis)

### Success Criteria
- ModelRegistry can discover and instantiate any registered model
- TimeSeriesDataset validated (correct windowing, no leakage, symbol isolation)
- Trainer works with ANY model via BaseModel interface
- 3+ model families implemented (boosting, time series, classical/neural)
- All models achieve F1 > 0.35 AND Sharpe > 0.3 on at least one horizon
- Consistent prediction format across all models for ensemble stacking
- Tests passing (>80% coverage on core infrastructure)

**Key Insight:** Phase 2 is about building INFRASTRUCTURE to train models, not about training specific models. Adding a new model should require:
1. Implement 4 methods (fit, predict, save, load)
2. Register with `@ModelRegistry.register(name, family)`
3. Add YAML config
4. Done - no changes to training infrastructure

---

## Phase 3: Ensemble Stacking (PLANNED)

**Status:** Not started

**Goal:** Combine predictions from multiple models using a meta-learner to improve overall performance.

### Prerequisites
- Trained models from Phase 2 (3+ models from different families)
- Validation predictions (probabilities) from all models
- Diverse models (low correlation between predictions)

### Infrastructure to Build
- [ ] `src/ensemble/stacker.py` - Meta-learner (stacks model predictions)
- [ ] `src/ensemble/blender.py` - Simple weighted averaging (baseline)
- [ ] `src/ensemble/diversity.py` - Measure model diversity
- [ ] Cross-validation for meta-learner training (avoid overfitting)

### Stacking Strategies
1. **Simple Averaging:** Baseline - average probabilities from all models
2. **Weighted Averaging:** Learn optimal weights via CV
3. **Meta-Learner:** Train classifier on model predictions (LogisticRegression, XGBoost)
4. **Rank-Based:** Combine rank predictions instead of probabilities

### Deliverables
- Ensemble model that outperforms individual models
- Ablation study (contribution of each base model)
- Final predictions ready for Phase 4 (live trading simulation)

### Success Criteria
- Ensemble F1 > max(individual model F1)
- Ensemble Sharpe > max(individual model Sharpe)
- Stable performance across multiple CV folds
- Interpretable weights (which models matter most)

---

## Phase 4: Live Trading Simulation (PLANNED)

**Status:** Not started

**Goal:** Simulate live trading to validate performance in production-like environment.

### Prerequisites
- Trained ensemble from Phase 3
- Test set (held out, never seen during training)
- Realistic trading assumptions (slippage, commissions, latency)

### Infrastructure to Build
- [ ] `src/backtest/live_simulator.py` - Realistic trading simulation
- [ ] `src/backtest/execution.py` - Order execution with slippage
- [ ] `src/backtest/risk_manager.py` - Position sizing, risk limits
- [ ] Real-time feature calculation pipeline

### Simulation Features
- Realistic order execution (market orders, slippage)
- Transaction costs (commissions, spread)
- Latency modeling (prediction to execution delay)
- Position sizing (risk-based, not fixed)
- Risk limits (max drawdown, position limits)
- Walk-forward testing (retrain periodically)

### Deliverables
- Live simulation results on test set
- Final performance metrics (Sharpe, drawdown, win rate, profit factor)
- Tearsheet with equity curve, trade distribution, etc.
- Risk analysis (VaR, CVaR, stress testing)

### Success Criteria
- Test Sharpe > 0.5 (realistic after costs)
- Max drawdown < 20%
- Profit factor > 1.3
- Stable performance across market regimes
- No catastrophic failures or edge cases

---

## Phase 5: Production Deployment (FUTURE)

**Status:** Not started

**Goal:** Deploy system to production for paper trading, then live trading.

### Prerequisites
- Validated performance from Phase 4
- Robust error handling and monitoring
- Infrastructure for real-time data ingestion
- Paper trading validation (1-3 months)

### Infrastructure to Build
- [ ] Real-time data pipeline (live market data ingestion)
- [ ] Model serving API (FastAPI, gRPC)
- [ ] Order management system (OMS) integration
- [ ] Monitoring and alerting (Prometheus, Grafana)
- [ ] Logging and audit trail
- [ ] Disaster recovery and failover

### Deployment Phases
1. **Paper Trading:** Simulate orders without real capital (validate in production environment)
2. **Small Capital:** Start with minimal capital (validate risk management)
3. **Scale Up:** Gradually increase capital as confidence grows

### Success Criteria
- Zero downtime during market hours
- <100ms prediction latency (real-time)
- Automated model retraining and deployment
- Comprehensive monitoring and alerting
- Paper trading results match backtest expectations

---

## Milestones & Timeline

| Phase | Duration | Completion Date | Status |
|-------|----------|-----------------|--------|
| Phase 1: Dynamic ML Factory | 3 weeks | 2025-12-18 | COMPLETED |
| Phase 2: Model Factory | 4-5 weeks | TBD | IN DESIGN |
| Phase 3: Ensemble Stacking | 2 weeks | TBD | PLANNED |
| Phase 4: Live Trading Simulation | 2 weeks | TBD | PLANNED |
| Phase 5: Production Deployment | 4+ weeks | TBD | FUTURE |

**Total Estimated Time:** 15-18 weeks from Phase 1 start to production deployment

---

## Key Principles (All Phases)

### Architecture
- **Modularity:** Components with clear contracts, minimal coupling
- **Plugin Architecture:** Add new capabilities without rewriting core
- **Fail Fast:** Validate at boundaries, crash early with clear errors
- **Less Code:** Simple > clever, boring > fancy

### Constraints
- **650-line limit** per file
- **No exception swallowing**
- **Explicit validation** at every boundary
- **Zero data leakage** (purge/embargo enforced)

### Testing
- Unit tests for pure logic
- Integration tests for pipelines
- Regression tests for fixed bugs
- >80% coverage on core infrastructure

---

## Current Focus

**Active Phase:** Phase 2 (Model Factory)

**Next Steps:**
1. Implement BaseModel + ModelRegistry (Week 1)
2. Implement first boosting models (XGBoost, LightGBM, CatBoost) (Week 2)
3. Build Trainer orchestration + ModelEvaluator (Week 3)
4. Add time series models (N-HiTS, TFT) (Week 4)
5. Run baseline experiments and comparative analysis (Week 4-5)

**Critical Path:** Building factory infrastructure, not specific models. Success = extensible system that makes adding new models trivial.

---

## References

- Phase 1 Documentation: `/home/jake/Desktop/Research/docs/phase1/`
- Phase 2 Documentation: `/home/jake/Desktop/Research/docs/phases/PHASE_2.md`
- Engineering Principles: `/home/jake/Desktop/Research/CLAUDE.md`
- Pipeline Config: `/home/jake/Desktop/Research/src/pipeline_config.py`

---

**Last Updated:** 2025-12-24
**Status:** Phase 2 architecture designed, ready for implementation
