# Phase 2 Architecture - Executive Summary

**Project:** Ensemble Trading Pipeline - Model Training System
**Date:** 2025-12-21
**Status:** Design Complete, Ready for Implementation

---

## What Was Designed

A **modular, extensible model training system** that supports many model families (time series, classical ML, neural networks) while maintaining strict adherence to your engineering principles:

- ✅ **650-line limit** per file
- ✅ **Fail-fast validation** at every boundary
- ✅ **No exception swallowing**
- ✅ **Clear separation of concerns**
- ✅ **Zero data leakage** guarantees
- ✅ **Less code is better** philosophy

---

## Core Design Patterns

### 1. Plugin Architecture (Model Registry)
```python
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    ...

# Auto-discovery + factory instantiation
model = ModelRegistry.create("xgboost", config, horizon, features)
```

### 2. Abstract Base Class (Common Interface)
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> Dict: ...
    @abstractmethod
    def predict(self, X, metadata) -> PredictionOutput: ...
    @abstractmethod
    def save(self, path: Path): ...
    @abstractmethod
    def load(self, path: Path): ...
```

### 3. Temporal Dataset (Zero Leakage)
```python
class TimeSeriesDataset:
    # Symbol-isolated windowing
    # Past features only (no future leakage)
    # Flexible sequence lengths (1 for boosting, 60+ for time series)
    def _create_sequences(self, df):
        for symbol in symbols:
            for i in range(seq_len, len(data)):
                X_window = data[i-seq_len : i]  # Past only
                y_label = data[i]               # Future
```

### 4. Orchestrated Training (Reusable Infrastructure)
```python
class Trainer:
    def run_full_pipeline(self):
        self.prepare_data()      # Load Phase 1 splits
        self.build_model()       # Instantiate via registry
        self.train()             # Delegate to model.fit()
        self.evaluate()          # Compute metrics, save predictions
        # MLflow tracks everything automatically
```

### 5. Structured Artifacts (Experiment Tracking)
```
experiments/runs/{model}_{timestamp}/
├── checkpoints/model/           # Model weights + config
├── predictions/                 # val/test predictions.parquet
├── metrics/metrics.json         # All metrics
└── plots/                       # Visualizations
```

---

## File Structure Created

```
src/
├── models/                      # Model implementations
│   ├── base.py                  # BaseModel, ModelConfig, PredictionOutput
│   ├── registry.py              # ModelRegistry (plugin architecture)
│   ├── boosting/                # XGBoost, LightGBM, CatBoost
│   ├── timeseries/              # N-HiTS, TFT, PatchTST, TimesFM
│   └── neural/                  # LSTM, GRU, Transformer
│
├── data/                        # Data loading
│   ├── dataset.py               # TimeSeriesDataset (windowing)
│   └── loaders.py               # DataLoader factories
│
├── training/                    # Training orchestration
│   ├── trainer.py               # Trainer (MLflow integration)
│   ├── evaluator.py             # ModelEvaluator (metrics)
│   └── callbacks.py             # EarlyStopping, Checkpointing
│
└── tuning/                      # Hyperparameter optimization
    ├── optuna_tuner.py          # Optuna integration
    └── search_spaces.py         # Model-specific search spaces

config/
├── models/                      # Model-specific YAML configs
│   ├── xgboost.yaml
│   ├── nhits.yaml
│   └── lstm.yaml
└── experiments/                 # Experiment definitions
    └── baseline.yaml

scripts/
├── train_model.py               # CLI: Train single model
├── run_experiment.py            # CLI: Run full experiment
└── tune_model.py                # CLI: Hyperparameter tuning

experiments/
├── runs/                        # Training run outputs
├── mlruns/                      # MLflow artifact store
└── registry/                    # Production models
```

---

## Integration with Phase 1

**Phase 1 Outputs** → **Phase 2 Inputs**

```python
# Phase 1 creates these files
data/splits/scaled/
├── train_scaled.parquet  (87,094 rows × 126 cols)
├── val_scaled.parquet    (18,591 rows × 126 cols)
└── test_scaled.parquet   (18,592 rows × 126 cols)

# Phase 2 loads them via TimeSeriesDataset
dataset = TimeSeriesDataset(DatasetConfig(
    train_path='data/splits/scaled/train_scaled.parquet',
    val_path='data/splits/scaled/val_scaled.parquet',
    test_path='data/splits/scaled/test_scaled.parquet',
    horizon=5,
    sequence_length=60
))

# Auto-detects:
# - Features: 107 columns (everything except datetime, symbol, labels)
# - Labels: label_h5, label_h20
# - Purge/embargo already applied by Phase 1
```

---

## Answers to Your Questions

### 1. Model Registry Pattern
**Answer:** Decorator-based plugin architecture with auto-discovery
- Models self-register via `@ModelRegistry.register`
- Factory pattern for instantiation: `ModelRegistry.create()`
- Fail-fast validation at registration time

### 2. Base Model Interface
**Answer:** Abstract base class with enforced contract
- Required methods: `fit()`, `predict()`, `save()`, `load()`
- Standardized output: `PredictionOutput` dataclass
- Input validation: `validate_inputs()` in base class

### 3. Data Loading
**Answer:** TimeSeriesDataset with temporal windowing
- Symbol-isolated windows (no cross-symbol leakage)
- Past features only (no future leakage)
- Flexible sequence lengths (1 for boosting, 60+ for time series)

### 4. Training Loop
**Answer:** Hybrid approach - reusable orchestration + model-specific loops
- Trainer handles workflow (data → model → train → evaluate)
- Models implement custom training loops in `fit()`
- MLflow tracks everything automatically

### 5. Artifact Management
**Answer:** Structured run directories + MLflow tracking
- Run-specific directories: `experiments/runs/{model}_{timestamp}/`
- MLflow UI for comparison and visualization
- Production registry separate from experiments

### 6. Configuration
**Answer:** Hybrid - global config.py + model-specific YAML files
- `config.py` for project-wide settings (Phase 1 params)
- `config/models/*.yaml` for model hyperparameters
- `config/experiments/*.yaml` for multi-model experiments

---

## Implementation Roadmap

### Week 1: Core Infrastructure
- [ ] BaseModel, ModelConfig, PredictionOutput (~250 lines)
- [ ] ModelRegistry with auto-discovery (~180 lines)
- [ ] TimeSeriesDataset with windowing (~200 lines)
- [ ] Unit tests for validation logic

### Week 2: First Model Family (Boosting)
- [ ] XGBoostModel (~180 lines)
- [ ] LightGBMModel (~170 lines)
- [ ] CatBoostModel (~170 lines)
- [ ] End-to-end test with real Phase 1 data

### Week 3: Training Infrastructure
- [ ] Trainer orchestration (~200 lines)
- [ ] ModelEvaluator (~150 lines)
- [ ] Training callbacks (~120 lines)
- [ ] CLI scripts (train_model.py, run_experiment.py)

### Week 4: Time Series Models
- [ ] N-HiTS implementation (~220 lines)
- [ ] TFT implementation (~230 lines)
- [ ] Baseline experiments (all models, both horizons)

### Week 5: Hyperparameter Tuning
- [ ] OptunaModelTuner (~200 lines)
- [ ] Search space definitions (~150 lines)
- [ ] Tuning experiments (50-100 trials per model)
- [ ] Lock in production configs

**Total:** ~20 files, ~6,000 lines of code, all files <650 lines

---

## Key Metrics to Track

### Model Performance
- Accuracy, Precision, Recall, F1 (per class: -1, 0, 1)
- Sharpe Ratio (simulated trading)
- Win Rate, Max Drawdown
- Profit Factor

### Training Efficiency
- Training time per epoch
- Convergence speed (early stopping)
- Memory usage

### Experiment Metadata
- Hyperparameters
- Model checkpoints
- Feature importance (for tree models)
- Predictions (val/test)

---

## Documentation Delivered

| Document | Purpose | Lines |
|----------|---------|-------|
| `PHASE2_ARCHITECTURE.md` | Comprehensive system design | 1,000+ |
| `PHASE2_ARCHITECTURE_DIAGRAM.md` | Visual diagrams (Mermaid) | 500+ |
| `PHASE2_IMPLEMENTATION_CHECKLIST.md` | Day-by-day implementation tasks | 800+ |
| `PHASE2_DESIGN_DECISIONS.md` | Q&A and design rationale | 800+ |
| `PHASE2_QUICKSTART.md` | 30-minute getting started guide | 600+ |
| `PHASE2_SUMMARY.md` | This document | 300+ |

**Total Documentation:** ~4,000 lines across 6 files

---

## Success Criteria

- [x] All design questions answered
- [x] Architecture respects 650-line limit
- [x] Fail-fast validation at every boundary
- [x] Zero-leakage guarantees
- [x] Clear separation of concerns
- [x] Extensible to new model families
- [x] MLflow integration for experiment tracking
- [x] Phase 1 integration points defined
- [x] Implementation roadmap (5 weeks)
- [x] Code examples provided (XGBoost complete)

---

## Next Steps

1. **Read PHASE2_QUICKSTART.md** to get started (30 min setup)
2. **Implement BaseModel + Registry** (Week 1, Days 1-2)
3. **Implement XGBoost model** (Week 1, Day 3)
4. **Test end-to-end with Phase 1 data** (Week 1, Day 3)
5. **Follow PHASE2_IMPLEMENTATION_CHECKLIST.md** for remaining tasks

---

## Design Principles Maintained

✅ **Modularity:** Each model in isolated module (<650 lines)
✅ **Fail-Fast:** Validation at every boundary (config, inputs, registry)
✅ **No Exception Swallowing:** Errors propagate with clear messages
✅ **Less Code:** Simple, boring solutions over clever abstractions
✅ **Clear Contracts:** Abstract base class enforces interface
✅ **Testing:** Unit/integration tests for all components
✅ **Documentation:** Comprehensive architecture + usage guides

---

## Questions or Clarifications?

All design documents are located in:
```
/home/jake/Desktop/Research/PHASE2_*.md
```

If you have questions during implementation:
1. Check `PHASE2_DESIGN_DECISIONS.md` for rationale
2. See `PHASE2_IMPLEMENTATION_CHECKLIST.md` for detailed tasks
3. Refer to `PHASE2_QUICKSTART.md` for code examples

---

**Architecture designed by:** Claude (Backend Architect Agent)
**Date:** 2025-12-21
**Status:** ✅ Design Complete, Ready for Implementation

**Estimated Implementation Time:** 4-5 weeks (following checklist)
**Expected Model Families:** 3+ (boosting, time series, neural)
**Expected Models:** 6+ (XGBoost, LightGBM, CatBoost, N-HiTS, TFT, LSTM)

---

**End of Phase 2 Summary**
