# Phase 2 Quick Reference Card

**One-page reference for Phase 2 architecture**

---

## Core Components (4 Key Pieces)

```
1. BaseModel (base.py)          → Interface all models implement
2. ModelRegistry (registry.py)  → Plugin system for models
3. TimeSeriesDataset (dataset.py) → Data loading with zero leakage
4. Trainer (trainer.py)         → Training orchestration + MLflow
```

---

## File Size Limits

| Component | File | Max Lines | Actual |
|-----------|------|-----------|--------|
| Base Interface | `base.py` | 650 | ~250 |
| Model Registry | `registry.py` | 650 | ~180 |
| Dataset | `dataset.py` | 650 | ~200 |
| Trainer | `trainer.py` | 650 | ~200 |
| XGBoost Model | `xgboost.py` | 650 | ~180 |
| Each Model | `*.py` | 650 | 150-230 |

**All files respect 650-line limit ✅**

---

## Model Registration Pattern

```python
# Step 1: Define model with decorator
@ModelRegistry.register(name="xgboost", family="boosting")
class XGBoostModel(BaseModel):
    def fit(self, ...): ...
    def predict(self, ...): ...
    def save(self, ...): ...
    def load(self, ...): ...

# Step 2: Import to trigger registration
from src.models.boosting.xgboost import XGBoostModel

# Step 3: Instantiate via registry
model = ModelRegistry.create(
    model_name="xgboost",
    config={'n_estimators': 100, 'max_depth': 6},
    horizon=5,
    feature_columns=['feat1', 'feat2', ...]
)
```

---

## Data Flow (Phase 1 → Phase 2)

```
Phase 1 Outputs:
  data/splits/scaled/train_scaled.parquet  (87,094 × 126)
  data/splits/scaled/val_scaled.parquet    (18,591 × 126)
  data/splits/scaled/test_scaled.parquet   (18,592 × 126)
         ↓
TimeSeriesDataset:
  - Auto-detect 107 features, 2 labels (label_h5, label_h20)
  - Create windowed sequences (symbol-isolated)
  - Enforce temporal ordering (past only)
         ↓
Model Input:
  Boosting:     X(n, 107) or X(n, 60×107) if flattened
  Time Series:  X(n, 60, 107) - 3D sequences
```

---

## Training Workflow

```python
# 1. Create trainer
trainer = Trainer(
    model_name='xgboost',
    model_config={'n_estimators': 100, 'max_depth': 6},
    dataset_config=DatasetConfig(
        train_path='data/splits/scaled/train_scaled.parquet',
        val_path='data/splits/scaled/val_scaled.parquet',
        test_path='data/splits/scaled/test_scaled.parquet',
        horizon=5,
        sequence_length=1  # or 60 for time series
    ),
    experiment_name='baseline',
    use_mlflow=True
)

# 2. Run full pipeline
results = trainer.run_full_pipeline()
# - Loads data
# - Builds model
# - Trains with early stopping
# - Evaluates on val/test
# - Saves checkpoints
# - Logs to MLflow

# 3. View results
print(f"Val F1: {results['evaluation']['val_metrics']['f1']:.4f}")
print(f"Test F1: {results['evaluation']['test_metrics']['f1']:.4f}")
```

---

## CLI Commands

```bash
# Train single model
python scripts/train_model.py \
    --model xgboost \
    --horizon 5 \
    --config config/models/xgboost.yaml

# Run full experiment (all models)
python scripts/run_experiment.py \
    --config config/experiments/baseline.yaml

# Hyperparameter tuning
python scripts/tune_model.py \
    --model xgboost \
    --horizon 5 \
    --n-trials 100

# View MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000
```

---

## Configuration Structure

```yaml
# config/models/xgboost.yaml
model:
  name: "xgboost"
  family: "boosting"

hyperparameters:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.05

training:
  early_stopping_rounds: 20
  random_seed: 42

dataset:
  sequence_length: 1
  exclude_neutrals: false
```

---

## Validation Checklist (Before Commit)

```bash
# 1. File size
wc -l src/models/boosting/xgboost.py  # Must be <650

# 2. Tests pass
pytest tests/test_xgboost.py -v

# 3. Validation works
python -c "
from src.models.boosting.xgboost import XGBoostModel
model = ModelRegistry.create('xgboost', config, horizon=5, features=['f1'])
# Should fail with clear error if invalid
"

# 4. No exception swallowing
grep -r "except.*pass" src/models/  # Should return nothing

# 5. Model registered
python -c "
from src.models.registry import ModelRegistry
print(ModelRegistry.list_models())
# Should include your model
"
```

---

## Common Patterns

### Adding a New Model

```python
# 1. Create file: src/models/{family}/{name}.py
# 2. Define config dataclass (inherits ModelConfig)
# 3. Define model class (inherits BaseModel)
# 4. Add @ModelRegistry.register decorator
# 5. Implement: _build_config, _build_model, fit, predict, save, load
# 6. Import in __init__.py to trigger registration
# 7. Create config/models/{name}.yaml
# 8. Write tests/test_{name}.py
# 9. Test end-to-end with real data
```

### Handling 3D Sequences

```python
# Boosting models (need 2D)
if X.ndim == 3:
    n_samples, seq_len, n_features = X.shape
    X = X.reshape(n_samples, seq_len * n_features)

# Time series models (use 3D directly)
# No reshaping needed
```

### Label Encoding (for XGBoost/LightGBM)

```python
# Models expect {0, 1, 2}, but data has {-1, 0, 1}
def encode_labels(y):
    return y + 1  # -1→0, 0→1, 1→2

def decode_labels(y):
    return y - 1  # 0→-1, 1→0, 2→1
```

---

## Error Messages Guide

| Error | Cause | Solution |
|-------|-------|----------|
| "Model not found" | Not imported/registered | Import model module |
| "Invalid horizon" | horizon not 5 or 20 | Check horizon value |
| "Invalid labels" | Labels not in {-1,0,1} | Filter NaNs, check data |
| "Shape mismatch" | X.shape[0] != len(y) | Check data alignment |
| "Config validation failed" | Invalid hyperparameters | Check config values |
| "Model not fitted" | predict() before fit() | Call fit() first |

---

## Directory Structure (Minimal)

```
src/
├── models/
│   ├── base.py              # START HERE (Day 1)
│   ├── registry.py          # Day 2
│   └── boosting/
│       └── xgboost.py       # Day 3
├── data/
│   └── dataset.py           # Day 4
└── training/
    └── trainer.py           # Day 5

config/models/xgboost.yaml   # Day 3
scripts/train_model.py       # Day 5
tests/test_xgboost.py        # Day 3
```

---

## Metrics to Track

```python
# Classification metrics
- accuracy: (predictions == labels).mean()
- precision, recall, f1: per class (-1, 0, 1)
- confusion matrix

# Trading metrics
- win_rate: % correct non-neutral predictions
- sharpe_ratio: returns.mean() / returns.std() * sqrt(252)
- max_drawdown: max cumulative loss
- profit_factor: gross_profit / gross_loss
```

---

## Key Design Decisions

| Question | Decision | Reason |
|----------|----------|--------|
| Registry pattern? | Decorator-based | Self-contained, fail-fast |
| Base interface? | Abstract class | Type safety, enforced contract |
| Data loading? | TimeSeriesDataset | Zero-leakage guarantee |
| Training loop? | Model-specific fit() | Flexibility for different frameworks |
| Artifacts? | Structured run_dir + MLflow | Easy comparison, recovery |
| Config? | YAML files | Non-dev friendly, version control |

---

## Implementation Order

```
Week 1: Infrastructure
  Day 1: BaseModel (250 lines)
  Day 2: ModelRegistry (180 lines)
  Day 3: XGBoost (180 lines) + test
  Day 4: TimeSeriesDataset (200 lines)
  Day 5: Trainer (200 lines)

Week 2: More Models
  LightGBM, CatBoost

Week 3: Time Series
  N-HiTS, TFT

Week 4-5: Experiments
  Baseline runs, hyperparameter tuning
```

---

## Documentation Map

| File | Purpose | Read When |
|------|---------|-----------|
| `PHASE2_SUMMARY.md` | Executive summary | Start here |
| `PHASE2_QUICKSTART.md` | 30-min setup | Ready to code |
| `PHASE2_ARCHITECTURE.md` | Full design | Need details |
| `PHASE2_IMPLEMENTATION_CHECKLIST.md` | Day-by-day tasks | During implementation |
| `PHASE2_DESIGN_DECISIONS.md` | Q&A, rationale | Need to understand why |
| `PHASE2_ARCHITECTURE_DIAGRAM.md` | Visual diagrams | Need visual overview |
| `PHASE2_QUICK_REFERENCE.md` | This file | Quick lookup |

---

**Keep this file open while coding!**
