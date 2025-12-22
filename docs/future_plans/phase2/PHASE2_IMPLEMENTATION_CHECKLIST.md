# Phase 2 Implementation Checklist

**Project:** Ensemble Trading Pipeline - Model Training System
**Date:** 2025-12-21
**Status:** Ready for Implementation

---

## Prerequisites

- [x] Phase 1 complete (data/splits/scaled/*.parquet exist)
- [x] 107 features, 2 active horizons (H5, H20)
- [x] Purge (60) + Embargo (1440) applied
- [x] Labels: {-1, 0, 1} for short/neutral/long

---

## Week 1: Core Infrastructure (Days 1-3)

### Day 1: Base Model Interface

**File:** `/home/jake/Desktop/Research/src/models/base.py` (~250 lines)

- [ ] Create `ModelConfig` dataclass
  - [ ] Required fields: model_name, model_family, horizon
  - [ ] Optional: early_stopping, patience, random_seed
  - [ ] `validate()` method with fail-fast checks
- [ ] Create `PredictionOutput` dataclass
  - [ ] Fields: predictions, probabilities, timestamps, symbols, horizons
  - [ ] `to_dataframe()` method
- [ ] Create `BaseModel` abstract class
  - [ ] Abstract methods: `fit()`, `predict()`, `save()`, `load()`
  - [ ] Abstract methods: `_build_config()`, `_build_model()`
  - [ ] Concrete: `validate_inputs()` - check shapes/values
  - [ ] Concrete: `get_feature_importance()` - optional, returns None
  - [ ] `__init__()`: validate horizon, features, call _build_config/_build_model
- [ ] Write unit tests for validation logic
  - [ ] Test invalid horizons (not 5 or 20)
  - [ ] Test empty feature_columns
  - [ ] Test invalid label values (not {-1, 0, 1})
  - [ ] Test shape mismatches (X vs y)

**Validation:**
```python
# Quick test
from src.models.base import BaseModel, ModelConfig, PredictionOutput
assert ModelConfig(model_name='test', model_family='test', horizon=5).validate() is None
```

---

### Day 2: Model Registry

**File:** `/home/jake/Desktop/Research/src/models/registry.py` (~180 lines)

- [ ] Create `ModelRegistry` class
  - [ ] Class variables: `_registry: Dict[str, Type[BaseModel]]`
  - [ ] Class variables: `_metadata: Dict[str, dict]`
- [ ] Implement `@register` decorator
  - [ ] Parameters: name, family, description, requires_gpu, supports_multivariate
  - [ ] Validation: Check BaseModel inheritance
  - [ ] Validation: Check required methods exist (fit, predict, save, load)
  - [ ] Store in `_registry[f"{family}:{name}"]`
- [ ] Implement `create()` factory method
  - [ ] Resolve short name to full name (family:name)
  - [ ] Validate config dict
  - [ ] Validate horizon (5 or 20)
  - [ ] Validate feature_columns (non-empty list)
  - [ ] Instantiate model class with try/except
- [ ] Implement helper methods
  - [ ] `list_models(family=None)` - return list of metadata
  - [ ] `get_metadata(model_name)` - return single model metadata
  - [ ] `_resolve_name(name)` - short name to full name
- [ ] Implement `auto_register_models()`
  - [ ] Scan `models/timeseries/*.py`
  - [ ] Scan `models/boosting/*.py`
  - [ ] Scan `models/neural/*.py`
  - [ ] Use `importlib.import_module()`
- [ ] Write unit tests
  - [ ] Test registration with valid model
  - [ ] Test registration fails for non-BaseModel class
  - [ ] Test create() with invalid name
  - [ ] Test name resolution (short vs full)

**Validation:**
```python
# Quick test
from src.models.registry import ModelRegistry

# Should be empty initially
assert len(ModelRegistry.list_models()) == 0

# After registering a model
@ModelRegistry.register(name='test', family='test')
class TestModel(BaseModel):
    ...

assert len(ModelRegistry.list_models()) == 1
```

---

### Day 3: TimeSeriesDataset

**File:** `/home/jake/Desktop/Research/src/data/dataset.py` (~200 lines)

- [ ] Create `DatasetConfig` dataclass
  - [ ] Required: train_path, val_path, test_path
  - [ ] Optional: sequence_length (default=60), horizon (default=5)
  - [ ] Optional: feature_columns, label_column, include_symbols, exclude_neutrals
  - [ ] `validate()` method - check paths exist, sequence_length > 0, valid horizon
- [ ] Create `TimeSeriesDataset` class
  - [ ] `__init__(config)` - validate config, load parquet files
  - [ ] `_detect_columns()` - auto-detect features/labels
    - [ ] Metadata cols: datetime, symbol, split
    - [ ] Label cols: label_h5, label_h20
    - [ ] Feature cols: everything else
  - [ ] `_create_sequences(df, split_name)` - create windowed sequences
    - [ ] Loop over symbols (prevent cross-symbol windows)
    - [ ] For each symbol: sort by datetime
    - [ ] Create windows: X[i-seq_len:i], y[i]
    - [ ] Skip NaN labels, optionally skip neutrals
    - [ ] Return (X, y, metadata)
  - [ ] `get_split(split)` - return train/val/test sequences
  - [ ] `get_feature_columns()` - return list of feature names
- [ ] Write unit tests
  - [ ] Test sequence creation with seq_len=10
  - [ ] Test symbol isolation (no cross-symbol windows)
  - [ ] Test temporal ordering (past features only)
  - [ ] Test exclude_neutrals filtering
  - [ ] Test invalid paths, invalid horizon

**Validation:**
```python
# Quick test with actual data
from src.data.dataset import TimeSeriesDataset, DatasetConfig
from pathlib import Path

config = DatasetConfig(
    train_path=Path("data/splits/scaled/train_scaled.parquet"),
    val_path=Path("data/splits/scaled/val_scaled.parquet"),
    test_path=Path("data/splits/scaled/test_scaled.parquet"),
    horizon=5,
    sequence_length=60
)

dataset = TimeSeriesDataset(config)
X_train, y_train, meta_train = dataset.get_split('train')

print(f"Train sequences: {X_train.shape}")  # Should be (n, 60, 107)
print(f"Labels: {y_train.shape}")            # Should be (n,)
print(f"Unique labels: {np.unique(y_train)}") # Should be {-1, 0, 1}
```

---

## Week 2: First Model Family - Boosting (Days 4-6)

### Day 4: XGBoost Model

**File:** `/home/jake/Desktop/Research/src/models/boosting/xgboost.py` (~180 lines)

- [ ] Create `XGBoostConfig` dataclass (inherits ModelConfig)
  - [ ] XGB hyperparameters: n_estimators, max_depth, learning_rate, etc.
  - [ ] Defaults: n_estimators=100, max_depth=6, learning_rate=0.1
- [ ] Create `XGBoostModel` class (inherits BaseModel)
  - [ ] Decorate with `@ModelRegistry.register(name="xgboost", family="boosting")`
  - [ ] Implement `_build_config(config, horizon)` - create XGBoostConfig
  - [ ] Implement `_build_model()` - initialize XGBClassifier
  - [ ] Implement `fit()`:
    - [ ] Flatten 3D sequences to 2D if needed: (n, seq, feat) -> (n, seq*feat)
    - [ ] Convert labels {-1,0,1} -> {0,1,2} via `_encode_labels()`
    - [ ] Call `xgb.fit(X, y, eval_set=[(X_val, y_val)])`
    - [ ] Store training history (train_loss, val_loss)
    - [ ] Return training results dict
  - [ ] Implement `predict()`:
    - [ ] Flatten 3D if needed
    - [ ] Call `xgb.predict_proba()` and `xgb.predict()`
    - [ ] Convert predictions {0,1,2} -> {-1,0,1} via `_decode_labels()`
    - [ ] Extract timestamps/symbols from metadata
    - [ ] Return PredictionOutput
  - [ ] Implement `save()` - save model + metadata
  - [ ] Implement `load()` - restore model + metadata
  - [ ] Implement `get_feature_importance()` - XGB feature_importances_
  - [ ] Helper: `_encode_labels(y)` - y + 1
  - [ ] Helper: `_decode_labels(y)` - y - 1
- [ ] Write unit tests
  - [ ] Test fit with dummy data
  - [ ] Test predict returns correct shapes
  - [ ] Test label encoding/decoding
  - [ ] Test save/load roundtrip
  - [ ] Test feature importance

**Validation:**
```python
# Quick test
from src.models.boosting.xgboost import XGBoostModel
from src.models.registry import ModelRegistry

# Should be registered
assert 'boosting:xgboost' in [m['name'] for m in ModelRegistry.list_models()]

# Should instantiate
model = ModelRegistry.create(
    model_name='xgboost',
    config={'n_estimators': 10, 'max_depth': 3},
    horizon=5,
    feature_columns=['feat1', 'feat2']
)

assert not model.is_fitted
```

---

### Day 5: LightGBM Model

**File:** `/home/jake/Desktop/Research/src/models/boosting/lightgbm.py` (~170 lines)

- [ ] Create `LightGBMConfig` dataclass
  - [ ] Hyperparameters: n_estimators, max_depth, learning_rate, num_leaves, etc.
- [ ] Create `LightGBMModel` class
  - [ ] Decorate with `@ModelRegistry.register(name="lightgbm", family="boosting")`
  - [ ] Similar structure to XGBoost
  - [ ] Use `lgb.LGBMClassifier`
  - [ ] Handle categorical features if needed
- [ ] Write unit tests

---

### Day 6: CatBoost Model

**File:** `/home/jake/Desktop/Research/src/models/boosting/catboost.py` (~170 lines)

- [ ] Create `CatBoostConfig` dataclass
  - [ ] Hyperparameters: iterations, depth, learning_rate, etc.
- [ ] Create `CatBoostModel` class
  - [ ] Decorate with `@ModelRegistry.register(name="catboost", family="boosting")`
  - [ ] Use `CatBoostClassifier`
  - [ ] Handle silent mode (verbose=False)
- [ ] Write unit tests

**End of Week Validation:**
```python
# All 3 boosting models should be registered
models = ModelRegistry.list_models(family='boosting')
assert len(models) == 3
assert set([m['name'] for m in models]) == {'xgboost', 'lightgbm', 'catboost'}
```

---

## Week 3: Training Infrastructure (Days 7-10)

### Day 7: Model Evaluator

**File:** `/home/jake/Desktop/Research/src/training/evaluator.py` (~150 lines)

- [ ] Create `ModelEvaluator` class
  - [ ] `__init__(model)` - store model reference
  - [ ] `evaluate(X, y, metadata, split_name)` - compute metrics
    - [ ] Get predictions via `model.predict(X, metadata)`
    - [ ] Compute classification metrics:
      - [ ] Accuracy (overall)
      - [ ] Precision, Recall, F1 (per class: -1, 0, 1)
      - [ ] Confusion matrix
    - [ ] Compute trading metrics:
      - [ ] Win rate (% of correct non-neutral predictions)
      - [ ] Sharpe ratio (simulated returns)
      - [ ] Max drawdown
    - [ ] Return dict of metrics
  - [ ] `plot_confusion_matrix(y_true, y_pred)` - optional
  - [ ] `generate_report(metrics)` - create summary report
- [ ] Write unit tests
  - [ ] Test metric calculation with known predictions
  - [ ] Test handling of all-neutral predictions

---

### Day 8: Training Callbacks

**File:** `/home/jake/Desktop/Research/src/training/callbacks.py` (~120 lines)

- [ ] Create `Callback` base class
  - [ ] Methods: `on_epoch_begin()`, `on_epoch_end()`, `on_train_begin()`, `on_train_end()`
- [ ] Create `CallbackList` class
  - [ ] Manages list of callbacks
  - [ ] Calls all callbacks in sequence
- [ ] Create `EarlyStoppingCallback`
  - [ ] Track validation metric
  - [ ] Stop if no improvement for `patience` epochs
- [ ] Create `CheckpointCallback`
  - [ ] Save model at best epoch
  - [ ] Save at regular intervals
- [ ] Write unit tests

---

### Day 9-10: Trainer

**File:** `/home/jake/Desktop/Research/src/training/trainer.py` (~200 lines)

- [ ] Create `Trainer` class
  - [ ] `__init__(model_name, model_config, dataset_config, experiment_name, output_dir, use_mlflow)`
    - [ ] Create run_id (model_name + timestamp)
    - [ ] Create run_dir (output_dir/run_id)
  - [ ] `prepare_data()` - instantiate TimeSeriesDataset
  - [ ] `build_model()` - instantiate model via ModelRegistry
  - [ ] `train(callbacks=None)` - execute training
    - [ ] Setup MLflow experiment
    - [ ] Log parameters (model config, dataset config)
    - [ ] Call `model.fit(X_train, y_train, X_val, y_val)`
    - [ ] Log metrics (training history)
    - [ ] Save model to run_dir
    - [ ] Log artifacts to MLflow
  - [ ] `evaluate()` - evaluate on val/test
    - [ ] Create ModelEvaluator
    - [ ] Evaluate on val set
    - [ ] Evaluate on test set
    - [ ] Save predictions to parquet
    - [ ] Log metrics/artifacts to MLflow
  - [ ] `run_full_pipeline()` - prepare -> build -> train -> evaluate
- [ ] Write integration tests
  - [ ] Test full pipeline with dummy data
  - [ ] Test MLflow logging
  - [ ] Test checkpoint saving

**Validation:**
```python
# End-to-end test with XGBoost
from src.training.trainer import Trainer
from src.data.dataset import DatasetConfig
from pathlib import Path

dataset_config = DatasetConfig(
    train_path=Path("data/splits/scaled/train_scaled.parquet"),
    val_path=Path("data/splits/scaled/val_scaled.parquet"),
    test_path=Path("data/splits/scaled/test_scaled.parquet"),
    horizon=5,
    sequence_length=1  # No windowing for XGBoost
)

trainer = Trainer(
    model_name='xgboost',
    model_config={'n_estimators': 10, 'max_depth': 3},
    dataset_config=dataset_config,
    experiment_name='test_run',
    use_mlflow=False
)

results = trainer.run_full_pipeline()
print(f"Validation F1: {results['evaluation']['val_metrics']['f1']:.4f}")
```

---

## Week 4: Time Series Models (Days 11-14)

### Day 11-12: N-HiTS Model

**File:** `/home/jake/Desktop/Research/src/models/timeseries/nhits.py` (~220 lines)

- [ ] Install `neuralforecast` library
- [ ] Create `NHiTSConfig` dataclass
  - [ ] Hyperparameters: input_size, h (horizon), hidden_layers, etc.
- [ ] Create `NHiTSModel` class
  - [ ] Decorate with `@ModelRegistry.register(name="nhits", family="timeseries")`
  - [ ] Use `NHiTS` from neuralforecast
  - [ ] Handle 3D inputs natively (no flattening needed)
  - [ ] Implement PyTorch training loop if needed
- [ ] Write unit tests

---

### Day 13-14: TFT Model

**File:** `/home/jake/Desktop/Research/src/models/timeseries/tft.py` (~230 lines)

- [ ] Create `TFTConfig` dataclass
- [ ] Create `TFTModel` class
  - [ ] Decorate with `@ModelRegistry.register(name="tft", family="timeseries")`
  - [ ] Use Temporal Fusion Transformer implementation
- [ ] Write unit tests

---

## Week 5: Experiments & Tuning (Days 15-20)

### Day 15: Configuration Files

**Files:** `config/models/*.yaml` and `config/experiments/*.yaml`

- [ ] Create `config/models/xgboost.yaml`
  - [ ] Hyperparameters section
  - [ ] Training section (early stopping, seed, etc.)
  - [ ] Dataset section (sequence_length, etc.)
- [ ] Create `config/models/lightgbm.yaml`
- [ ] Create `config/models/catboost.yaml`
- [ ] Create `config/models/nhits.yaml`
- [ ] Create `config/experiments/baseline.yaml`
  - [ ] Experiment name/description
  - [ ] List of models to run
  - [ ] Horizons: [5, 20]
  - [ ] Data paths
  - [ ] MLflow settings

---

### Day 16: CLI Scripts

**File:** `/home/jake/Desktop/Research/scripts/train_model.py` (~150 lines)

- [ ] Create CLI parser with argparse
  - [ ] Arguments: --model, --horizon, --config, --output, --no-mlflow
- [ ] Load YAML config
- [ ] Create DatasetConfig from paths
- [ ] Create Trainer instance
- [ ] Call `trainer.run_full_pipeline()`
- [ ] Print results summary
- [ ] Test with: `python scripts/train_model.py --model xgboost --horizon 5 --config config/models/xgboost.yaml`

**File:** `/home/jake/Desktop/Research/scripts/run_experiment.py` (~180 lines)

- [ ] Load experiment YAML config
- [ ] Loop over models and horizons
- [ ] Create Trainer for each combination
- [ ] Run training in sequence (or parallel if desired)
- [ ] Aggregate results
- [ ] Generate comparison report

---

### Day 17-18: Hyperparameter Tuning

**File:** `/home/jake/Desktop/Research/src/tuning/optuna_tuner.py` (~200 lines)

- [ ] Install `optuna` library
- [ ] Create `OptunaModelTuner` class
  - [ ] `__init__(model_name, dataset_config, search_space_fn, n_trials, direction, metric_name)`
  - [ ] `objective(trial)` - single Optuna trial
    - [ ] Sample hyperparameters via search_space_fn(trial)
    - [ ] Create Trainer with sampled config
    - [ ] Train and evaluate
    - [ ] Return metric value
  - [ ] `tune()` - run optimization
    - [ ] Create Optuna study
    - [ ] Run n_trials
    - [ ] Return best_params, best_value
- [ ] Write unit tests

**File:** `/home/jake/Desktop/Research/src/tuning/search_spaces.py` (~150 lines)

- [ ] Define `xgboost_search_space(trial)` function
  - [ ] Sample n_estimators (50-500)
  - [ ] Sample max_depth (3-12)
  - [ ] Sample learning_rate (0.01-0.3, log scale)
  - [ ] Sample subsample (0.6-1.0)
  - [ ] Sample colsample_bytree (0.6-1.0)
- [ ] Define search spaces for other models
- [ ] Write unit tests

**File:** `/home/jake/Desktop/Research/scripts/tune_model.py` (~100 lines)

- [ ] CLI for running hyperparameter tuning
- [ ] Test with: `python scripts/tune_model.py --model xgboost --horizon 5 --n-trials 50`

---

### Day 19-20: Baseline Experiments

- [ ] Run XGBoost for H5 and H20
  ```bash
  python scripts/train_model.py --model xgboost --horizon 5 --config config/models/xgboost.yaml
  python scripts/train_model.py --model xgboost --horizon 20 --config config/models/xgboost.yaml
  ```
- [ ] Run LightGBM for H5 and H20
- [ ] Run CatBoost for H5 and H20
- [ ] Run N-HiTS for H5 and H20 (if implemented)
- [ ] Compare results in MLflow UI
  ```bash
  mlflow ui --backend-store-uri experiments/mlruns
  ```
- [ ] Generate comparison report
- [ ] Document best performing models per horizon

---

## Validation Checkpoints

### After Week 1 (Infrastructure)
```python
# All core components exist and pass tests
pytest tests/test_base_model.py -v
pytest tests/test_registry.py -v
pytest tests/test_dataset.py -v
```

### After Week 2 (Boosting Models)
```python
# All 3 boosting models registered
from src.models.registry import ModelRegistry
models = ModelRegistry.list_models(family='boosting')
assert len(models) == 3

# Can train XGBoost end-to-end
# (See validation code above)
```

### After Week 3 (Training Infrastructure)
```python
# Can run full training pipeline
trainer = Trainer(...)
results = trainer.run_full_pipeline()
assert 'evaluation' in results
assert 'val_metrics' in results['evaluation']
```

### After Week 5 (Experiments)
```bash
# MLflow UI shows multiple runs
mlflow ui --backend-store-uri experiments/mlruns

# Comparison report exists
ls -lh experiments/runs/comparison_report.md
```

---

## Testing Strategy

### Unit Tests (~30 tests total)
- `tests/test_base_model.py` - BaseModel validation, input checks
- `tests/test_registry.py` - Registration, factory, name resolution
- `tests/test_dataset.py` - Sequence creation, filtering, temporal ordering
- `tests/test_xgboost.py` - XGBoost fit/predict/save/load
- `tests/test_evaluator.py` - Metric calculations
- `tests/test_trainer.py` - Training orchestration

### Integration Tests (~5 tests)
- `tests/test_end_to_end.py` - Full pipeline with dummy data
- `tests/test_mlflow_integration.py` - MLflow logging
- `tests/test_checkpoint_recovery.py` - Save/load models

### Smoke Tests (Manual)
- Train XGBoost on real data (should complete in <5 min)
- Verify predictions have correct shape and values
- Verify MLflow UI shows metrics
- Verify saved model can be loaded and used

---

## Success Criteria

- [ ] All unit tests pass (pytest coverage >80%)
- [ ] All integration tests pass
- [ ] At least 3 model families implemented (boosting, time series, neural)
- [ ] At least 6 models total (XGB, LightGBM, CatBoost, N-HiTS, TFT, LSTM)
- [ ] CLI scripts work end-to-end
- [ ] MLflow UI shows experiments with metrics
- [ ] Comparison report generated
- [ ] All files respect 650-line limit
- [ ] Documentation complete (architecture, usage, examples)

---

## Deliverables

1. **Code:**
   - `src/models/` - Model implementations
   - `src/data/` - Dataset classes
   - `src/training/` - Trainer, evaluator, callbacks
   - `src/tuning/` - Hyperparameter optimization
   - `scripts/` - CLI entry points

2. **Configuration:**
   - `config/models/` - Model-specific YAML configs
   - `config/experiments/` - Experiment definitions

3. **Tests:**
   - `tests/test_*.py` - Unit and integration tests
   - >80% code coverage

4. **Documentation:**
   - `PHASE2_ARCHITECTURE.md` - System design
   - `PHASE2_ARCHITECTURE_DIAGRAM.md` - Visual diagrams
   - `PHASE2_IMPLEMENTATION_CHECKLIST.md` - This file
   - `docs/PHASE2_USAGE_GUIDE.md` - User guide
   - `docs/PHASE2_MODEL_REFERENCE.md` - Model documentation

5. **Experiments:**
   - `experiments/runs/` - Training run outputs
   - `experiments/mlruns/` - MLflow artifacts
   - Baseline comparison report

---

## Notes

- Stick to 650-line limit per file (fail-fast if exceeded)
- Validate inputs at every boundary
- No exception swallowing - let errors propagate
- Prefer simple, boring solutions
- Test everything before moving to next stage
- Document design decisions in code comments
- Use type hints throughout

