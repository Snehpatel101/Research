# Phase 2: Model Training (“Model Factory”)

## Status: COMPLETE

Phase 2 trains any registered model against Phase 1’s leakage-safe datasets via a single trainer interface.

**Primary entrypoints**
- Notebook: `notebooks/ML_Pipeline.ipynb` (see `docs/notebook/README.md`)
- CLI: `python scripts/train_model.py ...`

---

## What Exists (Source of Truth)

- Model interface: `src/models/base.py` (`BaseModel`, `TrainingMetrics`, `PredictionOutput`)
- Registry/plugin system: `src/models/registry.py` + `@register(...)` decorators under `src/models/**`
- Trainer orchestration + artifact saving: `src/models/trainer.py`
- Global defaults: `config/training.yaml`
- Per-model defaults: `config/models/*.yaml`

**Registered models (13 total)**
- Boosting: `xgboost`, `lightgbm`, `catboost`
- Neural: `lstm`, `gru`, `tcn`, `transformer`
- Classical: `random_forest`, `logistic`, `svm`
- Ensemble: `voting`, `stacking`, `blending`

---

## Inputs

Phase 2 expects Phase 1 outputs in a “scaled splits” directory (usually not committed to git):

- `data/splits/scaled/train_scaled.parquet`
- `data/splits/scaled/val_scaled.parquet`
- `data/splits/scaled/test_scaled.parquet`
- Optional: `data/splits/scaled/scaling_metadata.json`

These are loaded via `src/phase1/stages/datasets/container.py` (`TimeSeriesDataContainer.from_parquet_dir(...)`).

---

## Training from the CLI

```bash
# List models
python scripts/train_model.py --list-models

# Train a single model (writes to experiments/runs/<run_id>/)
python scripts/train_model.py --model xgboost --horizon 20

# Override paths
python scripts/train_model.py --model lstm --horizon 20 --data-dir data/splits/scaled --output-dir experiments/runs
```

**Artifacts**

Each run is written under `experiments/runs/<run_id>/` (created by `src/models/trainer.py`), including:
- `checkpoints/best_model/` (via `BaseModel.save(...)`)
- `metrics/training_metrics.json`, `metrics/evaluation_metrics.json`
- `predictions/val_predictions.npz`
- `config/training_config.json`, `config/model_config.json`

---

## Model Data Requirements

### Tabular Models (2D Input)

**Models:** `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`

**Input Shape:** `(n_samples, n_features)`

**Data Access:** `container.get_sklearn_arrays("train")` → returns numpy arrays

**Current Features:** All ~180 indicator-derived features (base + MTF indicators)

**Optimal Data (Strategy 2):** MTF indicator features ✅ Currently receiving appropriate data

**Example:**
```python
X_train, y_train, w_train = container.get_sklearn_arrays("train")
# Shape: (50000, 180) - all indicator features
```

### Sequence Models (3D Input)

**Implemented (7):** `lstm`, `gru`, `tcn`, `transformer`

**Planned (6):** `inceptiontime`, `resnet1d`, `patchtst`, `itransformer`, `tft`, `n_beats`

**Total Sequence Models:** 13 (7 implemented + 6 planned)

**Input Shape:** `(n_samples, seq_len, n_features)`

**Data Access:** `container.get_pytorch_sequences("train", seq_len=60)` → returns SequenceDataset

**Current Features:** All ~180 indicator-derived features from 5 of 9 timeframes (windowed into sequences)

**Optimal Data (Strategy 3):** Multi-resolution raw OHLCV bars from 9 timeframes ❌ **Not implemented**

**Current Limitations:**
1. Sequence models receive pre-computed indicators when they should receive raw multi-resolution bars for temporal learning
2. Only 5 of 9 intended timeframes implemented (missing: 1min, 10min, 20min, 25min)

**Example of Current Implementation:**
```python
train_dataset = container.get_pytorch_sequences("train", seq_len=60)
# Shape: (49940, 60, 180) - same indicators, windowed
# (loses 60 samples due to windowing)
```

**Example of Intended Implementation (Strategy 3):**
```python
# This doesn't exist yet:
train_data = container.get_multi_resolution_bars("train", timeframes=['5min', '15min', '1h'])
# Returns dict of tensors:
# {
#   '5min':  torch.Tensor(50000, 60, 4),  # raw OHLCV
#   '15min': torch.Tensor(50000, 20, 4),  # raw OHLCV
#   '1h':    torch.Tensor(50000, 5, 4),   # raw OHLCV
# }
```

### Ensemble Models

**Models:** `voting`, `stacking`, `blending`

**Compatibility Rules:**
- ✅ All-tabular ensembles: `voting(['xgboost', 'lightgbm', 'catboost'])`
- ✅ All-sequence ensembles: `voting(['lstm', 'gru', 'tcn'])`
- ❌ Mixed ensembles: `voting(['xgboost', 'lstm'])` raises `EnsembleCompatibilityError`

**Reason:** Shape mismatch (2D vs 3D inputs) causes training failures

---

## Notes / Known Gaps (for "ML factory" usage)

- Phase 2 does **not** currently produce an inference “bundle” automatically. Phase 5 tooling (`scripts/serve_model.py`, `scripts/batch_inference.py`) expects a `ModelBundle` directory (see `src/inference/bundle.py`), so bundling is a separate step today (often done in the notebook).
- Symbol selection and feature generation are Phase 1 concerns; Phase 2 assumes features/labels already exist in the split parquet files.

---

## Adding a New Model

1. Implement `BaseModel` (`src/models/base.py`).
2. Register it with `@register(name=..., family=...)` (see `src/models/registry.py`).
3. Add a config file in `config/models/`.
4. Add/extend tests under `tests/`.
