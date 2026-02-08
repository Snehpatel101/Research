# ML Factory Architecture Guide

**Last Updated**: 2026-02-08
**Purpose**: Complete architecture reference for ML Factory pipeline

---

## Overview

ML Factory is a config-driven system for building production ML ensembles for financial time-series prediction.

```
Raw OHLCV → Pipeline (12 stages) → Features + Labels → Optimization → Models → Ensemble → Trading Signals
```

### Core Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| No data leakage | Purge/embargo in all CV splits |
| No lookahead | All features use `shift(1)` |
| Reproducible | Same config = same output |
| Realistic metrics | Transaction costs, slippage included |

---

## Model Registry

**Location**: `src/models/registry.py`

### Plugin System

Models are registered using decorators:

```python
@ModelRegistry.register("xgboost", family="boosting")
class XGBoostModel(BaseModel):
    pass

# Create by name:
model = ModelRegistry.create("xgboost", config={"max_depth": 6})

# List all models:
ModelRegistry.list_models()
```

### Registered Models (as of 2026-02-08)

| Family | Models | Data Shape |
|--------|--------|------------|
| **boosting** | xgboost, lightgbm, catboost | 2D `(samples, features)` |
| **neural** | lstm, gru, tcn, nbeats, resnet1d, inceptiontime, tft, transformer | 3D `(samples, seq_len, features)` |
| **transformer** | patchtst, itransformer | 4D `(samples, seq_len, n_vars, patch_len)` |
| **classical** | logistic, random_forest, svm | 2D |
| **ensemble** | stacking, voting, blending | Mixed |
| **meta_learner** | ridge_meta, xgboost_meta, mlp_meta, calibrated_meta | 2D (OOF predictions) |

### Data Shape Routing

The system automatically routes data based on model requirements:

```python
# In src/models/data_preparation.py
if model.requires_sequences:
    if model.requires_4d:
        data = prepare_4d_data(df)  # (batch, seq, vars, patch)
    else:
        data = prepare_3d_data(df)  # (batch, seq, features)
else:
    data = prepare_2d_data(df)      # (batch, features)
```

---

## Optuna Orchestration

**Location**: `src/optimization/pipeline.py`

### 4-Stage Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                 OptimizationPipeline                         │
├─────────────────────────────────────────────────────────────┤
│ Stage 1: Label Optimization                                  │
│   └─ Optuna tunes triple-barrier params (k_up, k_down, etc) │
│   └─ Uses walk-forward validation with purge/embargo        │
│                                                              │
│ Stage 2: Feature Selection                                   │
│   └─ MDA (Mean Decrease Accuracy) importance ranking        │
│   └─ Select top N features per importance threshold         │
│                                                              │
│ Stage 3: Feature Pruning                                     │
│   └─ Remove redundant/correlated features                   │
│   └─ Cluster-based deduplication                            │
│                                                              │
│ Stage 4: Hyperparameter Optimization                         │
│   └─ Model-specific Optuna trials                           │
│   └─ Per-model configs stored for ensemble                  │
└─────────────────────────────────────────────────────────────┘
```

### Walk-Forward Validation

Each optimization stage uses a **different temporal window** to prevent overfitting:

```
Data: |----Q1----|----Q2----|----Q3----|----Q4----|
       
Stage 2: Train=Q1      Val=Q2
Stage 3: Train=Q1+Q2   Val=Q3  
Stage 4: Train=Q1+Q2+Q3 Val=Q4
```

### Key Files

- `src/optimization/pipeline.py` - Main orchestrator
- `src/optimization/labels.py` - Label optimization (triple-barrier)
- `src/optimization/features.py` - Feature selection
- `src/optimization/hyperparameters.py` - Model tuning
- `src/optimization/scoring.py` - Purged scoring functions

---

## Pipeline Stages

**Location**: `src/data/pipeline/stages/`

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Data Loading | Load raw OHLCV from parquet/CSV |
| 2 | Cleaning | Handle missing values, outliers |
| 3 | Feature Engineering | Generate 400+ technical features |
| 4 | MTF (Multi-Timeframe) | Aggregate to multiple resolutions |
| 5 | Labeling | Triple-barrier method |
| 6 | Scaling | Train-only StandardScaler |
| 7 | Regime Detection | Classify market conditions |
| 8 | Splits | Purged train/val/test splits |
| 9 | Report Generation | Summary statistics |
| 10 | Evaluation | Post-training model evaluation |

### Running the Pipeline

```python
from src.data.pipeline import PipelineRunner
from src.config import PipelineConfig

config = PipelineConfig.from_yaml("config/pipeline.yaml")
runner = PipelineRunner(config)

# Run all stages
result = runner.run()

# Or specific stages
features_df = runner.run_stages(start=1, end=8)
```

---

## Ensemble System

**Location**: `src/models/ensemble/`

### Ensemble Types

| Type | Description | Use Case |
|------|-------------|----------|
| **StackingEnsemble** | Meta-learner on OOF predictions | Best accuracy |
| **VotingEnsemble** | Weighted probability averaging | Simple, robust |
| **BlendingEnsemble** | Holdout-based stacking | Fast training |

### Ensemble Training Flow

```python
from src.models.ensemble import EnsembleOrchestrator

orchestrator = EnsembleOrchestrator(config)

# Train base models + generate OOF predictions
# Train meta-learner on OOF
# Return combined ensemble
ensemble = orchestrator.train(X, y, base_model_configs)
```

### Why Models May Be Excluded from Ensemble

1. **Training failure** - Model threw exception during fit
2. **Shape incompatibility** - 2D models can not mix with 4D in stacking
3. **Config whitelist** - `base_model_names` restricts which models to use
4. **OOF generation failure** - Cross-validation failed for that model

---

## 4D Data Support (Transformers)

### When 4D is Used

- **PatchTST**: Patches time series into (batch, n_patches, patch_len, n_vars)
- **iTransformer**: Inverted attention over variables (batch, seq, n_vars, d_model)

### Enabling 4D

```python
# In model config
config = {
    "model_type": "patchtst",
    "patch_len": 16,
    "stride": 8,
    "n_vars": len(feature_columns),  # Number of input variables
}

# Data preparation automatically handles 4D
from src.data.adapters import get_adapter
adapter = get_adapter("patchtst")
X_4d = adapter.transform(df)  # Returns 4D tensor
```

### 4D Shape Reference

```
Standard 3D: (batch_size, sequence_length, n_features)
             e.g., (1000, 60, 100)

PatchTST 4D: (batch_size, n_patches, patch_len, n_vars)
             e.g., (1000, 8, 16, 100)

iTransformer: (batch_size, sequence_length, n_vars, d_model)
              e.g., (1000, 60, 100, 64)
```

---

## Jupyter Notebook Usage

**Location**: `notebooks/ml_factory_colab.ipynb`

### Typical Workflow

```python
# Cell 1: Install dependencies
!pip install -r requirements-colab.txt

# Cell 2: Mount Drive (for data persistence)
from google.colab import drive
drive.mount("/content/drive")

# Cell 3: Load and prepare data
from src.data.pipeline import PipelineRunner
runner = PipelineRunner.from_yaml("config/pipeline.yaml")
features_df = runner.run()

# Cell 4: Optimize
from src.optimization import OptimizationPipeline
optimizer = OptimizationPipeline(config)
best_params = optimizer.run(features_df)

# Cell 5: Train ensemble
from src.models.ensemble import EnsembleOrchestrator
ensemble = EnsembleOrchestrator(config)
result = ensemble.train(features_df, best_params)

# Cell 6: Evaluate
from src.inference import Backtester
backtester = Backtester(ensemble)
metrics = backtester.run(test_df)
```

---

## Google Colab Setup

### Requirements

1. **Runtime**: GPU (T4 minimum, A100 recommended for transformers)
2. **RAM**: High-RAM if using full pipeline (12GB+ standard may OOM)
3. **Storage**: Mount Google Drive for data and checkpoints

### Installation

```bash
# Option 1: Full install
!pip install -r requirements-colab.txt

# Option 2: Core only
!pip install torch xgboost lightgbm catboost optuna pandas numpy scikit-learn PyWavelets numba
```

### Colab-Specific Considerations

| Issue | Solution |
|-------|----------|
| 90-min timeout | Add checkpointing after each stage |
| RAM limits | Use `gc.collect()` between stages |
| GPU OOM | Reduce batch_size, use mixed precision |
| Disconnects | Save state to Drive frequently |

### Checkpointing Example

```python
import pickle
from pathlib import Path

CHECKPOINT_DIR = Path("/content/drive/MyDrive/ml_factory_checkpoints")

def save_checkpoint(name, obj):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CHECKPOINT_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)

def load_checkpoint(name):
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Usage
features_df = load_checkpoint("features") or runner.run_stages(1, 8)
save_checkpoint("features", features_df)
```

---

## Project Structure

```
Research/
├── config/                 # Configuration files
│   ├── global.yaml        # Global settings
│   └── pipeline.yaml      # Pipeline configuration
├── data/
│   ├── raw/               # Raw OHLCV data
│   └── processed/         # Intermediate outputs
├── notebooks/
│   └── ml_factory_colab.ipynb
├── src/
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration classes
│   ├── core/              # Types, contracts, base interfaces
│   ├── data/              # Adapters, features, pipeline, labeling
│   │   ├── adapters/      # Data shape transformers
│   │   ├── pipeline/      # Pipeline stages
│   │   └── store/         # Caching, persistence
│   ├── models/            # All model implementations
│   │   ├── boosting/      # XGBoost, LightGBM, CatBoost
│   │   ├── neural/        # LSTM, GRU, TCN, etc.
│   │   ├── ensemble/      # Stacking, Voting, Blending
│   │   └── training/      # Training services
│   ├── optimization/      # Optuna, feature selection
│   ├── validation/        # Leakage detection, CV
│   └── inference/         # Backtesting, prediction
├── tests/                 # Test suite
├── experiments/           # Training outputs
├── requirements.txt       # Local dependencies
├── requirements-colab.txt # Colab dependencies
└── ARCHITECTURE.md        # This file
```

---

## Quick Reference Commands

```bash
# Run full pipeline
python -m src.cli run --config config/pipeline.yaml

# Run specific stages
python -m src.cli run --stages 1-8

# Train single model
python -m src.cli train --model xgboost --data data/processed/features.parquet

# Run tests
python -m pytest tests/ -v

# Type checking
python -m mypy src/ --ignore-missing-imports

# Linting
ruff check src/ --fix
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Check venv activation, run `pip install -e .` |
| Shape mismatch | Verify model data rank (2D/3D/4D) matches data |
| OOM during training | Reduce batch_size, enable gradient checkpointing |
| Slow feature computation | Enable numba JIT, use cached features |
| Circular import | Use lazy imports or `TYPE_CHECKING` guard |
| Leaky validation | Ensure purge_bars and embargo_bars are set |

---

*Generated by Johnny (OpenClaw) - 2026-02-08*
