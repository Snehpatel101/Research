# Configuration Index

Comprehensive guide to all configuration files and settings in the ML Model Factory.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Structure](#configuration-structure)
3. [Model Configurations](#model-configurations)
4. [Pipeline Configurations](#pipeline-configurations)
5. [Ensemble Configurations](#ensemble-configurations)
6. [Experiment Templates](#experiment-templates)
7. [Configuration Reference](#configuration-reference)
8. [Validation Guidelines](#validation-guidelines)
9. [Environment-Specific Configs](#environment-specific-configs)
10. [Best Practices](#best-practices)

---

## Quick Start

### Load a Configuration
```python
from src.models.config.loaders import load_model_config, load_training_config

# Load model config
model_config = load_model_config("xgboost")

# Load global training config
training_config = load_training_config()

# Load CV config
cv_config = load_training_config("cv")
```

### Override Configuration Values
```bash
# CLI override
python scripts/train_model.py \
  --model xgboost \
  --override "defaults.n_estimators=1000" \
  --override "defaults.learning_rate=0.01"

# Programmatic override
from src.models.config.merging import merge_configs

base_config = load_model_config("xgboost")
overrides = {
    "defaults": {
        "n_estimators": 1000,
        "learning_rate": 0.01
    }
}
final_config = merge_configs(base_config, overrides)
```

---

## Configuration Structure

### Directory Layout
```
config/
├── models/              # 13 model configurations
├── pipeline/            # Global pipeline settings
├── ensembles/           # Ensemble templates
├── experiments/         # Experiment templates
├── optimization/        # GA optimization results
└── features/            # Feature engineering configs (reserved)
```

### File Naming Conventions
- Model configs: `{model_name}.yaml` (e.g., `xgboost.yaml`)
- Ensemble configs: `{ensemble_description}.yaml` (e.g., `boosting_trio.yaml`)
- Experiment configs: `{experiment_name}.yaml` (e.g., `baseline_experiment.yaml`)
- Use lowercase with underscores (snake_case)

---

## Model Configurations

All model configs follow the same structure:

### Model Config Template
```yaml
# Model identification
model:
  name: {model_name}           # Must match registry name
  family: {family}             # boosting, neural, classical, ensemble
  description: {description}   # Human-readable description

# Default hyperparameters
defaults:
  # Model-specific hyperparameters
  param1: value1
  param2: value2

# Training settings
training:
  feature_set: {boosting_optimal | neural_optimal | classical_optimal}
  random_seed: 42
  batch_size: 256              # For neural models
  max_epochs: 100              # For neural models

# Device settings
device:
  default: auto                # auto, cuda, cpu
  mixed_precision: true        # FP16 training (neural models)
```

### Model Families and Configs

#### 1. Boosting Models (3 models)

**XGBoost** (`config/models/xgboost.yaml`)
```yaml
model:
  name: xgboost
  family: boosting

defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  gamma: 0.1
  reg_alpha: 0.1
  reg_lambda: 1.0
  tree_method: hist          # Required for GPU
```

**LightGBM** (`config/models/lightgbm.yaml`)
```yaml
model:
  name: lightgbm
  family: boosting

defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_samples: 20
  reg_alpha: 0.1
  reg_lambda: 1.0
  boosting_type: gbdt
```

**CatBoost** (`config/models/catboost.yaml`)
```yaml
model:
  name: catboost
  family: boosting

defaults:
  iterations: 500
  depth: 6
  learning_rate: 0.05
  subsample: 0.8
  l2_leaf_reg: 1.0
  border_count: 128
  task_type: GPU             # CPU or GPU
```

#### 2. Neural Models (4 models)

**LSTM** (`config/models/lstm.yaml`)
```yaml
model:
  name: lstm
  family: neural

defaults:
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: false
  sequence_length: 60
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip: 1.0
```

**GRU** (`config/models/gru.yaml`)
```yaml
model:
  name: gru
  family: neural

defaults:
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: false
  sequence_length: 60
  learning_rate: 0.001
```

**TCN** (`config/models/tcn.yaml`)
```yaml
model:
  name: tcn
  family: neural

defaults:
  num_channels: [64, 128, 256]
  kernel_size: 3
  dropout: 0.3
  sequence_length: 60
  learning_rate: 0.001
```

**Transformer** (`config/models/transformer.yaml`)
```yaml
model:
  name: transformer
  family: neural

defaults:
  d_model: 256
  nhead: 8
  num_layers: 4
  dim_feedforward: 1024
  dropout: 0.3
  sequence_length: 60
  learning_rate: 0.0001
```

#### 3. Classical Models (3 models)

**Random Forest** (`config/models/random_forest.yaml`)
```yaml
model:
  name: random_forest
  family: classical

defaults:
  n_estimators: 500
  max_depth: 20
  min_samples_split: 10
  min_samples_leaf: 5
  max_features: sqrt
```

**Logistic Regression** (`config/models/logistic.yaml`)
```yaml
model:
  name: logistic
  family: classical

defaults:
  C: 1.0
  penalty: l2
  solver: saga
  max_iter: 1000
  multi_class: multinomial
```

**SVM** (`config/models/svm.yaml`)
```yaml
model:
  name: svm
  family: classical

defaults:
  C: 1.0
  kernel: rbf
  gamma: scale
  probability: true
```

#### 4. Ensemble Models (3 models)

**Voting** (`config/models/voting.yaml`)
```yaml
model:
  name: voting
  family: ensemble

defaults:
  voting: soft               # soft or hard
  weights: null              # Equal weights

base_models:
  # Specified at runtime via --base-models
  required: true
  min_count: 2
  max_count: 6
  family_constraint: same    # All tabular OR all sequence
```

**Stacking** (`config/models/stacking.yaml`)
```yaml
model:
  name: stacking
  family: ensemble

defaults:
  meta_learner: logistic
  use_oof: true
  passthrough: false
  cv_splits: 5
```

**Blending** (`config/models/blending.yaml`)
```yaml
model:
  name: blending
  family: ensemble

defaults:
  meta_learner: logistic
  holdout_ratio: 0.2
  passthrough: false
```

---

## Pipeline Configurations

### Global Training Config (`pipeline/training.yaml`)

**Data Settings:**
```yaml
data:
  horizon: 20                # Default label horizon
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  purge_bars: 60             # 3 * max_horizon
  embargo_bars: 1440         # ~5 days at 5-min
  default_symbols: [MES, MGC]
```

**Training Defaults:**
```yaml
training:
  batch_size: 256
  max_epochs: 100
  early_stopping_patience: 15
  min_delta: 0.0001
  random_seed: 42
  num_workers: 4
  pin_memory: true
```

**Device Settings:**
```yaml
device:
  default: auto              # auto, cuda, cpu
  mixed_precision: true      # FP16 for faster training
  max_memory_fraction: 0.9
```

**Logging & Checkpointing:**
```yaml
logging:
  level: INFO
  show_progress: true
  log_interval: 100

checkpointing:
  save_checkpoints: true
  keep_n_best: 3
  checkpoint_dir: experiments/checkpoints
```

**Experiment Tracking:**
```yaml
experiment:
  output_dir: experiments/runs
  mlflow_enabled: false
  wandb_enabled: false
```

**Environment Overrides:**
```yaml
environments:
  colab:
    device:
      default: auto
    training:
      batch_size: 512

  local_cpu:
    device:
      default: cpu
    training:
      batch_size: 128

  local_gpu:
    device:
      default: cuda
    training:
      batch_size: 512
```

### Cross-Validation Config (`pipeline/cv.yaml`)

**Time-Series CV:**
```yaml
time_series_cv:
  n_splits: 5
  min_train_samples: 10000
  gap: 60                    # Purge bars
  expanding: true            # Expanding window
```

**Purged K-Fold:**
```yaml
purged_kfold:
  n_splits: 5
  purge_overlap: 60
  embargo: 60
```

**Walk-Forward:**
```yaml
walk_forward:
  enabled: true
  retrain_every_days: 30
  min_train_days: 180
  validation_days: 30
```

**Hyperparameter Tuning:**
```yaml
hyperparameter_tuning:
  framework: optuna
  n_trials: 100
  timeout: 3600              # 1 hour
  pruning: true
  sampler: tpe
```

**Metrics:**
```yaml
metrics:
  primary: f1_macro

  additional:
    - accuracy
    - precision_macro
    - recall_macro
    - roc_auc_ovr

  trading:
    - sharpe_ratio
    - max_drawdown
    - win_rate
```

---

## Ensemble Configurations

### Boosting Trio (`ensembles/boosting_trio.yaml`)

Fast tabular ensemble combining three gradient boosting models.

```yaml
ensemble:
  name: boosting_trio
  method: voting
  description: XGBoost + LightGBM + CatBoost

base_models:
  - xgboost
  - lightgbm
  - catboost

voting:
  weights: null              # Equal weights
  strategy: soft             # Probability averaging

training:
  train_base_models: true
  load_existing: false
  use_cv: true
  n_splits: 5

data:
  horizon: 20
  feature_set: boosting_optimal

expected:
  training_time_minutes: 45
  memory_gb: 8
  baseline_f1: 0.52
```

### Temporal Stack (`ensembles/temporal_stack.yaml`)

Stacking ensemble for sequential pattern learning.

```yaml
ensemble:
  name: temporal_stack
  method: stacking
  description: LSTM + GRU + TCN with meta-learner

base_models:
  - lstm
  - gru
  - tcn

stacking:
  meta_learner: logistic
  use_oof: true
  passthrough: false
  cv_splits: 5
  cv_purge: 60
  cv_embargo: 60

training:
  train_base_models: true
  load_existing: false
  sequence_length: 60

data:
  horizon: 20
  feature_set: neural_optimal

expected:
  training_time_minutes: 180
  memory_gb: 12
  baseline_f1: 0.54
```

---

## Experiment Templates

### Baseline Experiment (`experiments/baseline_experiment.yaml`)

Quick baseline to validate pipeline and data quality.

```yaml
experiment:
  name: baseline_experiment
  description: Fast XGBoost baseline
  type: single_model

model:
  name: xgboost
  family: boosting

data:
  symbols: [MES]
  horizons: [20]

training:
  max_epochs: 50             # Reduced from 100
  early_stopping_patience: 10

validation:
  run_cv: false
  run_walk_forward: false
  run_cpcv: false

expected:
  runtime_minutes: 15
  f1_score: ">= 0.48"
  sharpe_ratio: ">= 0.5"

success_criteria:
  - "Pipeline completes without errors"
  - "Model trains successfully"
  - "Validation F1 > 0.45"
  - "No data leakage detected"
```

### Full Benchmark (`experiments/full_benchmark.yaml`)

Comprehensive benchmark across all 13 models.

```yaml
experiment:
  name: full_benchmark
  description: Benchmark all models
  type: multi_model

models:
  boosting: [xgboost, lightgbm, catboost]
  classical: [random_forest, logistic, svm]
  neural: [lstm, gru, tcn, transformer]
  ensemble: [voting, stacking, blending]

data:
  symbols: [MES]
  horizons: [5, 10, 15, 20]

validation:
  run_cv: true
  cv_splits: 5
  run_walk_forward: true
  wf_retrain_days: 30
  run_cpcv: true
  cpcv_n_paths: 16

metrics:
  classification:
    - accuracy
    - f1_macro
    - roc_auc_ovr
  trading:
    - sharpe_ratio
    - max_drawdown
    - win_rate

output:
  save_predictions: true
  save_models: true
  generate_comparison_report: true
  export_format: [json, csv, html]

expected:
  total_runtime_hours: 12
  best_f1_score: ">= 0.55"
  best_sharpe: ">= 1.0"

resources:
  min_gpu_memory_gb: 12
  min_ram_gb: 32
  recommended_gpu: "RTX 4070 Ti or better"
```

---

## Configuration Reference

### Required Fields

**All Model Configs:**
- `model.name` (string) - Registry name
- `model.family` (enum) - boosting | neural | classical | ensemble
- `model.description` (string) - Human-readable description
- `defaults` (dict) - Model-specific hyperparameters
- `training.feature_set` (string) - Feature set identifier
- `training.random_seed` (int) - Reproducibility seed
- `device.default` (enum) - auto | cuda | cpu

**Neural Models Only:**
- `defaults.sequence_length` (int) - Sequence window size
- `training.batch_size` (int) - Batch size
- `training.max_epochs` (int) - Maximum training epochs

**Ensemble Models Only:**
- `base_models` (list[string]) - List of base model names
- `defaults.meta_learner` (string) - Meta-learner name (stacking/blending)

### Optional Fields

- `defaults.use_gpu` (bool) - Enable GPU acceleration
- `training.early_stopping_patience` (int) - Early stopping patience
- `device.mixed_precision` (bool) - FP16 training
- `validation.*` - Validation settings

### Field Types and Constraints

| Field | Type | Constraints | Default |
|-------|------|-------------|---------|
| `model.name` | string | Must match registry | required |
| `model.family` | enum | boosting, neural, classical, ensemble | required |
| `defaults.n_estimators` | int | >= 1 | 500 |
| `defaults.learning_rate` | float | > 0, <= 1 | 0.05 |
| `defaults.max_depth` | int | >= 1, <= 30 | 6 |
| `defaults.hidden_size` | int | >= 16, power of 2 | 256 |
| `defaults.sequence_length` | int | >= 10, <= 500 | 60 |
| `training.batch_size` | int | >= 1, power of 2 | 256 |
| `training.random_seed` | int | >= 0 | 42 |
| `device.default` | enum | auto, cuda, cpu | auto |

---

## Validation Guidelines

### Automatic Validation

All configs are validated at load time:

```python
from src.models.config.loaders import load_model_config

try:
    config = load_model_config("xgboost")
    print("✓ Config valid")
except ValidationError as e:
    print(f"✗ Config invalid: {e}")
```

### Validation Rules

1. **Required fields present**
   - model.name, model.family, defaults, training, device

2. **Type checking**
   - Numeric fields are numbers
   - String fields are strings
   - Enum fields have valid values

3. **Range constraints**
   - learning_rate > 0
   - n_estimators >= 1
   - sequence_length >= 10

4. **Ensemble compatibility**
   - All base models exist in registry
   - All base models have same input shape (tabular OR sequence)
   - Meta-learner is tabular model

5. **File size limits**
   - Target: 150 lines per config
   - Maximum: 200 lines (with justification)

### Manual Validation Checklist

Before committing a new config:

- [ ] All required fields present
- [ ] Field types correct
- [ ] Values within valid ranges
- [ ] Ensemble compatibility verified (if applicable)
- [ ] File size within limits (< 150 lines)
- [ ] Comments explain non-obvious settings
- [ ] Tested with actual training run
- [ ] No sensitive data (API keys, paths)

---

## Environment-Specific Configs

### Setting Environment

```bash
# Export environment variable
export ML_ENV=colab

# Or pass to script
ML_ENV=local_gpu python scripts/train_model.py --model xgboost --horizon 20
```

### Supported Environments

**colab** - Google Colab (T4/A100 GPU)
```yaml
environments:
  colab:
    device:
      default: auto
      mixed_precision: true
    training:
      num_workers: 2
      batch_size: 512
```

**local_cpu** - Local development (CPU only)
```yaml
environments:
  local_cpu:
    device:
      default: cpu
      mixed_precision: false
    training:
      num_workers: 4
      batch_size: 128
```

**local_gpu** - Local GPU (RTX 4070 Ti, RTX 3090, etc.)
```yaml
environments:
  local_gpu:
    device:
      default: cuda
      mixed_precision: true
    training:
      num_workers: 4
      batch_size: 512
```

### Adding Custom Environments

```yaml
# In config/pipeline/training.yaml
environments:
  my_custom_env:
    device:
      default: cuda
      mixed_precision: true
    training:
      batch_size: 1024
    experiment:
      output_dir: /path/to/custom/output
```

---

## Best Practices

### 1. Configuration Organization
- One config file per model
- Use descriptive names (not `model1.yaml`)
- Group related settings together
- Keep files short (< 150 lines)

### 2. Configuration Reuse
- Use global settings in `training.yaml`
- Override only what's necessary in model configs
- Use experiment templates for common workflows
- Create ensemble configs for reusable combinations

### 3. Version Control
- Commit configs with descriptive messages
- Document breaking changes
- Keep configs in sync with code
- Use git tags for config versions

### 4. Documentation
- Add comments for non-obvious settings
- Include paper references for architectures
- Document expected ranges
- Explain environment-specific overrides

### 5. Validation
- Validate before committing
- Test with actual training run
- Check ensemble compatibility
- Verify file size limits

### 6. Security
- No API keys or secrets in configs
- Use environment variables for sensitive data
- No hardcoded absolute paths
- Use relative paths from project root

---

## Related Documentation

- [Configuration README](README.md) - Quick navigation
- [Model Integration Guide](../docs/guides/MODEL_INTEGRATION_GUIDE.md) - Adding new models
- [Hyperparameter Optimization Guide](../docs/guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) - Tuning strategies
- [Quick Reference](../docs/QUICK_REFERENCE.md) - Command cheatsheet

---

*Last Updated: 2025-12-30*
*Configuration Version: 2.0*
