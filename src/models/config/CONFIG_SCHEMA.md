# Configuration Schema Documentation

This document describes the configuration system for model training, including all
configuration keys, their sources, precedence rules, and how to verify which
configuration was used for a run.

## Configuration Precedence

Configuration values are merged from multiple sources, with higher-precedence
sources overriding lower ones:

```
1. CLI Arguments        (Highest)  --batch-size 128
2. Explicit Config File            --config my_experiment.yaml
3. Environment Overrides           config/training.yaml environments: section
4. Model-Specific YAML             config/models/{model_name}.yaml
5. Provided Defaults    (Lowest)   Built-in Python defaults
```

### Precedence Rules in Detail

| Priority | Source | Behavior on Error | Example |
|----------|--------|-------------------|---------|
| 1 (Highest) | CLI Arguments | N/A (validated by typer) | `--batch-size 128` |
| 2 | Explicit Config (`--config`) | **FAIL HARD** - User requested this file | `--config my_exp.yaml` |
| 3 | Environment Overrides | Warn and continue | Auto-detected from environment |
| 4 | Model YAML (auto-discovered) | Warn and continue | `config/models/xgboost.yaml` |
| 5 (Lowest) | Built-in Defaults | N/A | Python dataclass defaults |

### Example Override Flow

```python
# Starting with defaults
defaults = {"batch_size": 64, "max_epochs": 100}

# Model YAML overrides defaults
config/models/xgboost.yaml = {"batch_size": 256}
# Result: batch_size = 256

# Environment (GPU) overrides model YAML
config/training.yaml environments.local_gpu = {"batch_size": 512}
# Result: batch_size = 512

# CLI overrides everything
--batch-size 128
# Final Result: batch_size = 128
```

## Configuration Sources

### 1. CLI Arguments

Command-line arguments passed to training scripts. These have the highest priority.

```bash
python scripts/train_model.py \
    --model xgboost \
    --horizon 20 \
    --batch-size 256 \
    --max-epochs 100 \
    --feature-set boosting_optimal
```

### 2. Explicit Config File (`--config`)

User-provided YAML file. If specified and fails to load, training **fails hard**
with a clear error message.

```bash
python scripts/train_model.py --model xgboost --config experiments/my_config.yaml
```

### 3. Environment Overrides

Located in `config/training.yaml` under the `environments:` section. Auto-detected
based on execution environment:

```yaml
# config/training.yaml
environments:
  colab:
    batch_size: 64
    num_workers: 2
    mixed_precision: false

  local_gpu:
    batch_size: 512
    num_workers: 8
    mixed_precision: true

  local_cpu:
    batch_size: 64
    num_workers: 4
    mixed_precision: false
```

**Environment Detection:**
- `colab`: Google Colab environment (detected via `google.colab` import)
- `local_gpu`: Local machine with CUDA available
- `local_cpu`: Local machine without GPU

### 4. Model-Specific YAML

Located at `config/models/{model_name}.yaml`. Auto-discovered based on model name.

```yaml
# config/models/xgboost.yaml
model:
  name: xgboost
  family: boosting

defaults:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.1

training:
  batch_size: 256
  max_epochs: 100
```

### 5. Built-in Defaults

Python dataclass defaults in `TrainerConfig`:

```python
@dataclass
class TrainerConfig:
    model_name: str
    horizon: int = 20
    feature_set: str = "boosting_optimal"
    sequence_length: int = 60
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15
    # ... etc
```

## Configuration Keys Reference

### TrainerConfig Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_name` | str | (required) | Model to train (xgboost, lstm, etc.) |
| `horizon` | int | 20 | Label prediction horizon in bars |
| `feature_set` | str | "boosting_optimal" | Feature subset for training (see below) |
| `sequence_length` | int | 60 | Sequence length for neural models |
| `batch_size` | int | 256 | Training batch size |
| `max_epochs` | int | 100 | Maximum training epochs |
| `early_stopping_patience` | int | 15 | Epochs without improvement before stopping |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `experiment_name` | str | None | Optional experiment name |
| `output_dir` | Path | experiments/runs | Output directory for artifacts |
| `device` | str | "auto" | Device setting (auto/cuda/cpu) |
| `mixed_precision` | bool | True | Use mixed precision training |
| `num_workers` | int | 4 | DataLoader workers |
| `pin_memory` | bool | True | Pin memory for GPU transfers |
| `use_calibration` | bool | True | Apply probability calibration |
| `calibration_method` | str | "auto" | Calibration method (auto/isotonic/sigmoid) |
| `evaluate_test_set` | bool | True | Evaluate on test set (one-shot) |
| `use_feature_selection` | bool | True | Enable per-model feature selection |
| `feature_selection_n_features` | int | 50 | Number of features to select |
| `feature_selection_method` | str | "mda" | Selection method (mda/mdi/hybrid) |
| `feature_selection_cv_splits` | int | 5 | CV splits for stability analysis |

### Feature Set Values

| Value | Description | Typical Features |
|-------|-------------|------------------|
| `"full"` | All available features | ~180 features |
| `"boosting_optimal"` | Optimized for boosting models | ~50 features |
| `"neural_optimal"` | Optimized for LSTM/GRU/TCN | ~43 features |
| `"transformer_raw"` | Raw OHLCV for transformers | ~23 features |
| `"tcn_optimal"` | Optimized for TCN | ~50 features |
| `"patchtst_optimal"` | Optimized for PatchTST | ~23 features |

## Verifying Applied Configuration

### Using environment_info.json

Every training run saves `environment_info.json` in the `config/` subdirectory:

```json
{
  "environment": "local_gpu",
  "timestamp": "2025-01-14T10:30:00.123456",
  "run_id": "xgboost_h20_20250114_103000_123456_a3f9",
  "model_name": "xgboost",
  "device_resolved": "cuda",
  "applied_overrides": {
    "environment": "local_gpu",
    "environment_overrides": {
      "batch_size": 512,
      "mixed_precision": true
    },
    "model_yaml_path": "/path/to/config/models/xgboost.yaml",
    "model_yaml_keys": ["n_estimators", "max_depth", "learning_rate"],
    "explicit_config_path": null,
    "explicit_config_keys": [],
    "cli_overrides": {
      "batch_size": 256
    }
  }
}
```

### Programmatic Access

```python
from src.models.config import get_applied_overrides, build_config

# After calling build_config() or create_trainer_config()
config = build_config(model_name="xgboost", cli_args={"batch_size": 256})

# Get override information
overrides = get_applied_overrides()
print(f"Environment: {overrides['environment']}")
print(f"Env overrides: {overrides['environment_overrides']}")
print(f"CLI overrides: {overrides['cli_overrides']}")
```

## Common Configuration Scenarios

### Scenario 1: Local Development (CPU)

```bash
# Uses local_cpu environment overrides automatically
python scripts/train_model.py --model xgboost --horizon 20
```

### Scenario 2: GPU Training with Custom Batch Size

```bash
# CLI override takes precedence over environment
python scripts/train_model.py --model lstm --horizon 20 --batch-size 128
```

### Scenario 3: Experiment with Custom Config

```bash
# Explicit config file takes precedence over auto-discovered configs
python scripts/train_model.py --model xgboost --config experiments/ablation_study.yaml
```

### Scenario 4: Google Colab

```python
# In Colab notebook - environment auto-detected
from src.models.config import create_trainer_config

config = create_trainer_config(model_name="xgboost", horizon=20)
# batch_size will be 64 (from colab environment override)
```

## Troubleshooting

### Configuration Not Being Applied

1. Check `environment_info.json` for which overrides were actually applied
2. Verify YAML syntax in config files
3. Check environment detection is correct

### Environment Overrides Not Working

1. Verify `config/training.yaml` exists and has valid `environments:` section
2. Check that environment key matches: `colab`, `local_gpu`, or `local_cpu`
3. Review logs for "Applied environment overrides" message

### CLI Override Not Taking Effect

1. Ensure CLI argument name matches exactly (use hyphens, not underscores)
2. Check that value is not None (None values are filtered out)
3. Verify argument is supported by the script

## Related Files

- `src/models/config/merging.py` - Configuration merging logic and precedence
- `src/models/config/trainer_config.py` - TrainerConfig dataclass
- `src/models/config/loaders.py` - YAML loading utilities
- `src/models/config/environment.py` - Environment detection
- `config/training.yaml` - Global training config with environment overrides
- `config/models/*.yaml` - Model-specific configurations
