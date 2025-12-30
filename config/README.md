# Configuration Directory

This directory contains all configuration files for the ML Model Factory.

## Quick Navigation

| Directory | Description | Files |
|-----------|-------------|-------|
| [models/](models/) | Model-specific configurations | 13 model configs (XGBoost, LSTM, etc.) |
| [pipeline/](pipeline/) | Pipeline and training settings | training.yaml, cv.yaml, scaling_stats.json |
| [ensembles/](ensembles/) | Ensemble configurations | boosting_trio.yaml, temporal_stack.yaml |
| [experiments/](experiments/) | Experiment templates | baseline_experiment.yaml, full_benchmark.yaml |
| [optimization/](optimization/) | GA optimization results | ga_results/ (symbol-specific) |
| [features/](features/) | Feature engineering configs | (reserved for future use) |

## Configuration Structure

```
config/
├── README.md                           # This file
├── INDEX.md                            # Comprehensive configuration guide
│
├── models/                             # Model configurations (13 models)
│   ├── xgboost.yaml
│   ├── lightgbm.yaml
│   ├── catboost.yaml
│   ├── lstm.yaml
│   ├── gru.yaml
│   ├── tcn.yaml
│   ├── transformer.yaml
│   ├── random_forest.yaml
│   ├── logistic.yaml
│   ├── svm.yaml
│   ├── voting.yaml
│   ├── stacking.yaml
│   └── blending.yaml
│
├── pipeline/                           # Pipeline settings
│   ├── training.yaml                   # Global training config
│   ├── cv.yaml                         # Cross-validation config
│   └── scaling_stats.json              # Scaling statistics (auto-generated)
│
├── ensembles/                          # Ensemble configurations
│   ├── boosting_trio.yaml              # XGBoost + LightGBM + CatBoost
│   └── temporal_stack.yaml             # LSTM + GRU + TCN stacking
│
├── experiments/                        # Experiment templates
│   ├── baseline_experiment.yaml        # Quick baseline validation
│   └── full_benchmark.yaml             # Full 13-model benchmark
│
├── optimization/                       # Optimization artifacts
│   └── ga_results/                     # GA optimization results
│       ├── MES_ga_h5_best.json
│       ├── MES_ga_h10_best.json
│       ├── MES_ga_h15_best.json
│       ├── MES_ga_h20_best.json
│       ├── MGC_ga_h5_best.json
│       ├── MGC_ga_h10_best.json
│       ├── MGC_ga_h15_best.json
│       └── MGC_ga_h20_best.json
│
└── features/                           # Feature configs (reserved)
    └── (future feature engineering configs)
```

## Quick Start

### 1. Train a Single Model
Use default model configuration:
```bash
python scripts/train_model.py --model xgboost --horizon 20
```

Override config values:
```bash
python scripts/train_model.py \
  --model xgboost \
  --horizon 20 \
  --config config/models/xgboost.yaml \
  --override "defaults.n_estimators=1000" \
  --override "defaults.max_depth=8"
```

### 2. Train an Ensemble
```bash
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20
```

### 3. Run Cross-Validation
```bash
python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --n-splits 5
```

### 4. Full Benchmark
```bash
python scripts/run_cv.py \
  --models all \
  --horizons 5,10,15,20 \
  --tune
```

## Configuration Files

### Model Configurations
See [models/README.md](models/README.md) for detailed model-specific settings.

**Model Families:**
- **Boosting:** XGBoost, LightGBM, CatBoost (fast, interpretable)
- **Neural:** LSTM, GRU, TCN, Transformer (temporal dependencies)
- **Classical:** Random Forest, Logistic, SVM (robust baselines)
- **Ensemble:** Voting, Stacking, Blending (meta-learning)

### Pipeline Configurations
See [pipeline/README.md](pipeline/README.md) for global training and CV settings.

**Key Files:**
- `training.yaml` - Global training settings (batch size, epochs, device)
- `cv.yaml` - Cross-validation settings (n_splits, purge, embargo)
- `scaling_stats.json` - Auto-generated scaling statistics

### Ensemble Configurations
See [ensembles/README.md](ensembles/README.md) for ensemble templates.

**Available Ensembles:**
- `boosting_trio.yaml` - Fast tabular ensemble (XGB + LGB + CAT)
- `temporal_stack.yaml` - Sequence stacking (LSTM + GRU + TCN)

### Experiment Templates
See [experiments/README.md](experiments/README.md) for experiment templates.

**Available Templates:**
- `baseline_experiment.yaml` - Quick validation (15 mins)
- `full_benchmark.yaml` - Comprehensive benchmark (12 hours)

## Configuration Validation

All configurations are validated at load time. Common validation rules:

1. **Model compatibility** - Ensemble base models must have same input shape
2. **File size limits** - Target 650 lines, max 800 lines
3. **Required fields** - model.name, model.family, defaults.*
4. **Type checking** - Numeric values, valid enums, path existence
5. **Range constraints** - learning_rate > 0, n_estimators >= 1, etc.

### Validate a Configuration
```python
from src.models.config.loaders import load_model_config

# Load and validate
config = load_model_config("xgboost")
print(f"Model: {config['model']['name']}")
print(f"Family: {config['model']['family']}")
```

## Environment-Specific Overrides

Configurations support environment-specific overrides:

```yaml
# training.yaml
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
```

Set environment via:
```bash
export ML_ENV=colab
python scripts/train_model.py --model xgboost --horizon 20
```

## Adding New Configurations

### Adding a New Model Config
1. Create `config/models/{model_name}.yaml`
2. Follow template structure (see INDEX.md)
3. Validate required fields
4. Add model implementation in `src/models/{family}/`
5. Register via `@register(name="...", family="...")`

### Adding a New Ensemble Config
1. Create `config/ensembles/{ensemble_name}.yaml`
2. Specify base models (same family required)
3. Configure ensemble method (voting/stacking/blending)
4. Test compatibility

### Adding an Experiment Template
1. Create `config/experiments/{experiment_name}.yaml`
2. Define models, data, training, validation
3. Specify success criteria
4. Document expected runtime and resources

## Best Practices

### Configuration Organization
- One config file per model
- Use descriptive names (e.g., `boosting_trio.yaml`, not `ens1.yaml`)
- Include comments explaining non-obvious settings
- Keep files under 150 lines when possible

### Configuration Reuse
- Use global settings in `training.yaml` for defaults
- Override only what's necessary in model-specific configs
- Use experiment templates for common workflows

### Version Control
- Commit configuration changes with descriptive messages
- Document breaking changes in commit message
- Keep configs in sync with code

### Documentation
- Update README when adding new config files
- Document expected ranges for hyperparameters
- Include paper references for model architectures

## Troubleshooting

### Common Issues

**Issue:** Config file not found
```
FileNotFoundError: config/models/mymodel.yaml
```
**Solution:** Ensure file exists and path is correct. Use `--list-models` to see available models.

**Issue:** Invalid config format
```
ValidationError: Missing required field 'model.family'
```
**Solution:** Check config against template in INDEX.md. Ensure all required fields present.

**Issue:** Ensemble compatibility error
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models
```
**Solution:** Ensure all base models have same input shape (all tabular OR all sequence).

**Issue:** Environment override not applied
```
Expected batch_size=512 (colab), got batch_size=256 (default)
```
**Solution:** Set `ML_ENV=colab` environment variable before running.

## Related Documentation

- [Configuration Index (INDEX.md)](INDEX.md) - Comprehensive config guide
- [Model Integration Guide](../docs/guides/MODEL_INTEGRATION_GUIDE.md) - Adding new models
- [Hyperparameter Optimization Guide](../docs/guides/HYPERPARAMETER_OPTIMIZATION_GUIDE.md) - Tuning strategies
- [Quick Reference](../docs/QUICK_REFERENCE.md) - Command cheatsheet

## Questions?

- Check [INDEX.md](INDEX.md) for detailed configuration reference
- Review [docs/guides/](../docs/guides/) for implementation guides
- Open an issue for configuration-related questions

---

*Last Updated: 2025-12-30*
*Configuration Version: 2.0*
