# Experiment Templates

Pre-configured experiment templates for common research workflows.

## Available Templates

| Template | Type | Models | Runtime | Purpose |
|----------|------|--------|---------|---------|
| [Baseline Experiment](baseline_experiment.yaml) | Single model | XGBoost | ~15 min | Quick validation |
| [Full Benchmark](full_benchmark.yaml) | Multi-model | All 13 models | ~12 hours | Comprehensive comparison |

## Template Structure

```yaml
experiment:
  name: {experiment_name}
  description: {description}
  type: {single_model | multi_model}

model:                       # For single_model type
  name: {model_name}
  family: {family}

models:                      # For multi_model type
  boosting: [...]
  neural: [...]
  classical: [...]
  ensemble: [...]

data:
  symbols: [...]
  horizons: [...]

training:
  max_epochs: 100
  # ... training settings

validation:
  run_cv: true
  run_walk_forward: true
  run_cpcv: true

expected:
  runtime_minutes: {time}
  f1_score: {target}

success_criteria:
  - "Pipeline completes without errors"
  - "Validation F1 > {threshold}"
```

## Quick Start

### Run Baseline Experiment
```bash
# Quick validation (15 minutes)
python scripts/train_model.py \
  --model xgboost \
  --horizon 20 \
  --config config/experiments/baseline_experiment.yaml
```

### Run Full Benchmark
```bash
# Full benchmark (12 hours, requires GPU)
python scripts/run_cv.py \
  --models all \
  --horizons 5,10,15,20 \
  --n-splits 5
```

## Creating Custom Experiments

1. Copy a template
2. Modify models, data, validation settings
3. Update expected results and success criteria
4. Run and track results

See [config/INDEX.md](../INDEX.md) for detailed configuration reference.

---

*Last Updated: 2025-12-30*
