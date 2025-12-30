---
name: Model Implementation
about: Track implementation of a new model
title: '[MODEL] Implement {ModelName}'
labels: model, enhancement
assignees: ''
---

## Model Specification
- **Model Name:** [e.g., InceptionTime]
- **Model Family:** [boosting | neural | classical | ensemble]
- **Input Type:** [tabular (2D) | sequence (3D)]
- **Reference Paper:** [link or citation]

## Implementation Checklist
- [ ] Create model class implementing `BaseModel` interface
- [ ] Add `@register(name="...", family="...")` decorator
- [ ] Implement `fit()` method with validation data support
- [ ] Implement `predict()` method returning `PredictionOutput`
- [ ] Implement `save()` and `load()` methods
- [ ] Create model configuration file: `config/models/{model_name}.yaml`
- [ ] Add model to appropriate family directory: `src/models/{family}/`
- [ ] Write unit tests: `tests/unit/models/{family}/test_{model_name}.py`
- [ ] Write integration test: `tests/integration/models/test_{model_name}_integration.py`
- [ ] Add GPU support (if applicable)
- [ ] Add mixed precision training (if applicable)
- [ ] Update `ModelRegistry` to verify registration
- [ ] Add to documentation: `docs/phases/PHASE_2.md`
- [ ] Verify model appears in `./train_model.py --list-models`
- [ ] Run smoke test: `pytest tests/integration/models/test_{model_name}_integration.py -v`

## Configuration Template
```yaml
model:
  name: {model_name}
  family: {family}
  description: {description}

defaults:
  # Hyperparameters here

training:
  feature_set: {boosting_optimal | neural_optimal | classical_optimal}
  random_seed: 42

device:
  default: auto
```

## Testing Plan
- [ ] Train on MES symbol with h=20
- [ ] Verify training metrics logged
- [ ] Verify model saved correctly
- [ ] Test predict() on validation data
- [ ] Verify PredictionOutput format
- [ ] Run cross-validation: `python scripts/run_cv.py --models {model_name} --horizons 20 --n-splits 5`

## Expected Performance
- **Training Time:** [e.g., 30 mins on RTX 4070 Ti for 100 epochs]
- **GPU Memory:** [e.g., ~4GB for batch_size=512]
- **Baseline Metric:** [e.g., F1 > 0.50 on validation set]

## Dependencies
- New packages required: [list if any]
- Minimum GPU memory: [e.g., 4GB]
- Recommended hardware: [e.g., RTX 3060+ or T4+]
