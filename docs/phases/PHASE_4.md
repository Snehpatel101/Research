# Phase 4: Ensembles (Voting / Stacking / Blending)

## Status: IMPLEMENTED

Ensembles are implemented as model types in the model registry and are trained through the same `Trainer` as single models.

**Code:**
- Ensemble models: `src/models/ensemble/voting.py`, `src/models/ensemble/stacking.py`, `src/models/ensemble/blending.py`
- Ensemble validator: `src/models/ensemble/validator.py`
- Training orchestration: `src/models/trainer.py`
- YAML configs: `config/models/voting.yaml`, `config/models/stacking.yaml`, `config/models/blending.yaml`

---

## Ensemble Model Compatibility

### Critical Limitation: Tabular vs Sequence Models

**Ensembles CANNOT mix tabular and sequence models** due to incompatible input shapes:

- **Tabular models** expect 2D input: `(n_samples, n_features)`
  - Boosting: `xgboost`, `lightgbm`, `catboost`
  - Classical: `random_forest`, `logistic`, `svm`

- **Sequence models** expect 3D input: `(n_samples, seq_len, n_features)`
  - Neural: `lstm`, `gru`, `tcn`, `transformer`

**Mixing these families will cause shape mismatches and training failures.**

### Validation Behavior

All ensemble classes (`VotingEnsemble`, `StackingEnsemble`, `BlendingEnsemble`) now validate base model compatibility:

1. **At training time**: `fit()` method validates `base_model_names` before training
2. **At model setup**: `set_base_models()` validates pre-trained model instances
3. **Error type**: Raises `EnsembleCompatibilityError` with detailed error message

Example validation error:
```
Ensemble Compatibility Error: Cannot mix tabular and sequence models.

REASON:
  - Tabular models expect 2D input: (n_samples, n_features)
  - Sequence models expect 3D input: (n_samples, seq_len, n_features)
  - Mixed ensembles would cause shape mismatches during training/prediction

YOUR CONFIGURATION:
  Tabular models (2D): ['xgboost', 'lightgbm']
  Sequence models (3D): ['lstm']

SUPPORTED ENSEMBLE CONFIGURATIONS:
✅ All Tabular Models:
  - Example: base_model_names=['xgboost', 'lightgbm', 'random_forest']

✅ All Sequence Models:
  - Example: base_model_names=['lstm', 'gru', 'tcn']

❌ Mixed Models (NOT SUPPORTED):
  - Example: base_model_names=['xgboost', 'lstm']  # WILL FAIL
```

---

## Supported Ensemble Configurations

### Tabular-Only Ensembles (Recommended)

| Configuration | Models | Use Case | Expected Performance |
|--------------|---------|----------|---------------------|
| Boosting Trio | `xgboost`, `lightgbm`, `catboost` | Fast, tree-based diversity | Strong baseline |
| Boosting + Forest | `xgboost`, `lightgbm`, `random_forest` | Tree diversity + bagging | Balanced accuracy/speed |
| All Tabular | `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm` | Maximum diversity | High variance risk |

### Sequence-Only Ensembles

| Configuration | Models | Use Case | Expected Performance |
|--------------|---------|----------|---------------------|
| RNN Variants | `lstm`, `gru` | Temporal pattern diversity | Good for trend following |
| Temporal Stack | `lstm`, `gru`, `tcn` | Multi-architecture temporal | Better generalization |
| All Neural | `lstm`, `gru`, `tcn`, `transformer` | Maximum temporal diversity | Resource intensive |

### INVALID Configurations (Will Fail)

❌ **DO NOT USE THESE:**

```bash
# WILL FAIL: Mixing boosting + neural
python scripts/train_model.py --model voting \
  --base-models xgboost,lightgbm,lstm

# WILL FAIL: Mixing tabular + sequence
python scripts/train_model.py --model stacking \
  --base-models xgboost,random_forest,gru,tcn

# WILL FAIL: Mixing classical + sequence
python scripts/train_model.py --model blending \
  --base-models logistic,svm,transformer
```

---

## CLI Usage

### Voting Ensemble

```bash
# Tabular models (soft voting)
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence models (hard voting)
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru,tcn \
  --config '{"voting": "hard"}'
```

### Stacking Ensemble

```bash
# Tabular models with logistic meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,random_forest

# Sequence models with logistic meta-learner
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn \
  --config '{"meta_learner_name": "logistic", "n_folds": 5}'
```

### Blending Ensemble

```bash
# Tabular models with 20% holdout
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence models with custom holdout fraction
python scripts/train_model.py --model blending --horizon 20 \
  --base-models lstm,gru \
  --config '{"holdout_fraction": 0.3}'
```

---

## Validation Utilities

### Check Model Compatibility

```python
from src.models.ensemble import validate_ensemble_config, get_compatible_models

# Validate a configuration
is_valid, error = validate_ensemble_config(["xgboost", "lightgbm", "lstm"])
if not is_valid:
    print(error)  # Shows detailed error with suggestions

# Get compatible models for a reference model
tabular_models = get_compatible_models("xgboost")
print(tabular_models)
# ['catboost', 'lightgbm', 'logistic', 'random_forest', 'svm', 'xgboost']

sequence_models = get_compatible_models("lstm")
print(sequence_models)
# ['gru', 'lstm', 'tcn', 'transformer']
```

### Programmatic Validation

```python
from src.models.ensemble import validate_base_model_compatibility, EnsembleCompatibilityError

try:
    validate_base_model_compatibility(["xgboost", "lstm"])
except EnsembleCompatibilityError as e:
    print(f"Invalid ensemble: {e}")
    # Prints detailed error with compatible alternatives
```

---

## Future Enhancements

### Hybrid Ensembles (Not Currently Supported)

To support mixed tabular + sequence ensembles in the future, we would need:

1. **Architecture changes:**
   - Dual data preparation paths (2D for tabular, 3D for sequence)
   - Model-specific input shaping in ensemble predict()
   - Separate feature sets per model type

2. **Implementation complexity:**
   - Increased code complexity and maintenance burden
   - Higher risk of data leakage bugs
   - More difficult debugging

3. **Uncertain benefits:**
   - No empirical evidence that mixing families improves performance
   - May increase overfitting due to heterogeneous predictions
   - Resource intensive (must maintain both data formats)

**Current recommendation:** Use same-family ensembles for proven benefits and simpler architecture.

---

## Important Notes

- Ensemble composition (base models, meta-learner, folds, etc.) is controlled by the ensemble model config in `config/models/*.yaml` and/or `--config` overrides.
- If you want strict stacking hygiene (meta-learner trained only on OOF predictions), generate OOF predictions first via Phase 3 and then train the stacking model using those artifacts.
- All ensemble methods use the same validation logic to prevent incompatible model combinations.
- Validation happens at training time, providing clear error messages before expensive training begins.
