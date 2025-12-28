# Phase 4: Ensembles (Voting / Stacking / Blending)

## Status: IMPLEMENTED

Ensembles are implemented as model types in the model registry and are trained through the same `Trainer` as single models.

**Code:**
- Ensemble models: `src/models/ensemble/voting.py`, `src/models/ensemble/stacking.py`, `src/models/ensemble/blending.py`
- Training orchestration: `src/models/trainer.py`
- YAML configs: `config/models/voting.yaml`, `config/models/stacking.yaml`, `config/models/blending.yaml`

## CLI usage

```bash
# Voting ensemble
python scripts/train_model.py --model voting --horizon 20
python scripts/train_model.py --model voting --horizon 20 --base-models xgboost,lightgbm,catboost

# Stacking ensemble
python scripts/train_model.py --model stacking --horizon 20

# Blending ensemble
python scripts/train_model.py --model blending --horizon 20
```

## Important notes

- Ensemble composition (base models, meta-learner, folds, etc.) is controlled by the ensemble model config in `config/models/*.yaml` and/or `--config` overrides.
- If you want strict stacking hygiene (meta-learner trained only on OOF predictions), generate OOF predictions first via Phase 3 and then train the stacking model using those artifacts.
