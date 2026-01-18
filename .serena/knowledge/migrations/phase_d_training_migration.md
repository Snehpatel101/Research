# Phase D: Training Migration (Stages 13-15)

**Status:** Planning Complete
**Estimated Effort:** 6 days

---

## Current State Summary

### Stage 13: Hyperparameter Optimization (2,300 trials)
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/optimization/hyperparameters.py` | ~800 | HyperparameterOptimizer with 23 model spaces |
| `src/cross_validation/cv_tuner.py` | ~600 | TimeSeriesOptunaTuner with DSR |

### Stage 14: Training
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/models/trainer.py` | Re-exports | Trainer class |
| `src/models/training/` | ~2,000 | Core training implementation |
| `src/training/orchestrator.py` | ~500 | TrainingOrchestrator |

### Stage 15: Stacking
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/cross_validation/oof_stacking.py` | ~800 | HeterogeneousStackingBuilder |
| `src/cross_validation/oof_generator.py` | ~400 | OOFGenerator |
| `src/models/ensemble/stacking.py` | ~600 | StackingEnsemble |
| `src/models/ensemble/ridge_meta.py` | ~300 | RidgeMetaLearner |

---

## Target State

### In `src/pipeline/phases/training.py`

```python
class Stage13HyperparameterOptimization:
    """100 trials per model = 2,300 total trials."""

    def run(self, state: PipelineState) -> StageResult:
        for model_name in self.config.models:
            X, y = self._get_model_data(state, model_name)
            result = self.optimizer.optimize(model_name, X, y)
            state.optuna_studies[f"hyperparam_{model_name}"] = result.study

class Stage14Training:
    """Train 23 models with OOF generation."""

    def run(self, state: PipelineState) -> StageResult:
        for model_name in self.config.models:
            if self._is_meta_learner(model_name):
                continue  # Meta-learners in Stage 15

            best_params = state.stage_outputs[13][model_name].best_params
            model = ModelRegistry.create(model_name, config=best_params)
            oof_pred = self.oof_generator.generate(X, y, model)

            state.trained_models[model_name] = model
            state.oof_predictions[model_name] = oof_pred

class Stage15Stacking:
    """4 meta-learners stacking 3-4 heterogeneous bases."""

    def run(self, state: PipelineState) -> StageResult:
        # Build stacking dataset from OOF
        X_stack = self.stacking_builder.build_stacking_dataset(
            oof_results=self._select_base_models(state.oof_predictions)
        )

        # Train each meta-learner
        for meta_name in ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"]:
            meta = ModelRegistry.create(meta_name)
            meta.fit(X_stack_train, y_train)
            state.meta_learners[meta_name] = meta
```

---

## Model Registry Integration

### 23 Models by Family
| Family | Count | Models | Adapter |
|--------|-------|--------|---------|
| Boosting | 3 | xgboost, lightgbm, catboost | Tabular |
| Classical | 3 | random_forest, logistic, svm | Tabular |
| Neural Basic | 4 | lstm, gru, tcn, transformer | Sequence |
| Neural Advanced | 6 | patchtst, itransformer, tft, nbeats, inceptiontime, resnet1d | MultiRes |
| Ensemble | 3 | voting, stacking, blending | N/A |
| Meta-Learners | 4 | ridge_meta, mlp_meta, xgboost_meta, calibrated_meta | Tabular |

### Trial Budget
| Stage | Target | Trials |
|-------|--------|--------|
| 13 | Per-model hyperparams | 100 × 23 = 2,300 |
| 15 | Meta-learner + base selection | 50 |
| **Total** | | **2,350** |

---

## OOF Generation Flow

```
For each base model (19 models):
  5-Fold PurgedKFold CV
  ├── Fold 1: Train [2,3,4,5] → Predict [1] → OOF[1]
  ├── Fold 2: Train [1,3,4,5] → Predict [2] → OOF[2]
  ├── Fold 3: Train [1,2,4,5] → Predict [3] → OOF[3]
  ├── Fold 4: Train [1,2,3,5] → Predict [4] → OOF[4]
  └── Fold 5: Train [1,2,3,4] → Predict [5] → OOF[5]

  Result: OOF predictions for ALL training samples
  Shape: (N_train, 3) probabilities
```

---

## Meta-Learner Architecture

### Four Meta-Learners
| Meta | Type | Speed | Best For |
|------|------|-------|----------|
| ridge_meta | Linear | <1s | Default, fast |
| mlp_meta | Neural | 5-30s | Complex interactions |
| xgboost_meta | Boosting | 10-60s | Maximum power |
| calibrated_meta | Linear+Calibration | 2-5s | Probability calibration |

### Base Model Selection (3-4 heterogeneous)
```yaml
base_models:
  tabular: catboost      # Best from boosting
  cnn: tcn               # Best from CNN
  transformer: patchtst  # Best from transformers
  optional_4th: null     # OR nbeats, ridge
```

### Stacking Input Shape
```
Base OOF:
  CatBoost: (N, 3) + TCN: (N, 3) + PatchTST: (N, 3)
  = Concatenated: (N, 9) + derived features (4)
  = Final X_stack: (N, 13)
```

---

## Interface Contracts

| Stage | Input | Output |
|-------|-------|--------|
| 13 | Adapted tensors (Stage 12) | `{model}_best_params.json` × 23 |
| 14 | Best params (Stage 13) | Trained models + OOF predictions |
| 15 | OOF predictions (Stage 14) | 4 meta-learners + stacking report |

**Checkpoint:** `experiments/runs/{run_id}/models/`

---

## Migration Steps

1. **Stage 13** (2 days): Wrap HyperparameterOptimizer
2. **Stage 14** (2 days): Wrap Trainer + OOFGenerator
3. **Stage 15** (1.5 days): Wrap HeterogeneousStackingBuilder
4. **Testing** (0.5 days): End-to-end training validation

---

## Critical Files

1. `src/optimization/hyperparameters.py` - 23 model search spaces
2. `src/models/trainer.py` - Training workflow
3. `src/cross_validation/oof_generator.py` - OOF generation
4. `src/cross_validation/oof_stacking.py` - Heterogeneous stacking
5. `src/models/ensemble/stacking.py` - StackingEnsemble
