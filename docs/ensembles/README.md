# Ensemble Documentation

Comprehensive guides for voting, stacking, and blending ensembles in the ML Model Factory.

---

## Quick Start

### 1. Choose Your Ensemble Method

| Method | Training Time | Performance | Use Case |
|--------|--------------|-------------|----------|
| **Voting** | Fast (single pass) | Good baseline | Quick prototyping, simple averaging |
| **Blending** | Medium (2-3x faster than stacking) | Better than voting | Large datasets, speed matters |
| **Stacking** | Slow (k-fold) | Best performance | Production, maximum accuracy |

**Recommendation:** Start with voting, upgrade to stacking if you need extra performance.

### 2. Choose Your Base Models

**CRITICAL:** All base models must have the same input shape (tabular OR sequence, not mixed).

**Tabular-Only** (2D input):
- Boosting: `xgboost`, `lightgbm`, `catboost`
- Classical: `random_forest`, `logistic`, `svm`

**Sequence-Only** (3D input):
- Neural: `lstm`, `gru`, `tcn`, `transformer`

See `/home/user/Research/docs/ensembles/ENSEMBLE_COMPATIBILITY.md` for details.

### 3. Train Your Ensemble

```bash
# Voting ensemble (fast)
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Blending ensemble (medium)
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Stacking ensemble (slow, best)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

---

## Documentation Index

### Ensemble Method Guides

1. **[VOTING_GUIDE.md](VOTING_GUIDE.md)** - Voting ensemble configurations
   - Soft vs hard voting
   - Weighted voting
   - Low-latency parallel predictions
   - Recommended configurations (boosting trio, RNN variants, etc.)

2. **[STACKING_GUIDE.md](STACKING_GUIDE.md)** - Stacking ensemble configurations
   - Out-of-fold (OOF) predictions
   - Meta-learner selection
   - Leakage prevention
   - PurgedKFold cross-validation
   - Recommended configurations (boosting stack, RNN stack, etc.)

3. **[BLENDING_GUIDE.md](BLENDING_GUIDE.md)** - Blending ensemble configurations
   - Holdout validation approach
   - Time-based splits
   - Retraining strategies
   - Recommended configurations (boosting blend, RNN blend, etc.)

4. **[ENSEMBLE_COMPATIBILITY.md](ENSEMBLE_COMPATIBILITY.md)** - Model compatibility matrix
   - Tabular vs sequence model compatibility
   - Valid and invalid configurations
   - Validation utilities
   - Troubleshooting

### Configuration Templates

Located in `/home/user/Research/config/ensembles/`:

1. **[voting.yaml](../../config/ensembles/voting.yaml)** - Voting ensemble configs
   - Tabular ensembles: boosting trio, boosting+forest, all tabular
   - Sequence ensembles: RNN variants, temporal stack, all neural
   - Low-latency configurations

2. **[stacking.yaml](../../config/ensembles/stacking.yaml)** - Stacking ensemble configs
   - Tabular ensembles: boosting stack, boosting+classical, all tabular
   - Sequence ensembles: RNN stack, all neural
   - Meta-learner variants (logistic, XGBoost)
   - PurgedKFold settings

3. **[blending.yaml](../../config/ensembles/blending.yaml)** - Blending ensemble configs
   - Tabular ensembles: boosting blend, boosting pair, all tabular
   - Sequence ensembles: RNN blend, temporal blend, all neural
   - Holdout fraction variants
   - Retraining configurations

---

## Recommended Configurations

### Tabular Ensembles

#### Fast Baseline (3-5 minutes)

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Expected:** +0.01-0.03 F1 over single models

#### Best Performance (15-25 minutes)

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Expected:** +0.02-0.04 F1 over voting

#### Balanced (5-10 minutes)

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost
```

**Expected:** +0.01-0.03 F1 over voting, 2-3x faster than stacking

### Sequence Ensembles

#### Fast Baseline (15-30 minutes GPU)

```bash
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

**Expected:** +0.02-0.04 F1 over single models

#### Best Performance (45-90 minutes GPU)

```bash
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60
```

**Expected:** +0.03-0.06 F1 over voting

#### Balanced (20-40 minutes GPU)

```bash
python scripts/train_model.py --model blending --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

**Expected:** +0.02-0.04 F1 over voting, 2x faster than stacking

---

## Performance Comparison

| Ensemble | Training Time | Val F1 Lift | Complexity | Use Case |
|----------|--------------|-------------|------------|----------|
| **Voting** | Fast (1x) | Baseline | Simple | Prototyping |
| **Blending** | Medium (2-3x) | +0.01-0.02 | Medium | Large datasets |
| **Stacking** | Slow (5x) | +0.02-0.04 | High | Production |

**Note:** Times are relative to voting baseline. Absolute times depend on base models.

---

## Key Concepts

### Out-of-Fold (OOF) Predictions

**What:** Each sample predicted by a model that never saw that sample during training

**Why:** In-sample predictions are overconfident (overfit). OOF predictions reflect realistic model performance.

**Where:** Used in stacking to train meta-learner on honest predictions

**Example:**
```
5-fold cross-validation:
- Fold 1: Train on folds 2-5, predict fold 1 (OOF)
- Fold 2: Train on folds 1,3-5, predict fold 2 (OOF)
- ...

Result: 100% coverage with OOF predictions
Meta-learner trains on realistic performance
```

### PurgedKFold

**What:** Time-series aware cross-validation with purging and embargo

**Why:** Prevents information leakage from:
1. Overlapping labels (purge)
2. Serial correlation (embargo)

**Settings:**
- `purge_bars`: 60 (3x max horizon=20)
- `embargo_bars`: 1440 (5 days at 5min)

**Used in:** Stacking ensembles for OOF generation

### Meta-Learner

**What:** Second-layer model that combines base model predictions

**Default:** Logistic regression (fast, calibrated, regularized)

**Alternative:** XGBoost (slower, may overfit, better for complex interactions)

**Used in:** Stacking and blending ensembles

### Leakage Prevention

**Problem:** If OOF predictions use tuned hyperparameters, meta-learner trains on optimistically biased predictions → overfit

**Solution:** Stacking uses **default configs for OOF generation**:
```yaml
use_default_configs_for_oof: true  # Default (CRITICAL)
```

**Result:** Meta-learner trains on realistic base model performance

---

## Validation Utilities

### Check Configuration Before Training

```python
from src.models.ensemble import validate_ensemble_config

# Validate a configuration
is_valid, error = validate_ensemble_config(["xgboost", "lightgbm", "lstm"])

if not is_valid:
    print(error)  # Detailed error with suggestions
```

### Get Compatible Models

```python
from src.models.ensemble import get_compatible_models

# Get models compatible with xgboost
tabular_models = get_compatible_models("xgboost")
print(tabular_models)
# ['catboost', 'lightgbm', 'logistic', 'random_forest', 'svm', 'xgboost']

# Get models compatible with lstm
sequence_models = get_compatible_models("lstm")
print(sequence_models)
# ['gru', 'lstm', 'tcn', 'transformer']
```

### Programmatic Validation

```python
from src.models.ensemble import (
    validate_base_model_compatibility,
    EnsembleCompatibilityError,
)

try:
    validate_base_model_compatibility(["xgboost", "lstm"])
except EnsembleCompatibilityError as e:
    print(f"Invalid: {e}")
```

---

## Common Patterns

### Pattern 1: Quick Baseline → Production

```bash
# Step 1: Quick voting baseline (3 min)
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Step 2: If improvement is promising, upgrade to stacking (15 min)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Step 3: Cross-validate best ensemble
python scripts/run_cv.py --models stacking --horizons 20 --n-splits 5
```

### Pattern 2: Rapid Prototyping

```bash
# Test multiple ensemble combinations quickly
for models in "xgboost,lightgbm" "xgboost,catboost" "lightgbm,catboost"; do
  python scripts/train_model.py --model voting --horizon 20 \
    --base-models $models
done

# Pick best, upgrade to stacking
```

### Pattern 3: Separate Tabular and Sequence Experiments

```bash
# Experiment 1: Tabular ensemble
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Experiment 2: Sequence ensemble
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models lstm,gru,tcn --seq-len 60

# Compare results, pick better approach
```

### Pattern 4: Dataset Size Optimization

```bash
# Small dataset (<5k): Use stacking with fewer folds
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lightgbm \
  --config '{"n_folds": 3}'

# Medium dataset (5k-20k): Use blending (default)
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Large dataset (>20k): Use blending with small holdout
python scripts/train_model.py --model blending --horizon 20 \
  --base-models xgboost,lightgbm,catboost \
  --config '{"holdout_fraction": 0.15}'
```

---

## Troubleshooting

### Issue: "Cannot mix tabular and sequence models"

**Error:**
```
EnsembleCompatibilityError: Cannot mix tabular and sequence models.
```

**Solution:** Use only tabular OR only sequence models. See `ENSEMBLE_COMPATIBILITY.md`.

### Issue: Slow training

**Solutions:**
1. Use voting instead of stacking
2. Use blending instead of stacking
3. Reduce number of base models
4. For stacking: reduce `n_folds` (5→3)

### Issue: No improvement over single models

**Possible causes:**
1. Base models too similar (all boosting)
2. Base models overfit (need regularization)
3. Dataset too small (<5k samples)

**Solutions:**
1. Mix model families (boosting + classical)
2. Tune base model hyperparameters
3. Use cross-validation to verify improvement

### Issue: Meta-learner overfits (stacking/blending)

**Symptoms:**
- High train F1, low val F1
- Meta-learner outperforms base models on train but not val

**Solutions:**
1. Use logistic regression (default, more regularized)
2. For stacking: Enable `use_default_configs_for_oof=True`
3. For blending: Increase `holdout_fraction`

### Issue: OOF coverage < 95% (stacking)

**Solutions:**
1. Reduce `purge_bars` or `embargo_bars`
2. Use blending instead (no purging/embargo)
3. Check `label_end_times` are provided correctly

---

## Implementation Details

### Source Code

- **Voting:** `/home/user/Research/src/models/ensemble/voting.py`
- **Stacking:** `/home/user/Research/src/models/ensemble/stacking.py`
- **Blending:** `/home/user/Research/src/models/ensemble/blending.py`
- **Validator:** `/home/user/Research/src/models/ensemble/validator.py`

### OOF Generation

- **Core:** `/home/user/Research/src/cross_validation/oof_core.py`
- **Sequence:** `/home/user/Research/src/cross_validation/oof_sequence.py`
- **Stacking:** `/home/user/Research/src/cross_validation/oof_stacking.py`
- **Validation:** `/home/user/Research/src/cross_validation/oof_validation.py`
- **I/O:** `/home/user/Research/src/cross_validation/oof_io.py`

### Cross-Validation

- **PurgedKFold:** `/home/user/Research/src/cross_validation/purged_kfold.py`
- **Runner:** `/home/user/Research/src/cross_validation/cv_runner.py`

---

## Related Documentation

- **Phase 4 Overview:** `/home/user/Research/docs/phases/PHASE_4.md`
- **Model Registry:** `/home/user/Research/src/models/registry.py`
- **Trainer:** `/home/user/Research/src/models/trainer.py`
- **Base Model:** `/home/user/Research/src/models/base.py`

---

## Next Steps

1. **Read method-specific guide:** Start with `VOTING_GUIDE.md` for quick baseline
2. **Check compatibility:** Review `ENSEMBLE_COMPATIBILITY.md` before training
3. **Load config templates:** Browse configs in `/home/user/Research/config/ensembles/`
4. **Train your first ensemble:** Use CLI examples above
5. **Cross-validate:** Run `scripts/run_cv.py` for robust evaluation

---

## Summary

**Three Ensemble Methods:**
- **Voting:** Simple averaging, fast, good baseline
- **Blending:** Holdout validation, medium speed, better than voting
- **Stacking:** OOF predictions, slow, best performance

**Two Model Families:**
- **Tabular:** 2D input (boosting, classical)
- **Sequence:** 3D input (neural)
- **Cannot mix families** (incompatible input shapes)

**Recommended Starting Point:**
```bash
# Tabular
python scripts/train_model.py --model voting --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Sequence
python scripts/train_model.py --model voting --horizon 20 \
  --base-models lstm,gru --seq-len 60
```

**Performance Expectations:**
- Voting: +0.01-0.03 F1 over single models
- Blending: +0.01-0.03 F1 over voting
- Stacking: +0.02-0.04 F1 over voting

**Key Principles:**
1. ✅ Use same-family ensembles (tabular OR sequence)
2. ✅ Start simple (voting), upgrade if needed (stacking)
3. ✅ Validate configurations before training
4. ✅ Use cross-validation for robust evaluation
5. ✅ For stacking: enable leakage prevention (`use_default_configs_for_oof=true`)
