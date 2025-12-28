# Pipeline Validation Checklist

Comprehensive checklists for ensuring ML pipeline correctness before and after training. Copy and use these checklists to validate your experimental workflow.

**Last Updated:** 2025-12-28

---

## Pre-Training Checklist

Use this checklist **before running expensive training jobs** to catch configuration errors early.

### Phase 1: Data Pipeline Validation

#### Data Quality
- [ ] Raw OHLCV data exists in `data/raw/{symbol}_1m.parquet` or `.csv`
- [ ] Data covers sufficient time period (minimum 6 months recommended)
- [ ] No excessive gaps in time series (check with gap detection)
- [ ] OHLCV columns present: `['open', 'high', 'low', 'close', 'volume']`
- [ ] Timestamps are properly formatted and sorted
- [ ] No NaN values in OHLCV columns

#### Pipeline Configuration
- [ ] Symbol is correctly specified (`MES`, `MGC`, `SI`, etc.)
- [ ] Target timeframe is set (default: `5min`)
- [ ] Label horizons configured: `[5, 10, 15, 20]`
- [ ] Purge/embargo auto-scaling enabled (default: yes)
- [ ] Project root points to repo root (not `src/`)
- [ ] Output directories configured correctly:
  - [ ] `data/splits/`
  - [ ] `data/splits/scaled/`
  - [ ] `results/`

#### Feature Engineering
- [ ] Feature sets enabled in config (default: 150+ features)
- [ ] Multi-timeframe analysis configured (5min to daily)
- [ ] Wavelet features enabled (if desired)
- [ ] Microstructure features enabled (if desired)

#### Labeling
- [ ] Triple-barrier method configured
- [ ] Symbol-specific barriers set (e.g., MES: 1.5:1.0)
- [ ] Min position duration configured (default: 5 bars)
- [ ] Max position duration configured (default: max_horizon)
- [ ] Optuna optimization enabled (recommended)
- [ ] Transaction cost penalty included (recommended: 0.5 bps)

#### Splits
- [ ] Split ratios configured (default: 70/15/15)
- [ ] Purge configured (default: max_horizon * 3 = 60 bars)
- [ ] Embargo configured (default: 1440 bars ~5 days)
- [ ] No overlap between train/val/test sets

#### Run Pipeline
```bash
# Validate configuration
python -c "from src.phase1.pipeline_config import PipelineConfig; c = PipelineConfig(); print(c)"

# Run Phase 1 pipeline
./pipeline run --symbols MES

# Check outputs
ls data/splits/scaled/
ls results/
```

---

### Phase 2: Model Training Validation

#### Model Selection
- [ ] Model is registered and available:
  ```bash
  python scripts/train_model.py --list-models
  ```
- [ ] Model family chosen (boosting/neural/classical/ensemble)
- [ ] GPU availability checked for neural models:
  ```bash
  nvidia-smi  # Should show available GPU
  ```

#### Configuration
- [ ] Horizon specified and matches Phase 1 horizons
- [ ] Model-specific parameters configured:
  - **Boosting:** `n_estimators`, `max_depth`, `learning_rate`
  - **Neural:** `seq_len`, `hidden_size`, `num_layers`, `dropout`
  - **Classical:** `n_estimators` (RF), `C` (SVM), regularization
- [ ] Sequence length configured for neural models (e.g., 30, 60, 90)
- [ ] Output directory exists: `experiments/runs/`

#### Data Compatibility
- [ ] Phase 1 completed successfully
- [ ] Scaled datasets exist in `data/splits/scaled/`
- [ ] Datasets contain correct horizon labels
- [ ] For neural models: sequence length < training samples

#### Run Training
```bash
# Validate model config
python scripts/train_model.py --model xgboost --horizon 20 --dry-run

# Train model
python scripts/train_model.py --model xgboost --horizon 20

# Check outputs
ls experiments/runs/xgboost_h20_*/
```

---

### Phase 3: Cross-Validation Validation

#### CV Configuration
- [ ] Number of folds configured (recommended: 5)
- [ ] Purge/embargo match Phase 1 settings
- [ ] Models to validate are registered
- [ ] Horizons match Phase 1 horizons
- [ ] PurgedKFold enabled (default: yes)

#### Optuna Tuning (Optional)
- [ ] Number of trials configured (recommended: 50-100)
- [ ] Search space defined for each model
- [ ] Objective metric chosen (default: f1_weighted)
- [ ] Timeout configured if needed

#### Output Configuration
- [ ] Output directory configured (default: `data/stacking/`)
- [ ] Unique run ID or custom name specified
- [ ] Stacking data generation enabled (if planning ensembles)

#### Run CV
```bash
# Validate CV config
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 3 --dry-run

# Run CV
python scripts/run_cv.py \
  --models xgboost,lightgbm \
  --horizons 20 \
  --n-splits 5 \
  --tune \
  --n-trials 50 \
  --output-name "test_cv"

# Check outputs
ls data/stacking/test_cv/
cat data/stacking/test_cv/cv_results.json
```

---

### Phase 4: Ensemble Validation

#### Ensemble Type Selection
- [ ] Ensemble type chosen (voting/stacking/blending)
- [ ] Meta-learner configured for stacking/blending

#### Base Model Compatibility ⚠️ CRITICAL
- [ ] **All base models are same family (tabular OR sequence)**
  - ✅ Valid: `xgboost,lightgbm,catboost` (all tabular)
  - ✅ Valid: `lstm,gru,tcn` (all sequence)
  - ❌ Invalid: `xgboost,lstm` (mixed tabular + sequence)
- [ ] Validate ensemble config:
  ```python
  from src.models.ensemble import validate_ensemble_config
  is_valid, error = validate_ensemble_config(["xgboost", "lightgbm"])
  assert is_valid, error
  ```

#### Phase 3→4 Integration (Recommended)
- [ ] Phase 3 CV completed successfully
- [ ] CV run ID noted from Phase 3 output
- [ ] Stacking datasets exist:
  ```bash
  ls data/stacking/<cv_run_id>/stacking/
  ```
- [ ] Horizon matches between CV and ensemble training
- [ ] `--stacking-data` argument prepared

#### Configuration
- [ ] Number of folds for stacking (if not using Phase 3 data)
- [ ] Passthrough features enabled/disabled
- [ ] Use probabilities vs class predictions configured
- [ ] Weights for voting ensemble (if not uniform)

#### Run Ensemble Training
```bash
# Option 1: Using Phase 3 data (recommended)
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --stacking-data <cv_run_id> \
  --meta-learner logistic

# Option 2: Generate OOF on-the-fly (slower)
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# Check outputs
ls experiments/runs/stacking_h20_*/
```

---

## Post-Training Checklist

Use this checklist **after training completes** to validate results are meaningful.

### Training Output Validation

#### Files Created
- [ ] Run directory created: `experiments/runs/<run_id>/`
- [ ] Model artifacts saved (`.pkl`, `.pt`, `.joblib`)
- [ ] Training metrics saved: `training_metrics.json`
- [ ] Configuration saved: `config.json`
- [ ] Training log exists: `training.log`

#### Metrics Sanity Checks
- [ ] Training accuracy > random baseline (33% for 3-class)
- [ ] Validation accuracy > random baseline
- [ ] Training accuracy > validation accuracy (expected)
- [ ] No extreme overfitting (train_acc - val_acc < 0.3)
- [ ] Loss decreased during training (check logs)
- [ ] F1 score > 0.4 (reasonable threshold)

#### Trading Metrics (If Available)
- [ ] Sharpe ratio > 0.5 (minimum acceptable)
- [ ] Win rate > 0.4 (minimum acceptable)
- [ ] Max drawdown < 50% (conservative threshold)
- [ ] Transaction costs accounted for in backtest

---

### Data Leakage Audit

Use this checklist to audit for potential data leakage.

#### Temporal Integrity
- [ ] Train data timestamps < validation data timestamps
- [ ] Validation data timestamps < test data timestamps
- [ ] Purge window applied between splits (check logs)
- [ ] Embargo period applied between splits (check logs)
- [ ] No overlap in train/val/test indices

#### Feature Engineering
- [ ] All features computed using only past data
- [ ] No forward-looking indicators (SMA uses past data only)
- [ ] Multi-timeframe features use proper alignment
- [ ] Resampling uses forward-fill (not future data)

#### Label Generation
- [ ] Labels computed using future returns (expected)
- [ ] Label calculation respects horizon boundaries
- [ ] No label information leaks into features
- [ ] Quality weights computed before splitting

#### Cross-Validation
- [ ] PurgedKFold used (not plain KFold)
- [ ] Purge/embargo parameters set correctly
- [ ] OOF predictions truly out-of-sample
- [ ] No temporal overlap between folds

#### Verification Script
```python
# Verify no leakage in splits
import pandas as pd
from pathlib import Path

# Load split indices
train_idx = pd.read_parquet("data/splits/scaled/train_indices_h20.parquet")
val_idx = pd.read_parquet("data/splits/scaled/val_indices_h20.parquet")
test_idx = pd.read_parquet("data/splits/scaled/test_indices_h20.parquet")

# Check no overlap
assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"

print("✅ No index overlap between splits")
```

---

### Ensemble-Specific Validation

#### Stacking Ensemble
- [ ] Base models trained on different folds
- [ ] OOF predictions used for meta-learner training
- [ ] Meta-learner trained on non-overlapping data
- [ ] Final predictions use all base models
- [ ] Stacking diversity verified (base models disagree sometimes)

#### Voting Ensemble
- [ ] All base models have similar performance (within 10% accuracy)
- [ ] Weights sum to 1.0 (if using weighted voting)
- [ ] Base model predictions are diverse (not all identical)

#### Blending Ensemble
- [ ] Holdout set used for meta-learner (not validation set)
- [ ] Holdout set size appropriate (10-20% of training data)
- [ ] Base models trained on non-holdout data only

---

### Test Set Discipline Checklist

⚠️ **CRITICAL:** Test set must be used ONLY ONCE for final evaluation.

#### Before Touching Test Set
- [ ] All hyperparameter tuning done on validation set
- [ ] All model selection done on validation set
- [ ] All feature engineering finalized
- [ ] Validation performance acceptable
- [ ] Final model configuration frozen

#### Test Set Usage
- [ ] Test set used EXACTLY ONCE for final evaluation
- [ ] No decisions made based on test performance
- [ ] No re-training after seeing test results
- [ ] Test results reported honestly (no cherry-picking)

#### Post-Test Actions
- [ ] Document test performance in run artifacts
- [ ] Do NOT retrain based on test performance
- [ ] Do NOT try different models to improve test score
- [ ] If test performance poor, analyze on validation set

#### Warning Signs of Test Set Leakage
- ❌ "Let me try different features to improve test score"
- ❌ "Test performance is bad, let me retune hyperparameters"
- ❌ "I ran 10 models and picked the one with best test score"
- ✅ "Test performance validates our validation approach"
- ✅ "Test performance matches validation performance closely"

---

### Phase 3→4 Workflow Checklist

Specific checklist for Phase 3→4 integration workflow.

#### Phase 3 Completion
- [ ] CV run completed successfully
- [ ] CV run ID recorded: `___________________`
- [ ] CV results JSON exists: `data/stacking/<run_id>/cv_results.json`
- [ ] Stacking datasets created for all horizons:
  - [ ] `stacking_dataset_h5.parquet`
  - [ ] `stacking_dataset_h10.parquet`
  - [ ] `stacking_dataset_h15.parquet`
  - [ ] `stacking_dataset_h20.parquet`
- [ ] Tuned parameters saved: `data/stacking/<run_id>/tuned_params/`

#### Phase 4 Setup
- [ ] CV run ID specified in `--stacking-data` argument
- [ ] Horizon matches between CV and ensemble training
- [ ] Meta-learner selected (default: logistic)
- [ ] Phase 3 base directory configured (default: `data/stacking/`)

#### Load Validation
- [ ] Stacking data loads without errors:
  ```python
  from scripts.train_model import load_phase3_stacking_data
  data = load_phase3_stacking_data(
      cv_run_id="<run_id>",
      horizon=20,
      phase3_base_dir=Path("data/stacking")
  )
  assert "y_true" in data["data"].columns
  ```
- [ ] Base model predictions present in loaded data
- [ ] Sample counts match expected CV output size

#### Training
- [ ] Ensemble training completes in <5 minutes (vs 30+ without Phase 3 data)
- [ ] Training uses loaded OOF predictions (check logs)
- [ ] Meta-learner training successful
- [ ] Final model saved correctly

---

## Quick Validation Script

Run this script to perform automated validation:

```python
#!/usr/bin/env python3
"""
Quick validation script for ML pipeline.
Run before expensive training jobs to catch config errors.
"""
from pathlib import Path
import sys

def validate_phase1():
    """Validate Phase 1 setup."""
    print("Validating Phase 1...")

    # Check data exists
    raw_data_dir = Path("data/raw")
    assert raw_data_dir.exists(), "data/raw/ directory missing"

    # Check for OHLCV files
    data_files = list(raw_data_dir.glob("*_1m.parquet")) + list(raw_data_dir.glob("*_1m.csv"))
    assert len(data_files) > 0, "No OHLCV data found in data/raw/"

    # Check config
    from src.phase1.pipeline_config import PipelineConfig
    config = PipelineConfig()
    assert not str(config.project_root).endswith("src"), "Project root incorrectly set to src/"

    print("✅ Phase 1 validation passed")

def validate_phase2():
    """Validate Phase 2 setup."""
    print("Validating Phase 2...")

    # Check models registered
    from src.models import ModelRegistry
    models = ModelRegistry.list_all()
    assert len(models) >= 12, f"Expected 12+ models, got {len(models)}"

    # Check required models
    required = ["xgboost", "lightgbm", "catboost", "lstm", "gru"]
    for model in required:
        assert model in models, f"Model '{model}' not registered"

    print("✅ Phase 2 validation passed")

def validate_phase3():
    """Validate Phase 3 setup."""
    print("Validating Phase 3...")

    # Check CV tools available
    from src.cross_validation.purged_kfold import PurgedKFold
    from src.cross_validation.cv_runner import CrossValidationRunner

    print("✅ Phase 3 validation passed")

def validate_phase4():
    """Validate Phase 4 setup."""
    print("Validating Phase 4...")

    # Check ensemble models
    from src.models.ensemble import (
        VotingEnsemble,
        StackingEnsemble,
        BlendingEnsemble,
        validate_ensemble_config,
    )

    # Test validation
    is_valid, _ = validate_ensemble_config(["xgboost", "lightgbm"])
    assert is_valid, "Valid ensemble config rejected"

    is_valid, _ = validate_ensemble_config(["xgboost", "lstm"])
    assert not is_valid, "Invalid ensemble config accepted"

    print("✅ Phase 4 validation passed")

def main():
    """Run all validations."""
    print("="* 60)
    print("ML Pipeline Validation")
    print("=" * 60)

    try:
        validate_phase1()
        validate_phase2()
        validate_phase3()
        validate_phase4()

        print("\n" + "=" * 60)
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Usage:**

```bash
# Save as scripts/validate_pipeline.py
chmod +x scripts/validate_pipeline.py
./scripts/validate_pipeline.py
```

---

## Summary

### Pre-Training
1. ✅ Data quality and availability
2. ✅ Configuration correctness
3. ✅ Model compatibility
4. ✅ Ensemble validation (if applicable)

### Post-Training
1. ✅ Output files created
2. ✅ Metrics pass sanity checks
3. ✅ No data leakage
4. ✅ Test set discipline maintained

### Best Practices
- Run validation script before expensive jobs
- Check all boxes in relevant checklists
- Document any checklist failures
- Use validation set for all decisions
- Touch test set ONLY ONCE

---

**Questions?** See `docs/MIGRATION_GUIDE.md` or `docs/QUICK_REFERENCE.md`
