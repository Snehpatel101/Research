# Phase E: Deployment Migration (Stage 16)

**Status:** Planning Complete
**Estimated Effort:** 8-10 days

---

## Current State Summary

### Existing Bundling Infrastructure
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/inference/bundle.py` | 719 | ModelBundle V1.1.0 |
| `src/inference/ensemble_bundle.py` | 928 | EnsembleBundle for stacking |
| `src/inference/preprocessing_graph.py` | 908 | PreprocessingGraph for train/serve parity |

### Current ModelBundle Structure
```
bundle_dir/
    manifest.json           # File listing and checksums
    metadata.json           # BundleMetadata
    features.json           # Feature column names
    scaler.pkl              # Fitted scaler
    calibrator.pkl          # Calibrator (optional)
    preprocessing_graph.json
    model/                  # Model artifacts
```

### What's Missing
- Barrier parameters (from Stage 7)
- Feature mask (from Stages 8-9)
- Hyperparameters (from Stage 13)
- Pipeline integration wrapper

---

## Target State

### Enhanced ModelBundle V1.1.0 Structure

```
{model}_bundle/
    manifest.json               # File listing with checksums
    metadata.json               # Enhanced BundleMetadata
    model/
        model.pkl OR model.pt   # Trained weights
        config.json             # Model hyperparameters
    preprocessing/
        scaler.pkl              # Fitted RobustScaler (Stage 11)
        feature_mask.json       # Combined mask (Stages 8-9)
        preprocessing_graph.json
    labeling/
        barrier_params.json     # Optimized params (Stage 7)
    hyperparameters/
        best_params.json        # Optuna params (Stage 13)
    features.json               # Ordered column names
    calibrator.pkl              # Optional
```

### Stage16Bundling Class

```python
class Stage16Bundling:
    """Create production-ready ModelBundle V1.1.0 packages."""

    def run(self, state: PipelineState) -> Stage16Output:
        """
        Inputs (from state):
            - Stage 7:  optimal_barrier_params.json
            - Stage 8-9: feature_mask.json
            - Stage 11: scaler.pkl
            - Stage 13: {model}_best_params.json (×23)
            - Stage 14: trained models (×23)
            - Stage 15: meta-learner models (×4)

        Outputs:
            - experiments/runs/{run_id}/bundles/{model}_bundle/
        """
        bundles = {}
        for model_name in state.trained_models:
            bundle_path = self.bundle_single_model(model_name, state)
            bundles[model_name] = bundle_path

        if state.meta_learners:
            ensemble_path = self.bundle_ensemble(state)

        return Stage16Output(bundles=bundles, ensemble_bundle=ensemble_path)
```

---

## Bundle Contents by Source Stage

| Artifact | Source Stage | Format | Purpose |
|----------|--------------|--------|---------|
| Model weights | 14 | .pkl/.pt | Inference |
| Scaler | 11 | .pkl | Feature preprocessing |
| Feature mask | 8-9 | .json | Feature selection |
| Barrier params | 7 | .json | Label generation for live trading |
| Hyperparameters | 13 | .json | Model configuration |
| PreprocessingGraph | Config | .json | Train/serve parity |
| Calibrator | 14 | .pkl | Probability calibration |

---

## Enhanced BundleMetadata Schema

```python
@dataclass
class BundleMetadata:
    """Enhanced metadata for ModelBundle V1.1.0."""

    # Identity
    version: str = "1.1.0"
    created_at: str
    bundle_hash: str  # SHA256

    # Model Info
    model_name: str
    model_family: str  # boosting, classical, neural, ensemble
    horizon: int
    symbol: str

    # Feature Info
    n_features: int
    feature_hash: str
    feature_mask_hash: str  # NEW
    requires_sequences: bool
    sequence_length: int

    # Labeling Info (NEW)
    has_barrier_params: bool
    barrier_params_hash: str

    # Hyperparameter Info (NEW)
    has_hyperparameters: bool
    hyperparameters_hash: str
    optuna_trial_number: int | None

    # Training Lineage
    pipeline_run_id: str
    training_run_id: str
    git_commit: str

    # Performance Metrics
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
```

---

## Interface Contract

### Input (from PipelineState)
```python
@dataclass
class Stage16Input:
    run_id: str
    output_dir: Path
    barrier_params_path: Path      # Stage 7
    feature_mask_path: Path        # Stages 8-9
    scaler_path: Path              # Stage 11
    hyperparameters_dir: Path      # Stage 13
    models_dir: Path               # Stage 14
    meta_learners_dir: Path | None # Stage 15
    models_to_bundle: list[str]
    build_ensemble_bundle: bool
```

### Output
```python
@dataclass
class Stage16Output:
    model_bundles: dict[str, Path]  # model_name -> bundle_path
    ensemble_bundle: Path | None
    n_bundles_created: int
    total_bundle_size_bytes: int
    bundling_time_seconds: float
    all_bundles_valid: bool
    validation_issues: list[str]
```

---

## Migration Steps

### Step 1: Enhance ModelBundle (2-3 days)
- Add barrier_params, feature_mask, hyperparameters fields
- Update save() with new subdirectories
- Update load() for new structure

### Step 2: Create Stage16Bundling (2-3 days)
- Create `src/pipeline/phases/deployment.py`
- Implement artifact collection from PipelineState
- Implement batch bundling

### Step 3: Integrate with PipelineState (1 day)
- Add stage_16_output field
- Add checkpoint path

### Step 4: Update EnsembleBundle (1-2 days)
- Reference base bundles by path
- Add validation for integrity

### Step 5: Testing (2-3 days)
- Unit tests for bundling
- Integration tests for inference
- Roundtrip tests

---

## Testing Strategy

```python
def test_bundle_roundtrip():
    """Predictions identical before/after save/load."""
    pred_before = model.predict(X_test)

    bundle = ModelBundle.from_training(model, ...)
    bundle.save("path/to/bundle")

    loaded = ModelBundle.load("path/to/bundle")
    pred_after = loaded.predict(X_test)

    np.testing.assert_array_equal(pred_before, pred_after)
```

---

## Critical Files

1. `src/inference/bundle.py` - Core ModelBundle to enhance
2. `src/inference/preprocessing_graph.py` - Train/serve parity
3. `src/inference/ensemble_bundle.py` - EnsembleBundle
4. `src/feature_selection/result.py` - Feature mask serialization
5. `src/contracts/artifact_manifest.py` - Manifest pattern
