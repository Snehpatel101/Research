# Model Packaging & Serialization Audit

**Date:** 2026-02-15
**Scope:** All 12 model types - save/load, serialization formats, bundle architecture, gaps

---

## 1. Overview

ML Factory has a **two-tier serialization architecture**:

1. **Model-level save/load** - Each `BaseModel` subclass implements `save(path)` / `load(path)` with model-family-specific formats
2. **Bundle-level packaging** - `ModelBundle` wraps a model with all inference artifacts (scaler, features, calibrator, preprocessing graph, feature spec)

All 12 models implement the `BaseModel` abstract interface (`src/models/base.py:296-321`), which requires `save(path: Path)` and `load(path: Path)` methods.

---

## 2. Per-Family Analysis

### 2.1 Boosting Family (XGBoost, LightGBM, CatBoost)

**Pattern:** Native model format + pickle metadata sidecar

| Model | Weight File | Format | Metadata File |
|-------|------------|--------|---------------|
| **XGBoost** | `model.json` | XGBoost JSON (portable) | `metadata.pkl` |
| **LightGBM** | `model.txt` | LightGBM text (portable) | `metadata.pkl` |
| **CatBoost** | `model.cbm` | CatBoost binary | `metadata.pkl` |

**Metadata saved (all three):**
- `config`: Full hyperparameter dict
- `feature_names`: List of feature names (set via `set_feature_names()`)
- `n_classes`: Number of output classes (always 3)
- `use_gpu`: GPU training flag
- `best_iteration`: Best boosting iteration (LightGBM/CatBoost only)

**Serialization details:**
- XGBoost: `xgb.Booster.save_model()` / `load_model()` - JSON format, portable across platforms
- LightGBM: `lgb.Booster.save_model()` / `Booster(model_file=)` - text format, portable
- CatBoost: `CatBoostClassifier.save_model()` / `load_model()` - binary `.cbm` format

**Load process:** Create empty native object -> load weights -> restore metadata from pickle -> set `_is_fitted = True`

### 2.2 RNN Family (LSTM, GRU)

**Pattern:** PyTorch checkpoint (single `.pt` file)

Both LSTM and GRU inherit from `BaseRNNModel` (`src/models/neural/base_rnn.py`) which provides shared save/load.

**File:** `model.pt` (via `torch.save`)

**Checkpoint contents:**
- `model_state_dict`: PyTorch state dict (weights + buffers)
- `config`: Full hyperparameter dict
- `n_features`: Number of input features
- `n_classes`: Number of output classes
- `seq_len`: Sequence length (optional, used by N-BEATS)

**Load process:** Load checkpoint -> restore config/n_features/n_classes -> recreate network architecture via `_create_network(n_features)` -> load state dict -> move to device -> set eval mode -> set `_is_fitted = True`

**Key detail:** Network architecture is reconstructed from config at load time, not stored in the checkpoint. This means the code must match the version that trained the model.

### 2.3 CNN Family (TCN, InceptionTime, ResNet1D)

**Pattern:** Same as RNN - inherits from `BaseRNNModel`

All three CNN models extend `BaseRNNModel` despite being CNNs (the base class provides the training loop infrastructure, not RNN-specific logic). They use the **identical** save/load mechanism described in 2.2.

- **TCN** (`src/models/neural/tcn_model.py`): Extends `BaseRNNModel`
- **InceptionTime** (`src/models/neural/inceptiontime_model.py`): Extends `BaseRNNModel`
- **ResNet1D** (`src/models/neural/resnet1d_model.py`): Extends `BaseRNNModel`

### 2.4 Transformer Family (PatchTST, iTransformer, TFT)

**Pattern:** Same as RNN - inherits from `BaseRNNModel`

All three transformers extend `BaseRNNModel`:

- **PatchTST** (`src/models/neural/patchtst_model.py`): Extends `BaseRNNModel`
- **iTransformer** (`src/models/neural/itransformer_model.py`): Extends `BaseRNNModel`
- **TFT** (`src/models/neural/tft_model.py`): Extends `BaseRNNModel`

Same checkpoint format as 2.2 above.

### 2.5 MLP Family (N-BEATS)

**Pattern:** Same as RNN - inherits from `BaseRNNModel`

- **N-BEATS** (`src/models/neural/nbeats_model.py`): Extends `BaseRNNModel`

N-BEATS additionally stores `seq_len` in the checkpoint (line 647 of `base_rnn.py`), which is required for its basis expansion architecture.

---

## 3. Bundle Architecture

### 3.1 ModelBundle (`src/inference/bundle.py`)

**Version:** 1.2.0

**Directory structure:**
```
bundle_dir/
  manifest.json              # File listing + MD5 checksums
  metadata.json              # BundleMetadata (model info, horizon, shapes)
  features.json              # Ordered feature column names
  scaler.pkl                 # Fitted RobustScaler/StandardScaler (pickle)
  calibrator.pkl             # Probability calibrator (optional, pickle)
  preprocessing_graph.json   # PreprocessingGraph config (optional)
  feature_spec.json          # FeatureSpec - 5-dimension optimization (optional)
  model/                     # Model artifacts (via model.save())
```

**BundleMetadata fields** (JSON-serialized):
- `version`, `created_at`, `model_name`, `model_family`
- `horizon` (prediction horizon)
- `n_features`, `feature_hash` (MD5 of column names)
- `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes`
- `has_calibrator`, `has_preprocessing_graph`, `has_feature_spec`
- `preprocessing_graph_hash`, `feature_spec_hash`
- `symbol` (trading instrument)
- `training_metrics`, `extra`

**Load process:** Read manifest -> read metadata -> read features -> load scaler (pickle) -> load calibrator (pickle) -> load preprocessing graph -> load feature spec -> create model via `ModelRegistry.create(model_name)` -> call `model.load(model_dir)`

**Packaging:** `package_bundle()` creates a tar.gz for deployment; `extract_bundle()` unpacks it.

### 3.2 EnsembleBundle (`src/inference/ensemble_bundle.py`)

**Version:** 1.0.0

**Directory structure:**
```
bundle_dir/
  manifest.json               # File listing
  metadata.json               # EnsembleBundleMetadata
  stacking_features.json      # Stacking feature names
  base_bundles.json            # Paths to base model bundles
  alignment_config.json        # OOF alignment configuration
  scaler.pkl                   # Stacking feature scaler (optional)
  meta_learner/                # Meta-learner artifacts
    model.pkl                  # OR native save format
```

**Key design:** Base models are referenced by path (not embedded), so the ensemble bundle depends on base model bundles existing at known paths.

---

## 4. Preprocessing Persistence

### 4.1 PreprocessingGraph (`src/inference/preprocessing_graph.py`)

Captures the complete preprocessing pipeline for train/serve parity:
- `CleaningConfig`: Resampling, gap filling, outlier handling
- `MTFConfig`: Multi-timeframe settings
- Feature engineering configuration
- Scaling parameters (scaler is set via `set_scaler()`)

Serialized as JSON. Ensures raw OHLCV can be processed identically to training.

### 4.2 FeatureSpec (`src/core/contracts/feature_spec.py`)

Captures all 5 optimization dimensions:
1. Triple barrier parameters (profit/loss thresholds, holding period)
2. Selected features list
3. Feature parameters (e.g., RSI period=14)
4. Feature timeframes (which TF each feature uses)
5. Model hyperparameters

Serialized as JSON with a schema hash for validation.

---

## 5. Gaps & Recommendations

### 5.1 Critical Gaps

| Gap | Impact | Affected Models |
|-----|--------|----------------|
| **Feature names not auto-saved by boosting models** | Feature names require manual `set_feature_names()` call before save. If not called, `_feature_names` is `None` in the pickle metadata. Importance reporting falls back to `f0, f1, ...` names. | XGBoost, LightGBM, CatBoost |
| **Label mapping not stored** | The `-1,0,1` to `0,1,2` label mapping is hardcoded via `map_labels_to_classes`/`map_classes_to_labels`. If mapping ever changes, old models break silently. | All 12 |
| **No architecture version tag** | Neural model checkpoints don't store architecture version. If `_create_network()` changes between code versions, `load_state_dict()` will fail with cryptic shape mismatch errors. | All 9 neural models |
| **Scaler type not validated on load** | Bundle stores scaler as raw pickle. No check that loaded scaler type matches `model_contract.scaler_type`. | All models needing scaling |
| **Ensemble base paths depend on input** | Ensemble orchestrator uses relative paths by default (./experiments/exp_001). `EnsembleBundle` saves paths as raw `str(p)` at L447 — whether absolute depends on input. Not hardcoded absolute. Moving bundles may still break if absolute paths were used at training time. | Ensemble |

### 5.2 Security Considerations

| Issue | Location | Status |
|-------|----------|--------|
| Pickle deserialization | All `metadata.pkl`, `scaler.pkl`, `calibrator.pkl` loads | Documented with security comments; only trusted internal paths |
| `torch.load` with `weights_only=False` | `base_rnn.py:662` | Required for loading state dicts with config; potential risk if loading untrusted checkpoints |
| `tarfile.extractall` | `bundle.py:585` | Path traversal check exists (`..` and `/` prefix rejected) |

### 5.3 Recommendations

1. **Auto-capture feature names** - Trainer should call `set_feature_names()` automatically after training, before save. This ensures boosting models always have interpretable feature names.

2. **Add architecture version** - Neural models should save a version string (e.g., `"arch_version": "1.0"`) in the checkpoint. On load, compare against current version and warn/error on mismatch.

3. **Store label mapping** - Save the label encoding scheme (`{-1: 0, 0: 1, 1: 2}`) in metadata to detect changes.

4. **Enforce relative base bundle paths** - Ensemble orchestrator uses relative paths by default, but `EnsembleBundle` saves paths as raw `str(p)` at L447. Ensure all paths are stored relative to ensemble bundle root to guarantee portability.

5. **Bundle integrity validation** - `ModelBundle.load()` reads manifest checksums but doesn't verify them. Add optional checksum validation on load.

6. **Standardize metadata format** - Boosting models use pickle for metadata while neural models embed metadata in the torch checkpoint. Consider a unified JSON metadata sidecar for all model families.

7. **`weights_only=True` migration** - Investigate migrating `torch.load` calls to `weights_only=True` (PyTorch 2.6+ default). This requires saving config separately from the state dict.

### 5.4 Strengths

- **Consistent interface**: All 12 models implement the same `save()`/`load()` contract from `BaseModel`
- **Bundle architecture is well-designed**: `ModelBundle` captures scaler, features, calibrator, preprocessing graph, and feature spec - covering most inference needs
- **PreprocessingGraph ensures train/serve parity**: Raw OHLCV -> features pipeline is fully serialized
- **FeatureSpec captures all 5 optimization dimensions**: Full reproducibility of Optuna trials
- **Tarball packaging with path traversal protection**: Deployment-ready with security checks
- **MD5 checksums in manifest**: File integrity tracking (though not enforced on load)
- **Neural models share infrastructure**: `BaseRNNModel` provides consistent save/load for 9/12 models

---

## 6. Summary

| Model Family | Serialization | Metadata Format | Bundle Support |
|-------------|--------------|-----------------|----------------|
| Boosting (3) | Native format (JSON/text/cbm) | Pickle sidecar | Yes (ModelBundle) |
| RNN (2) | PyTorch checkpoint (.pt) | Embedded in checkpoint | Yes (ModelBundle) |
| CNN (3) | PyTorch checkpoint (.pt) | Embedded in checkpoint | Yes (ModelBundle) |
| Transformer (3) | PyTorch checkpoint (.pt) | Embedded in checkpoint | Yes (ModelBundle) |
| MLP (1) | PyTorch checkpoint (.pt) | Embedded in checkpoint | Yes (ModelBundle) |
| Ensemble | Meta-learner save + base refs | JSON + pickle | Yes (EnsembleBundle) |

**Overall assessment:** The packaging system is well-structured with a clean two-tier design. The main gaps are around metadata completeness (feature names, architecture versioning, label mapping) and portability (absolute ensemble paths). No blocking issues for production deployment.
