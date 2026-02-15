# TrainerProtocol & Bundle Extraction Plan

**Date:** 2026-02-15
**Scope:** TrainerProtocol definition, calibrator fix, FeatureSpec auto-generation, neural versioning, BundleMetadata additions, backward compatibility
**Priority:** Phase 2a foundation — unblocks reliable bundle extraction for all 12+ models

---

## 1. TrainerProtocol Definition

### Problem

`BundleBuilder._extract_model()` (src/inference/builder.py:557-580) uses duck-typing fallback chains to extract model, scaler, feature_columns, and calibrator from trainer instances:

```python
# Current fragile extraction
for attr in ["model", "_model", "estimator", "_estimator"]:  # model
for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:  # scaler
for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:  # features
for attr in ["calibrator", "_calibrator", "prob_calibrator"]:  # calibrator
```

If any trainer renames an attribute, extraction silently fails and the bundle is incomplete.

### Solution: Python Protocol Class

Define a `TrainerProtocol` using `typing.Protocol` (structural subtyping — no inheritance required):

```python
# src/core/protocols.py (NEW FILE)
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from src.models.base import BaseModel

@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol that any trainer must satisfy for bundle extraction.

    All trainers that produce models for bundling must expose these
    properties. Using Protocol (structural subtyping) means existing
    classes satisfy it automatically if they have the right attributes —
    no inheritance change needed.
    """

    @property
    def model(self) -> BaseModel:
        """The trained model instance."""
        ...

    @property
    def scaler(self) -> Any | None:
        """Fitted feature scaler (RobustScaler/StandardScaler), or None."""
        ...

    @property
    def feature_columns(self) -> list[str]:
        """Ordered list of feature column names used during training."""
        ...

    @property
    def calibrator(self) -> Any | None:
        """Fitted probability calibrator, or None."""
        ...

    @property
    def training_config(self) -> dict[str, Any]:
        """Training configuration dict (model hyperparams, etc.)."""
        ...

    @property
    def model_key(self) -> str:
        """Unique key for this trained model (e.g., 'xgboost_h20')."""
        ...
```

### Why Protocol (not ABC)

- **No inheritance change** — existing `Trainer` class already has `self.model`, `self.calibrator`. We just need to ensure consistent naming and add a few missing properties.
- **runtime_checkable** — `isinstance(trainer, TrainerProtocol)` works for validation.
- **Gradual adoption** — old code continues to work while new code uses the protocol.

---

## 2. Migration Path

### 2.1 Trainer Class (src/models/training/trainer.py)

The `Trainer` class already exposes:
- `self.model` (line 94) — BaseModel instance from ModelRegistry
- `self.calibrator` (line 107, set at line 822-830) — ProbabilityCalibrator or None
- `self.config` (line 89) — TrainerConfig

**Needs to add:**

| Property | Current State | Change Needed |
|----------|--------------|---------------|
| `scaler` | Not stored on Trainer | Add `self.scaler = None`, populate from prepared data or container's scaler |
| `feature_columns` | `self._feature_set_columns` (line 104, private) | Add public `@property feature_columns` exposing resolved feature list |
| `training_config` | `self.config.to_dict()` exists | Add `@property training_config` returning `self.config.to_dict()` |
| `model_key` | Computable from config | Add `@property model_key` → `f"{self.config.model_name}_h{self.config.horizon}"` |

**Concrete changes:**

```python
# src/models/training/trainer.py — Trainer class

# In __init__ (after line 107):
self.scaler: Any | None = None  # Set during run() from container/adapter

# New properties (after __init__):
@property
def feature_columns(self) -> list[str]:
    """Feature columns used for training (resolved after run())."""
    if self._feature_set_columns is not None:
        return self._feature_set_columns
    # Fallback: get from model if available
    if hasattr(self.model, '_feature_names') and self.model._feature_names:
        return list(self.model._feature_names)
    return []

@property
def training_config(self) -> dict[str, Any]:
    """Training configuration as dict."""
    return self.config.to_dict()

@property
def model_key(self) -> str:
    """Unique key for this trained model."""
    return f"{self.config.model_name}_h{self.config.horizon}"
```

**Scaler population** — In `run()` (around line 720, after data prep):
```python
# After data preparation, capture scaler from container
self.scaler = container.get_scaler() if hasattr(container, 'get_scaler') else None
```

And in `run_prepared()` (around line 948):
```python
# Capture scaler from PreparedData
self.scaler = prepared.scaler if hasattr(prepared, 'scaler') else None
```

### 2.2 ModelTrainingService Result (src/models/training/services/model_training.py)

`ModelTrainingResult.trainer` (line 44) stores the `Trainer` instance. No changes needed here — just ensure the Trainer it stores satisfies the protocol.

### 2.3 UnifiedTrainingOrchestrator Result (src/models/training/unified_orchestrator.py)

`ModelTrainingResult.trainer` (line 97) also stores the `Trainer` instance. Same as above — no changes to the orchestrator's result dataclass.

### 2.4 BundleBuilder Update (src/inference/builder.py)

Replace duck-typing with protocol-aware extraction:

```python
# src/inference/builder.py — replace _extract_* methods

def _extract_model(self, trainer: Any) -> Any | None:
    from src.core.protocols import TrainerProtocol
    if isinstance(trainer, TrainerProtocol):
        return trainer.model
    # Legacy fallback (deprecation warning)
    logger.warning("Trainer does not satisfy TrainerProtocol, using legacy extraction")
    for attr in ["model", "_model", "estimator", "_estimator"]:
        model = getattr(trainer, attr, None)
        if model is not None:
            return model
    if hasattr(trainer, "get_model") and callable(trainer.get_model):
        return trainer.get_model()
    return None

def _extract_scaler(self, trainer: Any) -> Any | None:
    from src.core.protocols import TrainerProtocol
    if isinstance(trainer, TrainerProtocol):
        return trainer.scaler
    # Legacy fallback
    for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
        scaler = getattr(trainer, attr, None)
        if scaler is not None:
            return scaler
    return None

def _extract_feature_columns(self, trainer: Any, n_features: int) -> list[str]:
    from src.core.protocols import TrainerProtocol
    if isinstance(trainer, TrainerProtocol):
        cols = trainer.feature_columns
        if cols:
            return cols
    # Legacy fallback (existing logic)
    ...

def _extract_calibrator(self, trainer: Any) -> Any | None:
    from src.core.protocols import TrainerProtocol
    if isinstance(trainer, TrainerProtocol):
        return trainer.calibrator
    # Legacy fallback
    for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
        calibrator = getattr(trainer, attr, None)
        if calibrator is not None:
            return calibrator
    return None
```

---

## 3. Calibrator Transfer Fix

### Tracing the Bug

1. **Calibrator is created** in `Trainer.run()` at line 822-830:
   ```python
   self.calibrator = ProbabilityCalibrator(cal_config)
   self.calibrator.fit(y_true=y_val_aligned, probabilities=val_predictions.class_probabilities)
   ```
   Same in `Trainer.run_prepared()` at line 1006-1018.

2. **Calibrator lives on** `trainer.calibrator` after training completes.

3. **ModelTrainingService** returns `ModelTrainingResult(trainer=trainer)` at line 143-151 of services/model_training.py.

4. **UnifiedTrainingOrchestrator** stores this in `TrainingRunResult.model_results[key].trainer`.

5. **BundleBuilder._extract_calibrator()** looks for `trainer.calibrator` — this SHOULD work since `Trainer.calibrator` exists.

### Where It Gets Lost

The issue is **not** that the attribute is missing — it's a **timing/conditionality** problem:

- `Trainer.calibrator` is only set if `self.config.use_calibration` is `True` (default may be `False`)
- When `use_calibration=False`, `self.calibrator` is explicitly set to `None` at line 822/1006
- The calibrator is **also** created by `EnsembleService` independently but stored on the service result, not transferred back to the trainer

### Fix

**A. Ensure calibrator propagation from EnsembleService:**

In `src/models/training/services/ensemble_service.py`, if the ensemble service fits a calibrator, it should be stored on the ensemble result in a way that BundleBuilder can find it.

**B. In BundleBuilder, also check model_result for calibrator:**

```python
# src/inference/builder.py — in build_from_training_result(), line 311-313:
# Extract calibrator if requested
calibrator = None
if include_calibrator:
    calibrator = self._extract_calibrator(trainer)
    # Also check model_result directly (ensemble service stores calibrator here)
    if calibrator is None and hasattr(model_result, 'calibrator'):
        calibrator = model_result.calibrator
```

**C. Add `calibrator` field to `ModelTrainingResult`:**

```python
# src/models/training/unified_orchestrator.py — ModelTrainingResult dataclass
@dataclass
class ModelTrainingResult:
    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    calibrator: Any | None = None  # NEW: explicit calibrator field
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
```

Then in the orchestrator's training loop, after training completes:
```python
result.calibrator = trainer.calibrator if hasattr(trainer, 'calibrator') else None
```

---

## 4. FeatureSpec Auto-Generation

### Problem

`BundleBuilder.build_from_training_result()` accepts `feature_specs` as an optional parameter, but it must be passed explicitly. No auto-generation from training config exists.

### Solution: Auto-generate from PipelineConfig + Training Results

```python
# src/inference/builder.py — new method on BundleBuilder

def _auto_generate_feature_spec(
    self,
    model_result: ModelTrainingResult,
    trainer: Any,
) -> FeatureSpec | None:
    """Auto-generate FeatureSpec from training config and results."""
    from src.core.contracts.feature_spec import FeatureSpec

    try:
        feature_columns = self._extract_feature_columns(trainer, model_result.n_features)

        return FeatureSpec(
            # Dimension 1: Triple barrier from PipelineConfig
            profit_threshold=self.config.profit_threshold,
            loss_threshold=self.config.loss_threshold,
            max_holding_bars=self.config.max_holding_bars,
            # Dimension 2: Selected features
            selected_features=feature_columns,
            # Dimension 3: Feature parameters (from config if available)
            feature_params=getattr(self.config, 'feature_params', {}),
            # Dimension 4: Feature timeframes
            feature_timeframes={f: self.config.primary_timeframe for f in feature_columns},
            # Dimension 5: Model hyperparameters
            hyperparameters=trainer.training_config if hasattr(trainer, 'training_config') else {},
            model_name=model_result.model_name,
        )
    except Exception as e:
        logger.warning(f"Failed to auto-generate FeatureSpec: {e}")
        return None
```

**Integration** — In `build_from_training_result()`, after line 316:
```python
# Auto-generate feature spec if not provided
if feature_spec is None:
    feature_spec = self._auto_generate_feature_spec(model_result, trainer)
```

---

## 5. Neural Architecture Versioning

### Problem

Neural model checkpoints (`model.pt`) don't store architecture version. If `_create_network()` changes between code versions, `load_state_dict()` fails with cryptic shape mismatch errors.

### Solution: Tag Checkpoints with `arch_version`

**A. Define version constants per model:**

```python
# src/models/neural/base_rnn.py — class-level constant
class BaseRNNModel(BaseModel):
    ARCH_VERSION: str = "1.0"  # Increment when network architecture changes
```

Each subclass can override:
```python
# src/models/neural/patchtst_model.py
class PatchTSTModel(BaseRNNModel):
    ARCH_VERSION: str = "1.0"
```

**B. Save arch_version in checkpoint:**

In `BaseRNNModel.save()` (base_rnn.py, around line 645-650):
```python
checkpoint = {
    "model_state_dict": self._network.state_dict(),
    "config": self._config,
    "n_features": self._n_features,
    "n_classes": self._n_classes,
    "arch_version": self.ARCH_VERSION,  # NEW
    # ... existing fields
}
```

**C. Validate on load:**

In `BaseRNNModel.load()` (base_rnn.py, around line 660-680):
```python
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Validate architecture version
saved_version = checkpoint.get("arch_version", "0.0")  # Default for old checkpoints
if saved_version != self.ARCH_VERSION:
    logger.warning(
        f"Architecture version mismatch: checkpoint has {saved_version}, "
        f"current code has {self.ARCH_VERSION}. "
        f"Model may fail to load if architecture changed."
    )
```

---

## 6. BundleMetadata Additions

### Current Fields (src/inference/bundle.py)

BundleMetadata already has: version, created_at, model_name, model_family, horizon, n_features, feature_hash, requires_sequences, requires_4d, sequence_length, n_timeframes, has_calibrator, has_preprocessing_graph, has_feature_spec, symbol, training_metrics, extra.

### New Fields Needed

| Field | Type | Purpose | Default |
|-------|------|---------|---------|
| `scaling_source` | `str` | Which scaler was used: "pipeline", "adapter", "none" | `"unknown"` |
| `arch_version` | `str \| None` | Neural architecture version tag | `None` |
| `label_mapping` | `dict[int, int]` | Maps original labels to training classes: `{-1: 0, 0: 1, 1: 2}` | `None` |
| `feature_names` | `list[str]` | Full ordered feature name list (separate from feature_hash) | `[]` |
| `scaler_type` | `str` | Scaler type used: "robust", "standard", "none" | `"unknown"` |
| `training_run_id` | `str \| None` | Links bundle to training run | `None` |

### Concrete Changes

```python
# src/inference/bundle.py — BundleMetadata dataclass

@dataclass
class BundleMetadata:
    # ... existing fields ...

    # NEW fields (with backward-compatible defaults)
    scaling_source: str = "unknown"     # "pipeline" | "adapter" | "none"
    arch_version: str | None = None      # Neural model architecture version
    label_mapping: dict[int, int] | None = None  # e.g., {-1: 0, 0: 1, 1: 2}
    feature_names: list[str] = field(default_factory=list)  # Full ordered feature list
    scaler_type: str = "unknown"         # "robust" | "standard" | "minmax" | "none"
    training_run_id: str | None = None   # Links to training run
```

**Population in BundleBuilder** (builder.py, in `build_from_training_result()`):
```python
# In the ModelBundle.from_training() call, pass new metadata:
extra_metadata={
    "training_run_id": training_result.run_id,
    "training_time_seconds": model_result.training_time_seconds,
    "data_rank": model_result.data_rank,
    "scaling_source": "pipeline",  # or determine from config
    "arch_version": getattr(model, 'ARCH_VERSION', None),
    "label_mapping": {-1: 0, 0: 1, 1: 2},  # Standard mapping
    "scaler_type": self._get_scaler_type(scaler),
}
```

---

## 7. Backward Compatibility

### Old Bundles Loading

Old bundles don't have the new metadata fields. The loading code must handle missing fields gracefully.

**Strategy: Default values + optional parsing**

```python
# src/inference/bundle.py — BundleMetadata.from_dict()

@classmethod
def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
    return cls(
        # ... existing fields with .get() ...
        # NEW fields with safe defaults
        scaling_source=data.get("scaling_source", "unknown"),
        arch_version=data.get("arch_version", None),
        label_mapping=data.get("label_mapping", None),
        feature_names=data.get("feature_names", []),
        scaler_type=data.get("scaler_type", "unknown"),
        training_run_id=data.get("training_run_id", None),
    )
```

**Neural checkpoint loading** — Old checkpoints without `arch_version`:
```python
saved_version = checkpoint.get("arch_version", "0.0")  # "0.0" = pre-versioning
# Warning only, not error — allows old checkpoints to load
```

**Version bump** — Increment BundleMetadata version from `1.2.0` to `1.3.0` for new field support, but keep the loader backward-compatible with `1.2.0` bundles.

### Migration Path

No migration needed. Old bundles load with defaults. New bundles populate all fields. No breaking changes.

---

## 8. Concrete Changes — File-by-File

### New Files

| File | Purpose |
|------|---------|
| `src/core/protocols.py` | `TrainerProtocol` definition |

### Modified Files

| File | Function/Class | Change |
|------|---------------|--------|
| **src/models/training/trainer.py** | `Trainer.__init__` | Add `self.scaler = None` |
| **src/models/training/trainer.py** | `Trainer` (new properties) | Add `feature_columns`, `training_config`, `model_key` properties |
| **src/models/training/trainer.py** | `Trainer.run()` | Capture scaler from container after data prep |
| **src/models/training/trainer.py** | `Trainer.run_prepared()` | Capture scaler from PreparedData |
| **src/models/training/unified_orchestrator.py** | `ModelTrainingResult` | Add `calibrator` field |
| **src/models/training/unified_orchestrator.py** | Training loop | Populate `result.calibrator` from trainer |
| **src/inference/builder.py** | `BundleBuilder._extract_model` | Add protocol check before legacy fallback |
| **src/inference/builder.py** | `BundleBuilder._extract_scaler` | Add protocol check before legacy fallback |
| **src/inference/builder.py** | `BundleBuilder._extract_feature_columns` | Add protocol check before legacy fallback |
| **src/inference/builder.py** | `BundleBuilder._extract_calibrator` | Add protocol check + model_result fallback |
| **src/inference/builder.py** | `BundleBuilder.build_from_training_result` | Add calibrator fallback from model_result, auto FeatureSpec |
| **src/inference/builder.py** | `BundleBuilder` (new method) | Add `_auto_generate_feature_spec()` |
| **src/inference/builder.py** | `BundleBuilder._create_preprocessing_graph` | Pull config values from PipelineConfig instead of hardcoding |
| **src/inference/bundle.py** | `BundleMetadata` | Add 6 new fields with defaults |
| **src/inference/bundle.py** | `BundleMetadata.from_dict` | Parse new fields with safe defaults |
| **src/inference/bundle.py** | `BundleMetadata.to_dict` | Serialize new fields |
| **src/inference/bundle.py** | `ModelBundle.from_training` | Accept and store new metadata fields |
| **src/models/neural/base_rnn.py** | `BaseRNNModel` | Add `ARCH_VERSION = "1.0"` class constant |
| **src/models/neural/base_rnn.py** | `BaseRNNModel.save()` | Include `arch_version` in checkpoint |
| **src/models/neural/base_rnn.py** | `BaseRNNModel.load()` | Validate `arch_version` on load (warning only) |

### Hardcoded Config Fix

`BundleBuilder._create_preprocessing_graph()` (builder.py:512-555) hardcodes:
- `source_timeframe: "1min"` — should come from `self.config.source_timeframe` or similar
- `target_timeframe: "5min"` — should come from `self.config.primary_timeframe`
- `scaler_type: "robust"` — should come from model contract's `scaler_type`

Fix: Read these from PipelineConfig:
```python
pipeline_config = {
    "clean": {
        "source_timeframe": getattr(self.config, 'source_timeframe', '1min'),
        "target_timeframe": getattr(self.config, 'primary_timeframe', '5min'),
    },
    "scaling": {
        "scaler_type": getattr(self.config, 'scaler_type', 'robust'),
    },
    # ... rest from config
}
```

---

## 9. Validation Criteria

After implementation, verify:

```bash
# 1. Protocol check
python -c "
from src.core.protocols import TrainerProtocol
from src.models.training.trainer import Trainer
from src.models.config import TrainerConfig
config = TrainerConfig(model_name='xgboost', horizon=20)
trainer = Trainer(config)
print(f'Trainer satisfies TrainerProtocol: {isinstance(trainer, TrainerProtocol)}')
"

# 2. Backward compat — old metadata loads
python -c "
from src.inference.bundle import BundleMetadata
old_data = {'version': '1.2.0', 'model_name': 'xgboost', 'model_family': 'boosting'}
meta = BundleMetadata.from_dict(old_data)
print(f'scaling_source: {meta.scaling_source}')  # Should be 'unknown'
print(f'arch_version: {meta.arch_version}')  # Should be None
"

# 3. Neural arch version
python -c "
from src.models.neural.base_rnn import BaseRNNModel
print(f'ARCH_VERSION defined: {hasattr(BaseRNNModel, \"ARCH_VERSION\")}')
"

# 4. No regression — existing extraction still works
python -c "
from src.inference.builder import BundleBuilder
# Verify legacy fallback still handles non-protocol trainers
class LegacyTrainer:
    model = 'dummy'
    _scaler = None
builder = BundleBuilder.__new__(BundleBuilder)
result = builder._extract_model(LegacyTrainer())
print(f'Legacy extraction works: {result == \"dummy\"}')
"
```

---

## 10. Execution Order

```
1. Create src/core/protocols.py (TrainerProtocol)          [no dependencies]
2. Update Trainer class properties                          [depends on 1]
3. Update BundleMetadata with new fields                    [no dependencies]
4. Update BundleBuilder extraction to use protocol          [depends on 1, 2]
5. Add calibrator field to ModelTrainingResult              [no dependencies]
6. Fix calibrator transfer in orchestrator                  [depends on 5]
7. Add FeatureSpec auto-generation                          [depends on 2]
8. Add neural architecture versioning                       [no dependencies]
9. Fix hardcoded preprocessing graph config                 [no dependencies]
10. Validation tests                                        [depends on all]
```

Steps 1, 3, 5, 8, 9 can be parallelized.
Steps 2, 4, 6, 7 are sequential.

**Estimated scope:** ~200 lines of new code, ~80 lines of modified code across 8 files.
