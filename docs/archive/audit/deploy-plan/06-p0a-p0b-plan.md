# 06 - Detailed Implementation Plan: P0-A (Foundation) and P0-B (Core Inference)

**Date:** 2026-02-15
**Agent:** 6/10 (Architecture Planner)
**Purpose:** Exact task specifications for P0-A and P0-B with file paths, API signatures, dependency order, backward compatibility, and acceptance criteria.

---

## References

- Reports 01-05 in `.audit/deploy-plan/`
- `PHASE0-DEPLOYABLE-ARTIFACT-PLAN.md` and `HIGH-LEVEL-DEPLOYABLE-ARTIFACT-ARCHITECTURE.md`
- MODEL CONTRACT TABLE from Agent 4 (Report 04, Section 3)

---

# P0-A: Foundation

## P0-A-1: Create `src/core/protocols.py` with TrainerProtocol and InferenceBundle

### Files to Create

- `/home/jake/Desktop/Research/src/core/protocols.py` (NEW, ~55 lines)

### API Changes

```python
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol defining what a Trainer must expose for bundle extraction."""

    @property
    def model(self) -> Any:
        """The trained model instance."""
        ...

    @property
    def scaler(self) -> Any | None:
        """The fitted feature scaler, or None."""
        ...

    @property
    def feature_columns(self) -> list[str]:
        """Ordered list of feature column names used during training."""
        ...

    @property
    def calibrator(self) -> Any | None:
        """Fitted probability calibrator, or None."""
        ...


@runtime_checkable
class InferenceBundle(Protocol):
    """Protocol for deployable inference artifacts (ModelBundle, EnsembleBundle)."""

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
    ) -> Any:
        """Raw OHLCV bars to prediction in a single call."""
        ...

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> Any:
        """Prediction from pre-shaped input."""
        ...

    def validate(self) -> dict[str, Any]:
        """Smoke test report."""
        ...

    def save(self, path: str | Any, overwrite: bool = False) -> Any:
        """Serialize to disk."""
        ...

    @classmethod
    def load(cls, path: str | Any) -> Any:
        """Deserialize from disk."""
        ...
```

### Metadata/Schema Changes

None.

### Dependency Order

- **Blocks:** P0-A-6 (protocol-aware BundleBuilder)
- **Blocked by:** Nothing (no dependencies)

### Backward Compatibility

- Purely additive -- new file, no existing code changes.
- `runtime_checkable` allows `isinstance()` checks without requiring explicit inheritance.
- Existing `ModelBundle` and `EnsembleBundle` do NOT need to declare they implement these protocols; structural subtyping handles it automatically once their APIs match.

### Acceptance Criteria

1. `python -c "from src.core.protocols import TrainerProtocol, InferenceBundle; print('OK')"` succeeds.
2. `grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l` returns `1`.
3. `grep -r "class InferenceBundle" src/ --include="*.py" | wc -l` returns `1`.
4. `ruff check src/core/protocols.py` passes with 0 errors.
5. File includes `from __future__ import annotations`.

### Effort Estimate

**S** -- 40-60 LOC

---

## P0-A-2: Add `ScalingSource` Enum to `src/core/types.py`

### Files to Modify

- `/home/jake/Desktop/Research/src/core/types.py` (MODIFY)

### API Changes

Add after the `LabelingMethod` enum (after ~L226):

```python
class ScalingSource(str, Enum):
    """
    Controls which component applies feature scaling during inference.

    Exactly one scaling source must be active per prediction path
    to prevent double-scaling.

    - BUNDLE: Bundle's own scaler applies (default for predict_from_raw).
    - PREPROCESSING: PreprocessingGraph applies scaling.
    - NONE: No scaling applied (caller is responsible).
    """

    BUNDLE = "bundle"
    PREPROCESSING = "preprocessing"
    NONE = "none"
```

### Metadata/Schema Changes

None (enum only).

### Dependency Order

- **Blocks:** Nothing in P0-A/P0-B directly (used by UIP in Phase 3B-2, which is outside P0-A/P0-B scope). Included here because it is a foundation type that must exist in `types.py` per CLAUDE.md.
- **Blocked by:** Nothing.

### Backward Compatibility

- Purely additive -- new enum, no existing code changes.
- Existing imports from `src/core/types.py` are unaffected.

### Acceptance Criteria

1. `python -c "from src.core.types import ScalingSource; print(ScalingSource.BUNDLE.value)"` prints `bundle`.
2. `grep -r "class ScalingSource" src/ --include="*.py" | wc -l` returns `1`.
3. `ruff check src/core/types.py` passes.

### Effort Estimate

**S** -- 10-15 LOC

---

## P0-A-3: Extend BundleMetadata with New Fields

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/bundle.py` (MODIFY)

### API Changes

Add new fields to `BundleMetadata` dataclass (after `n_timeframes` at L84, before `has_calibrator` at L85):

```python
@dataclass
class BundleMetadata:
    # ... existing required fields (version, created_at, model_name, etc.) ...
    requires_sequences: bool = False
    requires_4d: bool = False
    sequence_length: int = 0
    n_timeframes: int = 0
    # NEW FIELDS (v1.3.0):
    scaling_source: str = "bundle"           # ScalingSource enum value as string
    primary_timeframe: str = "5min"          # Base timeframe from contract
    mtf_timeframes: list[str] = field(default_factory=list)  # Additional TFs for 4D models
    feature_names: list[str] = field(default_factory=list)   # Ordered feature names (redundant backup of feature_columns)
    arch_version: str = "0.0"               # Neural model architecture version for compat checking
    label_mapping: dict[int, str] = field(default_factory=dict)  # Class index to label name
    scaler_type: str = "robust"             # Which scaler type (robust/standard/none)
    # ... existing optional fields continue ...
    has_calibrator: bool = False
    # ... rest unchanged ...
```

Also bump version constant:

```python
BUNDLE_VERSION = "1.3.0"  # Updated for inference routing metadata
```

Update `to_dict()` to include all new fields:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        # ... existing fields ...
        "scaling_source": self.scaling_source,
        "primary_timeframe": self.primary_timeframe,
        "mtf_timeframes": self.mtf_timeframes,
        "feature_names": self.feature_names,
        "arch_version": self.arch_version,
        "label_mapping": self.label_mapping,
        "scaler_type": self.scaler_type,
        # ... rest ...
    }
```

Update `from_dict()` to load new fields with safe defaults:

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
    return cls(
        # ... existing fields ...
        scaling_source=data.get("scaling_source", "bundle"),
        primary_timeframe=data.get("primary_timeframe", "5min"),
        mtf_timeframes=data.get("mtf_timeframes", []),
        feature_names=data.get("feature_names", []),
        arch_version=data.get("arch_version", "0.0"),
        label_mapping={int(k): v for k, v in data.get("label_mapping", {}).items()},
        scaler_type=data.get("scaler_type", "robust"),
        # ... rest ...
    )
```

### Metadata/Schema Changes

| New Field | Type | Default | Source at Bundle Creation Time |
|-----------|------|---------|-------------------------------|
| `scaling_source` | `str` | `"bundle"` | Hardcoded to `"bundle"` for `predict_from_raw` path |
| `primary_timeframe` | `str` | `"5min"` | `contract.primary_timeframe` via model lookup |
| `mtf_timeframes` | `list[str]` | `[]` | `contract.mtf_timeframes` via model lookup |
| `feature_names` | `list[str]` | `[]` | `feature_columns` list passed to bundle |
| `arch_version` | `str` | `"0.0"` | Version tag from neural model config (if available) |
| `label_mapping` | `dict[int, str]` | `{}` | From training config or label encoder |
| `scaler_type` | `str` | `"robust"` | `contract.scaler_type` via model lookup |

### Dependency Order

- **Blocks:** P0-B-1 (adapter routing reads `primary_timeframe`, `mtf_timeframes` from metadata), P0-B-2 (reads `sequence_length`), P0-B-3 (reads `mtf_timeframes`, `primary_timeframe`)
- **Blocked by:** Nothing.

### Backward Compatibility

- All new fields use `.get()` with safe defaults in `from_dict()`.
- Existing v1.2.0 bundles load successfully -- missing fields get defaults.
- `BUNDLE_VERSION` bump to `"1.3.0"` is informational; no loading check rejects older versions.
- `feature_names` is a backup copy; `feature_columns` on the ModelBundle instance remains the authoritative list.

### Acceptance Criteria

1. Existing bundles (v1.2.0 JSON) load without error via `BundleMetadata.from_dict()`.
2. New bundles include all 7 new fields in their serialized `metadata.json`.
3. `BUNDLE_VERSION` reads `"1.3.0"`.
4. Round-trip test: `BundleMetadata.from_dict(metadata.to_dict())` produces identical object.
5. `ruff check src/inference/bundle.py` passes.

### Effort Estimate

**M** -- 50-80 LOC changes across dataclass, `to_dict`, `from_dict`, and `from_training`.

---

## P0-A-4: Fix Calibrator Transfer (Orchestrator -> ModelTrainingResult -> BundleBuilder)

### Files to Modify

- `/home/jake/Desktop/Research/src/models/training/unified_orchestrator.py` (MODIFY)

### API Changes

**Option chosen:** Add `calibrator` field to the orchestrator's `ModelTrainingResult` and populate it during conversion.

Step 1: Add field to `ModelTrainingResult` at L78-111:

```python
@dataclass
class ModelTrainingResult:
    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
    calibrator: Any | None = None          # NEW: Probability calibrator from _calibrate_model
    calibration_metrics: Any | None = None  # NEW: Calibration quality metrics
```

Step 2: Update `_train_single_model()` conversion at L912-920:

```python
return ModelTrainingResult(
    model_name=result.model_name,
    horizon=result.horizon,
    metrics=result.metrics,
    trainer=result.trainer,
    training_time_seconds=result.training_time_seconds,
    n_features=result.n_features,
    data_rank=result.data_rank,
    calibrator=getattr(result, "calibrator", None),              # NEW
    calibration_metrics=getattr(result, "calibration_metrics", None),  # NEW
)
```

Step 3: Update `_train_boosting_parallel()` similarly (L799-807 area) with the same `calibrator=getattr(result, "calibrator", None)` addition.

Step 4: Update `to_dict()` at L102-111 to include calibrator status:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        # ... existing fields ...
        "has_calibrator": self.calibrator is not None,
    }
```

### Files Also Modified

- `/home/jake/Desktop/Research/src/inference/builder.py` (MODIFY)

Update `_extract_calibrator()` at L630-644 to ALSO check `model_result.calibrator`:

```python
def _extract_calibrator(
    self, trainer: Any, model_result: Any = None
) -> Any | None:
    """Extract probability calibrator from trainer or model result."""
    # First check model_result (preferred, set by orchestrator)
    if model_result is not None:
        calibrator = getattr(model_result, "calibrator", None)
        if calibrator is not None:
            return calibrator

    # Fallback: check trainer attributes (legacy duck-typing)
    for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
        calibrator = getattr(trainer, attr, None)
        if calibrator is not None:
            return calibrator
    return None
```

Update the call site in `build_from_training_result()` (~L313 area) to pass `model_result`:

```python
calibrator = self._extract_calibrator(trainer, model_result=model_result) if include_calibrator else None
```

### Dependency Order

- **Blocks:** Nothing directly (calibrators flowing into bundles is correctness improvement).
- **Blocked by:** Nothing.

### Backward Compatibility

- New `calibrator` field defaults to `None` -- existing code creating `ModelTrainingResult` without it works fine.
- `_extract_calibrator` still has the duck-typing fallback for old trainers that might have the calibrator set directly.
- `getattr(result, "calibrator", None)` is safe for service results that dynamically attach the attribute.

### Acceptance Criteria

1. After training with `auto_calibrate=True`, `training_result.model_results["xgboost_h20"].calibrator` is not None.
2. After bundling, loaded `ModelBundle.calibrator` is not None when calibration was enabled.
3. `ModelBundle.metadata.has_calibrator` is `True` in the saved bundle metadata.
4. Backward compat: Creating `ModelTrainingResult(model_name="test", horizon=20)` still works (calibrator defaults to None).
5. `ruff check src/models/training/unified_orchestrator.py src/inference/builder.py` passes.

### Effort Estimate

**M** -- 30-50 LOC across two files.

---

## P0-A-5: Fix Double-Scaling Bug in `ModelBundle.predict_from_raw()`

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/bundle.py` (MODIFY)

### API Changes

One-line change in `preprocess()` at L1041:

```python
# BEFORE (L1038-1042):
features = self.preprocessing_graph.transform(
    raw_df,
    skip_cleaning=skip_cleaning,
    skip_scaling=False,
)

# AFTER:
features = self.preprocessing_graph.transform(
    raw_df,
    skip_cleaning=skip_cleaning,
    skip_scaling=True,   # Bundle's own scaler applies in predict(); avoid double-scaling
)
```

### Metadata/Schema Changes

None.

### Dependency Order

- **Blocks:** P0-B-1 (adapter routing assumes correct single-scaled features)
- **Blocked by:** Nothing.

### Backward Compatibility

- For models WITH a bundle scaler: behavior changes from double-scaled to single-scaled (correct).
- For models WITHOUT a bundle scaler (boosting models with `requires_scaling=False`): the preprocessing graph was the only scaler, so this change means NO scaling applies. However, boosting contracts have `requires_scaling=False` and `scaler_type="none"`, so no scaling is correct.
- For models WITH a preprocessing graph scaler AND a bundle scaler: this is the double-scaling bug fix. The bundle scaler is the authoritative one (fitted on training data), so it should be the sole scaling source.

### Acceptance Criteria

1. `self.preprocessing_graph.transform()` is called with `skip_scaling=True` in `preprocess()`.
2. For a bundle with both a preprocessing graph scaler and bundle scaler, calling `predict_from_raw()` applies scaling exactly once (the bundle scaler in `predict()`).
3. `ruff check src/inference/bundle.py` passes.

### Effort Estimate

**S** -- 1 LOC change, but include a comment explaining why.

---

## P0-A-6: Make BundleBuilder Protocol-Aware with Legacy Fallback

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/builder.py` (MODIFY)

### API Changes

Update `_extract_model()`, `_extract_scaler()`, `_extract_feature_columns()`, and `_extract_calibrator()` to first check for `TrainerProtocol` compliance, then fall back to duck-typing.

```python
def _extract_model(self, trainer: Any) -> Any | None:
    """Extract model from trainer, preferring protocol if available."""
    from src.core.protocols import TrainerProtocol

    # Protocol path (preferred)
    if isinstance(trainer, TrainerProtocol):
        return trainer.model

    # Legacy duck-typing fallback
    for attr in ["model", "_model", "estimator", "_estimator"]:
        model = getattr(trainer, attr, None)
        if model is not None:
            return model
    if hasattr(trainer, "get_model") and callable(trainer.get_model):
        return trainer.get_model()
    return None


def _extract_scaler(self, trainer: Any) -> Any | None:
    """Extract scaler from trainer, preferring protocol if available."""
    from src.core.protocols import TrainerProtocol

    if isinstance(trainer, TrainerProtocol):
        return trainer.scaler

    for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
        scaler = getattr(trainer, attr, None)
        if scaler is not None:
            return scaler
    return None


def _extract_feature_columns(
    self, trainer: Any, n_features: int
) -> list[str]:
    """Extract feature columns, preferring protocol if available."""
    from src.core.protocols import TrainerProtocol

    if isinstance(trainer, TrainerProtocol):
        columns = trainer.feature_columns
        if columns and len(columns) > 0:
            return list(columns)

    # Legacy duck-typing fallback (unchanged)
    for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:
        columns = getattr(trainer, attr, None)
        if columns is not None and len(columns) > 0:
            return list(columns)

    scaler = self._extract_scaler(trainer)
    if scaler is not None:
        columns = getattr(scaler, "feature_names_in_", None)
        if columns is not None:
            return list(columns)

    logger.warning("No feature columns found, using generic names")
    return [f"f{i}" for i in range(n_features)]
```

Additionally, update `build_from_training_result()` to populate the new BundleMetadata fields from model contracts when available:

```python
# Inside the per-model loop, after extracting model:
# Try to get contract info for richer metadata
contract = None
try:
    from src.core.contracts import get_model_contract
    contract = get_model_contract(model_result.model_name)
except Exception:
    pass

# When creating ModelBundle.from_training(), pass contract info for metadata:
primary_timeframe = contract.primary_timeframe if contract else "5min"
mtf_timeframes = list(contract.mtf_timeframes) if contract else []
scaler_type = contract.scaler_type if contract else "robust"
arch_version = getattr(model, "_arch_version", "0.0") if model else "0.0"
```

### Dependency Order

- **Blocks:** Nothing directly.
- **Blocked by:** P0-A-1 (protocols.py must exist), P0-A-3 (BundleMetadata fields), P0-A-4 (calibrator field).

### Backward Compatibility

- All existing trainers continue to work via the duck-typing fallback.
- `isinstance(trainer, TrainerProtocol)` returns `False` for trainers that don't satisfy the protocol, so the fallback always runs for legacy code.
- No behavior change for existing callers.

### Acceptance Criteria

1. For a trainer implementing `TrainerProtocol`: model, scaler, feature_columns, calibrator extracted via protocol properties.
2. For a trainer NOT implementing protocol: extraction falls back to duck-typing chains (unchanged behavior).
3. New BundleMetadata fields (`primary_timeframe`, `mtf_timeframes`, `scaler_type`) populated from contract when available.
4. `ruff check src/inference/builder.py` passes.
5. `from src.core.protocols import TrainerProtocol` import does not fail.

### Effort Estimate

**M** -- 60-90 LOC changes across extraction methods and metadata population.

---

# P0-B: Core Inference (Adapter Routing)

## P0-B-1: Add `_apply_adapter()` Routing to ModelBundle

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/bundle.py` (MODIFY)

### API Changes

Add a new private method and update `predict_from_raw()`:

```python
def _apply_adapter(
    self,
    features_2d: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Route 2D preprocessed features through the appropriate adapter
    based on bundle metadata flags.

    Args:
        features_2d: 2D DataFrame from preprocess() with shape (n_rows, n_features)
        raw_df: Original raw OHLCV DataFrame (needed for 4D models only)

    Returns:
        ndarray of appropriate shape:
          - 2D (n, feat) for tabular models
          - 3D (n, seq, feat) for sequence models
          - 4D (n, tf, seq, feat) for multi-timeframe models

    Raises:
        InferenceShapeMismatchError: If adapter output shape is unexpected
        ValueError: If insufficient data for windowing
    """
    if self.metadata.requires_4d:
        if raw_df is None:
            raise ValueError(
                "4D models require raw_df for multi-timeframe preprocessing. "
                "Pass the original OHLCV DataFrame."
            )
        return self._build_4d_input(raw_df)
    elif self.metadata.requires_sequences:
        return self._build_3d_input(features_2d)
    else:
        # Tabular: convert DataFrame to ndarray, reorder columns
        return features_2d.values.astype(np.float32)
```

Update `predict_from_raw()` at L1056-1077:

```python
def predict_from_raw(
    self,
    raw_df: pd.DataFrame,
    calibrate: bool = True,
    skip_cleaning: bool = False,
) -> PredictionResult:
    """
    End-to-end prediction from raw OHLCV data.

    Handles all 12 core model families:
    - Tabular (2D): PreprocessingGraph -> pass through -> predict
    - Sequence (3D): PreprocessingGraph -> sliding window -> predict
    - Multi-TF (4D): Raw OHLCV resampling -> 4D tensor -> predict

    Args:
        raw_df: DataFrame with raw OHLCV data (datetime index, OHLCV columns)
        calibrate: Whether to apply probability calibration
        skip_cleaning: If True, skip resampling/cleaning step

    Returns:
        PredictionResult with predictions and probabilities
    """
    # For 4D models, use dedicated preprocessing path
    if self.metadata.requires_4d:
        adapted = self._build_4d_input(raw_df, skip_cleaning=skip_cleaning)
        return self.predict(adapted, calibrate=calibrate)

    # For 2D and 3D models, use PreprocessingGraph then adapt
    features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)
    adapted = self._apply_adapter(features, raw_df=raw_df)
    return self.predict(adapted, calibrate=calibrate)
```

### Metadata/Schema Changes

None (reads existing `metadata.requires_4d`, `metadata.requires_sequences`).

### Dependency Order

- **Blocks:** P0-B-4 (EnsembleBundle.predict_from_raw depends on base bundles having working predict_from_raw)
- **Blocked by:** P0-A-3 (BundleMetadata fields), P0-A-5 (double-scaling fix), P0-B-2 (_build_3d_input), P0-B-3 (_build_4d_input)

### Backward Compatibility

- `predict_from_raw()` signature unchanged -- same parameters, same return type.
- Existing callers with boosting models see identical behavior (2D passthrough).
- New callers with neural/transformer models now get working inference instead of ValueError.
- `predict(X)` with pre-shaped input is completely unaffected.

### Acceptance Criteria

1. `predict_from_raw(raw_df)` succeeds for XGBoost bundle (2D, tabular).
2. `predict_from_raw(raw_df)` succeeds for LSTM bundle (3D, sequence) with sufficient data rows >= `sequence_length`.
3. `predict_from_raw(raw_df)` succeeds for PatchTST bundle (4D, multi-TF) with raw 1min OHLCV data.
4. `predict(pre_shaped_X)` continues to work for all pre-existing use cases.
5. Returns `PredictionResult` in all cases.
6. `ruff check src/inference/bundle.py` passes.

### Effort Estimate

**M** -- 40-60 LOC (routing method + predict_from_raw update).

---

## P0-B-2: Add `_build_3d_input()` for Sequence Models (Sliding Window)

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/bundle.py` (MODIFY)

### API Changes

Add new private method:

```python
def _build_3d_input(
    self,
    features_2d: pd.DataFrame,
) -> np.ndarray:
    """
    Convert 2D feature DataFrame to 3D sliding-window tensor for sequence models.

    Replicates the windowing logic from SequenceAdapter._build_sequences()
    using numpy stride tricks for efficiency.

    Args:
        features_2d: DataFrame of shape (n_rows, n_features) from preprocess()

    Returns:
        ndarray of shape (n_sequences, sequence_length, n_features)
        For single-point inference (n_rows == sequence_length): returns (1, seq, feat)

    Raises:
        ValueError: If n_rows < sequence_length
    """
    seq_len = self.metadata.sequence_length
    if seq_len <= 0:
        seq_len = 60  # safe default from most contracts

    values = features_2d.values.astype(np.float32)
    n_rows, n_feat = values.shape

    if n_rows < seq_len:
        raise ValueError(
            f"Insufficient data for sequence model: got {n_rows} rows "
            f"but model requires sequence_length={seq_len}. "
            f"Provide at least {seq_len} rows of preprocessed features."
        )

    if n_rows == seq_len:
        # Exactly one window -- single-point inference
        return values.reshape(1, seq_len, n_feat)

    # Sliding window via stride tricks (matches SequenceAdapter)
    windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
    # windows shape: (n_rows - seq_len + 1, n_feat, seq_len)
    # Transpose to (n_sequences, seq_len, n_feat)
    result = windows.transpose(0, 2, 1).copy()

    return result
```

### Metadata/Schema Changes

None (reads `metadata.sequence_length`).

### Dependency Order

- **Blocks:** P0-B-1 (_apply_adapter calls this method)
- **Blocked by:** P0-A-3 (sequence_length must be in metadata -- already is, but new fields ensure contract info available)

### Backward Compatibility

- New private method, no public API change.
- Does not affect `predict()` or any existing code path.

### Acceptance Criteria

1. For 60-row DataFrame with 50 features and `sequence_length=60`: returns shape `(1, 60, 50)`.
2. For 120-row DataFrame with 50 features and `sequence_length=60`: returns shape `(61, 60, 50)`.
3. For 30-row DataFrame with `sequence_length=60`: raises `ValueError` with descriptive message.
4. Output values match what `SequenceAdapter._build_sequences()` produces for the same input.
5. `ruff check src/inference/bundle.py` passes.

### Effort Estimate

**S** -- 25-35 LOC.

---

## P0-B-3: Add `_build_4d_input()` for Transformer Models (MTF Generation)

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/bundle.py` (MODIFY)

### API Changes

Add new private method:

```python
def _build_4d_input(
    self,
    raw_df: pd.DataFrame,
    skip_cleaning: bool = False,
) -> np.ndarray:
    """
    Build 4D multi-timeframe tensor from raw OHLCV data for transformer models.

    Bypasses PreprocessingGraph entirely. Resamples raw OHLCV to each
    required timeframe, then builds sliding windows per timeframe and
    stacks them into a 4D tensor.

    Replicates the logic from MultiStreamAdapter._build_multi_stream().

    Args:
        raw_df: DataFrame with datetime index and OHLCV columns (at base resolution,
                typically 1min for patchtst/itransformer)
        skip_cleaning: If True, skip data cleaning/validation

    Returns:
        ndarray of shape (n_sequences, n_timeframes, sequence_length, n_features)
        where n_features is typically 5 (OHLCV) or as specified by contract.

    Raises:
        ValueError: If raw_df lacks required columns or datetime index
        ValueError: If insufficient data for windowing at any timeframe
    """
    import pandas as pd

    # Determine timeframes from metadata
    primary_tf = self.metadata.primary_timeframe or "1min"
    mtf_list = self.metadata.mtf_timeframes or []
    all_timeframes = [primary_tf] + list(mtf_list)
    seq_len = self.metadata.sequence_length or 60

    # Feature columns for 4D models: raw OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    available_cols = [c for c in ohlcv_cols if c in raw_df.columns]
    if len(available_cols) < 4:
        raise ValueError(
            f"4D models require OHLCV columns. Found: {list(raw_df.columns[:10])}. "
            f"Need at least: open, high, low, close"
        )
    feature_cols = available_cols
    n_feat = len(feature_cols)

    # Ensure datetime index
    if not isinstance(raw_df.index, pd.DatetimeIndex):
        raise ValueError("4D models require a DatetimeIndex on raw_df.")

    # Resample to each timeframe and build per-TF sliding windows
    tf_windows: list[np.ndarray] = []  # Each element: (n_seq, seq_len, n_feat)

    # Map timeframe strings to pandas resample rules
    tf_map = {
        "1min": "1min", "5min": "5min", "15min": "15min",
        "30min": "30min", "1h": "1h", "4h": "4h", "1d": "1D",
    }

    for tf in all_timeframes:
        resample_rule = tf_map.get(tf, tf)

        if tf == primary_tf:
            # Primary timeframe: use raw_df directly (or resample if needed)
            tf_df = raw_df
        else:
            # Resample to higher timeframe using standard OHLCV aggregation
            tf_df = raw_df[feature_cols].resample(resample_rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                **({"volume": "sum"} if "volume" in feature_cols else {}),
            }).dropna()

        tf_values = tf_df[feature_cols].values.astype(np.float32)

        if len(tf_values) < seq_len:
            raise ValueError(
                f"Insufficient data for timeframe '{tf}': "
                f"got {len(tf_values)} bars but need {seq_len}. "
                f"Provide more raw data."
            )

        # Sliding window for this timeframe
        windows = np.lib.stride_tricks.sliding_window_view(
            tf_values, seq_len, axis=0
        )
        # Shape: (n_windows, n_feat, seq_len) -> transpose to (n_windows, seq_len, n_feat)
        tf_3d = windows.transpose(0, 2, 1).copy()
        tf_windows.append(tf_3d)

    # Align window counts: use minimum across timeframes
    min_windows = min(w.shape[0] for w in tf_windows)
    # Take last min_windows from each (most recent data)
    aligned = [w[-min_windows:] for w in tf_windows]

    # Stack: list of (n_seq, seq_len, n_feat) -> (n_seq, n_tf, seq_len, n_feat)
    result = np.stack(aligned, axis=1)

    return result
```

### Metadata/Schema Changes

None (reads `metadata.primary_timeframe`, `metadata.mtf_timeframes`, `metadata.sequence_length`).

### Dependency Order

- **Blocks:** P0-B-1 (_apply_adapter calls this for 4D models)
- **Blocked by:** P0-A-3 (BundleMetadata must have `primary_timeframe`, `mtf_timeframes`)

### Backward Compatibility

- New private method, no public API change.
- Does not affect any existing code path.
- 4D models previously raised ValueError in `predict_from_raw()` -- now they work.

### Acceptance Criteria

1. For raw 1min OHLCV DataFrame with 200 rows, PatchTST bundle with `primary_timeframe="1min"`, `mtf_timeframes=["5min", "15min"]`, `sequence_length=60`: returns shape `(n, 3, 60, 5)` where n > 0.
2. For raw 1min data with only 30 rows (insufficient): raises `ValueError` with descriptive message.
3. Resampled higher-TF bars use correct OHLCV aggregation (first/max/min/last/sum).
4. No forward-looking data in windows (each window uses only past data).
5. `ruff check src/inference/bundle.py` passes.

### Effort Estimate

**L** -- 80-120 LOC (resampling, per-TF windowing, alignment, validation).

---

## P0-B-4: Add `predict_from_raw()` to EnsembleBundle

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` (MODIFY)

### API Changes

Add new public method after `predict_from_base_features()` (~L698):

```python
def predict_from_raw(
    self,
    raw_df: pd.DataFrame,
    calibrate: bool = True,
    skip_cleaning: bool = False,
) -> Any:
    """
    End-to-end ensemble prediction from raw OHLCV data.

    Loads base bundles, calls predict_from_raw() on each (with calibrate=False
    so calibration happens only at the ensemble level), stacks base outputs,
    and runs the meta-learner.

    Args:
        raw_df: DataFrame with raw OHLCV data (datetime index, OHLCV columns)
        calibrate: Whether to apply calibration at the ensemble level
        skip_cleaning: If True, skip resampling/cleaning in base bundles

    Returns:
        PredictionResult with ensemble predictions and probabilities

    Raises:
        ValueError: If no base bundles available or meta-learner not loaded
    """
    # Ensure base bundles are loaded
    self._ensure_base_bundles_loaded()

    if not self._base_bundles:
        raise ValueError(
            "No base bundles loaded. Ensure base_bundle_paths are valid."
        )

    if self.meta_learner is None:
        raise ValueError("Meta-learner not loaded. Load bundle first.")

    # Get predictions from each base model via their own predict_from_raw
    base_predictions: dict[str, np.ndarray] = {}

    for model_name, bundle in self._base_bundles.items():
        output = bundle.predict_from_raw(
            raw_df,
            calibrate=False,       # Calibration at ensemble level only
            skip_cleaning=skip_cleaning,
        )
        base_predictions[model_name] = output.class_probabilities

    # Combine via meta-learner
    return self.predict(base_predictions, calibrate=calibrate)
```

### Metadata/Schema Changes

None.

### Dependency Order

- **Blocks:** Nothing in P0-B.
- **Blocked by:** P0-B-1 (base ModelBundle.predict_from_raw must work for all model types).

### Backward Compatibility

- Purely additive -- new method, no existing methods changed.
- Existing `predict()`, `predict_proba()`, `predict_classes()`, `predict_from_base_features()` all unchanged.
- Satisfies the `InferenceBundle` protocol (P0-A-1) structurally.

### Acceptance Criteria

1. `ensemble_bundle.predict_from_raw(raw_ohlcv_df)` returns `PredictionResult`.
2. Base model `predict_from_raw()` is called with `calibrate=False`.
3. Works with heterogeneous ensembles (e.g., XGBoost + LSTM base models) because each base bundle handles its own adapter routing.
4. Raises `ValueError` if base bundles are missing or meta-learner is None.
5. `ruff check src/inference/ensemble_bundle.py` passes.

### Effort Estimate

**S** -- 30-40 LOC.

---

## P0-B-5: Fix EnsembleBundle Relative Paths (Save/Load)

### Files to Modify

- `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` (MODIFY)

### API Changes

Update `save()` at L442-452 to store relative paths:

```python
# In save(), when writing base_bundles.json:
# Convert absolute paths to relative (relative to the ensemble bundle directory)
bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
relative_paths = []
for p in self.base_bundle_paths:
    try:
        relative_paths.append(str(Path(p).relative_to(path.parent)))
    except ValueError:
        # Cannot make relative (different drive/root); fall back to absolute
        relative_paths.append(str(p))

with open(bundles_path, "w") as f:
    json.dump(
        {
            "paths": relative_paths,
            "model_names": self.metadata.base_model_names,
        },
        f,
        indent=2,
    )
```

Update `load()` at L538-543 to resolve relative paths:

```python
# In load(), when reading base_bundles.json:
base_bundle_paths: list[Path] = []
bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
if bundles_path.exists():
    with open(bundles_path) as f:
        raw_paths = json.load(f).get("paths", [])
        for p_str in raw_paths:
            p = Path(p_str)
            if not p.is_absolute():
                # Resolve relative to ensemble bundle's parent directory
                p = (path.parent / p).resolve()
            base_bundle_paths.append(p)
```

### Metadata/Schema Changes

- `base_bundles.json` `paths` field changes from absolute to relative paths.
- Old bundles with absolute paths still load correctly (the `is_absolute()` check handles both).

### Dependency Order

- **Blocks:** Nothing.
- **Blocked by:** Nothing (independent fix).

### Backward Compatibility

- Old ensemble bundles with absolute paths in `base_bundles.json`: still load correctly because `Path(abs_path).is_absolute()` returns `True` and the path is used as-is.
- New ensemble bundles: use relative paths, portable across machines.
- The `try/except ValueError` in save handles edge cases (different drives on Windows, etc.).

### Acceptance Criteria

1. Save an EnsembleBundle, move it to a different directory, load it: base bundle paths resolve correctly if base bundles are also moved alongside.
2. Load an old EnsembleBundle with absolute paths: works unchanged.
3. `base_bundles.json` in newly saved bundles contains relative paths.
4. `ruff check src/inference/ensemble_bundle.py` passes.

### Effort Estimate

**S** -- 20-30 LOC.

---

## P0-B-6: Create `src/inference/errors.py` with InferenceShapeMismatchError

### Files to Create

- `/home/jake/Desktop/Research/src/inference/errors.py` (NEW, ~30 lines)

### API Changes

```python
"""
Custom error types for the inference module.

Domain-specific errors provide clearer diagnostics than generic ValueError
when shapes, features, or configurations don't match expectations.
"""

from __future__ import annotations


class InferenceError(Exception):
    """Base class for inference-related errors."""

    pass


class InferenceShapeMismatchError(InferenceError):
    """
    Raised when input tensor shape does not match model expectations.

    Provides diagnostic information about expected vs actual shapes
    to help users debug inference failures.
    """

    def __init__(
        self,
        expected_shape: tuple[int, ...] | str,
        actual_shape: tuple[int, ...],
        model_name: str = "",
        hint: str = "",
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        self.model_name = model_name
        self.hint = hint

        msg = f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        if model_name:
            msg = f"[{model_name}] {msg}"
        if hint:
            msg = f"{msg}. {hint}"
        super().__init__(msg)
```

### Metadata/Schema Changes

None.

### Dependency Order

- **Blocks:** Nothing immediately (can be used by P0-B-1/P0-B-2/P0-B-3 to replace ValueError, but not required).
- **Blocked by:** Nothing.

### Backward Compatibility

- New file, no existing code changes.
- Adoption in existing code (replacing `ValueError` raises) is optional and can be done incrementally.

### Acceptance Criteria

1. `python -c "from src.inference.errors import InferenceShapeMismatchError; print('OK')"` succeeds.
2. `grep -r "class InferenceShapeMismatchError" src/ --include="*.py" | wc -l` returns `1`.
3. `ruff check src/inference/errors.py` passes.
4. File includes `from __future__ import annotations`.
5. Error message includes expected shape, actual shape, model name, and hint.

### Effort Estimate

**S** -- 25-35 LOC.

---

# Dependency Graph

```
P0-A Tasks (Foundation):
                            ┌─────────────────────────────────────────────┐
                            │                                             │
  P0-A-1 (protocols.py) ───┤                                             │
                            │                                             │
  P0-A-2 (ScalingSource) ──┤  All independent, no inter-deps             │
                            │                                             │
  P0-A-3 (BundleMetadata)──┤                                             │
                            │                                             │
  P0-A-4 (calibrator fix)──┤                                             │
                            │                                             │
  P0-A-5 (double-scaling)──┤                                             │
                            └───────────────┬─────────────────────────────┘
                                            │
                                            v
                            P0-A-6 (protocol-aware BundleBuilder)
                            Depends on: P0-A-1, P0-A-3, P0-A-4

P0-B Tasks (Core Inference):
                            ┌─────────────────────────────────────────────┐
                            │                                             │
  P0-B-2 (_build_3d)   ────┤  Depend on P0-A-3 (metadata fields)         │
                            │  and P0-A-5 (skip_scaling fix)              │
  P0-B-3 (_build_4d)   ────┤                                             │
                            │                                             │
  P0-B-6 (errors.py)   ────┤  Independent                                │
                            │                                             │
  P0-B-5 (relative paths)──┤  Independent                                │
                            └───────────────┬─────────────────────────────┘
                                            │
                                            v
                            P0-B-1 (_apply_adapter routing)
                            Depends on: P0-B-2, P0-B-3
                                            │
                                            v
                            P0-B-4 (EnsembleBundle.predict_from_raw)
                            Depends on: P0-B-1
```

## Recommended Execution Order

```
Batch 1 (parallel, no deps):
  P0-A-1  Create protocols.py
  P0-A-2  Add ScalingSource enum
  P0-A-3  Extend BundleMetadata
  P0-A-4  Fix calibrator transfer
  P0-A-5  Fix double-scaling bug
  P0-B-6  Create errors.py

Batch 2 (depends on Batch 1):
  P0-A-6  Protocol-aware BundleBuilder
  P0-B-2  _build_3d_input
  P0-B-3  _build_4d_input
  P0-B-5  Fix relative paths

Batch 3 (depends on Batch 2):
  P0-B-1  _apply_adapter routing (integrates B-2 and B-3)

Batch 4 (depends on Batch 3):
  P0-B-4  EnsembleBundle.predict_from_raw
```

## Total Effort Summary

| Task | Effort | LOC Range | Files |
|------|--------|-----------|-------|
| P0-A-1 | S | 40-60 | 1 new |
| P0-A-2 | S | 10-15 | 1 modify |
| P0-A-3 | M | 50-80 | 1 modify |
| P0-A-4 | M | 30-50 | 2 modify |
| P0-A-5 | S | 1-5 | 1 modify |
| P0-A-6 | M | 60-90 | 1 modify |
| P0-B-1 | M | 40-60 | 1 modify |
| P0-B-2 | S | 25-35 | 1 modify |
| P0-B-3 | L | 80-120 | 1 modify |
| P0-B-4 | S | 30-40 | 1 modify |
| P0-B-5 | S | 20-30 | 1 modify |
| P0-B-6 | S | 25-35 | 1 new |
| **Total** | | **411-620** | **2 new, 5 modify** |

## Model Contract Routing Reference (from Agent 4)

For implementer reference, the routing decisions in P0-B-1 through P0-B-3 follow this table:

| Model | requires_4d | requires_sequences | Adapter Path | sequence_length | primary_tf | mtf_timeframes |
|-------|-------------|-------------------|--------------|-----------------|------------|----------------|
| xgboost | False | False | 2D passthrough | N/A | 15min | () |
| lightgbm | False | False | 2D passthrough | N/A | 15min | () |
| catboost | False | False | 2D passthrough | N/A | 15min | () |
| lstm | False | True | `_build_3d_input` | 60 | 5min | () |
| gru | False | True | `_build_3d_input` | 60 | 5min | () |
| tcn | False | True | `_build_3d_input` | 120 | 5min | () |
| inceptiontime | False | True | `_build_3d_input` | 60 | 5min | () |
| resnet1d | False | True | `_build_3d_input` | 60 | 5min | () |
| tft | False | True | `_build_3d_input` | 60 | 5min | () |
| nbeats | False | True | `_build_3d_input` | 60 | 5min | () |
| patchtst | True | True | `_build_4d_input` | 60 | 1min | (5min, 15min) |
| itransformer | True | True | `_build_4d_input` | 60 | 1min | (5min, 15min) |

---

*This document is a planning artifact. No code has been modified.*
