# Phase 3D: Cleanup — Concrete Implementation Plan

**Date:** 2026-02-15
**Status:** Ready for implementation
**Dependencies:** None (parallel with 3A-3C)

---

## Task 3D-1: Remove `_apply_regime()` No-Op

**File:** `src/inference/preprocessing_graph.py`
**Complexity:** LOW | **Lines changed:** -8

### Call site removal (line 496-498)

**Old code** (lines 496-498):
```python
        # Step 4: Regime detection
        if self.config.regime.enabled:
            df = self._apply_regime(df)
```

**New code:** Delete these 3 lines entirely. The regime features are already handled inside `_apply_features()` via `add_regime_features`.

### Method removal (lines 702-706)

**Old code** (lines 702-706):
```python
    def _apply_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply regime detection."""
        # Regime features are already added in _apply_features via add_regime_features
        # This method is for any additional regime-specific processing
        return df
```

**New code:** Delete these 5 lines entirely.

### Verification
```bash
grep -n "_apply_regime" src/inference/preprocessing_graph.py  # Should return 0 results
python -c "from src.inference.preprocessing_graph import PreprocessingGraph; print('OK')"
```

---

## Task 3D-2: Consolidate `CVMethod` Enum

**File:** `src/config/cv.py`
**Complexity:** LOW | **Lines changed:** ~4

### Current state
- **Canonical:** `src/core/types.py:163` — `class CVMethod(str, Enum)` with 5 values
- **Duplicate:** `src/config/cv.py:28` — identical 5 values
- **Re-exports:** `src/config/__init__.py:142` imports `CVMethod` from `cv.py` → still works after change

### Consumers (all import from canonical paths, NOT directly from cv.py)
- `src/models/training/unified_orchestrator.py:43` — imports from `src.core`
- `src/core/config.py:60` — imports from `src.core.types`
- `src/validation/cv/cv_orchestrator.py:29` — imports from `src.core`
- `src/config/__init__.py:142` — imports from `src.config.cv` (re-export chain)

### Change in `src/config/cv.py`

**Old code** (lines 19, 28-35):
```python
from enum import Enum

from src.config.base import BaseConfig

# =============================================================================
# ENUMS
# =============================================================================


class CVMethod(str, Enum):
    """Supported cross-validation methods."""

    PURGED_KFOLD = "purged_kfold"
    CPCV = "cpcv"
    WALK_FORWARD = "walk_forward"
    PBO = "pbo"
    STANDARD = "standard"
```

**New code** (replace lines 19-35):
```python
from enum import Enum

from src.config.base import BaseConfig
from src.core.types import CVMethod  # Canonical source

# =============================================================================
# ENUMS
# =============================================================================
```

Note: Keep `from enum import Enum` since `WindowType` enum (line 38) still needs it.

### Verification
```bash
grep -c "class CVMethod" src/ -r --include="*.py"  # Should return 1 (only in types.py)
python -c "from src.config.cv import CVMethod; print(CVMethod.WALK_FORWARD)"
python -c "from src.config import CVMethod; print(CVMethod.PURGED_KFOLD)"
python -c "from src.core.types import CVMethod; print(CVMethod.CPCV)"
```

---

## Task 3D-3: Consolidate `LabelingMethod` Enum

**File:** `src/config/data.py`
**Complexity:** LOW | **Lines changed:** ~4

### Current state
- **Canonical:** `src/core/types.py:213` — `class LabelingMethod(str, Enum)` with 4 values
- **Duplicate:** `src/config/data.py:52` — identical 4 values
- **Re-exports:** `src/config/__init__.py:161` imports `LabelingMethod` from `data.py` → still works

### Consumers (imports)
- `src/core/config.py:60` — imports from `src.core.types` (canonical)
- `src/config/__init__.py:161` — imports from `src.config.data` (re-export chain)
- `src/config/experiment.py:41` — imports from `src.config.data` (uses other things too)
- No file imports `LabelingMethod` directly from `data.py` by name

### Change in `src/config/data.py`

**Old code** (lines 52-58):
```python
class LabelingMethod(str, Enum):
    """Supported labeling methods."""

    TRIPLE_BARRIER = "triple_barrier"
    DIRECTIONAL = "directional"
    THRESHOLD = "threshold"
    REGRESSION = "regression"
```

**New code** (replace lines 52-58):
```python
from src.core.types import LabelingMethod  # Canonical source  # noqa: E402
```

Note: The `from enum import Enum` on line 20 is still needed by `ScalerType`, `FeatureCategory`, and `MTFMode` enums in the same file.

### Check `src/config/experiment.py` import

**Line 41:** `from src.config.data import (` — verify `LabelingMethod` is among the imports.

```bash
grep -A 20 "from src.config.data import" src/config/experiment.py
```

If `LabelingMethod` is imported there, it will continue to work because `data.py` now re-exports it from `types.py`.

### Verification
```bash
grep -c "class LabelingMethod" src/ -r --include="*.py"  # Should return 1 (only in types.py)
python -c "from src.config.data import LabelingMethod; print(LabelingMethod.TRIPLE_BARRIER)"
python -c "from src.config import LabelingMethod; print(LabelingMethod.DIRECTIONAL)"
python -c "from src.core.types import LabelingMethod; print(LabelingMethod.THRESHOLD)"
```

---

## Task 3D-4: `safe_pickle_load()` Utility

**New file:** `src/core/utils/safe_pickle.py`
**Complexity:** MEDIUM | **Lines changed:** ~40 new + 16 call site modifications

### New file content

```python
"""
Safe pickle loading utility.

Centralizes pickle deserialization with path validation and type checking.
All pickle files in ML Factory are trusted internal artifacts (model weights,
scalers, calibrators, checkpoints) — this utility adds defense-in-depth.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_pickle_load(
    path: Path | str,
    expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    """
    Load a pickle file with path validation and optional type checking.

    Args:
        path: Path to the pickle file.
        expected_type: Expected type(s) of the deserialized object.
            If provided and the loaded object doesn't match, logs a warning.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        pickle.UnpicklingError: If the file is not a valid pickle.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)  # noqa: S301

    if expected_type is not None and not isinstance(obj, expected_type):
        logger.warning(
            "Pickle type mismatch at %s: expected %s, got %s",
            path,
            expected_type,
            type(obj).__name__,
        )

    return obj
```

### Export from `src/core/utils/__init__.py`

Add to `src/core/utils/__init__.py`:
```python
from src.core.utils.safe_pickle import safe_pickle_load
```

### Call site replacements (16 sites across 13 files)

Each replacement follows this pattern:
1. Add `from src.core.utils.safe_pickle import safe_pickle_load` to imports
2. Replace `pickle.load(f)` with `safe_pickle_load(path, expected_type=<type>)`
3. Remove the surrounding `with open(path, "rb") as f:` block
4. Remove the `# SECURITY:` comment block (now handled by the utility)
5. If `import pickle` is no longer needed for `pickle.dump`, keep it; otherwise remove

Below are exact replacements per file:

---

#### Site 1: `src/factory.py:474`

**Old** (lines 471-475):
```python
            with open(training_cache_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (training results from this system)
                # External/untrusted pickle files could execute arbitrary code
                return pickle.load(f)
```

**New:**
```python
            return safe_pickle_load(training_cache_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 405 uses `pickle.dump`)

---

#### Site 2: `src/models/boosting/xgboost_model.py:295`

**Old** (lines 292-295):
```python
            with open(metadata_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (model metadata from this system)
                # External/untrusted pickle files could execute arbitrary code
                metadata = pickle.load(f)
```

**New:**
```python
            metadata = safe_pickle_load(metadata_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 276 uses `pickle.dump`)

---

#### Site 3: `src/models/boosting/catboost_model.py:281`

**Old** (lines 278-281):
```python
            with open(metadata_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (model metadata from this system)
                # External/untrusted pickle files could execute arbitrary code
                metadata = pickle.load(f)
```

**New:**
```python
            metadata = safe_pickle_load(metadata_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 262 uses `pickle.dump`)

---

#### Site 4: `src/models/boosting/lightgbm_model.py:352`

**Old** (lines 349-352):
```python
            with open(metadata_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (model metadata from this system)
                # External/untrusted pickle files could execute arbitrary code
                metadata = pickle.load(f)
```

**New:**
```python
            metadata = safe_pickle_load(metadata_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 334 uses `pickle.dump`)

---

#### Site 5: `src/models/calibration/conformal.py:482`

**Old** (lines 479-482):
```python
        with open(path, "rb") as f:
            # SECURITY: Only load from trusted internal paths (conformal predictors fitted by this system)
            # External/untrusted pickle files could execute arbitrary code
            state = pickle.load(f)
```

**New:**
```python
        state = safe_pickle_load(path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 462 uses `pickle.dump`)

---

#### Site 6: `src/models/calibration/calibrator.py:307`

**Old** (lines 304-307):
```python
        with open(path, "rb") as f:
            # SECURITY: Only load from trusted internal paths (calibrators fitted by this system)
            # External/untrusted pickle files could execute arbitrary code
            state = pickle.load(f)
```

**New:**
```python
        state = safe_pickle_load(path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 285 uses `pickle.dump`)

---

#### Site 7: `src/models/ensemble/xgboost_meta.py:279`

**Old** (lines 276-279):
```python
            with open(metadata_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (model metadata from this system)
                # External/untrusted pickle files could execute arbitrary code
                metadata = pickle.load(f)
```

**New:**
```python
            metadata = safe_pickle_load(metadata_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Note:** `import pickle` is inside local function scopes (lines 241, 262). The `pickle.dump` on line 256 still needs it. Keep the local `import pickle` in the save function, remove from load function.

---

#### Site 8: `src/data/pipeline/stages/scaling/scaler.py:499`

**Old** (lines 496-499):
```python
        with open(path, "rb") as f:
            # SECURITY: Only load from trusted internal paths (scalers fitted by this system)
            # External/untrusted pickle files could execute arbitrary code
            state = pickle.load(f)
```

**New:**
```python
        state = safe_pickle_load(path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 441 uses `pickle.dump`)

---

#### Sites 9-10: `src/core/utils/checkpoint_manager.py:208,224`

**Site 9 — Old** (lines 205-208):
```python
        with open(latest_checkpoint, "rb") as f:
            # SECURITY: Only load from trusted internal paths (checkpoints created by this system)
            # External/untrusted pickle files could execute arbitrary code
            checkpoint: dict[str, Any] = pickle.load(f)
```

**Site 9 — New:**
```python
        checkpoint: dict[str, Any] = safe_pickle_load(latest_checkpoint, expected_type=dict)
```

**Site 10 — Old** (lines 221-224):
```python
                with open(checkpoint_files[0], "rb") as f:
                    # SECURITY: Only load from trusted internal paths (W&B artifacts from this system)
                    # External/untrusted pickle files could execute arbitrary code
                    result: dict[str, Any] = pickle.load(f)
```

**Site 10 — New:**
```python
                result: dict[str, Any] = safe_pickle_load(checkpoint_files[0], expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 141 uses `pickle.dump`)

---

#### Site 11: `src/core/utils/cache.py:190`

**Old** (lines 187-190):
```python
            with open(disk_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (cache files created by this system)
                # External/untrusted pickle files could execute arbitrary code
                cached = pickle.load(f)
```

**New:**
```python
            cached = safe_pickle_load(disk_path, expected_type=dict)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 174 uses `pickle.dump`, line 150 uses `pickle.dumps`)

---

#### Sites 12-13: `src/inference/bundle.py:644,653`

**Site 12 — Old** (lines 641-644):
```python
            with open(scaler_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (scalers fitted by this system)
                # External/untrusted pickle files could execute arbitrary code
                scaler = pickle.load(f)
```

**Site 12 — New:**
```python
            scaler = safe_pickle_load(scaler_path)
```

**Site 13 — Old** (lines 650-653):
```python
            with open(calibrator_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (calibrators trained by this system)
                # External/untrusted pickle files could execute arbitrary code
                calibrator = pickle.load(f)
```

**Site 13 — New:**
```python
            calibrator = safe_pickle_load(calibrator_path)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (lines 409, 417 use `pickle.dump`)

Note: No `expected_type` for scaler/calibrator since they can be various sklearn types.

---

#### Sites 14-15: `src/inference/ensemble_bundle.py:561,580`

**Site 14 — Old** (lines 558-561):
```python
            with open(meta_dir / "model.pkl", "rb") as f:
                # SECURITY: Only load from trusted internal paths (models trained by this system)
                # External/untrusted pickle files could execute arbitrary code
                meta_learner = pickle.load(f)
```

**Site 14 — New:**
```python
            meta_learner = safe_pickle_load(meta_dir / "model.pkl")
```

**Site 15 — Old** (lines 577-580):
```python
            with open(scaler_path, "rb") as f:
                # SECURITY: Only load from trusted internal paths (scalers fitted by this system)
                # External/untrusted pickle files could execute arbitrary code
                scaler = pickle.load(f)
```

**Site 15 — New:**
```python
            scaler = safe_pickle_load(scaler_path)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (lines 468-471 use `pickle.dump`)

---

#### Site 16: `src/inference/preprocessing_graph.py:431`

**Old** (lines 428-431):
```python
        with open(path, "rb") as f:
            # SECURITY: Only load from trusted internal paths (scalers fitted by this system)
            # External/untrusted pickle files could execute arbitrary code
            self._scaler = pickle.load(f)
```

**New:**
```python
        self._scaler = safe_pickle_load(path)
```

**Import to add:** `from src.core.utils.safe_pickle import safe_pickle_load`
**Keep `import pickle`:** Yes (line 438 uses `pickle.dump`)

---

### Verification
```bash
# Should return 0 raw pickle.load calls
grep -rn "pickle\.load(" src/ --include="*.py" | grep -v safe_pickle | grep -v "# noqa"

# Should return 1 (only in safe_pickle.py itself)
grep -rn "pickle\.load(" src/ --include="*.py" | wc -l

# Import check
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('OK')"
```

---

## Task 3D-5: Neural Architecture Versioning

**File:** `src/models/neural/base_rnn.py`
**Complexity:** LOW | **Lines changed:** ~15

### Add version constant (after imports, ~line 34)

**Add after line 33 (after all imports):**
```python

# Architecture version — increment when changing model structure (layers, sizes, etc.)
# This is saved in checkpoints so we can detect incompatible loads.
ARCH_VERSION = "1.0"
```

### Update `save()` method (line 641-649)

**Old** (lines 641-648):
```python
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": self._config,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "seq_len": getattr(self, "_seq_len", None),  # N-BEATS needs this
            },
            path / "model.pt",
        )
```

**New:**
```python
        torch.save(
            {
                "arch_version": ARCH_VERSION,
                "model_state_dict": self._model.state_dict(),
                "config": self._config,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "seq_len": getattr(self, "_seq_len", None),  # N-BEATS needs this
            },
            path / "model.pt",
        )
```

### Update `load()` method (after line 662)

**Old** (lines 662-669):
```python
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        self._config = checkpoint["config"]
        self._n_features = checkpoint["n_features"]
        self._n_classes = checkpoint["n_classes"]
        # Restore seq_len for N-BEATS (and other models that need it)
        if "seq_len" in checkpoint:
            self._seq_len = checkpoint["seq_len"]
```

**New** (insert version check after torch.load):
```python
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        # Validate architecture version (warning only, not error — allows loading older models)
        saved_version = checkpoint.get("arch_version", "unknown")
        if saved_version != ARCH_VERSION:
            logger.warning(
                "Architecture version mismatch: checkpoint=%s, current=%s. "
                "Model may behave unexpectedly if architecture has changed.",
                saved_version,
                ARCH_VERSION,
            )

        self._config = checkpoint["config"]
        self._n_features = checkpoint["n_features"]
        self._n_classes = checkpoint["n_classes"]
        # Restore seq_len for N-BEATS (and other models that need it)
        if "seq_len" in checkpoint:
            self._seq_len = checkpoint["seq_len"]
```

### Verification
```bash
grep -n "ARCH_VERSION" src/models/neural/base_rnn.py  # Should show constant + save + load
python -c "from src.models.neural.base_rnn import ARCH_VERSION; print(ARCH_VERSION)"
```

---

## Task 3D-6: Feature Names in BundleBuilder

**File:** `src/inference/builder.py`
**Complexity:** LOW | **Lines changed:** ~5

### Context
After extracting the model (line 298) and feature columns (line 308), if the model supports `set_feature_names()`, call it. This ensures models loaded from checkpoints (not trained fresh) have feature names set.

### Change (after line 308, before line 310)

**Old** (lines 307-310):
```python
            # Extract feature columns
            feature_columns = self._extract_feature_columns(trainer, model_result.n_features)

            # Extract calibrator if requested
```

**New:**
```python
            # Extract feature columns
            feature_columns = self._extract_feature_columns(trainer, model_result.n_features)

            # Ensure model has feature names set (may not be set if loaded from checkpoint)
            if hasattr(model, "set_feature_names") and callable(model.set_feature_names):
                model.set_feature_names(feature_columns)

            # Extract calibrator if requested
```

### Verification
```bash
grep -n "set_feature_names" src/inference/builder.py  # Should show the new call
python -c "from src.inference.builder import BundleBuilder; print('OK')"
```

---

## Task 3D-7: Deprecation Warnings for Old Inference Classes

**Files:** `src/inference/pipeline.py`, `src/inference/orchestrator.py`
**Complexity:** LOW | **Lines changed:** ~10

### Change in `src/inference/pipeline.py` (line 146-162)

**Old `__init__`** (lines 146-162):
```python
    def __init__(
        self,
        bundles: list[ModelBundle],
        default_voting: str = "soft_vote",
    ) -> None:
        """
        Initialize InferencePipeline.

        Args:
            bundles: List of ModelBundle instances
            default_voting: Default ensemble voting method
        """
        if not bundles:
            raise ValueError("At least one bundle is required")

        self.bundles = bundles
        self.default_voting = default_voting
```

**New `__init__`** (add import + warning):
```python
    def __init__(
        self,
        bundles: list[ModelBundle],
        default_voting: str = "soft_vote",
    ) -> None:
        """
        Initialize InferencePipeline.

        .. deprecated::
            Use ``UniversalInferencePipeline`` instead.

        Args:
            bundles: List of ModelBundle instances
            default_voting: Default ensemble voting method
        """
        import warnings

        warnings.warn(
            "InferencePipeline is deprecated. Use UniversalInferencePipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not bundles:
            raise ValueError("At least one bundle is required")

        self.bundles = bundles
        self.default_voting = default_voting
```

### Change in `src/inference/orchestrator.py` (lines 85-101)

**Old `__init__`** (lines 85-101):
```python
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        """
        Initialize InferenceOrchestrator.

        Args:
            config: Optional PipelineConfig from src/core
        """
        self.config = config
        self._bundles: dict[str, Any] = {}  # model_name -> ModelBundle
        self._ensemble_bundle: Any | None = None  # EnsembleBundle or meta-learner
        self._preprocessing_graph: Any | None = None
        self._is_loaded = False

        logger.info("InferenceOrchestrator initialized")
```

**New `__init__`:**
```python
    def __init__(
        self,
        config: PipelineConfig | None = None,
    ) -> None:
        """
        Initialize InferenceOrchestrator.

        .. deprecated::
            Use ``UniversalInferencePipeline`` instead.

        Args:
            config: Optional PipelineConfig from src/core
        """
        import warnings

        warnings.warn(
            "InferenceOrchestrator is deprecated. Use UniversalInferencePipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.config = config
        self._bundles: dict[str, Any] = {}  # model_name -> ModelBundle
        self._ensemble_bundle: Any | None = None  # EnsembleBundle or meta-learner
        self._preprocessing_graph: Any | None = None
        self._is_loaded = False

        logger.info("InferenceOrchestrator initialized")
```

### Verification
```bash
python -W all -c "
from src.inference.pipeline import InferencePipeline
from src.inference.orchestrator import InferenceOrchestrator
" 2>&1 | grep -c "DeprecationWarning"  # Should be 2
```

---

## Execution Order

All 7 tasks are independent. Recommended order for efficiency:

1. **3D-1** (remove dead code) — simplest, 2 deletions
2. **3D-2 + 3D-3** (enum consolidation) — parallel, 2 small changes
3. **3D-7** (deprecation warnings) — 2 small additions
4. **3D-6** (feature names) — 2-line addition
5. **3D-5** (arch versioning) — small addition to save/load
6. **3D-4** (safe_pickle) — largest task, new file + 16 call sites

## Total Impact

| Metric | Count |
|--------|-------|
| New files | 1 (`safe_pickle.py`) |
| Modified files | 16 |
| Lines added | ~80 |
| Lines removed | ~70 |
| `pickle.load` calls eliminated | 16 |
| Duplicate enums removed | 2 |
| Dead methods removed | 1 |
