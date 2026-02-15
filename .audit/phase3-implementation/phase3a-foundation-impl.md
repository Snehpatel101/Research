# Phase 3A: Foundation — Concrete Implementation Plan

**Goal:** Establish TrainerProtocol, add Trainer properties, extend BundleMetadata, make BundleBuilder protocol-aware, fix calibrator transfer, add FeatureSpec auto-generation.

**Execution Order:**
```
3A-1 (TrainerProtocol) → 3A-2 (Trainer properties) → 3A-5 (Calibrator transfer)
                                                    → 3A-3 (BundleMetadata) → 3A-4 (BundleBuilder) → 3A-6 (FeatureSpec auto-gen)
```

3A-1 and 3A-2 are foundational. 3A-3 is independent of 3A-2 but must precede 3A-4. 3A-5 can be done in parallel with 3A-3/3A-4. 3A-6 depends on 3A-4.

---

## 3A-1: Create TrainerProtocol

**File:** `src/core/protocols.py` (NEW FILE)

**Complete file content:**

```python
"""
Protocols — Structural typing contracts for cross-module interfaces.

TrainerProtocol defines the interface that BundleBuilder expects from any
trainer instance. This replaces fragile getattr() probing with a clear,
type-checkable contract.

Usage:
    from src.core.protocols import TrainerProtocol

    def build_bundle(trainer: TrainerProtocol) -> ModelBundle:
        model = trainer.model
        scaler = trainer.scaler
        ...
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.models.base import BaseModel


@runtime_checkable
class TrainerProtocol(Protocol):
    """
    Structural interface for trainer objects used by BundleBuilder.

    Any class with these properties satisfies this protocol without
    explicit inheritance. The Trainer class (src/models/training/trainer.py)
    is the primary implementor.
    """

    @property
    def model(self) -> BaseModel: ...

    @property
    def scaler(self) -> Any | None: ...

    @property
    def feature_columns(self) -> list[str]: ...

    @property
    def calibrator(self) -> Any | None: ...

    @property
    def training_config(self) -> dict[str, Any]: ...

    @property
    def model_key(self) -> str: ...
```

**Import registration:** Add to `src/core/__init__.py`:
```python
from .protocols import TrainerProtocol
```

Add `"TrainerProtocol"` to the `__all__` list in `src/core/__init__.py`.

**Validation:**
```bash
python -c "from src.core.protocols import TrainerProtocol; print('OK')"
python -c "from src.core import TrainerProtocol; print('OK')"
```

---

## 3A-2: Add Trainer Properties

**File:** `src/models/training/trainer.py`

### Change 1: Add `self.scaler` in `__init__` (after line 107)

**Current code (lines 106-110):**
```python
        # Calibrator (set during run() if calibration is enabled)
        self.calibrator: ProbabilityCalibrator | None = None

        # Initialize experiment tracker
        self.tracker: ExperimentTracker = self._setup_tracker()
```

**New code:**
```python
        # Calibrator (set during run() if calibration is enabled)
        self.calibrator: ProbabilityCalibrator | None = None

        # Scaler (set during run()/run_prepared() when data is prepared)
        self.scaler: Any | None = None

        # Initialize experiment tracker
        self.tracker: ExperimentTracker = self._setup_tracker()
```

### Change 2: Add property methods (after `_generate_run_id` method, after line 134)

**Insert after line 134** (after `_generate_run_id` closes):

```python
    @property
    def feature_columns(self) -> list[str]:
        """Feature columns used for training (after feature set filtering)."""
        if self._feature_set_columns is not None:
            return list(self._feature_set_columns)
        return []

    @property
    def training_config(self) -> dict[str, Any]:
        """Training configuration as a dictionary."""
        return self.config.to_dict()

    @property
    def model_key(self) -> str:
        """Unique key for this model+horizon combination."""
        return f"{self.config.model_name}_h{self.config.horizon}"
```

**Note:** `self.model` and `self.calibrator` already exist as attributes (lines 94-97, 107). No additional properties needed for those — they satisfy the protocol via attribute access.

### Change 3: Capture scaler in `run()` method

In `run()`, the scaler comes from the container. The container applies scaling during `get_sklearn_arrays()`. We need to capture it. However, looking at the code, the container-based `run()` doesn't use an explicit scaler — the data comes pre-scaled from the pipeline. The scaler lives in the `PreparedData.scaler` attribute when using `run_prepared()`.

**In `run_prepared()` (after line 951, where `X_train = prepared.X_train`):**

Current code (lines 948-954):
```python
        # Extract data directly from PreparedData (no reshaping needed)
        X_train = prepared.X_train
        y_train = prepared.y_train
        w_train = prepared.train_weights if prepared.has_weights else np.ones(len(y_train))
        X_val = prepared.X_val
        y_val = prepared.y_val
```

New code:
```python
        # Extract data directly from PreparedData (no reshaping needed)
        X_train = prepared.X_train
        y_train = prepared.y_train
        w_train = prepared.train_weights if prepared.has_weights else np.ones(len(y_train))
        X_val = prepared.X_val
        y_val = prepared.y_val

        # Capture scaler for bundle building (TrainerProtocol)
        if prepared.scaler is not None:
            self.scaler = prepared.scaler

        # Capture feature names for bundle building (TrainerProtocol)
        if prepared.feature_names and self._feature_set_columns is None:
            self._feature_set_columns = prepared.feature_names
```

**Note on `run()` path:** The container-based `run()` receives pre-scaled data from the pipeline. The scaler is not directly accessible in this path. For `run()`, we leave `self.scaler = None` — the BundleBuilder already handles this case (scaler is optional). If needed later, the scaler can be extracted from the pipeline artifacts.

**Validation:**
```bash
python -c "
from src.core.protocols import TrainerProtocol
from src.models.training.trainer import Trainer
# Verify structural compatibility (can't instantiate without config, but check attrs exist)
required = ['model', 'scaler', 'feature_columns', 'calibrator', 'training_config', 'model_key']
for attr in required:
    assert hasattr(Trainer, attr) or any(attr in str(m) for m in dir(Trainer)), f'Missing: {attr}'
print('OK')
"
```

---

## 3A-3: Extend BundleMetadata

**File:** `src/inference/bundle.py`

### Change 1: Bump version (line 54)

**Current:**
```python
BUNDLE_VERSION = "1.2.0"  # Updated for FeatureSpec support (5-dimension optimization)
```

**New:**
```python
BUNDLE_VERSION = "1.3.0"  # Updated for TrainerProtocol + extended metadata
```

### Change 2: Add new fields to BundleMetadata dataclass (after line 92)

**Current (lines 70-92):**
```python
@dataclass
class BundleMetadata:
    """Metadata for a model bundle."""

    version: str
    created_at: str
    model_name: str
    model_family: str
    horizon: int
    n_features: int
    feature_hash: str
    requires_sequences: bool = False
    requires_4d: bool = False
    sequence_length: int = 0
    n_timeframes: int = 0
    has_calibrator: bool = False
    has_preprocessing_graph: bool = False
    preprocessing_graph_hash: str = ""
    has_feature_spec: bool = False
    feature_spec_hash: str = ""
    symbol: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
```

**New:**
```python
@dataclass
class BundleMetadata:
    """Metadata for a model bundle."""

    version: str
    created_at: str
    model_name: str
    model_family: str
    horizon: int
    n_features: int
    feature_hash: str
    requires_sequences: bool = False
    requires_4d: bool = False
    sequence_length: int = 0
    n_timeframes: int = 0
    has_calibrator: bool = False
    has_preprocessing_graph: bool = False
    preprocessing_graph_hash: str = ""
    has_feature_spec: bool = False
    feature_spec_hash: str = ""
    symbol: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # Phase 3A extensions
    scaling_source: str = "unknown"
    arch_version: str | None = None
    label_mapping: dict[str, Any] | None = None
    feature_names: list[str] = field(default_factory=list)
    scaler_type: str = "unknown"
    training_run_id: str | None = None
```

### Change 3: Update `to_dict()` (lines 94-115)

**Current:**
```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "feature_hash": self.feature_hash,
            "requires_sequences": self.requires_sequences,
            "requires_4d": self.requires_4d,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "has_calibrator": self.has_calibrator,
            "has_preprocessing_graph": self.has_preprocessing_graph,
            "preprocessing_graph_hash": self.preprocessing_graph_hash,
            "has_feature_spec": self.has_feature_spec,
            "feature_spec_hash": self.feature_spec_hash,
            "symbol": self.symbol,
            "training_metrics": self.training_metrics,
            "extra": self.extra,
        }
```

**New:**
```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "feature_hash": self.feature_hash,
            "requires_sequences": self.requires_sequences,
            "requires_4d": self.requires_4d,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "has_calibrator": self.has_calibrator,
            "has_preprocessing_graph": self.has_preprocessing_graph,
            "preprocessing_graph_hash": self.preprocessing_graph_hash,
            "has_feature_spec": self.has_feature_spec,
            "feature_spec_hash": self.feature_spec_hash,
            "symbol": self.symbol,
            "training_metrics": self.training_metrics,
            "extra": self.extra,
            "scaling_source": self.scaling_source,
            "arch_version": self.arch_version,
            "label_mapping": self.label_mapping,
            "feature_names": self.feature_names,
            "scaler_type": self.scaler_type,
            "training_run_id": self.training_run_id,
        }
```

### Change 4: Update `from_dict()` (lines 117-139)

**Current:**
```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            model_name=data["model_name"],
            model_family=data.get("model_family", "unknown"),
            horizon=data["horizon"],
            n_features=data["n_features"],
            feature_hash=data["feature_hash"],
            requires_sequences=data.get("requires_sequences", False),
            requires_4d=data.get("requires_4d", False),
            sequence_length=data.get("sequence_length", 0),
            n_timeframes=data.get("n_timeframes", 0),
            has_calibrator=data.get("has_calibrator", False),
            has_preprocessing_graph=data.get("has_preprocessing_graph", False),
            preprocessing_graph_hash=data.get("preprocessing_graph_hash", ""),
            has_feature_spec=data.get("has_feature_spec", False),
            feature_spec_hash=data.get("feature_spec_hash", ""),
            symbol=data.get("symbol", ""),
            training_metrics=data.get("training_metrics", {}),
            extra=data.get("extra", {}),
        )
```

**New:**
```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        return cls(
            version=data["version"],
            created_at=data.get("created_at", ""),
            model_name=data["model_name"],
            model_family=data.get("model_family", "unknown"),
            horizon=data.get("horizon", 0),
            n_features=data.get("n_features", 0),
            feature_hash=data.get("feature_hash", ""),
            requires_sequences=data.get("requires_sequences", False),
            requires_4d=data.get("requires_4d", False),
            sequence_length=data.get("sequence_length", 0),
            n_timeframes=data.get("n_timeframes", 0),
            has_calibrator=data.get("has_calibrator", False),
            has_preprocessing_graph=data.get("has_preprocessing_graph", False),
            preprocessing_graph_hash=data.get("preprocessing_graph_hash", ""),
            has_feature_spec=data.get("has_feature_spec", False),
            feature_spec_hash=data.get("feature_spec_hash", ""),
            symbol=data.get("symbol", ""),
            training_metrics=data.get("training_metrics", {}),
            extra=data.get("extra", {}),
            scaling_source=data.get("scaling_source", "unknown"),
            arch_version=data.get("arch_version"),
            label_mapping=data.get("label_mapping"),
            feature_names=data.get("feature_names", []),
            scaler_type=data.get("scaler_type", "unknown"),
            training_run_id=data.get("training_run_id"),
        )
```

**Note:** Also made `created_at`, `horizon`, `n_features`, `feature_hash` use `.get()` with safe defaults for forward-compatibility when loading older bundles with minimal metadata.

**Validation:**
```bash
python -c "
from src.inference.bundle import BundleMetadata
# Test new fields with defaults
m = BundleMetadata.from_dict({'version':'1.2.0','model_name':'xgb', 'created_at':'2025-01-01', 'horizon':20, 'n_features':100, 'feature_hash':'abc'})
print(f'scaling_source={m.scaling_source}')
print(f'scaler_type={m.scaler_type}')
print(f'feature_names={m.feature_names}')
assert m.scaling_source == 'unknown'
assert m.scaler_type == 'unknown'
assert m.feature_names == []
# Test round-trip
d = m.to_dict()
m2 = BundleMetadata.from_dict(d)
assert m2.scaling_source == 'unknown'
print('OK')
"
```

---

## 3A-4: Protocol-Aware BundleBuilder

**File:** `src/inference/builder.py`

### Change 1: Add import for TrainerProtocol (after line 40)

**Current (lines 40-45):**
```python
from src.core import PipelineConfig

if TYPE_CHECKING:
    from src.models.ensemble.orchestrator import EnsembleResult
    from src.models.training.unified_orchestrator import TrainingRunResult
```

**New:**
```python
from src.core import PipelineConfig
from src.core.protocols import TrainerProtocol

if TYPE_CHECKING:
    from src.models.ensemble.orchestrator import EnsembleResult
    from src.models.training.unified_orchestrator import TrainingRunResult
```

### Change 2: Replace `_extract_model()` (lines 557-580)

**Current:**
```python
    def _extract_model(self, trainer: Any) -> Any | None:
        """
        Extract model from trainer instance.

        Tries multiple attribute names for compatibility with different
        trainer implementations.

        Args:
            trainer: Trainer instance

        Returns:
            Model instance or None
        """
        # Try common attribute names
        for attr in ["model", "_model", "estimator", "_estimator"]:
            model = getattr(trainer, attr, None)
            if model is not None:
                return model

        # Try callable model property
        if hasattr(trainer, "get_model") and callable(trainer.get_model):
            return trainer.get_model()

        return None
```

**New:**
```python
    def _extract_model(self, trainer: Any) -> Any | None:
        """
        Extract model from trainer instance.

        Uses TrainerProtocol for type-safe access when available,
        falls back to attribute probing for legacy trainers.

        Args:
            trainer: Trainer instance

        Returns:
            Model instance or None
        """
        if isinstance(trainer, TrainerProtocol):
            return trainer.model

        # Legacy fallback with warning
        logger.warning(
            f"Trainer {type(trainer).__name__} does not satisfy TrainerProtocol, "
            "using legacy attribute probing"
        )
        for attr in ["model", "_model", "estimator", "_estimator"]:
            model = getattr(trainer, attr, None)
            if model is not None:
                return model

        if hasattr(trainer, "get_model") and callable(trainer.get_model):
            return trainer.get_model()

        return None
```

### Change 3: Replace `_extract_scaler()` (lines 582-596)

**Current:**
```python
    def _extract_scaler(self, trainer: Any) -> Any | None:
        """
        Extract scaler from trainer instance.

        Args:
            trainer: Trainer instance

        Returns:
            Scaler instance or None
        """
        for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
            scaler = getattr(trainer, attr, None)
            if scaler is not None:
                return scaler
        return None
```

**New:**
```python
    def _extract_scaler(self, trainer: Any) -> Any | None:
        """
        Extract scaler from trainer instance.

        Args:
            trainer: Trainer instance

        Returns:
            Scaler instance or None
        """
        if isinstance(trainer, TrainerProtocol):
            return trainer.scaler

        # Legacy fallback
        for attr in ["scaler", "_scaler", "feature_scaler", "_feature_scaler"]:
            scaler = getattr(trainer, attr, None)
            if scaler is not None:
                return scaler
        return None
```

### Change 4: Replace `_extract_feature_columns()` (lines 598-628)

**Current:**
```python
    def _extract_feature_columns(
        self,
        trainer: Any,
        n_features: int,
    ) -> list[str]:
        """
        Extract feature column names from trainer.

        Args:
            trainer: Trainer instance
            n_features: Number of features (fallback for generic names)

        Returns:
            List of feature column names
        """
        # Try common attribute names
        for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:
            columns = getattr(trainer, attr, None)
            if columns is not None and len(columns) > 0:
                return list(columns)

        # Try to get from scaler if available
        scaler = self._extract_scaler(trainer)
        if scaler is not None:
            columns = getattr(scaler, "feature_names_in_", None)
            if columns is not None:
                return list(columns)

        # Fallback to generic names
        logger.warning("No feature columns found, using generic names")
        return [f"f{i}" for i in range(n_features)]
```

**New:**
```python
    def _extract_feature_columns(
        self,
        trainer: Any,
        n_features: int,
    ) -> list[str]:
        """
        Extract feature column names from trainer.

        Args:
            trainer: Trainer instance
            n_features: Number of features (fallback for generic names)

        Returns:
            List of feature column names
        """
        if isinstance(trainer, TrainerProtocol):
            columns = trainer.feature_columns
            if columns:
                return columns

        # Legacy fallback
        for attr in ["feature_columns", "_feature_columns", "feature_names", "_feature_names"]:
            columns = getattr(trainer, attr, None)
            if columns is not None and len(columns) > 0:
                return list(columns)

        # Try to get from scaler if available
        scaler = self._extract_scaler(trainer)
        if scaler is not None:
            columns = getattr(scaler, "feature_names_in_", None)
            if columns is not None:
                return list(columns)

        # Fallback to generic names
        logger.warning("No feature columns found, using generic names")
        return [f"f{i}" for i in range(n_features)]
```

### Change 5: Replace `_extract_calibrator()` (lines 630-644)

**Current:**
```python
    def _extract_calibrator(self, trainer: Any) -> Any | None:
        """
        Extract probability calibrator from trainer.

        Args:
            trainer: Trainer instance

        Returns:
            Calibrator instance or None
        """
        for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
            calibrator = getattr(trainer, attr, None)
            if calibrator is not None:
                return calibrator
        return None
```

**New:**
```python
    def _extract_calibrator(self, trainer: Any, model_result: Any = None) -> Any | None:
        """
        Extract probability calibrator from trainer or model result.

        Checks TrainerProtocol first, then model_result.calibrator (from
        orchestrator calibration), then legacy attribute probing.

        Args:
            trainer: Trainer instance
            model_result: Optional ModelTrainingResult (may have calibrator from orchestrator)

        Returns:
            Calibrator instance or None
        """
        if isinstance(trainer, TrainerProtocol):
            cal = trainer.calibrator
            if cal is not None:
                return cal

        # Check model_result.calibrator (from orchestrator auto-calibration)
        if model_result is not None:
            cal = getattr(model_result, "calibrator", None)
            if cal is not None:
                return cal

        # Legacy fallback
        for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
            calibrator = getattr(trainer, attr, None)
            if calibrator is not None:
                return calibrator
        return None
```

### Change 6: Update calibrator extraction call in `build_from_training_result()` (line 313)

**Current (line 311-313):**
```python
            if include_calibrator:
                calibrator = self._extract_calibrator(trainer)
```

**New:**
```python
            if include_calibrator:
                calibrator = self._extract_calibrator(trainer, model_result)
```

### Change 7: Replace hardcoded values in `_create_preprocessing_graph()` (lines 512-555)

**Current (lines 522-549):**
```python
        # Build pipeline config dict from PipelineConfig
        pipeline_config = {
            "horizons": self.config.horizons,
            "mtf_timeframes": self.config.mtf_timeframes,
            "clean": {
                "source_timeframe": "1min",
                "target_timeframe": "5min",
            },
            "features": {
                "scale_periods": True,
                "base_timeframe": "5min",
            },
            "mtf": {
                "enable_mtf": True,
                "base_timeframe": "5min",
                "mtf_timeframes": self.config.mtf_timeframes,
                "mode": "both",
            },
            "wavelets": {
                "enable_wavelets": True,
            },
            "regime": {
                "enabled": True,
            },
            "scaling": {
                "scaler_type": "robust",
                "clip_outliers": True,
            },
        }
```

**New:**
```python
        # Build pipeline config dict from PipelineConfig
        # Use config values instead of hardcoded defaults
        source_tf = getattr(self.config, "source_timeframe", "1min")
        target_tf = getattr(self.config, "target_timeframe", "5min")
        scaler_type = getattr(self.config, "scaler_type", "robust")

        pipeline_config = {
            "horizons": self.config.horizons,
            "mtf_timeframes": self.config.mtf_timeframes,
            "clean": {
                "source_timeframe": source_tf,
                "target_timeframe": target_tf,
            },
            "features": {
                "scale_periods": True,
                "base_timeframe": target_tf,
            },
            "mtf": {
                "enable_mtf": True,
                "base_timeframe": target_tf,
                "mtf_timeframes": self.config.mtf_timeframes,
                "mode": "both",
            },
            "wavelets": {
                "enable_wavelets": True,
            },
            "regime": {
                "enabled": True,
            },
            "scaling": {
                "scaler_type": scaler_type,
                "clip_outliers": True,
            },
        }
```

**Note:** Uses `getattr` with defaults to maintain backward compatibility — if PipelineConfig doesn't have `source_timeframe`/`target_timeframe`/`scaler_type` attributes yet, the defaults match the current hardcoded values.

---

## 3A-5: Calibrator Transfer Fix

**File:** `src/models/training/unified_orchestrator.py`

### Change 1: Add `calibrator` field to `ModelTrainingResult` (after line 100)

**Current (lines 77-101):**
```python
@dataclass
class ModelTrainingResult:
    """
    Result from training a single model.

    Attributes:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        horizon: Prediction horizon in bars
        metrics: Validation metrics dict (val_f1, val_accuracy, etc.)
        oof_prediction: Optional OOF predictions from CV
        trainer: Optional Trainer instance (for inference)
        training_time_seconds: Time taken to train
        n_features: Number of features used
        data_rank: Data dimensionality (2, 3, or 4)
    """

    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
```

**New:**
```python
@dataclass
class ModelTrainingResult:
    """
    Result from training a single model.

    Attributes:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        horizon: Prediction horizon in bars
        metrics: Validation metrics dict (val_f1, val_accuracy, etc.)
        oof_prediction: Optional OOF predictions from CV
        trainer: Optional Trainer instance (for inference)
        training_time_seconds: Time taken to train
        n_features: Number of features used
        data_rank: Data dimensionality (2, 3, or 4)
        calibrator: Optional probability calibrator (from auto-calibration)
    """

    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
    calibrator: Any | None = None
```

### Change 2: Capture calibrator from orchestrator's `_calibrate_model` result

In `_calibrate_model()` (line 922-999), the calibrator is attached to the service result via `result.calibrator = calibrator` (line 993). This service result is then used to construct `ModelTrainingResult` at lines 799-807 and 912-920.

**Update parallel path (lines 799-807):**

**Current:**
```python
            result = ModelTrainingResult(
                model_name=service_result.model_name,
                horizon=service_result.horizon,
                metrics=service_result.metrics,
                trainer=service_result.trainer,
                training_time_seconds=service_result.training_time_seconds,
                n_features=service_result.n_features,
                data_rank=service_result.data_rank,
            )
```

**New:**
```python
            result = ModelTrainingResult(
                model_name=service_result.model_name,
                horizon=service_result.horizon,
                metrics=service_result.metrics,
                trainer=service_result.trainer,
                training_time_seconds=service_result.training_time_seconds,
                n_features=service_result.n_features,
                data_rank=service_result.data_rank,
                calibrator=getattr(service_result, "calibrator", None),
            )
```

**Update sequential path (lines 912-920):**

**Current:**
```python
        return ModelTrainingResult(
            model_name=result.model_name,
            horizon=result.horizon,
            metrics=result.metrics,
            trainer=result.trainer,
            training_time_seconds=result.training_time_seconds,
            n_features=result.n_features,
            data_rank=result.data_rank,
        )
```

**New:**
```python
        return ModelTrainingResult(
            model_name=result.model_name,
            horizon=result.horizon,
            metrics=result.metrics,
            trainer=result.trainer,
            training_time_seconds=result.training_time_seconds,
            n_features=result.n_features,
            data_rank=result.data_rank,
            calibrator=getattr(result, "calibrator", None),
        )
```

### Change 3: Update `to_dict()` to include calibrator status (line 102-111)

**Current:**
```python
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "n_features": self.n_features,
            "data_rank": self.data_rank,
        }
```

**New:**
```python
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "n_features": self.n_features,
            "data_rank": self.data_rank,
            "has_calibrator": self.calibrator is not None,
        }
```

---

## 3A-6: FeatureSpec Auto-Generation

**File:** `src/inference/builder.py`

### Change 1: Add `_auto_generate_feature_spec()` method (after `_extract_calibrator`, ~line 644)

**Insert new method:**

```python
    def _auto_generate_feature_spec(
        self,
        model_result: Any,
        trainer: Any,
    ) -> Any | None:
        """
        Auto-generate a FeatureSpec from training artifacts when not explicitly provided.

        Uses the trainer's feature columns and model contract to build a minimal
        FeatureSpec for inference parity.

        Args:
            model_result: ModelTrainingResult with training metadata
            trainer: Trainer instance with feature information

        Returns:
            FeatureSpec instance or None if generation fails
        """
        try:
            from src.core.contracts.feature_spec import FeatureSpec
        except ImportError:
            logger.debug("FeatureSpec not available, skipping auto-generation")
            return None

        # Extract feature columns
        feature_columns = self._extract_feature_columns(trainer, model_result.n_features)
        if not feature_columns or all(c.startswith("f") and c[1:].isdigit() for c in feature_columns):
            # Generic names — not useful for a FeatureSpec
            return None

        try:
            feature_spec = FeatureSpec(
                selected_features=feature_columns,
                horizon=model_result.horizon,
                model_name=model_result.model_name,
                symbol=self.config.symbol,
            )
            logger.info(
                f"Auto-generated FeatureSpec for {model_result.model_name}: "
                f"{len(feature_columns)} features"
            )
            return feature_spec
        except Exception as e:
            logger.debug(f"Failed to auto-generate FeatureSpec: {e}")
            return None
```

### Change 2: Call auto-generation in `build_from_training_result()` (after line 318)

**Current (lines 315-318):**
```python
            # Get feature spec if provided
            feature_spec = None
            if feature_specs is not None:
                feature_spec = feature_specs.get(key)
```

**New:**
```python
            # Get feature spec if provided, or auto-generate
            feature_spec = None
            if feature_specs is not None:
                feature_spec = feature_specs.get(key)
            if feature_spec is None:
                feature_spec = self._auto_generate_feature_spec(model_result, trainer)
```

**Note:** This is a best-effort addition. If `FeatureSpec` requires additional constructor arguments that we don't have (like `triple_barrier_params`), the `try/except` will catch it gracefully. The exact `FeatureSpec` constructor signature should be verified before implementation — if it requires more fields, the auto-generation should populate what it can and leave others as defaults.

---

## Summary of Changes

| Task | File | Type | Lines Changed |
|------|------|------|---------------|
| 3A-1 | `src/core/protocols.py` | NEW | ~45 lines |
| 3A-1 | `src/core/__init__.py` | EDIT | +2 lines (import + __all__) |
| 3A-2 | `src/models/training/trainer.py` | EDIT | ~25 lines (scaler attr + 3 properties + scaler capture) |
| 3A-3 | `src/inference/bundle.py` | EDIT | ~30 lines (6 new fields, to_dict, from_dict, version bump) |
| 3A-4 | `src/inference/builder.py` | EDIT | ~80 lines (import + 4 extract methods + preprocessing graph) |
| 3A-5 | `src/models/training/unified_orchestrator.py` | EDIT | ~15 lines (calibrator field + 2 construction sites + to_dict) |
| 3A-6 | `src/inference/builder.py` | EDIT | ~40 lines (new method + call site) |

**Total: ~237 lines changed/added across 5 files (1 new, 4 modified)**

---

## Validation Script

```bash
#!/bin/bash
set -e

echo "=== Phase 3A Validation ==="

# 3A-1: TrainerProtocol importable
python -c "from src.core.protocols import TrainerProtocol; print('3A-1: TrainerProtocol import OK')"
python -c "from src.core import TrainerProtocol; print('3A-1: TrainerProtocol re-export OK')"

# 3A-2: Trainer has protocol properties
python -c "
from src.models.training.trainer import Trainer
for attr in ['model', 'scaler', 'feature_columns', 'calibrator', 'training_config', 'model_key']:
    assert hasattr(Trainer, attr), f'Missing: {attr}'
print('3A-2: Trainer properties OK')
"

# 3A-3: BundleMetadata round-trip with new fields
python -c "
from src.inference.bundle import BundleMetadata, BUNDLE_VERSION
m = BundleMetadata.from_dict({'version':'1.2.0','model_name':'xgb','created_at':'2025-01-01','horizon':20,'n_features':100,'feature_hash':'abc'})
assert m.scaling_source == 'unknown'
assert m.feature_names == []
d = m.to_dict()
assert 'scaling_source' in d
assert 'feature_names' in d
assert BUNDLE_VERSION == '1.3.0'
print('3A-3: BundleMetadata extensions OK')
"

# 3A-4: BundleBuilder uses protocol
python -c "
from src.inference.builder import BundleBuilder
from src.core.protocols import TrainerProtocol
print('3A-4: BundleBuilder + TrainerProtocol import OK')
"

# 3A-5: ModelTrainingResult has calibrator field
python -c "
from src.models.training.unified_orchestrator import ModelTrainingResult
r = ModelTrainingResult(model_name='xgb', horizon=20, calibrator='test')
assert r.calibrator == 'test'
d = r.to_dict()
assert 'has_calibrator' in d
print('3A-5: Calibrator transfer OK')
"

# Lint check
ruff check src/core/protocols.py src/inference/bundle.py src/inference/builder.py src/models/training/trainer.py src/models/training/unified_orchestrator.py

echo "=== All Phase 3A validations passed ==="
```

---

## Risk Notes

1. **`FeatureSpec` constructor signature** (3A-6): The auto-generation assumes `FeatureSpec(selected_features=..., horizon=..., model_name=..., symbol=...)`. If the constructor requires additional mandatory fields, the try/except handles it gracefully — just verify the actual signature before implementing.

2. **Trainer protocol compliance** (3A-2): The `Trainer` class already has `model` and `calibrator` as attributes. Adding `feature_columns` as a property (vs the existing `_feature_set_columns` private attr) means `feature_columns` shadows the existing `self.feature_columns` attribute on `ModelBundle`. This is fine — they serve different purposes in different classes.

3. **Backward compatibility** (3A-3): The `from_dict()` changes use `.get()` with safe defaults, so loading bundles saved with v1.2.0 metadata will work without errors — new fields simply get their default values.

4. **`PipelineConfig` attributes** (3A-4): The preprocessing graph now uses `getattr(self.config, "source_timeframe", "1min")` etc. If these attributes don't exist on `PipelineConfig` yet, the defaults match the currently hardcoded values — zero behavior change until the config is extended.
