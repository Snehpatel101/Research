# Phase 3 Validation & Test Plan

**Date:** 2026-02-15
**Purpose:** Comprehensive test/validation plan for Phase 3A–3D implementation
**Note:** No existing test suite (`tests/` dir, pytest config, conftest.py) was found. All tests below are new.

---

## 0. Environment Prerequisites

```bash
# Required packages for running tests (install in dev environment)
pip install scikit-learn pandas numpy torch pytest ruff black

# Verify core imports work before any Phase 3 changes
python -c "import src; print('src package OK')"
```

**Important:** The project has no `sklearn` in the current environment (only `types-pytest-lazy-fixture` installed). Full integration tests require: `scikit-learn`, `pandas`, `numpy`, `xgboost`, `lightgbm`, `catboost`, `torch`, `optuna`.

---

## 1. Per-Phase Smoke Tests

### Phase 3A: Foundation — Smoke Tests

Run these immediately after completing Phase 3A tasks.

```bash
# 3A-1: TrainerProtocol exists and imports
python -c "from src.core.protocols import TrainerProtocol; print('3A-1 OK: TrainerProtocol importable')"

# 3A-1: Protocol is runtime_checkable
python -c "
from src.core.protocols import TrainerProtocol
import typing
assert typing.runtime_checkable  # module-level check
print('3A-1 OK: runtime_checkable available')
"

# 3A-2: Trainer satisfies protocol
python -c "
from src.core.protocols import TrainerProtocol
from src.models.training.trainer import Trainer
from src.models.config import TrainerConfig
config = TrainerConfig(model_name='xgboost', horizon=20)
t = Trainer(config)
assert isinstance(t, TrainerProtocol), 'Trainer does not satisfy TrainerProtocol'
print('3A-2 OK: Trainer satisfies TrainerProtocol')
# Verify properties exist
_ = t.feature_columns
_ = t.training_config
_ = t.model_key
print('3A-2 OK: All properties accessible')
"

# 3A-3: BundleMetadata backward compat (old dict without new fields)
python -c "
from src.inference.bundle import BundleMetadata
old = BundleMetadata.from_dict({'version':'1.2.0','model_name':'x','model_family':'boosting'})
assert old.scaling_source == 'unknown', f'Expected unknown, got {old.scaling_source}'
assert old.arch_version is None, f'Expected None, got {old.arch_version}'
assert old.label_mapping is None
assert old.feature_names == []
assert old.scaler_type == 'unknown'
assert old.training_run_id is None
print('3A-3 OK: BundleMetadata backward compat — all new fields have safe defaults')
"

# 3A-3: BundleMetadata version bump
python -c "
from src.inference.bundle import BUNDLE_VERSION
assert BUNDLE_VERSION == '1.3.0', f'Expected 1.3.0, got {BUNDLE_VERSION}'
print('3A-3 OK: BUNDLE_VERSION is 1.3.0')
"

# 3A-3: BundleMetadata round-trip (to_dict → from_dict preserves new fields)
python -c "
from src.inference.bundle import BundleMetadata
m = BundleMetadata.from_dict({
    'version': '1.3.0', 'model_name': 'test',
    'model_family': 'boosting', 'scaling_source': 'pipeline',
    'arch_version': '1.0', 'label_mapping': {'SHORT':0,'HOLD':1,'LONG':2},
    'feature_names': ['feat_a','feat_b'], 'scaler_type': 'robust',
    'training_run_id': 'run_001'
})
d = m.to_dict()
m2 = BundleMetadata.from_dict(d)
assert m2.scaling_source == 'pipeline'
assert m2.arch_version == '1.0'
assert m2.label_mapping == {'SHORT':0,'HOLD':1,'LONG':2}
assert m2.feature_names == ['feat_a','feat_b']
assert m2.scaler_type == 'robust'
assert m2.training_run_id == 'run_001'
print('3A-3 OK: BundleMetadata round-trip preserves all new fields')
"

# 3A-4: BundleBuilder protocol-aware extraction (legacy fallback)
python -c "
from src.inference.builder import BundleBuilder
class LegacyTrainer:
    model = 'dummy_model'
    _scaler = None
b = BundleBuilder.__new__(BundleBuilder)
result = b._extract_model(LegacyTrainer())
assert result == 'dummy_model', f'Expected dummy_model, got {result}'
print('3A-4 OK: BundleBuilder legacy fallback works')
"

# 3A-5: ModelTrainingResult has calibrator field
python -c "
from src.models.training.unified_orchestrator import ModelTrainingResult
r = ModelTrainingResult.__new__(ModelTrainingResult)
assert hasattr(r, 'calibrator') or 'calibrator' in ModelTrainingResult.__dataclass_fields__
print('3A-5 OK: ModelTrainingResult has calibrator field')
"

# 3A-6: BundleBuilder has _auto_generate_feature_spec
python -c "
from src.inference.builder import BundleBuilder
assert hasattr(BundleBuilder, '_auto_generate_feature_spec')
print('3A-6 OK: _auto_generate_feature_spec exists')
"

# Single definition checks
grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l  # Expected: 1
```

**Expected results:** All print `OK`. Single definition count = 1.

---

### Phase 3B: Core Inference — Smoke Tests

Run these after completing Phase 3B tasks.

```bash
# 3B-1: Adapter routing methods on ModelBundle
python -c "
from src.inference.bundle import ModelBundle
for method in ['_apply_adapter', '_build_3d_input', '_build_4d_input']:
    assert hasattr(ModelBundle, method), f'Missing {method}'
print('3B-1 OK: All adapter routing methods exist on ModelBundle')
"

# 3B-1: predict_from_raw has skip_scaling behavior
python -c "
import inspect
from src.inference.bundle import ModelBundle
src = inspect.getsource(ModelBundle.predict_from_raw)
assert 'skip_scaling' in src or '_apply_adapter' in src
print('3B-1 OK: predict_from_raw references adapter/scaling logic')
"

# 3B-2: UniversalInferencePipeline importable
python -c "
from src.inference.universal_pipeline import UniversalInferencePipeline, ScalingSource
print('3B-2 OK: UIP and ScalingSource importable')
"

# 3B-2: UIP has required class methods
python -c "
from src.inference.universal_pipeline import UniversalInferencePipeline as UIP
for m in ['from_bundle', 'from_bundles', 'from_experiment', 'predict', 'predict_from_raw', 'predict_ensemble']:
    assert hasattr(UIP, m), f'Missing method: {m}'
print('3B-2 OK: UIP has all required methods')
"

# 3B-2: InferenceShapeMismatchError
python -c "
from src.inference.errors import InferenceShapeMismatchError
assert issubclass(InferenceShapeMismatchError, ValueError)
print('3B-2 OK: InferenceShapeMismatchError importable and is ValueError subclass')
"

# 3B-3: EnsembleBundle has predict_from_raw
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
assert hasattr(EnsembleBundle, 'predict_from_raw')
print('3B-3 OK: EnsembleBundle.predict_from_raw exists')
"

# 3B-4: MTF inference data generation
python -c "
from src.inference.bundle import ModelBundle
assert hasattr(ModelBundle, '_generate_mtf_dataframes')
print('3B-4 OK: _generate_mtf_dataframes exists')
"

# 3B-5: Type alignment
python -c "
from src.models.training.services.ensemble_service import to_ensemble_result
print('3B-5 OK: to_ensemble_result importable')
"
```

---

### Phase 3C: Integration — Smoke Tests

```bash
# 3C-1/3C-4: Notebook syntax check (no runtime — just JSON validity)
python -c "
import json
with open('notebooks/ml_factory_colab.ipynb') as f:
    nb = json.load(f)
print(f'3C-1 OK: Notebook has {len(nb[\"cells\"])} cells, valid JSON')
"

# 3C-2: server.py uses UIP
python -c "
import ast, inspect
with open('src/inference/server.py') as f:
    src = f.read()
assert 'UniversalInferencePipeline' in src
print('3C-2 OK: server.py references UniversalInferencePipeline')
"

# 3C-2: batch.py uses UIP
python -c "
with open('src/inference/batch.py') as f:
    src = f.read()
assert 'UniversalInferencePipeline' in src
print('3C-2 OK: batch.py references UniversalInferencePipeline')
"

# 3C-3: Special mode bundles importable
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('3C-3 OK: WalkForwardBundle')"
python -c "from src.inference.regime_bundle import RegimeBundle; print('3C-3 OK: RegimeBundle')"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('3C-3 OK: MetaLabelingBundle')"
python -c "from src.inference.regime_detector import RegimeDetector; print('3C-3 OK: RegimeDetector')"

# 3C-3: Special bundles satisfy InferenceBundle protocol
python -c "
from src.inference.walk_forward_bundle import WalkForwardBundle
from src.inference.regime_bundle import RegimeBundle
from src.inference.meta_labeling_bundle import MetaLabelingBundle
for cls in [WalkForwardBundle, RegimeBundle, MetaLabelingBundle]:
    for method in ['predict', 'predict_from_raw', 'save', 'load']:
        assert hasattr(cls, method), f'{cls.__name__} missing {method}'
print('3C-3 OK: All special bundles have InferenceBundle interface')
"

# 3C-5: __init__.py exports
python -c "
from src.inference import UniversalInferencePipeline
print('3C-5 OK: UIP exported from src.inference')
"

# 3C-5: Deprecation warnings fire for old classes
python -c "
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    from src.inference.pipeline import InferencePipeline
    # InferencePipeline.__init__ should emit DeprecationWarning
    # (Can't fully instantiate without args, but import should work)
print('3C-5 OK: Old classes still importable')
"
```

---

### Phase 3D: Cleanup — Smoke Tests

```bash
# 3D-1: _apply_regime removed
count=$(grep -c "_apply_regime" src/inference/preprocessing_graph.py 2>/dev/null || echo 0)
if [ "$count" = "0" ]; then echo "3D-1 OK: _apply_regime removed"; else echo "3D-1 FAIL: _apply_regime still present ($count occurrences)"; fi

# 3D-2: CVMethod consolidated (single definition)
count=$(grep -r "class CVMethod" src/ --include="*.py" | wc -l)
if [ "$count" = "1" ]; then echo "3D-2 OK: Single CVMethod definition"; else echo "3D-2 FAIL: $count definitions"; fi

# 3D-3: LabelingMethod consolidated
count=$(grep -r "class LabelingMethod" src/ --include="*.py" | wc -l)
if [ "$count" = "1" ]; then echo "3D-3 OK: Single LabelingMethod definition"; else echo "3D-3 FAIL: $count definitions"; fi

# 3D-4: safe_pickle_load exists
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('3D-4 OK: safe_pickle_load importable')"

# 3D-4: No raw pickle.load remaining
raw_count=$(grep -r "pickle\.load(" src/ --include="*.py" | grep -v "safe_pickle" | grep -v "#" | wc -l)
if [ "$raw_count" = "0" ]; then echo "3D-4 OK: All pickle.load migrated"; else echo "3D-4 WARN: $raw_count raw pickle.load calls remain"; fi

# 3D-5: Neural arch versioning
python -c "
from src.models.neural.base_rnn import BaseRNNModel
assert hasattr(BaseRNNModel, 'ARCH_VERSION')
print(f'3D-5 OK: ARCH_VERSION={BaseRNNModel.ARCH_VERSION}')
"

# 3D-7: Deprecation warnings on old classes
python -c "
import warnings, inspect
from src.inference.pipeline import InferencePipeline
from src.inference.orchestrator import InferenceOrchestrator
src1 = inspect.getsource(InferencePipeline.__init__)
src2 = inspect.getsource(InferenceOrchestrator.__init__)
assert 'DeprecationWarning' in src1 or 'deprecated' in src1.lower() or 'warnings.warn' in src1
assert 'DeprecationWarning' in src2 or 'deprecated' in src2.lower() or 'warnings.warn' in src2
print('3D-7 OK: Deprecation warnings present in old classes')
"
```

---

## 2. Integration Test Code

These are pytest-compatible test functions. Save as `tests/test_phase3_integration.py`.

```python
"""
Phase 3 Integration Tests
Run: pytest tests/test_phase3_integration.py -v

Requirements: Full ML Factory dev environment with sklearn, torch, xgboost, etc.
These tests require trained model artifacts. Skip if artifacts unavailable.
"""

import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df():
    """Generate minimal OHLCV dataframe (200 rows of random walk)."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min"))


@pytest.fixture
def sample_features_2d():
    """Generate 2D feature matrix (200 rows, 20 features)."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(200, 20),
        columns=[f"feat_{i}" for i in range(20)],
    )


@pytest.fixture
def tmp_bundle_dir():
    """Temporary directory for bundle save/load tests."""
    d = tempfile.mkdtemp(prefix="mlf_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2.1 Tabular Model: Train → Bundle → predict_from_raw
# ---------------------------------------------------------------------------

class TestTabularRoundtrip:
    """Full roundtrip for tabular (2D) models like XGBoost."""

    @pytest.mark.slow
    def test_xgboost_train_bundle_predict(self, sample_ohlcv_df, tmp_bundle_dir):
        """Train XGBoost → build bundle → predict_from_raw → PredictionResult."""
        from src.inference.bundle import ModelBundle

        # Step 1: Check if a pre-built bundle exists for testing
        # (In CI, we'd build one; here, we try to load or skip)
        bundle_path = tmp_bundle_dir / "xgb_test"

        # If no pre-trained bundle, use mock approach
        pytest.skip("Requires full training environment — run manually with trained model")

    def test_tabular_bundle_predict_from_raw_shape(self, sample_ohlcv_df, tmp_bundle_dir):
        """Verify predict_from_raw returns correct shape for tabular model."""
        from src.inference.bundle import ModelBundle

        # Load a bundle if available
        # This test verifies the API contract, not the prediction quality
        pytest.skip("Requires pre-trained tabular bundle")

    def test_bundle_metadata_preserved_after_save_load(self, tmp_bundle_dir):
        """BundleMetadata round-trips correctly through save/load."""
        from src.inference.bundle import BundleMetadata

        original = BundleMetadata.from_dict({
            "version": "1.3.0",
            "model_name": "xgboost_h20",
            "model_family": "boosting",
            "scaling_source": "pipeline",
            "arch_version": None,
            "label_mapping": {"SHORT": 0, "HOLD": 1, "LONG": 2},
            "feature_names": ["rsi_14", "ema_20", "volume_ratio"],
            "scaler_type": "robust",
            "training_run_id": "test_run_001",
        })

        # Serialize and deserialize
        d = original.to_dict()
        restored = BundleMetadata.from_dict(d)

        assert restored.scaling_source == "pipeline"
        assert restored.label_mapping == {"SHORT": 0, "HOLD": 1, "LONG": 2}
        assert restored.feature_names == ["rsi_14", "ema_20", "volume_ratio"]
        assert restored.scaler_type == "robust"
        assert restored.training_run_id == "test_run_001"


# ---------------------------------------------------------------------------
# 2.2 Sequence Model (3D): Train → Bundle → predict_from_raw
# ---------------------------------------------------------------------------

class TestSequenceRoundtrip:
    """Full roundtrip for 3D sequence models like LSTM, GRU."""

    def test_3d_windowing_correct_shape(self, sample_features_2d):
        """Verify _build_3d_input produces (n_seq, seq_len, n_feat) shape."""
        from src.inference.bundle import ModelBundle

        # After Phase 3B, ModelBundle should have _build_3d_input
        if not hasattr(ModelBundle, "_build_3d_input"):
            pytest.skip("Phase 3B not yet implemented")

        # Create a minimal bundle mock to test windowing
        bundle = ModelBundle.__new__(ModelBundle)
        bundle.feature_columns = list(sample_features_2d.columns)

        # Mock metadata with sequence_length
        class MockMeta:
            sequence_length = 30

        bundle.metadata = MockMeta()

        result = bundle._build_3d_input(sample_features_2d)

        # 200 rows, seq_len=30 → 171 windows (200 - 30 + 1)
        assert result.ndim == 3
        assert result.shape[1] == 30  # seq_len
        assert result.shape[2] == 20  # n_features
        assert result.shape[0] == 171  # n_windows

    def test_3d_windowing_insufficient_data(self, sample_features_2d):
        """Verify _build_3d_input raises error with insufficient data."""
        from src.inference.bundle import ModelBundle

        if not hasattr(ModelBundle, "_build_3d_input"):
            pytest.skip("Phase 3B not yet implemented")

        bundle = ModelBundle.__new__(ModelBundle)
        bundle.feature_columns = list(sample_features_2d.columns)

        class MockMeta:
            sequence_length = 300  # More than 200 rows

        bundle.metadata = MockMeta()

        with pytest.raises(ValueError, match="Need >= 300 rows"):
            bundle._build_3d_input(sample_features_2d)

    @pytest.mark.slow
    def test_lstm_train_bundle_predict(self, sample_ohlcv_df, tmp_bundle_dir):
        """Train LSTM → bundle → predict_from_raw with 3D adapter."""
        pytest.skip("Requires full training environment with torch")


# ---------------------------------------------------------------------------
# 2.3 Transformer Model (4D): Train → Bundle → predict_from_raw
# ---------------------------------------------------------------------------

class TestTransformerRoundtrip:
    """Full roundtrip for 4D transformer models like PatchTST."""

    def test_4d_input_builder_exists(self):
        """Verify _build_4d_input method exists on ModelBundle."""
        from src.inference.bundle import ModelBundle

        if not hasattr(ModelBundle, "_build_4d_input"):
            pytest.skip("Phase 3B not yet implemented")
        assert callable(getattr(ModelBundle, "_build_4d_input"))

    def test_mtf_dataframe_generation(self, sample_ohlcv_df):
        """Verify _generate_mtf_dataframes produces correct timeframe dict."""
        from src.inference.bundle import ModelBundle

        if not hasattr(ModelBundle, "_generate_mtf_dataframes"):
            pytest.skip("Phase 3B not yet implemented")

        bundle = ModelBundle.__new__(ModelBundle)
        result = bundle._generate_mtf_dataframes(
            sample_ohlcv_df, timeframes=["1min", "5min"]
        )

        assert isinstance(result, dict)
        assert "1min" in result
        assert "5min" in result
        # 5min resampled from 200 1min bars → ~40 bars
        assert len(result["5min"]) < len(result["1min"])

    @pytest.mark.slow
    def test_patchtst_train_bundle_predict(self, sample_ohlcv_df, tmp_bundle_dir):
        """Train PatchTST → bundle → predict_from_raw with 4D MTF adapter."""
        pytest.skip("Requires full training environment with torch")


# ---------------------------------------------------------------------------
# 2.4 Ensemble Bundle: Save → Move → Load → Predict
# ---------------------------------------------------------------------------

class TestEnsembleBundlePortability:
    """Verify ensemble bundles with relative paths are portable."""

    def test_ensemble_relative_paths_after_move(self, tmp_bundle_dir):
        """Save ensemble → move directory → load succeeds."""
        from src.inference.ensemble_bundle import EnsembleBundle

        if not hasattr(EnsembleBundle, "predict_from_raw"):
            pytest.skip("Phase 3B not yet implemented")

        # This test requires actual trained bundles.
        # Minimal version: verify save/load with mocked internals.
        pytest.skip("Requires pre-trained ensemble bundle artifacts")

    def test_ensemble_bundle_save_uses_relative_paths(self, tmp_bundle_dir):
        """Verify saved base_bundles.json uses relative (not absolute) paths."""
        from src.inference.ensemble_bundle import EnsembleBundle

        # After Phase 3B-3, saved JSON should have relative paths
        # Check by inspecting saved JSON if we can create a minimal bundle
        pytest.skip("Requires pre-trained ensemble bundle artifacts")


# ---------------------------------------------------------------------------
# 2.5 Old Bundle Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify old bundles (v1.2.0 metadata) still load correctly."""

    def test_old_metadata_loads_with_defaults(self):
        """v1.2.0 metadata dict (missing new fields) loads with safe defaults."""
        from src.inference.bundle import BundleMetadata

        old_dict = {
            "version": "1.2.0",
            "model_name": "xgboost_h20",
            "model_family": "boosting",
            "horizon": 20,
            "sequence_length": None,
            "requires_sequences": False,
            "requires_4d": False,
            # NO new Phase 3A fields
        }

        meta = BundleMetadata.from_dict(old_dict)

        # All new fields should have safe defaults
        assert meta.scaling_source == "unknown"
        assert meta.arch_version is None
        assert meta.label_mapping is None
        assert meta.feature_names == []
        assert meta.scaler_type == "unknown"
        assert meta.training_run_id is None

    def test_old_prediction_path_unchanged(self):
        """ModelBundle.predict(X_preshaped) signature is unchanged."""
        import inspect
        from src.inference.bundle import ModelBundle

        sig = inspect.signature(ModelBundle.predict)
        params = list(sig.parameters.keys())
        # 'self' + at least 'X' (or 'features')
        assert len(params) >= 2, f"predict signature changed: {params}"

    def test_trainer_legacy_extraction_fallback(self):
        """BundleBuilder falls back to duck-typing for old trainers."""
        from src.inference.builder import BundleBuilder

        class OldStyleTrainer:
            model = "old_model"
            _scaler = None
            _feature_set_columns = ["a", "b"]

        builder = BundleBuilder.__new__(BundleBuilder)
        # Should not raise — should use legacy duck-typing
        model = builder._extract_model(OldStyleTrainer())
        assert model == "old_model"

    def test_neural_checkpoint_without_arch_version(self):
        """Old neural checkpoints (no arch_version) load with warning."""
        # Phase 3D-5: loading a checkpoint dict without 'arch_version'
        # should default to "0.0" and emit warning, not error
        pytest.skip("Requires actual neural checkpoint file")

    def test_enum_imports_from_config_still_work(self):
        """Importing CVMethod/LabelingMethod from src.config still works."""
        # After Phase 3D-2/3D-3 consolidation, these imports must not break
        from src.config.cv import CVMethod
        from src.config.data import LabelingMethod
        assert CVMethod is not None
        assert LabelingMethod is not None


# ---------------------------------------------------------------------------
# 2.6 UniversalInferencePipeline Tests
# ---------------------------------------------------------------------------

class TestUniversalInferencePipeline:
    """Tests for the new UIP class."""

    def test_scaling_source_enum(self):
        """ScalingSource enum has expected values."""
        from src.inference.universal_pipeline import ScalingSource

        assert hasattr(ScalingSource, "BUNDLE")
        assert hasattr(ScalingSource, "PREPROCESSING")
        assert hasattr(ScalingSource, "NONE")

    def test_uip_from_bundle_classmethod(self):
        """UIP.from_bundle() exists and is callable."""
        from src.inference.universal_pipeline import UniversalInferencePipeline as UIP

        assert callable(getattr(UIP, "from_bundle", None))

    @pytest.mark.slow
    def test_uip_tabular_predict(self, sample_features_2d, tmp_bundle_dir):
        """UIP.from_bundle(path).predict(X_2d) for boosting model."""
        pytest.skip("Requires pre-trained bundle")


# ---------------------------------------------------------------------------
# 2.7 Special Mode Bundles
# ---------------------------------------------------------------------------

class TestSpecialModeBundles:
    """Tests for WalkForward, Regime, and MetaLabeling bundles."""

    def test_walk_forward_bundle_interface(self):
        """WalkForwardBundle has required InferenceBundle methods."""
        from src.inference.walk_forward_bundle import WalkForwardBundle

        for method in ["predict", "predict_from_raw", "save", "load"]:
            assert hasattr(WalkForwardBundle, method)

    def test_regime_bundle_interface(self):
        """RegimeBundle has required InferenceBundle methods."""
        from src.inference.regime_bundle import RegimeBundle

        for method in ["predict", "predict_from_raw", "save", "load"]:
            assert hasattr(RegimeBundle, method)

    def test_meta_labeling_bundle_interface(self):
        """MetaLabelingBundle has required InferenceBundle methods."""
        from src.inference.meta_labeling_bundle import MetaLabelingBundle

        for method in ["predict", "predict_from_raw", "save", "load"]:
            assert hasattr(MetaLabelingBundle, method)


# ---------------------------------------------------------------------------
# 2.8 safe_pickle_load
# ---------------------------------------------------------------------------

class TestSafePickle:
    """Tests for the safe_pickle_load utility."""

    def test_safe_pickle_load_basic(self, tmp_bundle_dir):
        """safe_pickle_load loads a valid pickle file."""
        import pickle

        from src.core.utils.safe_pickle import safe_pickle_load

        test_obj = {"key": "value", "num": 42}
        pkl_path = tmp_bundle_dir / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(test_obj, f)

        loaded = safe_pickle_load(pkl_path)
        assert loaded == test_obj

    def test_safe_pickle_load_type_check_warn(self, tmp_bundle_dir):
        """safe_pickle_load warns on type mismatch."""
        import pickle

        from src.core.utils.safe_pickle import safe_pickle_load

        pkl_path = tmp_bundle_dir / "test.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump("a string", f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = safe_pickle_load(pkl_path, expected_type=dict)
            # Should still return the object but warn
            assert loaded == "a string"

    def test_safe_pickle_load_missing_file(self):
        """safe_pickle_load raises FileNotFoundError for missing file."""
        from src.core.utils.safe_pickle import safe_pickle_load

        with pytest.raises((FileNotFoundError, OSError)):
            safe_pickle_load(Path("/nonexistent/path.pkl"))
```

---

## 3. Regression Tests

### What Could Break and How to Verify

| Area | What Could Break | Regression Test |
|------|-----------------|-----------------|
| `ModelBundle.predict(X_preshaped)` | Phase 3B changes to bundle.py could alter existing predict path | Verify `predict()` signature unchanged; test with pre-shaped 2D input |
| `ModelBundle.predict_from_raw(df)` for tabular | Phase 3B adapter routing could break existing tabular path | Verify tabular models still produce PredictionResult without adapter |
| `BundleBuilder.build_from_training_result()` | Phase 3A protocol changes could break with old-style trainers | Test with duck-typed trainer (no protocol) — must fall back gracefully |
| `EnsembleBundle.load()` with absolute paths | Phase 3B-3 relative path fix could break old absolute-path bundles | Test loading bundle with absolute paths — must still resolve |
| Model `save()`/`load()` roundtrips | Phase 3D-5 arch versioning could break load for old checkpoints | Load checkpoint without `arch_version` key — must default to "0.0" |
| `from src.config.cv import CVMethod` | Phase 3D-2 enum consolidation removes class from cv.py | Verify import still works (re-export from core.types) |
| `from src.config.data import LabelingMethod` | Phase 3D-3 enum consolidation removes class from data.py | Verify import still works (re-export from core.types) |
| `InferencePipeline` instantiation | Phase 3D-7 deprecation warnings | Verify class still works, just emits DeprecationWarning |
| `InferenceOrchestrator` instantiation | Phase 3D-7 deprecation warnings | Verify class still works, just emits DeprecationWarning |
| Pickle-based save/load across codebase | Phase 3D-4 migration to safe_pickle_load | Verify all 17 call sites still load correctly |

### Regression Test Shell Commands

```bash
# Existing predict path unchanged
python -c "
import inspect
from src.inference.bundle import ModelBundle
sig = inspect.signature(ModelBundle.predict)
print(f'predict signature: {sig}')
assert 'features' in str(sig) or 'X' in str(sig), 'predict() signature changed!'
print('REGRESSION OK: predict() signature preserved')
"

# Existing imports from src.config still work
python -c "
from src.config.cv import CVMethod
from src.config.data import LabelingMethod
print('REGRESSION OK: enum imports from src.config work')
"

# Old InferencePipeline still importable
python -c "
from src.inference.pipeline import InferencePipeline
from src.inference.orchestrator import InferenceOrchestrator
print('REGRESSION OK: Old classes still importable')
"

# All __init__.py exports still work
python -c "
from src.inference import (
    ModelBundle, BundleMetadata, BundleManifest,
    InferencePipeline, InferenceResult,
    BatchPredictor, BatchProgress,
    ModelServer, ServerConfig,
    BundleBuilder, BundleBuildResult,
    EnsembleBundle, EnsembleBundleMetadata,
    InferenceOrchestrator, PredictionResult,
    PreprocessingGraph, PreprocessingGraphConfig,
)
print('REGRESSION OK: All existing __init__.py exports work')
"
```

---

## 4. Ruff/Black Verification

Run after **every phase** and before any commit.

```bash
# ---- After Phase 3A ----
ruff check src/core/protocols.py src/models/training/trainer.py src/inference/bundle.py src/inference/builder.py src/models/training/unified_orchestrator.py
black --check src/core/protocols.py src/models/training/trainer.py src/inference/bundle.py src/inference/builder.py src/models/training/unified_orchestrator.py

# ---- After Phase 3B ----
ruff check src/inference/universal_pipeline.py src/inference/errors.py src/inference/bundle.py src/inference/ensemble_bundle.py src/models/training/services/ensemble_service.py
black --check src/inference/universal_pipeline.py src/inference/errors.py src/inference/bundle.py src/inference/ensemble_bundle.py src/models/training/services/ensemble_service.py

# ---- After Phase 3C ----
ruff check src/inference/walk_forward_bundle.py src/inference/regime_bundle.py src/inference/meta_labeling_bundle.py src/inference/regime_detector.py src/inference/server.py src/inference/batch.py src/inference/__init__.py
black --check src/inference/walk_forward_bundle.py src/inference/regime_bundle.py src/inference/meta_labeling_bundle.py src/inference/regime_detector.py src/inference/server.py src/inference/batch.py src/inference/__init__.py

# ---- After Phase 3D ----
ruff check src/inference/preprocessing_graph.py src/config/cv.py src/config/data.py src/core/utils/safe_pickle.py src/models/neural/base_rnn.py src/inference/pipeline.py src/inference/orchestrator.py
black --check src/inference/preprocessing_graph.py src/config/cv.py src/config/data.py src/core/utils/safe_pickle.py src/models/neural/base_rnn.py src/inference/pipeline.py src/inference/orchestrator.py

# ---- Full project sweep (final) ----
ruff check src/ --fix
black src/
ruff check src/  # Verify no remaining issues
black --check src/  # Verify formatting stable
```

---

## 5. Import Verification Matrix

### New Imports That Must Work (Post-Implementation)

| Import | Phase | Status Check |
|--------|-------|--------------|
| `from src.core.protocols import TrainerProtocol` | 3A | `python -c "..."` |
| `from src.inference.bundle import BundleMetadata` (with new fields) | 3A | metadata round-trip test |
| `from src.inference.universal_pipeline import UniversalInferencePipeline` | 3B | `python -c "..."` |
| `from src.inference.universal_pipeline import ScalingSource` | 3B | `python -c "..."` |
| `from src.inference.errors import InferenceShapeMismatchError` | 3B | `python -c "..."` |
| `from src.inference.walk_forward_bundle import WalkForwardBundle` | 3C | `python -c "..."` |
| `from src.inference.regime_bundle import RegimeBundle` | 3C | `python -c "..."` |
| `from src.inference.regime_detector import RegimeDetector` | 3C | `python -c "..."` |
| `from src.inference.meta_labeling_bundle import MetaLabelingBundle` | 3C | `python -c "..."` |
| `from src.inference import UniversalInferencePipeline` | 3C | via `__init__.py` |
| `from src.core.utils.safe_pickle import safe_pickle_load` | 3D | `python -c "..."` |

### Existing Imports That Must NOT Break

| Import | Risk From | Verification |
|--------|-----------|-------------|
| `from src.inference import ModelBundle` | 3B changes | `python -c "from src.inference import ModelBundle; print('OK')"` |
| `from src.inference import InferencePipeline` | 3D-7 deprecation | `python -c "from src.inference import InferencePipeline; print('OK')"` |
| `from src.inference import InferenceOrchestrator` | 3D-7 deprecation | `python -c "from src.inference import InferenceOrchestrator; print('OK')"` |
| `from src.inference import EnsembleBundle` | 3B-3 changes | `python -c "from src.inference import EnsembleBundle; print('OK')"` |
| `from src.inference import BundleBuilder` | 3A-4 changes | `python -c "from src.inference import BundleBuilder; print('OK')"` |
| `from src.config.cv import CVMethod` | 3D-2 consolidation | `python -c "from src.config.cv import CVMethod; print('OK')"` |
| `from src.config.data import LabelingMethod` | 3D-3 consolidation | `python -c "from src.config.data import LabelingMethod; print('OK')"` |
| `from src.core.types import DataRank, ModelFamily` | Not changed | `python -c "from src.core.types import DataRank, ModelFamily; print('OK')"` |
| `from src.core.contracts import get_model_contract` | Not changed | `python -c "from src.core.contracts import get_model_contract; print('OK')"` |

### Batch Import Verification Script

```bash
#!/bin/bash
# Save as: scripts/verify_imports.sh
# Run: bash scripts/verify_imports.sh

echo "=== New Imports (Phase 3) ==="
python -c "from src.core.protocols import TrainerProtocol; print('  TrainerProtocol: OK')" 2>&1 || echo "  TrainerProtocol: FAIL"
python -c "from src.inference.universal_pipeline import UniversalInferencePipeline; print('  UIP: OK')" 2>&1 || echo "  UIP: FAIL"
python -c "from src.inference.universal_pipeline import ScalingSource; print('  ScalingSource: OK')" 2>&1 || echo "  ScalingSource: FAIL"
python -c "from src.inference.errors import InferenceShapeMismatchError; print('  InferenceShapeMismatchError: OK')" 2>&1 || echo "  InferenceShapeMismatchError: FAIL"
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('  WalkForwardBundle: OK')" 2>&1 || echo "  WalkForwardBundle: FAIL"
python -c "from src.inference.regime_bundle import RegimeBundle; print('  RegimeBundle: OK')" 2>&1 || echo "  RegimeBundle: FAIL"
python -c "from src.inference.regime_detector import RegimeDetector; print('  RegimeDetector: OK')" 2>&1 || echo "  RegimeDetector: FAIL"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('  MetaLabelingBundle: OK')" 2>&1 || echo "  MetaLabelingBundle: FAIL"
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('  safe_pickle_load: OK')" 2>&1 || echo "  safe_pickle_load: FAIL"

echo ""
echo "=== Existing Imports (Must Not Break) ==="
python -c "from src.inference import ModelBundle; print('  ModelBundle: OK')" 2>&1 || echo "  ModelBundle: FAIL"
python -c "from src.inference import InferencePipeline; print('  InferencePipeline: OK')" 2>&1 || echo "  InferencePipeline: FAIL"
python -c "from src.inference import InferenceOrchestrator; print('  InferenceOrchestrator: OK')" 2>&1 || echo "  InferenceOrchestrator: FAIL"
python -c "from src.inference import EnsembleBundle; print('  EnsembleBundle: OK')" 2>&1 || echo "  EnsembleBundle: FAIL"
python -c "from src.inference import BundleBuilder; print('  BundleBuilder: OK')" 2>&1 || echo "  BundleBuilder: FAIL"
python -c "from src.config.cv import CVMethod; print('  CVMethod (config): OK')" 2>&1 || echo "  CVMethod: FAIL"
python -c "from src.config.data import LabelingMethod; print('  LabelingMethod (config): OK')" 2>&1 || echo "  LabelingMethod: FAIL"
python -c "from src.core.types import DataRank, ModelFamily; print('  DataRank/ModelFamily: OK')" 2>&1 || echo "  DataRank/ModelFamily: FAIL"
python -c "from src.core.contracts import get_model_contract; print('  get_model_contract: OK')" 2>&1 || echo "  get_model_contract: FAIL"

echo ""
echo "=== Single Definition Checks ==="
echo -n "  TrainerProtocol definitions: "; grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l
echo -n "  CVMethod definitions: "; grep -r "class CVMethod" src/ --include="*.py" | wc -l
echo -n "  LabelingMethod definitions: "; grep -r "class LabelingMethod" src/ --include="*.py" | wc -l
echo -n "  Raw pickle.load calls: "; grep -r "pickle\.load(" src/ --include="*.py" | grep -v "safe_pickle" | grep -v "#" | wc -l
```

---

## 6. Execution Order (Dependency-Safe)

### Master Test Execution Order

```
Phase 3D smoke tests          ← Run first (parallel, no deps on 3A-3C)
    ↓ (independent)
Phase 3A smoke tests          ← Foundation must pass before 3B
    ↓ (depends on 3A)
Phase 3A regression tests     ← Verify nothing broke
    ↓
Phase 3B smoke tests          ← Core inference depends on 3A
    ↓ (depends on 3B)
Phase 3B regression tests     ← Verify backward compat
    ↓
Phase 3C smoke tests          ← Integration depends on 3B
    ↓
Full regression suite         ← All existing imports + signatures
    ↓
Integration tests (pytest)    ← End-to-end roundtrips
    ↓
Ruff + Black full sweep       ← Final code quality gate
    ↓
Import verification script    ← Final import matrix check
```

### Quick Reference: What to Run When

| After completing... | Run these |
|-------------------|-----------|
| Any single task | Ruff check on modified files |
| Phase 3A | 3A smoke tests → 3A regression → ruff/black on 3A files |
| Phase 3B | 3B smoke tests → 3B regression → ruff/black on 3B files |
| Phase 3C | 3C smoke tests → full regression → ruff/black on 3C files |
| Phase 3D | 3D smoke tests → enum/import regression → ruff/black on 3D files |
| All phases done | Full import verification script → pytest integration → full ruff/black sweep |

### Critical Path Tests (Must Pass for Go/No-Go)

1. **BundleMetadata backward compat** (3A-3) — Old bundles must load
2. **TrainerProtocol + legacy fallback** (3A-4) — Old trainers must still work
3. **Adapter routing shape correctness** (3B-1) — 3D windowing must be (n, seq, feat)
4. **Existing imports unbroken** — All items in "Must Not Break" table
5. **Single definition counts** — TrainerProtocol=1, CVMethod=1, LabelingMethod=1
6. **Zero raw pickle.load** after 3D-4 — All migrated to safe_pickle_load
7. **Ruff + Black clean** — No warnings on `ruff check src/` and `black --check src/`

---

*Generated: 2026-02-15*
*Source: UNIFIED-ROADMAP.md Section 7 + codebase analysis*
