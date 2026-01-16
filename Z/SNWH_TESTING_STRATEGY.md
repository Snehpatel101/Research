# SNwH Testing Strategy: Comprehensive TDD Plan

## Document Purpose

This document defines a complete testing strategy for the SNwH (Unified Multi-Timeframe Model Factory) implementation. It covers unit tests, integration tests, regression tests, property-based tests, fixtures, and CI/CD integration to ensure all contracts are enforced and heterogeneous ensembles work by default.

---

## 1. Test Directory Structure

```
tests/
  snwh/
    __init__.py
    conftest.py                         # Shared SNwH fixtures

    # Phase 0 - Contracts
    contracts/
      __init__.py
      test_data_contract.py             # DataContract schema, hash
      test_model_contract.py            # ModelContract input_rank, feature_mode
      test_artifact_manifest.py         # ArtifactManifest hash verification

    # Phase 1 - Configuration
    config/
      __init__.py
      test_per_model_config.py          # PerModelConfig validation
      test_ensemble_plan.py             # EnsemblePlan expansion, resolution
      test_trainer_config_extensions.py # TrainerConfig timeframe, mtf_mode

    # Phase 2 - Adapters
    adapters/
      __init__.py
      test_tabular_adapter.py           # 2D output validation
      test_sequence_adapter.py          # 3D output validation
      test_multi_stream_adapter.py      # 4D output validation
      test_adapter_registry.py          # Routing by input_rank
      test_adapter_validation.py        # Shape, dtype, feature order

    # Phase 3 - Timeframe Coordination
    timeframe/
      __init__.py
      test_timeframe_coordinator.py     # Alignment, shift(1) enforcement
      test_multi_stream_alignment.py    # Timestamp alignment for 4D
      test_temporal_rules.py            # Anchor TF, resampling policy

    # Phase 4 - OOF Integrity
    oof/
      __init__.py
      test_oof_alignment_validator.py   # Coverage, shape alignment
      test_heterogeneous_stacking.py    # Mixed 2D+3D OOF handling
      test_oof_preflight.py             # Pre-flight validation
      test_oof_no_leakage.py            # Purge/embargo enforcement

    # Phase 5 - Feature Strategy
    features/
      __init__.py
      test_feature_strategy_manager.py  # Strategy resolution
      test_feature_optimizer.py         # Optuna pruning
      test_model_feature_strategies.py  # 23-model strategy definitions

    # Integration Tests
    integration/
      __init__.py
      test_heterogeneous_stacking_e2e.py  # Full stacking pipeline
      test_mixed_family_training.py       # Tabular + Sequence + Transformer
      test_per_model_timeframe_e2e.py     # Different TFs same source

    # Regression Tests
    regression/
      __init__.py
      test_oof_leakage_regression.py    # OOF no-leakage invariants
      test_shape_alignment_regression.py # Shape consistency
      test_config_backward_compat.py    # Config migration

    # Property-Based Tests
    property/
      __init__.py
      test_adapter_properties.py        # Adapter invariants
      test_oof_coverage_properties.py   # Coverage invariants
      test_timeframe_alignment_props.py # Alignment properties
```

---

## 2. Unit Tests

### 2.1 Phase 0 - Contracts

#### `tests/snwh/contracts/test_data_contract.py`

```python
"""
Tests for DataContract - canonical data schema validation.

Tests:
- Schema validation (required columns, types)
- Hash generation determinism
- Hash change detection on schema drift
- Label column naming conventions
- Timeframe metadata consistency
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestDataContractSchema:
    """Tests for DataContract schema validation."""

    def test_valid_schema_passes(self, sample_canonical_ohlcv):
        """DataContract accepts valid OHLCV with all required columns."""
        pass

    def test_missing_timestamp_raises(self, sample_ohlcv_no_timestamp):
        """DataContract rejects data without timestamp column."""
        pass

    def test_missing_ohlcv_raises(self, sample_ohlcv_missing_close):
        """DataContract rejects data missing OHLCV columns."""
        pass

    def test_invalid_dtypes_raises(self, sample_ohlcv_string_prices):
        """DataContract rejects non-numeric OHLCV columns."""
        pass

    def test_label_column_validation(self, sample_labeled_data):
        """DataContract validates label_h{horizon} naming convention."""
        pass

    def test_sample_weight_column_validation(self, sample_labeled_data):
        """DataContract validates sample_weight_h{horizon} pairing."""
        pass


class TestDataContractHash:
    """Tests for DataContract hash generation."""

    def test_hash_deterministic(self, sample_canonical_ohlcv):
        """Same data produces same hash."""
        pass

    def test_hash_changes_on_column_add(self, sample_canonical_ohlcv):
        """Hash changes when columns added."""
        pass

    def test_hash_changes_on_dtype_change(self, sample_canonical_ohlcv):
        """Hash changes when column dtype changes."""
        pass

    def test_hash_stable_on_data_change(self, sample_canonical_ohlcv):
        """Hash stable when only values change (schema same)."""
        pass


class TestDataContractTimeframes:
    """Tests for timeframe metadata in DataContract."""

    def test_timeframe_in_metadata(self, sample_mtf_data):
        """DataContract stores timeframe in metadata."""
        pass

    def test_all_9_timeframes_supported(self):
        """All 9 canonical timeframes are valid."""
        pass

    def test_invalid_timeframe_raises(self):
        """Invalid timeframe string raises ValueError."""
        pass
```

**Fixtures Needed:**
- `sample_canonical_ohlcv`: Valid 1-min OHLCV DataFrame
- `sample_ohlcv_no_timestamp`: OHLCV missing datetime column
- `sample_ohlcv_missing_close`: OHLCV missing close column
- `sample_ohlcv_string_prices`: OHLCV with string price columns
- `sample_labeled_data`: OHLCV with label_h{5,20} columns
- `sample_mtf_data`: Multi-timeframe data with MTF columns

---

#### `tests/snwh/contracts/test_model_contract.py`

```python
"""
Tests for ModelContract - model input requirements validation.

Tests:
- input_rank validation (2, 3, 4)
- feature_mode validation (engineered, raw, hybrid)
- mtf_mode validation (none, indicators, multi_stream)
- Sequence length requirements for 3D/4D
- Contract enforcement at model registration
"""
import pytest


class TestModelContractInputRank:
    """Tests for input_rank validation."""

    def test_rank_2_valid_for_tabular(self):
        """input_rank=2 valid for tabular models."""
        pass

    def test_rank_3_valid_for_sequence(self):
        """input_rank=3 valid for sequence models."""
        pass

    def test_rank_4_valid_for_multi_stream(self):
        """input_rank=4 valid for multi-stream models."""
        pass

    def test_rank_1_invalid(self):
        """input_rank=1 raises ValueError."""
        pass

    def test_rank_5_invalid(self):
        """input_rank=5 raises ValueError."""
        pass


class TestModelContractFeatureMode:
    """Tests for feature_mode validation."""

    def test_engineered_mode_valid(self):
        """feature_mode='engineered' for models using indicators."""
        pass

    def test_raw_mode_valid(self):
        """feature_mode='raw' for transformers using OHLCV only."""
        pass

    def test_hybrid_mode_valid(self):
        """feature_mode='hybrid' for mixed feature models."""
        pass

    def test_invalid_mode_raises(self):
        """Invalid feature_mode raises ValueError."""
        pass


class TestModelContractMTFMode:
    """Tests for mtf_mode validation."""

    def test_none_mode_valid(self):
        """mtf_mode='none' for single-timeframe models."""
        pass

    def test_indicators_mode_valid(self):
        """mtf_mode='indicators' adds MTF indicator features."""
        pass

    def test_multi_stream_mode_valid(self):
        """mtf_mode='multi_stream' for 4D multi-TF ingestion."""
        pass

    def test_multi_stream_requires_rank_4(self):
        """mtf_mode='multi_stream' requires input_rank=4."""
        pass


class TestModelContractSequenceParams:
    """Tests for sequence-related parameters."""

    def test_sequence_length_required_for_rank_3(self):
        """sequence_length required when input_rank >= 3."""
        pass

    def test_patch_length_optional_for_patchtst(self):
        """patch_length optional but validated for PatchTST."""
        pass

    def test_sequence_length_positive(self):
        """sequence_length must be positive."""
        pass


class TestModelContractRegistry:
    """Tests for contract enforcement at registration."""

    @pytest.mark.parametrize("model_name", [
        "xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm"
    ])
    def test_all_tabular_models_rank_2(self, model_name):
        """All tabular models have input_rank=2."""
        pass

    @pytest.mark.parametrize("model_name", [
        "lstm", "gru", "tcn", "inceptiontime", "resnet1d"
    ])
    def test_all_sequence_models_rank_3(self, model_name):
        """All sequence models have input_rank=3."""
        pass

    @pytest.mark.parametrize("model_name", [
        "patchtst", "itransformer"
    ])
    def test_multi_stream_transformers_rank_4(self, model_name):
        """Multi-stream transformers have input_rank=4 when using multi_stream."""
        pass
```

---

#### `tests/snwh/contracts/test_artifact_manifest.py`

```python
"""
Tests for ArtifactManifest - safe artifact loading and verification.

Tests:
- Hash verification on load
- Signature validation
- Schema version compatibility
- Metadata completeness
"""
import pytest
from pathlib import Path


class TestArtifactManifestCreation:
    """Tests for ArtifactManifest creation."""

    def test_create_manifest_includes_hash(self, tmp_model_dir):
        """Manifest includes content hash."""
        pass

    def test_create_manifest_includes_schema_version(self, tmp_model_dir):
        """Manifest includes schema version."""
        pass

    def test_create_manifest_includes_config_hash(self, tmp_model_dir):
        """Manifest includes training config hash."""
        pass

    def test_create_manifest_includes_code_version(self, tmp_model_dir):
        """Manifest includes git commit or version string."""
        pass


class TestArtifactManifestVerification:
    """Tests for manifest verification on load."""

    def test_verify_passes_for_valid_artifact(self, saved_model_with_manifest):
        """Verification passes for unmodified artifact."""
        pass

    def test_verify_fails_on_tampered_model(self, saved_model_with_manifest):
        """Verification fails if model file modified."""
        pass

    def test_verify_fails_on_missing_manifest(self, saved_model_no_manifest):
        """Verification fails if manifest missing."""
        pass

    def test_verify_fails_on_hash_mismatch(self, saved_model_bad_hash):
        """Verification fails if hash doesn't match content."""
        pass


class TestArtifactManifestSchemaVersion:
    """Tests for schema version handling."""

    def test_current_version_loads(self, saved_model_current_version):
        """Current schema version loads without issues."""
        pass

    def test_old_version_migration(self, saved_model_v1):
        """Old schema version triggers migration."""
        pass

    def test_future_version_fails(self, saved_model_future_version):
        """Future schema version fails with clear error."""
        pass
```

---

### 2.2 Phase 1 - Configuration

#### `tests/snwh/config/test_per_model_config.py`

```python
"""
Tests for PerModelConfig - per-model timeframe and feature configuration.

Tests:
- Per-model timeframe selection
- Per-model feature_mode
- Per-model mtf_mode
- Adapter ID resolution
- Config validation
"""
import pytest


class TestPerModelConfigTimeframe:
    """Tests for per-model timeframe configuration."""

    def test_default_timeframe_5min(self):
        """Default primary timeframe is 5min."""
        pass

    @pytest.mark.parametrize("timeframe", [
        "1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"
    ])
    def test_all_canonical_timeframes_valid(self, timeframe):
        """All 9 canonical timeframes are valid."""
        pass

    def test_invalid_timeframe_raises(self):
        """Invalid timeframe raises ValidationError."""
        pass

    def test_different_models_different_timeframes(self):
        """Different models can have different primary timeframes."""
        pass


class TestPerModelConfigFeatureMode:
    """Tests for per-model feature_mode configuration."""

    def test_boosting_default_engineered(self):
        """Boosting models default to feature_mode='engineered'."""
        pass

    def test_transformer_default_raw(self):
        """Transformer models default to feature_mode='raw'."""
        pass

    def test_override_feature_mode(self):
        """feature_mode can be overridden per model."""
        pass


class TestPerModelConfigMTFMode:
    """Tests for per-model mtf_mode configuration."""

    def test_xgboost_default_indicators(self):
        """XGBoost defaults to mtf_mode='indicators'."""
        pass

    def test_tcn_default_none(self):
        """TCN defaults to mtf_mode='none' (single-TF)."""
        pass

    def test_patchtst_default_multi_stream(self):
        """PatchTST defaults to mtf_mode='multi_stream'."""
        pass


class TestPerModelConfigAdapterID:
    """Tests for adapter_id resolution."""

    def test_tabular_adapter_for_rank_2(self):
        """Tabular adapter selected for input_rank=2."""
        pass

    def test_sequence_adapter_for_rank_3(self):
        """Sequence adapter selected for input_rank=3."""
        pass

    def test_multi_stream_adapter_for_rank_4(self):
        """MultiStream adapter selected for input_rank=4."""
        pass

    def test_explicit_adapter_id_override(self):
        """Explicit adapter_id overrides auto-detection."""
        pass
```

---

#### `tests/snwh/config/test_ensemble_plan.py`

```python
"""
Tests for EnsemblePlan - ensemble configuration expansion and validation.

Tests:
- Base model expansion from names
- Compatibility validation (homogeneous vs heterogeneous)
- Multi-adapter loading plan generation
- Per-model requirement resolution
"""
import pytest


class TestEnsemblePlanExpansion:
    """Tests for base model expansion."""

    def test_expand_base_model_names(self):
        """Expand base_model_names to full ModelConfig list."""
        pass

    def test_expand_with_configs(self):
        """Expand respects base_model_configs overrides."""
        pass

    def test_expand_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        pass


class TestEnsemblePlanCompatibility:
    """Tests for compatibility validation."""

    def test_homogeneous_tabular_valid_all_ensembles(self):
        """All-tabular valid for voting, stacking, blending."""
        pass

    def test_homogeneous_sequence_valid_all_ensembles(self):
        """All-sequence valid for voting, stacking, blending."""
        pass

    def test_heterogeneous_invalid_voting(self):
        """Mixed tabular+sequence invalid for voting."""
        pass

    def test_heterogeneous_invalid_blending(self):
        """Mixed tabular+sequence invalid for blending."""
        pass

    def test_heterogeneous_valid_stacking(self):
        """Mixed tabular+sequence VALID for stacking."""
        pass

    def test_error_suggests_stacking(self):
        """Error message suggests stacking for heterogeneous."""
        pass


class TestEnsemblePlanMultiAdapter:
    """Tests for multi-adapter loading plan."""

    def test_homogeneous_single_adapter(self):
        """Homogeneous ensemble needs single adapter type."""
        pass

    def test_heterogeneous_multiple_adapters(self):
        """Heterogeneous ensemble needs multiple adapter types."""
        pass

    def test_adapter_plan_per_model(self):
        """Plan specifies adapter per base model."""
        pass


class TestEnsemblePlanRequirements:
    """Tests for per-model requirement resolution."""

    def test_resolve_timeframe_requirements(self):
        """Resolve required timeframes across base models."""
        pass

    def test_resolve_feature_requirements(self):
        """Resolve feature sets per base model."""
        pass

    def test_resolve_sequence_lengths(self):
        """Resolve sequence lengths for sequence models."""
        pass
```

---

### 2.3 Phase 2 - Adapters

#### `tests/snwh/adapters/test_tabular_adapter.py`

```python
"""
Tests for TabularAdapter - 2D output for tabular models.

Tests:
- Output shape (n_samples, n_features)
- Feature ordering consistency
- Dtype validation
- Missing feature handling
"""
import pytest
import numpy as np


class TestTabularAdapterShape:
    """Tests for TabularAdapter output shape."""

    def test_output_shape_2d(self, sample_engineered_features):
        """Output is 2D (n_samples, n_features)."""
        pass

    def test_n_samples_matches_input(self, sample_engineered_features):
        """n_samples matches input DataFrame length."""
        pass

    def test_n_features_matches_strategy(self, sample_engineered_features, xgboost_strategy):
        """n_features matches feature strategy baseline."""
        pass


class TestTabularAdapterFeatureOrdering:
    """Tests for feature ordering consistency."""

    def test_feature_order_deterministic(self, sample_engineered_features):
        """Feature order is deterministic across calls."""
        pass

    def test_feature_order_matches_schema(self, sample_engineered_features, feature_schema):
        """Feature order matches stored schema."""
        pass

    def test_feature_names_accessible(self, sample_engineered_features):
        """Feature names accessible from adapter."""
        pass


class TestTabularAdapterDtype:
    """Tests for dtype validation."""

    def test_output_dtype_float32(self, sample_engineered_features):
        """Output dtype is float32."""
        pass

    def test_handles_int_columns(self, sample_features_with_int):
        """Int columns converted to float."""
        pass

    def test_rejects_string_columns(self, sample_features_with_string):
        """String columns raise TypeError."""
        pass


class TestTabularAdapterMissingFeatures:
    """Tests for missing feature handling."""

    def test_missing_feature_raises_by_default(self, sample_features_missing_rsi):
        """Missing feature raises KeyError by default."""
        pass

    def test_missing_feature_fills_with_mode(self, sample_features_missing_rsi):
        """Missing feature filled with pad_value when configured."""
        pass
```

---

#### `tests/snwh/adapters/test_sequence_adapter.py`

```python
"""
Tests for SequenceAdapter - 3D output for sequence models.

Tests:
- Output shape (n_samples, seq_len, n_features)
- Label alignment after windowing
- Symbol isolation (no cross-symbol sequences)
- Sequence coverage validation
"""
import pytest
import numpy as np


class TestSequenceAdapterShape:
    """Tests for SequenceAdapter output shape."""

    def test_output_shape_3d(self, sample_sequence_data):
        """Output is 3D (n_samples, seq_len, n_features)."""
        pass

    def test_seq_len_matches_config(self, sample_sequence_data, seq_len_30):
        """seq_len matches configuration."""
        pass

    def test_n_samples_reduced_by_window(self, sample_sequence_data, seq_len_30):
        """n_samples = original_len - seq_len + 1."""
        pass


class TestSequenceAdapterLabelAlignment:
    """Tests for label alignment after windowing."""

    def test_labels_aligned_to_window_end(self, sample_sequence_data):
        """Labels correspond to last timestep of window."""
        pass

    def test_no_label_shift_error(self, sample_sequence_data):
        """Labels not accidentally shifted."""
        pass

    def test_weights_aligned_to_window_end(self, sample_sequence_data):
        """Weights correspond to last timestep of window."""
        pass


class TestSequenceAdapterSymbolIsolation:
    """Tests for symbol isolation."""

    def test_no_cross_symbol_sequences(self, multi_symbol_data):
        """Sequences don't span symbol boundaries."""
        pass

    def test_symbol_boundary_detection(self, multi_symbol_data):
        """Symbol boundaries correctly detected."""
        pass

    def test_single_symbol_no_boundary_issues(self, single_symbol_data):
        """Single symbol data has no boundary issues."""
        pass


class TestSequenceAdapterCoverage:
    """Tests for sequence coverage."""

    def test_first_seq_len_minus_1_excluded(self, sample_sequence_data):
        """First seq_len-1 rows have no sequence (excluded)."""
        pass

    def test_coverage_calculation_correct(self, sample_sequence_data):
        """Coverage = (n_total - seq_len + 1) / n_total."""
        pass
```

---

#### `tests/snwh/adapters/test_multi_stream_adapter.py`

```python
"""
Tests for MultiStreamAdapter - 4D output for multi-TF models.

Tests:
- Output shape (n_samples, n_streams, seq_len, n_features)
- Stream alignment to anchor TF
- Feature consistency across streams
- Padding for missing timeframes
"""
import pytest
import numpy as np


class TestMultiStreamAdapterShape:
    """Tests for MultiStreamAdapter output shape."""

    def test_output_shape_4d(self, sample_mtf_data):
        """Output is 4D (n_samples, n_streams, seq_len, n_features)."""
        pass

    def test_n_streams_matches_timeframes(self, sample_mtf_data, tf_list_5):
        """n_streams matches number of timeframes."""
        pass

    def test_seq_len_matches_config(self, sample_mtf_data, seq_len_60):
        """seq_len matches configuration."""
        pass


class TestMultiStreamAdapterAlignment:
    """Tests for stream alignment."""

    def test_streams_aligned_to_anchor_tf(self, sample_mtf_data):
        """All streams aligned to anchor (smallest) timeframe."""
        pass

    def test_higher_tf_resampled_correctly(self, sample_mtf_data):
        """Higher TF data resampled to anchor grid."""
        pass

    def test_timestamps_consistent_across_streams(self, sample_mtf_data):
        """Timestamps consistent across all streams."""
        pass


class TestMultiStreamAdapterPadding:
    """Tests for padding missing timeframes."""

    def test_missing_tf_padded(self, sample_mtf_missing_1h):
        """Missing timeframe filled with pad_value."""
        pass

    def test_pad_value_configurable(self, sample_mtf_missing_1h):
        """pad_value can be configured (default 0.0)."""
        pass

    def test_padding_does_not_affect_valid_streams(self, sample_mtf_missing_1h):
        """Valid streams unaffected by padding."""
        pass
```

---

#### `tests/snwh/adapters/test_adapter_registry.py`

```python
"""
Tests for AdapterRegistry - automatic adapter routing.

Tests:
- Routing by input_rank
- Model family to adapter mapping
- All 23 models correctly routed
- Custom adapter registration
"""
import pytest


class TestAdapterRegistryRouting:
    """Tests for adapter routing."""

    def test_route_rank_2_to_tabular(self):
        """input_rank=2 routes to TabularAdapter."""
        pass

    def test_route_rank_3_to_sequence(self):
        """input_rank=3 routes to SequenceAdapter."""
        pass

    def test_route_rank_4_to_multi_stream(self):
        """input_rank=4 routes to MultiStreamAdapter."""
        pass


class TestAdapterRegistryModels:
    """Tests for model-to-adapter mapping."""

    @pytest.mark.parametrize("model_name", [
        "xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm"
    ])
    def test_tabular_models_use_tabular_adapter(self, model_name):
        """All tabular models use TabularAdapter."""
        pass

    @pytest.mark.parametrize("model_name", [
        "lstm", "gru", "tcn", "inceptiontime", "resnet1d"
    ])
    def test_sequence_models_use_sequence_adapter(self, model_name):
        """All sequence models use SequenceAdapter."""
        pass

    @pytest.mark.parametrize("model_name", [
        "patchtst", "itransformer"
    ])
    def test_multi_stream_models_use_multi_stream_adapter(self, model_name):
        """Multi-stream models use MultiStreamAdapter when mtf_mode=multi_stream."""
        pass

    def test_tft_uses_sequence_adapter(self):
        """TFT uses SequenceAdapter (not multi-stream by default)."""
        pass


class TestAdapterRegistryCustom:
    """Tests for custom adapter registration."""

    def test_register_custom_adapter(self):
        """Custom adapter can be registered."""
        pass

    def test_custom_adapter_takes_precedence(self):
        """Explicit adapter_id overrides auto-routing."""
        pass
```

---

### 2.4 Phase 3 - Timeframe Coordination

#### `tests/snwh/timeframe/test_timeframe_coordinator.py`

```python
"""
Tests for TimeframeCoordinator - alignment and shift(1) enforcement.

Tests:
- Timeframe existence validation
- shift(1) enforcement on MTF indicators
- Anchor TF determination
- Multi-stream timestamp alignment
"""
import pytest
import numpy as np
import pandas as pd


class TestTimeframeCoordinatorValidation:
    """Tests for timeframe validation."""

    def test_validates_requested_tf_exists(self, sample_5min_data):
        """Raises if requested timeframe data doesn't exist."""
        pass

    def test_all_9_timeframes_can_be_requested(self, sample_9tf_data):
        """All 9 canonical timeframes can be requested."""
        pass

    def test_invalid_tf_string_raises(self):
        """Invalid timeframe string raises ValueError."""
        pass


class TestTimeframeCoordinatorShift:
    """Tests for shift(1) enforcement."""

    def test_shift_1_applied_to_mtf_indicators(self, sample_mtf_indicators):
        """MTF indicators have shift(1) applied."""
        pass

    def test_first_row_nan_after_shift(self, sample_mtf_indicators):
        """First row of MTF indicators is NaN after shift."""
        pass

    def test_no_shift_for_base_tf_features(self, sample_5min_data):
        """Base TF features don't get additional shift."""
        pass

    def test_shift_applied_before_resampling(self, sample_raw_mtf):
        """shift(1) applied before any resampling."""
        pass


class TestTimeframeCoordinatorAnchor:
    """Tests for anchor TF determination."""

    def test_anchor_is_smallest_tf(self, multi_tf_config):
        """Anchor TF is smallest among requested timeframes."""
        pass

    def test_anchor_explicit_override(self, multi_tf_config):
        """Anchor TF can be explicitly overridden."""
        pass

    def test_higher_tfs_aligned_to_anchor(self, sample_mtf_data):
        """Higher TF data aligned to anchor TF grid."""
        pass


class TestTimeframeCoordinatorAlignment:
    """Tests for multi-stream timestamp alignment."""

    def test_all_streams_same_timestamps(self, sample_mtf_data):
        """All streams have identical timestamps."""
        pass

    def test_alignment_respects_market_hours(self, sample_mtf_with_sessions):
        """Alignment respects market session boundaries."""
        pass
```

---

### 2.5 Phase 4 - OOF Integrity

#### `tests/snwh/oof/test_oof_alignment_validator.py`

```python
"""
Tests for OOFAlignmentValidator - coverage and shape validation.

Tests:
- 100% coverage validation
- Shape alignment across models
- Tabular vs Sequence coverage differences
- Index alignment validation
"""
import pytest
import numpy as np


class TestOOFCoverageValidation:
    """Tests for OOF coverage validation."""

    def test_100_percent_coverage_passes(self, complete_oof_predictions):
        """100% coverage validation passes."""
        pass

    def test_missing_samples_fails(self, incomplete_oof_predictions):
        """Missing samples fails validation."""
        pass

    def test_coverage_threshold_configurable(self, partial_oof_predictions):
        """Coverage threshold can be configured."""
        pass


class TestOOFShapeAlignment:
    """Tests for shape alignment across models."""

    def test_all_models_same_n_samples(self, multi_model_oof):
        """All models have same number of predictions."""
        pass

    def test_tabular_vs_sequence_coverage_diff(self):
        """Validates that tabular=100%, sequence=100%-seq_len."""
        pass

    def test_sequence_coverage_aligned(self, sequence_oof):
        """Sequence OOF coverage aligned correctly."""
        pass


class TestOOFIndexAlignment:
    """Tests for index alignment."""

    def test_oof_indices_match_training_data(self, oof_with_indices, training_data):
        """OOF indices match training data indices."""
        pass

    def test_no_duplicate_indices(self, oof_predictions):
        """No duplicate indices in OOF."""
        pass

    def test_indices_sorted_chronologically(self, oof_predictions):
        """OOF indices are sorted chronologically."""
        pass
```

---

#### `tests/snwh/oof/test_heterogeneous_stacking.py`

```python
"""
Tests for HeterogeneousStackingBuilder - mixed 2D+3D OOF handling.

Tests:
- Combine tabular and sequence OOF predictions
- Index intersection for coverage alignment
- Stacking feature matrix construction
- Meta-learner input preparation
"""
import pytest
import numpy as np
import pandas as pd


class TestHeterogeneousStackingCombination:
    """Tests for combining heterogeneous OOF predictions."""

    def test_combine_tabular_and_sequence(self, tabular_oof, sequence_oof):
        """Combine tabular (100%) and sequence (100%-seq_len) OOF."""
        pass

    def test_index_intersection_used(self, tabular_oof, sequence_oof):
        """Index intersection used for combined dataset."""
        pass

    def test_combined_n_samples_min_coverage(self, tabular_oof, sequence_oof):
        """Combined n_samples = min coverage across models."""
        pass


class TestHeterogeneousStackingFeatures:
    """Tests for stacking feature matrix construction."""

    def test_stacking_features_2d(self, heterogeneous_oof):
        """Stacking features are always 2D (meta-learner input)."""
        pass

    def test_features_include_all_model_probs(self, heterogeneous_oof):
        """Features include probabilities from all models."""
        pass

    def test_feature_naming_convention(self, heterogeneous_oof):
        """Features named {model}_{prob_class}."""
        pass


class TestHeterogeneousStackingMetaLearner:
    """Tests for meta-learner input preparation."""

    def test_meta_learner_receives_2d_input(self, heterogeneous_oof):
        """Meta-learner always receives 2D input."""
        pass

    def test_meta_learner_y_aligned(self, heterogeneous_oof):
        """Meta-learner y aligned to stacking features."""
        pass

    def test_meta_learner_weights_aligned(self, heterogeneous_oof):
        """Meta-learner weights aligned to stacking features."""
        pass
```

---

#### `tests/snwh/oof/test_oof_no_leakage.py`

```python
"""
Tests for OOF no-leakage invariants.

Tests:
- Purge/embargo enforcement
- No future data in OOF predictions
- Train/test separation per fold
- Label end time handling
"""
import pytest
import numpy as np


class TestOOFPurgeEmbargo:
    """Tests for purge/embargo enforcement."""

    def test_purge_bars_applied(self, oof_with_purge):
        """Purge bars removed between train and test."""
        pass

    def test_embargo_bars_applied(self, oof_with_embargo):
        """Embargo bars removed after test set."""
        pass

    def test_purge_embargo_combined(self, oof_with_purge_embargo):
        """Both purge and embargo applied correctly."""
        pass


class TestOOFNoFutureData:
    """Tests for no future data leakage."""

    def test_train_indices_before_test(self, oof_fold_splits):
        """All train indices chronologically before test indices."""
        pass

    def test_no_overlapping_indices(self, oof_fold_splits):
        """No overlapping indices between train and test."""
        pass

    def test_features_at_time_t_use_only_past(self, oof_features_at_t):
        """Features at time t only use data from t-1 and earlier."""
        pass


class TestOOFLabelEndTime:
    """Tests for label end time handling."""

    def test_label_end_times_respected(self, oof_with_label_end_times):
        """Label end times respected in purge calculation."""
        pass

    def test_overlapping_labels_purged(self, oof_with_overlapping_labels):
        """Samples with overlapping labels purged."""
        pass
```

---

### 2.6 Phase 5 - Feature Strategy

#### `tests/snwh/features/test_feature_strategy_manager.py`

```python
"""
Tests for FeatureStrategyManager - strategy resolution.

Tests:
- Strategy lookup by model name
- Default strategy per family
- Strategy merging with overrides
- Feature set resolution
"""
import pytest


class TestFeatureStrategyLookup:
    """Tests for strategy lookup."""

    @pytest.mark.parametrize("model_name", [
        "xgboost", "lightgbm", "catboost", "random_forest", "logistic", "svm",
        "lstm", "gru", "tcn", "transformer", "patchtst", "itransformer",
        "tft", "nbeats", "inceptiontime", "resnet1d"
    ])
    def test_strategy_exists_for_all_models(self, model_name):
        """Strategy exists for all 16 base models."""
        pass

    def test_unknown_model_raises(self):
        """Unknown model name raises ValueError."""
        pass


class TestFeatureStrategyDefaults:
    """Tests for default strategies per family."""

    def test_boosting_default_engineered(self):
        """Boosting models default to engineered features."""
        pass

    def test_transformer_default_raw(self):
        """Transformer models default to raw OHLCV."""
        pass

    def test_neural_default_momentum_volatility(self):
        """Neural models default to momentum+volatility+wavelets."""
        pass


class TestFeatureStrategyOverrides:
    """Tests for strategy overrides."""

    def test_override_baseline_features(self):
        """Baseline features can be overridden."""
        pass

    def test_override_mtf_mode(self):
        """mtf_mode can be overridden."""
        pass

    def test_override_min_max_features(self):
        """min/max features can be overridden."""
        pass


class TestFeatureSetResolution:
    """Tests for feature set resolution."""

    def test_resolve_feature_columns(self, sample_features_df, xgboost_strategy):
        """Resolve actual column names from strategy."""
        pass

    def test_resolve_handles_missing_features(self, sample_features_df):
        """Resolution handles missing features gracefully."""
        pass

    def test_resolve_mtf_indicators(self, sample_mtf_features_df):
        """Resolution includes MTF indicator features."""
        pass
```

---

#### `tests/snwh/features/test_model_feature_strategies.py`

```python
"""
Tests for MODEL_FEATURE_STRATEGIES - 23-model strategy definitions.

Tests:
- All 23 models have strategy defined
- Strategy attributes valid
- Baseline features exist
- MTF mode consistency with input_rank
"""
import pytest
from src.features.strategies import MODEL_FEATURE_STRATEGIES, ModelFeatureStrategy


class TestAllModelsHaveStrategy:
    """Tests that all 23 models have strategies."""

    @pytest.mark.parametrize("model_name", [
        "xgboost", "lightgbm", "catboost",
        "random_forest", "logistic", "svm",
        "lstm", "gru", "tcn", "transformer", "patchtst",
        "itransformer", "tft", "nbeats", "inceptiontime", "resnet1d",
        "voting", "stacking", "blending",
        "ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"
    ])
    def test_strategy_defined(self, model_name):
        """Strategy defined for all 23 models."""
        assert model_name in MODEL_FEATURE_STRATEGIES

    def test_total_strategy_count(self):
        """Total strategies >= 23."""
        assert len(MODEL_FEATURE_STRATEGIES) >= 23


class TestStrategyAttributes:
    """Tests for strategy attribute validity."""

    @pytest.mark.parametrize("model_name", list(MODEL_FEATURE_STRATEGIES.keys()))
    def test_strategy_has_required_fields(self, model_name):
        """All strategies have required fields."""
        strategy = MODEL_FEATURE_STRATEGIES[model_name]
        assert isinstance(strategy, ModelFeatureStrategy)
        assert hasattr(strategy, "model_name")
        assert hasattr(strategy, "family")
        assert hasattr(strategy, "baseline_features")
        assert hasattr(strategy, "mtf_mode")
        assert hasattr(strategy, "min_features")
        assert hasattr(strategy, "max_features")

    @pytest.mark.parametrize("model_name", list(MODEL_FEATURE_STRATEGIES.keys()))
    def test_min_less_than_max_features(self, model_name):
        """min_features < max_features for all strategies."""
        strategy = MODEL_FEATURE_STRATEGIES[model_name]
        assert strategy.min_features <= strategy.max_features


class TestBaselineFeatures:
    """Tests for baseline feature validity."""

    def test_xgboost_has_mtf_indicators(self):
        """XGBoost baseline includes MTF indicators."""
        strategy = MODEL_FEATURE_STRATEGIES["xgboost"]
        assert any("_15min" in f or "_1h" in f for f in strategy.baseline_features)

    def test_patchtst_raw_ohlcv_only(self):
        """PatchTST baseline is raw OHLCV only."""
        strategy = MODEL_FEATURE_STRATEGIES["patchtst"]
        assert all(f in ["open", "high", "low", "close", "volume"]
                   for f in strategy.baseline_features)

    def test_lstm_includes_wavelets(self):
        """LSTM baseline includes wavelet features."""
        strategy = MODEL_FEATURE_STRATEGIES["lstm"]
        assert any("wavelet" in f for f in strategy.baseline_features)


class TestMTFModeConsistency:
    """Tests for MTF mode consistency."""

    def test_multi_stream_models_have_multi_stream_mode(self):
        """Models with multi_stream mtf_mode are transformers."""
        for name, strategy in MODEL_FEATURE_STRATEGIES.items():
            if strategy.mtf_mode == "multi_stream":
                assert strategy.family in ["transformer"], f"{name} unexpected family"

    def test_sequence_models_require_sequences(self):
        """Models with requires_sequences=True are neural/transformer."""
        for name, strategy in MODEL_FEATURE_STRATEGIES.items():
            if strategy.requires_sequences:
                assert strategy.family in ["neural", "transformer", "cnn"], f"{name} unexpected family"
```

---

## 3. Integration Tests

### `tests/snwh/integration/test_heterogeneous_stacking_e2e.py`

```python
"""
End-to-end tests for heterogeneous stacking.

Tests the complete flow:
XGBoost (15min, 2D) + LSTM (5min, 3D) + PatchTST (1min, 4D) -> Meta-learner
"""
import pytest
import numpy as np


class TestHeterogeneousStackingE2E:
    """End-to-end heterogeneous stacking tests."""

    @pytest.fixture
    def heterogeneous_base_models(self):
        """Configure heterogeneous base models."""
        return [
            {"name": "xgboost", "timeframe": "15min", "mtf_mode": "indicators"},
            {"name": "lstm", "timeframe": "5min", "mtf_mode": "none", "seq_len": 30},
            {"name": "patchtst", "timeframe": "1min", "mtf_mode": "multi_stream"},
        ]

    def test_all_base_models_train(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """All base models train successfully."""
        pass

    def test_oof_generated_for_all(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """OOF predictions generated for all base models."""
        pass

    def test_oof_coverage_aligned(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """OOF coverage aligned across heterogeneous models."""
        pass

    def test_stacking_dataset_built(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """Stacking dataset built from heterogeneous OOF."""
        pass

    def test_meta_learner_trains(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """Meta-learner trains on stacking dataset."""
        pass

    def test_full_retrain_succeeds(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """Full retrain of base models succeeds."""
        pass

    def test_test_predictions_combine(self, heterogeneous_base_models, sample_canonical_ohlcv):
        """Test predictions combine through meta-learner."""
        pass


class TestHeterogeneousStackingDataFlow:
    """Tests for data flow in heterogeneous stacking."""

    def test_same_1min_source_for_all(self, heterogeneous_base_models):
        """All models derive from same 1-min canonical source."""
        pass

    def test_different_timeframes_derived(self, heterogeneous_base_models):
        """Different primary timeframes derived correctly."""
        pass

    def test_different_feature_sets(self, heterogeneous_base_models):
        """Different models get different feature sets."""
        pass

    def test_same_labels_and_splits(self, heterogeneous_base_models):
        """Same labels and train/val/test splits for all."""
        pass
```

---

### `tests/snwh/integration/test_mixed_family_training.py`

```python
"""
Integration tests for mixed-family training.

Tests training tabular + sequence + transformer in single run.
"""
import pytest


class TestMixedFamilyTraining:
    """Tests for mixed-family training."""

    @pytest.fixture
    def mixed_family_config(self):
        """Configuration for mixed-family training."""
        return {
            "models": [
                {"name": "random_forest", "family": "classical"},
                {"name": "gru", "family": "neural"},
                {"name": "transformer", "family": "transformer"},
            ]
        }

    def test_all_families_initialize(self, mixed_family_config):
        """All model families initialize correctly."""
        pass

    def test_adapter_routing_correct(self, mixed_family_config):
        """Correct adapter routed to each family."""
        pass

    def test_data_shapes_match_requirements(self, mixed_family_config):
        """Data shapes match model requirements."""
        pass

    def test_training_completes(self, mixed_family_config, sample_canonical_ohlcv):
        """Training completes for all models."""
        pass

    def test_predictions_valid(self, mixed_family_config, sample_canonical_ohlcv):
        """All models produce valid predictions."""
        pass
```

---

## 4. Regression Tests

### `tests/snwh/regression/test_oof_leakage_regression.py`

```python
"""
Regression tests for OOF no-leakage invariants.

These tests prevent reintroduction of leakage bugs.
"""
import pytest
import numpy as np


class TestOOFLeakageRegression:
    """Regression tests for OOF leakage."""

    def test_no_future_labels_in_train(self, oof_fold_data):
        """REGRESSION: Train set never contains future labels."""
        pass

    def test_purge_always_positive(self, purge_config):
        """REGRESSION: Purge bars always positive."""
        pass

    def test_embargo_scaled_from_horizon(self, embargo_config):
        """REGRESSION: Embargo scaled from max horizon."""
        pass

    def test_mtf_features_shifted(self, mtf_features):
        """REGRESSION: MTF features have shift(1) applied."""
        pass

    def test_sequence_labels_window_aligned(self, sequence_data):
        """REGRESSION: Sequence labels aligned to window end."""
        pass


class TestSequenceTabularCoverageRegression:
    """Regression tests for coverage alignment."""

    def test_tabular_100_percent_coverage(self, tabular_oof):
        """REGRESSION: Tabular OOF has 100% coverage."""
        pass

    def test_sequence_coverage_formula(self, sequence_oof, seq_len):
        """REGRESSION: Sequence coverage = (n - seq_len + 1) / n."""
        pass

    def test_coverage_intersection_correct(self, heterogeneous_oof):
        """REGRESSION: Heterogeneous coverage uses intersection."""
        pass
```

---

## 5. Property-Based Tests

### `tests/snwh/property/test_adapter_properties.py`

```python
"""
Property-based tests for adapter invariants.

Uses hypothesis to verify adapter properties hold for arbitrary inputs.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np


class TestTabularAdapterProperties:
    """Property-based tests for TabularAdapter."""

    @given(
        n_samples=st.integers(min_value=10, max_value=1000),
        n_features=st.integers(min_value=5, max_value=200),
    )
    @settings(max_examples=50)
    def test_output_shape_invariant(self, n_samples, n_features):
        """Output shape always (n_samples, n_features)."""
        pass

    @given(
        data=st.data(),
    )
    def test_feature_order_stable(self, data):
        """Feature order stable across calls."""
        pass


class TestSequenceAdapterProperties:
    """Property-based tests for SequenceAdapter."""

    @given(
        n_samples=st.integers(min_value=100, max_value=1000),
        seq_len=st.integers(min_value=10, max_value=60),
    )
    @settings(max_examples=50)
    def test_coverage_invariant(self, n_samples, seq_len):
        """Coverage = (n - seq_len + 1) / n for valid inputs."""
        assume(seq_len < n_samples)
        pass

    @given(
        seq_len=st.integers(min_value=10, max_value=60),
    )
    def test_label_alignment_invariant(self, seq_len):
        """Labels aligned to window end for all seq_len."""
        pass


class TestMultiStreamAdapterProperties:
    """Property-based tests for MultiStreamAdapter."""

    @given(
        n_streams=st.integers(min_value=2, max_value=9),
        seq_len=st.integers(min_value=10, max_value=60),
    )
    @settings(max_examples=30)
    def test_stream_count_invariant(self, n_streams, seq_len):
        """n_streams dimension matches timeframe count."""
        pass
```

---

### `tests/snwh/property/test_oof_coverage_properties.py`

```python
"""
Property-based tests for OOF coverage invariants.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np


class TestOOFCoverageProperties:
    """Property-based tests for OOF coverage."""

    @given(
        n_samples=st.integers(min_value=100, max_value=10000),
        n_folds=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50)
    def test_coverage_100_percent(self, n_samples, n_folds):
        """OOF coverage is 100% for tabular models."""
        pass

    @given(
        purge_bars=st.integers(min_value=0, max_value=100),
        embargo_bars=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50)
    def test_no_train_test_overlap(self, purge_bars, embargo_bars):
        """Train and test sets never overlap after purge/embargo."""
        pass
```

---

## 6. Fixtures (conftest.py)

### `tests/snwh/conftest.py`

```python
"""
Shared fixtures for SNwH tests.

Provides:
- Synthetic OHLCV data generators
- MTF data generators
- Mock adapters and models
- Configuration fixtures
- OOF prediction fixtures
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# CANONICAL OHLCV DATA
# =============================================================================

@pytest.fixture
def sample_canonical_ohlcv() -> pd.DataFrame:
    """
    Generate canonical 1-min OHLCV data.

    Returns:
        DataFrame with 5000 1-min bars:
        - datetime, open, high, low, close, volume
        - symbol='MES'
        - Realistic price dynamics
    """
    np.random.seed(42)
    n = 5000

    start_time = datetime(2024, 1, 2, 9, 30)
    dates = pd.date_range(start=start_time, periods=n, freq="1min")

    # Generate realistic price series
    base_price = 4500.0
    returns = np.random.randn(n) * 0.0002  # 0.02% per minute
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    daily_range = np.abs(np.random.randn(n) * 0.0003)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.0001)

    # Ensure valid OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n)

    return pd.DataFrame({
        "datetime": dates,
        "symbol": "MES",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_5min_data(sample_canonical_ohlcv) -> pd.DataFrame:
    """Generate 5-min resampled data from canonical 1-min."""
    df = sample_canonical_ohlcv.set_index("datetime")
    resampled = df.resample("5min", closed="left", label="left").agg({
        "symbol": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return resampled.reset_index()


@pytest.fixture
def sample_9tf_data(sample_canonical_ohlcv) -> Dict[str, pd.DataFrame]:
    """Generate all 9 canonical timeframes from 1-min source."""
    df = sample_canonical_ohlcv.set_index("datetime")

    timeframes = {
        "1min": "1min", "5min": "5min", "10min": "10min",
        "15min": "15min", "20min": "20min", "25min": "25min",
        "30min": "30min", "45min": "45min", "1h": "1h",
    }

    result = {}
    for name, freq in timeframes.items():
        if name == "1min":
            result[name] = df.reset_index()
        else:
            resampled = df.resample(freq, closed="left", label="left").agg({
                "symbol": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            result[name] = resampled.reset_index()

    return result


# =============================================================================
# FEATURE DATA
# =============================================================================

@pytest.fixture
def sample_engineered_features(sample_5min_data) -> pd.DataFrame:
    """Generate sample engineered features."""
    df = sample_5min_data.copy()
    np.random.seed(42)
    n = len(df)

    # Momentum features
    df["returns"] = df["close"].pct_change().fillna(0)
    df["rsi_14"] = 50 + np.random.randn(n) * 15
    df["macd_line"] = np.random.randn(n) * 0.5
    df["macd_signal"] = df["macd_line"].rolling(9).mean().fillna(0)

    # Volatility features
    df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean().fillna(0.5)
    df["bb_width"] = np.random.uniform(0.02, 0.05, n)

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(20).mean().fillna(df["volume"].mean())

    # Apply shift(1) to prevent lookahead
    feature_cols = ["returns", "rsi_14", "macd_line", "macd_signal",
                    "atr_14", "bb_width", "volume_sma_20"]
    for col in feature_cols:
        df[col] = df[col].shift(1)

    return df.dropna().reset_index(drop=True)


@pytest.fixture
def sample_mtf_features_df(sample_engineered_features) -> pd.DataFrame:
    """Generate features with MTF indicators."""
    df = sample_engineered_features.copy()
    np.random.seed(42)
    n = len(df)

    # MTF indicators (simulated with shift(1) already applied)
    df["rsi_14_15min"] = 50 + np.random.randn(n) * 15
    df["atr_14_15min"] = np.random.uniform(0.3, 0.8, n)
    df["rsi_14_1h"] = 50 + np.random.randn(n) * 15
    df["atr_14_1h"] = np.random.uniform(0.4, 1.0, n)

    return df


# =============================================================================
# LABELED DATA
# =============================================================================

@pytest.fixture
def sample_labeled_data(sample_engineered_features) -> pd.DataFrame:
    """Generate labeled data with triple-barrier labels."""
    df = sample_engineered_features.copy()
    np.random.seed(42)
    n = len(df)

    # Add labels for multiple horizons
    for horizon in [5, 10, 15, 20]:
        labels = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
        df[f"label_h{horizon}"] = labels
        df[f"sample_weight_h{horizon}"] = np.random.uniform(0.5, 1.5, n)

    return df


# =============================================================================
# SEQUENCE DATA
# =============================================================================

@pytest.fixture
def sample_sequence_data(sample_labeled_data) -> Dict[str, Any]:
    """Generate sequence data for neural models."""
    df = sample_labeled_data.copy()
    seq_len = 30

    # Feature columns (excluding metadata and labels)
    feature_cols = [c for c in df.columns
                    if c not in ["datetime", "symbol", "open", "high", "low", "close", "volume"]
                    and not c.startswith("label_") and not c.startswith("sample_weight_")]

    X = df[feature_cols].values
    y = df["label_h20"].values

    # Create sequences
    n_sequences = len(X) - seq_len + 1
    X_seq = np.zeros((n_sequences, seq_len, len(feature_cols)), dtype=np.float32)
    y_seq = np.zeros(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        X_seq[i] = X[i:i+seq_len]
        y_seq[i] = y[i+seq_len-1]  # Label at window end

    return {
        "X": X_seq,
        "y": y_seq,
        "feature_names": feature_cols,
        "seq_len": seq_len,
    }


# =============================================================================
# MTF DATA
# =============================================================================

@pytest.fixture
def sample_mtf_data(sample_9tf_data) -> pd.DataFrame:
    """Generate combined MTF DataFrame for multi-stream adapter."""
    # Use 5min as base and add MTF columns
    base_df = sample_9tf_data["5min"].copy()
    np.random.seed(42)
    n = len(base_df)

    # Add MTF columns for each timeframe
    for tf in ["1min", "15min", "30min", "1h"]:
        for col in ["close", "volume", "atr_14"]:
            base_df[f"{col}_{tf}"] = np.random.randn(n) * 0.01 + base_df["close"].values

    return base_df


# =============================================================================
# OOF PREDICTIONS
# =============================================================================

@pytest.fixture
def tabular_oof(sample_labeled_data) -> Dict[str, np.ndarray]:
    """Generate mock tabular OOF predictions (100% coverage)."""
    n = len(sample_labeled_data)
    np.random.seed(42)

    probs = np.random.dirichlet([1, 1, 1], size=n)
    preds = np.argmax(probs, axis=1) - 1

    return {
        "model_name": "xgboost",
        "predictions": preds,
        "probabilities": probs,
        "indices": np.arange(n),
        "coverage": 1.0,
    }


@pytest.fixture
def sequence_oof(sample_labeled_data) -> Dict[str, np.ndarray]:
    """Generate mock sequence OOF predictions (100%-seq_len coverage)."""
    n = len(sample_labeled_data)
    seq_len = 30
    np.random.seed(42)

    # Sequence models cover fewer samples
    n_covered = n - seq_len + 1
    probs = np.random.dirichlet([1, 1, 1], size=n_covered)
    preds = np.argmax(probs, axis=1) - 1

    return {
        "model_name": "lstm",
        "predictions": preds,
        "probabilities": probs,
        "indices": np.arange(seq_len - 1, n),  # Start after warmup
        "coverage": n_covered / n,
        "seq_len": seq_len,
    }


@pytest.fixture
def heterogeneous_oof(tabular_oof, sequence_oof) -> Dict[str, Dict]:
    """Combined heterogeneous OOF predictions."""
    return {
        "xgboost": tabular_oof,
        "lstm": sequence_oof,
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

@pytest.fixture
def xgboost_strategy():
    """XGBoost feature strategy."""
    from src.features.strategies import MODEL_FEATURE_STRATEGIES
    return MODEL_FEATURE_STRATEGIES["xgboost"]


@pytest.fixture
def lstm_strategy():
    """LSTM feature strategy."""
    from src.features.strategies import MODEL_FEATURE_STRATEGIES
    return MODEL_FEATURE_STRATEGIES["lstm"]


@pytest.fixture
def patchtst_strategy():
    """PatchTST feature strategy."""
    from src.features.strategies import MODEL_FEATURE_STRATEGIES
    return MODEL_FEATURE_STRATEGIES["patchtst"]


@pytest.fixture
def multi_tf_config() -> Dict[str, Any]:
    """Multi-timeframe configuration."""
    return {
        "timeframes": ["1min", "5min", "15min", "30min", "1h"],
        "anchor_tf": "5min",
        "primary_tf": "15min",
    }


# =============================================================================
# TEMPORARY DIRECTORIES
# =============================================================================

@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model save/load tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for data tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
```

---

## 7. CI/CD Integration

### Test Execution Order

```yaml
# .github/workflows/test.yml (excerpt)
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      # 1. Unit tests first (fast, isolated)
      - name: Unit Tests
        run: |
          pytest tests/snwh/contracts/ -v --tb=short
          pytest tests/snwh/config/ -v --tb=short
          pytest tests/snwh/adapters/ -v --tb=short
          pytest tests/snwh/timeframe/ -v --tb=short
          pytest tests/snwh/oof/ -v --tb=short
          pytest tests/snwh/features/ -v --tb=short

      # 2. Property tests (medium, generative)
      - name: Property Tests
        run: pytest tests/snwh/property/ -v --tb=short

      # 3. Regression tests (critical invariants)
      - name: Regression Tests
        run: pytest tests/snwh/regression/ -v --tb=short

      # 4. Integration tests last (slow, end-to-end)
      - name: Integration Tests
        run: pytest tests/snwh/integration/ -v --tb=short --timeout=600
```

### Parallel Test Execution

```ini
# pytest.ini
[pytest]
addopts = -n auto --dist loadfile
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
```

### Coverage Requirements

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### Coverage Targets by Module

| Module | Target | Critical Paths |
|--------|--------|----------------|
| `src/contracts/` | 95% | Schema validation, hash generation |
| `src/adapters/` | 90% | Shape transformation, routing |
| `src/timeframe/` | 90% | shift(1) enforcement, alignment |
| `src/oof/` | 95% | Coverage validation, no-leakage |
| `src/features/strategies.py` | 85% | Strategy lookup |

---

## 8. Test Data Generators

### `tests/snwh/generators.py`

```python
"""
Test data generators for SNwH tests.

Provides functions to generate:
- Market scenarios (trending, ranging, volatile)
- MTF data with correct alignment
- OOF predictions with specified coverage
- Feature matrices with correlations
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def generate_market_scenario(
    scenario: str,
    n_bars: int = 1000,
    base_price: float = 4500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate OHLCV data for specific market scenario.

    Args:
        scenario: One of 'trending_up', 'trending_down', 'ranging', 'volatile'
        n_bars: Number of bars to generate
        base_price: Starting price
        seed: Random seed

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(seed)

    if scenario == "trending_up":
        drift = 0.0002  # Positive drift
        vol = 0.001
    elif scenario == "trending_down":
        drift = -0.0002  # Negative drift
        vol = 0.001
    elif scenario == "ranging":
        drift = 0.0
        vol = 0.0005
    elif scenario == "volatile":
        drift = 0.0
        vol = 0.003
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    returns = drift + np.random.randn(n_bars) * vol
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = close * (1 + np.abs(np.random.randn(n_bars)) * vol)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * vol)
    open_ = close * (1 + np.random.randn(n_bars) * vol * 0.5)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.random.randint(100, 10000, n_bars)

    dates = pd.date_range(
        start=datetime(2024, 1, 2, 9, 30),
        periods=n_bars,
        freq="1min"
    )

    return pd.DataFrame({
        "datetime": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def generate_mtf_aligned_data(
    base_df: pd.DataFrame,
    timeframes: list[str],
    shift_mtf: bool = True,
) -> pd.DataFrame:
    """
    Generate MTF-aligned data from base DataFrame.

    Args:
        base_df: Base OHLCV DataFrame (1-min)
        timeframes: List of timeframes to generate
        shift_mtf: Whether to apply shift(1) to MTF features

    Returns:
        DataFrame with MTF columns
    """
    result = base_df.copy()

    tf_minutes = {
        "1min": 1, "5min": 5, "10min": 10, "15min": 15,
        "20min": 20, "25min": 25, "30min": 30, "45min": 45, "1h": 60,
    }

    for tf in timeframes:
        if tf == "1min":
            continue

        minutes = tf_minutes.get(tf)
        if minutes is None:
            raise ValueError(f"Unknown timeframe: {tf}")

        # Simulate MTF feature with appropriate lag
        for col in ["close", "volume"]:
            mtf_col = f"{col}_{tf}"
            result[mtf_col] = result[col].rolling(minutes).mean()
            if shift_mtf:
                result[mtf_col] = result[mtf_col].shift(1)

    return result


def generate_oof_predictions(
    n_samples: int,
    coverage: float = 1.0,
    start_idx: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mock OOF predictions.

    Args:
        n_samples: Total samples in training data
        coverage: Fraction of samples covered (1.0 for tabular, <1.0 for sequence)
        start_idx: Starting index for covered samples
        seed: Random seed

    Returns:
        Tuple of (predictions, probabilities, indices)
    """
    np.random.seed(seed)

    n_covered = int(n_samples * coverage)
    probs = np.random.dirichlet([1, 1, 1], size=n_covered)
    preds = np.argmax(probs, axis=1) - 1
    indices = np.arange(start_idx, start_idx + n_covered)

    return preds, probs, indices


def generate_correlated_features(
    n_samples: int,
    n_features: int,
    correlation_groups: Optional[list[list[int]]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate feature matrix with correlation structure.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        correlation_groups: List of feature index groups that should be correlated
        seed: Random seed

    Returns:
        DataFrame with correlated features
    """
    np.random.seed(seed)

    if correlation_groups is None:
        # Default: first 5 correlated, rest independent
        correlation_groups = [list(range(5))]

    features = np.random.randn(n_samples, n_features)

    for group in correlation_groups:
        if len(group) < 2:
            continue

        # Make features in group correlated
        base = np.random.randn(n_samples)
        for idx in group:
            if idx < n_features:
                features[:, idx] = base + np.random.randn(n_samples) * 0.2

    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(features, columns=columns)
```

---

## 9. Summary

### Test Counts by Category

| Category | Files | Test Classes | Test Methods (est.) |
|----------|-------|--------------|---------------------|
| Contracts | 3 | 12 | 40 |
| Config | 3 | 12 | 45 |
| Adapters | 5 | 16 | 55 |
| Timeframe | 3 | 8 | 30 |
| OOF | 4 | 12 | 45 |
| Features | 3 | 8 | 35 |
| Integration | 2 | 4 | 20 |
| Regression | 2 | 4 | 15 |
| Property | 3 | 6 | 20 |
| **Total** | **28** | **82** | **~305** |

### Critical Test Scenarios

1. **Heterogeneous stacking end-to-end**: XGBoost (15min, 2D) + LSTM (5min, 3D) + PatchTST (1min, 4D)
2. **OOF coverage alignment**: Tabular 100% vs Sequence (100%-seq_len)
3. **Per-model feature selection**: Different models get different baseline features
4. **MTF leakage prevention**: shift(1) applied to all MTF indicators
5. **Adapter shape validation**: Correct routing for all 23 models

### TDD Workflow

1. **Red**: Write failing test for SNwH requirement
2. **Green**: Implement minimal code to pass
3. **Refactor**: Clean up while tests pass
4. **Commit**: Tests + implementation together

### Test Execution Commands

```bash
# Run all SNwH tests
pytest tests/snwh/ -v

# Run specific category
pytest tests/snwh/contracts/ -v
pytest tests/snwh/adapters/ -v

# Run with coverage
pytest tests/snwh/ --cov=src --cov-report=html

# Run property-based tests with more examples
pytest tests/snwh/property/ --hypothesis-seed=42 -v

# Run integration tests only
pytest tests/snwh/integration/ -v -m integration

# Run regression tests (critical)
pytest tests/snwh/regression/ -v --tb=long
```
