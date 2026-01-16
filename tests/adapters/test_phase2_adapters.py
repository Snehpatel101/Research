"""
Tests for Phase 2 Adapter Architecture.

Tests cover:
- AdapterResult dataclass and validation
- AdapterRegistry registration and lookup
- TabularAdapter (2D) transformation
- SequenceAdapter (3D) transformation
- MultiStreamAdapter (4D) transformation
- Integration with model contracts
"""

import numpy as np
import pandas as pd
import pytest

from src.contracts import DataRank


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame({
        # Features
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
        "rsi_14": np.random.uniform(20, 80, n_samples),
        "macd_line": np.random.randn(n_samples),
        # OHLCV (for multi-stream)
        "open": np.random.uniform(100, 110, n_samples),
        "high": np.random.uniform(105, 115, n_samples),
        "low": np.random.uniform(95, 105, n_samples),
        "close": np.random.uniform(100, 110, n_samples),
        "volume": np.random.uniform(1000, 10000, n_samples),
        # Labels
        "label_h20": np.random.randint(0, 3, n_samples),
        "sample_weight_h20": np.random.uniform(0.5, 1.5, n_samples),
        # Metadata
        "symbol": ["MES"] * n_samples,
        "timeframe": ["5min"] * n_samples,
    })


@pytest.fixture
def feature_columns():
    """Feature columns for testing."""
    return ["feature_1", "feature_2", "feature_3", "rsi_14", "macd_line"]


@pytest.fixture
def ohlcv_columns():
    """OHLCV columns for multi-stream testing."""
    return ["open", "high", "low", "close", "volume"]


# =============================================================================
# Test AdapterResult
# =============================================================================


class TestAdapterResult:
    """Tests for AdapterResult dataclass."""

    def test_adapter_result_creation(self):
        """Test basic AdapterResult creation."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.TABULAR_2D,
        )

        assert result.n_samples == 100
        assert result.n_features == 10
        assert result.data_rank == DataRank.TABULAR_2D

    def test_adapter_result_validation_passes(self):
        """Test validation passes for valid result."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.TABULAR_2D,
        )

        is_valid, issues = result.validate()
        assert is_valid
        assert len(issues) == 0

    def test_adapter_result_validation_rank_mismatch(self):
        """Test validation catches rank mismatch."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)  # 2D
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.SEQUENCE_3D,  # Wrong rank
        )

        is_valid, issues = result.validate()
        assert not is_valid
        assert any("rank mismatch" in issue for issue in issues)

    def test_adapter_result_validation_y_length_mismatch(self):
        """Test validation catches y length mismatch."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int64)  # Wrong length

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.TABULAR_2D,
        )

        is_valid, issues = result.validate()
        assert not is_valid
        assert any("length mismatch" in issue for issue in issues)

    def test_adapter_result_validation_nan_detection(self):
        """Test validation catches NaN values."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)
        X[0, 0] = np.nan  # Add NaN
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.TABULAR_2D,
        )

        is_valid, issues = result.validate()
        assert not is_valid
        assert any("NaN" in issue for issue in issues)

    def test_adapter_result_3d_features(self):
        """Test n_features computed correctly for 3D arrays."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 60, 10).astype(np.float32)  # 3D
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.SEQUENCE_3D,
        )

        assert result.n_samples == 100
        assert result.n_features == 10  # Last dimension
        assert result.sequence_length is None  # Not set automatically

    def test_adapter_result_4d_features(self):
        """Test n_features computed correctly for 4D arrays."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 3, 60, 5).astype(np.float32)  # 4D
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.MULTI_TF_4D,
        )

        assert result.n_samples == 100
        assert result.n_features == 5  # Last dimension

    def test_adapter_result_to_dict(self):
        """Test serialization to dict."""
        from src.adapters.base import AdapterResult

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100).astype(np.int64)

        result = AdapterResult(
            X=X,
            y=y,
            data_rank=DataRank.TABULAR_2D,
            adapter_name="tabular",
        )

        d = result.to_dict()
        assert d["n_samples"] == 100
        assert d["n_features"] == 10
        assert d["data_rank"] == 2
        assert d["adapter_name"] == "tabular"
        assert d["X_shape"] == [100, 10]


# =============================================================================
# Test AdapterRegistry
# =============================================================================


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_registry_has_adapters(self):
        """Test registry has registered adapters."""
        from src.adapters import AdapterRegistry

        adapters = AdapterRegistry.list_adapters()
        assert "tabular" in adapters
        assert "sequence" in adapters
        assert "multi_stream" in adapters

    def test_registry_get_by_id(self):
        """Test getting adapter by ID."""
        from src.adapters import AdapterRegistry, TabularAdapter

        adapter_class = AdapterRegistry.get("tabular")
        assert adapter_class == TabularAdapter

    def test_registry_get_unknown_raises(self):
        """Test getting unknown adapter raises error."""
        from src.adapters import AdapterRegistry

        with pytest.raises(ValueError, match="Unknown adapter"):
            AdapterRegistry.get("unknown_adapter")

    def test_registry_create(self):
        """Test creating adapter instance."""
        from src.adapters import AdapterRegistry, TabularAdapter

        adapter = AdapterRegistry.create("tabular")
        assert isinstance(adapter, TabularAdapter)

    def test_registry_get_for_model(self):
        """Test getting adapter for model."""
        from src.adapters import AdapterRegistry, TabularAdapter, SequenceAdapter

        # XGBoost uses tabular
        adapter = AdapterRegistry.get_for_model("xgboost")
        assert isinstance(adapter, TabularAdapter)

        # LSTM uses sequence
        adapter = AdapterRegistry.get_for_model("lstm")
        assert isinstance(adapter, SequenceAdapter)

    def test_registry_is_registered(self):
        """Test is_registered check."""
        from src.adapters import AdapterRegistry

        assert AdapterRegistry.is_registered("tabular")
        assert not AdapterRegistry.is_registered("unknown")


class TestGetAdapter:
    """Tests for get_adapter convenience function."""

    def test_get_adapter_by_model_name(self):
        """Test get_adapter by model name."""
        from src.adapters import get_adapter, TabularAdapter

        adapter = get_adapter(model_name="xgboost")
        assert isinstance(adapter, TabularAdapter)

    def test_get_adapter_by_adapter_id(self):
        """Test get_adapter by adapter ID."""
        from src.adapters import get_adapter, SequenceAdapter

        adapter = get_adapter(adapter_id="sequence")
        assert isinstance(adapter, SequenceAdapter)

    def test_get_adapter_with_kwargs(self):
        """Test get_adapter passes kwargs to constructor."""
        from src.adapters import get_adapter

        adapter = get_adapter(adapter_id="sequence", sequence_length=30)
        assert adapter.sequence_length == 30

    def test_get_adapter_requires_arg(self):
        """Test get_adapter requires model_name or adapter_id."""
        from src.adapters import get_adapter

        with pytest.raises(ValueError, match="Either model_name or adapter_id"):
            get_adapter()


# =============================================================================
# Test TabularAdapter
# =============================================================================


class TestTabularAdapter:
    """Tests for TabularAdapter (2D)."""

    def test_tabular_transform_basic(self, sample_df, feature_columns):
        """Test basic tabular transformation."""
        from src.adapters import TabularAdapter

        adapter = TabularAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
        )

        result = adapter.transform(sample_df)

        assert result.X.ndim == 2
        assert result.X.shape == (200, 5)
        assert result.y.shape == (200,)
        assert result.data_rank == DataRank.TABULAR_2D
        assert result.adapter_name == "tabular"

    def test_tabular_transform_with_weights(self, sample_df, feature_columns):
        """Test tabular transformation with weights."""
        from src.adapters import TabularAdapter

        adapter = TabularAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            weight_column="sample_weight_h20",
        )

        result = adapter.transform(sample_df)

        assert result.weights is not None
        assert result.weights.shape == (200,)

    def test_tabular_transform_auto_features(self, sample_df):
        """Test tabular transformation with auto-detected features."""
        from src.adapters import TabularAdapter

        adapter = TabularAdapter(
            label_column="label_h20",
        )

        result = adapter.transform(sample_df)

        # Should exclude OHLCV and metadata columns
        assert result.X.ndim == 2
        assert result.n_features > 0

    def test_tabular_transform_creates_data_contract(self, sample_df, feature_columns):
        """Test that transformation creates DataContract."""
        from src.adapters import TabularAdapter

        adapter = TabularAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
        )

        result = adapter.transform(sample_df)

        assert result.data_contract is not None
        assert result.data_contract.n_samples == 200
        assert result.data_contract.n_features == 5
        assert result.data_contract.data_rank == DataRank.TABULAR_2D

    def test_tabular_transform_with_contract(self, sample_df, feature_columns):
        """Test transformation with model contract validation."""
        from src.adapters import TabularAdapter
        from src.contracts import get_model_contract

        contract = get_model_contract("xgboost")
        adapter = TabularAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
        )

        # Transform without contract (contract enforces min 40 features)
        result = adapter.transform(sample_df)

        assert result.X.ndim == 2
        assert result.data_rank == DataRank.TABULAR_2D
        # Verify contract info is correct
        assert contract.adapter_id == "tabular"

    def test_tabular_invalid_input_raises(self, sample_df, feature_columns):
        """Test invalid input raises error."""
        from src.adapters import TabularAdapter

        adapter = TabularAdapter(
            feature_columns=feature_columns,
            label_column="nonexistent_label",
        )

        with pytest.raises(ValueError, match="validation failed"):
            adapter.transform(sample_df)


# =============================================================================
# Test SequenceAdapter
# =============================================================================


class TestSequenceAdapter:
    """Tests for SequenceAdapter (3D)."""

    def test_sequence_transform_basic(self, sample_df, feature_columns):
        """Test basic sequence transformation."""
        from src.adapters import SequenceAdapter

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=30,
        )

        result = adapter.transform(sample_df)

        assert result.X.ndim == 3
        # n_sequences = (200 - 30) // 1 + 1 = 171
        assert result.X.shape == (171, 30, 5)
        assert result.y.shape == (171,)
        assert result.data_rank == DataRank.SEQUENCE_3D
        assert result.sequence_length == 30

    def test_sequence_transform_with_stride(self, sample_df, feature_columns):
        """Test sequence transformation with stride."""
        from src.adapters import SequenceAdapter

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=30,
            stride=5,
        )

        result = adapter.transform(sample_df)

        # n_sequences = (200 - 30) // 5 + 1 = 35
        assert result.X.shape[0] == 35
        assert result.X.shape[1] == 30

    def test_sequence_transform_with_weights(self, sample_df, feature_columns):
        """Test sequence transformation preserves weights."""
        from src.adapters import SequenceAdapter

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            weight_column="sample_weight_h20",
            sequence_length=30,
        )

        result = adapter.transform(sample_df)

        assert result.weights is not None
        assert result.weights.shape[0] == result.X.shape[0]

    def test_sequence_original_indices(self, sample_df, feature_columns):
        """Test sequence transformation tracks original indices."""
        from src.adapters import SequenceAdapter

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=30,
        )

        result = adapter.transform(sample_df)

        assert result.original_indices is not None
        # First sequence ends at index 29
        assert result.original_indices[0] == 29
        # Second sequence ends at index 30
        assert result.original_indices[1] == 30

    def test_sequence_contract_overrides_seq_length(self, sample_df, feature_columns):
        """Test model contract overrides sequence length."""
        from src.adapters import SequenceAdapter
        from src.contracts import get_model_contract

        contract = get_model_contract("lstm")
        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=30,  # This will be overridden
        )

        result = adapter.transform(sample_df, model_contract=contract)

        # Contract's sequence_length should be used
        assert result.sequence_length == contract.sequence_length

    def test_sequence_too_short_df_returns_empty(self, feature_columns):
        """Test sequence adapter handles short DataFrames with warning."""
        from src.adapters import SequenceAdapter

        # Create very short DataFrame
        short_df = pd.DataFrame({
            "feature_1": np.random.randn(10),
            "feature_2": np.random.randn(10),
            "feature_3": np.random.randn(10),
            "rsi_14": np.random.randn(10),
            "macd_line": np.random.randn(10),
            "label_h20": np.random.randint(0, 3, 10),
        })

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=60,  # Longer than df
        )

        # Adapter returns empty result with warning (doesn't raise)
        result = adapter.transform(short_df)
        assert result.n_samples == 0
        assert result.X.shape[0] == 0

    def test_sequence_symbol_isolation(self, feature_columns):
        """Test sequences don't cross symbol boundaries."""
        from src.adapters import SequenceAdapter

        # Create df with two symbols
        df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "rsi_14": np.random.randn(100),
            "macd_line": np.random.randn(100),
            "label_h20": np.random.randint(0, 3, 100),
            "symbol": ["MES"] * 50 + ["MGC"] * 50,
        })

        adapter = SequenceAdapter(
            feature_columns=feature_columns,
            label_column="label_h20",
            sequence_length=30,
            symbol_column="symbol",
        )

        result = adapter.transform(df)

        # Each symbol has 50 rows, so (50-30)//1+1 = 21 sequences each
        # Total = 42 sequences
        assert result.X.shape[0] == 42


# =============================================================================
# Test MultiStreamAdapter
# =============================================================================


class TestMultiStreamAdapter:
    """Tests for MultiStreamAdapter (4D)."""

    def test_multi_stream_transform_basic(self, sample_df, ohlcv_columns):
        """Test basic multi-stream transformation."""
        from src.adapters import MultiStreamAdapter

        # Create additional timeframe dfs
        df_5min = sample_df.copy()
        df_15min = sample_df.iloc[::3].reset_index(drop=True)  # 1/3 rows

        adapter = MultiStreamAdapter(
            feature_columns=ohlcv_columns,
            label_column="label_h20",
            sequence_length=30,
            timeframes=["5min", "15min"],
        )

        result = adapter.transform(
            sample_df,
            additional_dfs={"15min": df_15min},
        )

        assert result.X.ndim == 4
        # Shape: (n_sequences, n_tfs, seq_len, n_features)
        assert result.X.shape[1] == 2  # 2 timeframes
        assert result.X.shape[2] == 30  # sequence length
        assert result.X.shape[3] == 5  # OHLCV features
        assert result.data_rank == DataRank.MULTI_TF_4D
        assert result.n_timeframes == 2
        assert result.timeframe_names == ["5min", "15min"]

    def test_multi_stream_default_ohlcv(self, sample_df):
        """Test multi-stream defaults to OHLCV features."""
        from src.adapters import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            label_column="label_h20",
            sequence_length=30,
            timeframes=["5min"],
        )

        result = adapter.transform(sample_df)

        # Should use OHLCV by default
        assert result.n_features == 5

    def test_multi_stream_with_weights(self, sample_df, ohlcv_columns):
        """Test multi-stream preserves weights."""
        from src.adapters import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            feature_columns=ohlcv_columns,
            label_column="label_h20",
            weight_column="sample_weight_h20",
            sequence_length=30,
            timeframes=["5min"],
        )

        result = adapter.transform(sample_df)

        assert result.weights is not None
        assert result.weights.shape[0] == result.X.shape[0]

    def test_multi_stream_missing_tf_raises(self, sample_df, ohlcv_columns):
        """Test missing timeframe data raises error."""
        from src.adapters import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            feature_columns=ohlcv_columns,
            label_column="label_h20",
            sequence_length=30,
            timeframes=["5min", "15min", "60min"],
        )

        with pytest.raises((ValueError, FileNotFoundError)):
            adapter.transform(sample_df)  # No additional_dfs provided


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 adapters."""

    def test_adapter_for_all_tabular_models(self, sample_df, feature_columns):
        """Test adapter works for all tabular models."""
        from src.adapters import get_adapter
        from src.contracts import get_model_contract

        tabular_models = ["xgboost", "lightgbm", "random_forest", "logistic", "svm"]

        for model_name in tabular_models:
            contract = get_model_contract(model_name)
            assert contract.adapter_id == "tabular"

            adapter = get_adapter(
                model_name=model_name,
                feature_columns=feature_columns,
                label_column="label_h20",
            )

            # Transform without contract (contract enforces min features)
            result = adapter.transform(sample_df)
            assert result.X.ndim == 2
            assert result.data_rank == DataRank.TABULAR_2D

    def test_adapter_for_sequence_models(self, sample_df, feature_columns):
        """Test adapter works for sequence models."""
        from src.adapters import get_adapter
        from src.contracts import get_model_contract

        sequence_models = ["lstm", "gru", "tcn"]

        for model_name in sequence_models:
            contract = get_model_contract(model_name)
            assert contract.adapter_id == "sequence"

            adapter = get_adapter(
                model_name=model_name,
                feature_columns=feature_columns,
                label_column="label_h20",
            )

            result = adapter.transform(sample_df, model_contract=contract)
            assert result.X.ndim == 3
            assert result.data_rank == DataRank.SEQUENCE_3D

    def test_heterogeneous_ensemble_adapters(self, sample_df, feature_columns):
        """Test adapters for heterogeneous ensemble (Phase 7 prep)."""
        from src.adapters import get_adapter
        from src.contracts import get_model_contract

        # Simulate heterogeneous ensemble with different model types
        models = ["xgboost", "lstm"]
        results = {}

        for model_name in models:
            contract = get_model_contract(model_name)
            adapter = get_adapter(
                model_name=model_name,
                feature_columns=feature_columns,
                label_column="label_h20",
            )
            # Transform without contract (contract enforces min features)
            results[model_name] = adapter.transform(sample_df)

        # XGBoost should be 2D
        assert results["xgboost"].X.ndim == 2

        # LSTM should be 3D
        assert results["lstm"].X.ndim == 3

        # But y arrays should align
        assert results["xgboost"].y.shape[0] == 200
        # LSTM has fewer samples due to windowing
        assert results["lstm"].y.shape[0] < 200

    def test_adapter_result_validation_chain(self, sample_df, feature_columns):
        """Test full chain: transform -> validate -> use."""
        from src.adapters import get_adapter

        adapter = get_adapter(
            model_name="xgboost",
            feature_columns=feature_columns,
            label_column="label_h20",
            weight_column=None,  # Explicitly disable weights for test
        )

        result = adapter.transform(sample_df)

        # Validate result
        is_valid, issues = result.validate()
        assert is_valid, f"Validation failed: {issues}"

        # Check metadata
        meta = result.to_dict()
        assert meta["adapter_name"] == "tabular"
        assert meta["has_weights"] is False

        # Data contract should be set
        assert result.data_contract is not None
