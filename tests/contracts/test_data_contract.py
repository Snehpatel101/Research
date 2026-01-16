"""
Tests for DataContract, DataRank, FeatureMode, MTFMode.

Phase 0 SNwH implementation tests.
"""

import json
import pytest
import numpy as np
import pandas as pd

from src.contracts import (
    DataRank,
    FeatureMode,
    MTFMode,
    DataContract,
    DataContractSchema,
    DATA_SCHEMA,
)


class TestDataRank:
    """Tests for DataRank enum."""

    def test_values(self):
        """Test DataRank values."""
        assert DataRank.TABULAR_2D.value == 2
        assert DataRank.SEQUENCE_3D.value == 3
        assert DataRank.MULTI_TF_4D.value == 4

    def test_from_ndim(self):
        """Test creating DataRank from ndim."""
        assert DataRank.from_ndim(2) == DataRank.TABULAR_2D
        assert DataRank.from_ndim(3) == DataRank.SEQUENCE_3D
        assert DataRank.from_ndim(4) == DataRank.MULTI_TF_4D

    def test_from_ndim_invalid(self):
        """Test invalid ndim raises error."""
        with pytest.raises(ValueError, match="Unsupported array dimension"):
            DataRank.from_ndim(5)
        with pytest.raises(ValueError, match="Unsupported array dimension"):
            DataRank.from_ndim(1)


class TestFeatureMode:
    """Tests for FeatureMode enum."""

    def test_values(self):
        """Test FeatureMode values."""
        assert FeatureMode.ENGINEERED.value == "engineered"
        assert FeatureMode.RAW.value == "raw"
        assert FeatureMode.HYBRID.value == "hybrid"
        assert FeatureMode.OOF_PROBS.value == "oof_probs"

    def test_string_comparison(self):
        """Test FeatureMode string comparison."""
        assert FeatureMode.ENGINEERED == "engineered"
        assert FeatureMode.RAW == "raw"


class TestMTFMode:
    """Tests for MTFMode enum."""

    def test_values(self):
        """Test MTFMode values."""
        assert MTFMode.NONE.value == "none"
        assert MTFMode.INDICATORS.value == "indicators"
        assert MTFMode.MULTI_STREAM.value == "multi_stream"


class TestDataContractSchema:
    """Tests for DataContractSchema."""

    def test_required_ohlcv(self):
        """Test required OHLCV columns."""
        assert "timestamp" in DATA_SCHEMA.REQUIRED_OHLCV
        assert "open" in DATA_SCHEMA.REQUIRED_OHLCV
        assert "high" in DATA_SCHEMA.REQUIRED_OHLCV
        assert "low" in DATA_SCHEMA.REQUIRED_OHLCV
        assert "close" in DATA_SCHEMA.REQUIRED_OHLCV
        assert "volume" in DATA_SCHEMA.REQUIRED_OHLCV

    def test_label_column_patterns(self):
        """Test label column pattern generation."""
        assert DATA_SCHEMA.get_label_column(20) == "label_h20"
        assert DATA_SCHEMA.get_weight_column(20) == "sample_weight_h20"
        assert DATA_SCHEMA.get_label_end_time_column(20) == "label_end_time_h20"

    def test_valid_labels(self):
        """Test valid label values."""
        assert -1 in DATA_SCHEMA.VALID_LABELS
        assert 0 in DATA_SCHEMA.VALID_LABELS
        assert 1 in DATA_SCHEMA.VALID_LABELS
        assert DATA_SCHEMA.INVALID_LABEL_SENTINEL == -99

    def test_is_metadata_column(self):
        """Test metadata column detection."""
        assert DATA_SCHEMA.is_metadata_column("timestamp")
        assert DATA_SCHEMA.is_metadata_column("datetime")
        assert DATA_SCHEMA.is_metadata_column("label_h20")
        assert not DATA_SCHEMA.is_metadata_column("close")
        assert not DATA_SCHEMA.is_metadata_column("rsi_14")


class TestDataContract:
    """Tests for DataContract."""

    def test_basic_creation(self):
        """Test basic DataContract creation."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        assert contract.symbol == "MES"
        assert contract.timeframe == "15min"
        assert contract.horizon == 20
        assert contract.split == "train"
        assert contract.n_samples == 1000
        assert contract.n_features == 100
        assert contract.data_rank == DataRank.TABULAR_2D

    def test_auto_label_column(self):
        """Test automatic label column setting."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        assert contract.label_column == "label_h20"
        assert contract.weight_column == "sample_weight_h20"

    def test_schema_hash_deterministic(self):
        """Test schema hash is deterministic."""
        contract1 = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=["a", "b", "c"],
        )
        contract2 = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=["c", "a", "b"],  # Different order, same set
        )
        # Hash should be same for same schema
        assert contract1.schema_hash == contract2.schema_hash

    def test_schema_hash_different_for_different_schema(self):
        """Test schema hash differs for different schemas."""
        contract1 = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        contract2 = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=200,  # Different feature count
            data_rank=DataRank.TABULAR_2D,
        )
        assert contract1.schema_hash != contract2.schema_hash

    def test_from_array_2d(self):
        """Test creating contract from 2D array."""
        X = np.random.randn(1000, 100)
        contract = DataContract.from_array(
            X, symbol="MES", timeframe="15min", horizon=20, split="train"
        )
        assert contract.n_samples == 1000
        assert contract.n_features == 100
        assert contract.data_rank == DataRank.TABULAR_2D
        assert contract.sequence_length is None

    def test_from_array_3d(self):
        """Test creating contract from 3D array."""
        X = np.random.randn(1000, 60, 50)  # (samples, seq_len, features)
        contract = DataContract.from_array(
            X, symbol="MES", timeframe="5min", horizon=20, split="train"
        )
        assert contract.n_samples == 1000
        assert contract.sequence_length == 60
        assert contract.n_features == 50
        assert contract.data_rank == DataRank.SEQUENCE_3D

    def test_from_array_4d(self):
        """Test creating contract from 4D array."""
        X = np.random.randn(1000, 3, 60, 4)  # (samples, timeframes, seq_len, features)
        contract = DataContract.from_array(
            X, symbol="MES", timeframe="1min", horizon=20, split="train"
        )
        assert contract.n_samples == 1000
        assert contract.n_timeframes == 3
        assert contract.sequence_length == 60
        assert contract.n_features == 4
        assert contract.data_rank == DataRank.MULTI_TF_4D

    def test_from_array_invalid_dim(self):
        """Test invalid array dimension raises error."""
        X = np.random.randn(1000)  # 1D
        with pytest.raises(ValueError, match="Unsupported array rank"):
            DataContract.from_array(
                X, symbol="MES", timeframe="15min", horizon=20, split="train"
            )

    def test_from_dataframe(self):
        """Test creating contract from DataFrame."""
        df = pd.DataFrame(np.random.randn(1000, 100))
        feature_columns = [f"feature_{i}" for i in range(100)]
        contract = DataContract.from_dataframe(
            df,
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            feature_columns=feature_columns,
        )
        assert contract.n_samples == 1000
        assert contract.n_features == 100
        assert contract.data_rank == DataRank.TABULAR_2D

    def test_validate_array_2d_valid(self):
        """Test valid 2D array validation."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        X = np.random.randn(1000, 100)
        y = np.random.randint(0, 3, 1000)
        is_valid, issues = contract.validate_array(X, y)
        assert is_valid
        assert len(issues) == 0

    def test_validate_array_2d_wrong_samples(self):
        """Test 2D array validation with wrong sample count."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        X = np.random.randn(500, 100)  # Wrong sample count
        is_valid, issues = contract.validate_array(X)
        assert not is_valid
        assert any("Sample count mismatch" in issue for issue in issues)

    def test_validate_array_wrong_rank(self):
        """Test array validation with wrong rank."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        X = np.random.randn(1000, 60, 100)  # 3D instead of 2D
        is_valid, issues = contract.validate_array(X)
        assert not is_valid
        assert any("Rank mismatch" in issue for issue in issues)

    def test_validate_array_3d(self):
        """Test 3D array validation."""
        contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=50,
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )
        X = np.random.randn(1000, 60, 50)
        is_valid, issues = contract.validate_array(X)
        assert is_valid
        assert len(issues) == 0

    def test_validate_array_label_mismatch(self):
        """Test array validation with label count mismatch."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        X = np.random.randn(1000, 100)
        y = np.random.randint(0, 3, 500)  # Wrong label count
        is_valid, issues = contract.validate_array(X, y)
        assert not is_valid
        assert any("Label count mismatch" in issue for issue in issues)

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=["a", "b", "c"],
            feature_mode=FeatureMode.ENGINEERED,
            pipeline_run_id="test-run-123",
        )
        # Roundtrip
        data = contract.to_dict()
        contract2 = DataContract.from_dict(data)

        assert contract2.symbol == contract.symbol
        assert contract2.timeframe == contract.timeframe
        assert contract2.horizon == contract.horizon
        assert contract2.n_samples == contract.n_samples
        assert contract2.n_features == contract.n_features
        assert contract2.data_rank == contract.data_rank
        assert contract2.feature_columns == contract.feature_columns
        assert contract2.feature_mode == contract.feature_mode
        assert contract2.schema_hash == contract.schema_hash

    def test_to_dict_json_serializable(self):
        """Test to_dict produces JSON-serializable output."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        data = contract.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_repr(self):
        """Test string representation."""
        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        repr_str = repr(contract)
        assert "MES" in repr_str
        assert "15min" in repr_str
        assert "h20" in repr_str
        assert "train" in repr_str


class TestDataContractValidateDataFrame:
    """Tests for DataFrame validation."""

    def test_validate_dataframe_valid(self):
        """Test valid DataFrame validation."""
        feature_columns = [f"feature_{i}" for i in range(100)]
        df = pd.DataFrame(
            np.random.randn(1000, 100),
            columns=feature_columns,
        )
        df["label_h20"] = np.random.randint(0, 3, 1000)
        df["sample_weight_h20"] = np.ones(1000)

        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_columns,
        )
        is_valid, issues = contract.validate_dataframe(df)
        assert is_valid
        assert len(issues) == 0

    def test_validate_dataframe_missing_features(self):
        """Test DataFrame validation with missing features."""
        feature_columns = [f"feature_{i}" for i in range(100)]
        df = pd.DataFrame(
            np.random.randn(1000, 50),  # Only 50 columns
            columns=[f"feature_{i}" for i in range(50)],
        )
        df["label_h20"] = np.random.randint(0, 3, 1000)

        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_columns,
        )
        is_valid, issues = contract.validate_dataframe(df)
        assert not is_valid
        assert any("Missing feature columns" in issue for issue in issues)

    def test_validate_dataframe_missing_label(self):
        """Test DataFrame validation with missing label column."""
        feature_columns = [f"feature_{i}" for i in range(100)]
        df = pd.DataFrame(
            np.random.randn(1000, 100),
            columns=feature_columns,
        )
        # No label column

        contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_columns,
        )
        is_valid, issues = contract.validate_dataframe(df)
        assert not is_valid
        assert any("Missing label column" in issue for issue in issues)
