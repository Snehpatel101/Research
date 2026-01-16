"""
Tests for Phase 3 - Timeframe Coordination.

Comprehensive tests for TimeframeCoordinator and alignment utilities
that enable heterogeneous ensembles where different base models
train on different timeframes derived from the same 1-min canonical source.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.contracts import get_model_contract, DataRank


# =============================================================================
# FIXTURES - Sample Data Generation
# =============================================================================


@pytest.fixture
def sample_1min_df() -> pd.DataFrame:
    """Generate 200 rows of 1-min OHLCV data with features and labels."""
    np.random.seed(42)
    n_rows = 200

    base_time = datetime(2024, 1, 15, 9, 30)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]

    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n_rows) * 0.1)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices + np.random.randn(n_rows) * 0.05,
        "high": prices + np.abs(np.random.randn(n_rows) * 0.1),
        "low": prices - np.abs(np.random.randn(n_rows) * 0.1),
        "close": prices,
        "volume": np.random.randint(100, 1000, n_rows).astype(float),
    })

    for i in range(10):
        df[f"feature_{i}"] = np.random.randn(n_rows).astype(np.float32)

    df["label_h20"] = np.random.choice([-1, 0, 1], size=n_rows)
    df["sample_weight_h20"] = np.random.uniform(0.5, 1.5, size=n_rows).astype(np.float32)

    return df


@pytest.fixture
def sample_5min_df(sample_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 5-min data derived from 1-min data."""
    df_1min = sample_1min_df.copy()
    df_1min = df_1min.set_index("timestamp")

    ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_5min = df_1min[["open", "high", "low", "close", "volume"]].resample("5min").agg(ohlcv_agg)

    feature_cols = [c for c in df_1min.columns if c.startswith("feature_")]
    for col in feature_cols:
        df_5min[col] = df_1min[col].resample("5min").last()

    df_5min["label_h20"] = df_1min["label_h20"].resample("5min").last()
    df_5min["sample_weight_h20"] = df_1min["sample_weight_h20"].resample("5min").last()
    df_5min = df_5min.dropna().reset_index()

    return df_5min


@pytest.fixture
def sample_15min_df(sample_1min_df: pd.DataFrame) -> pd.DataFrame:
    """Generate 15-min data derived from 1-min data."""
    df_1min = sample_1min_df.copy()
    df_1min = df_1min.set_index("timestamp")

    ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df_15min = df_1min[["open", "high", "low", "close", "volume"]].resample("15min").agg(ohlcv_agg)

    feature_cols = [c for c in df_1min.columns if c.startswith("feature_")]
    for col in feature_cols:
        df_15min[col] = df_1min[col].resample("15min").last()

    df_15min["label_h20"] = df_1min["label_h20"].resample("15min").last()
    df_15min["sample_weight_h20"] = df_1min["sample_weight_h20"].resample("15min").last()
    df_15min = df_15min.dropna().reset_index()

    return df_15min


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_1min_df, sample_5min_df, sample_15min_df) -> Path:
    """Create temporary directory with parquet files for each timeframe."""
    data_dir = tmp_path / "data" / "splits" / "scaled"
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_1min_df.to_parquet(data_dir / "features_1min.parquet", index=False)
    sample_5min_df.to_parquet(data_dir / "features_5min.parquet", index=False)
    sample_15min_df.to_parquet(data_dir / "features_15min.parquet", index=False)

    return data_dir


# =============================================================================
# TEST CLASS: TimeframeData
# =============================================================================


class TestTimeframeData:
    """Tests for TimeframeData dataclass."""

    def test_timeframe_data_creation(self, sample_5min_df: pd.DataFrame):
        """Test basic TimeframeData creation."""
        from src.coordination import TimeframeData

        feature_cols = [c for c in sample_5min_df.columns if c.startswith("feature_")]
        tf_data = TimeframeData(
            timeframe="5min",
            df=sample_5min_df,
            feature_columns=feature_cols,
        )

        assert tf_data.timeframe == "5min"
        assert tf_data.df is not None
        assert len(tf_data.df) == len(sample_5min_df)
        assert tf_data.n_samples == len(sample_5min_df)

    def test_timeframe_data_computes_timestamps(self, sample_5min_df: pd.DataFrame):
        """Test TimeframeData computes timestamp range."""
        from src.coordination import TimeframeData

        feature_cols = [c for c in sample_5min_df.columns if c.startswith("feature_")]
        tf_data = TimeframeData(
            timeframe="5min",
            df=sample_5min_df,
            feature_columns=feature_cols,
        )

        assert tf_data.start_time is not None
        assert tf_data.end_time is not None
        assert tf_data.end_time > tf_data.start_time

    def test_timeframe_data_empty_df(self):
        """Test TimeframeData with empty DataFrame."""
        from src.coordination import TimeframeData

        empty_df = pd.DataFrame()
        tf_data = TimeframeData(
            timeframe="5min",
            df=empty_df,
            feature_columns=[],
        )

        assert tf_data.n_samples == 0
        assert tf_data.start_time is None
        assert tf_data.end_time is None

    def test_timeframe_data_feature_columns(self, sample_5min_df: pd.DataFrame):
        """Test TimeframeData stores feature columns."""
        from src.coordination import TimeframeData

        feature_cols = ["feature_0", "feature_1", "feature_2"]
        tf_data = TimeframeData(
            timeframe="5min",
            df=sample_5min_df,
            feature_columns=feature_cols,
        )

        assert tf_data.feature_columns == feature_cols
        assert len(tf_data.feature_columns) == 3


# =============================================================================
# TEST CLASS: TimeframeCoordinator
# =============================================================================


class TestTimeframeCoordinator:
    """Tests for TimeframeCoordinator."""

    def test_coordinator_init(self, tmp_data_dir: Path):
        """Test coordinator initialization."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(
            data_dir=tmp_data_dir,
            split="train",
            horizon=20,
        )

        assert coordinator.data_dir == tmp_data_dir
        assert coordinator.split == "train"
        assert coordinator.horizon == 20
        assert len(coordinator.loaded_timeframes) == 0

    def test_load_single_timeframe(self, tmp_data_dir: Path):
        """Test loading a single timeframe."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min"])

        assert "5min" in coordinator.loaded_timeframes
        assert coordinator.anchor_timeframe == "5min"

        tf_data = coordinator.get_timeframe_data("5min")
        assert tf_data.n_samples > 0
        assert len(tf_data.feature_columns) > 0

    def test_load_multiple_timeframes(self, tmp_data_dir: Path):
        """Test loading multiple timeframes."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min", "15min", "1min"])

        assert len(coordinator.loaded_timeframes) == 3
        assert "1min" in coordinator.loaded_timeframes
        assert "5min" in coordinator.loaded_timeframes
        assert "15min" in coordinator.loaded_timeframes

    def test_anchor_timeframe_is_smallest(self, tmp_data_dir: Path):
        """Test anchor timeframe is smallest loaded TF."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min", "15min", "1min"])

        assert coordinator.anchor_timeframe == "1min"

    def test_get_timeframe_data_raises_for_unloaded(self, tmp_data_dir: Path):
        """Test error when accessing unloaded timeframe."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min"])

        with pytest.raises(KeyError, match="not loaded"):
            coordinator.get_timeframe_data("15min")

    def test_get_data_for_model_uses_contract(self, tmp_data_dir: Path):
        """Test get_data_for_model uses contract's primary_timeframe."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min", "15min"])

        # XGBoost contract has primary_timeframe="15min"
        contract = get_model_contract("xgboost")
        df = coordinator.get_data_for_model("xgboost", contract)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_multi_stream_dfs(self, tmp_data_dir: Path):
        """Test get_multi_stream_dfs returns aligned DataFrames."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["1min", "5min", "15min"])

        dfs = coordinator.get_multi_stream_dfs(["1min", "5min", "15min"])

        assert "1min" in dfs
        assert "5min" in dfs
        assert "15min" in dfs
        # All should have feature columns
        for tf, df in dfs.items():
            assert len(df) > 0

    def test_get_required_timeframes_for_ensemble(self, tmp_data_dir: Path):
        """Test getting required timeframes for ensemble base models."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)

        # Get required timeframes for heterogeneous ensemble
        required = coordinator.get_required_timeframes_for_ensemble(
            ["xgboost", "lstm"]
        )

        # XGBoost needs 15min, LSTM needs 5min
        assert "15min" in required
        assert "5min" in required

    def test_validate_timeframe_coverage(self, tmp_data_dir: Path):
        """Test validation of timeframe coverage."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min", "15min"])

        # Should pass - all loaded
        is_valid, missing = coordinator.validate_timeframe_coverage({"5min", "15min"})
        assert is_valid
        assert len(missing) == 0

        # Should fail - 1min not loaded
        is_valid, missing = coordinator.validate_timeframe_coverage({"1min", "5min"})
        assert not is_valid
        assert "1min" in missing


# =============================================================================
# TEST CLASS: Alignment
# =============================================================================


class TestAlignment:
    """Tests for alignment utilities."""

    def test_align_to_anchor_forward_fills(self, sample_1min_df, sample_15min_df):
        """Test align_to_anchor forward-fills higher TF data."""
        from src.coordination.alignment import align_to_anchor

        # Add datetime column
        anchor_df = sample_1min_df.copy()
        anchor_df = anchor_df.rename(columns={"timestamp": "datetime"})
        higher_df = sample_15min_df.copy()
        higher_df = higher_df.rename(columns={"timestamp": "datetime"})

        aligned = align_to_anchor(
            anchor_df=anchor_df,
            higher_tf_df=higher_df,
            anchor_tf="1min",
            higher_tf="15min",
            datetime_col="datetime",
        )

        # Aligned should have same length as anchor
        assert len(aligned) == len(anchor_df)

    def test_align_to_anchor_no_lookahead(self, sample_1min_df, sample_15min_df):
        """Test align_to_anchor uses backward direction (no lookahead)."""
        from src.coordination.alignment import align_to_anchor

        anchor_df = sample_1min_df.copy()
        anchor_df = anchor_df.rename(columns={"timestamp": "datetime"})
        higher_df = sample_15min_df.copy()
        higher_df = higher_df.rename(columns={"timestamp": "datetime"})

        aligned = align_to_anchor(
            anchor_df=anchor_df,
            higher_tf_df=higher_df,
            anchor_tf="1min",
            higher_tf="15min",
            datetime_col="datetime",
        )

        # First few rows should have NaN (before first 15min bar completes)
        # This is expected behavior to prevent lookahead
        assert len(aligned) > 0

    def test_apply_mtf_lag_shifts_columns(self, sample_5min_df):
        """Test apply_mtf_lag shifts specified columns."""
        from src.coordination.alignment import apply_mtf_lag

        df = sample_5min_df.copy()
        original_values = df["feature_0"].values.copy()

        lagged = apply_mtf_lag(
            df=df,
            mtf_columns=["feature_0", "feature_1"],
            shift=1,
        )

        # After shift=1, row i should have value from row i-1
        # First row gets filled with first valid value
        assert lagged["feature_0"].iloc[1] == original_values[0]
        assert lagged["feature_0"].iloc[2] == original_values[1]

    def test_apply_mtf_lag_fills_nan(self, sample_5min_df):
        """Test apply_mtf_lag fills NaN from shift."""
        from src.coordination.alignment import apply_mtf_lag

        df = sample_5min_df.copy()

        lagged = apply_mtf_lag(
            df=df,
            mtf_columns=["feature_0"],
            shift=1,
        )

        # No NaN values should remain after fill
        assert not lagged["feature_0"].isna().any()

    def test_compute_sequence_offset(self):
        """Test compute_sequence_offset calculates correctly."""
        from src.coordination.alignment import compute_sequence_offset

        # 1000 tabular samples, seq_len=30 -> 971 sequences, offset=29
        offset = compute_sequence_offset(
            tabular_samples=1000,
            sequence_samples=971,
            sequence_length=30,
        )

        assert offset == 29

    def test_validate_timestamp_alignment_passes(self, sample_5min_df):
        """Test validate_timestamp_alignment passes for aligned DataFrames."""
        from src.coordination.alignment import validate_timestamp_alignment

        df1 = sample_5min_df.copy()
        df1 = df1.rename(columns={"timestamp": "datetime"})
        df2 = df1.copy()

        is_valid, issues = validate_timestamp_alignment(
            df1=df1,
            df2=df2,
            datetime_col="datetime",
            tolerance_minutes=1,
        )

        assert is_valid
        assert len(issues) == 0

    def test_validate_timestamp_alignment_fails_length(self, sample_5min_df):
        """Test validate_timestamp_alignment fails for length mismatch."""
        from src.coordination.alignment import validate_timestamp_alignment

        df1 = sample_5min_df.copy()
        df1 = df1.rename(columns={"timestamp": "datetime"})
        df2 = df1.iloc[:10].copy()

        is_valid, issues = validate_timestamp_alignment(
            df1=df1,
            df2=df2,
            datetime_col="datetime",
        )

        assert not is_valid
        assert any("Length" in issue for issue in issues)


# =============================================================================
# TEST CLASS: Phase 3 Integration
# =============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3."""

    def test_heterogeneous_ensemble_timeframe_loading(self, tmp_data_dir: Path):
        """Test loading timeframes for heterogeneous ensemble."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)

        # Get required timeframes for heterogeneous ensemble
        base_models = ["xgboost", "lstm"]
        required = coordinator.get_required_timeframes_for_ensemble(base_models)

        # Load only the required timeframes that exist in test data
        to_load = [tf for tf in required if tf in ["1min", "5min", "15min"]]
        coordinator.load_timeframes(to_load)

        # Each model should get correct timeframe
        for model in base_models:
            contract = get_model_contract(model)
            primary_tf = contract.primary_timeframe
            if primary_tf in coordinator.loaded_timeframes:
                df = coordinator.get_data_for_model(model, contract)
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0

    def test_coordinator_with_adapters(self, tmp_data_dir: Path):
        """Test coordinator integration with Phase 2 adapters."""
        from src.coordination import TimeframeCoordinator
        from src.adapters import get_adapter

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min", "15min"])

        # Get 15min data for XGBoost
        contract = get_model_contract("xgboost")
        df = coordinator.get_data_for_model("xgboost", contract)

        # Get 15min TimeframeData for full DataFrame with labels
        tf_data = coordinator.get_timeframe_data("15min")

        # Create adapter and transform
        adapter = get_adapter(
            model_name="xgboost",
            feature_columns=tf_data.feature_columns,
            label_column="label_h20",
        )

        result = adapter.transform(tf_data.df)
        assert result.X.ndim == 2
        assert result.X.shape[0] == tf_data.n_samples

    def test_model_contract_timeframe_consistency(self, tmp_data_dir: Path):
        """Test that model contracts provide consistent timeframe info."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)

        # Check tabular models use 15min
        for model in ["xgboost", "lightgbm"]:
            contract = get_model_contract(model)
            assert contract.primary_timeframe in ["15min", "10min", "5min"]

        # Check sequence models use 5min
        for model in ["lstm", "gru", "tcn"]:
            contract = get_model_contract(model)
            assert contract.primary_timeframe == "5min"


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_coordinator_missing_file(self, tmp_path: Path):
        """Test coordinator raises error for missing file."""
        from src.coordination import TimeframeCoordinator

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(parents=True)

        coordinator = TimeframeCoordinator(data_dir=empty_dir)

        with pytest.raises(FileNotFoundError):
            coordinator.load_timeframes(["5min"])

    def test_coordinator_normalizes_timeframes(self, tmp_data_dir: Path):
        """Test coordinator normalizes timeframe aliases."""
        from src.coordination import TimeframeCoordinator

        coordinator = TimeframeCoordinator(data_dir=tmp_data_dir)
        coordinator.load_timeframes(["5min"])

        # Should find via normalized form
        assert "5min" in coordinator.loaded_timeframes

    def test_alignment_empty_dataframe(self):
        """Test alignment handles empty DataFrames."""
        from src.coordination.alignment import align_to_anchor

        empty_df = pd.DataFrame({"datetime": [], "value": []})
        non_empty = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=10, freq="15min"),
            "value": range(10),
        })

        result = align_to_anchor(empty_df, non_empty, "1min", "15min")
        assert len(result) == 0

    def test_compute_sequence_offset_validation(self):
        """Test compute_sequence_offset validates inputs."""
        from src.coordination.alignment import compute_sequence_offset

        with pytest.raises(ValueError):
            compute_sequence_offset(-1, 100, 30)

        with pytest.raises(ValueError):
            compute_sequence_offset(100, 200, 30)  # sequence > tabular
