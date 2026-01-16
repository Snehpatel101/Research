"""
Tests for feature store lineage tracking module.

Tests cover:
- DataSource creation and serialization
- Transformation recording
- FeatureLineage management
- LineageTracker operations
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.feature_store.lineage import (
    DataSource,
    FeatureLineage,
    LineageTracker,
    Transformation,
    TransformationType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_lineage_dir(tmp_path):
    """Create a temporary lineage directory."""
    lineage_dir = tmp_path / "lineage"
    lineage_dir.mkdir()
    return lineage_dir


@pytest.fixture
def sample_data_source():
    """Create a sample DataSource."""
    return DataSource(
        path="/data/raw/MES_1m.parquet",
        checksum="abc123def456",
        symbol="MES",
        timeframe="1min",
        row_count=100000,
        date_start=datetime(2024, 1, 1),
        date_end=datetime(2024, 6, 30),
    )


# =============================================================================
# TRANSFORMATION TYPE TESTS
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_types_exist(self):
        """Test that all expected transformation types exist."""
        assert TransformationType.INGESTION.value == "ingestion"
        assert TransformationType.CLEANING.value == "cleaning"
        assert TransformationType.RESAMPLING.value == "resampling"
        assert TransformationType.FEATURE_ENGINEERING.value == "feature_engineering"
        assert TransformationType.MTF_AGGREGATION.value == "mtf_aggregation"
        assert TransformationType.LABELING.value == "labeling"
        assert TransformationType.SCALING.value == "scaling"
        assert TransformationType.SELECTION.value == "selection"

    def test_type_from_string(self):
        """Test creating enum from string."""
        t = TransformationType("feature_engineering")
        assert t == TransformationType.FEATURE_ENGINEERING


# =============================================================================
# DATA SOURCE TESTS
# =============================================================================


class TestDataSource:
    """Tests for DataSource dataclass."""

    def test_create_data_source(self):
        """Test creating a data source."""
        source = DataSource(
            path="/data/raw/MES_1m.parquet",
            checksum="hash123",
            symbol="MES",
            timeframe="1min",
            row_count=50000,
            date_start=datetime(2024, 1, 1),
            date_end=datetime(2024, 3, 31),
        )

        assert source.path == "/data/raw/MES_1m.parquet"
        assert source.symbol == "MES"
        assert source.row_count == 50000

    def test_to_dict(self, sample_data_source):
        """Test serialization to dict."""
        d = sample_data_source.to_dict()

        assert d["path"] == "/data/raw/MES_1m.parquet"
        assert d["symbol"] == "MES"
        assert "created_at" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "path": "/data/test.parquet",
            "checksum": "xyz789",
            "symbol": "MGC",
            "timeframe": "5min",
            "row_count": 10000,
            "date_start": "2024-02-01T00:00:00",
            "date_end": "2024-02-28T23:59:59",
            "created_at": "2024-03-01T12:00:00",
            "metadata": {"extra": "info"},
        }

        source = DataSource.from_dict(data)

        assert source.symbol == "MGC"
        assert source.timeframe == "5min"
        assert source.metadata == {"extra": "info"}

    def test_roundtrip_serialization(self, sample_data_source):
        """Test to_dict/from_dict roundtrip."""
        restored = DataSource.from_dict(sample_data_source.to_dict())

        assert restored.path == sample_data_source.path
        assert restored.checksum == sample_data_source.checksum
        assert restored.symbol == sample_data_source.symbol


# =============================================================================
# TRANSFORMATION TESTS
# =============================================================================


class TestTransformation:
    """Tests for Transformation dataclass."""

    def test_create_transformation(self):
        """Test creating a transformation."""
        t = Transformation(
            type=TransformationType.FEATURE_ENGINEERING,
            name="add_rsi",
            config={"period": 14},
            config_hash="hash123",
            input_columns=["close"],
            output_columns=["rsi_14"],
            duration_seconds=0.5,
        )

        assert t.type == TransformationType.FEATURE_ENGINEERING
        assert t.name == "add_rsi"
        assert t.output_columns == ["rsi_14"]

    def test_to_dict(self):
        """Test serialization to dict."""
        t = Transformation(
            type=TransformationType.SCALING,
            name="robust_scaler",
            config={"quantile_range": (25, 75)},
            config_hash="scale_hash",
            input_columns=["feature_1", "feature_2"],
            output_columns=["feature_1", "feature_2"],
        )

        d = t.to_dict()

        assert d["type"] == "scaling"
        assert d["name"] == "robust_scaler"
        assert "executed_at" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "type": "labeling",
            "name": "triple_barrier",
            "config": {"horizon": 20},
            "config_hash": "label_hash",
            "input_columns": ["close", "high", "low"],
            "output_columns": ["label", "return"],
            "executed_at": "2024-01-15T10:00:00",
            "duration_seconds": 2.5,
            "metadata": {"num_classes": 3},
        }

        t = Transformation.from_dict(data)

        assert t.type == TransformationType.LABELING
        assert t.name == "triple_barrier"
        assert t.duration_seconds == 2.5


# =============================================================================
# FEATURE LINEAGE TESTS
# =============================================================================


class TestFeatureLineage:
    """Tests for FeatureLineage dataclass."""

    @pytest.fixture
    def sample_lineage(self):
        """Create a sample lineage."""
        return FeatureLineage(
            feature_set="core_features",
            version="1.0.0",
            symbol="MES",
        )

    def test_create_lineage(self, sample_lineage):
        """Test creating a lineage."""
        assert sample_lineage.feature_set == "core_features"
        assert sample_lineage.version == "1.0.0"
        assert sample_lineage.symbol == "MES"
        assert sample_lineage.source is None
        assert len(sample_lineage.transformations) == 0

    def test_lineage_id_deterministic(self, sample_lineage):
        """Test that lineage ID is deterministic."""
        id1 = sample_lineage.lineage_id
        id2 = sample_lineage.lineage_id

        assert id1 == id2
        assert len(id1) == 16

    def test_lineage_id_unique(self):
        """Test that different lineages have different IDs."""
        l1 = FeatureLineage(feature_set="core", version="1.0.0", symbol="MES")
        l2 = FeatureLineage(feature_set="core", version="1.0.0", symbol="MGC")
        l3 = FeatureLineage(feature_set="mtf", version="1.0.0", symbol="MES")

        assert l1.lineage_id != l2.lineage_id
        assert l1.lineage_id != l3.lineage_id

    def test_set_source(self, sample_lineage):
        """Test setting data source."""
        source = sample_lineage.set_source(
            path="/data/raw/MES_1m.parquet",
            checksum="hash123",
            symbol="MES",
            timeframe="1min",
            row_count=100000,
            date_start=datetime(2024, 1, 1),
            date_end=datetime(2024, 6, 30),
        )

        assert sample_lineage.source is not None
        assert source.path == "/data/raw/MES_1m.parquet"

    def test_add_transformation(self, sample_lineage):
        """Test adding transformations."""
        t = sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING,
            name="add_indicators",
            config={"indicators": ["rsi", "macd"]},
            input_columns=["close"],
            output_columns=["rsi_14", "macd", "macd_signal"],
        )

        assert len(sample_lineage.transformations) == 1
        assert t.name == "add_indicators"

    def test_add_multiple_transformations(self, sample_lineage):
        """Test adding multiple transformations in sequence."""
        sample_lineage.add_transformation(
            type=TransformationType.CLEANING,
            name="fill_gaps",
            config={},
            input_columns=["close"],
            output_columns=["close"],
        )
        sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING,
            name="add_sma",
            config={"periods": [20, 50]},
            input_columns=["close"],
            output_columns=["sma_20", "sma_50"],
        )
        sample_lineage.add_transformation(
            type=TransformationType.SCALING,
            name="robust_scale",
            config={},
            input_columns=["sma_20", "sma_50"],
            output_columns=["sma_20", "sma_50"],
        )

        assert len(sample_lineage.transformations) == 3

    def test_get_transformation_chain(self, sample_lineage):
        """Test getting transformation chain."""
        sample_lineage.add_transformation(
            type=TransformationType.CLEANING, name="clean", config={},
            input_columns=[], output_columns=[]
        )
        sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING, name="engineer", config={},
            input_columns=[], output_columns=[]
        )
        sample_lineage.add_transformation(
            type=TransformationType.SCALING, name="scale", config={},
            input_columns=[], output_columns=[]
        )

        chain = sample_lineage.get_transformation_chain()

        assert chain == ["clean", "engineer", "scale"]

    def test_get_all_output_columns(self, sample_lineage):
        """Test getting all output columns."""
        sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING, name="t1", config={},
            input_columns=[], output_columns=["a", "b"]
        )
        sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING, name="t2", config={},
            input_columns=[], output_columns=["b", "c", "d"]
        )

        columns = sample_lineage.get_all_output_columns()

        assert columns == {"a", "b", "c", "d"}

    def test_to_dict(self, sample_lineage):
        """Test serialization to dict."""
        sample_lineage.set_source(
            path="/data/test.parquet",
            checksum="hash",
            symbol="MES",
            timeframe="5min",
            row_count=1000,
            date_start=datetime(2024, 1, 1),
            date_end=datetime(2024, 1, 31),
        )
        sample_lineage.add_transformation(
            type=TransformationType.FEATURE_ENGINEERING,
            name="test",
            config={"param": 1},
            input_columns=["in"],
            output_columns=["out"],
        )

        d = sample_lineage.to_dict()

        assert "lineage_id" in d
        assert d["feature_set"] == "core_features"
        assert d["source"] is not None
        assert len(d["transformations"]) == 1

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "lineage_id": "test123",
            "feature_set": "mtf_features",
            "version": "2.0.0",
            "symbol": "MGC",
            "source": {
                "path": "/data/raw.parquet",
                "checksum": "hash",
                "symbol": "MGC",
                "timeframe": "1min",
                "row_count": 5000,
                "date_start": "2024-01-01T00:00:00",
                "date_end": "2024-01-31T23:59:59",
                "created_at": "2024-02-01T00:00:00",
                "metadata": {},
            },
            "transformations": [
                {
                    "type": "feature_engineering",
                    "name": "test",
                    "config": {},
                    "config_hash": "hash",
                    "input_columns": [],
                    "output_columns": ["out"],
                    "executed_at": "2024-02-01T12:00:00",
                    "duration_seconds": 1.0,
                    "metadata": {},
                }
            ],
            "output_columns": ["out"],
            "output_checksum": "checksum",
            "created_at": "2024-02-01T00:00:00",
            "parent_lineages": [],
            "metadata": {},
        }

        lineage = FeatureLineage.from_dict(data)

        assert lineage.feature_set == "mtf_features"
        assert lineage.source is not None
        assert len(lineage.transformations) == 1


# =============================================================================
# LINEAGE TRACKER TESTS
# =============================================================================


class TestLineageTracker:
    """Tests for LineageTracker class."""

    def test_create_tracker_in_memory(self):
        """Test creating an in-memory tracker."""
        tracker = LineageTracker()

        assert len(tracker) == 0

    def test_create_tracker_with_storage(self, temp_lineage_dir):
        """Test creating a tracker with storage."""
        tracker = LineageTracker(storage_path=temp_lineage_dir)

        assert len(tracker) == 0

    def test_create_lineage(self):
        """Test creating a lineage through tracker."""
        tracker = LineageTracker()

        lineage = tracker.create_lineage(
            feature_set="core",
            version="1.0.0",
            symbol="MES",
        )

        assert lineage.feature_set == "core"
        assert len(tracker) == 1

    def test_get_lineage(self):
        """Test getting lineage by ID."""
        tracker = LineageTracker()
        lineage = tracker.create_lineage(
            feature_set="core", version="1.0.0", symbol="MES"
        )

        retrieved = tracker.get_lineage(lineage.lineage_id)

        assert retrieved is not None
        assert retrieved.feature_set == "core"

    def test_get_lineage_nonexistent(self):
        """Test getting nonexistent lineage returns None."""
        tracker = LineageTracker()
        result = tracker.get_lineage("nonexistent")

        assert result is None

    def test_get_lineage_for_feature_set(self):
        """Test getting lineage by feature set/version/symbol."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MGC")

        lineage = tracker.get_lineage_for_feature_set(
            feature_set="core", version="1.0.0", symbol="MES"
        )

        assert lineage is not None
        assert lineage.symbol == "MES"

    def test_save_lineage_in_memory(self):
        """Test saving lineage without storage (in-memory)."""
        tracker = LineageTracker()
        lineage = tracker.create_lineage(
            feature_set="core", version="1.0.0", symbol="MES"
        )

        result = tracker.save_lineage(lineage)

        assert result is None  # No storage path

    def test_save_lineage_with_storage(self, temp_lineage_dir):
        """Test saving lineage to storage."""
        tracker = LineageTracker(storage_path=temp_lineage_dir)
        lineage = tracker.create_lineage(
            feature_set="core", version="1.0.0", symbol="MES"
        )

        result = tracker.save_lineage(lineage)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".json"

    def test_find_lineages_all(self):
        """Test finding all lineages."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="mtf", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MGC")

        results = tracker.find_lineages()

        assert len(results) == 3

    def test_find_lineages_by_feature_set(self):
        """Test finding lineages by feature set."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="mtf", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MGC")

        results = tracker.find_lineages(feature_set="core")

        assert len(results) == 2

    def test_find_lineages_by_symbol(self):
        """Test finding lineages by symbol."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MGC")

        results = tracker.find_lineages(symbol="MES")

        assert len(results) == 1
        assert results[0].symbol == "MES"

    def test_find_lineages_combined_filters(self):
        """Test finding lineages with multiple filters."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="2.0.0", symbol="MES")
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MGC")

        results = tracker.find_lineages(feature_set="core", version="1.0.0")

        assert len(results) == 2

    def test_iteration(self):
        """Test iteration over lineages."""
        tracker = LineageTracker()
        tracker.create_lineage(feature_set="core", version="1.0.0", symbol="MES")
        tracker.create_lineage(feature_set="mtf", version="1.0.0", symbol="MES")

        lineages = list(tracker)

        assert len(lineages) == 2
        assert all(isinstance(l, FeatureLineage) for l in lineages)

    def test_parent_child_relationships(self):
        """Test parent-child lineage relationships."""
        tracker = LineageTracker()

        parent = tracker.create_lineage(
            feature_set="raw", version="1.0.0", symbol="MES"
        )
        child = tracker.create_lineage(
            feature_set="features",
            version="1.0.0",
            symbol="MES",
            parent_lineages=[parent.lineage_id],
        )

        downstream = tracker.get_downstream_lineages(parent.lineage_id)
        upstream = tracker.get_upstream_lineages(child.lineage_id)

        assert len(downstream) == 1
        assert downstream[0].lineage_id == child.lineage_id
        assert len(upstream) == 1
        assert upstream[0].lineage_id == parent.lineage_id

    def test_load_existing_lineages(self, temp_lineage_dir):
        """Test loading existing lineages from storage."""
        # Create and save lineages
        tracker1 = LineageTracker(storage_path=temp_lineage_dir)
        lineage = tracker1.create_lineage(
            feature_set="core", version="1.0.0", symbol="MES"
        )
        tracker1.save_lineage(lineage)

        # Create new tracker pointing to same storage
        tracker2 = LineageTracker(storage_path=temp_lineage_dir)

        assert len(tracker2) == 1
        assert tracker2.get_lineage(lineage.lineage_id) is not None
