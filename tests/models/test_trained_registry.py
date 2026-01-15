"""Tests for trained model registry."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models.trained_registry import ModelQuery, TrainedModelEntry, TrainedModelRegistry


class TestModelQuery:
    """Tests for ModelQuery builder."""

    def test_empty_query_matches_all(self):
        """Test that empty query matches all entries."""
        query = ModelQuery()
        entry = {
            "model_name": "xgboost",
            "horizon": 20,
            "metrics": {"val_f1": 0.5},
        }
        assert query.matches(entry)

    def test_model_name_filter(self):
        """Test model name filtering."""
        query = ModelQuery().model_name("xgboost")

        assert query.matches({"model_name": "xgboost"})
        assert not query.matches({"model_name": "lightgbm"})

    def test_multiple_model_names(self):
        """Test filtering by multiple model names."""
        query = ModelQuery().model_name("xgboost", "lightgbm")

        assert query.matches({"model_name": "xgboost"})
        assert query.matches({"model_name": "lightgbm"})
        assert not query.matches({"model_name": "catboost"})

    def test_horizon_filter(self):
        """Test horizon filtering."""
        query = ModelQuery().horizon(20)

        assert query.matches({"horizon": 20})
        assert not query.matches({"horizon": 10})

    def test_min_metric_filter(self):
        """Test minimum metric filtering."""
        query = ModelQuery().min_metric("val_f1", 0.5)

        assert query.matches({"metrics": {"val_f1": 0.6}})
        assert query.matches({"metrics": {"val_f1": 0.5}})
        assert not query.matches({"metrics": {"val_f1": 0.4}})
        assert not query.matches({"metrics": {}})

    def test_max_metric_filter(self):
        """Test maximum metric filtering."""
        query = ModelQuery().max_metric("val_loss", 0.5)

        assert query.matches({"metrics": {"val_loss": 0.4}})
        assert query.matches({"metrics": {"val_loss": 0.5}})
        assert not query.matches({"metrics": {"val_loss": 0.6}})

    def test_date_filters(self):
        """Test date range filtering."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        query = ModelQuery().after(yesterday).before(tomorrow)

        assert query.matches({"train_date": now.isoformat()})
        assert not query.matches({"train_date": (now - timedelta(days=2)).isoformat()})

    def test_tag_filter(self):
        """Test tag filtering."""
        query = ModelQuery().with_tag("env", "production")

        assert query.matches({"tags": {"env": "production"}})
        assert not query.matches({"tags": {"env": "test"}})
        assert not query.matches({"tags": {}})

    def test_combined_filters(self):
        """Test combining multiple filters."""
        query = (
            ModelQuery()
            .model_name("xgboost")
            .horizon(20)
            .min_metric("val_f1", 0.5)
        )

        # All conditions met
        assert query.matches({
            "model_name": "xgboost",
            "horizon": 20,
            "metrics": {"val_f1": 0.6},
        })

        # Wrong model
        assert not query.matches({
            "model_name": "lightgbm",
            "horizon": 20,
            "metrics": {"val_f1": 0.6},
        })

        # Wrong horizon
        assert not query.matches({
            "model_name": "xgboost",
            "horizon": 10,
            "metrics": {"val_f1": 0.6},
        })

        # Metric too low
        assert not query.matches({
            "model_name": "xgboost",
            "horizon": 20,
            "metrics": {"val_f1": 0.4},
        })

    def test_fluent_interface(self):
        """Test fluent interface returns self."""
        query = ModelQuery()
        assert query.model_name("x") is query
        assert query.horizon(20) is query
        assert query.min_metric("m", 0.5) is query
        assert query.sort_by("m") is query
        assert query.limit(10) is query

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        original = (
            ModelQuery()
            .model_name("xgboost")
            .horizon(20)
            .min_metric("val_f1", 0.5)
            .sort_by("val_f1")
            .limit(10)
        )

        data = original.to_dict()
        restored = ModelQuery.from_dict(data)

        assert restored.model_names == original.model_names
        assert restored.horizons == original.horizons
        assert restored.min_metrics == original.min_metrics
        assert restored.sort_metric == original.sort_metric
        assert restored.result_limit == original.result_limit


class TestTrainedModelEntry:
    """Tests for TrainedModelEntry dataclass."""

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        entry = TrainedModelEntry(
            run_id="xgboost_h20_20240101_120000_abc123",
            model_name="xgboost",
            model_family="boosting",
            horizon=20,
            run_dir=Path("/tmp/test_run"),
            train_date=datetime(2024, 1, 1, 12, 0, 0),
            feature_set="boosting_optimal",
            metrics={"val_f1": 0.6, "val_accuracy": 0.7},
            tags={"env": "test"},
        )

        data = entry.to_dict()
        restored = TrainedModelEntry.from_dict(data)

        assert restored.run_id == entry.run_id
        assert restored.model_name == entry.model_name
        assert restored.model_family == entry.model_family
        assert restored.horizon == entry.horizon
        assert restored.feature_set == entry.feature_set
        assert restored.metrics == entry.metrics
        assert restored.tags == entry.tags

    def test_model_path_property(self):
        """Test model_path property."""
        entry = TrainedModelEntry(
            run_id="test",
            model_name="xgboost",
            model_family="boosting",
            horizon=20,
            run_dir=Path("/tmp/test_run"),
            train_date=datetime.now(),
        )
        assert entry.model_path == Path("/tmp/test_run/checkpoints")

    def test_is_valid(self, tmp_path):
        """Test is_valid check."""
        # Create valid directory structure
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()

        entry = TrainedModelEntry(
            run_id="test",
            model_name="xgboost",
            model_family="boosting",
            horizon=20,
            run_dir=run_dir,
            train_date=datetime.now(),
        )
        assert entry.is_valid()

        # Invalid if directory doesn't exist
        entry_invalid = TrainedModelEntry(
            run_id="test",
            model_name="xgboost",
            model_family="boosting",
            horizon=20,
            run_dir=Path("/nonexistent"),
            train_date=datetime.now(),
        )
        assert not entry_invalid.is_valid()


class TestTrainedModelRegistry:
    """Tests for TrainedModelRegistry."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create registry with temp index."""
        index_path = tmp_path / "model_index.json"
        return TrainedModelRegistry(index_path, auto_load=False)

    @pytest.fixture
    def sample_run_dir(self, tmp_path):
        """Create a sample run directory structure."""
        run_dir = tmp_path / "runs" / "xgboost_h20_20240101_120000_abc123"
        run_dir.mkdir(parents=True)

        # Create config
        config_dir = run_dir / "config"
        config_dir.mkdir()
        config = {
            "model_name": "xgboost",
            "horizon": 20,
            "feature_set": "boosting_optimal",
            "batch_size": 256,
        }
        with open(config_dir / "trainer_config.json", "w") as f:
            json.dump(config, f)

        # Create metrics
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir()
        metrics = {
            "accuracy": 0.7,
            "macro_f1": 0.65,
            "precision": 0.68,
            "recall": 0.62,
            "trading": {"win_rate": 0.55, "profit_factor": 1.2},
        }
        with open(metrics_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f)

        # Create checkpoints dir
        (run_dir / "checkpoints").mkdir()

        return run_dir

    def test_register_model(self, registry, sample_run_dir):
        """Test registering a model."""
        entry = registry.register(sample_run_dir)

        assert entry.model_name == "xgboost"
        assert entry.horizon == 20
        assert entry.feature_set == "boosting_optimal"
        assert entry.metrics["val_macro_f1"] == 0.65
        assert registry.count == 1

    def test_register_nonexistent_raises(self, registry):
        """Test registering nonexistent directory raises."""
        with pytest.raises(FileNotFoundError):
            registry.register(Path("/nonexistent"))

    def test_unregister_model(self, registry, sample_run_dir):
        """Test unregistering a model."""
        entry = registry.register(sample_run_dir)
        run_id = entry.run_id

        assert registry.count == 1
        assert registry.unregister(run_id)
        assert registry.count == 0
        assert not registry.unregister(run_id)  # Already removed

    def test_get_model(self, registry, sample_run_dir):
        """Test getting a specific model."""
        entry = registry.register(sample_run_dir)

        retrieved = registry.get(entry.run_id)
        assert retrieved is not None
        assert retrieved.model_name == "xgboost"

        assert registry.get("nonexistent") is None

    def test_query_all(self, registry, sample_run_dir):
        """Test querying all models."""
        registry.register(sample_run_dir)

        results = registry.query()
        assert len(results) == 1

    def test_query_with_filter(self, registry, sample_run_dir):
        """Test querying with filters."""
        registry.register(sample_run_dir)

        # Matching query
        query = ModelQuery().model_name("xgboost").horizon(20)
        results = registry.query(query)
        assert len(results) == 1

        # Non-matching query
        query = ModelQuery().model_name("lightgbm")
        results = registry.query(query)
        assert len(results) == 0

    def test_query_with_sorting(self, registry, tmp_path):
        """Test query with sorting."""
        # Create multiple runs
        for i, f1 in enumerate([0.5, 0.7, 0.6]):
            run_dir = tmp_path / "runs" / f"xgboost_run_{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "config").mkdir()
            (run_dir / "metrics").mkdir()
            (run_dir / "checkpoints").mkdir()

            with open(run_dir / "config" / "trainer_config.json", "w") as f:
                json.dump({"model_name": "xgboost", "horizon": 20}, f)
            with open(run_dir / "metrics" / "evaluation_metrics.json", "w") as f:
                json.dump({"macro_f1": f1}, f)

            registry.register(run_dir)

        # Sort by val_f1 descending
        query = ModelQuery().sort_by("val_macro_f1", descending=True)
        results = registry.query(query)

        assert len(results) == 3
        assert results[0].metrics["val_macro_f1"] == 0.7
        assert results[1].metrics["val_macro_f1"] == 0.6
        assert results[2].metrics["val_macro_f1"] == 0.5

    def test_query_with_limit(self, registry, tmp_path):
        """Test query with limit."""
        # Create multiple runs
        for i in range(5):
            run_dir = tmp_path / "runs" / f"xgboost_run_{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "config").mkdir()
            (run_dir / "metrics").mkdir()
            (run_dir / "checkpoints").mkdir()

            with open(run_dir / "config" / "trainer_config.json", "w") as f:
                json.dump({"model_name": "xgboost", "horizon": 20}, f)
            with open(run_dir / "metrics" / "evaluation_metrics.json", "w") as f:
                json.dump({"macro_f1": 0.5 + i * 0.1}, f)

            registry.register(run_dir)

        query = ModelQuery().sort_by("val_macro_f1", descending=True).limit(3)
        results = registry.query(query)

        assert len(results) == 3

    def test_get_best(self, registry, tmp_path):
        """Test getting best model."""
        # Create multiple runs with different metrics
        for i, f1 in enumerate([0.5, 0.8, 0.6]):
            run_dir = tmp_path / "runs" / f"xgboost_run_{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "config").mkdir()
            (run_dir / "metrics").mkdir()
            (run_dir / "checkpoints").mkdir()

            with open(run_dir / "config" / "trainer_config.json", "w") as f:
                json.dump({"model_name": "xgboost", "horizon": 20}, f)
            with open(run_dir / "metrics" / "evaluation_metrics.json", "w") as f:
                json.dump({"macro_f1": f1}, f)

            registry.register(run_dir)

        best = registry.get_best("xgboost", horizon=20, metric="val_macro_f1")

        assert best is not None
        assert best.metrics["val_macro_f1"] == 0.8

    def test_get_best_no_match(self, registry, sample_run_dir):
        """Test get_best returns None when no match."""
        registry.register(sample_run_dir)

        best = registry.get_best("nonexistent", horizon=20)
        assert best is None

    def test_list_by_model(self, registry, sample_run_dir):
        """Test listing by model name."""
        registry.register(sample_run_dir)

        results = registry.list_by_model("xgboost")
        assert len(results) == 1

        results = registry.list_by_model("lightgbm")
        assert len(results) == 0

    def test_list_by_horizon(self, registry, sample_run_dir):
        """Test listing by horizon."""
        registry.register(sample_run_dir)

        results = registry.list_by_horizon(20)
        assert len(results) == 1

        results = registry.list_by_horizon(10)
        assert len(results) == 0

    def test_rebuild_index(self, registry, tmp_path):
        """Test rebuilding index from runs directory."""
        runs_dir = tmp_path / "runs"

        # Create multiple valid runs
        for i in range(3):
            run_dir = runs_dir / f"xgboost_run_{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "config").mkdir()
            (run_dir / "metrics").mkdir()
            (run_dir / "checkpoints").mkdir()

            with open(run_dir / "config" / "trainer_config.json", "w") as f:
                json.dump({"model_name": "xgboost", "horizon": 20}, f)
            with open(run_dir / "metrics" / "evaluation_metrics.json", "w") as f:
                json.dump({"macro_f1": 0.5 + i * 0.1}, f)

        # Create invalid directory (no config)
        invalid_dir = runs_dir / "invalid_run"
        invalid_dir.mkdir()

        count = registry.rebuild_index(runs_dir)

        assert count == 3
        assert registry.count == 3

    def test_validate_and_prune(self, registry, sample_run_dir, tmp_path):
        """Test validation and pruning."""
        # Register valid model
        entry = registry.register(sample_run_dir)

        # Verify it's valid
        validation = registry.validate_entries()
        assert validation[entry.run_id] is True

        # Remove the directory
        import shutil
        shutil.rmtree(sample_run_dir)

        # Now it should be invalid
        validation = registry.validate_entries()
        assert validation[entry.run_id] is False

        # Prune invalid
        pruned = registry.prune_invalid()
        assert pruned == 1
        assert registry.count == 0

    def test_get_summary(self, registry, tmp_path):
        """Test getting summary statistics."""
        # Create multiple runs
        for model in ["xgboost", "xgboost", "lightgbm"]:
            for horizon in [10, 20]:
                run_dir = tmp_path / "runs" / f"{model}_h{horizon}_{datetime.now().timestamp()}"
                run_dir.mkdir(parents=True)
                (run_dir / "config").mkdir()
                (run_dir / "metrics").mkdir()
                (run_dir / "checkpoints").mkdir()

                with open(run_dir / "config" / "trainer_config.json", "w") as f:
                    json.dump({"model_name": model, "horizon": horizon}, f)
                with open(run_dir / "metrics" / "evaluation_metrics.json", "w") as f:
                    json.dump({"macro_f1": 0.6}, f)

                registry.register(run_dir)

        summary = registry.get_summary()

        assert summary["count"] == 6
        assert summary["models"]["xgboost"] == 4
        assert summary["models"]["lightgbm"] == 2
        assert summary["horizons"][10] == 3
        assert summary["horizons"][20] == 3

    def test_persistence(self, tmp_path, sample_run_dir):
        """Test that registry persists and loads correctly."""
        index_path = tmp_path / "model_index.json"

        # Create registry and add model
        registry1 = TrainedModelRegistry(index_path, auto_load=False)
        entry = registry1.register(sample_run_dir)

        # Create new registry and verify it loads
        registry2 = TrainedModelRegistry(index_path, auto_load=True)

        assert registry2.count == 1
        loaded = registry2.get(entry.run_id)
        assert loaded is not None
        assert loaded.model_name == "xgboost"
