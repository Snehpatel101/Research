"""Tests for experiment tracking module."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.tracking import (
    ExperimentTracker,
    LocalTracker,
    TrackerConfig,
    get_tracker,
)
from src.models.tracking.base import DisabledTracker


class TestTrackerConfig:
    """Tests for TrackerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrackerConfig()
        assert config.enabled is True
        assert config.backend == "local"
        assert config.log_artifacts is True
        assert config.log_metrics_every_n_epochs == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrackerConfig(
            backend="mlflow",
            experiment_name="test_exp",
            tracking_uri="http://localhost:5000",
            log_metrics_every_n_epochs=5,
        )
        assert config.backend == "mlflow"
        assert config.experiment_name == "test_exp"
        assert config.tracking_uri == "http://localhost:5000"

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            TrackerConfig(backend="invalid")

    def test_output_dir_conversion(self):
        """Test output_dir string to Path conversion."""
        config = TrackerConfig(output_dir="/tmp/test")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/tmp/test")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrackerConfig(
            experiment_name="test",
            tags={"env": "test"},
        )
        d = config.to_dict()
        assert d["experiment_name"] == "test"
        assert d["tags"] == {"env": "test"}
        assert isinstance(d["output_dir"], str)


class TestGetTracker:
    """Tests for get_tracker factory function."""

    def test_get_disabled_tracker(self):
        """Test getting disabled tracker."""
        config = TrackerConfig(enabled=False)
        tracker = get_tracker(config=config)
        assert isinstance(tracker, DisabledTracker)

    def test_get_disabled_by_backend(self):
        """Test getting disabled tracker by backend name."""
        tracker = get_tracker(backend="disabled")
        assert isinstance(tracker, DisabledTracker)

    def test_get_local_tracker(self, tmp_path):
        """Test getting local tracker."""
        config = TrackerConfig(backend="local", output_dir=tmp_path)
        tracker = get_tracker(config=config)
        assert isinstance(tracker, LocalTracker)

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown tracking backend"):
            get_tracker(backend="invalid_backend")


class TestDisabledTracker:
    """Tests for DisabledTracker (no-op)."""

    def test_start_end_run(self):
        """Test start and end run do nothing."""
        config = TrackerConfig(enabled=False)
        tracker = DisabledTracker(config)

        run_id = tracker.start_run("test_run")
        assert run_id is not None
        assert tracker.is_active

        tracker.end_run()
        assert not tracker.is_active

    def test_log_methods_are_noop(self):
        """Test that logging methods are no-ops."""
        config = TrackerConfig(enabled=False)
        tracker = DisabledTracker(config)

        tracker.start_run()

        # These should not raise
        tracker.log_params({"lr": 0.001, "batch_size": 32})
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        tracker.log_artifact(Path("/nonexistent/path"))
        tracker.set_tags({"test": "value"})

        tracker.end_run()

    def test_run_context(self):
        """Test run context manager."""
        config = TrackerConfig(enabled=False)
        tracker = DisabledTracker(config)

        with tracker.run_context("test") as run_id:
            assert run_id is not None
            assert tracker.is_active

        assert not tracker.is_active


class TestLocalTracker:
    """Tests for LocalTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create local tracker with temp directory."""
        config = TrackerConfig(
            backend="local",
            output_dir=tmp_path,
            experiment_name="test_experiment",
        )
        return LocalTracker(config)

    def test_start_run_creates_directory(self, tracker, tmp_path):
        """Test that start_run creates run directory."""
        run_id = tracker.start_run("my_run")

        assert run_id is not None
        assert tracker.is_active
        assert tracker.run_dir is not None
        assert tracker.run_dir.exists()
        assert (tracker.run_dir / "run_info.json").exists()

    def test_end_run_saves_state(self, tracker, tmp_path):
        """Test that end_run saves all state."""
        tracker.start_run("my_run")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=1)

        run_dir = tracker.run_dir
        tracker.end_run()

        # Verify files exist
        assert (run_dir / "run_info.json").exists()
        assert (run_dir / "params.json").exists()
        assert (run_dir / "metrics.json").exists()

        # Verify run_info has end status
        with open(run_dir / "run_info.json") as f:
            run_info = json.load(f)
        assert run_info["status"] == "FINISHED"
        assert "end_time" in run_info

    def test_log_params(self, tracker):
        """Test parameter logging."""
        tracker.start_run()
        tracker.log_params({
            "lr": 0.001,
            "batch_size": 32,
            "model": "lstm",
            "path": Path("/some/path"),
        })

        params = tracker.get_run_params()
        assert params["lr"] == 0.001
        assert params["batch_size"] == 32
        assert params["model"] == "lstm"
        assert params["path"] == "/some/path"

        tracker.end_run()

    def test_log_metrics(self, tracker):
        """Test metrics logging."""
        tracker.start_run()

        for step in range(5):
            tracker.log_metrics({
                "loss": 1.0 - step * 0.1,
                "accuracy": 0.5 + step * 0.1,
            }, step=step)

        metrics = tracker.get_run_metrics()
        assert len(metrics) == 5
        assert metrics[0]["metrics"]["loss"] == 1.0
        assert metrics[4]["metrics"]["accuracy"] == 0.9

        tracker.end_run()

    def test_log_metrics_handles_numpy(self, tracker):
        """Test that numpy values are handled."""
        tracker.start_run()
        tracker.log_metrics({
            "np_float": np.float32(0.5),
            "np_int": np.int64(10),
        }, step=0)

        metrics = tracker.get_run_metrics()
        assert metrics[0]["metrics"]["np_float"] == 0.5
        assert metrics[0]["metrics"]["np_int"] == 10.0

        tracker.end_run()

    def test_log_artifact_file(self, tracker, tmp_path):
        """Test artifact logging for files."""
        tracker.start_run()

        # Create test file
        test_file = tmp_path / "test_artifact.txt"
        test_file.write_text("test content")

        tracker.log_artifact(test_file, artifact_type="data")

        # Check artifact was copied
        artifacts_dir = tracker.run_dir / "artifacts" / "data"
        assert (artifacts_dir / "test_artifact.txt").exists()

        tracker.end_run()

    def test_log_artifact_directory(self, tracker, tmp_path):
        """Test artifact logging for directories."""
        tracker.start_run()

        # Create test directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        tracker.log_artifact(test_dir, artifact_type="model")

        # Check directory was copied
        artifacts_dir = tracker.run_dir / "artifacts" / "model" / "test_dir"
        assert artifacts_dir.exists()
        assert (artifacts_dir / "file1.txt").exists()

        tracker.end_run()

    def test_set_tags(self, tracker):
        """Test tag setting."""
        tracker.start_run()
        tracker.set_tags({"env": "test", "version": "1.0"})

        run_dir = tracker.run_dir
        tracker.end_run()

        with open(run_dir / "tags.json") as f:
            tags = json.load(f)
        assert tags["env"] == "test"
        assert tags["version"] == "1.0"

    def test_run_context_success(self, tracker):
        """Test run context manager on success."""
        with tracker.run_context("test_run") as run_id:
            assert run_id is not None
            tracker.log_params({"test": "value"})
            run_dir = tracker.run_dir

        # Verify ended with FINISHED
        with open(run_dir / "run_info.json") as f:
            run_info = json.load(f)
        assert run_info["status"] == "FINISHED"

    def test_run_context_failure(self, tracker):
        """Test run context manager on exception."""
        with pytest.raises(ValueError):
            with tracker.run_context("test_run") as run_id:
                run_dir = tracker.run_dir
                raise ValueError("Test error")

        # Verify ended with FAILED
        with open(run_dir / "run_info.json") as f:
            run_info = json.load(f)
        assert run_info["status"] == "FAILED"

    def test_list_runs(self, tracker, tmp_path):
        """Test listing runs."""
        # Create multiple runs
        for i in range(3):
            tracker.start_run(f"run_{i}")
            tracker.log_metrics({"metric": i}, step=0)
            tracker.end_run()

        runs = LocalTracker.list_runs(tmp_path, "test_experiment")
        assert len(runs) == 3

    def test_load_run(self, tracker, tmp_path):
        """Test loading a completed run."""
        tracker.start_run("test")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=0)
        run_dir = tracker.run_dir
        tracker.end_run()

        data = LocalTracker.load_run(run_dir)
        assert data["params"]["lr"] == 0.001
        assert len(data["metrics"]) == 1


class TestExperimentTrackerProtocol:
    """Tests for ExperimentTracker protocol compliance."""

    @pytest.fixture(params=["local", "disabled"])
    def tracker(self, request, tmp_path):
        """Get tracker for each backend."""
        config = TrackerConfig(
            backend=request.param,
            output_dir=tmp_path,
        )
        return get_tracker(config=config)

    def test_implements_protocol(self, tracker):
        """Test that tracker implements ExperimentTracker."""
        assert isinstance(tracker, ExperimentTracker)

    def test_basic_workflow(self, tracker):
        """Test basic tracking workflow."""
        run_id = tracker.start_run("test")
        assert run_id is not None
        assert tracker.is_active
        assert tracker.run_id == run_id

        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 0.5}, step=0)
        tracker.set_tags({"tag1": "value"})

        tracker.end_run()
        assert not tracker.is_active

    def test_run_context_interface(self, tracker):
        """Test run_context interface."""
        with tracker.run_context("context_test", tags={"test": "yes"}) as run_id:
            assert run_id is not None
            tracker.log_metrics({"x": 1.0})

        assert not tracker.is_active


class TestMLflowTrackerImport:
    """Tests for MLflow tracker availability."""

    def test_mlflow_import_optional(self):
        """Test that MLflow import is optional."""
        from src.models.tracking.mlflow_tracker import MLFLOW_AVAILABLE

        # Just verify we can import and check availability
        assert isinstance(MLFLOW_AVAILABLE, bool)

    def test_get_tracker_mlflow_fallback(self, tmp_path):
        """Test that get_tracker falls back to local if MLflow unavailable."""
        from src.models.tracking.mlflow_tracker import MLFLOW_AVAILABLE

        config = TrackerConfig(backend="mlflow", output_dir=tmp_path)

        if not MLFLOW_AVAILABLE:
            # Should fall back to LocalTracker
            tracker = get_tracker(config=config)
            assert isinstance(tracker, LocalTracker)
