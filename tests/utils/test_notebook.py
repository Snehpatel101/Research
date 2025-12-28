"""
Tests for Notebook Utilities.

Tests cover:
- Environment setup
- Display utilities
- Configuration helpers
- Sample data generation
- Progress callbacks

Note: Some functions require specific environments (Colab, Kaggle)
and are tested with mocking.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# ENVIRONMENT SETUP TESTS
# =============================================================================

class TestSetupNotebook:
    """Tests for setup_notebook function."""

    def test_setup_returns_dict(self):
        """setup_notebook should return environment info dict."""
        from src.utils.notebook import setup_notebook

        with patch("src.models.device.get_environment_info") as mock_env, \
             patch("src.models.device.get_best_gpu") as mock_gpu:
            mock_env.return_value = {
                "platform": "linux",
                "python_version": "3.11.0",
                "is_colab": False,
                "is_kaggle": False,
            }
            mock_gpu.return_value = None

            result = setup_notebook()

            assert isinstance(result, dict)
            assert "platform" in result
            assert "python_version" in result
            assert "gpu_available" in result

    def test_setup_sets_random_seed(self):
        """setup_notebook should set random seed."""
        from src.utils.notebook import setup_notebook

        with patch("src.models.device.get_environment_info") as mock_env, \
             patch("src.models.device.get_best_gpu") as mock_gpu:
            mock_env.return_value = {
                "platform": "linux",
                "python_version": "3.11.0",
                "is_colab": False,
                "is_kaggle": False,
            }
            mock_gpu.return_value = None

            result = setup_notebook(seed=123)

            assert result["seed"] == 123


class TestInstallDependencies:
    """Tests for install_dependencies function."""

    def test_install_returns_dict(self):
        """install_dependencies should return results dict."""
        from src.utils.notebook import install_dependencies

        # Use a definitely-installed package
        with patch("subprocess.check_call") as mock_call:
            mock_call.return_value = 0
            result = install_dependencies(packages=["numpy"], quiet=True)

            assert isinstance(result, dict)
            assert "numpy" in result


class TestMountDrive:
    """Tests for mount_drive function."""

    def test_mount_drive_not_colab_returns_false(self):
        """mount_drive should return False when not in Colab."""
        from src.utils.notebook import mount_drive

        with patch("src.models.device.is_colab") as mock_colab:
            mock_colab.return_value = False
            result = mount_drive()

            assert result is False


# =============================================================================
# SAMPLE DATA GENERATION TESTS
# =============================================================================

class TestDownloadSampleData:
    """Tests for download_sample_data function."""

    def test_generates_sample_data(self, tmp_path):
        """Should generate synthetic OHLCV data."""
        from src.utils.notebook import download_sample_data

        result = download_sample_data(
            output_dir=str(tmp_path / "sample"),
            symbols=["TEST"],
        )

        assert "TEST" in result
        assert Path(result["TEST"]).exists()

    def test_generates_expected_columns(self, tmp_path):
        """Generated data should have OHLCV columns."""
        import pandas as pd
        from src.utils.notebook import download_sample_data

        result = download_sample_data(
            output_dir=str(tmp_path / "sample"),
            symbols=["TEST"],
        )

        df = pd.read_parquet(result["TEST"])
        expected_cols = ["datetime", "open", "high", "low", "close", "volume", "symbol"]

        for col in expected_cols:
            assert col in df.columns

    def test_generates_valid_ohlcv(self, tmp_path):
        """Generated data should have valid OHLCV relationships."""
        import pandas as pd
        from src.utils.notebook import download_sample_data

        result = download_sample_data(
            output_dir=str(tmp_path / "sample"),
            symbols=["TEST"],
        )

        df = pd.read_parquet(result["TEST"])

        # High >= max(open, close)
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        # Low <= min(open, close)
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        # Volume >= 0
        assert (df["volume"] >= 0).all()


# =============================================================================
# DISPLAY UTILITIES TESTS
# =============================================================================

class TestDisplayMetrics:
    """Tests for display_metrics function."""

    def test_display_metrics_no_error(self, capsys):
        """display_metrics should print without errors."""
        from src.utils.notebook import display_metrics

        results = {
            "model_name": "xgboost",
            "horizon": 20,
            "training_metrics": {
                "train_loss": 0.5,
                "val_loss": 0.6,
                "train_f1": 0.7,
                "val_f1": 0.65,
                "epochs_trained": 10,
                "training_time_seconds": 30.5,
            },
            "evaluation_metrics": {
                "accuracy": 0.65,
                "macro_f1": 0.6,
                "weighted_f1": 0.62,
                "precision": 0.58,
                "recall": 0.55,
            },
        }

        # Should not raise
        display_metrics(results, title="Test Results", show_confusion=False)

        captured = capsys.readouterr()
        assert "xgboost" in captured.out
        assert "20" in captured.out


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_plot_creates_figure(self, tmp_path):
        """plot_confusion_matrix should create a plot."""
        from src.utils.notebook import plot_confusion_matrix
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Trading labels: -1=short, 0=neutral, 1=long
        y_true = np.array([-1, -1, 0, 0, 1, 1])
        y_pred = np.array([-1, -1, 0, 0, 1, 1])

        save_path = tmp_path / "confusion.png"

        # Should not raise
        plot_confusion_matrix(y_true, y_pred, save_path=str(save_path))

        assert save_path.exists()
        plt.close("all")


class TestPlotTrainingHistory:
    """Tests for plot_training_history function."""

    def test_plot_history_with_losses(self, tmp_path):
        """plot_training_history should plot loss curves."""
        from src.utils.notebook import plot_training_history
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history = {
            "train_loss": [1.0, 0.8, 0.6, 0.5],
            "val_loss": [1.1, 0.9, 0.7, 0.6],
        }

        save_path = tmp_path / "history.png"

        plot_training_history(history, save_path=str(save_path))

        assert save_path.exists()
        plt.close("all")

    def test_plot_history_empty_does_not_error(self):
        """Empty history should not raise."""
        from src.utils.notebook import plot_training_history
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Should not raise
        plot_training_history({})
        plt.close("all")


class TestPlotModelComparison:
    """Tests for plot_model_comparison function."""

    def test_plot_comparison_with_results(self, tmp_path):
        """plot_model_comparison should create comparison chart."""
        from src.utils.notebook import plot_model_comparison
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = {
            "xgboost": {
                "evaluation_metrics": {"macro_f1": 0.65},
            },
            "random_forest": {
                "evaluation_metrics": {"macro_f1": 0.60},
            },
        }

        save_path = tmp_path / "comparison.png"

        plot_model_comparison(results, metric="macro_f1", save_path=str(save_path))

        assert save_path.exists()
        plt.close("all")


# =============================================================================
# CONFIGURATION HELPERS TESTS
# =============================================================================

class TestGetSampleConfig:
    """Tests for get_sample_config function."""

    def test_get_sample_config_basic(self):
        """Should return config dict."""
        from src.utils.notebook import get_sample_config

        config = get_sample_config("xgboost", horizon=20)

        assert config["model_type"] == "xgboost"
        assert config["horizon"] == 20

    def test_get_sample_config_quick_mode(self):
        """Quick mode should provide faster settings."""
        from src.utils.notebook import get_sample_config

        config = get_sample_config("xgboost", horizon=20, quick_mode=True)

        assert "model_config" in config
        # Quick mode should have reduced estimators
        assert config["model_config"]["n_estimators"] == 50


class TestCreateProgressCallback:
    """Tests for create_progress_callback function."""

    def test_callback_created(self):
        """Should create a callback function."""
        from src.utils.notebook import create_progress_callback

        callback = create_progress_callback(total_epochs=10, description="Test")

        assert callable(callback)

    def test_callback_accepts_metrics(self):
        """Callback should accept epoch and metrics."""
        from src.utils.notebook import create_progress_callback

        callback = create_progress_callback(total_epochs=10, description="Test")

        # Should not raise
        callback(epoch=0, metrics={"loss": 0.5, "accuracy": 0.6})


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_results() -> Dict[str, Any]:
    """Sample training results for testing display functions."""
    return {
        "run_id": "xgboost_h20_20231201_120000",
        "model_name": "xgboost",
        "horizon": 20,
        "training_metrics": {
            "train_loss": 0.45,
            "val_loss": 0.52,
            "train_accuracy": 0.72,
            "val_accuracy": 0.68,
            "train_f1": 0.70,
            "val_f1": 0.65,
            "epochs_trained": 50,
            "training_time_seconds": 120.5,
        },
        "evaluation_metrics": {
            "accuracy": 0.68,
            "macro_f1": 0.65,
            "weighted_f1": 0.66,
            "precision": 0.64,
            "recall": 0.63,
            "per_class_f1": {
                "short": 0.60,
                "neutral": 0.65,
                "long": 0.70,
            },
            "confusion_matrix": [[20, 5, 3], [4, 25, 6], [2, 5, 30]],
            "trading": {
                "long_signals": 35,
                "short_signals": 28,
                "neutral_signals": 37,
                "position_win_rate": 0.55,
            },
        },
        "total_time_seconds": 180.0,
        "output_path": "/tmp/experiments/runs/xgboost_h20",
    }
