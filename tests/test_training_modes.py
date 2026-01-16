"""
Test Training Modes - Walk-forward, Regime-aware, and Meta-labeling trainers.

Tests cover:
- WalkForwardTrainer initialization and configuration
- WalkForwardTrainerConfig validation
- RegimeAwareTrainer regime detection
- MetaLabelingTrainer two-stage training flow
- Configuration dataclasses and result types
"""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from src.training.config import ExperimentConfig, ModelConfig
from src.training.modes.walk_forward import (
    WalkForwardTrainer,
    WalkForwardTrainerConfig,
    WalkForwardTrainingResult,
    compute_classification_metrics,
)
from src.training.modes.regime_aware import (
    RegimeAwareTrainer,
    RegimeAwareConfig,
    RegimeAwareTrainingResult,
    RegimeTrainingResult,
    detect_volatility_regime,
    detect_trend_regime,
    detect_composite_regime,
    detect_regimes,
    COMPOSITE_REGIMES,
    VOLATILITY_REGIMES,
    TREND_REGIMES,
)
from src.training.modes.meta_labeling import (
    MetaLabelingTrainer,
    MetaLabelingConfig,
    MetaLabelingResult,
    compute_bet_size_labels,
    compute_directional_meta_labels,
)


class TestWalkForwardTrainerConfig:
    """Test WalkForwardTrainerConfig dataclass."""

    def test_default_values(self):
        """WalkForwardTrainerConfig should have sensible defaults."""
        config = WalkForwardTrainerConfig()
        assert config.n_windows == 5
        assert config.window_type == "expanding"
        assert config.min_train_pct == 0.4
        assert config.test_pct == 0.1
        assert config.gap_bars == 0
        assert config.embargo_bars == 0

    def test_custom_values(self):
        """WalkForwardTrainerConfig should accept custom values."""
        config = WalkForwardTrainerConfig(
            n_windows=10,
            window_type="rolling",
            min_train_pct=0.5,
            test_pct=0.15,
            gap_bars=10,
            embargo_bars=20,
        )
        assert config.n_windows == 10
        assert config.window_type == "rolling"
        assert config.min_train_pct == 0.5

    def test_to_walk_forward_config_conversion(self):
        """to_walk_forward_config should convert to WalkForwardConfig."""
        config = WalkForwardTrainerConfig(n_windows=7, window_type="rolling")
        wf_config = config.to_walk_forward_config()

        assert wf_config.n_windows == 7
        assert wf_config.window_type == "rolling"


class TestWalkForwardTrainingResult:
    """Test WalkForwardTrainingResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize the result."""
        result = WalkForwardTrainingResult(
            model_name="xgboost",
            horizon=20,
            aggregated_metrics={"mean_accuracy": 0.65},
            total_time=120.5,
        )
        data = result.to_dict()

        assert data["model_name"] == "xgboost"
        assert data["horizon"] == 20
        assert data["total_time"] == 120.5
        assert "aggregated_metrics" in data


class TestWalkForwardTrainer:
    """Test WalkForwardTrainer initialization and methods."""

    @pytest.fixture
    def experiment_config(self, tmp_path):
        """Create test experiment configuration."""
        return ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )

    def test_initialization(self, experiment_config):
        """WalkForwardTrainer should initialize correctly."""
        trainer = WalkForwardTrainer(experiment_config)

        assert trainer.config == experiment_config
        assert trainer.wf_config.n_windows == 5  # default
        assert trainer.output_dir.exists()

    def test_initialization_with_wf_config(self, experiment_config):
        """WalkForwardTrainer should accept custom WF config."""
        wf_config = WalkForwardTrainerConfig(n_windows=10)
        trainer = WalkForwardTrainer(experiment_config, wf_config=wf_config)

        assert trainer.wf_config.n_windows == 10


class TestComputeClassificationMetrics:
    """Test compute_classification_metrics helper function."""

    def test_perfect_predictions(self):
        """Metrics should be 1.0 for perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self):
        """Metrics should be low for random predictions."""
        np.random.seed(42)
        y_true = np.array([0, 1, 2] * 10)
        y_pred = np.random.choice([0, 1, 2], size=30)

        metrics = compute_classification_metrics(y_true, y_pred)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


class TestRegimeAwareConfig:
    """Test RegimeAwareConfig dataclass."""

    def test_default_values(self):
        """RegimeAwareConfig should have sensible defaults."""
        config = RegimeAwareConfig()
        assert config.regime_type == "composite"
        assert config.train_separate_models is True
        assert config.use_regime_as_feature is False
        assert config.min_samples_per_regime == 100
        assert config.adx_threshold == 25.0
        assert config.volatility_multiplier == 1.5

    def test_custom_values(self):
        """RegimeAwareConfig should accept custom values."""
        config = RegimeAwareConfig(
            regime_type="volatility",
            train_separate_models=False,
            use_regime_as_feature=True,
            adx_threshold=30.0,
        )
        assert config.regime_type == "volatility"
        assert config.train_separate_models is False
        assert config.use_regime_as_feature is True


class TestRegimeDetection:
    """Test regime detection functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with ATR and ADX columns."""
        np.random.seed(42)
        n_samples = 200

        # Create data with clear regime patterns
        atr = np.concatenate([
            np.random.uniform(0.5, 1.0, n_samples // 2),  # Low volatility
            np.random.uniform(2.0, 3.0, n_samples // 2),  # High volatility
        ])

        adx = np.concatenate([
            np.random.uniform(15, 20, n_samples // 2),  # Mean reverting
            np.random.uniform(30, 40, n_samples // 2),  # Trending
        ])

        close = np.cumsum(np.random.randn(n_samples)) + 100

        return pd.DataFrame({
            "atr_14": atr,
            "adx_14": adx,
            "close": close,
        })

    def test_detect_volatility_regime(self, sample_data):
        """detect_volatility_regime should classify high/low vol."""
        regimes = detect_volatility_regime(sample_data, volatility_multiplier=1.5)

        assert regimes.isin(VOLATILITY_REGIMES).all()
        assert "low_vol" in regimes.values
        assert "high_vol" in regimes.values

    def test_detect_volatility_regime_missing_atr(self):
        """detect_volatility_regime should raise error without ATR."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="ATR column not found"):
            detect_volatility_regime(df)

    def test_detect_trend_regime(self, sample_data):
        """detect_trend_regime should classify trending/mean_reverting."""
        regimes = detect_trend_regime(sample_data, adx_threshold=25.0)

        assert regimes.isin(TREND_REGIMES).all()
        assert "trending" in regimes.values
        assert "mean_reverting" in regimes.values

    def test_detect_composite_regime(self, sample_data):
        """detect_composite_regime should produce 4 regime types."""
        regimes = detect_composite_regime(sample_data)

        # Should have 4 possible regimes
        assert regimes.isin(COMPOSITE_REGIMES).all()

    def test_detect_regimes_unified(self, sample_data):
        """detect_regimes should dispatch to correct detector."""
        # Volatility mode
        vol_regimes = detect_regimes(sample_data, regime_type="volatility")
        assert vol_regimes.isin(VOLATILITY_REGIMES).all()

        # Trend mode
        trend_regimes = detect_regimes(sample_data, regime_type="trend")
        assert trend_regimes.isin(TREND_REGIMES).all()

        # Composite mode (default)
        comp_regimes = detect_regimes(sample_data, regime_type="composite")
        assert comp_regimes.isin(COMPOSITE_REGIMES).all()


class TestRegimeTrainingResult:
    """Test RegimeTrainingResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize the result."""
        result = RegimeTrainingResult(
            regime="trending_high_vol",
            n_samples=1000,
            sample_fraction=0.25,
            val_f1=0.65,
            val_accuracy=0.60,
            training_time=45.2,
        )
        data = result.to_dict()

        assert data["regime"] == "trending_high_vol"
        assert data["n_samples"] == 1000
        assert data["val_f1"] == 0.65


class TestRegimeAwareTrainingResult:
    """Test RegimeAwareTrainingResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize all regime results."""
        regime_results = {
            "trending": RegimeTrainingResult(
                regime="trending",
                n_samples=500,
                sample_fraction=0.5,
                val_f1=0.7,
                val_accuracy=0.65,
            ),
        }
        result = RegimeAwareTrainingResult(
            model_name="xgboost",
            horizon=20,
            regime_type="trend",
            regime_results=regime_results,
            aggregated_metrics={"weighted_f1": 0.68},
            total_time=90.0,
        )
        data = result.to_dict()

        assert data["model_name"] == "xgboost"
        assert "trending" in data["regime_results"]
        assert data["aggregated_metrics"]["weighted_f1"] == 0.68


class TestRegimeAwareTrainer:
    """Test RegimeAwareTrainer initialization."""

    @pytest.fixture
    def experiment_config(self, tmp_path):
        """Create test experiment configuration."""
        return ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )

    def test_initialization(self, experiment_config):
        """RegimeAwareTrainer should initialize correctly."""
        trainer = RegimeAwareTrainer(experiment_config)

        assert trainer.config == experiment_config
        assert trainer.regime_config.regime_type == "composite"
        assert trainer.output_dir.exists()

    def test_initialization_with_regime_config(self, experiment_config):
        """RegimeAwareTrainer should accept custom regime config."""
        regime_config = RegimeAwareConfig(
            regime_type="volatility",
            train_separate_models=False,
        )
        trainer = RegimeAwareTrainer(experiment_config, regime_config=regime_config)

        assert trainer.regime_config.regime_type == "volatility"
        assert trainer.regime_config.train_separate_models is False


class TestMetaLabelingConfig:
    """Test MetaLabelingConfig dataclass."""

    def test_default_values(self):
        """MetaLabelingConfig should have sensible defaults."""
        config = MetaLabelingConfig()
        assert config.primary_model == "xgboost"
        assert config.meta_model == "logistic"
        assert config.meta_threshold == 0.3
        assert config.use_quality_weights is True

    def test_custom_values(self):
        """MetaLabelingConfig should accept custom values."""
        config = MetaLabelingConfig(
            primary_model="lightgbm",
            meta_model="random_forest",
            meta_threshold=0.5,
            calibrate_probabilities=True,
        )
        assert config.primary_model == "lightgbm"
        assert config.meta_threshold == 0.5


class TestMetaLabelingResult:
    """Test MetaLabelingResult dataclass."""

    def test_to_dict(self):
        """to_dict should serialize the result."""
        result = MetaLabelingResult(
            primary_model="xgboost",
            meta_model="logistic",
            horizon=20,
            primary_metrics={"val_f1": 0.65},
            meta_metrics={"val_f1": 0.70},
            combined_metrics={"combined_accuracy": 0.72},
            total_time=150.0,
        )
        data = result.to_dict()

        assert data["primary_model"] == "xgboost"
        assert data["meta_model"] == "logistic"
        assert data["combined_metrics"]["combined_accuracy"] == 0.72


class TestComputeBetSizeLabels:
    """Test compute_bet_size_labels function."""

    def test_correct_predictions_full_bet(self):
        """Correct predictions should have bet size 1.0."""
        y_actual = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])  # All correct

        bet_sizes = compute_bet_size_labels(y_actual, y_pred)

        np.testing.assert_array_equal(bet_sizes, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_wrong_predictions_zero_bet(self):
        """Wrong predictions should have bet size 0.0."""
        y_actual = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])  # All wrong

        bet_sizes = compute_bet_size_labels(y_actual, y_pred)

        np.testing.assert_array_equal(bet_sizes, [0.0, 0.0, 0.0])

    def test_quality_weights_applied(self):
        """Quality weights should scale bet sizes."""
        y_actual = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])  # All correct
        quality = np.array([0.5, 0.8, 1.0])

        bet_sizes = compute_bet_size_labels(y_actual, y_pred, quality_weights=quality)

        np.testing.assert_array_almost_equal(bet_sizes, [0.5, 0.8, 1.0])


class TestComputeDirectionalMetaLabels:
    """Test compute_directional_meta_labels function."""

    def test_with_probabilities(self):
        """Meta-labels should incorporate prediction confidence."""
        y_actual = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])  # All correct
        proba = np.array([
            [0.9, 0.05, 0.05],  # High confidence class 0
            [0.1, 0.8, 0.1],    # High confidence class 1
            [0.1, 0.2, 0.7],    # Moderate confidence class 2
        ])

        bet_sizes = compute_directional_meta_labels(y_actual, y_pred, proba)

        # Bet sizes should be scaled by max probability
        assert bet_sizes[0] == pytest.approx(0.9, rel=0.01)
        assert bet_sizes[1] == pytest.approx(0.8, rel=0.01)
        assert bet_sizes[2] == pytest.approx(0.7, rel=0.01)

    def test_wrong_predictions_zero_bet(self):
        """Wrong predictions should have zero bet regardless of confidence."""
        y_actual = np.array([0, 1])
        y_pred = np.array([1, 0])  # All wrong
        proba = np.array([
            [0.1, 0.9, 0.0],  # Confident but wrong
            [0.95, 0.05, 0.0],  # Very confident but wrong
        ])

        bet_sizes = compute_directional_meta_labels(y_actual, y_pred, proba)

        np.testing.assert_array_equal(bet_sizes, [0.0, 0.0])

    def test_with_quality_weights(self):
        """Quality weights should further scale bet sizes."""
        y_actual = np.array([0, 1])
        y_pred = np.array([0, 1])
        proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.9, 0.0],
        ])
        quality = np.array([0.5, 1.0])

        bet_sizes = compute_directional_meta_labels(
            y_actual, y_pred, proba, quality_weights=quality
        )

        assert bet_sizes[0] == pytest.approx(0.8 * 0.5, rel=0.01)
        assert bet_sizes[1] == pytest.approx(0.9 * 1.0, rel=0.01)


class TestMetaLabelingTrainer:
    """Test MetaLabelingTrainer initialization."""

    @pytest.fixture
    def experiment_config(self, tmp_path):
        """Create test experiment configuration."""
        return ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
        )

    def test_initialization(self, experiment_config):
        """MetaLabelingTrainer should initialize correctly."""
        trainer = MetaLabelingTrainer(experiment_config)

        assert trainer.config == experiment_config
        assert trainer.meta_config.primary_model == "xgboost"
        assert trainer.meta_config.meta_model == "logistic"
        assert trainer.output_dir.exists()

    def test_initialization_with_meta_config(self, experiment_config):
        """MetaLabelingTrainer should accept custom meta config."""
        meta_config = MetaLabelingConfig(
            primary_model="lightgbm",
            meta_model="random_forest",
            meta_threshold=0.4,
        )
        trainer = MetaLabelingTrainer(experiment_config, meta_config=meta_config)

        assert trainer.meta_config.primary_model == "lightgbm"
        assert trainer.meta_config.meta_model == "random_forest"
        assert trainer.meta_config.meta_threshold == 0.4


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_default_values(self):
        """ExperimentConfig should have sensible defaults."""
        config = ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
        )
        assert config.cv_splits == 5
        assert config.cross_validate is True
        assert config.build_ensemble is False

    def test_models_converted_to_model_config(self):
        """String model names should be converted to ModelConfig."""
        config = ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost", "lightgbm"],
        )
        # After __post_init__, models should be ModelConfig instances
        assert all(isinstance(m, ModelConfig) for m in config.models)
        assert config.models[0].name == "xgboost"
        assert config.models[1].name == "lightgbm"

    def test_path_string_conversion(self):
        """String paths should be converted to Path objects."""
        config = ExperimentConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
            data_dir="data/test",
            output_dir="output/test",
        )
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.output_dir, Path)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """ModelConfig should have sensible defaults."""
        config = ModelConfig(name="xgboost")
        assert config.optimize_features is False
        assert config.optimize_hyperparams is False
        assert config.timeframe is None
        assert config.sequence_length is None

    def test_custom_values(self):
        """ModelConfig should accept custom values."""
        config = ModelConfig(
            name="lstm",
            timeframe="5min",
            optimize_features=True,
            sequence_length=60,
        )
        assert config.name == "lstm"
        assert config.timeframe == "5min"
        assert config.optimize_features is True
        assert config.sequence_length == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
