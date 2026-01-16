"""
Tests for unified ML pipeline.

Tests both basic functionality and config normalization.
"""

import pytest
from pathlib import Path
from src.ml_pipeline import MLPipeline, MLConfig, ModelConfig, PipelineState


class TestMLConfig:
    """Test MLConfig creation and validation."""

    def test_create_basic_config(self):
        """Test creating basic config."""
        config = MLConfig(
            symbol="MES",
            horizons=[20],
            models=["xgboost"],
        )

        assert config.symbol == "MES"
        assert config.horizons == [20]
        assert len(config.models) == 1
        assert config.models[0].name == "xgboost"
        assert config.training_mode == "standard"

    def test_create_with_model_configs(self):
        """Test creating config with ModelConfig objects."""
        config = MLConfig(
            symbol="MES",
            horizons=[20],
            models=[
                ModelConfig(name="xgboost", timeframe="15min", optimize_features=True),
                ModelConfig(name="lstm", timeframe="5min", optimize_features=True),
            ],
            build_ensemble=True,
        )

        assert len(config.models) == 2
        assert config.models[0].name == "xgboost"
        assert config.models[0].timeframe == "15min"
        assert config.models[0].optimize_features is True
        assert config.build_ensemble is True

    def test_config_validation(self):
        """Test config validation catches errors."""
        with pytest.raises(ValueError, match="horizons must be positive"):
            MLConfig(symbol="MES", horizons=[0, -5])

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["symbol"] == "MES"
        assert d["horizons"] == [20]

    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        d = {
            "symbol": "MES",
            "horizons": [20],
            "models": ["xgboost"],
            "training_mode": "standard",
        }

        config = MLConfig.from_dict(d)
        assert config.symbol == "MES"
        assert config.horizons == [20]


class TestPipelineState:
    """Test PipelineState functionality."""

    def test_create_state(self):
        """Test creating pipeline state."""
        state = PipelineState(run_id="test_20260116_120000")

        assert state.run_id == "test_20260116_120000"
        assert state.status == "running"
        assert not state.is_complete("data")

    def test_mark_complete(self):
        """Test marking phase complete."""
        state = PipelineState(run_id="test_20260116_120000")

        state.mark_start("data")
        state.mark_complete("data", {"success": True})

        assert state.is_complete("data")
        assert state.get_result("data")["success"] is True

    def test_mark_failed(self):
        """Test marking phase failed."""
        state = PipelineState(run_id="test_20260116_120000")

        state.mark_start("training")
        state.mark_failed("training", "Model training failed")

        assert not state.is_complete("training")
        assert state.status == "failed"
        assert "Model training failed" in state.error_message

    def test_get_summary(self):
        """Test getting state summary."""
        state = PipelineState(run_id="test_20260116_120000")

        state.mark_start("data")
        state.mark_complete("data", {"success": True})

        summary = state.get_summary()

        assert summary["run_id"] == "test_20260116_120000"
        assert summary["status"] == "running"
        assert "data" in summary["completed_phases"]


class TestMLPipeline:
    """Test MLPipeline initialization and config normalization."""

    def test_create_pipeline_with_mlconfig(self):
        """Test creating pipeline with MLConfig."""
        config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
        pipeline = MLPipeline(config)

        assert pipeline.config.symbol == "MES"
        assert isinstance(pipeline.state, PipelineState)

    def test_create_pipeline_with_dict(self):
        """Test creating pipeline with dict config."""
        config_dict = {
            "symbol": "MES",
            "horizons": [20],
            "models": ["xgboost"],
        }

        pipeline = MLPipeline(config_dict)

        assert pipeline.config.symbol == "MES"
        assert pipeline.config.horizons == [20]

    def test_invalid_config_type(self):
        """Test error handling for invalid config type."""
        with pytest.raises(ValueError, match="Invalid config type"):
            MLPipeline(12345)  # Invalid type

    def test_yaml_config_not_found(self):
        """Test error handling for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            MLPipeline("nonexistent_config.yaml")

    def test_state_initialization(self):
        """Test state is properly initialized."""
        config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
        pipeline = MLPipeline(config)

        assert pipeline.state.run_id == config.run_id
        assert not pipeline.state.is_complete("data")
        assert not pipeline.state.is_complete("training")


class TestConfigConversion:
    """Test MLConfig → PipelineConfig/ExperimentConfig conversion."""

    def test_mlconfig_to_pipeline_config(self):
        """Test converting MLConfig to PipelineConfig."""
        config = MLConfig(
            symbol="MES",
            horizons=[20],
            start_date="2024-01-01",
            end_date="2024-12-31",
            target_timeframe="5min",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=1440,
        )

        pipeline = MLPipeline(config)
        pipeline_config = pipeline._mlconfig_to_pipeline_config()

        # Check key fields mapped correctly
        assert pipeline_config.symbols == ["MES"]
        assert pipeline_config.start_date == "2024-01-01"
        assert pipeline_config.end_date == "2024-12-31"
        assert pipeline_config.target_timeframe == "5min"
        assert pipeline_config.label_horizons == [20]
        assert pipeline_config.train_ratio == 0.70
        assert pipeline_config.purge_bars == 60

    def test_mlconfig_to_experiment_config_requires_data_phase(self):
        """Test that ExperimentConfig conversion requires completed data phase."""
        config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
        pipeline = MLPipeline(config)

        with pytest.raises(
            (RuntimeError, ValueError),
            match="(Data phase results not found|Phase .* is not complete)",
        ):
            pipeline._mlconfig_to_experiment_config()


class TestTrainingModeDispatch:
    """Test training mode dispatchers."""

    def test_standard_training_not_implemented_without_data(self):
        """Test standard training requires data phase complete."""
        config = MLConfig(symbol="MES", horizons=[20], models=["xgboost"])
        pipeline = MLPipeline(config)

        with pytest.raises(RuntimeError, match="data phase not complete"):
            pipeline.run_training()

    def test_walk_forward_not_implemented(self):
        """Test walk-forward mode raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], training_mode="walk_forward"
        )
        pipeline = MLPipeline(config)

        # Mock data phase complete
        pipeline.state.mark_complete("data", {"success": True})

        with pytest.raises(NotImplementedError, match="(Walk-forward|walk_forward)"):
            pipeline.run_training()

    def test_regime_aware_not_implemented(self):
        """Test regime-aware mode raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], training_mode="regime_aware"
        )
        pipeline = MLPipeline(config)

        # Mock data phase complete
        pipeline.state.mark_complete("data", {"success": True})

        with pytest.raises(NotImplementedError, match="(Regime-aware|regime_aware)"):
            pipeline.run_training()

    def test_meta_labeling_not_implemented(self):
        """Test meta-labeling mode raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], training_mode="meta_labeling"
        )
        pipeline = MLPipeline(config)

        # Mock data phase complete
        pipeline.state.mark_complete("data", {"success": True})

        with pytest.raises(NotImplementedError, match="(Meta-labeling|meta_labeling)"):
            pipeline.run_training()


class TestEvaluationDispatch:
    """Test evaluation method dispatchers."""

    def test_cv_evaluation_not_implemented(self):
        """Test CV evaluation raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], evaluation_methods=["cv"]
        )
        pipeline = MLPipeline(config)

        # Mock phases complete
        pipeline.state.mark_complete("data", {"success": True})
        pipeline.state.mark_complete("training", {"model_paths": {}})

        with pytest.raises(NotImplementedError, match="CV evaluation"):
            pipeline.run_evaluation()

    def test_walk_forward_evaluation_not_implemented(self):
        """Test walk-forward evaluation raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], evaluation_methods=["walk_forward"]
        )
        pipeline = MLPipeline(config)

        # Mock phases complete
        pipeline.state.mark_complete("data", {"success": True})
        pipeline.state.mark_complete("training", {"model_paths": {}})

        with pytest.raises(NotImplementedError, match="Walk-forward evaluation"):
            pipeline.run_evaluation()

    def test_cpcv_pbo_evaluation_not_implemented(self):
        """Test CPCV-PBO evaluation raises NotImplementedError."""
        config = MLConfig(
            symbol="MES", horizons=[20], models=["xgboost"], evaluation_methods=["cpcv_pbo"]
        )
        pipeline = MLPipeline(config)

        # Mock phases complete
        pipeline.state.mark_complete("data", {"success": True})
        pipeline.state.mark_complete("training", {"model_paths": {}})

        with pytest.raises(NotImplementedError, match="CPCV-PBO evaluation"):
            pipeline.run_evaluation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
