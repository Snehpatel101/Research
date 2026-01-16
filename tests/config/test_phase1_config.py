"""
Tests for Phase 1 Configuration Layer (SNwH Implementation).

Tests:
- TrainerConfig new fields and from_model_contract()
- PerModelConfig and EnsemblePlan
- ModelDataRequirements new fields
- ModelConfigSection in UnifiedConfig
"""

import pytest
from pathlib import Path

from src.models.config import TrainerConfig, PerModelConfig, EnsemblePlan
from src.models.config.data_requirements import (
    MODEL_DATA_REQUIREMENTS,
    ModelDataRequirements,
    get_model_requirements,
)


class TestTrainerConfigPhase1:
    """Tests for TrainerConfig Phase 1 SNwH extensions."""

    def test_new_fields_have_defaults(self):
        """Test that new fields have appropriate defaults."""
        config = TrainerConfig(model_name="xgboost", horizon=20)

        # New fields should have defaults
        assert config.primary_timeframe == "5min"  # Default
        assert config.mtf_mode == "indicators"
        assert config.mtf_timeframes == []
        assert config.feature_mode == "engineered"
        assert config.adapter_id == "tabular"  # Auto-resolved from input_rank=2
        assert config.input_rank == 2
        assert config.min_features == 4
        assert config.max_features == 200

    def test_adapter_id_auto_resolution(self):
        """Test that adapter_id is auto-resolved from input_rank."""
        # Tabular (2D)
        config = TrainerConfig(model_name="test", horizon=20, input_rank=2)
        assert config.adapter_id == "tabular"

        # Sequence (3D)
        config = TrainerConfig(model_name="test", horizon=20, input_rank=3)
        assert config.adapter_id == "sequence"

        # Multi-stream (4D)
        config = TrainerConfig(model_name="test", horizon=20, input_rank=4)
        assert config.adapter_id == "multi_stream"

    def test_mtf_mode_validation(self):
        """Test that invalid mtf_mode raises ValueError."""
        with pytest.raises(ValueError, match="mtf_mode must be one of"):
            TrainerConfig(model_name="test", horizon=20, mtf_mode="invalid")

    def test_feature_mode_validation(self):
        """Test that invalid feature_mode raises ValueError."""
        with pytest.raises(ValueError, match="feature_mode must be one of"):
            TrainerConfig(model_name="test", horizon=20, feature_mode="invalid")

    def test_input_rank_validation(self):
        """Test that invalid input_rank raises ValueError."""
        with pytest.raises(ValueError, match="input_rank must be 2, 3, or 4"):
            TrainerConfig(model_name="test", horizon=20, input_rank=5)

    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes all new Phase 1 fields."""
        config = TrainerConfig(
            model_name="lstm",
            horizon=20,
            primary_timeframe="5min",
            mtf_mode="indicators",
            mtf_timeframes=["15min", "60min"],
            feature_mode="engineered",
            input_rank=3,
        )
        data = config.to_dict()

        assert data["primary_timeframe"] == "5min"
        assert data["mtf_mode"] == "indicators"
        assert data["mtf_timeframes"] == ["15min", "60min"]
        assert data["feature_mode"] == "engineered"
        assert data["adapter_id"] == "sequence"
        assert data["input_rank"] == 3

    def test_from_model_contract_xgboost(self):
        """Test from_model_contract for XGBoost."""
        config = TrainerConfig.from_model_contract("xgboost", horizon=20)

        assert config.model_name == "xgboost"
        assert config.horizon == 20
        assert config.primary_timeframe == "15min"
        assert config.mtf_mode == "indicators"
        assert config.feature_mode == "engineered"
        assert config.input_rank == 2
        assert config.adapter_id == "tabular"

    def test_from_model_contract_lstm(self):
        """Test from_model_contract for LSTM."""
        config = TrainerConfig.from_model_contract("lstm", horizon=20)

        assert config.model_name == "lstm"
        assert config.primary_timeframe == "5min"
        assert config.mtf_mode == "indicators"
        assert config.input_rank == 3
        assert config.adapter_id == "sequence"
        assert config.sequence_length == 60

    def test_from_model_contract_patchtst(self):
        """Test from_model_contract for PatchTST."""
        config = TrainerConfig.from_model_contract("patchtst", horizon=20)

        assert config.model_name == "patchtst"
        assert config.primary_timeframe == "1min"
        assert config.mtf_mode == "multi_stream"
        assert config.feature_mode == "raw"
        assert config.input_rank == 3
        assert config.adapter_id == "sequence"

    def test_from_model_contract_with_overrides(self):
        """Test from_model_contract with user overrides."""
        config = TrainerConfig.from_model_contract(
            "xgboost",
            horizon=20,
            batch_size=512,
            max_epochs=200,
            primary_timeframe="10min",  # Override contract default
        )

        assert config.batch_size == 512
        assert config.max_epochs == 200
        assert config.primary_timeframe == "10min"  # User override

    def test_from_model_contract_case_insensitive(self):
        """Test from_model_contract is case insensitive."""
        config1 = TrainerConfig.from_model_contract("XGBOOST", horizon=20)
        config2 = TrainerConfig.from_model_contract("XGBoost", horizon=20)
        config3 = TrainerConfig.from_model_contract("xgboost", horizon=20)

        assert config1.model_name == config2.model_name == config3.model_name


class TestPerModelConfig:
    """Tests for PerModelConfig dataclass."""

    def test_basic_creation(self):
        """Test basic PerModelConfig creation."""
        config = PerModelConfig(name="xgboost")

        assert config.name == "xgboost"
        assert config.timeframe is None
        assert config.feature_mode is None
        assert config.mtf_mode is None

    def test_resolved_properties_from_contract(self):
        """Test that resolved properties use contract defaults."""
        config = PerModelConfig(name="xgboost")

        assert config.resolved_timeframe == "15min"  # From contract
        assert config.resolved_feature_mode == "engineered"  # From contract
        assert config.resolved_mtf_mode == "indicators"  # From contract
        assert config.input_rank == 2
        assert config.adapter_id == "tabular"

    def test_resolved_properties_with_overrides(self):
        """Test that explicit values override contract defaults."""
        config = PerModelConfig(
            name="xgboost",
            timeframe="10min",
            feature_mode="raw",
            mtf_mode="none",
        )

        assert config.resolved_timeframe == "10min"  # Override
        assert config.resolved_feature_mode == "raw"  # Override
        assert config.resolved_mtf_mode == "none"  # Override

    def test_to_trainer_config_kwargs(self):
        """Test conversion to TrainerConfig kwargs."""
        config = PerModelConfig(name="lstm", timeframe="10min")
        kwargs = config.to_trainer_config_kwargs()

        assert kwargs["model_name"] == "lstm"
        assert kwargs["primary_timeframe"] == "10min"  # Override
        assert kwargs["input_rank"] == 3
        assert kwargs["adapter_id"] == "sequence"

    def test_from_string(self):
        """Test creation from model name string."""
        config = PerModelConfig.from_string("lstm")

        assert config.name == "lstm"
        assert config.resolved_timeframe == "5min"  # Contract default

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = PerModelConfig(
            name="lstm",
            timeframe="10min",
            feature_mode="engineered",
            optimize_features=True,
            hyperparameters={"hidden_size": 128},
        )

        data = original.to_dict()
        restored = PerModelConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.timeframe == original.timeframe
        assert restored.feature_mode == original.feature_mode
        assert restored.optimize_features == original.optimize_features
        assert restored.hyperparameters == original.hyperparameters


class TestEnsemblePlan:
    """Tests for EnsemblePlan dataclass."""

    def test_basic_creation(self):
        """Test basic EnsemblePlan creation."""
        base_models = [
            PerModelConfig(name="xgboost"),
            PerModelConfig(name="lstm"),
        ]
        plan = EnsemblePlan(base_models=base_models)

        assert len(plan.base_models) == 2
        assert plan.meta_learner == "ridge_meta"
        assert plan.stacking_method == "soft"

    def test_minimum_base_models_validation(self):
        """Test that at least 2 base models are required."""
        with pytest.raises(ValueError, match="at least 2 base models"):
            EnsemblePlan(base_models=[PerModelConfig(name="xgboost")])

    def test_is_heterogeneous(self):
        """Test is_heterogeneous property."""
        # Heterogeneous (different families)
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm", "patchtst"])
        assert plan.is_heterogeneous is True

        # Homogeneous (same family)
        plan = EnsemblePlan.from_model_names(["xgboost", "lightgbm"])
        assert plan.is_heterogeneous is False

    def test_required_timeframes(self):
        """Test required_timeframes property."""
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm", "patchtst"])
        tfs = plan.required_timeframes

        assert "15min" in tfs  # XGBoost
        assert "5min" in tfs   # LSTM
        assert "1min" in tfs   # PatchTST

    def test_has_tabular_models(self):
        """Test has_tabular_models property."""
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm"])
        assert plan.has_tabular_models is True

        plan = EnsemblePlan.from_model_names(["lstm", "gru"])
        assert plan.has_tabular_models is False

    def test_has_sequence_models(self):
        """Test has_sequence_models property."""
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm"])
        assert plan.has_sequence_models is True

        plan = EnsemblePlan.from_model_names(["xgboost", "lightgbm"])
        assert plan.has_sequence_models is False

    def test_get_models_by_adapter(self):
        """Test get_models_by_adapter method."""
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm", "patchtst"])
        groups = plan.get_models_by_adapter()

        assert "tabular" in groups
        assert len(groups["tabular"]) == 1
        assert groups["tabular"][0].name == "xgboost"

        assert "sequence" in groups
        assert len(groups["sequence"]) == 2  # lstm and patchtst

    def test_from_model_names(self):
        """Test from_model_names factory method."""
        plan = EnsemblePlan.from_model_names(
            base_models=["xgboost", "lstm", "patchtst"],
            meta_learner="xgboost_meta",
            n_folds=3,
        )

        assert len(plan.base_models) == 3
        assert plan.base_model_names == ["xgboost", "lstm", "patchtst"]
        assert plan.meta_learner == "xgboost_meta"
        assert plan.n_folds == 3

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = EnsemblePlan.from_model_names(
            ["xgboost", "lstm"],
            meta_learner="ridge_meta",
            n_folds=3,
        )

        data = original.to_dict()
        restored = EnsemblePlan.from_dict(data)

        assert len(restored.base_models) == 2
        assert restored.meta_learner == "ridge_meta"
        assert restored.n_folds == 3


class TestModelDataRequirementsPhase1:
    """Tests for ModelDataRequirements Phase 1 SNwH fields."""

    def test_all_models_have_phase1_fields(self):
        """Test all 23 models have Phase 1 fields."""
        assert len(MODEL_DATA_REQUIREMENTS) >= 23  # 23-24 models (includes mlp)

        for name, req in MODEL_DATA_REQUIREMENTS.items():
            assert hasattr(req, "input_rank"), f"{name} missing input_rank"
            assert hasattr(req, "feature_mode"), f"{name} missing feature_mode"
            assert hasattr(req, "mtf_mode"), f"{name} missing mtf_mode"
            assert hasattr(req, "primary_timeframe"), f"{name} missing primary_timeframe"
            assert hasattr(req, "min_features"), f"{name} missing min_features"

    def test_boosting_models_phase1_fields(self):
        """Test boosting models have correct Phase 1 fields."""
        for model in ["xgboost", "lightgbm", "catboost"]:
            if model not in MODEL_DATA_REQUIREMENTS:
                continue
            req = get_model_requirements(model)
            assert req.input_rank == 2
            assert req.feature_mode == "engineered"
            assert req.mtf_mode == "indicators"
            assert req.primary_timeframe == "15min"

    def test_neural_models_phase1_fields(self):
        """Test neural models have correct Phase 1 fields."""
        for model in ["lstm", "gru"]:
            req = get_model_requirements(model)
            assert req.input_rank == 3
            assert req.feature_mode == "engineered"
            assert req.primary_timeframe == "5min"

    def test_patchtst_multi_stream(self):
        """Test PatchTST has multi_stream MTF mode."""
        req = get_model_requirements("patchtst")
        assert req.input_rank == 3
        assert req.mtf_mode == "multi_stream"
        assert req.feature_mode == "raw"
        assert req.primary_timeframe == "1min"
        assert len(req.mtf_timeframes) > 0

    def test_meta_learner_oof_probs(self):
        """Test meta-learners use oof_probs feature mode."""
        for model in ["ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]:
            req = get_model_requirements(model)
            assert req.input_rank == 2
            assert req.feature_mode == "oof_probs"
            assert req.mtf_mode == "none"

    def test_adapter_id_property(self):
        """Test adapter_id property works correctly."""
        # Tabular
        req = get_model_requirements("xgboost")
        assert req.adapter_id == "tabular"

        # Sequence
        req = get_model_requirements("lstm")
        assert req.adapter_id == "sequence"


class TestUnifiedConfigPhase1:
    """Tests for UnifiedConfig Phase 1 extensions."""

    def test_model_config_section_exists(self):
        """Test ModelConfigSection is in UnifiedConfig."""
        from src.config.unified import UnifiedConfig, ModelConfigSection

        config = UnifiedConfig()
        assert hasattr(config, "model_config")
        assert isinstance(config.model_config, ModelConfigSection)

    def test_model_config_section_defaults(self):
        """Test ModelConfigSection has family keys (defaults are empty)."""
        from src.config.unified import UnifiedConfig

        config = UnifiedConfig()
        defaults = config.model_config.defaults

        # Family keys exist (empty by default - settings come from contracts)
        assert "boosting" in defaults
        assert "neural" in defaults
        assert "transformer" in defaults
        assert "classical" in defaults
        assert "ensemble" in defaults
        assert "meta_learner" in defaults

        # Defaults are empty - model-specific settings come from ModelContract
        assert defaults["boosting"] == {}
        assert defaults["neural"] == {}

    def test_get_model_config(self):
        """Test get_model_config method."""
        from src.config.unified import UnifiedConfig

        config = UnifiedConfig()

        # With empty defaults, get_model_config returns empty dict
        boosting_config = config.model_config.get_model_config("xgboost", "boosting")
        assert boosting_config == {}

        # Test with model override
        config.model_config.overrides["xgboost"] = {"batch_size": 512}
        boosting_config = config.model_config.get_model_config("xgboost", "boosting")
        assert boosting_config["batch_size"] == 512

        # Can also set family defaults via config
        config.model_config.defaults["boosting"]["custom_setting"] = "value"
        boosting_config = config.model_config.get_model_config("xgboost", "boosting")
        assert boosting_config["custom_setting"] == "value"
        assert boosting_config["batch_size"] == 512

    def test_get_trainer_config_for_model(self):
        """Test get_trainer_config_for_model method."""
        from src.config.unified import UnifiedConfig

        unified = UnifiedConfig()
        trainer_config = unified.get_trainer_config_for_model("xgboost", horizon=20)

        assert trainer_config.model_name == "xgboost"
        assert trainer_config.horizon == 20
        assert trainer_config.primary_timeframe == "15min"  # From contract
        assert trainer_config.batch_size == unified.training.batch_size

    def test_get_trainer_config_for_model_with_overrides(self):
        """Test get_trainer_config_for_model with user overrides."""
        from src.config.unified import UnifiedConfig

        unified = UnifiedConfig()
        trainer_config = unified.get_trainer_config_for_model(
            "lstm",
            horizon=20,
            batch_size=512,
            max_epochs=200,
        )

        assert trainer_config.model_name == "lstm"
        assert trainer_config.batch_size == 512  # User override
        assert trainer_config.max_epochs == 200  # User override
        assert trainer_config.primary_timeframe == "5min"  # From contract

    def test_to_dict_includes_model_config(self):
        """Test to_dict includes model_config section."""
        from src.config.unified import UnifiedConfig

        config = UnifiedConfig()
        data = config.to_dict()

        assert "model_config" in data
        assert "defaults" in data["model_config"]
        assert "overrides" in data["model_config"]

    def test_from_dict_loads_model_config(self):
        """Test from_dict loads model_config section."""
        from src.config.unified import UnifiedConfig

        data = {
            "symbol": "MES",
            "model_config": {
                "defaults": {"boosting": {"primary_timeframe": "10min"}},
                "overrides": {"xgboost": {"batch_size": 512}},
            },
        }

        config = UnifiedConfig.from_dict(data, validate=False)

        assert config.model_config.defaults["boosting"]["primary_timeframe"] == "10min"
        assert config.model_config.overrides["xgboost"]["batch_size"] == 512


class TestPhase1Integration:
    """Integration tests for Phase 1 configuration layer."""

    def test_contract_to_trainer_config_flow(self):
        """Test flow from contract -> TrainerConfig."""
        from src.contracts import get_model_contract

        # Get contract
        contract = get_model_contract("lstm")
        assert contract.input_rank.value == 3
        assert contract.primary_timeframe == "5min"

        # Create TrainerConfig from contract
        config = TrainerConfig.from_model_contract("lstm", horizon=20)
        assert config.input_rank == 3
        assert config.primary_timeframe == "5min"
        assert config.adapter_id == "sequence"

    def test_ensemble_plan_to_trainer_configs(self):
        """Test creating TrainerConfigs from EnsemblePlan."""
        plan = EnsemblePlan.from_model_names(["xgboost", "lstm", "patchtst"])

        configs = []
        for model_config in plan.base_models:
            kwargs = model_config.to_trainer_config_kwargs()
            tc = TrainerConfig(horizon=20, **kwargs)
            configs.append(tc)

        assert len(configs) == 3

        # Check XGBoost config
        xgb_config = configs[0]
        assert xgb_config.model_name == "xgboost"
        assert xgb_config.input_rank == 2
        assert xgb_config.adapter_id == "tabular"

        # Check LSTM config
        lstm_config = configs[1]
        assert lstm_config.model_name == "lstm"
        assert lstm_config.input_rank == 3
        assert lstm_config.adapter_id == "sequence"

        # Check PatchTST config
        patchtst_config = configs[2]
        assert patchtst_config.model_name == "patchtst"
        assert patchtst_config.feature_mode == "raw"
        assert patchtst_config.mtf_mode == "multi_stream"

    def test_unified_config_heterogeneous_ensemble(self):
        """Test UnifiedConfig supports heterogeneous ensemble configuration."""
        from src.config.unified import UnifiedConfig

        unified = UnifiedConfig()

        # Create configs for heterogeneous ensemble
        xgb_config = unified.get_trainer_config_for_model("xgboost", horizon=20)
        lstm_config = unified.get_trainer_config_for_model("lstm", horizon=20)
        patchtst_config = unified.get_trainer_config_for_model("patchtst", horizon=20)

        # All should work and have appropriate settings
        assert xgb_config.adapter_id == "tabular"
        assert lstm_config.adapter_id == "sequence"
        assert patchtst_config.adapter_id == "sequence"

        # But different timeframes
        assert xgb_config.primary_timeframe == "15min"
        assert lstm_config.primary_timeframe == "5min"
        assert patchtst_config.primary_timeframe == "1min"
