"""
Tests for ModelContract and MODEL_CONTRACTS registry.

Phase 0 SNwH implementation tests.
"""

import json
import pytest

from src.contracts import (
    DataRank,
    FeatureMode,
    MTFMode,
    DataContract,
    ModelContract,
    MODEL_CONTRACTS,
    get_model_contract,
    list_model_contracts,
    get_models_by_rank,
    get_models_requiring_scaling,
    get_models_by_mtf_mode,
)


class TestModelContract:
    """Tests for ModelContract dataclass."""

    def test_basic_creation(self):
        """Test basic ModelContract creation."""
        contract = ModelContract(
            model_name="test_model",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            feature_mode=FeatureMode.ENGINEERED,
            mtf_mode=MTFMode.INDICATORS,
            primary_timeframe="5min",
            sequence_length=60,
            requires_scaling=True,
            scaler_type="robust",
        )
        assert contract.model_name == "test_model"
        assert contract.model_family == "neural"
        assert contract.input_rank == DataRank.SEQUENCE_3D
        assert contract.sequence_length == 60

    def test_requires_sequences_property(self):
        """Test requires_sequences property."""
        tabular = ModelContract(
            model_name="test",
            model_family="boosting",
            input_rank=DataRank.TABULAR_2D,
        )
        sequence = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
        )
        multi_tf = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.MULTI_TF_4D,
        )

        assert not tabular.requires_sequences
        assert sequence.requires_sequences
        assert multi_tf.requires_sequences

    def test_requires_multi_timeframe_property(self):
        """Test requires_multi_timeframe property."""
        tabular = ModelContract(
            model_name="test",
            model_family="boosting",
            input_rank=DataRank.TABULAR_2D,
        )
        sequence = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
        )
        multi_tf = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.MULTI_TF_4D,
        )

        assert not tabular.requires_multi_timeframe
        assert not sequence.requires_multi_timeframe
        assert multi_tf.requires_multi_timeframe

    def test_adapter_id_property(self):
        """Test adapter_id property."""
        tabular = ModelContract(
            model_name="test",
            model_family="boosting",
            input_rank=DataRank.TABULAR_2D,
        )
        sequence = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
        )
        multi_tf = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.MULTI_TF_4D,
        )

        assert tabular.adapter_id == "tabular"
        assert sequence.adapter_id == "sequence"
        assert multi_tf.adapter_id == "multi_stream"

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        contract = ModelContract(
            model_name="test_model",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            feature_mode=FeatureMode.ENGINEERED,
            mtf_mode=MTFMode.INDICATORS,
            primary_timeframe="5min",
            mtf_timeframes=("15min", "60min"),
            sequence_length=60,
            patch_length=16,
            requires_scaling=True,
            scaler_type="robust",
            min_features=30,
            max_features=150,
            description="Test model",
        )

        # Roundtrip
        data = contract.to_dict()
        contract2 = ModelContract.from_dict(data)

        assert contract2.model_name == contract.model_name
        assert contract2.model_family == contract.model_family
        assert contract2.input_rank == contract.input_rank
        assert contract2.feature_mode == contract.feature_mode
        assert contract2.mtf_mode == contract.mtf_mode
        assert contract2.mtf_timeframes == contract.mtf_timeframes
        assert contract2.sequence_length == contract.sequence_length
        assert contract2.patch_length == contract.patch_length

    def test_to_dict_json_serializable(self):
        """Test to_dict produces JSON-serializable output."""
        contract = ModelContract(
            model_name="test",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            mtf_timeframes=("5min", "15min"),
        )
        data = contract.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_repr(self):
        """Test string representation."""
        contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            primary_timeframe="5min",
        )
        repr_str = repr(contract)
        assert "lstm" in repr_str
        assert "neural" in repr_str
        assert "3D" in repr_str


class TestModelContractValidation:
    """Tests for ModelContract.validate_data_contract."""

    def test_validate_compatible_contract(self):
        """Test validation of compatible contracts."""
        model_contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
            min_features=30,
            max_features=150,
        )
        data_contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=80,
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert is_valid
        assert len(issues) == 0

    def test_validate_rank_mismatch(self):
        """Test validation catches rank mismatch."""
        model_contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
        )
        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,  # Wrong rank
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert not is_valid
        assert any("rank mismatch" in issue.lower() for issue in issues)

    def test_validate_sequence_length_mismatch(self):
        """Test validation catches sequence length mismatch."""
        model_contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )
        data_contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=80,
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=30,  # Wrong sequence length
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert not is_valid
        assert any("sequence length mismatch" in issue.lower() for issue in issues)

    def test_validate_too_few_features(self):
        """Test validation catches too few features."""
        model_contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            min_features=50,
        )
        data_contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=20,  # Too few
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert not is_valid
        assert any("too few features" in issue.lower() for issue in issues)

    def test_validate_too_many_features(self):
        """Test validation catches too many features."""
        model_contract = ModelContract(
            model_name="lstm",
            model_family="neural",
            input_rank=DataRank.SEQUENCE_3D,
            max_features=100,
        )
        data_contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=1000,
            n_features=200,  # Too many
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert not is_valid
        assert any("too many features" in issue.lower() for issue in issues)


class TestModelContractsRegistry:
    """Tests for MODEL_CONTRACTS registry."""

    def test_has_23_models(self):
        """Test registry has all 23 models."""
        assert len(MODEL_CONTRACTS) == 23

    def test_boosting_models(self):
        """Test boosting models are registered."""
        boosting_models = ["xgboost", "lightgbm", "catboost"]
        for model in boosting_models:
            assert model in MODEL_CONTRACTS
            contract = MODEL_CONTRACTS[model]
            assert contract.model_family == "boosting"
            assert contract.input_rank == DataRank.TABULAR_2D
            assert not contract.requires_scaling

    def test_classical_models(self):
        """Test classical models are registered."""
        classical_models = ["random_forest", "logistic", "svm"]
        for model in classical_models:
            assert model in MODEL_CONTRACTS
            contract = MODEL_CONTRACTS[model]
            assert contract.model_family == "classical"
            assert contract.input_rank == DataRank.TABULAR_2D

    def test_neural_models(self):
        """Test neural models are registered."""
        neural_models = [
            "lstm",
            "gru",
            "tcn",
            "transformer",
            "patchtst",
            "itransformer",
            "tft",
            "nbeats",
            "inceptiontime",
            "resnet1d",
        ]
        for model in neural_models:
            assert model in MODEL_CONTRACTS
            contract = MODEL_CONTRACTS[model]
            assert contract.model_family == "neural"
            assert contract.input_rank == DataRank.SEQUENCE_3D

    def test_ensemble_models(self):
        """Test ensemble models are registered."""
        ensemble_models = ["voting", "stacking", "blending"]
        for model in ensemble_models:
            assert model in MODEL_CONTRACTS
            contract = MODEL_CONTRACTS[model]
            assert contract.model_family == "ensemble"

    def test_meta_learner_models(self):
        """Test meta-learner models are registered."""
        meta_models = ["ridge_meta", "mlp_meta", "calibrated_meta", "xgboost_meta"]
        for model in meta_models:
            assert model in MODEL_CONTRACTS
            contract = MODEL_CONTRACTS[model]
            assert contract.model_family == "meta_learner"
            assert contract.feature_mode == FeatureMode.OOF_PROBS

    def test_all_contracts_immutable(self):
        """Test all contracts are frozen (immutable)."""
        for name, contract in MODEL_CONTRACTS.items():
            with pytest.raises(AttributeError):
                contract.model_name = "changed"

    def test_patchtst_multi_stream(self):
        """Test PatchTST uses multi-stream MTF mode."""
        contract = MODEL_CONTRACTS["patchtst"]
        assert contract.mtf_mode == MTFMode.MULTI_STREAM
        assert contract.feature_mode == FeatureMode.RAW
        assert len(contract.mtf_timeframes) > 0

    def test_tcn_longer_sequence(self):
        """Test TCN has longer sequence length."""
        contract = MODEL_CONTRACTS["tcn"]
        assert contract.sequence_length == 120  # Longer than default 60


class TestModelContractHelpers:
    """Tests for helper functions."""

    def test_get_model_contract_valid(self):
        """Test get_model_contract with valid name."""
        contract = get_model_contract("xgboost")
        assert contract.model_name == "xgboost"
        assert contract.model_family == "boosting"

    def test_get_model_contract_case_insensitive(self):
        """Test get_model_contract is case insensitive."""
        contract1 = get_model_contract("XGBOOST")
        contract2 = get_model_contract("XGBoost")
        contract3 = get_model_contract("xgboost")
        assert contract1 == contract2 == contract3

    def test_get_model_contract_invalid(self):
        """Test get_model_contract with invalid name."""
        with pytest.raises(ValueError, match="No contract for model"):
            get_model_contract("invalid_model")

    def test_list_model_contracts_all(self):
        """Test list_model_contracts returns all."""
        contracts = list_model_contracts()
        assert len(contracts) == 23

    def test_list_model_contracts_by_family(self):
        """Test list_model_contracts filtered by family."""
        boosting = list_model_contracts("boosting")
        assert len(boosting) == 3
        assert all(c.model_family == "boosting" for c in boosting.values())

        neural = list_model_contracts("neural")
        assert len(neural) == 10
        assert all(c.model_family == "neural" for c in neural.values())

    def test_get_models_by_rank(self):
        """Test get_models_by_rank."""
        tabular = get_models_by_rank(DataRank.TABULAR_2D)
        sequence = get_models_by_rank(DataRank.SEQUENCE_3D)

        assert "xgboost" in tabular
        assert "lstm" in sequence
        assert "xgboost" not in sequence
        assert "lstm" not in tabular

    def test_get_models_requiring_scaling(self):
        """Test get_models_requiring_scaling."""
        scaling_models = get_models_requiring_scaling()

        # Neural models require scaling
        assert "lstm" in scaling_models
        assert "gru" in scaling_models

        # Boosting models don't require scaling
        assert "xgboost" not in scaling_models
        assert "lightgbm" not in scaling_models

    def test_get_models_by_mtf_mode(self):
        """Test get_models_by_mtf_mode."""
        none_mode = get_models_by_mtf_mode(MTFMode.NONE)
        indicators_mode = get_models_by_mtf_mode(MTFMode.INDICATORS)
        multi_stream = get_models_by_mtf_mode(MTFMode.MULTI_STREAM)

        assert "tcn" in none_mode  # TCN uses single-TF
        assert "xgboost" in indicators_mode  # XGBoost uses MTF indicators
        assert "patchtst" in multi_stream  # PatchTST uses multi-stream


class TestModelContractCrossValidation:
    """Tests for ModelContract cross-validation with DataContract."""

    def test_xgboost_compatibility(self):
        """Test XGBoost contract compatibility."""
        model_contract = get_model_contract("xgboost")
        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert is_valid

    def test_lstm_compatibility(self):
        """Test LSTM contract compatibility."""
        model_contract = get_model_contract("lstm")
        data_contract = DataContract(
            symbol="MES",
            timeframe="5min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=80,
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert is_valid

    def test_patchtst_compatibility(self):
        """Test PatchTST contract compatibility."""
        model_contract = get_model_contract("patchtst")
        data_contract = DataContract(
            symbol="MES",
            timeframe="1min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=5,  # Raw OHLCV + volume
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=60,
        )

        is_valid, issues = model_contract.validate_data_contract(data_contract)
        assert is_valid
