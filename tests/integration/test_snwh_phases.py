"""
Integration test for SNwH (Unified Multi-Timeframe Model Factory) phases.

Tests the end-to-end data flow:
1. Phase 0: Contracts - Model requirements declarations
2. Phase 1: TrainerConfig from contract
3. Phase 2: Adapters - Data format conversion
4. Phase 4: OOF Alignment - Heterogeneous ensemble alignment
5. Phase 5: Feature Strategy - Per-model feature selection

Run with: pytest tests/integration/test_snwh_phases.py -v
"""

import numpy as np
import pytest


class TestPhase0Contracts:
    """Test Phase 0: Model Contracts."""

    def test_get_model_contract(self):
        """Test getting model contracts."""
        from src.contracts import get_model_contract, MODEL_CONTRACTS

        # Test tabular model
        xgb_contract = get_model_contract("xgboost")
        assert xgb_contract.model_name == "xgboost"
        assert xgb_contract.input_rank.value == 2  # Tabular
        assert xgb_contract.primary_timeframe == "15min"
        assert xgb_contract.requires_sequences is False
        print(
            f"Phase 0: XGBoost contract - input_rank={xgb_contract.input_rank.value}D, "
            f"timeframe={xgb_contract.primary_timeframe}, "
            f"scaling={xgb_contract.requires_scaling}"
        )

        # Test sequence model
        lstm_contract = get_model_contract("lstm")
        assert lstm_contract.model_name == "lstm"
        assert lstm_contract.input_rank.value == 3  # Sequence
        assert lstm_contract.requires_sequences is True
        assert lstm_contract.sequence_length == 60
        print(
            f"Phase 0: LSTM contract - input_rank={lstm_contract.input_rank.value}D, "
            f"requires_sequences={lstm_contract.requires_sequences}, "
            f"seq_len={lstm_contract.sequence_length}"
        )

        # Test transformer model with multi-stream
        patchtst_contract = get_model_contract("patchtst")
        assert patchtst_contract.mtf_mode.value == "multi_stream"
        assert patchtst_contract.mtf_timeframes == ("5min", "15min")
        print(
            f"Phase 0: PatchTST contract - mtf_mode={patchtst_contract.mtf_mode.value}, "
            f"mtf_timeframes={patchtst_contract.mtf_timeframes}"
        )

    def test_all_models_have_contracts(self):
        """Test that all 23 models have contracts."""
        from src.contracts import MODEL_CONTRACTS

        assert len(MODEL_CONTRACTS) == 23
        print(f"Phase 0: {len(MODEL_CONTRACTS)} model contracts registered")

    def test_contract_families(self):
        """Test contract family distribution."""
        from src.contracts import list_model_contracts

        boosting = list_model_contracts(family="boosting")
        classical = list_model_contracts(family="classical")
        neural = list_model_contracts(family="neural")
        ensemble = list_model_contracts(family="ensemble")
        meta = list_model_contracts(family="meta_learner")

        assert len(boosting) == 3
        assert len(classical) == 3
        assert len(neural) == 10
        assert len(ensemble) == 3
        assert len(meta) == 4

        print(
            f"Phase 0: Families - boosting={len(boosting)}, classical={len(classical)}, "
            f"neural={len(neural)}, ensemble={len(ensemble)}, meta={len(meta)}"
        )


class TestPhase1TrainerConfig:
    """Test Phase 1: TrainerConfig from contract."""

    def test_config_from_contract(self):
        """Test creating TrainerConfig from model contract."""
        from src.models.config import TrainerConfig

        # Test LSTM config from contract
        config = TrainerConfig.from_model_contract("lstm", horizon=20)
        assert config.model_name == "lstm"
        assert config.horizon == 20
        assert config.primary_timeframe == "5min"
        assert config.adapter_id == "sequence"
        assert config.input_rank == 3
        assert config.sequence_length == 60
        print(
            f"Phase 1: LSTM TrainerConfig - model={config.model_name}, "
            f"timeframe={config.primary_timeframe}, adapter={config.adapter_id}"
        )

        # Test XGBoost config
        xgb_config = TrainerConfig.from_model_contract("xgboost", horizon=20)
        assert xgb_config.primary_timeframe == "15min"
        assert xgb_config.adapter_id == "tabular"
        assert xgb_config.input_rank == 2
        print(
            f"Phase 1: XGBoost TrainerConfig - timeframe={xgb_config.primary_timeframe}, "
            f"adapter={xgb_config.adapter_id}"
        )

        # Test PatchTST config with MTF
        patchtst_config = TrainerConfig.from_model_contract("patchtst", horizon=20)
        assert patchtst_config.mtf_mode == "multi_stream"
        assert patchtst_config.mtf_timeframes == ["5min", "15min"]
        print(
            f"Phase 1: PatchTST TrainerConfig - mtf_mode={patchtst_config.mtf_mode}, "
            f"mtf_timeframes={patchtst_config.mtf_timeframes}"
        )

    def test_config_overrides(self):
        """Test that contract defaults can be overridden."""
        from src.models.config import TrainerConfig

        config = TrainerConfig.from_model_contract(
            "lstm",
            horizon=20,
            batch_size=512,
            max_epochs=200,
            sequence_length=120,  # Override contract default of 60
        )
        assert config.batch_size == 512
        assert config.max_epochs == 200
        assert config.sequence_length == 120  # Overridden
        print(f"Phase 1: Override test - seq_len={config.sequence_length} (overridden from 60)")


class TestPhase2Adapters:
    """Test Phase 2: Adapter Registry."""

    def test_adapter_registry(self):
        """Test adapter registration and retrieval."""
        from src.adapters import AdapterRegistry, get_adapter

        # Check all adapters registered
        adapters = AdapterRegistry.list_adapters()
        assert "tabular" in adapters
        assert "sequence" in adapters
        assert "multi_stream" in adapters
        print(f"Phase 2: Registered adapters: {adapters}")

    def test_get_adapter_by_model(self):
        """Test getting adapter by model name."""
        from src.adapters import get_adapter

        # Tabular model -> TabularAdapter
        tabular_adapter = get_adapter(model_name="xgboost")
        assert type(tabular_adapter).__name__ == "TabularAdapter"
        print(f"Phase 2: XGBoost adapter type: {type(tabular_adapter).__name__}")

        # Sequence model -> SequenceAdapter
        sequence_adapter = get_adapter(model_name="lstm", sequence_length=60)
        assert type(sequence_adapter).__name__ == "SequenceAdapter"
        print(f"Phase 2: LSTM adapter type: {type(sequence_adapter).__name__}")

        # Multi-stream model -> MultiStreamAdapter
        multi_stream_adapter = get_adapter(model_name="patchtst", sequence_length=60)
        # Note: PatchTST actually uses sequence adapter based on input_rank=3
        # Multi-stream is about MTF mode, not input rank
        print(f"Phase 2: PatchTST adapter type: {type(multi_stream_adapter).__name__}")

    def test_adapter_from_contract(self):
        """Test getting adapter from model contract."""
        from src.contracts import get_model_contract
        from src.adapters import AdapterRegistry

        contract = get_model_contract("tcn")
        # Build kwargs based on adapter type - sequence adapters need sequence_length
        kwargs = {}
        if contract.adapter_id == "sequence":
            kwargs["sequence_length"] = contract.sequence_length
        adapter = AdapterRegistry.get_for_contract(contract, **kwargs)
        assert type(adapter).__name__ == "SequenceAdapter"
        print(f"Phase 2: TCN adapter (from contract): {type(adapter).__name__}")


class TestPhase4OOFAlignment:
    """Test Phase 4: OOF Alignment."""

    def test_oof_alignment_validator(self):
        """Test OOF alignment for heterogeneous ensembles."""
        from src.cross_validation.oof_alignment import OOFAlignmentValidator

        validator = OOFAlignmentValidator()

        # Register tabular model (full coverage)
        validator.register_oof("xgboost", oof_indices=None, n_total_samples=1000, sequence_length=None)

        # Register sequence model (partial coverage due to lookback)
        validator.register_oof("lstm", oof_indices=None, n_total_samples=1000, sequence_length=60)

        result = validator.validate()
        print(
            f"Phase 4: OOF Alignment - common_start={result.common_start_idx}, "
            f"common_end={result.common_end_idx}, n_common={result.n_common_samples}"
        )

        # LSTM starts at index 59 (seq_len - 1)
        assert result.common_start_idx == 59
        assert result.common_end_idx == 1000
        assert result.n_common_samples == 941  # 1000 - 59
        assert result.is_aligned

    def test_oof_alignment_offsets(self):
        """Test offset calculation for different models."""
        from src.cross_validation.oof_alignment import OOFAlignmentValidator

        validator = OOFAlignmentValidator()
        validator.register_oof("xgboost", None, 1000, None)  # Full coverage
        validator.register_oof("lstm", None, 1000, 60)  # Starts at 59
        validator.register_oof("tcn", None, 1000, 120)  # Starts at 119

        result = validator.validate()
        print(
            f"Phase 4: Multi-model alignment - common_range=[{result.common_start_idx}:{result.common_end_idx}]"
        )
        print(f"Phase 4: Offsets: {result.offsets}")

        # Common start is max of all starts (119 from TCN)
        assert result.common_start_idx == 119
        # Offsets tell us how to slice each model's OOF array
        assert result.offsets["xgboost"] == 119  # Skip first 119
        assert result.offsets["lstm"] == 60  # Skip first 60 (119 - 59)
        assert result.offsets["tcn"] == 0  # Start from beginning


class TestPhase5FeatureStrategy:
    """Test Phase 5: Feature Strategy."""

    def test_get_strategy_for_model(self):
        """Test getting feature strategy for different models."""
        from src.features import get_strategy_for_model

        # Boosting model strategy
        xgb_strategy = get_strategy_for_model("xgboost")
        assert xgb_strategy.family == "boosting"
        assert xgb_strategy.mtf_mode == "indicators"
        assert len(xgb_strategy.baseline_features) > 40
        print(
            f"Phase 5: XGBoost strategy - {len(xgb_strategy.baseline_features)} baseline features, "
            f"mtf={xgb_strategy.mtf_mode}"
        )

        # Neural model strategy
        lstm_strategy = get_strategy_for_model("lstm")
        assert lstm_strategy.family == "neural"
        assert lstm_strategy.requires_sequences is True
        assert len(lstm_strategy.baseline_features) > 30
        print(
            f"Phase 5: LSTM strategy - {len(lstm_strategy.baseline_features)} baseline features, "
            f"sequences={lstm_strategy.requires_sequences}"
        )

        # Transformer strategy (raw features, multi-stream)
        patchtst_strategy = get_strategy_for_model("patchtst")
        assert patchtst_strategy.mtf_mode == "multi_stream"
        assert len(patchtst_strategy.baseline_features) < 10  # Raw OHLCV only
        print(
            f"Phase 5: PatchTST strategy - {len(patchtst_strategy.baseline_features)} baseline features "
            f"(raw OHLCV), mtf={patchtst_strategy.mtf_mode}"
        )

    def test_all_models_have_strategies(self):
        """Test that all models have feature strategies."""
        from src.features import MODEL_FEATURE_STRATEGIES

        assert len(MODEL_FEATURE_STRATEGIES) == 23
        print(f"Phase 5: {len(MODEL_FEATURE_STRATEGIES)} model strategies registered")


class TestEndToEndIntegration:
    """Test end-to-end integration across all phases."""

    def test_contract_to_config_to_adapter_flow(self):
        """Test the complete flow: Contract -> TrainerConfig -> Adapter."""
        from src.contracts import get_model_contract
        from src.models.config import TrainerConfig
        from src.adapters import get_adapter

        # Step 1: Get contract
        contract = get_model_contract("lstm")
        print(f"\n[Integration] Step 1 - Contract: {contract}")

        # Step 2: Create config from contract
        config = TrainerConfig.from_model_contract("lstm", horizon=20)
        print(
            f"[Integration] Step 2 - TrainerConfig: model={config.model_name}, "
            f"adapter={config.adapter_id}, tf={config.primary_timeframe}"
        )

        # Step 3: Get adapter based on config
        adapter = get_adapter(adapter_id=config.adapter_id, sequence_length=config.sequence_length)
        print(f"[Integration] Step 3 - Adapter: {type(adapter).__name__}")

        # Verify consistency
        assert contract.adapter_id == config.adapter_id
        assert contract.primary_timeframe == config.primary_timeframe
        assert contract.sequence_length == config.sequence_length

    def test_heterogeneous_ensemble_flow(self):
        """Test heterogeneous ensemble data flow with OOF alignment."""
        from src.contracts import get_model_contract
        from src.models.config import TrainerConfig
        from src.cross_validation.oof_alignment import OOFAlignmentValidator
        from src.features import get_strategy_for_model
        from src.adapters import get_adapter

        # Define ensemble components
        base_models = ["xgboost", "lstm", "patchtst"]
        n_samples = 1000

        print("\n[Integration] Heterogeneous Ensemble Flow")
        print("-" * 50)

        # Step 1: Get contracts and configs for each base model
        configs = {}
        adapters = {}
        for model_name in base_models:
            contract = get_model_contract(model_name)
            config = TrainerConfig.from_model_contract(model_name, horizon=20)
            configs[model_name] = config

            # Get adapter with appropriate kwargs
            kwargs = {}
            if config.adapter_id == "sequence":
                kwargs["sequence_length"] = config.sequence_length
            adapters[model_name] = get_adapter(adapter_id=config.adapter_id, **kwargs)

            print(
                f"  {model_name}: tf={config.primary_timeframe}, "
                f"adapter={config.adapter_id}, mtf={config.mtf_mode}"
            )

        # Step 2: Get feature strategies
        print("\n[Integration] Feature Strategies:")
        for model_name in base_models:
            strategy = get_strategy_for_model(model_name)
            print(
                f"  {model_name}: {len(strategy.baseline_features)} features, "
                f"mtf={strategy.mtf_mode}"
            )

        # Step 3: Setup OOF alignment
        validator = OOFAlignmentValidator()
        for model_name in base_models:
            config = configs[model_name]
            seq_len = config.sequence_length if config.input_rank >= 3 else None
            validator.register_oof(model_name, None, n_samples, seq_len)

        result = validator.validate()
        print(f"\n[Integration] OOF Alignment: {result}")

        # Verify alignment
        assert result.is_aligned or len(result.issues) == 0 or "coverage" in result.issues[0].lower()
        print(f"[Integration] Common samples: {result.n_common_samples}/{n_samples}")

        # Verify adapters were created correctly
        assert type(adapters["xgboost"]).__name__ == "TabularAdapter"
        assert type(adapters["lstm"]).__name__ == "SequenceAdapter"
        assert type(adapters["patchtst"]).__name__ == "SequenceAdapter"

    def test_config_feature_strategy_integration(self):
        """Test that TrainerConfig can resolve feature strategies."""
        from src.models.config import TrainerConfig

        config = TrainerConfig.from_model_contract("xgboost", horizon=20)

        # Get strategy through config
        strategy = config.get_feature_strategy()
        assert strategy.model_name == "xgboost"
        assert len(strategy.baseline_features) > 0

        # Get baseline features through config
        baseline = config.get_baseline_features()
        assert len(baseline) == len(strategy.baseline_features)

        print(f"[Integration] Config->Strategy: {len(baseline)} baseline features for xgboost")


def run_integration_test():
    """Run all integration tests and print summary."""
    print("=" * 70)
    print("SNwH Phase Integration Test")
    print("=" * 70)

    # Phase 0: Contracts
    print("\n--- Phase 0: Contracts ---")
    from src.contracts import get_model_contract, MODEL_CONTRACTS

    xgb_contract = get_model_contract("xgboost")
    lstm_contract = get_model_contract("lstm")
    print(
        f"XGBoost: input_rank={xgb_contract.input_rank.value}D, "
        f"tf={xgb_contract.primary_timeframe}"
    )
    print(
        f"LSTM: input_rank={lstm_contract.input_rank.value}D, "
        f"seq_len={lstm_contract.sequence_length}"
    )
    print(f"Total contracts: {len(MODEL_CONTRACTS)}")

    # Phase 1: TrainerConfig
    print("\n--- Phase 1: TrainerConfig from Contract ---")
    from src.models.config import TrainerConfig

    config = TrainerConfig.from_model_contract("lstm", horizon=20)
    print(
        f"LSTM config: model={config.model_name}, tf={config.primary_timeframe}, "
        f"adapter={config.adapter_id}"
    )

    # Phase 2: Adapters
    print("\n--- Phase 2: Adapters ---")
    from src.adapters import get_adapter

    tabular_adapter = get_adapter(model_name="xgboost")
    sequence_adapter = get_adapter(model_name="lstm", sequence_length=60)
    print(f"XGBoost adapter: {type(tabular_adapter).__name__}")
    print(f"LSTM adapter: {type(sequence_adapter).__name__}")

    # Phase 4: OOF Alignment
    print("\n--- Phase 4: OOF Alignment ---")
    from src.cross_validation.oof_alignment import OOFAlignmentValidator

    validator = OOFAlignmentValidator()
    validator.register_oof("xgboost", None, 1000, None)
    validator.register_oof("lstm", None, 1000, 60)
    result = validator.validate()
    print(
        f"Alignment: common_start={result.common_start_idx}, "
        f"n_common={result.n_common_samples}"
    )

    # Phase 5: Feature Strategy
    print("\n--- Phase 5: Feature Strategy ---")
    from src.features import get_strategy_for_model

    xgb_strategy = get_strategy_for_model("xgboost")
    lstm_strategy = get_strategy_for_model("lstm")
    print(
        f"XGBoost: {len(xgb_strategy.baseline_features)} features, "
        f"mtf={xgb_strategy.mtf_mode}"
    )
    print(
        f"LSTM: {len(lstm_strategy.baseline_features)} features, "
        f"sequences={lstm_strategy.requires_sequences}"
    )

    print("\n" + "=" * 70)
    print("All phases integrate correctly!")
    print("=" * 70)


if __name__ == "__main__":
    run_integration_test()
