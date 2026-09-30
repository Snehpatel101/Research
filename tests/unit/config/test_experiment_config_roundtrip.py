"""Test D1: ExperimentConfig round-trip (to_dict -> from_dict)."""

from src.config.experiment import ExperimentConfig


def test_roundtrip_preserves_key_fields():
    """from_dict(config.to_dict()) produces equivalent config."""
    config = ExperimentConfig(
        name="test_experiment",
        output_dir="test_output",
        run_id="fixed_run_id",
        random_seed=123,
        verbose=2,
    )
    config.data.symbol = "MGC"
    config.training.models = ["xgboost", "lightgbm", "lstm"]
    config.training.horizons = [10, 20]

    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)

    assert restored.symbol == "MGC"
    assert restored.horizons == [10, 20]
    assert restored.models == ["xgboost", "lightgbm", "lstm"]
    assert str(restored.output_dir) == str(config.output_dir)
    assert restored.name == "test_experiment"
    assert restored.run_id == "fixed_run_id"
    assert restored.random_seed == 123
    assert restored.verbose == 2


def test_roundtrip_preserves_training_section():
    """Training section fields survive round-trip."""
    config = ExperimentConfig(
        run_id="fixed",
    )
    config.training.training_mode = "walk_forward"
    config.training.cv_method = "cpcv"
    config.training.n_splits = 10
    config.training.batch_size = 256
    config.training.max_epochs = 50

    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.training.training_mode == "walk_forward"
    assert restored.training.cv_method == "cpcv"
    assert restored.training.n_splits == 10
    assert restored.training.batch_size == 256
    assert restored.training.max_epochs == 50


def test_roundtrip_preserves_evaluation_section():
    """Evaluation section fields survive round-trip."""
    config = ExperimentConfig(run_id="fixed")
    config.evaluation.run_backtest = True
    config.evaluation.initial_equity = 50000.0

    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.evaluation.run_backtest is True
    assert restored.evaluation.initial_equity == 50000.0


def test_to_dict_idempotent():
    """Calling to_dict() twice gives the same dict."""
    config = ExperimentConfig(
        name="idempotent_test",
        run_id="stable",
    )
    config.data.symbol = "MNQ"
    config.training.models = ["catboost"]
    config.training.horizons = [5]

    d1 = config.to_dict()
    d2 = config.to_dict()

    assert d1 == d2


# =============================================================================
# Phase 117 options: event sampling, fractional differentiation, probability sizing
# =============================================================================


def _configured() -> ExperimentConfig:
    config = ExperimentConfig(run_id="fixed")
    config.data.labeling.event_sampling = "cusum"
    config.data.labeling.cusum_threshold = 0.004
    config.data.labeling.cusum_vol_multiple = 2.5
    config.data.features.frac_diff.enabled = True
    config.data.features.frac_diff.d = 0.35
    config.data.features.frac_diff.columns = ["close", "high"]
    config.data.features.frac_diff.window = 64
    config.data.features.frac_diff.threshold = 1e-4
    config.evaluation.position_sizing = "probability"
    config.evaluation.bet_max_contracts = 8
    config.evaluation.bet_step_size = 0.1
    return config


def test_new_options_round_trip_through_dict():
    config = _configured()
    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.data.labeling.event_sampling == "cusum"
    assert restored.data.labeling.cusum_threshold == 0.004
    assert restored.data.labeling.cusum_vol_multiple == 2.5
    frac = restored.data.features.frac_diff
    assert (frac.enabled, frac.d, frac.columns, frac.window, frac.threshold) == (
        True,
        0.35,
        ["close", "high"],
        64,
        1e-4,
    )
    assert restored.evaluation.position_sizing == "probability"
    assert restored.evaluation.bet_max_contracts == 8
    assert restored.evaluation.bet_step_size == 0.1
    assert restored.to_dict() == config.to_dict()


def test_new_options_round_trip_through_yaml(tmp_path):
    config = _configured()
    config.data.labeling.cusum_threshold = "auto"
    config.data.features.frac_diff.d = "auto"
    path = tmp_path / "config.yaml"
    config.save_yaml(path)
    restored = ExperimentConfig.from_yaml(path)

    assert restored.data.labeling.cusum_threshold == "auto"
    assert restored.data.features.frac_diff.d == "auto"
    assert restored.to_dict() == config.to_dict()


def test_defaults_keep_todays_behavior():
    config = ExperimentConfig()

    assert config.data.labeling.event_sampling == "none"
    assert config.data.features.frac_diff.enabled is False
    assert config.evaluation.position_sizing == "fixed"
    assert config.validate() == []


def test_yaml_written_before_the_new_options_loads_with_defaults():
    old = {
        "data": {"symbol": "MES", "labeling": {"atr_period": 14}, "features": {}},
        "evaluation": {"run_backtest": True, "position_sizing": "kelly"},
    }
    config = ExperimentConfig.from_dict(old)

    assert config.data.labeling.event_sampling == "none"
    assert config.data.features.frac_diff.enabled is False
    assert config.evaluation.position_sizing == "kelly"


def test_validate_reports_bad_new_options():
    config = ExperimentConfig()
    config.data.labeling.event_sampling = "renko"
    config.data.labeling.cusum_threshold = -1.0
    config.data.labeling.cusum_vol_multiple = 0.0
    config.data.features.frac_diff.d = 1.7
    config.data.features.frac_diff.columns = ["volume"]
    config.data.features.frac_diff.window = 1

    text = " | ".join(config.validate())
    for fragment in (
        "event_sampling",
        "cusum_threshold",
        "cusum_vol_multiple",
        "frac_diff.d",
        "frac_diff.columns",
        "frac_diff.window",
    ):
        assert fragment in text


def test_factory_refuses_an_invalid_config(tmp_path):
    import pytest

    from src.factory import MLFactory

    config = ExperimentConfig(output_dir=tmp_path)
    config.data.labeling.event_sampling = "renko"
    with pytest.raises(ValueError, match="event_sampling"):
        MLFactory(config, verbose=0, enable_checkpoints=False)
