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
