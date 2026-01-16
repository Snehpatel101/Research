from pathlib import Path

import pytest

from src.config.global_config import (
    GlobalConfig,
    load_global_config,
    get_global_config,
    set_global_config,
)


class TestGlobalConfig:
    def test_load_global_config_from_default_path(self):
        config = load_global_config()

        assert config.random_seed == 42
        assert config.timeframes.default_primary == "5min"
        assert len(config.timeframes.canonical_ladder) == 9
        assert config.splits.train == 0.70
        assert config.splits.val == 0.15
        assert config.splits.test == 0.15

    def test_timeframe_config(self):
        config = load_global_config()

        assert config.timeframes.default_primary == "5min"
        assert "1min" in config.timeframes.canonical_ladder
        assert "60min" in config.timeframes.canonical_ladder
        assert "240min" in config.timeframes.extended
        assert "1440min" in config.timeframes.extended

    def test_split_config(self):
        config = load_global_config()

        assert config.splits.train == 0.70
        assert config.splits.val == 0.15
        assert config.splits.test == 0.15
        assert config.splits.train + config.splits.val + config.splits.test == 1.0

    def test_purge_embargo_config(self):
        config = load_global_config()

        assert config.purge_embargo.purge_multiplier == 3.0
        assert config.purge_embargo.embargo_time_minutes == 7200
        assert config.purge_embargo.min_embargo_bars == 1440

    def test_horizons_config(self):
        config = load_global_config()

        assert config.horizons.supported == [1, 5, 10, 15, 20, 30, 60, 120]
        assert config.horizons.active == [5, 10, 15, 20]
        assert config.horizons.default == [5, 10, 15, 20]

    def test_features_config(self):
        config = load_global_config()

        assert config.features.sma_periods == [10, 20, 50, 100, 200]
        assert config.features.ema_periods == [9, 21, 50]
        assert config.features.atr_periods == [7, 14, 21]
        assert config.features.rsi_period == 14
        assert config.features.macd["fast"] == 12
        assert config.features.macd["slow"] == 26
        assert config.features.macd["signal"] == 9
        assert config.features.bollinger["period"] == 20
        assert config.features.bollinger["std"] == 2.0

    def test_feature_selection_config(self):
        config = load_global_config()

        assert config.features.selection.enabled is True
        assert config.features.selection.method == "mda"
        assert config.features.selection.cv_splits == 5

    def test_feature_generation_config(self):
        config = load_global_config()

        assert config.features.generation.default == "full"
        assert "full" in config.features.generation.modes
        assert "minimal" in config.features.generation.modes
        assert "custom" in config.features.generation.modes

    def test_mtf_config(self):
        config = load_global_config()

        assert config.mtf.default_mode == "indicators"
        assert config.mtf.default_timeframes == ["1min", "15min", "60min"]
        assert config.mtf.enabled is True

    def test_training_config(self):
        config = load_global_config()

        assert config.training.sequence_length == 60
        assert config.training.batch_size == 256
        assert config.training.max_epochs == 100
        assert config.training.early_stopping_patience == 15
        assert config.training.device == "auto"
        assert config.training.mixed_precision is True
        assert config.training.num_workers == 4
        assert config.training.pin_memory is True

    def test_calibration_config(self):
        config = load_global_config()

        assert config.calibration.enabled is True
        assert config.calibration.method == "auto"

    def test_ga_config(self):
        config = load_global_config()

        assert config.optimization.ga.population_size == 50
        assert config.optimization.ga.generations == 100
        assert config.optimization.ga.crossover_rate == 0.8
        assert config.optimization.ga.mutation_rate == 0.1
        assert config.optimization.ga.elite_size == 5
        assert config.optimization.ga.safe_mode is True

    def test_optuna_config(self):
        config = load_global_config()

        assert config.optimization.optuna.n_trials == 100
        assert config.optimization.optuna.timeout == 3600
        assert config.optimization.optuna.n_jobs == -1

    def test_cross_validation_config(self):
        config = load_global_config()

        assert config.cross_validation.n_splits == 5
        assert config.cross_validation.purge_multiplier == 3.0
        assert config.cross_validation.embargo_time_minutes == 7200

    def test_processing_config(self):
        config = load_global_config()

        assert config.processing.n_jobs == -1
        assert config.processing.allow_batch_symbols is False

    def test_scaler_config(self):
        config = load_global_config()

        assert config.scaler.default == "robust"

    def test_tracking_config(self):
        config = load_global_config()

        assert config.tracking.enabled is True
        assert config.tracking.backend == "local"

    def test_oom_recovery_config(self):
        config = load_global_config()

        assert config.oom_recovery.enabled is True
        assert config.oom_recovery.max_retries == 3
        assert config.oom_recovery.batch_reduction_factor == 0.5
        assert config.oom_recovery.min_batch_size == 8

    def test_to_dict_roundtrip(self):
        config = load_global_config()
        data = config.to_dict()
        config2 = GlobalConfig.from_dict(data)

        assert config.random_seed == config2.random_seed
        assert config.timeframes.default_primary == config2.timeframes.default_primary
        assert config.splits.train == config2.splits.train
        assert config.horizons.active == config2.horizons.active

    def test_get_global_config_singleton(self):
        config1 = get_global_config()
        config2 = get_global_config()

        assert config1 is config2

    def test_set_global_config(self):
        original = get_global_config()

        new_config = load_global_config()
        new_config.random_seed = 999
        set_global_config(new_config)

        retrieved = get_global_config()
        assert retrieved.random_seed == 999

        set_global_config(original)
