"""Validation functions for PipelineConfig."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.phase1.pipeline_config import PipelineConfig


def validate_timeframe_config(target_timeframe: str) -> list[str]:
    """Validate target timeframe."""
    from src.phase1.config import SUPPORTED_TIMEFRAMES

    issues = []
    if target_timeframe not in SUPPORTED_TIMEFRAMES:
        issues.append(
            f"target_timeframe '{target_timeframe}' is not supported. "
            f"Supported: {SUPPORTED_TIMEFRAMES}"
        )
    return issues


def validate_ratios(train: float, val: float, test: float) -> list[str]:
    """Validate train/val/test ratios."""
    issues = []
    total = train + val + test
    if not (0.99 <= total <= 1.01):
        issues.append(f"Train/val/test ratios must sum to 1.0, got {total}")
    if not (0 < train < 1):
        issues.append(f"train_ratio must be between 0 and 1, got {train}")
    if not (0 < val < 1):
        issues.append(f"val_ratio must be between 0 and 1, got {val}")
    if not (0 < test < 1):
        issues.append(f"test_ratio must be between 0 and 1, got {test}")
    return issues


def validate_symbols(symbols: list[str], allow_batch: bool) -> list[str]:
    """Validate symbol configuration."""
    issues = []
    if not symbols:
        issues.append(
            "At least one symbol must be specified. " "Use --symbols MES or symbols=['MES']."
        )
    if len(symbols) > 1 and not allow_batch:
        issues.append(
            f"Batch processing of multiple symbols requires explicit opt-in. "
            f"Got {len(symbols)} symbols: {symbols}. "
            f"Use --batch-symbols flag or set allow_batch_symbols=True."
        )
    return issues


def validate_horizons(label_horizons: list[int], max_bars_ahead: int) -> list[str]:
    """Validate horizon configuration."""
    issues = []
    if not label_horizons:
        issues.append("At least one label horizon must be specified")
    for h in label_horizons:
        if h < 1:
            issues.append(f"Label horizon must be >= 1, got {h}")
    if max_bars_ahead < max(label_horizons, default=1):
        issues.append(
            f"max_bars_ahead ({max_bars_ahead}) must be >= max horizon ({max(label_horizons)})"
        )
    return issues


def validate_purge_embargo(purge_bars: int, embargo_bars: int) -> list[str]:
    """Validate purge and embargo settings."""
    issues = []
    if purge_bars < 0:
        issues.append(f"purge_bars must be >= 0, got {purge_bars}")
    if embargo_bars < 0:
        issues.append(f"embargo_bars must be >= 0, got {embargo_bars}")
    return issues


def validate_ga_params(
    pop_size: int, generations: int, crossover: float, mutation: float, elite: int
) -> list[str]:
    """Validate genetic algorithm parameters."""
    issues = []
    if pop_size < 2:
        issues.append(f"ga_population_size must be >= 2, got {pop_size}")
    if generations < 1:
        issues.append(f"ga_generations must be >= 1, got {generations}")
    if not (0 <= crossover <= 1):
        issues.append(f"ga_crossover_rate must be between 0 and 1, got {crossover}")
    if not (0 <= mutation <= 1):
        issues.append(f"ga_mutation_rate must be between 0 and 1, got {mutation}")
    if elite >= pop_size:
        issues.append(f"ga_elite_size ({elite}) must be < ga_population_size ({pop_size})")
    return issues


def validate_feature_params(
    sma_periods: list[int], ema_periods: list[int], atr_periods: list[int], rsi_period: int
) -> list[str]:
    """Validate feature engineering parameters."""
    issues = []
    if not sma_periods:
        issues.append("At least one SMA period must be specified")
    if not ema_periods:
        issues.append("At least one EMA period must be specified")
    if not atr_periods:
        issues.append("At least one ATR period must be specified")
    if rsi_period < 2:
        issues.append(f"rsi_period must be >= 2, got {rsi_period}")
    return issues


def validate_mtf_config(mtf_mode: str, mtf_timeframes: list[str]) -> list[str]:
    """Validate MTF configuration."""
    from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES

    issues = []
    valid_modes = ["bars", "indicators", "both"]
    if mtf_mode not in valid_modes:
        issues.append(f"mtf_mode must be one of {valid_modes}, got '{mtf_mode}'")
    for tf in mtf_timeframes:
        if tf not in MTF_TIMEFRAMES:
            issues.append(
                f"Unsupported MTF timeframe: '{tf}'. " f"Supported: {list(MTF_TIMEFRAMES.keys())}"
            )
    return issues


def validate_scaler_type(scaler_type: str) -> list[str]:
    """Validate scaler type."""
    valid = ["robust", "standard", "minmax", "quantile", "none"]
    if scaler_type not in valid:
        return [f"scaler_type must be one of {valid}, got '{scaler_type}'"]
    return []


def validate_feature_toggles(toggles: dict[str, bool] | None) -> list[str]:
    """Validate feature toggle keys."""
    if toggles is None:
        return []
    valid_keys = {"wavelets", "microstructure", "volume", "volatility"}
    issues = []
    for key in toggles.keys():
        if key not in valid_keys:
            issues.append(f"Unknown feature toggle key: '{key}'. Valid keys: {valid_keys}")
    return issues


def validate_barrier_overrides(overrides: dict[str, float] | None) -> list[str]:
    """Validate barrier override configuration."""
    if overrides is None:
        return []
    valid_keys = {"k_up", "k_down", "max_bars"}
    issues = []
    for key, value in overrides.items():
        if key not in valid_keys:
            issues.append(f"Unknown barrier override key: '{key}'. Valid keys: {valid_keys}")
        elif key in ("k_up", "k_down") and value <= 0:
            issues.append(f"barrier_overrides['{key}'] must be > 0, got {value}")
        elif key == "max_bars" and value < 1:
            issues.append(f"barrier_overrides['max_bars'] must be >= 1, got {value}")
    return issues


def validate_model_config_dict(model_config: dict[str, Any] | None) -> list[str]:
    """Validate model configuration dictionary."""
    if model_config is None:
        return []
    valid_keys = {"model_type", "base_models", "meta_learner", "sequence_length"}
    issues = []
    for key in model_config.keys():
        if key not in valid_keys:
            issues.append(f"Unknown model_config key: '{key}'. Valid keys: {valid_keys}")
    if "sequence_length" in model_config:
        seq = model_config["sequence_length"]
        if not isinstance(seq, int) or seq < 1:
            issues.append(f"model_config['sequence_length'] must be a positive integer, got {seq}")
    return issues


def validate_pipeline_config(config: "PipelineConfig") -> list[str]:
    """
    Run all validation checks on a PipelineConfig.

    Returns list of validation error messages (empty if valid).
    """
    from src.phase1.config import validate_feature_set_config

    issues = []
    issues.extend(validate_timeframe_config(config.target_timeframe))
    issues.extend(validate_ratios(config.train_ratio, config.val_ratio, config.test_ratio))
    issues.extend(validate_symbols(config.symbols, config.allow_batch_symbols))
    issues.extend(validate_horizons(config.label_horizons, config.max_bars_ahead))
    issues.extend(validate_purge_embargo(config.purge_bars, config.embargo_bars))
    issues.extend(
        validate_ga_params(
            config.ga_population_size,
            config.ga_generations,
            config.ga_crossover_rate,
            config.ga_mutation_rate,
            config.ga_elite_size,
        )
    )
    issues.extend(
        validate_feature_params(
            config.sma_periods, config.ema_periods, config.atr_periods, config.rsi_period
        )
    )
    issues.extend(validate_mtf_config(config.mtf_mode, config.mtf_timeframes))
    issues.extend(validate_scaler_type(config.scaler_type))
    issues.extend(validate_feature_set_config(config.feature_set))
    issues.extend(validate_feature_toggles(config.feature_toggles))
    issues.extend(validate_barrier_overrides(config.barrier_overrides))
    issues.extend(validate_model_config_dict(config.model_config))
    return issues
