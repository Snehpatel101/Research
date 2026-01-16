"""
Training modes package.

Provides specialized training strategies:
- walk_forward: Walk-forward validation training
- regime_aware: Regime-specific model training
- meta_labeling: Meta-labeling approach (Lopez de Prado 2018)

Usage:
    from src.training.modes import (
        WalkForwardTrainer,
        WalkForwardTrainerConfig,
        RegimeAwareTrainer,
        RegimeAwareConfig,
        MetaLabelingTrainer,
        MetaLabelingConfig,
    )

    # Walk-forward training
    wf_config = WalkForwardTrainerConfig(n_windows=5, window_type="expanding")
    wf_trainer = WalkForwardTrainer(experiment_config, wf_config)
    results = wf_trainer.run(container)

    # Regime-aware training
    regime_config = RegimeAwareConfig(regime_type="composite", train_separate_models=True)
    regime_trainer = RegimeAwareTrainer(experiment_config, regime_config)
    results = regime_trainer.run(container)

    # Meta-labeling training
    meta_config = MetaLabelingConfig(primary_model="xgboost", meta_model="logistic")
    meta_trainer = MetaLabelingTrainer(experiment_config, meta_config)
    results = meta_trainer.run(container)
"""

from .walk_forward import (
    WalkForwardTrainer,
    WalkForwardTrainerConfig,
    WalkForwardTrainingResult,
)
from .regime_aware import (
    RegimeAwareTrainer,
    RegimeAwareConfig,
    RegimeAwareTrainingResult,
    RegimeTrainingResult,
    detect_regimes,
    detect_volatility_regime,
    detect_trend_regime,
    detect_composite_regime,
    COMPOSITE_REGIMES,
    VOLATILITY_REGIMES,
    TREND_REGIMES,
)
from .meta_labeling import (
    MetaLabelingTrainer,
    MetaLabelingConfig,
    MetaLabelingResult,
    compute_bet_size_labels,
    compute_directional_meta_labels,
)

__all__ = [
    # Walk-forward
    "WalkForwardTrainer",
    "WalkForwardTrainerConfig",
    "WalkForwardTrainingResult",
    # Regime-aware
    "RegimeAwareTrainer",
    "RegimeAwareConfig",
    "RegimeAwareTrainingResult",
    "RegimeTrainingResult",
    "detect_regimes",
    "detect_volatility_regime",
    "detect_trend_regime",
    "detect_composite_regime",
    "COMPOSITE_REGIMES",
    "VOLATILITY_REGIMES",
    "TREND_REGIMES",
    # Meta-labeling
    "MetaLabelingTrainer",
    "MetaLabelingConfig",
    "MetaLabelingResult",
    "compute_bet_size_labels",
    "compute_directional_meta_labels",
]
