"""
Training modes package.

Provides specialized training strategies:
- walk_forward: Walk-forward validation training

The regime-aware and meta-labeling modes are implemented in the canonical
training chain (src/models/training/regime_trainer.py and inline in
training_ops.py), not here.

Usage:
    from src.models.training.modes import (
        WalkForwardTrainer,
        WalkForwardTrainerConfig,
    )

    wf_config = WalkForwardTrainerConfig(n_windows=5, window_type="expanding")
    wf_trainer = WalkForwardTrainer(experiment_config, wf_config)
    results = wf_trainer.run(container)
"""

from .walk_forward import (
    WalkForwardTrainer,
    WalkForwardTrainerConfig,
    WalkForwardTrainingResult,
)

__all__ = [
    "WalkForwardTrainer",
    "WalkForwardTrainerConfig",
    "WalkForwardTrainingResult",
]
