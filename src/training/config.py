from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    timeframe: str | None = None
    optimize_features: bool = False
    feature_opt_trials: int = 50
    optimize_hyperparams: bool = False
    hyperparam_opt_trials: int = 100
    sequence_length: int | None = None


@dataclass
class ExperimentConfig:
    symbol: str
    horizons: list[int]
    models: list[ModelConfig] | list[str]

    data_dir: Path = field(default_factory=lambda: Path("data/splits/scaled"))
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))

    cross_validate: bool = True
    cv_splits: int = 5

    build_ensemble: bool = False
    ensemble_method: str = "stacking"
    meta_learner: str = "ridge_meta"

    global_feature_optimization: bool = False
    global_hyperparam_optimization: bool = False

    parallel_training: bool = False
    n_jobs: int = -1

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        if self.models and isinstance(self.models[0], str):
            self.models = [
                ModelConfig(
                    name=m,
                    optimize_features=self.global_feature_optimization,
                    optimize_hyperparams=self.global_hyperparam_optimization,
                )
                for m in self.models
            ]
