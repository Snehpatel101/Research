from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _ModeConfig:
    """Lightweight config for mode trainers (walk-forward, regime-aware, meta-labeling).

    This is an internal shim — not the canonical ExperimentConfig.
    The canonical ExperimentConfig lives in src.config.experiment.
    """

    symbol: str
    horizons: list[int]
    models: list[str]
    data_dir: Path = field(default_factory=lambda: Path("data/splits/scaled"))
    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
