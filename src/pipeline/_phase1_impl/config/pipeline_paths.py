"""Path properties mixin for PipelineConfig."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelinePathMixin:
    """Mixin providing directory path properties for PipelineConfig.

    Path Strategy for Reproducibility:
    - Source data (raw_data_dir): Global path, immutable input
    - All outputs: Run-scoped under runs/{run_id}/data/

    This ensures each pipeline run produces isolated, reproducible outputs.
    """

    # These attributes are defined in PipelineConfig dataclass
    project_root: Path
    run_id: str

    @property
    def data_dir(self) -> Path:
        """Global data directory (for source/raw data only)."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Raw data directory - global, immutable source data."""
        return self.data_dir / "raw"

    @property
    def run_dir(self) -> Path:
        """Directory for this specific run - all outputs go here."""
        return self.project_root / "runs" / self.run_id

    @property
    def run_data_dir(self) -> Path:
        """Run-scoped data directory for all pipeline outputs."""
        return self.run_dir / "data"

    @property
    def clean_data_dir(self) -> Path:
        """Run-scoped cleaned data directory."""
        return self.run_data_dir / "clean"

    @property
    def features_dir(self) -> Path:
        """Run-scoped features directory."""
        return self.run_data_dir / "features"

    @property
    def final_data_dir(self) -> Path:
        """Run-scoped final labeled data directory."""
        return self.run_data_dir / "final"

    @property
    def splits_dir(self) -> Path:
        """Run-scoped train/val/test splits directory."""
        return self.run_data_dir / "splits"

    @property
    def run_config_dir(self) -> Path:
        """Config directory for this run."""
        return self.run_dir / "config"

    @property
    def run_logs_dir(self) -> Path:
        """Logs directory for this run."""
        return self.run_dir / "logs"

    @property
    def run_artifacts_dir(self) -> Path:
        """Artifacts directory for this run."""
        return self.run_dir / "artifacts"

    @property
    def results_dir(self) -> Path:
        """Results directory."""
        return self.project_root / "results"

    def create_directories(self) -> None:
        """Create all required directories for this run."""
        directories: list[Path] = [
            self.raw_data_dir,
            self.run_data_dir,
            self.clean_data_dir,
            self.features_dir,
            self.final_data_dir,
            self.splits_dir,
            self.run_config_dir,
            self.run_logs_dir,
            self.run_artifacts_dir,
            self.results_dir,
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
