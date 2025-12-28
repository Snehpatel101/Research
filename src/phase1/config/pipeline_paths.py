"""Path properties mixin for PipelineConfig."""
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class PipelinePathMixin:
    """Mixin providing directory path properties for PipelineConfig."""

    # These attributes are defined in PipelineConfig dataclass
    project_root: Path
    run_id: str

    @property
    def data_dir(self) -> Path:
        """Data directory for this run."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def clean_data_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def final_data_dir(self) -> Path:
        return self.data_dir / "final"

    @property
    def splits_dir(self) -> Path:
        return self.data_dir / "splits"

    @property
    def run_dir(self) -> Path:
        """Directory for this specific run."""
        return self.project_root / "runs" / self.run_id

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
        directories: List[Path] = [
            self.raw_data_dir,
            self.clean_data_dir,
            self.features_dir,
            self.final_data_dir,
            self.splits_dir,
            self.run_dir,
            self.run_config_dir,
            self.run_logs_dir,
            self.run_artifacts_dir,
            self.results_dir,
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
