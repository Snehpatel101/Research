"""Persistence functions for PipelineConfig."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def save_config_to_file(config: Any, path: Path) -> Path:
    """
    Save configuration to JSON file.

    Args:
        config: PipelineConfig instance
        path: Path to save config

    Returns:
        Path where config was saved
    """
    config_dict = asdict(config)
    config_dict["project_root"] = str(config.project_root)

    # Add metadata
    config_dict["_metadata"] = {
        "created_at": datetime.now().isoformat(),
        "config_version": "1.0",
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Configuration saved to {path}")
    return path


def load_config_from_file(cls: type[T], path: Path) -> T:
    """
    Load configuration from JSON file.

    Args:
        cls: PipelineConfig class
        path: Path to config JSON file

    Returns:
        PipelineConfig instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        config_dict = json.load(f)

    # Remove metadata if present
    config_dict.pop("_metadata", None)

    # Convert project_root back to Path
    if "project_root" in config_dict:
        config_dict["project_root"] = Path(config_dict["project_root"])

    logger.info(f"Configuration loaded from {path}")
    return cls(**config_dict)


def load_config_from_run_id(cls: type[T], run_id: str, project_root: Path | None = None) -> T:
    """
    Load configuration from a run ID.

    Args:
        cls: PipelineConfig class
        run_id: Run identifier
        project_root: Project root path

    Returns:
        PipelineConfig instance
    """
    if project_root is None:
        # Navigate from src/phase1/config/ to project root
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
    else:
        project_root = Path(project_root)

    config_path = project_root / "runs" / run_id / "config" / "config.json"
    return load_config_from_file(cls, config_path)


class PipelinePersistenceMixin:
    """Mixin providing save/load methods for PipelineConfig."""

    def save_config(self, path: Path | None = None) -> Path:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save config. If None, saves to run_config_dir/config.json

        Returns:
            Path where config was saved
        """
        if path is None:
            self.create_directories()
            path = self.run_config_dir / "config.json"
        return save_config_to_file(self, path)

    @classmethod
    def load_config(cls, path: Path) -> "PipelinePersistenceMixin":
        """Load configuration from JSON file."""
        return load_config_from_file(cls, path)

    @classmethod
    def load_from_run_id(
        cls, run_id: str, project_root: Path | None = None
    ) -> "PipelinePersistenceMixin":
        """Load configuration from a run ID."""
        return load_config_from_run_id(cls, run_id, project_root)
