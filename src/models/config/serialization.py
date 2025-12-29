"""Configuration serialization functions."""
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}")


def save_config_json(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Saved config to {path}")
