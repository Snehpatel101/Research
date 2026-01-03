"""Configuration path constants."""

from pathlib import Path

# Note: 3 levels up from src/models/config/paths.py -> project root
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"
CONFIG_DIR = CONFIG_ROOT / "models"
TRAINING_CONFIG_PATH = CONFIG_ROOT / "pipeline" / "training.yaml"
CV_CONFIG_PATH = CONFIG_ROOT / "pipeline" / "cv.yaml"
