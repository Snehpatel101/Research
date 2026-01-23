"""Model instantiation factory."""

import logging
from typing import Any

from src.models import ModelRegistry, TrainerConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Creates model instances from configuration."""

    @staticmethod
    def create_trainer(model_config: dict[str, Any], horizon: int) -> TrainerConfig:
        """
        Create TrainerConfig from model configuration.

        Args:
            model_config: Dict with model configuration
            horizon: Label horizon

        Returns:
            TrainerConfig instance
        """
        model_name = model_config["name"]

        if not ModelRegistry.is_registered(model_name):
            raise ValueError(f"Model not registered: {model_name}")

        config_kwargs = {
            "model_name": model_name,
            "horizon": horizon,
        }

        if "seq_len" in model_config:
            config_kwargs["sequence_length"] = model_config["seq_len"]

        if "batch_size" in model_config:
            config_kwargs["batch_size"] = model_config["batch_size"]

        if "max_epochs" in model_config:
            config_kwargs["max_epochs"] = model_config["max_epochs"]

        return TrainerConfig(**config_kwargs)

    @staticmethod
    def get_model_family(model_name: str) -> str:
        """Get model family for given model."""
        model_info = ModelRegistry.get_model_info(model_name)
        return str(model_info["family"])


__all__ = ["ModelFactory"]
