"""Config loading and validation for unified training system."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates training configurations."""

    @staticmethod
    def load_from_yaml(path: str | Path) -> dict:
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        ConfigLoader.validate(config)
        return config

    @staticmethod
    def load_from_params(**kwargs) -> dict:
        """Create config dict from parameters."""
        config = {
            "experiment": {
                "name": kwargs.get("experiment_name", "Untitled Experiment"),
            },
            "data": {
                "symbol": kwargs.get("symbol", "MES"),
                "horizons": kwargs.get("horizons", [20]),
                "timeframe": kwargs.get("timeframe", "5min"),
                "data_dir": kwargs.get("data_dir", "data/splits/scaled"),
            },
            "features": {
                "mode": kwargs.get("feature_mode", "full"),
                "optimize": kwargs.get("optimize_features", False),
                "mtf_strategy": kwargs.get("mtf_strategy", "indicators"),
            },
            "models": {
                "train_multiple": isinstance(kwargs.get("models"), list)
                and len(kwargs.get("models", [])) > 1,
                "model_list": [],
            },
            "ensemble": {
                "enabled": kwargs.get("build_ensemble", False),
                "method": kwargs.get("ensemble_method", "stacking"),
                "meta_learner": kwargs.get("meta_learner", "ridge_meta"),
            },
            "advanced": {
                "regime_aware": kwargs.get("regime_aware", False),
                "meta_labeling": kwargs.get("meta_labeling", False),
                "cross_validation": {
                    "enabled": kwargs.get("cross_validate", True),
                    "n_splits": kwargs.get("cv_splits", 5),
                },
            },
            "optimization": {
                "features": {
                    "n_trials": kwargs.get("feature_opt_trials", 50),
                },
                "hyperparams": {
                    "n_trials": kwargs.get("hyperparam_opt_trials", 100),
                },
            },
            "output": {
                "save_dir": kwargs.get("output_dir", "experiments/unified"),
                "save_predictions": kwargs.get("save_predictions", True),
                "save_feature_importance": kwargs.get("save_feature_importance", True),
                "generate_report": kwargs.get("generate_report", True),
            },
        }

        models = kwargs.get("models", ["xgboost"])
        if isinstance(models, str):
            if models == "all":
                from src.models import ModelRegistry

                models = ModelRegistry.list_all()
            else:
                models = [models]

        for model_name in models:
            model_config = {
                "name": model_name,
                "optimize_hyperparams": kwargs.get("optimize_hyperparams", False),
            }

            if model_name in [
                "lstm",
                "gru",
                "tcn",
                "transformer",
                "patchtst",
                "itransformer",
                "tft",
                "nbeats",
                "inceptiontime",
                "resnet1d",
            ]:
                model_config["seq_len"] = kwargs.get("seq_len", 60)

            config["models"]["model_list"].append(model_config)

        ConfigLoader.validate(config)
        return config

    @staticmethod
    def validate(config: dict) -> bool:
        """Validate config structure."""
        required_keys = ["data", "models", "features"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        if "symbol" not in config["data"]:
            raise ValueError("Missing data.symbol in config")

        if "horizons" not in config["data"]:
            raise ValueError("Missing data.horizons in config")

        if "model_list" not in config["models"] or not config["models"]["model_list"]:
            raise ValueError("Must specify at least one model in models.model_list")

        valid_feature_modes = ["full", "hft_only", "classical", "minimal", "research_2024"]
        feature_mode = config["features"].get("mode", "full")
        if feature_mode not in valid_feature_modes:
            raise ValueError(
                f"Invalid feature mode: {feature_mode}. Must be one of {valid_feature_modes}"
            )

        logger.info("Config validation passed")
        return True


def load_config_from_yaml(path: str | Path) -> dict:
    """Convenience function to load config from YAML."""
    return ConfigLoader.load_from_yaml(path)


def load_config_from_params(**kwargs) -> dict:
    """Convenience function to create config from parameters."""
    return ConfigLoader.load_from_params(**kwargs)


__all__ = ["ConfigLoader", "load_config_from_yaml", "load_config_from_params"]
