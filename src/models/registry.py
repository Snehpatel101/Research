"""
ModelRegistry - Plugin system for model types.

The registry enables dynamic model discovery and instantiation:
- Register models using the @register decorator
- Create models by name
- List available models by family
- Get model metadata and requirements

Example:
    >>> @ModelRegistry.register("xgboost", family="boosting")
    ... class XGBoostModel(BaseModel):
    ...     pass
    ...
    >>> model = ModelRegistry.create("xgboost", config={"max_depth": 6})
    >>> ModelRegistry.list_models()
    {'boosting': ['xgboost', 'lightgbm'], 'neural': ['lstm', 'gru']}
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Plugin registry for model types.

    Provides a centralized mechanism for model registration and
    instantiation. Models are registered using the @register decorator
    and can be created by name.

    Class Attributes:
        _models: Dict mapping model names to model classes
        _families: Dict mapping family names to lists of model names
        _metadata: Dict mapping model names to metadata dicts

    Example:
        >>> # Register a model
        >>> @ModelRegistry.register("my_model", family="boosting")
        ... class MyModel(BaseModel):
        ...     pass
        ...
        >>> # Create an instance
        >>> model = ModelRegistry.create("my_model")
        ...
        >>> # List all models
        >>> ModelRegistry.list_models()
        {'boosting': ['my_model']}
    """

    _models: dict[str, type[BaseModel]] = {}
    _families: dict[str, list[str]] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        family: str,
        description: str = "",
        aliases: list[str] | None = None,
    ) -> Callable[[type[BaseModel]], type[BaseModel]]:
        """
        Decorator to register a model class.

        Args:
            name: Unique model identifier (e.g., "xgboost", "lstm")
            family: Model family (e.g., "boosting", "neural", "transformer")
            description: Human-readable description
            aliases: Alternative names for the model

        Returns:
            Decorator function that registers the model class

        Raises:
            ValueError: If model name is already registered

        Example:
            >>> @ModelRegistry.register(
            ...     name="xgboost",
            ...     family="boosting",
            ...     description="XGBoost gradient boosting classifier"
            ... )
            ... class XGBoostModel(BaseModel):
            ...     pass
        """
        aliases = aliases or []

        def decorator(model_class: type[BaseModel]) -> type[BaseModel]:
            # Validate model class
            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"Model class must be a subclass of BaseModel, "
                    f"got {model_class.__name__}"
                )

            # Check for duplicate registration
            if name in cls._models:
                raise ValueError(
                    f"Model '{name}' is already registered to "
                    f"{cls._models[name].__name__}"
                )

            # Register the model
            cls._models[name] = model_class

            # Register aliases
            for alias in aliases:
                if alias in cls._models:
                    logger.warning(
                        f"Alias '{alias}' already registered, skipping"
                    )
                else:
                    cls._models[alias] = model_class

            # Add to family
            if family not in cls._families:
                cls._families[family] = []
            cls._families[family].append(name)

            # Store metadata
            cls._metadata[name] = {
                "name": name,
                "family": family,
                "description": description,
                "aliases": aliases,
                "class": model_class.__name__,
            }

            logger.debug(
                f"Registered model '{name}' ({model_class.__name__}) "
                f"in family '{family}'"
            )

            return model_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Instantiate a registered model.

        Args:
            name: Model name or alias
            config: Model configuration dict
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Instantiated model

        Raises:
            ValueError: If model name is not registered

        Example:
            >>> model = ModelRegistry.create(
            ...     "xgboost",
            ...     config={"max_depth": 6, "n_estimators": 100}
            ... )
        """
        name_lower = name.lower().strip()

        if name_lower not in cls._models:
            available = sorted(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        model_class = cls._models[name_lower]
        return model_class(config=config, **kwargs)

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """
        Get a model class by name.

        Args:
            name: Model name or alias

        Returns:
            Model class (not instantiated)

        Raises:
            ValueError: If model name is not registered
        """
        name_lower = name.lower().strip()

        if name_lower not in cls._models:
            available = sorted(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        return cls._models[name_lower]

    @classmethod
    def list_models(cls) -> dict[str, list[str]]:
        """
        List all registered models by family.

        Returns:
            Dict mapping family names to lists of model names

        Example:
            >>> ModelRegistry.list_models()
            {
                'boosting': ['xgboost', 'lightgbm', 'catboost'],
                'neural': ['lstm', 'gru', 'tcn'],
                'transformer': ['patchtst', 'informer'],
                'classical': ['random_forest', 'logistic']
            }
        """
        return {family: list(models) for family, models in cls._families.items()}

    @classmethod
    def list_all(cls) -> list[str]:
        """
        List all registered model names.

        Returns:
            Sorted list of all model names (excluding aliases)
        """
        return sorted(
            name for name, meta in cls._metadata.items()
            if name == meta["name"]  # Exclude aliases
        )

    @classmethod
    def list_family(cls, family: str) -> list[str]:
        """
        List all models in a specific family.

        Args:
            family: Family name (e.g., "boosting", "neural")

        Returns:
            List of model names in the family

        Raises:
            ValueError: If family is not found
        """
        family_lower = family.lower().strip()

        if family_lower not in cls._families:
            available = sorted(cls._families.keys())
            raise ValueError(
                f"Unknown family '{family}'. Available families: {available}"
            )

        return list(cls._families[family_lower])

    @classmethod
    def get_metadata(cls, name: str) -> dict[str, Any]:
        """
        Get metadata for a registered model.

        Args:
            name: Model name

        Returns:
            Dict with model metadata including:
            - name: Model name
            - family: Model family
            - description: Human-readable description
            - aliases: Alternative names
            - class: Class name

        Raises:
            ValueError: If model name is not registered
        """
        name_lower = name.lower().strip()

        if name_lower not in cls._models:
            available = sorted(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        # Find the canonical name (in case alias was passed)
        model_class = cls._models[name_lower]
        for canonical, meta in cls._metadata.items():
            if meta["class"] == model_class.__name__:
                return meta.copy()

        # Fallback if not found in metadata
        return {
            "name": name_lower,
            "family": "unknown",
            "description": "",
            "aliases": [],
            "class": model_class.__name__,
        }

    @classmethod
    def get_model_info(cls, name: str) -> dict[str, Any]:
        """
        Get detailed info about a model including runtime properties.

        Args:
            name: Model name

        Returns:
            Dict with model info including:
            - name: Model name
            - family: Model family
            - requires_scaling: Whether scaling is needed
            - requires_sequences: Whether sequential input is needed
            - default_config: Default hyperparameters

        Raises:
            ValueError: If model name is not registered
        """
        name_lower = name.lower().strip()

        if name_lower not in cls._models:
            available = sorted(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        # Create a temporary instance to access properties
        model_class = cls._models[name_lower]
        instance = model_class()

        metadata = cls.get_metadata(name_lower)

        return {
            "name": metadata["name"],
            "family": instance.model_family,
            "description": metadata.get("description", ""),
            "requires_scaling": instance.requires_scaling,
            "requires_sequences": instance.requires_sequences,
            "default_config": instance.get_default_config(),
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Model name to check

        Returns:
            True if model is registered, False otherwise
        """
        return name.lower().strip() in cls._models

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered models.

        Primarily used for testing.
        """
        cls._models.clear()
        cls._families.clear()
        cls._metadata.clear()
        logger.debug("Cleared all registered models")

    @classmethod
    def families(cls) -> list[str]:
        """
        Get list of all model families.

        Returns:
            Sorted list of family names
        """
        return sorted(cls._families.keys())

    @classmethod
    def count(cls) -> int:
        """
        Get total number of registered models.

        Returns:
            Number of registered models (excluding aliases)
        """
        return len(cls._metadata)

    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a model is registered AND can be instantiated.

        This differs from is_registered() in that it also verifies the model's
        dependencies are available. Useful for optional dependencies like CatBoost.

        Args:
            name: Model name to check

        Returns:
            True if model is registered and can be instantiated, False otherwise
        """
        if not cls.is_registered(name):
            return False

        try:
            # Try to instantiate the model
            model_class = cls._models[name.lower().strip()]
            instance = model_class()
            return True
        except ImportError:
            return False
        except Exception:
            # Other errors might indicate configuration issues, not availability
            return True


# Convenience function for cleaner imports
def register(
    name: str,
    family: str,
    description: str = "",
    aliases: list[str] | None = None,
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """
    Convenience decorator for model registration.

    Equivalent to ModelRegistry.register().

    Example:
        >>> from src.models import register
        >>> @register("my_model", family="boosting")
        ... class MyModel(BaseModel):
        ...     pass
    """
    return ModelRegistry.register(
        name=name,
        family=family,
        description=description,
        aliases=aliases,
    )


__all__ = [
    "ModelRegistry",
    "register",
]
