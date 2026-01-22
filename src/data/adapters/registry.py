"""
Adapter Registry - Maps model requirements to adapters.

Phase 2 SNwH Implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.contracts import ModelContract

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """
    Registry for data adapters.

    Maps adapter IDs to adapter classes and provides
    automatic routing based on model contracts.
    """

    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_id: str):
        """
        Decorator to register an adapter class.

        Args:
            adapter_id: Unique adapter identifier

        Example:
            @AdapterRegistry.register("tabular")
            class TabularAdapter(BaseAdapter):
                pass
        """

        def decorator(adapter_class: type[BaseAdapter]) -> type[BaseAdapter]:
            if adapter_id in cls._adapters:
                logger.warning(f"Adapter '{adapter_id}' already registered, overwriting")
            cls._adapters[adapter_id] = adapter_class
            adapter_class.adapter_id = adapter_id
            logger.debug(f"Registered adapter: {adapter_id}")
            return adapter_class

        return decorator

    @classmethod
    def get(cls, adapter_id: str) -> type[BaseAdapter]:
        """
        Get adapter class by ID.

        Args:
            adapter_id: Adapter identifier

        Returns:
            Adapter class

        Raises:
            ValueError: If adapter not found
        """
        if adapter_id not in cls._adapters:
            raise ValueError(
                f"Unknown adapter '{adapter_id}'. " f"Available: {sorted(cls._adapters.keys())}"
            )
        return cls._adapters[adapter_id]

    @classmethod
    def create(
        cls,
        adapter_id: str,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Create adapter instance by ID.

        Args:
            adapter_id: Adapter identifier
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance
        """
        adapter_class = cls.get(adapter_id)
        return adapter_class(**kwargs)

    @classmethod
    def get_for_model(
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Get adapter for a model based on its contract.

        Args:
            model_name: Name of the model
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance appropriate for the model
        """
        from src.core.contracts import get_model_contract

        contract = get_model_contract(model_name)
        return cls.create(contract.adapter_id, **kwargs)

    @classmethod
    def get_for_contract(
        cls,
        contract: ModelContract,
        **kwargs: Any,
    ) -> BaseAdapter:
        """
        Get adapter for a model contract.

        Args:
            contract: ModelContract instance
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance appropriate for the contract
        """
        return cls.create(contract.adapter_id, **kwargs)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """List all registered adapter IDs."""
        return sorted(cls._adapters.keys())

    @classmethod
    def is_registered(cls, adapter_id: str) -> bool:
        """Check if an adapter is registered."""
        return adapter_id in cls._adapters

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (for testing)."""
        cls._adapters.clear()


def get_adapter(
    model_name: str | None = None,
    adapter_id: str | None = None,
    **kwargs: Any,
) -> BaseAdapter:
    """
    Convenience function to get an adapter.

    Either model_name or adapter_id must be provided.

    Args:
        model_name: Model name (uses contract to determine adapter)
        adapter_id: Direct adapter ID
        **kwargs: Arguments to pass to adapter constructor

    Returns:
        Adapter instance

    Example:
        # Get adapter by model name
        adapter = get_adapter(model_name="xgboost")
        result = adapter.transform(df)

        # Get adapter by ID
        adapter = get_adapter(adapter_id="sequence", sequence_length=60)
        result = adapter.transform(df)
    """
    if adapter_id:
        return AdapterRegistry.create(adapter_id, **kwargs)
    elif model_name:
        return AdapterRegistry.get_for_model(model_name, **kwargs)
    else:
        raise ValueError("Either model_name or adapter_id must be provided")


__all__ = [
    "AdapterRegistry",
    "get_adapter",
]
