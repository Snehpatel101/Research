"""
Labeling Strategy Factory.

Provides factory functions for creating labeling strategy instances
based on configuration. This centralizes strategy instantiation and
makes it easy to switch between different labeling approaches.
"""

import logging
from typing import Any

from .adaptive_barriers import AdaptiveTripleBarrierLabeler
from .base import LabelingStrategy, LabelingType
from .directional import DirectionalLabeler
from .meta import MetaLabeler
from .regression import RegressionLabeler
from .threshold import ThresholdLabeler
from .triple_barrier import TripleBarrierLabeler

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Registry mapping LabelingType to strategy class
_STRATEGY_REGISTRY: dict[LabelingType, type[LabelingStrategy]] = {
    LabelingType.TRIPLE_BARRIER: TripleBarrierLabeler,
    LabelingType.ADAPTIVE_TRIPLE_BARRIER: AdaptiveTripleBarrierLabeler,
    LabelingType.DIRECTIONAL: DirectionalLabeler,
    LabelingType.THRESHOLD: ThresholdLabeler,
    LabelingType.REGRESSION: RegressionLabeler,
    LabelingType.META: MetaLabeler,
}


def get_labeler(strategy: LabelingType | str, **config: Any) -> LabelingStrategy:
    """
    Create a labeling strategy instance.

    Parameters
    ----------
    strategy : LabelingType or str
        The type of labeling strategy to create.
        Can be a LabelingType enum or its string value.
    **config : Any
        Strategy-specific configuration parameters.
        See individual strategy classes for available parameters.

    Returns
    -------
    LabelingStrategy
        Configured labeling strategy instance

    Raises
    ------
    ValueError
        If strategy type is not recognized

    Examples
    --------
    >>> labeler = get_labeler(LabelingType.TRIPLE_BARRIER, k_up=1.5, k_down=1.0)
    >>> labeler = get_labeler('directional', threshold=0.001)
    >>> labeler = get_labeler('threshold', pct_up=0.01, pct_down=0.01)
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = LabelingType(strategy)
        except ValueError:
            valid_values = [t.value for t in LabelingType]
            raise ValueError(
                f"Unknown labeling strategy: '{strategy}'. " f"Valid values are: {valid_values}"
            )

    if strategy not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"No implementation registered for strategy: {strategy}. "
            f"Available strategies: {list(_STRATEGY_REGISTRY.keys())}"
        )

    strategy_class = _STRATEGY_REGISTRY[strategy]
    logger.debug(f"Creating {strategy_class.__name__} with config: {config}")

    return strategy_class(**config)


def get_available_strategies() -> list[LabelingType]:
    """
    Get list of available labeling strategies.

    Returns
    -------
    list[LabelingType]
        List of registered labeling strategy types
    """
    return list(_STRATEGY_REGISTRY.keys())


def register_strategy(labeling_type: LabelingType, strategy_class: type[LabelingStrategy]) -> None:
    """
    Register a new labeling strategy.

    This allows external code to add custom labeling strategies
    to the factory without modifying this module.

    Parameters
    ----------
    labeling_type : LabelingType
        The type identifier for the strategy
    strategy_class : type[LabelingStrategy]
        The strategy class to register

    Raises
    ------
    TypeError
        If strategy_class is not a subclass of LabelingStrategy
    """
    if not issubclass(strategy_class, LabelingStrategy):
        raise TypeError(
            f"strategy_class must be a subclass of LabelingStrategy, "
            f"got {strategy_class.__name__}"
        )

    logger.info(f"Registering strategy: {labeling_type} -> {strategy_class.__name__}")
    _STRATEGY_REGISTRY[labeling_type] = strategy_class


def create_multi_labeler(strategies: list[dict[str, Any]]) -> list[LabelingStrategy]:
    """
    Create multiple labeling strategies from configuration.

    This is useful when you want to generate multiple label types
    for the same dataset.

    Parameters
    ----------
    strategies : list[dict]
        List of strategy configurations, each containing:
        - 'type': LabelingType or string
        - Additional strategy-specific parameters

    Returns
    -------
    list[LabelingStrategy]
        List of configured labeling strategy instances

    Examples
    --------
    >>> strategies = [
    ...     {'type': 'triple_barrier', 'k_up': 1.5, 'k_down': 1.0},
    ...     {'type': 'directional', 'threshold': 0.001},
    ...     {'type': 'regression', 'scale_factor': 100}
    ... ]
    >>> labelers = create_multi_labeler(strategies)
    """
    labelers = []

    for config in strategies:
        if "type" not in config:
            raise ValueError("Each strategy config must have a 'type' key")

        strategy_type = config.pop("type")
        labeler = get_labeler(strategy_type, **config)
        labelers.append(labeler)

        # Restore config dict
        config["type"] = strategy_type

    return labelers
