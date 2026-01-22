# src/models/training/services/mode_router.py
"""
Service for routing to appropriate training mode.

Extracted from UnifiedTrainingOrchestrator to reduce its size and
isolate training mode selection logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from src.core import TrainingMode

if TYPE_CHECKING:
    import pandas as pd
    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ModeHandler:
    """Handler for a specific training mode."""

    mode: TrainingMode
    handler: Callable
    description: str


class ModeRouter:
    """
    Routes training to the appropriate mode handler.

    Responsibilities:
    - Register mode handlers
    - Select and invoke the appropriate handler based on config
    - Validate mode configuration

    Does NOT:
    - Implement training logic (handled by mode-specific trainers)
    - Manage training state (handled by orchestrator)
    """

    def __init__(self) -> None:
        """Initialize ModeRouter with empty handler registry."""
        self._handlers: dict[TrainingMode, ModeHandler] = {}

    def register(
        self,
        mode: TrainingMode,
        handler: Callable,
        description: str = "",
    ) -> None:
        """
        Register a handler for a training mode.

        Args:
            mode: TrainingMode enum value
            handler: Callable that accepts (df, additional_dfs) and performs training
            description: Optional description of the handler
        """
        self._handlers[mode] = ModeHandler(
            mode=mode,
            handler=handler,
            description=description,
        )
        logger.debug(f"Registered handler for mode: {mode.value}")

    def route(
        self,
        config: "PipelineConfig",
        df: "pd.DataFrame",
        additional_dfs: dict[str, "pd.DataFrame"] | None = None,
    ) -> None:
        """
        Route to appropriate training mode handler.

        Args:
            config: PipelineConfig with training_mode set
            df: Input DataFrame for training
            additional_dfs: Optional additional DataFrames

        Raises:
            ValueError: If training mode is not registered
        """
        mode = TrainingMode(config.training_mode)

        if mode not in self._handlers:
            available = [m.value for m in self._handlers.keys()]
            raise ValueError(
                f"Unknown training mode: {mode.value}. "
                f"Available modes: {available}"
            )

        handler = self._handlers[mode]
        logger.info(f"Routing to training mode: {mode.value}")

        if handler.description:
            logger.debug(f"  Description: {handler.description}")

        handler.handler(df, additional_dfs)

    def get_available_modes(self) -> list[str]:
        """Get list of available training modes."""
        return [m.value for m in self._handlers.keys()]

    def is_registered(self, mode: TrainingMode) -> bool:
        """Check if a mode is registered."""
        return mode in self._handlers


__all__ = ["ModeRouter", "ModeHandler"]
