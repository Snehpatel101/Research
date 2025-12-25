"""
GRU Model - Gated Recurrent Unit for 3-class prediction.

GPU-accelerated GRU with:
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)
- AdamW optimizer with cosine annealing
- Gradient clipping and early stopping
- Simpler architecture than LSTM (fewer parameters)

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..registry import register
from .base_rnn import BaseRNNModel, RNNNetwork

logger = logging.getLogger(__name__)


class GRUNetwork(RNNNetwork):
    """
    GRU neural network architecture.

    Architecture:
        Input (batch, seq_len, features)
        -> GRU layers
        -> Take last hidden state
        -> LayerNorm + Dropout
        -> Linear -> hidden_size
        -> ReLU + Dropout
        -> Linear -> 3 classes

    GRU uses fewer gates than LSTM (2 vs 3), resulting in:
    - Fewer parameters
    - Faster training
    - Often comparable performance
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
        n_classes: int = 3,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            n_classes=n_classes,
        )

        # Initialize GRU
        self.rnn = self._init_rnn(input_size, hidden_size, num_layers, dropout, bidirectional)

    def _init_rnn(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> nn.Module:
        """Initialize the GRU layer."""
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


@register(
    name="gru",
    family="neural",
    description="GRU recurrent network with GPU support and mixed precision",
    aliases=["gated_recurrent_unit"],
)
class GRUModel(BaseRNNModel):
    """
    Gated Recurrent Unit classifier with GPU support.

    GRU is a simpler alternative to LSTM with:
    - Fewer parameters (2 gates vs 3)
    - Faster training
    - Often comparable performance on many tasks

    Features:
    - CUDA GPU acceleration (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing LR
    - Gradient clipping for stability
    - Early stopping on validation loss

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("gru", config={"hidden_size": 256})
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized GRUModel with config: {self._config}")

    def get_default_config(self) -> Dict[str, Any]:
        """Return default GRU hyperparameters."""
        defaults = super().get_default_config()
        # GRU-specific defaults
        defaults.update({
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            # GRU doesn't support bidirectional in this implementation
            # but we keep the parameter for consistency
            "bidirectional": False,
        })
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the GRU network."""
        return GRUNetwork(
            input_size=input_size,
            hidden_size=self._config.get("hidden_size", 256),
            num_layers=self._config.get("num_layers", 2),
            dropout=self._config.get("dropout", 0.3),
            bidirectional=self._config.get("bidirectional", False),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "gru"

    def get_hidden_states(self, X: "np.ndarray") -> Optional["np.ndarray"]:
        """
        Return hidden states from the GRU for interpretability.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            Hidden states array, shape (n_samples, seq_len, hidden_size)
            or None if model is not fitted
        """
        import numpy as np

        if not self._is_fitted:
            return None

        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            # Get RNN output (hidden states at each timestep)
            output, _ = self._model.rnn(X_tensor)
            return output.cpu().numpy()

    def get_gate_values(self, X: "np.ndarray") -> Optional[Dict[str, "np.ndarray"]]:
        """
        Return gate values from the GRU for interpretability.

        GRU has two gates:
        - Reset gate (r): controls how much past info to forget
        - Update gate (z): controls how much new info to accept

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            Dict with 'reset' and 'update' gate values,
            or None if model is not fitted
        """
        # Note: PyTorch doesn't expose gate values directly
        # This would require a custom GRU implementation
        # Returning None for now as a placeholder
        return None


__all__ = ["GRUModel", "GRUNetwork"]
