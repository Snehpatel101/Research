"""
LSTM Model - Long Short-Term Memory for 3-class prediction.

GPU-accelerated LSTM with:
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)
- AdamW optimizer with cosine annealing
- Gradient clipping and early stopping
- Bidirectional support

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from ..registry import register
from .base_rnn import BaseRNNModel, RNNNetwork

logger = logging.getLogger(__name__)


class LSTMNetwork(RNNNetwork):
    """
    LSTM neural network architecture.

    Architecture:
        Input (batch, seq_len, features)
        -> LSTM layers
        -> Take last hidden state
        -> LayerNorm + Dropout
        -> Linear -> hidden_size
        -> ReLU + Dropout
        -> Linear -> 3 classes
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

        # Initialize LSTM
        self.rnn = self._init_rnn(input_size, hidden_size, num_layers, dropout, bidirectional)

    def _init_rnn(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> nn.Module:
        """Initialize the LSTM layer."""
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )


@register(
    name="lstm",
    family="neural",
    description="LSTM recurrent network with GPU support and mixed precision",
    aliases=["long_short_term_memory"],
)
class LSTMModel(BaseRNNModel):
    """
    Long Short-Term Memory classifier with GPU support.

    Features:
    - CUDA GPU acceleration (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing LR
    - Gradient clipping for stability
    - Early stopping on validation loss
    - Bidirectional support

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("lstm", config={"hidden_size": 256})
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized LSTMModel with config: {self._config}")

    def get_default_config(self) -> dict[str, Any]:
        """Return default LSTM hyperparameters."""
        defaults = super().get_default_config()
        # LSTM-specific defaults (can override base)
        defaults.update({
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": False,
        })
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the LSTM network."""
        return LSTMNetwork(
            input_size=input_size,
            hidden_size=self._config.get("hidden_size", 256),
            num_layers=self._config.get("num_layers", 2),
            dropout=self._config.get("dropout", 0.3),
            bidirectional=self._config.get("bidirectional", False),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "lstm"

    def get_hidden_states(self, X: np.ndarray) -> np.ndarray | None:
        """
        Return hidden states from the LSTM for interpretability.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            Hidden states array, shape (n_samples, seq_len, hidden_size * num_directions)
            or None if model is not fitted
        """

        if not self._is_fitted:
            return None

        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            # Get RNN output (hidden states at each timestep)
            output, _ = self._model.rnn(X_tensor)
            return output.cpu().numpy()


__all__ = ["LSTMModel", "LSTMNetwork"]
