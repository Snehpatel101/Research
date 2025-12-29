"""
Transformer Model - Vanilla Transformer encoder for 3-class prediction.

GPU-accelerated Transformer with:
- Positional encoding (sinusoidal)
- Multi-head self-attention
- Feed-forward networks with GELU activation
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)
- Layer normalization and dropout

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import PredictionOutput
from ..registry import register
from .base_rnn import BaseRNNModel

logger = logging.getLogger(__name__)


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds positional information to input embeddings using sine/cosine functions
    of different frequencies.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Handle odd d_model: slice div_term to match the number of even/odd positions
        pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# =============================================================================
# TRANSFORMER NETWORK
# =============================================================================


class TransformerNetwork(nn.Module):
    """
    Vanilla Transformer encoder for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Linear projection to d_model
        -> Positional encoding
        -> TransformerEncoder (n_layers)
           - Multi-head self-attention
           - Feed-forward network (d_ff hidden units)
           - Layer normalization
           - Residual connections
        -> Global average pooling
        -> LayerNorm + Dropout
        -> Linear -> d_model // 2
        -> GELU + Dropout
        -> Linear -> 3 classes
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str = "gelu",
        max_seq_len: int = 5000,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projection (features -> d_model)
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # (batch, seq, feature)
            norm_first=True,  # Pre-LN architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.gelu = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, n_classes)

        # Initialize weights
        self._init_weights()

        # Store attention weights for interpretability
        self._last_attention_weights: torch.Tensor | None = None

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            x: Input tensor, shape (batch, seq_len, features)
            return_attention: If True, store attention weights

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification head
        x = self.layer_norm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # (batch, n_classes)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from all layers.

        This is a simplified version that computes attention patterns
        by manually running through encoder layers.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Attention weights, shape (n_layers, batch, n_heads, seq_len, seq_len)
        """
        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        attention_weights = []

        # Manually iterate through encoder layers to capture attention
        for layer in self.transformer_encoder.layers:
            # Run self-attention (simplified - actual implementation is more complex)
            # Note: This is a basic extraction; production code would use hooks
            attn_output, attn_weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weights.detach())

            # Complete the layer forward pass
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))

        return torch.stack(attention_weights, dim=0)


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================


@register(
    name="transformer",
    family="neural",
    description="Transformer encoder with self-attention for time series",
    aliases=["tfm"],
)
class TransformerModel(BaseRNNModel):
    """
    Transformer encoder classifier with GPU support.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection:
      - bfloat16 for Ampere+ (RTX 30xx/40xx, A100, H100)
      - float16 for Volta/Turing (RTX 20xx, GTX 16xx, T4, V100)
      - float32 for older GPUs or CPU
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Features:
    - Multi-head self-attention for global context
    - Positional encoding for temporal awareness
    - Feed-forward networks with GELU activation
    - Layer normalization and residual connections
    - Attention weight extraction for interpretability

    Note on Causality:
        Standard Transformer self-attention is inherently bidirectional - each
        position attends to ALL other positions in the sequence (past and future
        within the window). This is fundamentally non-causal. For production
        trading models requiring strict causality, consider using LSTM/GRU with
        bidirectional=False, or TCN which uses causal convolutions.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("transformer", config={
        ...     "d_model": 256,
        ...     "n_heads": 8,
        ...     "n_layers": 3
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> attention = model.get_attention_weights(X_test[:10])
    """

    # Track whether the non-causal warning has been logged
    _noncausal_warning_logged: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._noncausal_warning_logged = False
        logger.debug(f"Initialized TransformerModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        Standard Transformer self-attention is inherently non-causal (attends
        to all positions). This implementation does NOT use causal masking,
        so it always returns False.

        Returns:
            False - standard Transformer is not production-safe for trading.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """
        Log a warning about non-causal self-attention (only once).

        Overrides parent method since Transformer has different concerns than
        bidirectional RNNs.
        """
        if self._noncausal_warning_logged:
            return

        logger.warning(
            "TRANSFORMER NON-CAUSAL ATTENTION: Standard self-attention allows each "
            "position to attend to ALL other positions in the sequence window, including "
            "future positions. This is inherently non-causal.\n"
            "Implications:\n"
            "  - Each prediction uses information from later timesteps in the window\n"
            "  - Patterns learned may not be available during real-time inference\n"
            "  - Model may perform differently in live trading vs backtesting\n"
            "Recommendations:\n"
            "  - For production trading: Use LSTM/GRU (bidirectional=False) or TCN\n"
            "  - For research/pattern analysis: Transformer is acceptable\n"
            "  - To add causality: Would require implementing causal attention mask"
        )
        self._noncausal_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default Transformer hyperparameters."""
        defaults = super().get_default_config()
        # Transformer-specific defaults
        defaults.update(
            {
                # Architecture
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 3,
                "d_ff": 512,
                "dropout": 0.1,
                "activation": "gelu",
                "max_seq_len": 5000,
                # Training
                "sequence_length": 128,  # Longer sequences for Transformer
                "batch_size": 128,  # Smaller batch for memory efficiency
                "max_epochs": 50,
                "learning_rate": 0.0001,  # Lower LR for Transformer
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "early_stopping_patience": 10,
                "warmup_epochs": 3,
            }
        )
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the Transformer network."""
        return TransformerNetwork(
            input_size=input_size,
            d_model=self._config.get("d_model", 256),
            n_heads=self._config.get("n_heads", 8),
            n_layers=self._config.get("n_layers", 3),
            d_ff=self._config.get("d_ff", 512),
            dropout=self._config.get("dropout", 0.1),
            activation=self._config.get("activation", "gelu"),
            max_seq_len=self._config.get("max_seq_len", 5000),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "transformer"

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions with class probabilities.

        Overrides parent to add attention weights to metadata.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionOutput with predictions, probabilities, and attention weights
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        self._model.eval()
        amp_dtype = self._amp_dtype

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        all_probs = []
        batch_size = self._config.get("batch_size", 128)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(batch)
                    probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)
        class_predictions_int = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_class(class_predictions_int)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={
                "model": "transformer",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "n_layers": self._config.get("n_layers"),
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance based on input projection weights.

        For Transformers, we use the magnitude of input projection weights
        as a proxy for feature importance.

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        # Get input projection weights: (d_model, input_size)
        weights = self._model.input_projection.weight.detach().cpu().numpy()

        # Compute L2 norm across d_model dimension for each feature
        importance = np.linalg.norm(weights, axis=0)

        # Normalize to sum to 1
        importance = importance / importance.sum()

        # Return as dict with feature indices
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def get_attention_weights(
        self, X: np.ndarray, sample_idx: int = 0
    ) -> np.ndarray | None:
        """
        Extract attention weights for interpretability.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Index of sample to extract attention for

        Returns:
            Attention weights, shape (n_layers, n_heads, seq_len, seq_len)
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        self._validate_input_shape(X, "X")

        if sample_idx >= len(X):
            logger.warning(
                f"sample_idx {sample_idx} >= n_samples {len(X)}, using idx 0"
            )
            sample_idx = 0

        self._model.eval()
        X_tensor = torch.tensor(X[sample_idx : sample_idx + 1], dtype=torch.float32).to(
            self._device
        )

        with torch.no_grad():
            # Extract attention weights
            attention = self._model.get_attention_weights(X_tensor)
            # Shape: (n_layers, 1, n_heads, seq_len, seq_len)
            return attention[:, 0, :, :, :].cpu().numpy()

    def get_attention_summary(
        self, X: np.ndarray, n_samples: int = 10
    ) -> dict[str, np.ndarray]:
        """
        Get attention statistics across multiple samples.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with attention statistics:
            - mean_attention: Mean attention across samples
            - std_attention: Std deviation of attention
            - max_attention: Maximum attention values
        """
        if not self._is_fitted:
            return {}

        n_samples = min(n_samples, len(X))
        all_attention = []

        for i in range(n_samples):
            attn = self.get_attention_weights(X, sample_idx=i)
            if attn is not None:
                all_attention.append(attn)

        if not all_attention:
            return {}

        attention_stack = np.stack(all_attention, axis=0)
        # Shape: (n_samples, n_layers, n_heads, seq_len, seq_len)

        return {
            "mean_attention": attention_stack.mean(axis=0),
            "std_attention": attention_stack.std(axis=0),
            "max_attention": attention_stack.max(axis=0),
            "min_attention": attention_stack.min(axis=0),
        }

    def visualize_attention_pattern(
        self, X: np.ndarray, sample_idx: int = 0, layer_idx: int = -1
    ) -> np.ndarray | None:
        """
        Get attention pattern suitable for visualization.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Sample to visualize
            layer_idx: Layer to visualize (-1 for last layer)

        Returns:
            Attention matrix averaged across heads, shape (seq_len, seq_len)
        """
        attention = self.get_attention_weights(X, sample_idx)
        if attention is None:
            return None

        # Select layer and average across heads
        layer_attention = attention[layer_idx]  # (n_heads, seq_len, seq_len)
        return layer_attention.mean(axis=0)  # (seq_len, seq_len)


__all__ = [
    "TransformerModel",
    "TransformerNetwork",
    "PositionalEncoding",
]
