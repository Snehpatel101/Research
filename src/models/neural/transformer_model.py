"""
Transformer Model - Vanilla Transformer encoder for 3-class prediction.

GPU-accelerated Transformer with:
- Positional encoding (sinusoidal)
- Causal (masked) multi-head self-attention: position t attends to <= t only
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

from ..base import PredictionResult
from ..registry import register
from .base_rnn import BaseRNNModel
from .layers import pre_ln_layer_with_attention

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Handle odd d_model: slice div_term to match the number of even/odd positions
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
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
        pe_tensor: torch.Tensor = self.pe  # type: ignore[assignment]
        x = x + pe_tensor[:, : x.size(1), :]
        result: torch.Tensor = self.dropout(x)
        return result


# =============================================================================
# TRANSFORMER NETWORK
# =============================================================================


class TransformerNetwork(nn.Module):
    """
    Causal Transformer encoder for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Linear projection to d_model
        -> Positional encoding
        -> TransformerEncoder (n_layers, Pre-LN)
           - Causal multi-head self-attention (position t sees positions <= t)
           - Feed-forward network (d_ff hidden units)
           - Layer normalization
           - Residual connections
        -> Last position (the only one that has attended to the whole window)
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

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask to prevent attending to future positions.

        This is CRITICAL for production trading - without this mask, the model
        can "see" future positions during training, causing train-test mismatch.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        # Upper triangular mask: True where attention should be blocked
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # Convert to float mask: -inf where blocked, 0 where allowed
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Generate causal mask to prevent attending to future positions
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Transformer encoder with causal mask
        x = self.transformer_encoder(x, mask=causal_mask)  # (batch, seq_len, d_model)

        # Under the causal mask only the last position has attended to the whole
        # window; earlier positions saw progressively shorter prefixes.
        x = x[:, -1, :]  # (batch, d_model)

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
        Extract the attention weights ``forward`` actually uses, layer by layer.

        Replays each Pre-LN encoder layer with the same causal mask, so the
        weights are exactly zero above the diagonal (no attention to later
        positions).

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Attention weights, shape (n_layers, batch, n_heads, seq_len, seq_len)
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        causal_mask = self._generate_causal_mask(x.size(1), x.device)

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            assert isinstance(layer, nn.TransformerEncoderLayer)
            x, attn_weights = pre_ln_layer_with_attention(layer, x, causal_mask)
            attention_weights.append(attn_weights)

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
    - Causal multi-head self-attention (each position sees only its past)
    - Positional encoding for temporal awareness
    - Feed-forward networks with GELU activation
    - Layer normalization and residual connections
    - Attention weight extraction for interpretability

    Note on Causality:
        Self-attention runs under a causal mask: position t attends only to
        positions <= t, and the classifier reads the last position. The model
        is therefore causal, like TCN or a unidirectional LSTM/GRU.

    Architecture version 2.0: the classifier reads the last position instead of
    mean-pooling all positions (1.0 checkpoints are refused on load).

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

    ARCH_VERSION = "2.0"  # 2.0: last-position head under the causal mask

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized TransformerModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        Check if this model configuration is safe for production trading.

        Attention is causally masked (position t sees only positions <= t),
        so the model is causal.

        Returns:
            True - causal self-attention.
        """
        return True

    def _log_bidirectional_warning(self) -> None:
        """Causal attention: nothing to warn about."""

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

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with class probabilities.

        Overrides parent to add attention weights to metadata.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionResult with predictions, probabilities, and attention weights
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if self._model is None:
            raise RuntimeError("Model is not fitted")

        self._model.eval()
        amp_dtype = self._amp_dtype

        # Zero-copy tensor on CPU; each batch moves to GPU individually
        X_tensor = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))

        all_probs = []
        batch_size = self._config.get("batch_size", 128)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size].to(self._device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=self._use_amp):
                    logits = self._model(batch)
                    probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())

        probabilities = np.concatenate(all_probs, axis=0)
        class_predictions_int = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_class(class_predictions_int)
        confidence = np.max(probabilities, axis=1)

        return PredictionResult(
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
        transformer_network = self._unwrapped_model()
        if not isinstance(transformer_network, TransformerNetwork):
            return None
        weights = transformer_network.input_projection.weight.detach().cpu().numpy()

        # Compute L2 norm across d_model dimension for each feature
        importance = np.linalg.norm(weights, axis=0)

        # Normalize to sum to 1
        importance = importance / importance.sum()

        # Return as dict with feature indices
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def get_attention_weights(  # type: ignore[override]
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
            logger.warning(f"sample_idx {sample_idx} >= n_samples {len(X)}, using idx 0")
            sample_idx = 0

        transformer_network = self._unwrapped_model()
        if not isinstance(transformer_network, TransformerNetwork):
            return None
        transformer_network.eval()
        X_tensor = torch.from_numpy(
            np.ascontiguousarray(X[sample_idx : sample_idx + 1]).astype(np.float32)
        ).to(self._device)

        with torch.no_grad():
            # Extract attention weights
            attention = transformer_network.get_attention_weights(X_tensor)
            # Shape: (n_layers, 1, n_heads, seq_len, seq_len)
            return attention[:, 0, :, :, :].cpu().numpy()

    def get_attention_summary(self, X: np.ndarray, n_samples: int = 10) -> dict[str, np.ndarray]:
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
        result: np.ndarray = layer_attention.mean(axis=0)  # (seq_len, seq_len)
        return result


__all__ = [
    "TransformerModel",
    "TransformerNetwork",
    "PositionalEncoding",
]
