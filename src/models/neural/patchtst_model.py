"""
PatchTST Model - Patched Time Series Transformer for 3-class prediction.

GPU-accelerated PatchTST with:
- Patch embedding: segments time series into non-overlapping patches
- Channel-independence: each feature processed independently then aggregated
- Learnable positional encoding for patch sequences
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers" (ICLR 2023)

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


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for time series.

    Splits the input sequence into non-overlapping patches and projects
    them to the model dimension. Supports optional stride for overlapping patches.

    Args:
        input_size: Number of input features per timestep
        patch_len: Length of each patch in timesteps
        stride: Stride between patches (default: patch_len for non-overlapping)
        d_model: Output dimension of patch embeddings
    """

    def __init__(
        self,
        input_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_size = input_size

        # Linear projection from patch to d_model
        # Each patch has shape (patch_len * input_size)
        self.projection = nn.Linear(patch_len * input_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings from input sequence.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Patch embeddings, shape (batch, n_patches, d_model)
        """
        batch_size, seq_len, n_features = x.shape

        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches using unfold
        # Reshape to (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Use unfold to extract patches: (batch, features, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Reshape to (batch, n_patches, features * patch_len)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, n_patches, -1)

        # Project to d_model
        return self.projection(patches)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patch sequences.

    Unlike sinusoidal encoding, positions are learned during training,
    which can capture task-specific positional patterns.
    """

    def __init__(self, d_model: int, max_patches: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.pe = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to patch embeddings.

        Args:
            x: Patch embeddings, shape (batch, n_patches, d_model)

        Returns:
            Position-encoded embeddings, shape (batch, n_patches, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PatchTSTNetwork(nn.Module):
    """
    PatchTST network architecture for sequence classification.

    Architecture:
        Input (batch, seq_len, features)
        -> Patch embedding (segment into patches)
        -> Learnable positional encoding
        -> Transformer encoder (n_layers)
        -> Global average pooling over patches
        -> LayerNorm + Dropout
        -> Linear classifier -> 3 classes

    Key features:
        - Patch-based: reduces sequence length, enabling longer effective context
        - Pre-LN architecture for stable training
        - GELU activation in feed-forward layers
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        patch_len: int,
        stride: int,
        dropout: float,
        activation: str = "gelu",
        max_patches: int = 512,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_len = patch_len
        self.stride = stride

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )

        # Positional encoding (learnable)
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_patches, dropout)

        # Transformer encoder layers (Pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Classification head
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PatchTST.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Create patch embeddings: (batch, n_patches, d_model)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification head
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_n_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length."""
        return (seq_len - self.patch_len) // self.stride + 1


@register(
    name="patchtst",
    family="neural",
    description="PatchTST: Patched Time Series Transformer with channel-independent patches",
    aliases=["patch_tst", "ptst"],
)
class PatchTSTModel(BaseRNNModel):
    """
    PatchTST classifier with GPU support.

    PatchTST segments time series into patches before applying transformer
    encoding. This reduces the effective sequence length while maintaining
    long-range dependencies, making it more efficient for long sequences.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Key Features:
    - Patch embedding: reduces sequence length by patch_len/stride factor
    - Learnable positional encoding
    - Pre-LN transformer architecture for stability
    - Efficient for long sequences (128+ timesteps)

    Note on Causality:
        Like vanilla Transformer, PatchTST uses bidirectional self-attention
        within the patch sequence. Each patch can attend to all other patches.
        For strict causality, consider TCN or LSTM with bidirectional=False.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("patchtst", config={
        ...     "d_model": 256,
        ...     "patch_len": 16,
        ...     "stride": 8,
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    _noncausal_warning_logged: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._noncausal_warning_logged = False
        logger.debug(f"Initialized PatchTSTModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        PatchTST uses bidirectional attention, so not production-safe.

        Returns:
            False - PatchTST is not production-safe for trading.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """Log warning about non-causal attention (only once)."""
        if self._noncausal_warning_logged:
            return

        logger.warning(
            "PATCHTST NON-CAUSAL ATTENTION: Patches can attend to all other patches "
            "including future patches within the sequence window. This is inherently "
            "non-causal.\n"
            "Implications:\n"
            "  - Each patch attends to patches from later timesteps\n"
            "  - Patterns may not generalize to real-time inference\n"
            "Recommendations:\n"
            "  - For production trading: Use TCN or LSTM (bidirectional=False)\n"
            "  - For research/pattern analysis: PatchTST is acceptable"
        )
        self._noncausal_warning_logged = True

    def get_default_config(self) -> dict[str, Any]:
        """Return default PatchTST hyperparameters."""
        defaults = super().get_default_config()
        defaults.update(
            {
                # Architecture
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 3,
                "d_ff": 512,
                "patch_len": 16,  # Patch length in timesteps
                "stride": 8,  # Stride (8 = 50% overlap with patch_len=16)
                "dropout": 0.1,
                "activation": "gelu",
                "max_patches": 512,
                # Training
                "sequence_length": 128,  # Longer sequences benefit from patching
                "batch_size": 128,
                "max_epochs": 50,
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "early_stopping_patience": 10,
                "warmup_epochs": 3,
            }
        )
        return defaults

    def _create_network(self, input_size: int) -> nn.Module:
        """Create the PatchTST network."""
        return PatchTSTNetwork(
            input_size=input_size,
            d_model=self._config.get("d_model", 256),
            n_heads=self._config.get("n_heads", 8),
            n_layers=self._config.get("n_layers", 3),
            d_ff=self._config.get("d_ff", 512),
            patch_len=self._config.get("patch_len", 16),
            stride=self._config.get("stride", 8),
            dropout=self._config.get("dropout", 0.1),
            activation=self._config.get("activation", "gelu"),
            max_patches=self._config.get("max_patches", 512),
            n_classes=self._n_classes,
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "patchtst"

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log PatchTST-specific information at training start.

        Args:
            train_config: Training configuration
            seq_len: Sequence length of training data

        Returns:
            Dict with metadata for TrainingMetrics
        """
        patch_len = train_config.get("patch_len", 16)
        stride = train_config.get("stride", 8)
        n_patches = self._model.get_n_patches(seq_len)

        logger.info(
            f"PatchTST: seq_len={seq_len}, patch_len={patch_len}, "
            f"stride={stride}, n_patches={n_patches}"
        )

        if n_patches < 4:
            logger.warning(
                f"Very few patches ({n_patches}). Consider decreasing patch_len "
                f"or stride, or increasing sequence_length."
            )

        return {
            "patch_len": patch_len,
            "stride": stride,
            "n_patches": n_patches,
        }

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions with class probabilities.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionOutput with predictions, probabilities, and metadata
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
                "model": "patchtst",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "n_layers": self._config.get("n_layers"),
                "patch_len": self._config.get("patch_len"),
                "stride": self._config.get("stride"),
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance based on patch embedding weights.

        For PatchTST, we analyze the patch projection weights to estimate
        which input features contribute most to the patch representations.

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        # Get patch projection weights: (d_model, patch_len * input_size)
        weights = self._model.patch_embed.projection.weight.detach().cpu().numpy()

        # Reshape to (d_model, patch_len, input_size)
        patch_len = self._model.patch_len
        input_size = self._model.input_size
        weights = weights.reshape(self._model.d_model, patch_len, input_size)

        # Compute importance per feature across all timesteps and d_model dims
        # Sum L2 norms across d_model and patch_len dimensions
        importance = np.sqrt((weights**2).sum(axis=(0, 1)))

        # Normalize to sum to 1
        importance = importance / importance.sum()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}


__all__ = [
    "PatchTSTModel",
    "PatchTSTNetwork",
    "PatchEmbedding",
    "LearnablePositionalEncoding",
]
