"""
iTransformer Model - Inverted Transformer for 3-class prediction.

GPU-accelerated iTransformer with:
- Inverted attention: attention over variates (features) instead of time
- Per-window, per-variate instance normalisation (reference ``use_norm``)
- Each variate's whole window is embedded as one token (Linear over time)
- Cross-variate attention captures feature correlations
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference: Liu et al., "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (ICLR 2024); classification head as in the
thuml Time-Series-Library iTransformer (flatten all variate tokens -> linear).

Supports any NVIDIA GPU (GTX 10xx, RTX 20xx/30xx/40xx, Tesla T4/V100/A100).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import PredictionResult
from ..device import get_optimal_gpu_settings
from ..registry import register
from .base_rnn import BaseRNNModel
from .layers import WindowInstanceNorm, pre_ln_layer_with_attention

logger = logging.getLogger(__name__)


class TemporalEmbedding(nn.Module):
    """
    Inverted (variate-token) embedding for iTransformer.

    Projects each variate's full window of ``seq_len`` values to a
    ``d_model`` token with one shared linear map, followed by dropout
    (reference ``DataEmbedding_inverted``).

    Converts (batch, seq_len, n_variates) to (batch, n_variates, d_model).
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.temporal_proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, n_variates)

        Returns:
            Variate tokens, shape (batch, n_variates, d_model)
        """
        tokens: torch.Tensor = self.dropout(self.temporal_proj(x.transpose(1, 2)))
        return tokens


class iTransformerNetwork(nn.Module):
    """
    iTransformer network architecture for sequence classification.

    Architecture:
        Input (batch, seq_len, n_variates)   [4D inputs are flattened first]
        -> Instance norm per window and variate (use_norm)
        -> Variate-token embedding: (batch, n_variates, d_model)
        -> Transformer encoder (Pre-LN; attention over variates)
        -> LayerNorm -> GELU -> Dropout
        -> Flatten all variate tokens (+ the window statistics removed by the
           instance norm) -> Linear -> n_classes

    Why no positional encoding: like the reference, variate tokens carry no
    position embedding. Attention is permutation-equivariant over variates;
    identity is preserved because the head flattens the tokens in variate
    order, so each variate has its own head weights.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        activation: str = "gelu",
        n_classes: int = 3,
        use_norm: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_norm = use_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.instance_norm = WindowInstanceNorm(input_size) if use_norm else None

        # Variate-token embedding: project each variate's window to d_model
        self.temporal_embed = TemporalEmbedding(seq_len, d_model, dropout)

        # Transformer encoder (attention over variates)
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
        self.encoder_norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout_head = nn.Dropout(dropout)
        n_stats = 2 * input_size if use_norm else 0
        self.fc = nn.Linear(input_size * d_model + n_stats, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _flatten_timeframes(x: torch.Tensor) -> torch.Tensor:
        """(batch, n_tf, seq, feat) -> (batch, seq, n_tf * feat); 3D passes through."""
        if x.ndim == 4:
            batch, n_tf, seq, feat = x.shape
            x = x.permute(0, 2, 1, 3).reshape(batch, seq, n_tf * feat)
        return x

    def _embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Instance-normalise (optional) and embed: returns (tokens, window stats)."""
        x = self._flatten_timeframes(x)
        stats: torch.Tensor | None = None
        if self.instance_norm is not None:
            x, stats = self.instance_norm(x)
        return self.temporal_embed(x), stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through iTransformer.

        Args:
            x: Input tensor, shape (batch, seq_len, features)
               or 4D shape (batch, n_timeframes, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        tokens, stats = self._embed(x)  # (batch, n_variates, d_model)

        # Transformer encoder with attention over variates
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            tokens = checkpoint(self.transformer_encoder, tokens, use_reentrant=False)
        else:
            tokens = self.transformer_encoder(tokens)

        h = self.dropout_head(self.gelu(self.encoder_norm(tokens))).flatten(1)
        if stats is not None:
            h = torch.cat([h, stats], dim=-1)
        logits: torch.Tensor = self.fc(h)
        return logits

    def get_feature_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the variate-to-variate attention weights ``forward`` uses.

        Args:
            x: Input tensor, shape (batch, seq_len, features) or 4D

        Returns:
            Attention weights, shape (n_layers, batch, n_heads, n_features, n_features)
        """
        tokens, _ = self._embed(x)

        attention_weights = []
        for layer in self.transformer_encoder.layers:
            assert isinstance(layer, nn.TransformerEncoderLayer)
            tokens, attn_weights = pre_ln_layer_with_attention(layer, tokens)
            attention_weights.append(attn_weights)

        return torch.stack(attention_weights, dim=0)


@register(
    name="itransformer",
    family="transformer",
    description="iTransformer: Inverted Transformer with attention over features",
    aliases=["i_transformer", "inverted_transformer"],
)
class iTransformerModel(BaseRNNModel):
    """
    iTransformer classifier with GPU support.

    iTransformer inverts the attention mechanism of standard transformers:
    instead of attending over time positions, it attends over features.
    This allows the model to learn cross-feature correlations effectively.

    Inherits training infrastructure from BaseRNNModel:
    - GPU training with CUDA (any NVIDIA GPU)
    - Mixed precision with automatic dtype selection
    - AdamW optimizer with cosine annealing
    - Gradient clipping and early stopping

    Key Features:
    - Inverted attention: attends over features, not time
    - Instance normalisation per window and variate (``use_norm``, default
      True as in the reference); the removed window mean/log-std are fed to
      the classification head so level information is not lost
    - Temporal embedding: projects each feature's time series to d_model
    - Effective for multivariate time series with many correlated features
    - Typically needs fewer layers than standard transformers

    Architecture version 2.0: reference instance norm + flatten head, no
    learned feature positional encoding (1.0 checkpoints are refused on load).

    Note on Sequence Length:
        iTransformer is sensitive to sequence length since it's used in the
        temporal embedding. Changing seq_len at inference requires re-training.

    Example:
        >>> from src.models import ModelRegistry
        >>> model = ModelRegistry.create("itransformer", config={
        ...     "d_model": 256,
        ...     "n_heads": 8,
        ...     "n_layers": 2,
        ...     "sequence_length": 60,
        ... })
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """

    ARCH_VERSION = "2.0"  # 2.0: use_norm + flatten head, no feature positional encoding

    @property
    def requires_4d(self) -> bool:
        """iTransformer supports 4D multi-resolution input."""
        return True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        logger.debug(f"Initialized iTransformerModel with config: {self._config}")

    @property
    def is_production_safe(self) -> bool:
        """
        iTransformer processes all timesteps jointly in temporal embedding.

        Since the temporal embedding uses the full sequence, this model
        is not strictly causal. However, it does not have attention over
        time (only over features), so it's somewhat different from standard
        transformers.

        Returns:
            False - for consistency with other transformer models.
        """
        return False

    def _log_bidirectional_warning(self) -> None:
        """iTransformer doesn't have traditional bidirectional concerns."""
        pass  # Attention is over features, not time

    def get_default_config(self) -> dict[str, Any]:
        """Return default iTransformer hyperparameters.

        Architecture params (d_model, d_ff, n_heads, n_layers, batch_size)
        are VRAM-adaptive: they scale based on detected GPU memory via
        ``get_optimal_gpu_settings``.  Experiment-level overrides still win.
        """
        defaults = super().get_default_config()

        # Query VRAM-aware settings — returns conservative values for small GPUs
        gpu_settings = get_optimal_gpu_settings("itransformer")

        defaults.update(
            {
                # Architecture — adaptive to VRAM
                "d_model": gpu_settings.get("d_model", 128),
                "n_heads": gpu_settings.get("n_heads", 4),
                "n_layers": gpu_settings.get("n_layers", 2),
                "d_ff": gpu_settings.get("d_ff", 256),
                "dropout": 0.1,
                "activation": "gelu",
                "use_norm": True,  # per-window, per-variate instance norm
                # Training — adaptive batch size
                "sequence_length": gpu_settings.get("sequence_length", 60),
                "batch_size": gpu_settings.get("batch_size", 128),
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
        """Create the iTransformer network."""
        # Sequence length from the training data (restored by load()) or config
        seq_len = self._seq_len or self._config.get("sequence_length", 60)

        return iTransformerNetwork(
            input_size=input_size,
            seq_len=seq_len,
            d_model=self._config.get("d_model", 256),
            n_heads=self._config.get("n_heads", 8),
            n_layers=self._config.get("n_layers", 2),
            d_ff=self._config.get("d_ff", 512),
            dropout=self._config.get("dropout", 0.1),
            activation=self._config.get("activation", "gelu"),
            n_classes=self._n_classes,
            use_norm=self._config.get("use_norm", True),
            use_gradient_checkpointing=self._config.get("gradient_checkpointing", False),
        )

    def _get_model_type(self) -> str:
        """Return model type string."""
        return "itransformer"

    def _on_training_start(self, train_config: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """
        Log iTransformer-specific information at training start.

        Args:
            train_config: Training configuration
            seq_len: Sequence length of training data

        Returns:
            Dict with metadata for TrainingMetrics
        """
        n_features = self._n_features if self._n_features is not None else 0

        logger.info(
            f"iTransformer: seq_len={seq_len}, n_features={n_features}, "
            f"d_model={train_config.get('d_model', 256)}, "
            f"n_layers={train_config.get('n_layers', 2)}"
        )

        if n_features > 256:
            logger.warning(
                f"Large number of features ({n_features}). iTransformer attention "
                f"complexity is O(n_features^2). Consider feature selection."
            )

        return {
            "seq_len_embedded": seq_len,
            "n_feature_tokens": n_features,
        }

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with class probabilities.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            PredictionResult with predictions, probabilities, and metadata
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        # Validate sequence length matches training
        # For 4D input (batch, n_timeframes, seq_len, features), seq_len is dim 2
        input_seq_len = X.shape[2] if X.ndim == 4 else X.shape[1]
        if self._seq_len is not None and input_seq_len != self._seq_len:
            raise ValueError(
                f"Input sequence length ({input_seq_len}) does not match "
                f"training sequence length ({self._seq_len}). "
                f"iTransformer requires fixed sequence length."
            )

        self._model.eval()
        amp_dtype = self._amp_dtype

        # Zero-copy tensor on CPU; each batch moves to GPU individually
        X_tensor = torch.from_numpy(np.ascontiguousarray(X).astype(np.float32))

        all_probs = []
        batch_size = self._config.get("batch_size", 512)

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
                "model": "itransformer",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "n_layers": self._config.get("n_layers"),
                "seq_len": self._seq_len,
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importance from the classification head.

        The head flattens the variate tokens in feature order, so each feature
        owns a block of head weights (its token, plus its window mean/log-std
        when ``use_norm``). Importance is the L2 norm of that block.

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        itransformer_network = self._unwrapped_model()
        if not isinstance(itransformer_network, iTransformerNetwork):
            return None

        n_features = itransformer_network.input_size
        d_model = itransformer_network.d_model
        weights = itransformer_network.fc.weight.detach().cpu().numpy()  # (n_classes, in)
        n_classes = weights.shape[0]
        token_part = weights[:, : n_features * d_model].reshape(n_classes, n_features, d_model)
        sq = (token_part**2).sum(axis=(0, 2))
        if itransformer_network.use_norm:
            stats_part = weights[:, n_features * d_model :].reshape(n_classes, 2, n_features)
            sq = sq + (stats_part**2).sum(axis=(0, 1))
        importance = np.sqrt(sq)
        importance = importance / importance.sum()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def get_feature_attention_matrix(self, X: np.ndarray, sample_idx: int = 0) -> np.ndarray | None:
        """
        Extract feature-to-feature attention weights.

        Unlike standard transformers, iTransformer attention shows
        how features attend to each other, revealing correlations.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            sample_idx: Index of sample to extract attention for

        Returns:
            Attention matrix, shape (n_layers, n_heads, n_features, n_features)
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        self._validate_input_shape(X, "X")

        if sample_idx >= len(X):
            logger.warning(f"sample_idx {sample_idx} >= n_samples {len(X)}, using idx 0")
            sample_idx = 0

        itransformer_network = self._unwrapped_model()
        if not isinstance(itransformer_network, iTransformerNetwork):
            return None
        itransformer_network.eval()
        X_tensor = torch.from_numpy(
            np.ascontiguousarray(X[sample_idx : sample_idx + 1]).astype(np.float32)
        ).to(self._device)

        with torch.no_grad():
            attention = itransformer_network.get_feature_attention(X_tensor)
            # Shape: (n_layers, 1, n_heads, n_features, n_features)
            return attention[:, 0, :, :, :].cpu().numpy()


__all__ = [
    "iTransformerModel",
    "iTransformerNetwork",
    "TemporalEmbedding",
]
