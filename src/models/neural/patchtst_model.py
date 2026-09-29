"""
PatchTST Model - Patched Time Series Transformer for 3-class prediction.

GPU-accelerated PatchTST with:
- Patching: the window is cut into (overlapping) patches, each patch a token
- End padding by ``stride`` (reference ``padding_patch='end'``) so the newest
  bars are always inside the last patch
- RevIN-style per-window, per-variate instance normalisation (reference
  ``revin=1``), default on
- Two patch embeddings:
    * default (``channel_independent=False``): one token per patch that mixes
      all features (Linear over patch_len * n_features) — cheap on CPU
    * ``channel_independent=True``: the reference design — every feature is
      patched and encoded on its own with shared weights (n_features times
      the encoder work), then all per-feature tokens are flattened into the
      classification head
- Learnable positional encoding for patch sequences
- Mixed precision with automatic dtype selection (bfloat16/float16/float32)

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers" (ICLR 2023)

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
from .layers import WindowInstanceNorm

logger = logging.getLogger(__name__)


def patch_count(seq_len: int, patch_len: int, stride: int) -> int:
    """Number of patches after end-padding the window by ``stride`` bars.

    Matches the reference ``padding_patch='end'``:
    ``(seq_len - patch_len) // stride + 1`` patches plus one more, taken from
    the replication-padded tail, so the last bars are always covered.
    """
    padded = seq_len + stride
    if padded < patch_len:
        raise ValueError(
            f"PatchTST needs seq_len + stride >= patch_len, got seq_len={seq_len}, "
            f"stride={stride}, patch_len={patch_len}"
        )
    return (padded - patch_len) // stride + 1


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for time series.

    End-pads the window by ``stride`` bars (replicating the last bar), cuts it
    into patches of ``patch_len`` bars every ``stride`` bars and projects each
    patch to ``d_model``.

    Args:
        input_size: Number of input features per timestep
        patch_len: Length of each patch in timesteps
        stride: Stride between patches (patch_len for non-overlapping)
        d_model: Output dimension of patch embeddings
        channel_independent: If True, every feature is patched separately and
            projected by one shared Linear(patch_len -> d_model); the output
            has shape (batch * n_features, n_patches, d_model). If False, one
            token per patch mixes all features: Linear(patch_len * n_features
            -> d_model), output (batch, n_patches, d_model).
    """

    def __init__(
        self,
        input_size: int,
        patch_len: int,
        stride: int,
        d_model: int,
        channel_independent: bool = False,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.input_size = input_size
        self.channel_independent = channel_independent

        self.padding = nn.ReplicationPad1d((0, stride))
        projection_in = patch_len if channel_independent else patch_len * input_size
        self.projection = nn.Linear(projection_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings from input sequence.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            (batch, n_patches, d_model), or (batch * features, n_patches,
            d_model) when channel independent
        """
        batch_size, _, n_features = x.shape

        # (batch, seq_len, features) -> (batch, features, seq_len + stride)
        x = self.padding(x.transpose(1, 2))

        # (batch, features, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        n_patches = patches.shape[2]

        if self.channel_independent:
            patches = patches.reshape(batch_size * n_features, n_patches, self.patch_len)
        else:
            # (batch, n_patches, features * patch_len), feature-major per patch
            patches = patches.permute(0, 2, 1, 3).reshape(batch_size, n_patches, -1)

        result: torch.Tensor = self.projection(patches)
        return result


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
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"{x.size(1)} patches exceed max_patches={self.pe.size(1)}; "
                "increase max_patches or shorten the window"
            )
        x = x + self.pe[:, : x.size(1), :]
        result: torch.Tensor = self.dropout(x)
        return result


class PatchTSTNetwork(nn.Module):
    """
    PatchTST network architecture for sequence classification.

    Architecture:
        Input (batch, seq_len, features)   [4D inputs are flattened first]
        -> RevIN instance norm per window and variate (optional, default on)
        -> Patch embedding (end-padded patches; mixed or channel independent)
        -> Learnable positional encoding
        -> Transformer encoder (n_layers, Pre-LN)
        -> mixed:  mean over patches -> LayerNorm
           channel independent: LayerNorm -> flatten all features' patch tokens
        -> Dropout -> (+ window mean/log-std from RevIN) -> Linear -> n_classes

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
        revin: bool = True,
        revin_affine: bool = True,
        channel_independent: bool = False,
        seq_len: int | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_len = patch_len
        self.stride = stride
        self.revin = revin
        self.channel_independent = channel_independent
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.instance_norm = WindowInstanceNorm(input_size, affine=revin_affine) if revin else None

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            channel_independent=channel_independent,
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
        n_stats = 2 * input_size if revin else 0
        self.n_patches: int | None = (
            patch_count(seq_len, patch_len, stride) if seq_len is not None else None
        )
        if self.n_patches is not None and self.n_patches > max_patches:
            raise ValueError(f"{self.n_patches} patches exceed max_patches={max_patches}")
        if channel_independent:
            if self.n_patches is None:
                raise ValueError("channel_independent PatchTST needs seq_len to size its head")
            head_in = input_size * self.n_patches * d_model
        else:
            head_in = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_in + n_stats, n_classes)

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
               or 4D shape (batch, n_timeframes, seq_len, features)

        Returns:
            Output logits, shape (batch, n_classes)
        """
        # Handle 4D multi-resolution input: flatten timeframes into features
        if x.ndim == 4:
            batch, n_tf, seq, feat = x.shape
            # (batch, n_tf, seq, feat) -> (batch, seq, n_tf * feat)
            x = x.permute(0, 2, 1, 3).reshape(batch, seq, n_tf * feat)
        batch_size = x.shape[0]

        stats: torch.Tensor | None = None
        if self.instance_norm is not None:
            x, stats = self.instance_norm(x)

        # Patch tokens: (batch, n_patches, d_model) or (batch * features, ...)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
        else:
            x = self.transformer_encoder(x)

        if self.channel_independent:
            # (batch * features, n_patches, d_model) -> (batch, features * n_patches * d_model)
            x = self.layer_norm(x).reshape(batch_size, -1)
        else:
            x = self.layer_norm(x.mean(dim=1))  # (batch, d_model)

        x = self.dropout(x)
        if stats is not None:
            x = torch.cat([x, stats], dim=-1)
        logits: torch.Tensor = self.fc(x)
        return logits

    def get_n_patches(self, seq_len: int) -> int:
        """Number of patches for a given sequence length (end padding included)."""
        return patch_count(seq_len, self.patch_len, self.stride)


@register(
    name="patchtst",
    family="transformer",
    description="PatchTST: Patched Time Series Transformer (RevIN, end-padded patches)",
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
    - Patch embedding: reduces sequence length by patch_len/stride factor;
      the window is end-padded by ``stride`` so the newest bars are always in
      the last patch
    - RevIN instance normalisation (``revin``, default True, learnable affine
      via ``revin_affine``); the removed window mean/log-std feed the head
    - ``channel_independent`` (default False): the reference per-feature
      patching. The default mixes all features inside each patch token, which
      costs n_features times less encoder work (CPU friendly)
    - Learnable positional encoding
    - Pre-LN transformer architecture for stability

    Architecture version 2.0: end-padded patches and RevIN (1.0 checkpoints
    are refused on load).

    Note on Causality:
        PatchTST self-attention is unmasked: each patch attends to all other
        patches of the window. For strict causality, consider TCN, the causal
        Transformer, or LSTM/GRU with bidirectional=False.

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

    ARCH_VERSION = "2.0"  # 2.0: end-padded patches, RevIN, optional channel independence

    _noncausal_warning_logged: bool = False

    @property
    def requires_4d(self) -> bool:
        """PatchTST supports 4D multi-resolution input."""
        return True

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
        """Return default PatchTST hyperparameters.

        Architecture params (d_model, d_ff, n_heads, n_layers, batch_size)
        are VRAM-adaptive: they scale based on detected GPU memory via
        ``get_optimal_gpu_settings``.  Experiment-level overrides still win.
        """
        defaults = super().get_default_config()

        # Query VRAM-aware settings — returns conservative values for small GPUs
        gpu_settings = get_optimal_gpu_settings("patchtst")

        defaults.update(
            {
                # Architecture — adaptive to VRAM
                "d_model": gpu_settings.get("d_model", 128),
                "n_heads": gpu_settings.get("n_heads", 4),
                "n_layers": gpu_settings.get("n_layers", 3),
                "d_ff": gpu_settings.get("d_ff", 256),
                "patch_len": gpu_settings.get("patch_len", 16),
                "stride": gpu_settings.get("stride", 8),
                "dropout": 0.1,
                "activation": "gelu",
                "max_patches": 512,
                "revin": True,  # per-window, per-variate instance norm
                "revin_affine": True,
                "channel_independent": False,  # True = reference CI (n_features x cost)
                # Training — adaptive batch size
                "sequence_length": gpu_settings.get("sequence_length", 60),
                "batch_size": gpu_settings.get("batch_size", 64),
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
            revin=self._config.get("revin", True),
            revin_affine=self._config.get("revin_affine", True),
            channel_independent=self._config.get("channel_independent", False),
            seq_len=self._seq_len,
            use_gradient_checkpointing=self._config.get("gradient_checkpointing", False),
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
        n_patches = patch_count(seq_len, patch_len, stride)

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

        if self._model is None:
            raise RuntimeError("Model is not fitted")

        # The channel-independent head is sized for the training window
        input_seq_len = X.shape[2] if X.ndim == 4 else X.shape[1]
        if (
            self._config.get("channel_independent", False)
            and self._seq_len is not None
            and input_seq_len != self._seq_len
        ):
            raise ValueError(
                f"Input sequence length ({input_seq_len}) does not match training "
                f"sequence length ({self._seq_len}); channel-independent PatchTST "
                f"requires a fixed window."
            )

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
                "model": "patchtst",
                "d_model": self._config.get("d_model"),
                "n_heads": self._config.get("n_heads"),
                "n_layers": self._config.get("n_layers"),
                "patch_len": self._config.get("patch_len"),
                "stride": self._config.get("stride"),
                "channel_independent": self._config.get("channel_independent", False),
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return per-feature importance from the network weights.

        - Mixed patches (default): L2 norm of each feature's slice of the patch
          projection (patch vectors are laid out feature-major:
          ``index = feature * patch_len + t``).
        - Channel independent: the projection is shared by all features, so
          importance is the L2 norm of each feature's block of head weights
          (its flattened patch tokens plus, with RevIN, its window stats).

        Returns:
            Dict mapping feature indices to importance scores,
            or None if model is not fitted
        """
        if not self._is_fitted:
            return None

        patchtst_network = self._unwrapped_model()
        if not isinstance(patchtst_network, PatchTSTNetwork):
            return None

        n_features = patchtst_network.input_size
        if patchtst_network.channel_independent:
            weights = patchtst_network.fc.weight.detach().cpu().numpy()  # (n_classes, in)
            n_classes = weights.shape[0]
            n_token = weights.shape[1] - (2 * n_features if patchtst_network.revin else 0)
            token_part = weights[:, :n_token].reshape(n_classes, n_features, -1)
            sq = (token_part**2).sum(axis=(0, 2))
            if patchtst_network.revin:
                stats_part = weights[:, n_token:].reshape(n_classes, 2, n_features)
                sq = sq + (stats_part**2).sum(axis=(0, 1))
        else:
            # (d_model, n_features * patch_len) -> (d_model, n_features, patch_len)
            weights = patchtst_network.patch_embed.projection.weight.detach().cpu().numpy()
            weights = weights.reshape(patchtst_network.d_model, n_features, -1)
            sq = (weights**2).sum(axis=(0, 2))

        importance = np.sqrt(sq)
        importance = importance / importance.sum()

        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}


__all__ = [
    "PatchTSTModel",
    "PatchTSTNetwork",
    "PatchEmbedding",
    "LearnablePositionalEncoding",
    "patch_count",
]
