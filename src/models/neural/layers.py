"""
Shared building blocks for the attention-based neural models.

- ``WindowInstanceNorm``: per-window, per-variate instance normalisation
  (the ``use_norm`` of iTransformer / ``revin`` of PatchTST).
- ``pre_ln_layer_with_attention``: re-runs one Pre-LN
  ``nn.TransformerEncoderLayer`` step while returning its attention weights,
  so interpretability extraction computes exactly what ``forward`` computes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WindowInstanceNorm(nn.Module):
    """Normalise every variate of every window by its own mean and std over time.

    Mirrors the reference implementations: ``use_norm`` in thuml/iTransformer
    (Liu et al., 2024) and RevIN in PatchTST (Nie et al., 2023; Kim et al.,
    2022), both computing ``(x - mean_t) / sqrt(var_t + eps)`` per variate,
    with RevIN's optional learnable per-variate affine transform.

    The forecasting references *de-normalise* their output with the same
    statistics, which restores each window's level and scale. A classifier has
    no output to de-normalise, so ``forward`` also returns the statistics
    (mean and log-std per variate) for the caller to feed to its head — the
    classification analogue of that de-normalisation step. Without it, the
    network could not tell a window of, e.g., an RSI pinned high from one pinned
    low, which is often exactly the signal.

    Input and output shape: (batch, seq_len, n_variates). Stats shape:
    (batch, 2 * n_variates) laid out as ``[means..., log_stds...]``.
    """

    def __init__(self, n_variates: int, affine: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_variates = n_variates
        self.eps = eps
        self.affine_weight: nn.Parameter | None = None
        self.affine_bias: nn.Parameter | None = None
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_variates))
            self.affine_bias = nn.Parameter(torch.zeros(n_variates))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        std = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        normed = centered / std
        if self.affine_weight is not None and self.affine_bias is not None:
            normed = normed * self.affine_weight + self.affine_bias
        stats = torch.cat([mean.squeeze(1), torch.log(std.squeeze(1))], dim=-1)
        return normed, stats


def pre_ln_layer_with_attention(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Pre-LN (``norm_first=True``) encoder-layer step plus its attention.

    Reproduces ``nn.TransformerEncoderLayer.forward`` for ``norm_first=True``:
    ``x + SA(LN1(x))`` then ``x + FF(LN2(x))``, with the same attention mask.

    Returns:
        (layer output, attention weights of shape (batch, n_heads, L, L))
    """
    if not layer.norm_first:
        raise ValueError("pre_ln_layer_with_attention requires a norm_first=True layer")
    h = layer.norm1(x)
    attn_out, attn_weights = layer.self_attn(
        h, h, h, attn_mask=attn_mask, need_weights=True, average_attn_weights=False
    )
    x = x + layer.dropout1(attn_out)
    h = layer.norm2(x)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
    x = x + layer.dropout2(ff)
    return x, attn_weights.detach()


__all__ = ["WindowInstanceNorm", "pre_ln_layer_with_attention"]
