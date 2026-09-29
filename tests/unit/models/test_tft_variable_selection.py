"""TFT variable selection: the batched per-variable GRNs equal independent GRNs.

``VariableGRNs`` stacks the weights of one GRN per variable and computes all of
them with batched matmuls (and recomputes activations in backward while
training). These tests rebuild the architecture-1.0 formulation — a shared
``Linear(1, d_model)`` embedding followed by one ``GatedResidualNetwork`` module
per variable — copy its weights into the batched module and check that
outputs and gradients match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models import ModelRegistry
from src.models.neural.tft_model import GatedResidualNetwork, VariableGRNs

D = 8
N_FEATURES = 37  # not a multiple of 16: exercises a ragged last chunk


def _reference(n_features: int, d: int) -> tuple[nn.Linear, nn.ModuleList]:
    torch.manual_seed(1)
    embedding = nn.Linear(1, d)
    grns = nn.ModuleList([GatedResidualNetwork(d, d, d, dropout=0.0) for _ in range(n_features)])
    for grn in grns:
        for p in grn.parameters():  # non-trivial LayerNorm affine too
            nn.init.normal_(p, std=0.3)
    return embedding, grns


def _batched_from(embedding: nn.Linear, grns: nn.ModuleList) -> VariableGRNs:
    batched = VariableGRNs(len(grns), D, dropout=0.0)
    with torch.no_grad():
        batched.embedding.load_state_dict(embedding.state_dict())
        for i, grn in enumerate(grns):
            batched.fc1_weight[i] = grn.fc1.weight.T
            batched.fc1_bias[i] = grn.fc1.bias
            batched.fc2_weight[i] = grn.fc2.weight.T
            batched.fc2_bias[i] = grn.fc2.bias
            batched.glu_weight[i] = grn.glu.linear.weight.T
            batched.glu_bias[i] = grn.glu.linear.bias
            batched.norm_weight[i] = grn.layer_norm.weight
            batched.norm_bias[i] = grn.layer_norm.bias
    return batched


def _reference_forward(embedding: nn.Linear, grns: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    embedded = embedding(x.unsqueeze(-1))  # (rows, n_features, d)
    return torch.stack([grn(embedded[:, i]) for i, grn in enumerate(grns)], dim=1)


def _set_chunk(monkeypatch: pytest.MonkeyPatch, rows: int, variables: int) -> None:
    monkeypatch.setattr(VariableGRNs, "CHUNK_BYTES", rows * D * 4 * variables)


@pytest.mark.parametrize("variables_per_chunk", [1, 16, N_FEATURES])
@pytest.mark.parametrize("training", [False, True])
def test_batched_grns_equal_independent_grns(
    training: bool, variables_per_chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding, grns = _reference(N_FEATURES, D)
    batched = _batched_from(embedding, grns)
    batched.train(training)  # training recomputes chunks in backward
    x = torch.randn(50, N_FEATURES)
    _set_chunk(monkeypatch, 50, variables_per_chunk)
    assert batched.chunk_size(50, 4) == variables_per_chunk

    expected = _reference_forward(embedding, grns, x)
    got = batched(x)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    upstream = torch.randn_like(expected)
    ref_grads = torch.autograd.grad(
        (expected * upstream).sum(), [grns[3].fc1.weight, grns[-1].glu.linear.bias]
    )
    new_grads = torch.autograd.grad((got * upstream).sum(), [batched.fc1_weight, batched.glu_bias])
    torch.testing.assert_close(new_grads[0][3], ref_grads[0].T, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(new_grads[1][-1], ref_grads[1], atol=1e-5, rtol=1e-4)


def test_recomputation_reuses_the_dropout_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_chunk(monkeypatch, 40, 16)  # 37 variables: chunks of 16, 16, 5
    batched = VariableGRNs(N_FEATURES, D, dropout=0.3).train()
    x = torch.randn(40, N_FEATURES)
    upstream = torch.randn(40, N_FEATURES, D)

    torch.manual_seed(5)
    recomputed = batched(x)
    grad_recomputed = torch.autograd.grad((recomputed * upstream).sum(), batched.fc2_weight)[0]

    torch.manual_seed(5)
    xt = x.t().unsqueeze(-1)
    step = 16
    stored = torch.cat(
        [
            batched._chunk(xt[s : s + step], s, min(s + step, N_FEATURES))
            for s in range(0, N_FEATURES, step)
        ],
        dim=1,
    )
    grad_stored = torch.autograd.grad((stored * upstream).sum(), batched.fc2_weight)[0]

    torch.testing.assert_close(recomputed, stored)
    torch.testing.assert_close(grad_recomputed, grad_stored)


def test_per_variable_init_matches_independent_linears() -> None:
    torch.manual_seed(0)
    batched = VariableGRNs(200, 64)
    bound = np.sqrt(6.0 / (64 + 64))
    assert batched.fc1_weight.abs().max() <= bound
    assert batched.fc1_weight.std().item() == pytest.approx(bound / np.sqrt(3), rel=0.02)
    assert torch.equal(batched.norm_weight, torch.ones(200, 64))


def test_tft_refuses_architecture_1_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 20, 5)).astype(np.float32)
    y = rng.integers(-1, 2, size=64)
    config = {
        "device": "cpu",
        "max_epochs": 1,
        "batch_size": 32,
        "warmup_epochs": 0,
        "d_model": 8,
        "d_ff": 16,
        "n_heads": 2,
        "lstm_layers": 1,
    }
    model = ModelRegistry.create("tft", config=config)
    model.fit(X[:48], y[:48], X[48:], y[48:])
    model.save(tmp_path)
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    assert checkpoint["arch_version"] == "2.0"
    checkpoint["arch_version"] = "1.0"
    torch.save(checkpoint, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="architecture version"):
        ModelRegistry.create("tft", config={"device": "cpu"}).load(tmp_path)
