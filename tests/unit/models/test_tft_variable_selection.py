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
from src.models.neural.tft_model import (
    GatedResidualNetwork,
    VariableGRNs,
    convert_tft_state_dict_v1,
)

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


def test_empty_batch() -> None:
    batched = VariableGRNs(N_FEATURES, D).train()
    out = batched(torch.zeros(0, N_FEATURES))
    assert out.shape == (0, N_FEATURES, D)
    out.sum().backward()


def _v1_fragment(embedding: nn.Linear, grns: nn.ModuleList) -> dict[str, torch.Tensor]:
    """The architecture-1.0 state-dict keys of the embedding and per-feature GRNs."""
    fragment = {f"input_embedding.{k}": v for k, v in embedding.state_dict().items()}
    for i, grn in enumerate(grns):
        fragment |= {f"vsn.feature_grns.{i}.{k}": v for k, v in grn.state_dict().items()}
    return fragment


def test_v1_conversion_reproduces_the_independent_grns() -> None:
    embedding, grns = _reference(N_FEATURES, D)
    converted = convert_tft_state_dict_v1(_v1_fragment(embedding, grns), N_FEATURES)
    batched = VariableGRNs(N_FEATURES, D, dropout=0.0).eval()
    batched.load_state_dict(
        {k.removeprefix("vsn.variable_grns."): v for k, v in converted.items()}, strict=True
    )
    x = torch.randn(30, N_FEATURES)
    with torch.no_grad():
        torch.testing.assert_close(
            batched(x), _reference_forward(embedding, grns, x), atol=1e-5, rtol=1e-5
        )


def _small_tft(tmp_path: Path):  # noqa: ANN202
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
    return model, X, checkpoint


def _as_v1(state_dict: dict[str, torch.Tensor], n_features: int) -> dict[str, torch.Tensor]:
    """Rewrite a 2.0 state dict in the 1.0 layout (one GRN module per feature)."""
    prefix = "vsn.variable_grns."
    names = {
        "fc1_weight": ("fc1.weight", True),
        "fc1_bias": ("fc1.bias", False),
        "fc2_weight": ("fc2.weight", True),
        "fc2_bias": ("fc2.bias", False),
        "glu_weight": ("glu.linear.weight", True),
        "glu_bias": ("glu.linear.bias", False),
        "norm_weight": ("layer_norm.weight", False),
        "norm_bias": ("layer_norm.bias", False),
    }
    v1: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix + "embedding."):
            v1["input_embedding." + key.removeprefix(prefix + "embedding.")] = value
        elif key.startswith(prefix):
            param, transpose = names[key.removeprefix(prefix)]
            for i in range(n_features):
                v1[f"vsn.feature_grns.{i}.{param}"] = value[i].t() if transpose else value[i]
        else:
            v1[key] = value
    return v1


def test_tft_converts_architecture_1_checkpoint(tmp_path: Path) -> None:
    model, X, checkpoint = _small_tft(tmp_path)
    expected = model.predict(X[48:]).class_probabilities
    checkpoint["model_state_dict"] = _as_v1(checkpoint["model_state_dict"], 5)
    checkpoint["arch_version"] = "1.0"
    torch.save(checkpoint, tmp_path / "model.pt")

    loaded = ModelRegistry.create("tft", config={"device": "cpu"})
    loaded.load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(X[48:]).class_probabilities, expected)


def test_tft_refuses_unknown_architecture(tmp_path: Path) -> None:
    _, _, checkpoint = _small_tft(tmp_path)
    checkpoint["arch_version"] = "0.9"
    torch.save(checkpoint, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="architecture version"):
        ModelRegistry.create("tft", config={"device": "cpu"}).load(tmp_path)


def test_bundle_records_the_architecture_version(tmp_path: Path) -> None:
    from src.inference.bundle import ModelBundle

    model, _, _ = _small_tft(tmp_path)
    bundle = ModelBundle.from_training(
        model, None, [f"f{i}" for i in range(5)], horizon=5, model_name="tft"
    )
    assert bundle.metadata.arch_version == "2.0"
    assert bundle.metadata.to_dict()["arch_version"] == "2.0"
