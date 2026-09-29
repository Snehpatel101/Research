"""Phase 116 regression tests: neural model correctness and reference fidelity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import ModelRegistry
from src.models.neural.base_rnn import WarmupCosineSchedule
from src.models.neural.tft_model import InterpretableMultiHeadAttention

N, L, F = 160, 60, 6
FAST = {"device": "cpu", "max_epochs": 2, "batch_size": 32, "warmup_epochs": 0}
SMALL = {
    "lstm": {"hidden_size": 16},
    "tcn": {"num_channels": [8, 8, 8, 8]},
    "transformer": {"d_model": 16, "d_ff": 32, "n_heads": 2, "n_layers": 1},
    "nbeats": {"hidden_size": 32, "n_blocks_per_stack": 2},
    "tft": {"d_model": 8, "d_ff": 16, "n_heads": 2, "lstm_layers": 1},
    "patchtst": {"d_model": 16, "d_ff": 32, "n_heads": 2, "n_layers": 1},
    "itransformer": {"d_model": 16, "d_ff": 32, "n_heads": 2, "n_layers": 1},
}
FOUR_D = {"patchtst", "itransformer"}


def _data(name: str, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, L, F)).astype(np.float32)
    signal = X[:, -5:, 0].mean(axis=1)
    if n_classes == 3:
        y = np.where(signal > 0.3, 1, np.where(signal < -0.3, -1, 0))
    else:
        y = (signal > 0).astype(int)
    if name in FOUR_D:
        X = np.ascontiguousarray(X.reshape(N, L, 2, F // 2).transpose(0, 2, 1, 3))
    return X, y


def _fit(name: str, n_classes: int = 3, **extra: object):
    X, y = _data(name, n_classes)
    model = ModelRegistry.create(
        name, config={**FAST, **SMALL.get(name, {}), "n_classes": n_classes, **extra}
    )
    metrics = model.fit(X[:120], y[:120], X[120:], y[120:])
    return model, metrics, X, y


@pytest.mark.parametrize("name", sorted(SMALL))
def test_train_metrics_aligned_and_save_load_exact(name: str, tmp_path: Path) -> None:
    model, metrics, X, y = _fit(name)
    predicted = model.predict(X[:120]).class_predictions
    assert metrics.train_accuracy == pytest.approx(float(np.mean(predicted == y[:120])))

    probs = model.predict(X[120:]).class_probabilities
    model.save(tmp_path)
    loaded = ModelRegistry.create(name, config={"device": "cpu"})
    loaded.load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(X[120:]).class_probabilities, probs)


@pytest.mark.parametrize("name", ["lstm", "patchtst"])
def test_binary_mode(name: str) -> None:
    model, _, X, _ = _fit(name, n_classes=2)
    assert model.predict(X[:5]).class_probabilities.shape == (5, 2)


@pytest.mark.parametrize("channel_independent", [False, True])
def test_patchtst_sees_newest_bars(channel_independent: bool) -> None:
    model, _, X, _ = _fit("patchtst", channel_independent=channel_independent)
    perturbed = X[:20].copy()
    perturbed[:, :, 56:, :] += 3.0  # bars the unpadded unfold never reached
    delta = model.predict(perturbed).class_probabilities - model.predict(X[:20]).class_probabilities
    assert np.abs(delta).max() > 1e-4
    assert model._unwrapped_model().get_n_patches(60) == 7


def test_transformer_is_causal() -> None:
    model, _, X, _ = _fit("transformer")
    attn = model.get_attention_weights(X[:1])
    assert attn is not None
    assert np.all(attn[:, :, np.triu(np.ones((L, L), dtype=bool), k=1)] == 0)
    assert model.is_production_safe


def test_tcn_last_step_head() -> None:
    model, _, X, _ = _fit("tcn")
    assert model._unwrapped_model().receptive_field == 121  # kernel 5, 4 levels
    # Last-step head: outputs depend on the final bar
    perturbed = X[:10].copy()
    perturbed[:, -1, :] += 3.0
    delta = model.predict(perturbed).class_probabilities - model.predict(X[:10]).class_probabilities
    assert np.abs(delta).max() > 1e-5


def test_nbeats_basis_per_block() -> None:
    model, _, _, _ = _fit("nbeats")
    for stack in model._unwrapped_model().stacks:
        assert len({id(block.basis_function) for block in stack.blocks}) == len(stack.blocks)


def test_tft_mask_semantics_match_on_both_paths() -> None:
    torch.manual_seed(0)
    attn = InterpretableMultiHeadAttention(8, 2, dropout=0.0)
    q = torch.randn(2, 5, 8)
    allowed = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    additive = torch.zeros(5, 5).masked_fill(~allowed, float("-inf"))
    outputs = []
    for training in (True, False):
        attn.train(training)
        outputs += [attn(q, q, q, mask=allowed), attn(q, q, q, mask=additive)]
    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], atol=1e-5, rtol=1e-5)


def test_changed_architecture_refuses_old_checkpoint(tmp_path: Path) -> None:
    model, _, _, _ = _fit("tcn")
    model.save(tmp_path)
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=False)
    checkpoint["arch_version"] = "1.0"
    torch.save(checkpoint, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="architecture version"):
        ModelRegistry.create("tcn", config={"device": "cpu"}).load(tmp_path)


def test_accessors_see_through_compile_wrapper() -> None:
    model, _, _, _ = _fit("itransformer")

    class Wrapped(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    eager = model._model
    model._model = Wrapped(eager)
    try:
        assert model.get_feature_importance() is not None
    finally:
        model._model = eager


def test_schedule_is_batch_size_invariant() -> None:
    a = WarmupCosineSchedule(max_epochs=10, warmup_epochs=2, steps_per_epoch=5)
    b = WarmupCosineSchedule(max_epochs=10, warmup_epochs=2, steps_per_epoch=5)
    a.end_epoch(completed_epochs=3, step=15)
    b.end_epoch(completed_epochs=3, step=15)
    b.restart_epoch(steps_per_epoch=10, step=17)  # OOM two steps into epoch 4
    assert b(17) == pytest.approx(a(15))
    c = WarmupCosineSchedule(max_epochs=10, warmup_epochs=2, steps_per_epoch=10)
    c.end_epoch(completed_epochs=3, step=30)
    assert b(22) == pytest.approx(c(35))  # halfway through the retried epoch


def test_label_smoothing_default_off() -> None:
    model = ModelRegistry.create("lstm", config={"device": "cpu", "label_smoothing": 0.1})
    y = np.array([-1, 0, 1, 0])
    assert model._create_criterion(y, model.config).label_smoothing == pytest.approx(0.1)
    assert ModelRegistry.create("lstm", config={"device": "cpu"}).config["label_smoothing"] == 0.0
