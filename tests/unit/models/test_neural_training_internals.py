"""
Neural training internals (BaseRNNModel): loss weighting, loader wiring.

- Class weights survive when per-sample weights are active (both are applied).
- Validation batches are twice the training batch size.
- Multi-worker DataLoaders get a per-worker seed function.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.models.neural.lstm_model import LSTMModel

N_FEATURES, N_CLASSES = 4, 3


@pytest.fixture
def model() -> LSTMModel:
    return LSTMModel({"device": "cpu"})


def _one_epoch(model, loader, criterion) -> tuple[float, torch.Tensor]:
    """Run _train_epoch with a frozen (lr=0) linear head; return (loss, logits of the batch)."""
    torch.manual_seed(0)
    net = nn.Linear(N_FEATURES, N_CLASSES)
    model._model = net
    model._use_amp = False
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    X, _ = next(iter(loader))[:2]
    with torch.no_grad():
        logits = net(X)
    loss, _acc, _grad = model._train_epoch(
        loader, optimizer, criterion, scheduler, None, torch.float32, grad_clip=1.0
    )
    return loss, logits


class TestLossWeighting:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(12, N_FEATURES)).astype(np.float32)
        y = np.array([-1, 0, 1, 1, 0, -1, 1, 1, 0, 0, -1, 1])
        w = rng.uniform(0.2, 3.0, size=12).astype(np.float32)
        return X, y, w

    def test_class_and_sample_weights_are_both_applied(self, model, data) -> None:
        X, y, w = data
        class_weights = torch.tensor([1.0, 2.0, 0.5])
        loader = model._create_dataloader(X, y, w, {"batch_size": 64}, shuffle=False)

        loss, logits = _one_epoch(model, loader, nn.CrossEntropyLoss(weight=class_weights))

        targets = torch.from_numpy(model._convert_labels_to_class(y).astype(np.int64))
        per_sample = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
        expected = (per_sample * torch.from_numpy(w)).mean().item()
        without_class_weights = (
            F.cross_entropy(logits, targets, reduction="none") * torch.from_numpy(w)
        ).mean()

        assert loss == pytest.approx(expected, rel=1e-5)
        assert abs(loss - without_class_weights.item()) > 1e-3, "class weights were dropped"

    def test_label_smoothing_is_preserved_with_sample_weights(self, model, data) -> None:
        X, y, w = data
        loader = model._create_dataloader(X, y, w, {"batch_size": 64}, shuffle=False)

        loss, logits = _one_epoch(model, loader, nn.CrossEntropyLoss(label_smoothing=0.2))

        targets = torch.from_numpy(model._convert_labels_to_class(y).astype(np.int64))
        per_sample = F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.2)
        assert loss == pytest.approx((per_sample * torch.from_numpy(w)).mean().item(), rel=1e-5)

    def test_unweighted_batches_use_plain_criterion(self, model, data) -> None:
        X, y, _ = data
        loader = model._create_dataloader(X, y, None, {"batch_size": 64}, shuffle=False)

        loss, logits = _one_epoch(model, loader, nn.CrossEntropyLoss())

        targets = torch.from_numpy(model._convert_labels_to_class(y).astype(np.int64))
        assert loss == pytest.approx(F.cross_entropy(logits, targets).item(), rel=1e-5)


class TestLoaders:
    def _arrays(self, n: int = 40):
        rng = np.random.default_rng(2)
        return rng.normal(size=(n, N_FEATURES)).astype(np.float32), rng.integers(-1, 2, size=n)

    def test_validation_loader_batch_is_double_the_training_batch(self, model) -> None:
        X, y = self._arrays()
        train_loader, val_loader = model._create_fit_loaders(X, y, None, X, y, {"batch_size": 8})
        assert train_loader.batch_size == 8
        assert val_loader.batch_size == 16

    def test_default_batch_size_also_doubles_for_validation(self, model) -> None:
        X, y = self._arrays()
        train_loader, val_loader = model._create_fit_loaders(X, y, None, X, y, {})
        assert val_loader.batch_size == 2 * train_loader.batch_size

    def test_worker_loaders_get_a_seeding_function(self, model) -> None:
        X, y = self._arrays()
        loader = model._create_dataloader(X, y, None, {"num_workers": 2}, shuffle=False)
        assert loader.worker_init_fn is not None

    def test_single_process_loaders_need_no_worker_function(self, model) -> None:
        X, y = self._arrays()
        loader = model._create_dataloader(X, y, None, {"num_workers": 0}, shuffle=False)
        assert loader.worker_init_fn is None

    def test_loader_tensors_share_memory_with_the_input_array(self, model) -> None:
        """The training set is wrapped, not duplicated (torch.from_numpy, not torch.tensor)."""
        X, y = self._arrays()
        loader = model._create_dataloader(X, y, None, {"batch_size": 8}, shuffle=False)
        assert np.shares_memory(loader.dataset.tensors[0].numpy(), X)
