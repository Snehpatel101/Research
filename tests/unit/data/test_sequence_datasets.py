"""
Sequence datasets: correct windows/targets and zero-copy tensors.

Both datasets hand training loops tensors that VIEW the dataset's own float32
storage (no per-sample copy): writing through a returned tensor is visible in
the dataset. That is what keeps large 3D/4D training sets from being duplicated
per batch, so a reintroduced ``.copy()`` / ``torch.tensor(...)`` must fail here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.datasets.sequences import SequenceDataset
from src.data.adapters.multi_resolution import MultiResolution4DDataset

N, SEQ_LEN = 40, 5


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=N),
            "f2": rng.normal(size=N),
            "g1": rng.normal(size=N),
            "label": rng.integers(-1, 2, size=N),
            "w": rng.uniform(0.5, 1.5, size=N),
        }
    )


class TestSequenceDataset:
    @pytest.fixture
    def ds(self, frame: pd.DataFrame) -> SequenceDataset:
        return SequenceDataset(frame, ["f1", "f2"], "label", seq_len=SEQ_LEN, weight_column="w")

    def test_window_target_and_weight_come_from_the_last_bar(self, ds, frame) -> None:
        x, y, w = ds[3]
        assert x.shape == (SEQ_LEN, 2)
        np.testing.assert_allclose(x.numpy(), frame[["f1", "f2"]].to_numpy()[3 : 3 + SEQ_LEN])
        assert y.item() == frame["label"].iloc[3 + SEQ_LEN - 1]
        assert w.item() == pytest.approx(frame["w"].iloc[3 + SEQ_LEN - 1])

    def test_out_of_range_index_raises(self, ds) -> None:
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_item_tensor_is_a_view_of_dataset_storage(self, ds) -> None:
        x, _, _ = ds[2]
        assert np.shares_memory(x.numpy(), ds._features)
        x[0, 0] = 123.0
        assert ds._features[2, 0] == 123.0


class TestMultiResolution4DDataset:
    @pytest.fixture
    def ds(self, frame: pd.DataFrame) -> MultiResolution4DDataset:
        return MultiResolution4DDataset(
            frame,
            feature_map={"5min": ["f1", "f2"], "15min": ["g1"]},
            timeframes=["5min", "15min"],
            seq_len=SEQ_LEN,
            label_column="label",
            symbol_column=None,
        )

    def test_shape_padding_and_target(self, ds, frame) -> None:
        x, y, _ = ds[4]
        assert x.shape == (2, SEQ_LEN, 2)
        np.testing.assert_allclose(
            x[0].numpy(), frame[["f1", "f2"]].to_numpy(dtype=np.float32)[4 : 4 + SEQ_LEN]
        )
        # the 15min stream has 1 feature; the second column is padding
        np.testing.assert_allclose(x[1, :, 0].numpy(), frame["g1"].to_numpy(dtype=np.float32)[4:9])
        assert (x[1, :, 1] == 0).all()
        assert y.item() == frame["label"].iloc[4 + SEQ_LEN - 1]

    def test_item_tensor_is_a_view_of_dataset_storage(self, ds) -> None:
        x, _, _ = ds[1]
        assert np.shares_memory(x.numpy(), ds._timeframe_data)
        x[0, 0, 0] = 77.0
        assert ds._timeframe_data[0, 1, 0] == 77.0
