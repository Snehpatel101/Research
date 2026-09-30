"""
Evaluation containers: MLFactory-prepared data in the shape the evaluators expect.

``ml cv``, ``ml walk-forward`` and ``ml cpcv-pbo`` evaluate tabular models with
their own fold/window/CPCV loops (per-fold scaling, per-fold fits). They read a
``TimeSeriesDataContainer`` whose train split holds

- the UNSCALED features exactly as the model adapter produced them,
- the triple-barrier label ``label_h{horizon}`` and sample weights,
- ``label_end_h{horizon}``: row position (within the split) at which each label
  resolves, so purging drops every training sample whose label overlaps a test
  block,
- ``datetime`` (bar time), ``close`` and ``symbol`` for strategy returns and costs (CPCV/PBO).

Only the chronological TRAIN split is exposed: the validation and test splits
stay untouched for the final holdout, exactly as in ``MLFactory.run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.core.container import TimeSeriesDataContainer
from src.core.label_spans import label_end_column, remap_label_ends

if TYPE_CHECKING:
    from src.data.adapters import PreparedData


def build_evaluation_container(
    prepared: PreparedData,
    close: pd.Series,
    symbol: str,
    horizon: int,
    n_classes: int = 3,
) -> TimeSeriesDataContainer:
    """
    Wrap the train split of tabular ``PreparedData`` in a container.

    Args:
        prepared: Unscaled, valid-label tabular data (``data_rank == 2``) from
            ``DataPreparer.prepare(..., apply_scaling=False)``, filtered with
            ``filter_invalid_labels()``.
        close: Close price of the source frame, positionally aligned with the
            rows ``prepared.train_indices`` refer to.
        symbol: Trading symbol (looks up costs for CPCV/PBO strategy returns).
        horizon: Label horizon (the container's target is ``label_h{horizon}``).
        n_classes: 2 for binary labels, 3 for short/neutral/long.

    Raises:
        ValueError: If the data is not tabular or has no row indices.
    """
    if prepared.data_rank != 2:
        raise ValueError(
            f"Evaluation commands need tabular (2D) data, got rank {prepared.data_rank} "
            f"for model '{prepared.model_name}'"
        )
    rows = prepared.train_indices
    if rows is None:
        raise ValueError("PreparedData has no train_indices; cannot align labels and prices")

    label_col = f"label_h{horizon}"
    train_df = pd.DataFrame(prepared.X_train, columns=prepared.feature_names)
    if isinstance(close.index, pd.DatetimeIndex):
        # Real bar times: the container hands them out as the index of its frames, so
        # walk-forward windows, OOF predictions and stacking datasets carry timestamps
        train_df["datetime"] = close.index[rows]
    train_df["close"] = close.to_numpy()[rows]
    train_df["symbol"] = symbol
    train_df[label_col] = prepared.y_train
    train_df[f"sample_weight_h{horizon}"] = (
        prepared.train_weights
        if prepared.train_weights is not None
        else np.ones(len(prepared.y_train))
    )
    if prepared.label_end_positions is not None:
        # Source-frame positions -> positions within this split's rows
        train_df[label_end_column(label_col)] = remap_label_ends(
            prepared.label_end_positions[rows], rows
        )

    return TimeSeriesDataContainer.from_dataframes(
        train_df=train_df,
        horizon=horizon,
        feature_columns=list(prepared.feature_names),
        n_classes=n_classes,
    )
