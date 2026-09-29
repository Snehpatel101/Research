"""OOF probability column names: named for 3 classes, numbered otherwise (binary mode)."""

from __future__ import annotations

from src.validation.cv.oof_core import _get_prob_column_names


def test_three_class_columns_are_named_short_neutral_long() -> None:
    assert _get_prob_column_names("xgboost", n_classes=3) == [
        "xgboost_prob_short",
        "xgboost_prob_neutral",
        "xgboost_prob_long",
    ]


def test_binary_columns_are_numbered() -> None:
    assert _get_prob_column_names("lstm", n_classes=2) == ["lstm_prob_0", "lstm_prob_1"]


def test_other_class_counts_get_one_numbered_column_per_class() -> None:
    cols = _get_prob_column_names("model", n_classes=5)
    assert cols == [f"model_prob_{i}" for i in range(5)]
