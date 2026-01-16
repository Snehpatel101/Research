import numpy as np
import pandas as pd
import pytest

from src.cross_validation.oof_core import OOFPrediction
from src.cross_validation.timestamp_alignment import (
    align_predictions_on_datetime,
    get_datetime_alignment_report,
    validate_datetime_alignment,
)


@pytest.fixture
def aligned_oof_predictions():
    datetimes = pd.date_range("2024-01-01", periods=100, freq="5min")

    oof_pred_1 = OOFPrediction(
        model_name="model1",
        predictions=pd.DataFrame(
            {
                "datetime": datetimes,
                "model1_prob_short": np.random.rand(100),
                "model1_prob_neutral": np.random.rand(100),
                "model1_prob_long": np.random.rand(100),
                "model1_pred": np.random.choice([-1, 0, 1], 100),
                "model1_confidence": np.random.rand(100),
            }
        ),
        fold_info=[],
        coverage=1.0,
    )

    oof_pred_2 = OOFPrediction(
        model_name="model2",
        predictions=pd.DataFrame(
            {
                "datetime": datetimes,
                "model2_prob_short": np.random.rand(100),
                "model2_prob_neutral": np.random.rand(100),
                "model2_prob_long": np.random.rand(100),
                "model2_pred": np.random.choice([-1, 0, 1], 100),
                "model2_confidence": np.random.rand(100),
            }
        ),
        fold_info=[],
        coverage=1.0,
    )

    return {"model1": oof_pred_1, "model2": oof_pred_2}


@pytest.fixture
def misaligned_oof_predictions():
    datetimes_1 = pd.date_range("2024-01-01", periods=100, freq="5min")
    datetimes_2 = pd.date_range("2024-01-01 00:30", periods=90, freq="5min")

    oof_pred_1 = OOFPrediction(
        model_name="model1",
        predictions=pd.DataFrame(
            {
                "datetime": datetimes_1,
                "model1_prob_short": np.random.rand(100),
                "model1_prob_neutral": np.random.rand(100),
                "model1_prob_long": np.random.rand(100),
                "model1_pred": np.random.choice([-1, 0, 1], 100),
                "model1_confidence": np.random.rand(100),
            }
        ),
        fold_info=[],
        coverage=1.0,
    )

    oof_pred_2 = OOFPrediction(
        model_name="model2",
        predictions=pd.DataFrame(
            {
                "datetime": datetimes_2,
                "model2_prob_short": np.random.rand(90),
                "model2_prob_neutral": np.random.rand(90),
                "model2_prob_long": np.random.rand(90),
                "model2_pred": np.random.choice([-1, 0, 1], 90),
                "model2_confidence": np.random.rand(90),
            }
        ),
        fold_info=[],
        coverage=0.9,
    )

    return {"model1": oof_pred_1, "model2": oof_pred_2}


def test_validate_datetime_alignment_perfect(aligned_oof_predictions):
    is_valid, issues = validate_datetime_alignment(aligned_oof_predictions, strict=True)

    assert is_valid
    assert len(issues) == 0


def test_validate_datetime_alignment_misaligned(misaligned_oof_predictions):
    is_valid, issues = validate_datetime_alignment(misaligned_oof_predictions, strict=True)

    assert not is_valid
    assert len(issues) > 0
    assert any("Length mismatch" in issue for issue in issues)


def test_validate_datetime_alignment_misaligned_non_strict(misaligned_oof_predictions):
    is_valid, issues = validate_datetime_alignment(misaligned_oof_predictions, strict=False)

    assert len(issues) > 0


def test_validate_datetime_alignment_single_model():
    datetimes = pd.date_range("2024-01-01", periods=100, freq="5min")
    oof_pred = OOFPrediction(
        model_name="model1",
        predictions=pd.DataFrame(
            {
                "datetime": datetimes,
                "model1_prob_short": np.random.rand(100),
                "model1_prob_neutral": np.random.rand(100),
                "model1_prob_long": np.random.rand(100),
                "model1_pred": np.random.choice([-1, 0, 1], 100),
                "model1_confidence": np.random.rand(100),
            }
        ),
        fold_info=[],
        coverage=1.0,
    )

    is_valid, issues = validate_datetime_alignment({"model1": oof_pred}, strict=True)
    assert is_valid
    assert len(issues) == 0


def test_align_predictions_inner(misaligned_oof_predictions):
    aligned = align_predictions_on_datetime(misaligned_oof_predictions, method="inner")

    assert "model1" in aligned
    assert "model2" in aligned

    len_1 = len(aligned["model1"].predictions)
    len_2 = len(aligned["model2"].predictions)

    assert len_1 == len_2
    assert len_1 < 100


def test_align_predictions_outer(misaligned_oof_predictions):
    aligned = align_predictions_on_datetime(misaligned_oof_predictions, method="outer")

    assert "model1" in aligned
    assert "model2" in aligned

    len_1 = len(aligned["model1"].predictions)
    len_2 = len(aligned["model2"].predictions)

    assert len_1 == len_2

    has_nans_1 = aligned["model1"].predictions["model1_pred"].isna().any()
    has_nans_2 = aligned["model2"].predictions["model2_pred"].isna().any()

    assert has_nans_1 or has_nans_2


def test_align_predictions_invalid_method(aligned_oof_predictions):
    with pytest.raises(ValueError, match="Unknown alignment method"):
        align_predictions_on_datetime(aligned_oof_predictions, method="invalid")


def test_get_datetime_alignment_report(aligned_oof_predictions):
    report = get_datetime_alignment_report(aligned_oof_predictions)

    assert "n_models" in report
    assert report["n_models"] == 2
    assert "model_names" in report
    assert set(report["model_names"]) == {"model1", "model2"}
    assert "common_timestamps" in report
    assert "common_coverage" in report
    assert report["common_coverage"] == 1.0


def test_get_datetime_alignment_report_misaligned(misaligned_oof_predictions):
    report = get_datetime_alignment_report(misaligned_oof_predictions)

    assert report["n_models"] == 2
    assert report["common_coverage"] < 1.0
    assert "pairwise_overlaps" in report
    assert "model1_vs_model2" in report["pairwise_overlaps"]
