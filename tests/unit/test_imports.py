"""Import smoke test: a broken dependency or circular import fails here first."""

from __future__ import annotations

import importlib

import pytest

CRITICAL_SYMBOLS = [
    ("src.factory", "MLFactory"),
    ("src.config.experiment", "ExperimentConfig"),
    ("src.inference.backtesting.backtest", "BacktestConfig"),
    ("src.validation.cv.purged_kfold", "PurgedKFold"),
    ("src.optimization.feature_selection.result", "FeatureSelectionResult"),
    ("src.validation.cv.oof_core", "OOFPrediction"),
    ("src.validation.cv.oof_stacking", "StackingDatasetBuilder"),
    ("src.models.ensemble.stacking", "StackingEnsemble"),
    ("src.models.ensemble.calibrated_meta", "CalibratedMetaLearner"),
    ("src.data.pipeline.stages.regime.composite", "CompositeRegimeDetector"),
    ("src.data.pipeline.stages.regime.unified", "get_regime_labels"),
    ("src.core.datasets.sequences", "SequenceDataset"),
]


@pytest.mark.parametrize(("module", "name"), CRITICAL_SYMBOLS)
def test_critical_symbol_is_importable(module: str, name: str) -> None:
    assert hasattr(importlib.import_module(module), name)
