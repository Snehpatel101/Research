"""
Tests for ensemble validator input-rank compatibility (2D vs 3D vs 4D).
"""

from __future__ import annotations

import pytest

from src.core.exceptions import EnsembleCompatibilityError
from src.models.ensemble.validator import get_compatible_models, validate_base_model_compatibility
from src.models.ensemble.voting import VotingEnsemble


def test_validator_rejects_mixing_3d_and_4d_in_strict_mode() -> None:
    # Voting/Blending use strict validation (no mixed ranks)
    with pytest.raises(EnsembleCompatibilityError):
        validate_base_model_compatibility(["lstm", "patchtst"])


def test_validator_allows_homogeneous_4d() -> None:
    # Both models require 4D multi-timeframe input
    validate_base_model_compatibility(["patchtst", "itransformer"])


def test_validator_allows_stacking_2d_plus_3d() -> None:
    validate_base_model_compatibility(["xgboost", "lstm"], ensemble_type="stacking")


def test_validator_rejects_stacking_with_4d_mixed_in() -> None:
    with pytest.raises(EnsembleCompatibilityError):
        validate_base_model_compatibility(["xgboost", "patchtst"], ensemble_type="stacking")


def test_get_compatible_models_uses_input_rank() -> None:
    compatible = get_compatible_models("patchtst")
    assert "itransformer" in compatible
    assert "lstm" not in compatible


def test_voting_ensemble_requires_4d_when_all_base_models_are_4d() -> None:
    ensemble = VotingEnsemble(config={"base_model_names": ["patchtst", "itransformer"]})
    assert ensemble.requires_4d is True


def test_voting_ensemble_not_requires_4d_for_3d_sequence_models() -> None:
    ensemble = VotingEnsemble(config={"base_model_names": ["lstm", "gru"]})
    assert ensemble.requires_4d is False

