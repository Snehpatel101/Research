"""Config must not promise behaviour it does not have (Stage 7).

Two findings drove these tests.

1. Seven of thirteen `LabelingConfig` fields have ZERO consumers anywhere in
   `src/`. Not "ignored downstream" — nothing reads them. Worse, they read as
   meaningful, and `optimize_barriers` DEFAULTED TO TRUE, telling every reader
   that an expensive optimisation step was running.

2. The config layer has a full `validate()` chain, but all 20 call sites were
   `super().validate()` inside other config classes. Nothing outside the
   config package ever invoked it, so every bounds check was dead code.

Fixing (1) inside a validator that nobody calls would have repeated the exact
bug, so both halves are tested here: the fields declare their own inertness,
and validation actually runs.
"""

from __future__ import annotations

import logging
import tempfile

import pytest

from src.config.data import LabelingConfig


class TestUnwiredFieldsAreDeclared:
    """The dead fields must be enumerated, and the list must stay honest."""

    def test_unwired_map_matches_reality(self):
        """Every field in UNWIRED_FIELDS must really have no consumers.

        Greps `src/` for the field name. If someone wires one up without
        removing it from the map, this fails and tells them to.
        """
        import pathlib
        import re

        src = pathlib.Path("src")
        for field_name in LabelingConfig.UNWIRED_FIELDS:
            pattern = re.compile(rf"labeling\.{re.escape(field_name)}\b")
            hits = [
                f"{p}:{i}"
                for p in src.rglob("*.py")
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1)
                if pattern.search(line)
            ]
            assert not hits, (
                f"LabelingConfig.{field_name} is listed as UNWIRED but is read "
                f"at {hits[:3]}. If you wired it, remove it from "
                f"UNWIRED_FIELDS."
            )

    def test_defaults_are_the_inert_values(self):
        """A default must never advertise behaviour that does not exist."""
        cfg = LabelingConfig()
        for field_name, inert in LabelingConfig.UNWIRED_FIELDS.items():
            assert getattr(cfg, field_name) == inert, (
                f"LabelingConfig.{field_name} defaults to "
                f"{getattr(cfg, field_name)!r} but does nothing. A default "
                f"that implies a working feature is a lie in the API."
            )

    def test_optimize_barriers_no_longer_defaults_true(self):
        """Regression pin for the worst offender."""
        assert LabelingConfig().optimize_barriers is False, (
            "optimize_barriers defaulting True claimed that barrier "
            "optimisation runs. It does not."
        )


class TestValidationDetectsInertSettings:
    def test_setting_an_unwired_field_produces_an_issue(self):
        cfg = LabelingConfig(optimization_trials=500)
        issues = cfg.validate()
        assert any(
            "optimization_trials" in i and "NO EFFECT" in i for i in issues
        ), f"expected a NO EFFECT issue for optimization_trials, got {issues}"

    def test_default_config_is_clean(self):
        assert LabelingConfig().validate() == []

    def test_real_bounds_checks_still_work(self):
        """The pre-existing validation must not be shadowed by the new checks."""
        assert any("atr_period" in i for i in LabelingConfig(atr_period=0).validate())


class TestValidationActuallyRuns:
    """The systemic half: validate() was never called from outside config."""

    def test_factory_logs_config_issues_on_construction(self, caplog):
        from src.config.experiment import ExperimentConfig
        from src.factory import MLFactory

        with tempfile.TemporaryDirectory() as d:
            cfg = ExperimentConfig(name="t", output_dir=d)
            cfg.data.labeling.optimization_trials = 500
            with caplog.at_level(logging.WARNING):
                MLFactory(cfg, verbose=0)

        # getMessage() applies the %-args, so no manual formatting is needed.
        assert any("optimization_trials" in r.getMessage() for r in caplog.records), (
            "MLFactory did not surface the config issue. If validation is not "
            "invoked at construction, every bounds check is dead code again."
        )

    def test_default_config_construction_is_quiet(self, caplog):
        from src.config.experiment import ExperimentConfig
        from src.factory import MLFactory

        with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.WARNING):
            MLFactory(ExperimentConfig(name="t", output_dir=d), verbose=0)

        noisy = [r.getMessage() for r in caplog.records if "config issue" in r.getMessage()]
        assert not noisy, f"default config should validate cleanly, got: {noisy}"


class TestBarrierParamsRemainLive:
    """Guard the fields that DO work, so this stage cannot regress them."""

    @pytest.mark.parametrize(
        "field_name,value", [("upper_mult", 2.5), ("lower_mult", 1.5), ("atr_period", 21)]
    )
    def test_live_fields_are_not_flagged_inert(self, field_name, value):
        issues = LabelingConfig(**{field_name: value}).validate()
        assert not any(
            "NO EFFECT" in i for i in issues
        ), f"{field_name} is wired and must not be reported inert: {issues}"
