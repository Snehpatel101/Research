"""Regression tests for Phase 0 finding F15 (Stage 6).

`MLFactory._run_data_pipeline` used to construct `FeatureEngineer` with 2 of
its 18 parameters, so every `data.features.*` and `data.mtf.*` setting was
dead on the documented entry point. Agent 6 observed the visible symptom:
a run with `cfg.data.mtf.enabled = False` computed MTF features anyway.

Two properties must hold together, and they pull against each other:

  A. The DEFAULT config must reproduce the OLD feature set exactly, so wiring
     the knobs does not silently change anyone's existing results.
  B. NON-DEFAULT settings must actually change behaviour, or the wiring is
     decorative.

Full proof: docs/program/evidence/F5_feature_config_wiring.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# MTF columns are suffixed via get_timeframe_suffix(): "15min" -> "_15m",
# "60min" -> "_1h". Probing for "_15min"/"_60min" finds nothing and looks
# like dead wiring -- that mistake is why this constant is spelled out.
MTF_SUFFIXES = ("_15m", "_1h", "_4h", "_1d")

N = 3000


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, N))
    spread = np.abs(rng.normal(0, 0.05, N))
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=N, freq="1min"),
            "open": close + rng.normal(0, 0.02, N),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 1000, N).astype("float64"),
        }
    )


def _run(df: pd.DataFrame, tmp_path, **kwargs) -> set[str]:
    from src.data.pipeline.stages.features import FeatureEngineer

    eng = FeatureEngineer(input_dir=str(tmp_path), output_dir=str(tmp_path), **kwargs)
    out, _ = eng.engineer_features(df.copy(), symbol="MES")
    return set(out.columns)


class TestFactoryPassesConfig:
    """The wiring itself — asserted on the real construction site."""

    def test_factory_forwards_mtf_and_feature_settings(self):
        """Guards against a regression to the 2-argument construction."""
        import inspect

        from src.factory import MLFactory

        src = inspect.getsource(MLFactory._run_data_pipeline)
        # Deliberately narrow: this asserts the ARGUMENTS are forwarded, which
        # the behavioural tests below then prove actually work. It exists only
        # to fail loudly if someone reverts to FeatureEngineer(input, output).
        for kwarg in ("enable_mtf=", "mtf_timeframes=", "enable_wavelets="):
            assert kwarg in src, (
                f"MLFactory no longer forwards {kwarg!r} to FeatureEngineer — "
                f"Phase 0 finding F15 has regressed."
            )


class TestDefaultsUnchanged:
    """Property A: wiring the knobs must not move anyone's results."""

    def test_default_config_matches_legacy_defaults(self, ohlcv, tmp_path):
        legacy = _run(ohlcv, tmp_path)
        wired = _run(
            ohlcv,
            tmp_path,
            timeframe="5min",
            base_timeframe="5min",
            enable_mtf=True,
            mtf_timeframes=["5min", "15min", "60min"],
            mtf_include_ohlcv=True,
            mtf_include_indicators=True,
            enable_wavelets=True,
            enable_microstructure=True,
            enable_volume_features=True,
            enable_volatility_features=True,
        )
        assert wired == legacy, (
            f"Default ExperimentConfig no longer reproduces the legacy feature "
            f"set. Only in legacy: {sorted(legacy - wired)[:5]}; "
            f"only in wired: {sorted(wired - legacy)[:5]}"
        )


class TestSettingsAreLive:
    """Property B: the knobs must actually do something."""

    def test_mtf_enabled_false_removes_mtf_columns(self, ohlcv, tmp_path):
        on = _run(ohlcv, tmp_path, enable_mtf=True)
        off = _run(ohlcv, tmp_path, enable_mtf=False)

        mtf_on = {c for c in on if c.endswith(MTF_SUFFIXES)}
        mtf_off = {c for c in off if c.endswith(MTF_SUFFIXES)}

        assert mtf_on, "no MTF columns produced with enable_mtf=True — probe is wrong"
        assert not mtf_off, (
            f"enable_mtf=False still produced {len(mtf_off)} MTF columns "
            f"(e.g. {sorted(mtf_off)[:3]}). This is the exact symptom Agent 6 "
            f"observed in the Phase 0 end-to-end run."
        )

    def test_minimal_mode_produces_fewer_features(self, ohlcv, tmp_path):
        full = _run(ohlcv, tmp_path)
        minimal = _run(
            ohlcv,
            tmp_path,
            enable_wavelets=False,
            enable_microstructure=False,
            enable_volume_features=False,
            enable_volatility_features=False,
        )
        assert len(minimal) < len(full), (
            "features.mode='minimal' did not reduce the feature count; the "
            "fast test tier depends on this being real."
        )
        # Wavelets were the specific family Agent 6 saw emitting 100%-NaN
        # columns that were then dropped — pure wasted work.
        assert not {c for c in minimal if c.startswith("wavelet_")}

    def test_mtf_timeframes_selection_is_honoured(self, ohlcv, tmp_path):
        only_15 = _run(ohlcv, tmp_path, enable_mtf=True, mtf_timeframes=["15min"])
        suffixes = {c[-4:] for c in only_15 if c.endswith(MTF_SUFFIXES)}
        assert not any(
            c.endswith("_1h") for c in only_15
        ), f"requested only 15min MTF but got 1h columns; suffixes seen: {suffixes}"
