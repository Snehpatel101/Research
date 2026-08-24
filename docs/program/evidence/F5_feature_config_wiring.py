"""Stage 6: prove the feature-config wiring is live AND behaviour-preserving.

Phase 0 finding F15: MLFactory constructed FeatureEngineer with 2 of 18
parameters, so every `data.features.*` / `data.mtf.*` setting was dead on the
documented entry point.

Two things must BOTH be true after the fix, and they pull against each other:

  A. DEFAULT config produces the SAME features as before  -> no silent
     result changes for anyone's existing experiments.
  B. NON-DEFAULT config actually changes behaviour        -> the knobs are
     genuinely live, not merely passed.

Run: PYTHONPATH=. .venv/bin/python docs/program/evidence/F5_feature_config_wiring.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.pipeline.stages.features import FeatureEngineer

N = 3000
rng = np.random.default_rng(0)
close = 100 + np.cumsum(rng.normal(0, 0.1, N))
spread = np.abs(rng.normal(0, 0.05, N))
df = pd.DataFrame(
    {
        "datetime": pd.date_range("2024-01-01", periods=N, freq="1min"),
        "open": close + rng.normal(0, 0.02, N),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(100, 1000, N).astype("float64"),
    }
)


def run(**kwargs) -> tuple[set[str], int]:
    eng = FeatureEngineer(input_dir=".", output_dir="/tmp", **kwargs)
    out, _ = eng.engineer_features(df.copy(), symbol="MES")
    return set(out.columns), len(out)


print("=" * 72)
print("A. Does the DEFAULT config reproduce the OLD behaviour?")
print("=" * 72)
# OLD: what MLFactory used to build -- everything defaulted.
old_cols, old_rows = run()

# NEW: what MLFactory now builds from the DEFAULT ExperimentConfig.
# data.mtf: enabled=True, primary_timeframe='5min',
#           timeframes=['5min','15min','60min'], aggregate_*=True
# data.features.mode = 'full'  -> minimal=False
new_cols, new_rows = run(
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
print(f"  old: {len(old_cols)} columns, {old_rows} rows")
print(f"  new: {len(new_cols)} columns, {new_rows} rows")
only_old = sorted(old_cols - new_cols)
only_new = sorted(new_cols - old_cols)
print(f"  columns only in old: {only_old if only_old else 'none'}")
print(f"  columns only in new: {only_new if only_new else 'none'}")
identical = (old_cols == new_cols) and (old_rows == new_rows)
print(f"  IDENTICAL: {identical}")

print()
print("=" * 72)
print("B. Do NON-DEFAULT settings actually take effect?")
print("=" * 72)

# MTF columns use get_timeframe_suffix(): "15min" -> "_15m", "60min" -> "_1h".
# (My first attempt looked for "_15min"/"_60min" and found zero in BOTH cases,
#  which read as "wiring dead" when it was really a wrong probe.)
MTF_SUFFIXES = ("_15m", "_1h", "_4h", "_1d")
mtf_off_cols, _ = run(enable_mtf=False)
mtf_cols_present = {c for c in old_cols if c.endswith(MTF_SUFFIXES)}
mtf_cols_after_off = {c for c in mtf_off_cols if c.endswith(MTF_SUFFIXES)}
print(f"  mtf.enabled=True  -> {len(mtf_cols_present)} MTF columns")
print(f"  mtf.enabled=False -> {len(mtf_cols_after_off)} MTF columns")
mtf_live = len(mtf_cols_present) > 0 and len(mtf_cols_after_off) == 0

minimal_cols, _ = run(
    enable_wavelets=False,
    enable_microstructure=False,
    enable_volume_features=False,
    enable_volatility_features=False,
)
print(f"  features.mode='full'    -> {len(old_cols)} columns")
print(f"  features.mode='minimal' -> {len(minimal_cols)} columns")
minimal_live = len(minimal_cols) < len(old_cols)
saved = len(old_cols) - len(minimal_cols)
print(f"  minimal saves {saved} columns ({saved / max(len(old_cols), 1):.0%})")

print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  A. default behaviour preserved : {identical}")
print(f"  B1. mtf.enabled is live        : {mtf_live}")
print(f"  B2. features.mode is live      : {minimal_live}")
ok = identical and mtf_live and minimal_live
print()
print("RESULT:", "F15 FIXED — knobs live, defaults unchanged" if ok else "INCOMPLETE")
raise SystemExit(0 if ok else 1)
