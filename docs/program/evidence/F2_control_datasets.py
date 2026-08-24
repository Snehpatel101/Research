"""Control-pair validation: the falsifiability instrument for this program.

The mission requires proving that ensembles add real value rather than merely
running. That is impossible without a control. This builds two datasets with
KNOWN ground truth and checks that the repo's own models respond correctly:

  noise_ohlcv  -- a driftless random walk. Labels are unpredictable by
                  construction. NO model may beat the majority baseline.
  signal_ohlcv -- an injected, learnable autoregressive signal. EVERY
                  competent model must beat the majority baseline.

A metric that cannot separate these two datasets is not measuring skill, and
must not be used to justify any claim later in the program.

Run: PYTHONPATH=. .venv/bin/python docs/program/evidence/F2_control_datasets.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

warnings.filterwarnings("ignore")

SEED = 7
N = 6000


def _ohlcv_from_close(close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    spread = np.abs(rng.normal(0, 0.05, len(close)))
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=len(close), freq="1min"),
            "open": close + rng.normal(0, 0.02, len(close)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 1000, len(close)).astype("float64"),
        }
    )


def make_noise(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Driftless random walk. Future returns are independent of the past."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return _ohlcv_from_close(close, rng)


def make_signal(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Close carries a genuine AR(1) momentum signal that a model can learn."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        # strong positive autocorrelation -> next return is partly predictable
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.05)
    close = 100 + np.cumsum(r)
    return _ohlcv_from_close(close, rng)


def featurize(df: pd.DataFrame, horizon: int = 5):
    """Deliberately minimal, strictly causal features + a 3-class forward label."""
    c = df["close"]
    X = pd.DataFrame(
        {
            "ret1": c.pct_change(1),
            "ret5": c.pct_change(5),
            "ret10": c.pct_change(10),
            "vol10": c.pct_change().rolling(10).std(),
            "mom20": c / c.rolling(20).mean() - 1.0,
        }
    )
    fwd = c.shift(-horizon) / c - 1.0
    thr = fwd.std() * 0.5
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd > thr] = 1
    y[fwd < -thr] = -1
    ok = X.notna().all(axis=1) & fwd.notna()
    return X[ok].to_numpy(np.float32), y[ok].to_numpy(), ok.sum()


def evaluate(name: str, df: pd.DataFrame) -> dict:
    X, y, n = featurize(df)
    cut = int(len(X) * 0.7)  # strict temporal split, no shuffling
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    from src.models.registry import ModelRegistry

    model = ModelRegistry.create("xgboost")
    model.fit(Xtr, ytr, Xte, yte)
    pred = model.predict(Xte).class_predictions

    # Baseline: always predict the majority class of the TRAINING set.
    maj = np.bincount(ytr + 1).argmax() - 1
    base = np.full_like(yte, maj)

    # Control: same model, labels shuffled -> destroys any real relationship.
    rng = np.random.default_rng(SEED)
    ysh = rng.permutation(ytr)
    m2 = ModelRegistry.create("xgboost")
    m2.fit(Xtr, ysh, Xte, yte)
    pred_sh = m2.predict(Xte).class_predictions

    out = {
        "dataset": name,
        "n": int(n),
        "acc_model": accuracy_score(yte, pred),
        "acc_base": accuracy_score(yte, base),
        "mcc_model": matthews_corrcoef(yte, pred),
        "mcc_base": matthews_corrcoef(yte, base),
        "f1m_model": f1_score(yte, pred, average="macro"),
        "f1m_base": f1_score(yte, base, average="macro"),
        "acc_shuffled": accuracy_score(yte, pred_sh),
        "mcc_shuffled": matthews_corrcoef(yte, pred_sh),
    }
    return out


def main() -> int:
    import src.models  # noqa: F401  (triggers registration)

    rows = [evaluate("noise", make_noise()), evaluate("signal", make_signal())]

    print(f"\n{'dataset':9s}{'n':>7s}{'acc':>9s}{'base':>9s}{'MCC':>9s}"
          f"{'MCCbase':>9s}{'macroF1':>9s}{'F1base':>9s}{'accShuf':>9s}{'MCCshuf':>9s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['dataset']:9s}{r['n']:>7d}{r['acc_model']:>9.4f}{r['acc_base']:>9.4f}"
              f"{r['mcc_model']:>9.4f}{r['mcc_base']:>9.4f}{r['f1m_model']:>9.4f}"
              f"{r['f1m_base']:>9.4f}{r['acc_shuffled']:>9.4f}{r['mcc_shuffled']:>9.4f}")

    noise, signal = rows[0], rows[1]
    print("\n--- control assertions ---")
    checks = [
        ("noise: model must NOT beat baseline accuracy",
         noise["acc_model"] <= noise["acc_base"] + 0.02),
        ("noise: MCC must be ~0 (no skill)", abs(noise["mcc_model"]) < 0.10),
        ("signal: model MUST beat baseline accuracy",
         signal["acc_model"] > signal["acc_base"] + 0.05),
        ("signal: MCC must show real skill", signal["mcc_model"] > 0.10),
        ("shuffled labels destroy skill on signal",
         abs(signal["mcc_shuffled"]) < 0.10),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed

    print("\n--- metric suitability ---")
    f1_sep = signal["f1m_model"] - signal["f1m_base"]
    f1_noise_gain = noise["f1m_model"] - noise["f1m_base"]
    print(f"  macro-F1 gain on NOISE  = {f1_noise_gain:+.4f}  "
          f"(if > 0, macro-F1 rewards a no-skill model)")
    print(f"  macro-F1 gain on SIGNAL = {f1_sep:+.4f}")
    if f1_noise_gain > 0.01:
        print("  => macro-F1 is NOT a safe headline metric: it credits skill on"
              " pure noise.\n     Use accuracy-vs-majority and MCC for go/no-go"
              " claims.")

    print(f"\nRESULT: {'CONTROLS VALID' if ok else 'CONTROLS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
