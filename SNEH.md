# SNEH Investigation Notes

## Scope
- Request followed: investigate only, no pipeline/code changes.
- Target notebook analyzed: `notebooks/ml_factory_colab.runcheck.2pYcgH.ipynb`.
- Relevant EDA logic located in cell 7 (JSON lines around `notebooks/ml_factory_colab.runcheck.2pYcgH.ipynb:506`, `notebooks/ml_factory_colab.runcheck.2pYcgH.ipynb:529`, `notebooks/ml_factory_colab.runcheck.2pYcgH.ipynb:542`).

## What I Verified

### 1) The EDA code currently mixes returns across session boundaries
- Notebook computes returns with:
  - `returns = df['close'].pct_change().dropna()`
- Gap count (`>4h`) is only printed for diagnostics and is not used to segment returns.

### 2) Reproduced dataset-level facts from `data/raw/MGC_1m_5year.parquet`
- Rows: `1,601,940`
- Date range: `2021-01-03 17:00:00` -> `2025-07-16 00:00:00`
- Missing OHLCV: `0`
- Gaps > 4h: `244`
- Most common large gap: `2 days 01:01:00` (count `220`)
- Sunday reopen-like timing (Sun 16:00-19:00 timestamps): `232` of those large gaps

This supports your statement that most large gaps are market-closure shaped.

### 3) Kurtosis finding is real, but not only a weekend-gap artifact
Naive 1-bar returns (current notebook behavior):
- Kurtosis: `107.5403`
- Min/Max: `-0.01922` / `+0.01981`

If only removing first bar after `>4h` gaps:
- Kurtosis: `82.3930`

If removing first bar after `>10m` gaps:
- Kurtosis: `60.3813`

If removing first bar after `>2x median bar` gaps:
- Kurtosis: `60.4023`

If removing first **two** bars after `>10m` gaps:
- Kurtosis: `37.1039`

Interpretation:
- Yes, crossing gaps inflates kurtosis materially.
- But even with gap-boundary handling, tails remain very heavy; not all outliers come from weekend/holiday jumps.

### 4) Outlier clustering indicates two dominant mechanisms
For `|ret| >= 0.003` (0.3% in one bar), top time-of-day buckets:
- `07:30` (57 events)
- `17:00` (29 events)

For `|ret| >= 0.01` (1.0% in one bar):
- `17:00` appears 4 times
- `07:30` appears 1 time
- `23:51` appears 1 time

Interpretation:
- `17:00` events are strongly session-reopen related.
- `07:30` events are likely scheduled macro-release windows.
- There are also non-gap outliers that are genuine intraminute shocks (or potential feed artifacts).

### 5) Specific high-outlier context
Largest absolute return in file:
- `2023-07-30 23:51:00`, `+1.9806%`, **not** tagged as immediate post-gap bar
- It occurs one minute after a `33-minute` gap ended at `23:50`

This is important because a "drop first bar after gap" rule does not catch this event; dropping first 2 bars after long gaps does.

### 6) Additional integrity signal
- Duplicate timestamps found: `6` (all in June/July 2025, at `00:00:00`, duplicated values identical).
- This is small but worth tracking as a data hygiene note.

## Conclusion
- Your pasted diagnosis is directionally correct on major points:
  - Missing OHLCV is clean.
  - Most `>4h` gaps are expected market closures.
  - Naive 1-bar return kurtosis is inflated by boundary mixing.
- Additional finding:
  - Gap handling alone does not fully explain kurtosis `107`; substantial in-session/event-driven outliers remain.
- Practical implication for diagnostics:
  - If the goal is "within-session minute-return behavior," segmenting by gap boundary is necessary but likely should drop at least 2 bars after long gaps (not just 1).

---

## Preserved Messages (Verbatim)

### Message Block 1
This looks clean on the OHLCV side, and the “gaps” you’re seeing are almost certainly expected market-closure boundaries, not missing data.

What your summary is telling you
    •    Missing OHLCV: 0
✅ Every bar that exists has complete O/H/L/C/V.
    •    Gaps > 4h: 244
That count is very “weekend-shaped.” Over ~4.5 years, you’ll have ~230–240 weekends, plus a handful of holidays/early closes.
The examples confirm it:
    •    2 days 01:01:00 starting at 17:00:00 → that’s a classic Fri close → Sun reopen style gap (plus an extra ~hour-ish depending on your bar alignment/timezone/session rules).
    •    0 days 05:01:00 → that smells like a holiday halt / special schedule (e.g., MLK day-type behavior), where the market is closed longer than the normal daily maintenance window.

The one thing that is a red flag: return kurtosis = 107

That huge kurtosis (and min/max “1-bar” returns around ±1.9%) usually happens when you compute “1-bar returns” across multi-hour/multi-day gaps and treat them like normal adjacent minutes.

So your “1-bar return” stats are likely mixing:
    •    true minute-to-minute returns within sessions
    •    weekend/holiday gap jumps (which are not minute returns)

That will absolutely nuke kurtosis and inflate min/max.

What to do (the fix)

Treat gaps as segment boundaries and do not compute “1-bar returns” across them.

In pandas terms:

# df indexed by timestamp ascending, 1-minute bars
dt = df.index.to_series().diff()

# infer typical bar interval (e.g., 1 minute)
bar = dt.median()

# mark boundaries where the gap is "too large" to be a normal bar step
is_gap = dt > (bar * 2)

# compute returns, but break them at gaps
ret = df["close"].pct_change()
ret[is_gap] = float("nan")          # or 0.0, but NaN is better for stats/backtests

df["ret_1bar"] = ret
df["segment_id"] = is_gap.cumsum()  # useful for rolling features that must reset

Also recommended:
    •    Drop or ignore the first bar after a gap for indicator continuity (VWAP/EMA/rolling stats), or reset rolling windows per segment_id.
    •    If you’re using a continuous futures series, verify you’re not seeing roll jumps; those can also create huge outliers unless you back-adjust or handle rolls explicitly.

A quick sanity check after the fix

Recompute the “1-bar” return stats on ret_1bar.dropna() and you should see:
    •    kurtosis drop dramatically
    •    min/max tighten a lot (you’ll still have fat tails, but not “weekend jump pretending to be 1 minute” tails)

If you paste how you computed those “Return Stats (1-bar)” (log vs pct, close-to-close vs something else), I can tell you exactly which boundary rule to use (2×bar, 5×bar, session-clock-based, etc.).

### Message Block 2
This looks clean on the OHLCV side, and the “gaps” you’re seeing are almost certainly expected market-closure boundaries, not missing data.

What your summary is telling you
    •    Missing OHLCV: 0
✅ Every bar that exists has complete O/H/L/C/V.
    •    Gaps > 4h: 244
That count is very “weekend-shaped.” Over ~4.5 years, you’ll have ~230–240 weekends, plus a handful of holidays/early closes.
The examples confirm it:
    •    2 days 01:01:00 starting at 17:00:00 → that’s a classic Fri close → Sun reopen style gap (plus an extra ~hour-ish depending on your bar alignment/timezone/session rules).
    •    0 days 05:01:00 → that smells like a holiday halt / special schedule (e.g., MLK day-type behavior), where the market is closed longer than the normal daily maintenance window.

The one thing that is a red flag: return kurtosis = 107

That huge kurtosis (and min/max “1-bar” returns around ±1.9%) usually happens when you compute “1-bar returns” across multi-hour/multi-day gaps and treat them like normal adjacent minutes.

So your “1-bar return” stats are likely mixing:
    •    true minute-to-minute returns within sessions
    •    weekend/holiday gap jumps (which are not minute returns)

That will absolutely nuke kurtosis and inflate min/max.

What to do (the fix)

Treat gaps as segment boundaries and do not compute “1-bar returns” across them.

In pandas terms:

# df indexed by timestamp ascending, 1-minute bars
dt = df.index.to_series().diff()

# infer typical bar interval (e.g., 1 minute)
bar = dt.median()

# mark boundaries where the gap is "too large" to be a normal bar step
is_gap = dt > (bar * 2)

# compute returns, but break them at gaps
ret = df["close"].pct_change()
ret[is_gap] = float("nan")          # or 0.0, but NaN is better for stats/backtests

df["ret_1bar"] = ret
df["segment_id"] = is_gap.cumsum()  # useful for rolling features that must reset

Also recommended:
    •    Drop or ignore the first bar after a gap for indicator continuity (VWAP/EMA/rolling stats), or reset rolling windows per segment_id.
    •    If you’re using a continuous futures series, verify you’re not seeing roll jumps; those can also create huge outliers unless you back-adjust or handle rolls explicitly.

A quick sanity check after the fix

Recompute the “1-bar” return stats on ret_1bar.dropna() and you should see:
    •    kurtosis drop dramatically
    •    min/max tighten a lot (you’ll still have fat tails, but not “weekend jump pretending to be 1 minute” tails)

If you paste how you computed those “Return Stats (1-bar)” (log vs pct, close-to-close vs something else), I can tell you exactly which boundary rule to use (2×bar, 5×bar, session-clock-based, etc.).
