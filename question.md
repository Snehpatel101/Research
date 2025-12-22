1) Ingestion: get raw data in, keep it auditable

    Pull from vendor/API/CSV (ticks, 1m bars, L2, fundamentals, calendar events).

    Store raw, immutable copies (so you can reproduce results later).

    Track metadata: symbol, venue, timezone, session template, source, gaps, revisions.


2) Cleaning: fix the stuff that breaks models quietly

    Sort + dedupe, enforce monotonic timestamps.

    Handle missing bars/ticks (forward-fill? insert “empty” bars? mark missingness).

    Remove obvious bad prints (zero/negative prices, extreme spikes, out-of-range OHLC).

    Correct corporate actions (for equities) or contract rolls (for futures).


3) Normalization: make everything comparable and stable

    Resampling to standard intervals (1m/5m/1h) with consistent OHLCV rules.

    Time alignment across symbols/features (same timestamp grid).

    Session logic (RTH vs Globex, holidays, partial days) and timezone unification.

    Rolling/expanding stats that avoid lookahead (compute using only past data).

    Scaling transforms:

        price → returns/log-returns

        volume → z-scores or log(volume)

        volatility normalization (ATR, realized vol, etc.)

    Produce a canonical “features-ready” table: one row per timestamp per instrument.


4) Feature building: convert bars into signals the model can use


Common families:

    Returns/volatility (multi-horizon, realized vol, ATR, range metrics)

    Trend + momentum (MA/EMA slopes, RSI, MACD variants)

    Microstructure proxies (if you have it): spread, imbalance, trade intensity

    Regime/context: session, day-of-week, news windows, volatility regime flags

    Cross-asset features (indexes, correlated contracts, risk-on/off baskets)


Key rule: features must be computed with information available at that time (no leakage).


5) Labeling: define “what is the model trying to predict?”


This is where pipelines differ most. Common label styles:


Directional / horizon returns

    “Will return over next N bars be > 0?” (classification)

    “Next N-bar return value” (regression)


Threshold / event labels

    “Will price move +X ticks before -Y ticks within T minutes?”

    Fixed take-profit/stop outcomes


Triple-barrier labeling (common in trading ML)

    For each time t, define:

        upper barrier (+profit)

        lower barrier (-loss)

        vertical barrier (time limit)

    Label is which barrier hits first (up/down/none)


Meta-labeling

    First model proposes “trade/no trade”

    Second model predicts “will this trade work?” (filters bad signals)


6) Splitting + leakage controls: make it trainable without lying to yourself

    Time-based splits (train → validation → test) rather than random.

    Purging/embargo around split boundaries (prevents overlapping label windows).

    Ensure scaler/normalizer is fit on train only, applied forward.


7) Packaging: deliverables most pipelines produce

    Clean canonical bars (per symbol, per timeframe)

    Feature matrix (wide or long format) + feature dictionary

    Labels aligned to rows (with the exact spec recorded)

    Train/val/test indices (reproducible)

    Quality reports (gap %, outliers removed, label distribution, leakage checks)

