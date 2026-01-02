"""
Fitness functions for GA optimization.

Contains:
    - calculate_fitness: Main fitness function for label quality evaluation
    - evaluate_individual: Evaluate a GA individual (parameter set)
"""

import numpy as np

from src.phase1.config import LABEL_BALANCE_CONSTRAINTS, TICK_VALUES, get_total_trade_cost

# Import labeling function and config
from src.phase1.stages.labeling import triple_barrier_numba


def calculate_fitness(
    labels: np.ndarray,
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    horizon: int,
    atr_mean: float,
    symbol: str = "MES",
    k_up: float = 1.0,
    k_down: float = 1.0,
    regime: str = "low_vol",
    include_slippage: bool = True,
) -> float:
    """
    Calculate fitness score for a set of labels.

    CRITICAL: This function MUST preserve neutral labels (target 20-30%).
    Neutral labels represent "no trade" / timeout scenarios which are essential for:
    1. Avoiding overtrading and transaction costs
    2. Filtering low-confidence / noisy signals
    3. Realistic signal rates in production

    Components:
    1. HARD constraints: Reject solutions with <10% neutral or <10% any class
    2. Neutral score: Strong reward for 20-30% neutral, heavy penalty below 15%
    3. Balance score: Reward balanced long/short ratio
    4. Speed score: Prefer faster resolution (within reason)
    5. Profit factor: Using actual MFE/MAE from labeling
    6. Transaction cost penalty: Round-trip costs (commission + slippage)

    Parameters:
    -----------
    labels : array of -1, 0, 1
    bars_to_hit : array of bars until barrier hit
    mae : Maximum Adverse Excursion
    mfe : Maximum Favorable Excursion
    horizon : horizon identifier (5, 10, 15, or 20)
    atr_mean : mean ATR for the sample (for transaction cost normalization)
    symbol : 'MES' or 'MGC' (for symbol-specific transaction costs)
    k_up : upper barrier multiplier (ATR units)
    k_down : lower barrier multiplier (ATR units)
    regime : str, optional
        Volatility regime: 'low_vol' or 'high_vol' (default: 'low_vol')
    include_slippage : bool, optional
        Whether to include slippage in cost calculation (default: True)

    Returns:
    --------
    fitness : float (higher is better)
    """
    total = len(labels)
    if total == 0:
        return -10000.0  # Catastrophic failure

    n_long = (labels == 1).sum()
    n_short = (labels == -1).sum()
    n_neutral = (labels == 0).sum()
    long_pct = n_long / total
    short_pct = n_short / total
    neutral_pct = n_neutral / total

    # ==========================================================================
    # 0. HARD CONSTRAINTS - ABSOLUTELY NON-NEGOTIABLE
    # These return values that CANNOT be recovered from by other components
    # ==========================================================================
    min_long_pct = LABEL_BALANCE_CONSTRAINTS["min_long_pct"]
    min_short_pct = LABEL_BALANCE_CONSTRAINTS["min_short_pct"]
    min_neutral_pct = LABEL_BALANCE_CONSTRAINTS["min_neutral_pct"]  # 10%
    max_neutral_pct = LABEL_BALANCE_CONSTRAINTS["max_neutral_pct"]  # 40%
    min_any_class_pct = LABEL_BALANCE_CONSTRAINTS["min_any_class_pct"]  # 10%

    # HARD CONSTRAINT 1: Minimum neutral percentage (CRITICAL)
    # This is THE most important constraint to prevent neutral destruction
    if neutral_pct < min_neutral_pct:
        # Return extremely negative value - no recovery possible
        # Scale: -10000 base, plus small gradient to guide optimization
        return -10000.0 + (neutral_pct * 10.0)

    # HARD CONSTRAINT 2: Maximum neutral percentage (not enough signals)
    if neutral_pct > max_neutral_pct:
        return -10000.0 + ((1.0 - neutral_pct) * 10.0)

    # HARD CONSTRAINT 3: Minimum long/short percentages
    if long_pct < min_long_pct or short_pct < min_short_pct:
        return -10000.0 + ((long_pct + short_pct) * 10.0)

    # HARD CONSTRAINT 4: Any class below absolute minimum (fail-safe)
    if long_pct < min_any_class_pct or short_pct < min_any_class_pct:
        return -10000.0 + (min(long_pct, short_pct) * 10.0)

    # HARD CONSTRAINT 5: Long/short signal ratio balance
    signal_count = n_long + n_short
    if signal_count > 0:
        short_signal_ratio = n_short / signal_count
        min_short_ratio = LABEL_BALANCE_CONSTRAINTS["min_short_signal_ratio"]
        max_short_ratio = LABEL_BALANCE_CONSTRAINTS["max_short_signal_ratio"]
        if short_signal_ratio < min_short_ratio or short_signal_ratio > max_short_ratio:
            distance = abs(short_signal_ratio - 0.5)
            return -10000.0 + ((1.0 - distance) * 10.0)

    # ==========================================================================
    # 1. NEUTRAL SCORE (HIGH WEIGHT - target 20-30% neutral)
    # This is the PRIMARY driver for preserving neutral labels
    # ==========================================================================
    target_neutral_low = LABEL_BALANCE_CONSTRAINTS["target_neutral_low"]  # 0.20
    target_neutral_high = LABEL_BALANCE_CONSTRAINTS["target_neutral_high"]  # 0.30

    if target_neutral_low <= neutral_pct <= target_neutral_high:
        # Perfect range: maximum reward
        neutral_score = 10.0
    elif neutral_pct < target_neutral_low:
        # Below target: graduated penalty (steeper as we approach minimum)
        # At 15%: penalty = (0.20 - 0.15) * 40 = 2.0, score = 10 - 2 = 8.0
        # At 12%: penalty = (0.20 - 0.12) * 40 = 3.2, score = 10 - 3.2 = 6.8
        # At 10%: penalty = (0.20 - 0.10) * 40 = 4.0, score = 10 - 4 = 6.0
        deviation = target_neutral_low - neutral_pct
        neutral_score = 10.0 - (deviation * 40.0)
        # Additional steep penalty below 15% (approaching hard constraint)
        if neutral_pct < 0.15:
            neutral_score -= (0.15 - neutral_pct) * 50.0
    else:
        # Above target: moderate penalty
        deviation = neutral_pct - target_neutral_high
        neutral_score = 10.0 - (deviation * 20.0)

    # ==========================================================================
    # 3. LONG/SHORT BALANCE SCORE
    # ==========================================================================
    if (n_long + n_short) > 0:
        long_ratio = n_long / (n_long + n_short)
        # Perfect balance = 0.5, max score = 2.0
        balance_score = 2.0 - abs(long_ratio - 0.5) * 8.0

        # Additional penalty for extreme imbalance
        if long_ratio < 0.30 or long_ratio > 0.70:
            balance_score -= 3.0
    else:
        balance_score = -10.0

    # ==========================================================================
    # 4. SPEED SCORE (prefer faster resolution within reason)
    # ==========================================================================
    hit_mask = labels != 0
    if hit_mask.sum() > 0:
        avg_bars = bars_to_hit[hit_mask].mean()
        max_expected = horizon * 2.0
        speed_score = 1.0 - (avg_bars / max_expected)
        speed_score = np.clip(speed_score, -2.0, 1.5)
    else:
        speed_score = -2.0

    # ==========================================================================
    # 5. PROFIT FACTOR SCORE (using actual MFE/MAE from labeling)
    #
    # CORRECTED (2025-12): Use actual Maximum Favorable Excursion (MFE) instead
    # of theoretical barrier distances for more accurate profit estimation.
    #
    # For a LONG trade (label=1): profit = actual MFE (how far price moved up)
    # For a SHORT trade (label=-1): profit = abs(MAE) (how far price moved down)
    # ==========================================================================
    long_mask = labels == 1
    short_mask = labels == -1

    n_longs = long_mask.sum()
    n_shorts = short_mask.sum()

    # Use actual MFE/MAE for profit calculation (more accurate than theoretical)
    # MFE = Maximum Favorable Excursion (positive for longs moving up)
    # MAE = Maximum Adverse Excursion (negative for longs moving down)
    if n_longs > 0:
        long_profit = mfe[long_mask].sum()  # Actual favorable excursion for longs
        long_risk = np.abs(mae[long_mask]).sum()  # Actual adverse excursion (risk)
    else:
        long_profit = 0.0
        long_risk = 0.0

    if n_shorts > 0:
        # For shorts, MAE (price going down) is actually profit
        short_profit = np.abs(mae[short_mask]).sum()  # Price drop = short profit
        short_risk = mfe[short_mask].sum()  # Price rise = short risk
    else:
        short_profit = 0.0
        short_risk = 0.0

    total_profit = long_profit + short_profit
    total_risk = long_risk + short_risk

    if total_risk > 0:
        profit_factor = total_profit / total_risk

        if 1.0 <= profit_factor <= 2.0:
            pf_score = 2.0
        elif 0.8 <= profit_factor < 1.0:
            pf_score = 0.5
        elif profit_factor > 2.0:
            pf_score = 1.5
        else:
            pf_score = -2.0
    else:
        pf_score = 0.0

    # ==========================================================================
    # 6. TRANSACTION COST PENALTY (round-trip: commission + slippage)
    # ==========================================================================
    # Get total round-trip cost (commission + slippage) in ticks
    cost_ticks = get_total_trade_cost(symbol, regime, include_slippage)
    tick_value = TICK_VALUES.get(symbol, 1.0)

    # Convert to price units
    cost_in_price_units = cost_ticks * tick_value

    n_trades = n_longs + n_shorts
    if n_trades > 0 and atr_mean > 0:
        # Convert profit from ATR units to price units for comparison
        avg_profit_per_trade_atr = total_profit / n_trades
        avg_profit_per_trade = avg_profit_per_trade_atr * atr_mean
        cost_ratio = cost_in_price_units / (avg_profit_per_trade + 1e-6)

        if cost_ratio > 0.20:
            # Cap penalty at -10.0 to prevent extreme values from low profit trades
            transaction_penalty = max(-10.0, -(cost_ratio - 0.20) * 10.0)
        else:
            transaction_penalty = 0.5
    else:
        transaction_penalty = 0.0

    # ==========================================================================
    # 7. CLASS BALANCE PENALTY (penalize any class being too low)
    # Additional safety to prevent any single class from being minimized
    # ==========================================================================
    min_class_pct = min(long_pct, short_pct, neutral_pct)
    if min_class_pct < 0.15:
        # Graduated penalty as any class approaches minimum
        # At 12%: penalty = (0.15 - 0.12) * 30 = 0.9
        # At 10%: penalty = (0.15 - 0.10) * 30 = 1.5
        class_balance_penalty = -(0.15 - min_class_pct) * 30.0
    else:
        # Small reward for balanced distribution
        class_balance_penalty = 0.5

    # ==========================================================================
    # COMBINED FITNESS
    # ==========================================================================
    # Component weights (approximate max values):
    # - neutral_score: up to 10.0 (PRIMARY driver for neutral preservation)
    # - balance_score: up to 2.0 (long/short balance)
    # - speed_score: up to 1.5 (resolution speed)
    # - pf_score: up to 2.0 (profit factor)
    # - transaction_penalty: -10 to 0.5 (cost awareness)
    # - class_balance_penalty: -4.5 to 0.5 (class distribution safety)
    #
    # Total possible range: approximately -10 to +16
    # The neutral_score dominates, ensuring neutral preservation is prioritized
    fitness = (
        neutral_score
        + balance_score
        + speed_score
        + pf_score
        + transaction_penalty
        + class_balance_penalty
    )

    return fitness


def evaluate_individual(
    individual: list[float],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    symbol: str = "MES",
    regime: str = "low_vol",
    include_slippage: bool = True,
) -> tuple[float]:
    """
    Evaluate a GA individual (parameter set).

    Parameters:
    -----------
    individual : [k_up, k_down, max_bars_multiplier]
    close, high, low, open_prices, atr : price/indicator arrays
    horizon : horizon value
    symbol : 'MES' or 'MGC' for symbol-specific constraints
    regime : str, optional
        Volatility regime: 'low_vol' or 'high_vol' (default: 'low_vol')
    include_slippage : bool, optional
        Whether to include slippage in cost calculation (default: True)

    Returns:
    --------
    (fitness,) : tuple with single fitness value (DEAP convention)
    """
    k_up, k_down, max_bars_mult = individual

    # ==========================================================================
    # SYMBOL-SPECIFIC ASYMMETRY CONSTRAINT
    #
    # CORRECTED (2025-12): Fixed asymmetry direction for MES
    #
    # MES (S&P 500 E-mini): Has ~7% annual equity drift (long bias).
    #   - MES naturally drifts UP, making UPPER barrier easier to hit
    #   - To counteract this drift, we WANT k_up > k_down
    #   - This makes the upper barrier HARDER to hit, reducing long signals
    #   - Target asymmetry: k_up ~1.5x k_down (e.g., k_up=1.5, k_down=1.0)
    #
    # MGC (Micro Gold): Mean-reverting, no directional drift.
    #   - Symmetric barriers are appropriate (k_down ≈ k_up)
    #   - Penalize any significant asymmetry
    # ==========================================================================
    avg_k = (k_up + k_down) / 2.0

    if symbol == "MGC":
        # MGC: Strict symmetry - penalize asymmetry > 10%
        if abs(k_up - k_down) > avg_k * 0.10:
            # Cap penalty at -5.0 to prevent extreme values
            asymmetry_bonus = max(-5.0, -abs(k_up - k_down) * 5.0)
        else:
            asymmetry_bonus = 0.5  # Small reward for good symmetry
    else:
        # MES: REWARD k_up > k_down to counteract equity drift (CORRECTED)
        # Upper barrier should be HARDER (larger k_up) to balance upward drift
        if k_up > k_down:
            # Correct configuration - reward the asymmetry
            # Bonus scales with asymmetry magnitude (capped at reasonable levels)
            asymmetry_ratio = k_up / k_down if k_down > 0 else 1.0
            if 1.2 <= asymmetry_ratio <= 1.8:
                # Ideal range: k_up is 20-80% larger than k_down
                asymmetry_bonus = min(3.0, (k_up - k_down) * 2.0)
            elif asymmetry_ratio > 1.8:
                # Too extreme - still positive but reduced
                asymmetry_bonus = min(1.5, (k_up - k_down) * 0.5)
            else:
                # Slight asymmetry (ratio < 1.2) - small reward
                asymmetry_bonus = min(2.0, (k_up - k_down) * 1.0)
        else:
            # WRONG direction: k_down > k_up amplifies long bias
            # This makes shorts harder when market already drifts up - penalize
            # Cap penalty at -5.0 to prevent extreme values
            asymmetry_bonus = max(-5.0, -(k_down - k_up) * 3.0)

    # Decode max_bars
    max_bars = int(horizon * max_bars_mult)
    max_bars = max(horizon * 2, min(max_bars, horizon * 3))

    labels, bars_to_hit, mae, mfe, _ = triple_barrier_numba(
        close, high, low, open_prices, atr, k_up, k_down, max_bars
    )

    atr_mean = np.mean(atr[atr > 0]) if np.any(atr > 0) else 1.0

    fitness = calculate_fitness(
        labels,
        bars_to_hit,
        mae,
        mfe,
        horizon,
        atr_mean=atr_mean,
        symbol=symbol,
        k_up=k_up,
        k_down=k_down,
        regime=regime,
        include_slippage=include_slippage,
    )
    fitness += asymmetry_bonus

    return (fitness,)
