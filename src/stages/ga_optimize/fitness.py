"""
Fitness functions for GA optimization.

Contains:
    - calculate_fitness: Main fitness function for label quality evaluation
    - evaluate_individual: Evaluate a GA individual (parameter set)
"""

import logging
from typing import List, Tuple

import numpy as np

# Import labeling function and config
from src.stages.labeling import triple_barrier_numba
from src.config import TICK_VALUES, TRANSACTION_COSTS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
) -> float:
    """
    Calculate fitness score for a set of labels.

    IMPROVEMENTS:
    1. Require at least 60% directional signals (target 70-80% signal rate)
    2. Target 20-30% neutral rate (was <2%)
    3. Reward balanced long/short ratio
    4. Speed score prefers faster resolution
    5. Profit factor using theoretical barrier distances (not MAE/MFE)
    6. TRANSACTION COST penalty with round-trip costs

    Parameters:
    -----------
    labels : array of -1, 0, 1
    bars_to_hit : array of bars until barrier hit
    mae : Maximum Adverse Excursion (unused - kept for API compatibility)
    mfe : Maximum Favorable Excursion (unused - kept for API compatibility)
    horizon : horizon identifier (5 or 20)
    atr_mean : mean ATR for the sample (for transaction cost normalization)
    symbol : 'MES' or 'MGC' (for symbol-specific transaction costs)
    k_up : upper barrier multiplier (ATR units)
    k_down : lower barrier multiplier (ATR units)

    Returns:
    --------
    fitness : float (higher is better)
    """
    total = len(labels)
    if total == 0:
        return -1000.0

    n_long = (labels == 1).sum()
    n_short = (labels == -1).sum()
    n_neutral = (labels == 0).sum()

    # ==========================================================================
    # 1. SIGNAL RATE REQUIREMENT (target 70-80% signals, 20-30% neutral)
    # ==========================================================================
    neutral_pct = n_neutral / total

    # Target: 20-30% neutral rate
    if neutral_pct < 0.15:
        # Too few neutrals - trading on noise
        return -1000.0 + (neutral_pct * 100)
    elif neutral_pct > 0.40:
        # Too many neutrals - not enough trading signals
        return -1000.0 + ((1 - neutral_pct) * 100)

    # ==========================================================================
    # 2. NEUTRAL SCORE (target 20-30% neutral)
    # ==========================================================================
    TARGET_NEUTRAL_LOW = 0.20
    TARGET_NEUTRAL_HIGH = 0.30

    if TARGET_NEUTRAL_LOW <= neutral_pct <= TARGET_NEUTRAL_HIGH:
        neutral_score = 3.0  # Perfect range
    else:
        # Penalize deviation from target range
        if neutral_pct < TARGET_NEUTRAL_LOW:
            deviation = TARGET_NEUTRAL_LOW - neutral_pct
        else:
            deviation = neutral_pct - TARGET_NEUTRAL_HIGH
        neutral_score = 3.0 - (deviation * 15.0)

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
    # 6. TRANSACTION COST PENALTY (round-trip: entry + exit)
    # ==========================================================================
    cost_ticks = TRANSACTION_COSTS.get(symbol, 0.5)
    tick_value = TICK_VALUES.get(symbol, 1.0)
    # Round-trip cost: entry + exit = 2x one-way cost
    cost_in_price_units = 2 * cost_ticks * tick_value

    n_trades = n_longs + n_shorts
    if n_trades > 0 and atr_mean > 0:
        # Convert profit from ATR units to price units for comparison
        avg_profit_per_trade_atr = total_profit / n_trades
        avg_profit_per_trade = avg_profit_per_trade_atr * atr_mean
        cost_ratio = cost_in_price_units / (avg_profit_per_trade + 1e-6)

        if cost_ratio > 0.20:
            transaction_penalty = -(cost_ratio - 0.20) * 10.0
        else:
            transaction_penalty = 0.5
    else:
        transaction_penalty = 0.0

    # ==========================================================================
    # COMBINED FITNESS
    # ==========================================================================
    fitness = neutral_score + balance_score + speed_score + pf_score + transaction_penalty

    return fitness


def evaluate_individual(
    individual: List[float],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    symbol: str = "MES",
) -> Tuple[float]:
    """
    Evaluate a GA individual (parameter set).

    Parameters:
    -----------
    individual : [k_up, k_down, max_bars_multiplier]
    close, high, low, open_prices, atr : price/indicator arrays
    horizon : horizon value
    symbol : 'MES' or 'MGC' for symbol-specific constraints

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
            asymmetry_bonus = -abs(k_up - k_down) * 5.0
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
                asymmetry_bonus = (k_up - k_down) * 2.0
            elif asymmetry_ratio > 1.8:
                # Too extreme - still positive but reduced
                asymmetry_bonus = (k_up - k_down) * 0.5
            else:
                # Slight asymmetry (ratio < 1.2) - small reward
                asymmetry_bonus = (k_up - k_down) * 1.0
        else:
            # WRONG direction: k_down > k_up amplifies long bias
            # This makes shorts harder when market already drifts up - penalize
            asymmetry_bonus = -(k_down - k_up) * 3.0

    # Decode max_bars
    max_bars = int(horizon * max_bars_mult)
    max_bars = max(horizon * 2, min(max_bars, horizon * 3))

    # Run labeling
    try:
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
        )
        fitness += asymmetry_bonus

        return (fitness,)

    except Exception as e:
        logger.warning(
            f"Fitness evaluation failed for individual "
            f"[k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}]: {e}"
        )
        return (float("-inf"),)
