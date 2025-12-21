"""
Constants for feature engineering.

These constants are used for annualization of volatility metrics
and other time-based calculations in 5-minute bar data.
"""

import numpy as np

# Bars per trading day for 5-minute data
# 390 trading minutes / 5 minutes per bar = 78 bars
BARS_PER_DAY = 78

# Standard trading days per year
TRADING_DAYS_PER_YEAR = 252

# Annualization factor for volatility calculations
# Annualization: daily_vol = bar_vol * sqrt(bars_per_day)
#                annual_vol = daily_vol * sqrt(trading_days)
#                           = bar_vol * sqrt(bars_per_day * trading_days)
#                           = bar_vol * sqrt(78 * 252) = bar_vol * 140.07
ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR * BARS_PER_DAY)  # ~140.07

__all__ = [
    'BARS_PER_DAY',
    'TRADING_DAYS_PER_YEAR',
    'ANNUALIZATION_FACTOR',
]
