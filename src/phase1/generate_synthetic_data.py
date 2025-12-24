"""
Generate synthetic OHLCV data for MES and MGC futures.
This creates realistic-looking 1-minute bar data for pipeline testing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def generate_gbm_prices(n_bars: int, initial_price: float, mu: float = 0.0,
                        sigma: float = 0.001, dt: float = 1/252/1440) -> np.ndarray:
    """Generate prices using Geometric Brownian Motion."""
    np.random.seed(42)
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_bars)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    return np.exp(log_prices)

def generate_ohlcv(symbol: str, start_date: str, n_days: int = 500) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV data for a symbol."""
    logger.info(f"Generating {n_days} days of 1-min data for {symbol}")

    # Symbol-specific parameters
    if symbol == 'MES':
        initial_price = 4500.0
        tick_size = 0.25
        avg_volume = 1000
        volatility = 0.0012
    elif symbol == 'MGC':
        initial_price = 1950.0
        tick_size = 0.10
        avg_volume = 500
        volatility = 0.0010
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

    # Generate trading sessions (exclude weekends, assume 23hr trading)
    bars_per_day = 1380  # 23 hours * 60 minutes
    n_bars = n_days * bars_per_day

    # Generate close prices using GBM
    close_prices = generate_gbm_prices(n_bars, initial_price, sigma=volatility)

    # Generate OHLC from close
    high_low_range = np.abs(np.random.normal(0.002, 0.001, n_bars)) * close_prices
    high = close_prices + high_low_range * np.random.uniform(0.3, 0.7, n_bars)
    low = close_prices - high_low_range * np.random.uniform(0.3, 0.7, n_bars)

    # Open is previous close with small gap
    open_prices = np.roll(close_prices, 1) * (1 + np.random.normal(0, 0.0001, n_bars))
    open_prices[0] = initial_price

    # Ensure OHLC validity
    high = np.maximum(high, np.maximum(open_prices, close_prices))
    low = np.minimum(low, np.minimum(open_prices, close_prices))

    # Round to tick size
    close_prices = np.round(close_prices / tick_size) * tick_size
    open_prices = np.round(open_prices / tick_size) * tick_size
    high = np.round(high / tick_size) * tick_size
    low = np.round(low / tick_size) * tick_size

    # Generate volume (higher during RTH)
    volume = np.random.exponential(avg_volume, n_bars).astype(int) + 1

    # Generate timestamps (skip weekends)
    start = pd.Timestamp(start_date)
    timestamps = []
    current = start
    for i in range(n_bars):
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(days=1)
            current = current.replace(hour=0, minute=0)

        timestamps.append(current)
        current += timedelta(minutes=1)

        # Reset to next day at 23:00
        if current.hour == 23 and current.minute == 0:
            current = current.replace(hour=0, minute=0)
            current += timedelta(days=1)

    df = pd.DataFrame({
        'datetime': timestamps,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    })

    logger.info(f"Generated {len(df):,} bars for {symbol}")
    return df

def main():
    """Generate and save synthetic data for both symbols."""
    from config import RAW_DATA_DIR

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in ['MES', 'MGC']:
        df = generate_ohlcv(symbol, '2020-01-02', n_days=500)

        # Save as parquet (no need for CSV intermediate)
        output_path = RAW_DATA_DIR / f"{symbol}_1m.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {output_path}")

        # Print summary
        print(f"\n{symbol} Summary:")
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

if __name__ == "__main__":
    main()
