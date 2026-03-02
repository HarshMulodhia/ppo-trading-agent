"""
Technical Indicators Module

Provides efficient implementations of common technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
"""

from typing import Tuple

import numpy as np


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index (RSI).

    RSI measures the magnitude of recent price changes to evaluate overbought
    or oversold conditions. Values range from 0 to 100.
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)

    Args:
        prices: Array of closing prices
        period: Lookback period (default 14)

    Returns:
        RSI values (0-100)
    """
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    # Calculate price changes
    deltas = np.diff(prices)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate exponential moving averages
    avg_gain = np.zeros_like(prices, dtype=np.float64)
    avg_loss = np.zeros_like(prices, dtype=np.float64)

    # First period uses simple average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Subsequent periods use exponential smoothing
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # Calculate RS and RSI
    rs = np.divide(
        avg_gain, avg_loss, where=avg_loss != 0, out=np.full_like(avg_loss, 0.0)
    )
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator. Components:
    - MACD Line: 12-period EMA - 26-period EMA
    - Signal Line: 9-period EMA of MACD
    - Histogram: MACD - Signal

    Args:
        prices: Array of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        (macd_line, signal_line, histogram)
    """
    # Compute exponential moving averages
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line (EMA of MACD)
    signal_line = exponential_moving_average(macd_line, signal)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Bollinger Bands.

    Bollinger Bands measure volatility and potential overbought/oversold levels.
    Components:
    - Middle Band: SMA
    - Upper Band: SMA + (std_dev × StdDev)
    - Lower Band: SMA - (std_dev × StdDev)

    Args:
        prices: Array of closing prices
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        return np.full_like(prices, 0.0), prices, np.full_like(prices, 0.0)

    # Simple moving average
    middle = np.convolve(prices, np.ones(period) / period, mode="valid")

    # Pad to match input length
    middle = np.concatenate([np.full(period - 1, prices[0]), middle])

    # Standard deviation
    std = np.full_like(prices, 0.0, dtype=np.float64)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1 : i + 1])

    # Bands
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def exponential_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Exponential Moving Average (EMA).

    EMA gives more weight to recent prices using a smoothing factor.

    Args:
        prices: Array of prices
        period: EMA period

    Returns:
        EMA values
    """
    if len(prices) == 0:
        return prices

    ema = np.zeros_like(prices, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)

    # First EMA is SMA
    ema[0] = np.mean(prices[: min(period, len(prices))])

    # Subsequent values use exponential smoothing
    for i in range(1, len(prices)):
        ema[i] = prices[i] * alpha + ema[i - 1] * (1 - alpha)

    return ema


def simple_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Simple Moving Average (SMA).

    Args:
        prices: Array of prices
        period: SMA period

    Returns:
        SMA values
    """
    if len(prices) < period:
        return np.full_like(prices, np.mean(prices))

    sma = np.convolve(prices, np.ones(period) / period, mode="valid")
    return np.concatenate([np.full(period - 1, sma[0]), sma])
