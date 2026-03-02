"""
Performance Metrics

Comprehensive performance metrics for trading agent evaluation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_return: float
    std_return: float
    annual_return: float
    num_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "std_return": self.std_return,
            "annual_return": self.annual_return,
            "num_trades": self.num_trades,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"PerformanceMetrics(\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.3f}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Calmar Ratio: {self.calmar_ratio:.3f}\n"
            f"  Win Rate: {self.win_rate:.1%}\n"
            f"  Annual Return: {self.annual_return:.2%}\n"
            f")"
        )


def normalize_features(
    features: np.ndarray, method: str = "zscore", epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize features using specified method.

    Methods:
    - 'zscore': (x - mean) / std (zero mean, unit variance)
    - 'minmax': (x - min) / (max - min) (scale to [0, 1])

    Args:
        features: Feature array (can be 1D or 2D)
        method: Normalization method ('zscore' or 'minmax')
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized features
    """
    if method == "zscore":
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        return (features - mean) / (std + epsilon)

    elif method == "minmax":
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        return (features - min_val) / (max_val - min_val + epsilon)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_drawdown(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Compute drawdown (underwater curve) for portfolio.

    Drawdown = (Current Value - Peak Value) / Peak Value
    Negative value indicates decline from peak.

    Args:
        portfolio_values: Array of portfolio values over time

    Returns:
        Drawdown values (negative = underwater, 0 = at peak)
    """
    if len(portfolio_values) == 0:
        return np.array([])

    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max

    return drawdown


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Compute maximum drawdown percentage.

    Maximum drawdown is the largest peak-to-trough decline.

    Args:
        portfolio_values: Array of portfolio values

    Returns:
        Maximum drawdown as negative percentage
    """
    if len(portfolio_values) < 2:
        return 0.0

    drawdown = compute_drawdown(portfolio_values)
    return float(np.min(drawdown))


def compute_sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Compute Sharpe Ratio.

    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns
    Annualized: multiply by sqrt(periods_per_year)

    Args:
        returns: Array of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Risk-free rate (default 0%)
        periods_per_year: Periods per year for annualization (default 252 for daily)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = np.array(returns) - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return float(sharpe)


def compute_sortino_ratio(
    returns: np.ndarray, target_return: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Compute Sortino Ratio.

    Sortino Ratio = (Mean Return - Target Return) / Downside Deviation
    Only penalizes downside volatility (not upside).

    Args:
        returns: Array of period returns
        target_return: Target return (default 0%)
        periods_per_year: Periods per year for annualization

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = np.array(returns) - target_return

    # Downside deviation (only negative returns)
    downside = np.minimum(excess_returns, 0)
    downside_std = np.std(downside)

    if downside_std == 0:
        return 0.0

    mean_return = np.mean(excess_returns)
    sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)

    return float(sortino)


def compute_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Compute Calmar Ratio.

    Calmar Ratio = Annual Return / Max Drawdown
    Higher is better; directly compares absolute return to worst-case loss.

    Args:
        returns: Array of period returns
        periods_per_year: Periods per year

    Returns:
        Calmar Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Compute cumulative return
    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
    annual_return = (1 + cumulative_returns[-1]) ** (periods_per_year / len(returns)) - 1

    # Compute max drawdown from returns
    portfolio_values = np.cumprod(1 + np.array(returns))
    max_dd = compute_max_drawdown(portfolio_values)

    if max_dd == 0 or max_dd > 0:  # No drawdown or error
        return 0.0

    calmar = annual_return / abs(max_dd)
    return float(calmar)


def compute_win_rate(returns: np.ndarray) -> float:
    """
    Compute win rate (percentage of positive returns).

    Args:
        returns: Array of period returns

    Returns:
        Win rate (0.0 to 1.0)
    """
    if len(returns) == 0:
        return 0.0

    positive_returns = np.sum(np.array(returns) > 0)
    return float(positive_returns / len(returns))


def compute_total_return(prices: np.ndarray) -> float:
    """
    Compute total return from prices.

    Args:
        prices: Array of prices

    Returns:
        Total return as decimal
    """
    if len(prices) < 2 or prices[0] == 0:
        return 0.0

    return float((prices[-1] - prices[0]) / prices[0])


def clip_values(values: np.ndarray, min_val: float = -5.0, max_val: float = 5.0) -> np.ndarray:
    """
    Clip values to range to handle outliers.

    Args:
        values: Values to clip
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped values
    """
    return np.clip(values, min_val, max_val)


def validate_prices(prices: np.ndarray, min_length: int = 100) -> bool:
    """
    Validate price array.

    Args:
        prices: Price array
        min_length: Minimum required length

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(prices, np.ndarray):
        return False

    if len(prices) < min_length:
        return False

    if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
        return False

    if np.any(prices <= 0):
        return False

    return True


def compute_metrics(
    portfolio_values: np.ndarray,
    num_trades: int = 0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Args:
        portfolio_values: Array of portfolio values
        num_trades: Number of trades executed
        periods_per_year: Periods per year

    Returns:
        PerformanceMetrics object
    """
    # Compute returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Compute metrics
    total_return = compute_total_return(portfolio_values)
    sharpe = compute_sharpe_ratio(returns, periods_per_year)
    sortino = compute_sortino_ratio(returns, periods_per_year)
    max_dd = compute_max_drawdown(portfolio_values)
    calmar = compute_calmar_ratio(returns, periods_per_year)
    win_rate = compute_win_rate(returns)

    # Annual return
    annual_return = total_return * (periods_per_year / len(returns))

    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        avg_return=np.mean(returns),
        std_return=np.std(returns),
        annual_return=annual_return,
        num_trades=num_trades,
    )
