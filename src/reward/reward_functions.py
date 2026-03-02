"""
Reward Functions Module

Multiple reward function implementations for reinforcement learning.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sharpe_reward(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.02,
    **kwargs,
) -> float:
    """
    Compute Sharpe ratio as reward.

    Args:
        portfolio_values: Historical portfolio values
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - (risk_free_rate / 252)

    sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    return float(sharpe)


def pnl_reward(
    current_portfolio_value: float,
    initial_portfolio_value: float,
    transaction_cost: float = 0.0,
    **kwargs,
) -> float:
    """
    Profit/loss based reward.

    Args:
        current_portfolio_value: Current portfolio value
        initial_portfolio_value: Initial portfolio value
        transaction_cost: Transaction cost

    Returns:
        PnL reward
    """
    pnl = (current_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    pnl = pnl - transaction_cost  # Penalize transaction costs
    return float(pnl)


def risk_adjusted_reward(
    portfolio_values: np.ndarray,
    risk_penalty: float = 0.1,
    **kwargs,
) -> float:
    """
    Risk-adjusted reward.

    Args:
        portfolio_values: Historical portfolio values
        risk_penalty: Risk penalty coefficient

    Returns:
        Risk-adjusted reward
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    volatility = np.std(returns)

    reward = mean_return - risk_penalty * volatility
    return float(reward)


def sortino_reward(
    portfolio_values: np.ndarray,
    risk_free_rate: float = 0.02,
    target_return: float = 0.0,
    **kwargs,
) -> float:
    """
    Sortino ratio as reward.

    Args:
        portfolio_values: Historical portfolio values
        risk_free_rate: Risk-free rate
        target_return: Target return threshold

    Returns:
        Sortino ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - (risk_free_rate / 252)

    downside_returns = np.minimum(excess_returns - target_return, 0)
    downside_volatility = np.sqrt(np.mean(downside_returns**2))

    sortino = np.mean(excess_returns) / (downside_volatility + 1e-8) * np.sqrt(252)
    return float(sortino)


def calmar_reward(
    portfolio_values: np.ndarray,
    periods_per_year: int = 252,
    **kwargs,
) -> float:
    """
    Calmar ratio as reward.

    Args:
        portfolio_values: Historical portfolio values
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annual_return = total_return * periods_per_year / len(returns)

    # Max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = np.abs(np.min(drawdown))

    if max_drawdown == 0:
        return float(annual_return)

    calmar = annual_return / max_drawdown
    return float(calmar)


class RewardFunction:
    """
    Configurable reward function.

    Combines multiple reward components with weights.
    """

    def __init__(
        self,
        components: dict,
        weights: dict,
    ):
        """
        Initialize reward function.

        Args:
            components: Dict of reward component names and callables
            weights: Dict of component weights
        """
        self.components = components
        self.weights = weights

        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}

    def compute(self, **kwargs) -> float:
        """
        Compute total reward.

        Args:
            **kwargs: Arguments for reward components

        Returns:
            Total reward
        """
        total_reward = 0.0

        for component_name, weight in self.weights.items():
            if component_name in self.components:
                component_func = self.components[component_name]
                component_reward = component_func(**kwargs)
                total_reward += weight * component_reward

        return float(total_reward)
