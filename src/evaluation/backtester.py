"""
Backtester Module

Historical backtesting engine for trading agents.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float


class Portfolio:
    """Portfolio tracking during backtest."""

    def __init__(self, initial_capital: float, trading_costs: float = 0.001):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital
            trading_costs: Cost per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.trading_costs = trading_costs

        self.equity_curve = [initial_capital]
        self.trades: List[Trade] = []
        self.entry_price = 0
        self.entry_time = 0

    def buy(self, price: float, size: float, time: int) -> None:
        """Execute buy order."""
        cost = price * size * (1 + self.trading_costs)
        self.cash -= cost
        self.position += size
        self.entry_price = price
        self.entry_time = time

    def sell(self, price: float, size: float, time: int) -> None:
        """Execute sell order."""
        revenue = price * size * (1 - self.trading_costs)
        self.cash += revenue

        if self.position > 0:
            pnl = (price - self.entry_price) * size
            pnl_pct = (price - self.entry_price) / self.entry_price

            trade = Trade(
                entry_time=self.entry_time,
                exit_time=time,
                entry_price=self.entry_price,
                exit_price=price,
                size=size,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
            self.trades.append(trade)

        self.position -= size

    def update_equity(self, current_price: float) -> float:
        """Update equity value."""
        position_value = self.position * current_price
        total_equity = self.cash + position_value
        self.equity_curve.append(total_equity)
        self.current_capital = total_equity
        return total_equity

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        return np.array(self.equity_curve)

    def get_trades(self) -> List[Trade]:
        """Get list of trades."""
        return self.trades

    def get_returns(self) -> np.ndarray:
        """Get daily returns."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        return returns


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.

    Features:
    - Historical data replay
    - Position tracking
    - Trade execution
    - Performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        trading_costs: float = 0.001,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            trading_costs: Cost per trade
        """
        self.initial_capital = initial_capital
        self.trading_costs = trading_costs
        self.portfolio = None

    def run(
        self,
        agent,
        data: pd.DataFrame,
        env_reset_kwargs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Run backtest.

        Args:
            agent: Trading agent
            data: OHLCV data
            env_reset_kwargs: Kwargs for environment reset

        Returns:
            Dictionary with backtest metrics
        """
        self.portfolio = Portfolio(
            self.initial_capital,
            self.trading_costs,
        )

        # Create environment for agent inference
        from ..environment import TradingEnv

        env = TradingEnv(data.values)

        obs, _ = env.reset()
        done = False

        while not done:
            # Agent makes decision
            action, _ = agent.predict(obs, deterministic=True)

            # Get current price
            current_price = data.iloc[env.current_step]["close"]

            # Execute action
            if action == 1:  # Buy
                self.portfolio.buy(current_price, 1.0, env.current_step)
            elif action == 2:  # Sell
                self.portfolio.sell(current_price, 1.0, env.current_step)

            # Update portfolio value
            self.portfolio.update_equity(current_price)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

        # Compute metrics
        metrics = self.compute_metrics()
        logger.info(f"Backtest complete. Metrics: {metrics}")

        return metrics

    def compute_metrics(self) -> Dict[str, float]:
        """Compute backtest metrics."""
        if self.portfolio is None:
            return {}

        equity = self.portfolio.get_equity_curve()
        returns = self.portfolio.get_returns()
        trades = self.portfolio.get_trades()

        total_return = (equity[-1] - equity[0]) / equity[0]
        annual_return = total_return * 252  # Approximate annualization

        # Sharpe ratio
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        # Win rate
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / max(len(trades), 1)

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(equity)

        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "num_trades": len(trades),
            "final_equity": float(equity[-1]),
        }

        return metrics

    @staticmethod
    def _compute_max_drawdown(equity: np.ndarray) -> float:
        """Compute maximum drawdown."""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return float(np.min(drawdown))

    def get_equity_curve(self) -> np.ndarray:
        """Get equity curve."""
        if self.portfolio is None:
            return np.array([])
        return self.portfolio.get_equity_curve()

    def get_trades(self) -> List[Trade]:
        """Get all trades."""
        if self.portfolio is None:
            return []
        return self.portfolio.get_trades()
