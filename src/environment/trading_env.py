"""
Trading Environment - Gymnasium Custom Environment for PPO Trading Agent

This module implements a custom Gymnasium environment for trading that:
- Provides OHLCV + technical indicator states
- Supports Buy/Sell/Hold discrete actions
- Computes Sharpe ratio-based rewards
- Handles transaction costs and position constraints
- Integrates with Stable-Baselines3
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Custom Gymnasium trading environment for single-asset trading with PPO.

    State Space:
        - OHLCV normalized features
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Portfolio state (current position, portfolio value)

    Action Space:
        - 0: HOLD - Maintain current position
        - 1: BUY - Increase position by 1 unit
        - 2: SELL - Decrease position by 1 unit

    Reward:
        - Composite signal: PnL + Sharpe ratio + risk penalty
        - Penalizes trading frequency and drawdown
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}
    portfolio_history: list[float]
    pnl_history: list[float]
    action_history: list[int]

    def __init__(
        self,
        df: np.ndarray,
        initial_capital: float = 100000.0,
        max_position: int = 10,
        transaction_cost_pct: float = 0.001,  # 0.1% per trade
        lookback_window: int = 20,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV and indicators (numpy array format)
            initial_capital: Starting portfolio value
            max_position: Maximum position size (long and short)
            transaction_cost_pct: Transaction cost as percentage
            lookback_window: Window for computing rolling metrics
            render_mode: Rendering mode ('human', None)
        """
        super().__init__()

        self.df = df
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window = lookback_window
        self.render_mode = render_mode

        # Define observation and action spaces
        # State: close_norm, rsi_norm, macd_norm, bb_position, position_norm, volume_norm
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32  # 6 state features
        )

        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        # Initialize episode tracking
        self.current_step = 0
        self.position = 0.0  # Current position size
        self.portfolio_value = initial_capital
        self.cash_balance = initial_capital
        self.portfolio_history = []
        self.pnl_history = []
        self.action_history = []

        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            observation: Initial state observation
            info: Additional information dict
        """
        super().reset(seed=seed)

        # Start trading after indicators stabilize (100 bars)
        self.current_step = max(100, self.lookback_window)
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.cash_balance = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.pnl_history = [0.0]
        self.action_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation (normalized).

        Returns:
            Normalized state vector [close, rsi, macd, bb_pos, position, volume]
        """
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        row = self.df[self.current_step]

        # Extract normalized features
        close_norm = row[6] if len(row) > 6 else 0.0  # close_norm
        rsi_norm = row[7] if len(row) > 7 else 0.0  # rsi_norm
        macd_norm = row[8] if len(row) > 8 else 0.0  # macd_norm
        bb_position = row[9] if len(row) > 9 else 0.0  # bb_position
        volume_norm = row[10] if len(row) > 10 else 0.0  # volume_norm

        # Normalize position
        position_norm = self.position / max(self.max_position, 1.0)

        obs = np.array(
            [
                float(close_norm),
                float(rsi_norm),
                float(macd_norm),
                float(bb_position),
                position_norm,
                float(volume_norm),
            ],
            dtype=np.float32,
        )

        # Clip to prevent extreme values
        obs = np.clip(obs, -5.0, 5.0)

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Agent action (0=Hold, 1=Buy, 2=Sell)

        Returns:
            observation: Next state
            reward: Reward signal
            terminated: Episode finished
            truncated: Time limit reached
            info: Additional information
        """
        current_price = float(self.df[self.current_step, 4])  # Close price

        # Execute action with position constraints
        if action == 1 and self.position < self.max_position:  # BUY
            self.position += 1.0
            transaction_cost = current_price * self.transaction_cost_pct
            self.cash_balance -= transaction_cost
        elif action == 2 and self.position > -self.max_position:  # SELL
            self.position -= 1.0
            transaction_cost = current_price * self.transaction_cost_pct
            self.cash_balance -= transaction_cost
        # action == 0: HOLD (no change)

        # Advance to next timestep
        self.current_step += 1
        self.action_history.append(action)

        # Get next price and compute P&L
        next_price = float(self.df[self.current_step, 4])
        price_change = next_price - current_price
        pnl = self.position * price_change

        # Update portfolio
        self.portfolio_value = self.cash_balance + (self.position * next_price)
        self.portfolio_history.append(self.portfolio_value)
        self.pnl_history.append(pnl)

        # Compute reward
        reward = self._compute_reward()

        # Check termination conditions
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        info = {
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "pnl": pnl,
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    def _compute_reward(self) -> float:
        """
        Compute composite reward signal.

        Components:
        1. PnL return (0.5 weight)
        2. Sharpe ratio approximation (0.3 weight)
        3. Risk penalty for drawdown (0.2 weight)

        Returns:
            Normalized reward signal
        """
        if len(self.portfolio_history) < 2:
            return 0.0

        # Component 1: Recent return
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) > 0:
            recent_return = returns[-1]
        else:
            recent_return = 0.0

        # Component 2: Rolling Sharpe ratio
        if len(returns) >= self.lookback_window:
            recent_returns = returns[-self.lookback_window :]
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns) + 1e-8
            sharpe_approx = (mean_ret / std_ret) * np.sqrt(252)  # Annualize
        else:
            sharpe_approx = 0.0

        # Component 3: Drawdown penalty
        max_dd = self._compute_max_drawdown(portfolio_values)
        dd_penalty = max(-0.2 * max(0, -max_dd - 0.15), -0.5)  # Penalize if DD > 15%

        # Component 4: Trading frequency penalty
        trade_penalty = 0.0
        if len(self.action_history) > 1 and self.action_history[-1] != 0:
            trade_penalty = -0.01

        # Composite reward
        reward = (
            0.5 * recent_return + 0.3 * sharpe_approx + 0.2 * dd_penalty + trade_penalty
        )

        return float(reward)

    @staticmethod
    def _compute_max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Compute maximum drawdown percentage.

        Args:
            portfolio_values: Array of portfolio values

        Returns:
            Maximum drawdown as negative percentage
        """
        if len(portfolio_values) < 2:
            return 0.0

        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max

        return float(np.min(drawdown))

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Render environment state (optional).

        Returns:
            String representation if human mode
        """
        if self.render_mode == "human":
            price = self.df[self.current_step, 4]
            print(
                f"Step {self.current_step}: Price={price:.2f}, "
                f"Position={self.position}, Portfolio={self.portfolio_value:.2f}"
            )

    def close(self):
        """Clean up resources."""
        pass

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics for the episode.

        Returns:
            Dictionary with performance metrics
        """
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) == 0:
            return {}

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        max_dd = self._compute_max_drawdown(portfolio_values)
        win_rate = np.mean(returns > 0)

        # Calmar ratio
        if max_dd != 0:
            calmar = (total_return / 252) / abs(max_dd)
        else:
            calmar = 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "calmar_ratio": calmar,
            "avg_trade_return": np.mean(np.abs(returns)),
            "num_trades": np.sum(np.array(self.action_history) != 0),
        }
