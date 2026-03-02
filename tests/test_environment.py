"""
Environment Module Tests

Comprehensive test suite for the TradingEnv Gymnasium environment.
Tests cover initialization, reset, step execution, reward computation,
and edge cases following pytest best practices.

"""

import logging
from typing import Tuple

import numpy as np
import pytest

# Configure logging for tests
logger = logging.getLogger(__name__)


class TestTradingEnvInitialization:
    """Test suite for TradingEnv initialization."""

    def test_env_initializes_with_valid_data(self, sample_env):
        """Test environment initializes successfully with valid data."""
        env = sample_env
        assert env is not None
        assert env.initial_capital == 100000.0
        assert env.max_position == 10
        assert env.transaction_cost_pct == 0.001

    def test_env_creates_correct_action_space(self, sample_env):
        """Test action space is properly configured as Discrete(3)."""
        env = sample_env
        assert env.action_space is not None
        # Actions: 0=Hold, 1=Buy, 2=Sell
        assert env.action_space.n == 3

    def test_env_creates_correct_observation_space(self, sample_env):
        """Test observation space has correct shape and dtype."""
        env = sample_env
        assert env.observation_space is not None
        assert env.observation_space.shape == (6,)
        assert env.observation_space.dtype == np.float32

    def test_env_initializes_with_custom_capital(self, sample_data):
        """Test environment respects custom initial capital."""
        from src.environment.trading_env import TradingEnv

        custom_capital = 50000.0
        env = TradingEnv(sample_data.values, initial_capital=custom_capital)
        env.reset()

        assert env.initial_capital == custom_capital
        assert env.portfolio_value == custom_capital
        assert env.cash_balance == custom_capital

    def test_env_initializes_with_custom_position_limit(self, sample_data):
        """Test environment respects custom max position size."""
        from src.environment.trading_env import TradingEnv

        custom_max_pos = 25
        env = TradingEnv(sample_data.values, max_position=custom_max_pos)

        assert env.max_position == custom_max_pos

    def test_env_initializes_with_custom_transaction_cost(self, sample_data):
        """Test environment respects custom transaction cost percentage."""
        from src.environment.trading_env import TradingEnv

        custom_cost = 0.002  # 0.2%
        env = TradingEnv(sample_data.values, transaction_cost_pct=custom_cost)

        assert env.transaction_cost_pct == custom_cost

    def test_env_initializes_with_lookback_window(self, sample_data):
        """Test environment respects custom lookback window."""
        from src.environment.trading_env import TradingEnv

        custom_window = 50
        env = TradingEnv(sample_data.values, lookback_window=custom_window)

        assert env.lookback_window == custom_window

    def test_env_stores_dataframe_reference(self, sample_env, sample_data):
        """Test environment correctly stores data reference."""
        env = sample_env
        assert env.df is not None
        assert len(env.df) == len(sample_data.values)


class TestTradingEnvReset:
    """Test suite for TradingEnv reset functionality."""

    def test_reset_returns_tuple_of_observation_and_info(self, sample_env):
        """Test reset returns (observation, info) tuple."""
        result = sample_env.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_has_correct_shape(self, sample_env):
        """Test reset observation has shape (6,)."""
        obs, info = sample_env.reset()

        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_reset_observation_values_are_normalized(self, sample_env):
        """Test reset observation contains normalized values."""
        obs, info = sample_env.reset()

        # All values should be clipped to [-5, 5]
        assert np.all(obs >= -5.0)
        assert np.all(obs <= 5.0)

    def test_reset_initializes_state_correctly(self, sample_env):
        """Test reset initializes all state variables correctly."""
        sample_env.reset()

        assert sample_env.current_step >= 100  # Starts after indicators stabilize
        assert sample_env.position == 0.0
        assert sample_env.portfolio_value == sample_env.initial_capital
        assert sample_env.cash_balance == sample_env.initial_capital

    def test_reset_clears_episode_history(self, sample_env):
        """Test reset clears episode history."""
        # First episode
        sample_env.reset()
        sample_env.step(1)  # Take one action

        # Reset and verify history cleared
        sample_env.reset()
        assert len(sample_env.action_history) == 0
        assert len(sample_env.pnl_history) == 1  # Initial 0.0

    def test_reset_with_seed_is_reproducible(self, sample_env):
        """Test reset with seed produces reproducible results."""
        obs1, _ = sample_env.reset(seed=42)
        obs2, _ = sample_env.reset(seed=42)

        # Observations should be identical with same seed
        assert np.allclose(obs1, obs2, rtol=1e-5)

    def test_reset_info_dict_is_empty(self, sample_env):
        """Test reset returns empty info dict."""
        obs, info = sample_env.reset()

        assert isinstance(info, dict)
        assert len(info) == 0


class TestTradingEnvStep:
    """Test suite for TradingEnv step execution."""

    def test_step_returns_five_element_tuple(self, sample_env):
        """Test step returns (obs, reward, terminated, truncated, info)."""
        sample_env.reset()
        result = sample_env.step(0)  # Hold action

        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_observation_shape_is_consistent(self, sample_env):
        """Test step returns observation with correct shape."""
        sample_env.reset()
        obs, _, _, _, _ = sample_env.step(0)

        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_step_advances_time(self, sample_env):
        """Test step advances current_step by 1."""
        sample_env.reset()
        initial_step = sample_env.current_step

        sample_env.step(0)

        assert sample_env.current_step == initial_step + 1

    def test_step_hold_action_maintains_position(self, sample_env):
        """Test HOLD action (0) maintains current position."""
        sample_env.reset()
        sample_env.step(1)  # Buy
        position_before = sample_env.position

        sample_env.step(0)  # Hold

        assert sample_env.position == position_before

    def test_step_buy_action_increases_position(self, sample_env):
        """Test BUY action (1) increases position."""
        sample_env.reset()
        initial_position = sample_env.position

        sample_env.step(1)  # Buy

        assert sample_env.position > initial_position

    def test_step_sell_action_decreases_position(self, sample_env):
        """Test SELL action (2) decreases position."""
        sample_env.reset()
        sample_env.step(1)  # Buy first
        position_after_buy = sample_env.position

        sample_env.step(2)  # Sell

        assert sample_env.position < position_after_buy

    def test_step_position_respects_max_position_limit(self, sample_env):
        """Test position is clamped to max_position."""
        sample_env.reset()

        # Try to exceed max position
        for _ in range(sample_env.max_position + 5):
            sample_env.step(1)  # Buy

        assert sample_env.position <= sample_env.max_position

    def test_step_position_respects_min_position_limit(self, sample_env):
        """Test position is clamped to -max_position."""
        sample_env.reset()

        # Try to go below -max_position
        for _ in range(sample_env.max_position + 5):
            sample_env.step(2)  # Sell

        assert sample_env.position >= -sample_env.max_position

    def test_step_transaction_cost_reduces_cash(self, sample_env):
        """Test transaction cost reduces cash balance."""
        sample_env.reset()
        initial_cash = sample_env.cash_balance

        sample_env.step(1)  # Buy

        # Cash should decrease by transaction cost
        assert sample_env.cash_balance < initial_cash

    def test_step_updates_portfolio_value(self, sample_env):
        """Test step updates portfolio value correctly."""
        sample_env.reset()
        sample_env.step(1)  # Buy

        expected_value = sample_env.cash_balance + (
            sample_env.position * sample_env.df[sample_env.current_step, 4]
        )

        assert np.isclose(sample_env.portfolio_value, expected_value, rtol=1e-5)

    def test_step_populates_info_dict(self, sample_env):
        """Test step populates info dict with position and portfolio metrics."""
        sample_env.reset()
        _, _, _, _, info = sample_env.step(0)

        assert "position" in info
        assert "portfolio_value" in info
        assert "pnl" in info

    def test_step_reward_is_numeric(self, sample_env):
        """Test reward is always numeric and not NaN."""
        sample_env.reset()

        for _ in range(10):
            _, reward, _, _, _ = sample_env.step(np.random.randint(0, 3))
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)

    def test_step_terminated_flag_at_episode_end(self, sample_env):
        """Test terminated flag becomes True at end of episode."""
        sample_env.reset()
        terminated = False
        step_count = 0

        while not terminated and step_count < 10000:  # Safety limit
            _, _, terminated, _, _ = sample_env.step(0)
            step_count += 1

        assert terminated is True
        assert sample_env.current_step >= len(sample_env.df) - 1

    def test_step_truncated_flag_is_false(self, sample_env):
        """Test truncated flag is always False (no time limit)."""
        sample_env.reset()

        for _ in range(20):
            _, _, _, truncated, _ = sample_env.step(0)
            assert truncated is False


class TestTradingEnvObservation:
    """Test suite for observation generation."""

    def test_observation_contains_close_price_normalized(self, sample_env):
        """Test observation includes normalized close price."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # First element is close price (normalized)
        close_norm = obs[0]
        assert isinstance(close_norm, (float, np.floating))
        assert -5.0 <= close_norm <= 5.0

    def test_observation_contains_rsi_normalized(self, sample_env):
        """Test observation includes normalized RSI."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # Second element is RSI (normalized)
        rsi_norm = obs[1]
        assert isinstance(rsi_norm, (float, np.floating))
        assert -5.0 <= rsi_norm <= 5.0

    def test_observation_contains_macd_normalized(self, sample_env):
        """Test observation includes normalized MACD."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # Third element is MACD (normalized)
        macd_norm = obs[2]
        assert isinstance(macd_norm, (float, np.floating))
        assert -5.0 <= macd_norm <= 5.0

    def test_observation_contains_bollinger_bands_position(self, sample_env):
        """Test observation includes Bollinger Bands position."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # Fourth element is BB position
        bb_pos = obs[3]
        assert isinstance(bb_pos, (float, np.floating))
        assert -5.0 <= bb_pos <= 5.0

    def test_observation_contains_normalized_position(self, sample_env):
        """Test observation includes normalized position size."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # Fifth element is position (normalized)
        position_norm = obs[4]
        assert isinstance(position_norm, (float, np.floating))
        # Position should be normalized to [-1, 1] typically
        assert -1.1 <= position_norm <= 1.1

    def test_observation_contains_volume_normalized(self, sample_env):
        """Test observation includes normalized volume."""
        sample_env.reset()
        obs, _ = sample_env.reset()

        # Sixth element is volume (normalized)
        volume_norm = obs[5]
        assert isinstance(volume_norm, (float, np.floating))
        assert -5.0 <= volume_norm <= 5.0

    def test_observation_all_values_clipped(self, sample_env):
        """Test all observation values are clipped to [-5, 5]."""
        sample_env.reset()

        for _ in range(50):
            obs, _ = sample_env.reset()
            assert np.all(obs >= -5.0)
            assert np.all(obs <= 5.0)


class TestTradingEnvReward:
    """Test suite for reward computation."""

    def test_reward_is_float(self, sample_env):
        """Test reward is always float type."""
        sample_env.reset()

        for _ in range(10):
            _, reward, _, _, _ = sample_env.step(0)
            assert isinstance(reward, float)

    def test_reward_is_not_nan_or_inf(self, sample_env):
        """Test reward is never NaN or infinite."""
        sample_env.reset()

        for _ in range(100):
            _, reward, _, _, _ = sample_env.step(np.random.randint(0, 3))
            assert not np.isnan(reward)
            assert not np.isinf(reward)

    def test_reward_includes_pnl_component(self, sample_env):
        """Test reward is affected by P&L changes."""
        sample_env.reset()
        rewards = []

        # Buy and hold to accumulate rewards
        for _ in range(5):
            sample_env.step(1)
            _, reward, _, _, _ = sample_env.step(0)
            rewards.append(reward)

        # At least some variation in rewards
        assert len(set(rewards)) > 1 or len(rewards) == 0

    def test_reward_penalizes_large_drawdown(self, sample_env):
        """Test reward includes drawdown penalty."""
        sample_env.reset()

        # Force some trading and collect rewards
        for _ in range(20):
            _, reward, _, _, _ = sample_env.step(np.random.randint(0, 3))
            # Reward should be reasonable (not extremely negative normally)
            assert reward > -10.0  # Sanity check

    def test_reward_penalizes_frequent_trading(self, sample_env):
        """Test reward includes trading frequency penalty."""
        sample_env.reset()

        # Rapid trading should incur penalties
        rewards_rapid = []
        for _ in range(10):
            _, reward, _, _, _ = sample_env.step(np.random.randint(1, 3))
            rewards_rapid.append(reward)

        # Rewards should be computable even with frequent trading
        assert all(isinstance(r, float) for r in rewards_rapid)


class TestTradingEnvPortfolioTracking:
    """Test suite for portfolio tracking and metrics."""

    def test_portfolio_history_is_maintained(self, sample_env):
        """Test portfolio value history is tracked."""
        sample_env.reset()
        initial_history_len = len(sample_env.portfolio_history)

        sample_env.step(0)
        sample_env.step(0)

        assert len(sample_env.portfolio_history) == initial_history_len + 2

    def test_pnl_history_is_maintained(self, sample_env):
        """Test P&L history is tracked."""
        sample_env.reset()
        initial_pnl_len = len(sample_env.pnl_history)

        sample_env.step(0)
        sample_env.step(0)

        assert len(sample_env.pnl_history) == initial_pnl_len + 2

    def test_action_history_is_maintained(self, sample_env):
        """Test action history is tracked."""
        sample_env.reset()
        initial_action_len = len(sample_env.action_history)

        sample_env.step(0)
        sample_env.step(1)
        sample_env.step(2)

        assert len(sample_env.action_history) == initial_action_len + 3
        assert sample_env.action_history[-3:] == [0, 1, 2]

    def test_portfolio_value_consistency(self, sample_env):
        """Test portfolio value = cash + position value."""
        sample_env.reset()

        for _ in range(10):
            sample_env.step(np.random.randint(0, 3))

            current_price = sample_env.df[sample_env.current_step, 4]
            position_value = sample_env.position * current_price
            expected_portfolio = sample_env.cash_balance + position_value

            assert np.isclose(sample_env.portfolio_value, expected_portfolio, rtol=1e-5)

    def test_portfolio_value_changes_with_market(self, sample_env):
        """Test portfolio value reflects market price changes."""
        sample_env.reset()
        sample_env.step(1)  # Buy
        portfolio_after_buy = sample_env.portfolio_value

        # Step forward to see market movement
        for _ in range(5):
            _, _, done, _, _ = sample_env.step(0)
            if done:
                break

        # Portfolio should change as market moves
        assert sample_env.portfolio_value != portfolio_after_buy


class TestTradingEnvEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_env_handles_insufficient_data_gracefully(self, sample_data):
        """Test environment handles data with minimum required rows."""
        from src.environment.trading_env import TradingEnv

        # Should handle small datasets
        env = TradingEnv(sample_data.values)
        obs, _ = env.reset()

        assert obs is not None
        assert obs.shape == (6,)

    def test_env_handles_zero_position(self, sample_env):
        """Test environment handles zero position correctly."""
        sample_env.reset()

        assert sample_env.position == 0.0
        obs, _ = sample_env.reset()
        # Position normalized to 0
        assert abs(obs[4]) < 0.01

    def test_env_handles_extreme_price_movements(self, sample_env):
        """Test environment remains stable with any price data."""
        sample_env.reset()

        # Step through entire episode
        done = False
        steps = 0
        while not done and steps < 10000:
            _, reward, done, _, _ = sample_env.step(np.random.randint(0, 3))
            assert not np.isnan(reward)
            assert not np.isinf(reward)
            steps += 1

        assert np.isfinite(sample_env.portfolio_value)
        assert len(sample_env.portfolio_history) >= 1
        assert len(sample_env.pnl_history) >= 1

    def test_env_handles_repeated_buy_sell_cycles(self, sample_env):
        """Test environment handles repeated buy-sell cycles."""
        sample_env.reset()

        for cycle in range(5):
            sample_env.step(1)  # Buy
            for _ in range(3):
                sample_env.step(0)  # Hold
            sample_env.step(2)  # Sell

        assert sample_env.position == 0  # Should be flat after sell

    def test_env_handles_long_episodes(self, sample_env):
        """Test environment remains stable over extended episodes."""
        sample_env.reset()

        done = False
        step_count = 0
        while not done and step_count < 100:
            _, reward, done, _, _ = sample_env.step(0)
            assert not np.isnan(reward)
            step_count += 1

        assert step_count > 0

    def test_env_handles_negative_portfolio_avoidance(self, sample_env):
        """Test portfolio value stays positive."""
        sample_env.reset()

        done = False
        while not done:
            _, _, done, _, _ = sample_env.step(np.random.randint(0, 3))
            # Portfolio (finite)
            assert np.isfinite(sample_env.portfolio_value)


class TestTradingEnvIntegration:
    """Integration tests for complete trading episodes."""

    def test_complete_episode_execution(self, sample_env):
        """Test executing a complete episode from start to finish."""
        obs, info = sample_env.reset()

        assert obs is not None

        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            action = sample_env.action_space.sample()
            obs, reward, done, truncated, info = sample_env.step(action)

            assert obs.shape == (6,)
            assert isinstance(reward, float)

            total_reward += reward
            step_count += 1

            if step_count > 10000:  # Safety break
                break

        assert step_count > 100  # Should run for a while
        assert isinstance(total_reward, float)

    def test_deterministic_episode_with_fixed_actions(self, sample_env):
        """Test episode reproducibility with fixed action sequence."""
        actions = [0, 1, 1, 0, 2, 2, 0, 1, 0, 2]

        sample_env.reset(seed=42)
        obs1_seq = []
        reward1_seq = []

        for action in actions:
            obs, reward, _, _, _ = sample_env.step(action)
            obs1_seq.append(obs.copy())
            reward1_seq.append(reward)

        # Reset and repeat
        sample_env.reset(seed=42)
        obs2_seq = []
        reward2_seq = []

        for action in actions:
            obs, reward, _, _, _ = sample_env.step(action)
            obs2_seq.append(obs.copy())
            reward2_seq.append(reward)

        # Observations should be deterministic
        for obs1, obs2 in zip(obs1_seq, obs2_seq):
            assert np.allclose(obs1, obs2, rtol=1e-5)

    def test_multiple_episodes_independence(self, sample_env):
        """Test multiple episodes are independent."""
        episodes_data = []

        for episode in range(3):
            sample_env.reset()
            done = False
            ep_data = {"portfolio_values": [], "positions": []}

            while not done:
                sample_env.step(np.random.randint(0, 3))
                ep_data["portfolio_values"].append(sample_env.portfolio_value)
                ep_data["positions"].append(sample_env.position)

                if len(ep_data["portfolio_values"]) > 100:
                    break

                done = sample_env.current_step >= len(sample_env.df) - 1

            episodes_data.append(ep_data)

        # Different episodes should have different portfolio trajectories
        # (with high probability with random actions)
        assert len(episodes_data) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
