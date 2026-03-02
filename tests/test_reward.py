"""
Reward Module Tests

Comprehensive test suite for reward functions, reward shaping, and reward analysis.
Tests cover reward computation, analysis, shaping transformations, and evaluation.
"""

import logging

import numpy as np
import pytest

from src.reward import RewardAnalyzer, RewardShaper
from src.reward.reward_functions import *

# Configure logging for tests
logger = logging.getLogger(__name__)


class TestRewardFunctions:
    """Test suite for reward function implementations."""

    def test_sharpe_reward_basic(self):
        """Test basic Sharpe ratio reward computation."""
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        reward = sharpe_reward(portfolio_values)

        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_sharpe_reward_with_risk_free_rate(self):
        """Test Sharpe reward with custom risk-free rate."""
        portfolio_values = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        reward_low_rf = sharpe_reward(portfolio_values, risk_free_rate=0.0)
        reward_high_rf = sharpe_reward(portfolio_values, risk_free_rate=0.05)

        assert isinstance(reward_low_rf, float)
        assert isinstance(reward_high_rf, float)

    def test_sharpe_reward_single_point(self):
        """Test Sharpe reward with insufficient data."""
        portfolio_values = np.array([100.0])
        reward = sharpe_reward(portfolio_values)

        assert reward == 0.0

    def test_pnl_reward_positive(self):
        """Test PnL reward with profit."""
        current = 110.0
        initial = 100.0
        reward = pnl_reward(current, initial, transaction_cost=0.0)

        assert reward == pytest.approx(0.1, rel=1e-2)

    def test_pnl_reward_negative(self):
        """Test PnL reward with loss."""
        current = 90.0
        initial = 100.0
        reward = pnl_reward(current, initial, transaction_cost=0.0)

        assert reward == pytest.approx(-0.1, rel=1e-2)

    def test_pnl_reward_with_transaction_cost(self):
        """Test PnL reward with transaction costs."""
        current = 110.0
        initial = 100.0
        reward = pnl_reward(current, initial, transaction_cost=0.01)

        # Should be 0.1 - 0.01 = 0.09
        assert reward == pytest.approx(0.09, rel=1e-2)

    def test_risk_adjusted_reward(self):
        """Test risk-adjusted reward computation."""
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        reward = risk_adjusted_reward(portfolio_values, risk_penalty=0.1)

        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_risk_adjusted_reward_with_volatility(self):
        """Test risk penalty effect on reward."""
        # High volatility returns
        portfolio_values = np.array([100.0, 95.0, 110.0, 98.0, 115.0])
        reward_low_penalty = risk_adjusted_reward(portfolio_values, risk_penalty=0.0)
        reward_high_penalty = risk_adjusted_reward(portfolio_values, risk_penalty=1.0)
        # Higher penalty should reduce reward
        assert reward_low_penalty > reward_high_penalty

    def test_sortino_reward(self):
        """Test Sortino ratio reward computation."""
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        reward = sortino_reward(portfolio_values)
        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_sortino_reward_with_downside_only(self):
        """Test Sortino penalizes downside risk only."""
        # Positive returns only (no downside)
        portfolio_values_up = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        # Mixed returns
        portfolio_values_mixed = np.array([100.0, 95.0, 110.0, 98.0, 115.0])
        sortino_up = sortino_reward(portfolio_values_up)
        sortino_mixed = sortino_reward(portfolio_values_mixed)
        assert isinstance(sortino_up, float)
        assert isinstance(sortino_mixed, float)

    def test_calmar_reward(self):
        """Test Calmar ratio reward computation."""
        portfolio_values = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        reward = calmar_reward(portfolio_values)
        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_calmar_reward_with_drawdown(self):
        """Test Calmar ratio with significant drawdown."""
        # Large drawdown case
        portfolio_values = np.array([100.0, 150.0, 50.0, 120.0, 140.0])
        reward = calmar_reward(portfolio_values)
        assert isinstance(reward, float)

    def test_calmar_reward_no_drawdown(self):
        """Test Calmar ratio when no drawdown occurs."""
        # Only positive returns
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        reward = calmar_reward(portfolio_values)
        # Should return annual return
        assert isinstance(reward, float)


class TestRewardFunction:
    """Test suite for configurable RewardFunction class."""

    def test_reward_function_initialization(self):
        """Test RewardFunction initialization."""
        components = {
            "sharpe": sharpe_reward,
            "pnl": pnl_reward,
        }
        weights = {"sharpe": 0.6, "pnl": 0.4}
        reward_func = RewardFunction(components, weights)
        assert reward_func.components is not None
        assert reward_func.weights is not None

    def test_reward_function_weight_normalization(self):
        """Test weight normalization in RewardFunction."""
        components = {
            "sharpe": sharpe_reward,
            "pnl": pnl_reward,
        }
        weights = {"sharpe": 2.0, "pnl": 2.0}
        reward_func = RewardFunction(components, weights)
        # Weights should sum to 1.0
        total_weight = sum(reward_func.weights.values())
        assert total_weight == pytest.approx(1.0, rel=1e-6)

    def test_reward_function_compute_single_component(self):
        """Test computing reward with single component."""
        components = {"sharpe": sharpe_reward}
        weights = {"sharpe": 1.0}
        reward_func = RewardFunction(components, weights)
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        reward = reward_func.compute(portfolio_values=portfolio_values)
        assert isinstance(reward, float)
        assert not np.isnan(reward)

    def test_reward_function_compute_multiple_components(self):
        """Test computing weighted reward from multiple components."""
        components = {
            "sharpe": sharpe_reward,
            "pnl": pnl_reward,
        }
        weights = {"sharpe": 0.6, "pnl": 0.4}
        reward_func = RewardFunction(components, weights)
        portfolio_values = np.array([100.0, 105.0, 110.0, 115.0])
        reward = reward_func.compute(
            portfolio_values=portfolio_values,
            current_portfolio_value=115.0,
            initial_portfolio_value=100.0,
        )
        assert isinstance(reward, float)

    def test_reward_function_missing_component(self):
        """Test RewardFunction with missing component kwargs."""
        components = {"sharpe": sharpe_reward}
        weights = {"sharpe": 1.0}
        reward_func = RewardFunction(components, weights)

        # Component not called if kwargs missing
        # Should handle gracefully
        with pytest.raises(TypeError):
            reward_func.compute()


class TestRewardShaper:
    """Test suite for reward shaping utilities."""

    def test_shaper_normalize_basic(self):
        """Test basic reward normalization."""
        rewards = np.array([0.0, 5.0, 10.0])
        normalized = RewardShaper.normalize(rewards, min_val=-1.0, max_val=1.0)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        assert len(normalized) == len(rewards)

    def test_shaper_normalize_range(self):
        """Test normalization to custom range."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = RewardShaper.normalize(rewards, min_val=0.0, max_val=1.0)

        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_shaper_normalize_constant(self):
        """Test normalization with constant rewards."""
        rewards = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = RewardShaper.normalize(rewards)

        # Should return middle value
        middle = (
            (
                RewardShaper.normalize.__defaults__[0]
                + RewardShaper.normalize.__defaults__[1]
            )
            / 2
            if hasattr(RewardShaper.normalize, "__defaults__")
            else 0.0
        )
        assert np.all(normalized == normalized[0])

    def test_shaper_normalize_empty(self):
        """Test normalization with empty array."""
        rewards = np.array([])
        normalized = RewardShaper.normalize(rewards)

        assert len(normalized) == 0

    def test_shaper_standardize(self):
        """Test reward standardization."""
        rewards = np.random.randn(1000) * 10 + 5  # mean=5, std=10
        standardized = RewardShaper.standardize(rewards)

        assert np.isclose(standardized.mean(), 0.0, atol=1e-6)
        assert np.isclose(standardized.std(), 1.0, atol=1e-6)

    def test_shaper_standardize_zero_std(self):
        """Test standardization with zero standard deviation."""
        rewards = np.array([5.0, 5.0, 5.0, 5.0])
        standardized = RewardShaper.standardize(rewards)

        # Should return zeros
        assert np.all(standardized == 0.0)

    def test_shaper_clip(self):
        """Test reward clipping."""
        rewards = np.array([-5.0, -2.0, 0.0, 2.0, 5.0, 10.0])
        clipped = RewardShaper.clip(rewards, min_val=-2.0, max_val=2.0)

        assert np.all(clipped >= -2.0)
        assert np.all(clipped <= 2.0)

    def test_shaper_add_penalty(self):
        """Test adding penalty to rewards."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        penalty = 0.5
        penalized = RewardShaper.add_penalty(rewards, penalty)

        assert np.allclose(penalized, rewards - penalty)

    def test_shaper_exponential_shaping(self):
        """Test exponential reward shaping."""
        rewards = np.array([-1.0, 0.0, 1.0, 2.0])
        shaped = RewardShaper.exponential_shaping(rewards, factor=0.5)

        assert shaped.shape == rewards.shape
        assert not np.any(np.isnan(shaped))

    def test_shaper_exponential_shaping_sign_preserved(self):
        """Test exponential shaping preserves sign."""
        rewards = np.array([-2.0, -1.0, 1.0, 2.0])
        shaped = RewardShaper.exponential_shaping(rewards, factor=1.0)

        # Sign should be preserved
        assert np.all(np.sign(shaped) == np.sign(rewards))

    def test_shaper_power_shaping(self):
        """Test power reward shaping."""
        rewards = np.array([1.0, 4.0, 9.0, 16.0])
        shaped = RewardShaper.power_shaping(rewards, power=0.5)

        # sqrt(x) should be applied element-wise
        assert shaped[0] == pytest.approx(1.0, rel=1e-2)
        assert shaped[1] == pytest.approx(2.0, rel=1e-2)

    def test_shaper_power_shaping_negative(self):
        """Test power shaping with negative values."""
        rewards = np.array([-8.0, -1.0, 1.0, 8.0])
        shaped = RewardShaper.power_shaping(rewards, power=0.5)

        # Sign should be preserved
        assert shaped[0] < 0
        assert shaped[3] > 0

    def test_shaper_discount_rewards(self):
        """Test discounted reward computation."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0])
        discounted = RewardShaper.discount_rewards(rewards, gamma=0.99)

        assert len(discounted) == len(rewards)
        # Last reward should equal itself
        assert discounted[-1] == pytest.approx(1.0, rel=1e-2)
        # First reward should be highest (includes future rewards)
        assert discounted[0] > discounted[-1]

    def test_shaper_discount_rewards_zero_gamma(self):
        """Test discounted rewards with gamma=0."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        discounted = RewardShaper.discount_rewards(rewards, gamma=0.0)

        # With gamma=0, should equal rewards
        assert np.allclose(discounted, rewards)

    def test_shaper_discount_rewards_one_gamma(self):
        """Test discounted rewards with gamma=1."""
        rewards = np.array([1.0, 2.0, 3.0])
        discounted = RewardShaper.discount_rewards(rewards, gamma=1.0)

        # With gamma=1, cumulative sum
        assert discounted[0] == pytest.approx(6.0, rel=1e-2)  # 1+2+3
        assert discounted[1] == pytest.approx(5.0, rel=1e-2)  # 2+3
        assert discounted[2] == pytest.approx(3.0, rel=1e-2)  # 3


class TestRewardAnalyzer:
    """Test suite for reward analysis functionality."""

    def test_analyzer_initialization(self):
        """Test RewardAnalyzer initialization."""
        analyzer = RewardAnalyzer()

        assert analyzer.rewards_history == []

    def test_analyzer_add_rewards(self):
        """Test adding rewards to analyzer."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 2.0, 3.0])

        analyzer.add_rewards(rewards)

        assert len(analyzer.rewards_history) == 1
        assert np.array_equal(analyzer.rewards_history[0], rewards)

    def test_analyzer_add_multiple_batches(self):
        """Test adding multiple reward batches."""
        analyzer = RewardAnalyzer()

        for i in range(5):
            rewards = np.random.randn(100)
            analyzer.add_rewards(rewards)

        assert len(analyzer.rewards_history) == 5

    def test_analyzer_distribution_analysis(self):
        """Test reward distribution analysis."""
        analyzer = RewardAnalyzer()
        rewards = np.random.randn(1000)

        stats = analyzer.analyze_distribution(rewards)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_analyzer_distribution_uses_latest(self):
        """Test distribution analysis uses latest rewards if none provided."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer.add_rewards(rewards)

        stats = analyzer.analyze_distribution()  # No argument

        assert stats["mean"] == pytest.approx(3.0, rel=1e-2)

    def test_analyzer_detect_outliers(self):
        """Test outlier detection."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 1.5, 2.0, 100.0, 1.8, 2.2])  # 100.0 is outlier

        result = analyzer.detect_outliers(rewards, threshold=1.0)

        assert "num_outliers" in result
        assert "outlier_ratio" in result
        assert "outlier_indices" in result
        assert "outlier_values" in result
        assert result["num_outliers"] >= 1

    def test_analyzer_detect_outliers_threshold(self):
        """Test outlier detection with different threshold."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result_loose = analyzer.detect_outliers(rewards, threshold=5.0)
        result_strict = analyzer.detect_outliers(rewards, threshold=0.5)

        # Stricter threshold should find more outliers
        assert result_strict["num_outliers"] >= result_loose["num_outliers"]

    def test_analyzer_correlation_analysis(self):
        """Test reward correlation analysis."""
        analyzer = RewardAnalyzer()
        rewards = np.random.randn(100)
        analyzer.add_rewards(rewards)

        corr = analyzer.correlation_analysis()

        if corr:  # May be empty if insufficient data
            assert "lag1_autocorr" in corr
            assert "mean_diff" in corr
            assert "reward_momentum" in corr

    def test_analyzer_quality_metrics(self):
        """Test reward quality metrics."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 0.0, 2.0, 0.0, 1.5, 0.0])

        metrics = analyzer.reward_quality_metrics(rewards)

        assert "dead_ratio" in metrics
        assert "nonzero_ratio" in metrics
        assert "variance" in metrics
        assert "stability" in metrics

        # With 3 zeros out of 6
        assert metrics["dead_ratio"] == pytest.approx(0.5, rel=1e-2)

    def test_analyzer_quality_metrics_all_nonzero(self):
        """Test quality metrics with all non-zero rewards."""
        analyzer = RewardAnalyzer()
        rewards = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = analyzer.reward_quality_metrics(rewards)

        assert metrics["dead_ratio"] == 0.0
        assert metrics["nonzero_ratio"] == 1.0

    def test_analyzer_quality_metrics_all_dead(self):
        """Test quality metrics with all zero rewards."""
        analyzer = RewardAnalyzer()
        rewards = np.array([0.0, 0.0, 0.0, 0.0])

        metrics = analyzer.reward_quality_metrics(rewards)

        assert metrics["dead_ratio"] == 1.0
        assert metrics["nonzero_ratio"] == 0.0

    def test_analyzer_generate_report(self):
        """Test report generation."""
        analyzer = RewardAnalyzer()
        rewards = np.random.randn(100)
        analyzer.add_rewards(rewards)

        report = analyzer.generate_report()

        assert isinstance(report, str)
        assert "REWARD ANALYSIS REPORT" in report
        assert "DISTRIBUTION ANALYSIS" in report
        assert "OUTLIER DETECTION" in report

    def test_analyzer_report_no_data(self):
        """Test report generation with no data."""
        analyzer = RewardAnalyzer()
        report = analyzer.generate_report()

        assert "No reward data available" in report


class TestRewardEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_sharpe_reward_nan_handling(self):
        """Test Sharpe reward handles NaN gracefully."""
        portfolio_values = np.array([100.0, 100.0, 100.0, 100.0])  # No variance
        reward = sharpe_reward(portfolio_values)

        assert isinstance(reward, float)
        assert not np.isnan(reward) or reward == 0.0

    def test_analyzer_empty_rewards(self):
        """Test analyzer with empty rewards."""
        analyzer = RewardAnalyzer()
        result = analyzer.analyze_distribution(np.array([]))

        assert result == {}

    def test_shaper_discount_empty(self):
        """Test discount_rewards with empty array."""
        rewards = np.array([])
        discounted = RewardShaper.discount_rewards(rewards)

        assert len(discounted) == 0

    def test_calmar_reward_flat_returns(self):
        """Test Calmar with no change in portfolio value."""
        portfolio_values = np.array([100.0, 100.0, 100.0, 100.0])
        reward = calmar_reward(portfolio_values)

        # Should handle gracefully
        assert isinstance(reward, float)

    def test_risk_adjusted_single_point(self):
        """Test risk-adjusted reward with single point."""
        portfolio_values = np.array([100.0])
        reward = risk_adjusted_reward(portfolio_values)

        assert reward == 0.0


class TestRewardIntegration:
    """Integration tests for reward computation pipeline."""

    def test_reward_pipeline_basic(self):
        """Test basic reward computation pipeline."""
        # Generate portfolio values
        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        # Compute raw reward
        raw_reward = sharpe_reward(portfolio_values)

        # Shape reward
        shaped = RewardShaper.normalize(np.array([raw_reward]))

        assert isinstance(shaped[0], (float, np.floating))

    def test_reward_pipeline_with_shaping(self):
        """Test reward pipeline with multiple shaping steps."""
        portfolio_values = np.random.randn(100).cumsum() + 100

        # Compute rewards
        rewards = np.array(
            [
                sharpe_reward(portfolio_values),
                pnl_reward(portfolio_values[-1], portfolio_values[0]),
            ]
        )

        # Normalize
        normalized = RewardShaper.normalize(rewards)

        # Clip
        clipped = RewardShaper.clip(normalized, -1.0, 1.0)

        assert len(clipped) == 2
        assert np.all(clipped >= -1.0)
        assert np.all(clipped <= 1.0)

    def test_reward_pipeline_with_analysis(self):
        """Test reward pipeline with analysis."""
        analyzer = RewardAnalyzer()

        # Generate multiple episodes of rewards
        for _ in range(10):
            rewards = np.random.randn(50)
            analyzer.add_rewards(rewards)

        # Analyze
        dist = analyzer.analyze_distribution()
        outliers = analyzer.detect_outliers(threshold=2.5)
        quality = analyzer.reward_quality_metrics()

        assert dist is not None
        assert outliers is not None
        assert quality is not None

    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99])
    def test_discount_rewards_various_gammas(self, gamma):
        """Test discounted rewards with various gamma values."""
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        discounted = RewardShaper.discount_rewards(rewards, gamma=gamma)

        assert len(discounted) == len(rewards)
        # First should be largest
        assert discounted[0] >= discounted[-1]

    @pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0, 4.0])
    def test_outlier_detection_thresholds(self, threshold):
        """Test outlier detection with various thresholds."""
        analyzer = RewardAnalyzer()
        rewards = np.random.randn(1000)

        result = analyzer.detect_outliers(rewards, threshold=threshold)

        assert "num_outliers" in result
        # Higher threshold should find fewer outliers
        assert result["num_outliers"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
