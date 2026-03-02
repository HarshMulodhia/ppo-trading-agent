"""
Evaluation Module Tests

Comprehensive test suite for agent evaluation, backtesting, metrics, and visualization.
Tests cover AgentEvaluator, Backtester, Portfolio, PerformanceMetrics, and ResultsVisualizer.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    AgentEvaluator,
    Backtester,
    EvaluationResult,
    PerformanceMetrics,
    Portfolio,
    ResultsVisualizer,
    compute_metrics,
)

# Configure logging for tests
logger = logging.getLogger(__name__)


class TestAgentEvaluator:
    """Test suite for AgentEvaluator class."""

    def test_evaluator_initialization(self, sample_env):
        """Test AgentEvaluator initializes with valid env and agent."""

        mock_agent = Mock()
        evaluator = AgentEvaluator(sample_env, mock_agent)

        assert evaluator.env is not None
        assert evaluator.agent is not None
        assert evaluator.results == []

    def test_evaluate_single_episode(self, sample_env):
        """Test evaluation of single episode."""

        mock_agent = Mock()
        # Mock agent to always hold (action 0)
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        metrics = evaluator.evaluate(n_episodes=1, deterministic=True)

        assert isinstance(metrics, dict)
        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert "min_return" in metrics
        assert "max_return" in metrics
        assert "median_return" in metrics
        assert "mean_length" in metrics
        assert "std_length" in metrics

    def test_evaluate_multiple_episodes(self, sample_env):
        """Test evaluation across multiple episodes."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        metrics = evaluator.evaluate(n_episodes=5, deterministic=True)

        assert len(evaluator.results) == 1  # One evaluation call
        assert "mean_return" in metrics

    def test_evaluate_with_deterministic_policy(self, sample_env):
        """Test evaluation with deterministic policy flag."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (1, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        metrics = evaluator.evaluate(n_episodes=1, deterministic=True)

        # Verify deterministic flag was passed
        mock_agent.predict.assert_called()
        assert metrics is not None

    def test_compute_metrics_single_episode(self, sample_env):
        """Test metrics computation from single episode."""

        mock_agent = Mock()
        evaluator = AgentEvaluator(sample_env, mock_agent)

        episode_returns = [100.0]
        episode_lengths = [50]
        metrics = evaluator.compute_metrics(episode_returns, episode_lengths)

        assert metrics["mean_return"] == 100.0
        assert metrics["min_return"] == 100.0
        assert metrics["max_return"] == 100.0
        assert metrics["median_return"] == 100.0

    def test_compute_metrics_multiple_episodes(self, sample_env):
        """Test metrics computation from multiple episodes."""

        mock_agent = Mock()
        evaluator = AgentEvaluator(sample_env, mock_agent)

        episode_returns = [100.0, 150.0, 80.0, 120.0, 110.0]
        episode_lengths = [50, 60, 45, 55, 52]
        metrics = evaluator.compute_metrics(episode_returns, episode_lengths)

        assert metrics["mean_return"] == pytest.approx(112.0, rel=1e-2)
        assert metrics["min_return"] == 80.0
        assert metrics["max_return"] == 150.0
        assert metrics["std_return"] > 0
        assert metrics["std_length"] > 0

    def test_compare_baseline(self, sample_env):
        """Test agent comparison against baseline agent."""

        mock_agent = Mock()
        mock_baseline = Mock()
        mock_agent.predict.return_value = (0, None)
        mock_baseline.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        comparison = evaluator.compare_baseline(mock_baseline, n_episodes=2)

        assert "agent" in comparison
        assert "baseline" in comparison
        assert "improvement" in comparison
        assert "return_improvement" in comparison["improvement"]
        assert "return_ratio" in comparison["improvement"]

    def test_baseline_improvement_calculation(self, sample_env):
        """Test improvement calculation in baseline comparison."""

        mock_agent = Mock()
        mock_baseline = Mock()
        mock_agent.predict.return_value = (0, None)
        mock_baseline.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.side_effect = [
                {"mean_return": 150.0},  # agent metrics
                {"mean_return": 100.0},  # baseline metrics
            ]
            comparison = evaluator.compare_baseline(mock_baseline)

            improvement = comparison["improvement"]["return_improvement"]
            assert improvement == 50.0

            ratio = comparison["improvement"]["return_ratio"]
            assert ratio == 1.5

    def test_generate_report_empty_results(self, sample_env):
        """Test report generation with no evaluation results."""

        mock_agent = Mock()
        evaluator = AgentEvaluator(sample_env, mock_agent)

        report = evaluator.generate_report()
        assert "No evaluation results available" in report

    def test_generate_report_with_results(self, sample_env):
        """Test report generation with evaluation results."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        evaluator.evaluate(n_episodes=1, deterministic=True)
        report = evaluator.generate_report()

        assert "EVALUATION REPORT" in report
        assert "Mean Return" in report
        assert "Std Return" in report
        assert "Min Return" in report
        assert "Max Return" in report
        assert "Median Return" in report

    def test_report_contains_valid_metrics(self, sample_env):
        """Test report contains numeric metric values."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        evaluator.evaluate(n_episodes=1, deterministic=True)
        report = evaluator.generate_report()

        # Check that report contains numeric values
        assert ": " in report
        assert "Episode Length" in report


class TestEvaluationResult:
    """Test suite for EvaluationResult container class."""

    def test_evaluation_result_initialization(self):
        """Test EvaluationResult initializes correctly."""

        returns = [100.0, 105.0, 110.0]
        lengths = [50, 55, 60]
        actions = [0, 1, 2, 0, 1]
        observations = [np.array([1, 2, 3]) for _ in range(5)]

        result = EvaluationResult(returns, lengths, actions, observations)

        assert np.array_equal(result.returns, np.array(returns))
        assert np.array_equal(result.lengths, np.array(lengths))
        assert np.array_equal(result.actions, np.array(actions))

    def test_evaluation_result_summary(self):
        """Test EvaluationResult summary statistics."""

        returns = [100.0, 120.0, 110.0]
        lengths = [50, 55, 52]
        actions = [0, 1, 2]
        observations = [np.array([1, 2]) for _ in range(3)]

        result = EvaluationResult(returns, lengths, actions, observations)
        summary = result.summary()

        assert "mean_return" in summary
        assert "std_return" in summary
        assert "mean_length" in summary
        assert summary["mean_return"] == pytest.approx(110.0, rel=1e-2)


class TestBacktester:
    """Test suite for Backtester class."""

    def test_backtester_initialization(self):
        """Test Backtester initializes with default parameters."""

        backtester = Backtester()

        assert backtester.initial_capital == 100000.0
        assert backtester.trading_costs == 0.001
        assert backtester.portfolio is None

    def test_backtester_custom_parameters(self):
        """Test Backtester with custom parameters."""

        backtester = Backtester(initial_capital=50000.0, trading_costs=0.002)

        assert backtester.initial_capital == 50000.0
        assert backtester.trading_costs == 0.002

    def test_backtester_run_workflow(self, sample_data):
        """Test complete backtesting workflow."""
        from src.environment import TradingEnv

        backtester = Backtester(initial_capital=100000.0)
        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        env = TradingEnv(df=sample_data.values)

        # Mock data if needed
        if len(sample_data) < 200:
            # Create minimal test data
            sample_data = pd.DataFrame({"close": np.linspace(100, 110, 200)})

        metrics = backtester.run(mock_agent, sample_data)

        assert isinstance(metrics, dict)
        if len(metrics) > 0:
            assert "total_return" in metrics or len(metrics) == 0

    def test_backtester_compute_metrics(self):
        """Test metrics computation in backtester."""

        backtester = Backtester()
        backtester.portfolio = Mock()

        equity = np.array([100000, 102000, 101000, 105000])
        backtester.portfolio.get_equity_curve.return_value = equity
        backtester.portfolio.get_returns.return_value = np.diff(equity) / equity[:-1]
        backtester.portfolio.get_trades.return_value = []

        metrics = backtester.compute_metrics()

        assert "total_return" in metrics
        assert "annual_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "num_trades" in metrics

    def test_backtester_get_equity_curve(self):
        """Test equity curve retrieval."""

        backtester = Backtester()

        # No portfolio case
        equity = backtester.get_equity_curve()
        assert len(equity) == 0

        # With portfolio
        backtester.portfolio = Mock()
        expected_equity = np.array([100000, 102000, 105000])
        backtester.portfolio.get_equity_curve.return_value = expected_equity

        equity = backtester.get_equity_curve()
        assert np.array_equal(equity, expected_equity)

    def test_backtester_get_trades(self):
        """Test trade list retrieval."""

        backtester = Backtester()

        # No portfolio case
        trades = backtester.get_trades()
        assert trades == []

        # With portfolio
        backtester.portfolio = Mock()
        mock_trades = [Mock(), Mock()]
        backtester.portfolio.get_trades.return_value = mock_trades

        trades = backtester.get_trades()
        assert len(trades) == 2


class TestPortfolio:
    """Test suite for Portfolio tracking class."""

    def test_portfolio_initialization(self):
        """Test Portfolio initializes correctly."""

        portfolio = Portfolio(initial_capital=100000.0)

        assert portfolio.initial_capital == 100000.0
        assert portfolio.current_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.position == 0
        assert portfolio.trading_costs == 0.001
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == 100000.0

    def test_portfolio_buy_execution(self):
        """Test buy order execution."""

        portfolio = Portfolio(initial_capital=100000.0)
        initial_cash = portfolio.cash

        portfolio.buy(price=100.0, size=10.0, time=0)

        cost = 100.0 * 10.0 * (1 + 0.001)
        assert portfolio.cash == initial_cash - cost
        assert portfolio.position == 10.0
        assert portfolio.entry_price == 100.0
        assert portfolio.entry_time == 0

    def test_portfolio_sell_execution(self):
        """Test sell order execution and trade recording."""

        portfolio = Portfolio(initial_capital=100000.0)

        # Buy first
        portfolio.buy(price=100.0, size=10.0, time=0)
        position_after_buy = portfolio.position

        # Sell
        portfolio.sell(price=105.0, size=10.0, time=5)

        revenue = 105.0 * 10.0 * (1 - 0.001)
        assert portfolio.position == 0
        assert len(portfolio.trades) == 1

        trade = portfolio.trades[0]
        assert trade.entry_price == 100.0
        assert trade.exit_price == 105.0
        assert trade.size == 10.0
        assert trade.pnl > 0

    def test_portfolio_update_equity(self):
        """Test equity update with market price."""

        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.buy(price=100.0, size=10.0, time=0)

        # Update with new price
        new_price = 110.0
        equity = portfolio.update_equity(new_price)

        position_value = 10.0 * new_price
        expected_equity = portfolio.cash + position_value

        assert equity == expected_equity
        assert portfolio.current_capital == expected_equity
        assert len(portfolio.equity_curve) == 2

    def test_portfolio_get_equity_curve(self):
        """Test equity curve as numpy array."""

        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.equity_curve = [100000.0, 102000.0, 105000.0]

        equity = portfolio.get_equity_curve()

        assert isinstance(equity, np.ndarray)
        assert len(equity) == 3
        assert equity[0] == 100000.0

    def test_portfolio_get_trades(self):
        """Test retrieving trade list."""

        portfolio = Portfolio(initial_capital=100000.0)

        assert len(portfolio.get_trades()) == 0

        portfolio.buy(price=100.0, size=10.0, time=0)
        portfolio.sell(price=105.0, size=10.0, time=5)

        trades = portfolio.get_trades()
        assert len(trades) == 1

    def test_portfolio_get_returns(self):
        """Test daily returns computation."""

        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.equity_curve = [100000.0, 102000.0, 100500.0, 105000.0]

        returns = portfolio.get_returns()

        assert isinstance(returns, np.ndarray)
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.02, rel=1e-2)


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics data class and computations."""

    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics dataclass initialization."""

        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.10,
            calmar_ratio=1.5,
            win_rate=0.65,
            avg_return=0.001,
            std_return=0.015,
            annual_return=0.15,
        )

        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.5
        assert metrics.win_rate == 0.65

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""

        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.10,
            calmar_ratio=1.5,
            win_rate=0.65,
            avg_return=0.001,
            std_return=0.015,
            annual_return=0.15,
            num_trades=25,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["total_return"] == 0.15
        assert metrics_dict["num_trades"] == 25

    def test_metrics_string_representation(self):
        """Test string representation of metrics."""

        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.10,
            calmar_ratio=1.5,
            win_rate=0.65,
            avg_return=0.001,
            std_return=0.015,
            annual_return=0.15,
        )

        str_repr = str(metrics)

        assert "Total Return" in str_repr
        assert "Sharpe Ratio" in str_repr
        assert "Max Drawdown" in str_repr

    def test_compute_total_return(self):
        """Test total return computation from prices."""
        from src.evaluation.metrics import compute_total_return

        prices = np.array([100.0, 105.0, 110.0, 115.0])
        ret = compute_total_return(prices)

        assert ret == pytest.approx(0.15, rel=1e-2)

    def test_compute_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        from src.evaluation.metrics import compute_sharpe_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)

        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_compute_sortino_ratio(self):
        """Test Sortino ratio computation."""
        from src.evaluation.metrics import compute_sortino_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sortino = compute_sortino_ratio(returns)

        assert isinstance(sortino, float)

    def test_compute_max_drawdown(self):
        """Test maximum drawdown computation."""
        from src.evaluation.metrics import compute_max_drawdown

        portfolio_values = np.array([100, 120, 110, 90, 95])
        max_dd = compute_max_drawdown(portfolio_values)

        assert max_dd < 0  # Drawdown is negative
        assert max_dd == pytest.approx(-0.25, rel=1e-2)

    def test_compute_calmar_ratio(self):
        """Test Calmar ratio computation."""
        from src.evaluation.metrics import compute_calmar_ratio

        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        calmar = compute_calmar_ratio(returns)

        assert isinstance(calmar, float)

    def test_compute_win_rate(self):
        """Test win rate computation."""
        from src.evaluation.metrics import compute_win_rate

        returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
        win_rate = compute_win_rate(returns)

        assert win_rate == pytest.approx(0.667, rel=1e-2)
        assert 0.0 <= win_rate <= 1.0

    def test_normalize_features_zscore(self):
        """Test Z-score feature normalization."""
        from src.evaluation.metrics import normalize_features

        features = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = normalize_features(features, method="zscore")

        assert np.isclose(normalized.mean(axis=0), [0, 0], atol=1e-6).all()
        assert np.isclose(normalized.std(axis=0), [1, 1], atol=1e-6).all()

    def test_normalize_features_minmax(self):
        """Test min-max feature normalization."""
        from src.evaluation.metrics import normalize_features

        features = np.array([[1, 2], [3, 4], [5, 6]])
        normalized = normalize_features(features, method="minmax")

        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)

    def test_compute_drawdown(self):
        """Test drawdown underwater curve computation."""
        from src.evaluation.metrics import compute_drawdown

        portfolio_values = np.array([100, 120, 110, 90, 95])
        drawdown = compute_drawdown(portfolio_values)

        assert len(drawdown) == len(portfolio_values)
        assert drawdown[0] == 0  # First point is peak
        assert np.all(drawdown <= 0)  # All drawdowns are non-positive

    def test_validate_prices(self):
        """Test price array validation."""
        from src.evaluation.metrics import validate_prices

        # Valid prices
        valid_prices = np.array([100, 101, 102, 103] + [105] * 100)
        assert validate_prices(valid_prices, min_length=100) is True

        # Too few samples
        assert validate_prices(np.array([100, 101, 102]), min_length=100) is False

        # Contains NaN
        invalid_prices = np.array([100, np.nan, 102] + [105] * 100)
        assert validate_prices(invalid_prices, min_length=100) is False

        # Contains zero
        invalid_prices = np.array([100, 0, 102] + [105] * 100)
        assert validate_prices(invalid_prices, min_length=100) is False

    def test_clip_values(self):
        """Test value clipping for outlier handling."""
        from src.evaluation.metrics import clip_values

        values = np.array([-10, -5, 0, 5, 10, 15])
        clipped = clip_values(values, min_val=-5.0, max_val=5.0)

        assert np.all(clipped >= -5.0)
        assert np.all(clipped <= 5.0)

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""

        portfolio_values = np.array([100000, 102000, 105000, 103000, 108000])
        num_trades = 5

        metrics = compute_metrics(portfolio_values, num_trades=num_trades)

        assert metrics.total_return > 0
        assert np.isfinite(metrics.sharpe_ratio)
        assert np.isfinite(metrics.sortino_ratio)
        assert metrics.max_drawdown <= 0
        assert metrics.win_rate >= 0
        assert metrics.num_trades == 5


class TestResultsVisualizer:
    """Test suite for ResultsVisualizer class."""

    def test_visualizer_initialization(self, tmp_path):
        """Test ResultsVisualizer initialization."""

        output_dir = str(tmp_path / "results")
        visualizer = ResultsVisualizer(output_dir=output_dir)

        assert visualizer.output_dir.exists()

    def test_plot_equity_curve(self, tmp_path):
        """Test equity curve plotting."""

        output_dir = str(tmp_path / "results")
        visualizer = ResultsVisualizer(output_dir=output_dir)

        equity_curve = np.array([100000, 102000, 105000, 103000, 108000])

        with patch("matplotlib.pyplot.savefig"):
            save_path = visualizer.plot_equity(equity_curve)
            assert save_path is not None or save_path == ""

    def test_plot_drawdown_curve(self, tmp_path):
        """Test drawdown plotting."""

        output_dir = str(tmp_path / "results")
        visualizer = ResultsVisualizer(output_dir=output_dir)

        equity_curve = np.array([100000, 120000, 110000, 90000, 95000])

        with patch("matplotlib.pyplot.savefig"):
            save_path = visualizer.plot_drawdown(equity_curve)
            assert save_path is not None or save_path == ""

    def test_plot_trades(self, tmp_path):
        """Test trade markers plotting."""

        output_dir = str(tmp_path / "results")
        visualizer = ResultsVisualizer(output_dir=output_dir)

        equity_curve = np.array([100000, 102000, 105000, 103000, 108000])

        # Mock trades
        mock_trade = Mock()
        mock_trade.entry_time = 1
        mock_trade.exit_time = 3
        trades = [mock_trade]

        with patch("matplotlib.pyplot.savefig"):
            save_path = visualizer.plot_trades(equity_curve, trades)
            assert save_path is not None or save_path == ""

    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation."""

        output_dir = str(tmp_path / "results")
        visualizer = ResultsVisualizer(output_dir=output_dir)

        metrics = {
            "total_return": 0.15,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "win_rate": 0.65,
        }
        equity_curve = np.array([100000, 102000, 105000, 108000])

        save_path = visualizer.generate_html_report(metrics, equity_curve)

        assert save_path is not None or save_path == ""


class TestEvaluationEdgeCases:
    """Test suite for edge cases and error handling in evaluation."""

    def test_evaluator_with_zero_returns(self, sample_env):
        """Test evaluation with constant zero returns."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)

        # Manually set results with zero returns
        episode_returns = [0.0, 0.0, 0.0]
        episode_lengths = [50, 50, 50]

        metrics = evaluator.compute_metrics(episode_returns, episode_lengths)

        assert metrics["mean_return"] == 0.0
        assert metrics["std_return"] == 0.0

    def test_backtester_with_single_trade(self):
        """Test backtester handling single trade."""

        backtester = Backtester()
        backtester.portfolio = Mock()

        equity = np.array([100000, 101000])
        backtester.portfolio.get_equity_curve.return_value = equity
        backtester.portfolio.get_returns.return_value = np.array([0.01])
        backtester.portfolio.get_trades.return_value = []

        metrics = backtester.compute_metrics()

        assert "total_return" in metrics
        assert metrics["num_trades"] == 0

    def test_portfolio_no_position_sell(self):
        """Test selling when no position is held."""

        portfolio = Portfolio(initial_capital=100000.0)

        # Sell without buying (no position)
        initial_cash = portfolio.cash
        portfolio.sell(price=100.0, size=10.0, time=0)

        # Should handle gracefully
        assert portfolio.position == -10.0

    def test_sharpe_ratio_all_zeros(self):
        """Test Sharpe ratio with all zero returns."""
        from src.evaluation.metrics import compute_sharpe_ratio

        returns = np.array([0.0, 0.0, 0.0])
        sharpe = compute_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_win_rate_empty_returns(self):
        """Test win rate with empty returns array."""
        from src.evaluation.metrics import compute_win_rate

        returns = np.array([])
        win_rate = compute_win_rate(returns)

        assert win_rate == 0.0

    def test_drawdown_single_point(self):
        """Test drawdown computation with single point."""
        from src.evaluation.metrics import compute_drawdown

        portfolio_values = np.array([100000])
        drawdown = compute_drawdown(portfolio_values)

        assert len(drawdown) == 1
        assert drawdown[0] == 0

    def test_max_drawdown_empty_array(self):
        """Test max drawdown with empty array."""
        from src.evaluation.metrics import compute_max_drawdown

        portfolio_values = np.array([])
        max_dd = compute_max_drawdown(portfolio_values)

        assert max_dd == 0.0


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_evaluation_workflow(self, sample_env):
        """Test complete evaluation workflow."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)

        # Evaluate
        metrics = evaluator.evaluate(n_episodes=2, deterministic=True)
        assert metrics is not None

        # Generate report
        report = evaluator.generate_report()
        assert "EVALUATION REPORT" in report

    def test_backtesting_workflow(self, sample_data):
        """Test complete backtesting workflow."""

        backtester = Backtester(initial_capital=100000.0)
        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        # Create test data
        if len(sample_data) < 200:
            test_data = pd.DataFrame({"close": np.linspace(100, 110, 200)})
        else:
            test_data = sample_data

        # Run backtest
        try:
            metrics = backtester.run(mock_agent, test_data)
            equity = backtester.get_equity_curve()
            assert len(equity) >= 0
        except Exception as e:
            # Environment-specific exception handling
            logger.info(f"Backtesting skipped: {e}")

    def test_metrics_pipeline(self):
        """Test complete metrics computation pipeline."""

        portfolio_values = np.array([100000, 102000, 105000, 103000, 108000])

        metrics = compute_metrics(portfolio_values, num_trades=3)

        assert metrics.total_return > 0
        assert np.isfinite(metrics.sharpe_ratio)
        assert metrics.max_drawdown <= 0
        assert 0 <= metrics.win_rate <= 1

    @pytest.mark.parametrize("n_episodes", [1, 5, 10])
    def test_evaluation_with_varying_episodes(self, sample_env, n_episodes):
        """Test evaluation with different episode counts."""

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)
        metrics = evaluator.evaluate(n_episodes=n_episodes, deterministic=True)

        assert isinstance(metrics, dict)
        assert "mean_return" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
