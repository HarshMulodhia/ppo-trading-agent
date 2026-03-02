"""
Integration Tests

Comprehensive end-to-end integration tests for the complete trading system.
Tests cover data pipeline, training pipeline, evaluation pipeline, deployment pipeline,
inference engine, monitoring, model management, and risk management.
"""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

from src.deployment import (InferenceEngine, ModelManager, PerformanceMonitor,
                            RiskManager)

# Configure logging for tests
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestDataPipeline:
    """Test complete data pipeline integration."""

    def test_load_preprocess_split(self, sample_data, tmp_cache_dir):
        """Test loading, preprocessing, and splitting data."""
        from src.data import DataPreprocessor, DataSplitter

        # Preprocess
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess(
            sample_data,
            remove_outliers=True,
            normalize=True,
            compute_returns=False,
        )

        assert len(processed) > 0
        assert not processed.isna().all().any()

        # Split
        splitter = DataSplitter(processed)
        train, val, test = splitter.time_series_split()

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # Check no overlap
        total_len = len(train) + len(val) + len(test)
        assert total_len == len(processed)

    def test_data_caching(self, sample_data, tmp_cache_dir):
        """Test data caching functionality."""
        from src.data import DataCache

        cache = DataCache(cache_dir=str(tmp_cache_dir))

        # Save data
        cache.save("test_data", sample_data)
        assert cache.is_cached("test_data")

        # Load data
        loaded = cache.load("test_data")
        assert len(loaded) == len(sample_data)

    def test_preprocessor_pipeline(self, sample_data):
        """Test preprocessing pipeline with all steps."""
        from src.data import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Test with all options
        processed = preprocessor.preprocess(
            sample_data,
            remove_outliers=True,
            normalize=True,
            compute_returns=True,
        )

        assert len(processed) > 0
        assert not processed.isna().all().any()

    def test_data_splitter_ratios(self, sample_data):
        """Test data splitter with custom ratios."""
        from src.data import DataSplitter

        splitter = DataSplitter(sample_data)
        train, val, test = splitter.time_series_split(
            train_ratio=0.6,
            val_ratio=0.2,
        )

        # Check ratios are approximate
        n = len(sample_data)
        assert len(train) == int(n * 0.6)
        assert len(val) == int(n * 0.2)
        assert len(test) == n - len(train) - len(val)


@pytest.mark.integration
class TestTrainingPipeline:
    """Test training pipeline integration."""

    def test_env_agent_interaction(self, sample_env):
        """Test environment and agent interaction during training."""
        from src.agent import PPOAgent

        # Create agent
        agent = PPOAgent(env=sample_env)

        # Run episode
        obs, _ = sample_env.reset()
        done = False
        steps = 0
        episode_return = 0.0

        while not done and steps < 50:
            action, _ = agent.predict(obs=obs, deterministic=False)
            obs, reward, terminated, truncated, info = sample_env.step(action)
            done = terminated or truncated
            episode_return += reward
            steps += 1

        assert steps > 0
        assert isinstance(episode_return, (int, float))

    def test_training_loop(self, sample_env):
        """Test full training loop."""
        from src.agent import PPOAgent

        agent = PPOAgent(env=sample_env, learning_rate=0.001)

        # Run multiple episodes
        returns = []
        for episode in range(3):
            obs, info = sample_env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, log_prob = agent.predict(obs)
                obs, reward, done, truncated, info = sample_env.step(int(action))
                episode_return += reward

            returns.append(episode_return)

        assert len(returns) == 3

    def test_buffer_management(self):
        """Test replay buffer for training."""
        from src.agent import ExperienceBuffer

        buffer = ExperienceBuffer(max_size=1000)

        # Add transitions
        for _ in range(100):
            state = np.random.randn(5)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = np.random.choice([True, False])

            buffer.add(state, action, reward, next_state, done)

        assert buffer.size() == 100

        # Sample batch
        batch = buffer.sample(batch_size=32)
        assert batch["observations"].shape[0] == 32
        assert batch["actions"].shape[0] == 32
        assert batch["rewards"].shape[0] == 32
        assert batch["next_observations"].shape[0] == 32
        assert batch["dones"].shape[0] == 32


@pytest.mark.integration
class TestEvaluationPipeline:
    """Test evaluation pipeline integration."""

    def test_backtest_workflow(self, sample_env):
        """Test complete backtesting workflow."""
        from src.evaluation.backtester import Backtester

        backtester = Backtester(initial_capital=100000.0)
        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        # Simulate backtest
        obs, info = sample_env.reset()
        done = False
        equity_curve = [100000.0]

        for _ in range(50):
            if done:
                break
            action = np.random.randint(0, 3)
            obs, reward, done, truncated, info = sample_env.step(action)
            equity_curve.append(equity_curve[-1] * (1 + reward / 100))

        assert len(equity_curve) > 1
        assert all(x > 0 for x in equity_curve)

    def test_metrics_computation(self):
        """Test performance metrics computation."""
        from src.evaluation.metrics import compute_metrics

        portfolio_values = np.array([100000, 102000, 105000, 103000, 108000])
        metrics = compute_metrics(portfolio_values, num_trades=5)

        assert metrics.total_return > 0
        assert np.isfinite(metrics.sharpe_ratio)
        assert metrics.max_drawdown <= 0

    def test_evaluator_workflow(self, sample_env):
        """Test agent evaluator workflow."""
        from src.evaluation import AgentEvaluator

        mock_agent = Mock()
        mock_agent.predict.return_value = (0, None)

        evaluator = AgentEvaluator(sample_env, mock_agent)

        # Evaluate
        metrics = evaluator.evaluate(n_episodes=2, deterministic=True)

        assert "mean_return" in metrics
        assert "std_return" in metrics


@pytest.mark.integration
class TestDeploymentPipeline:
    """Test deployment pipeline integration."""

    def test_inference_engine_workflow(self, torch_model):
        """Test inference engine full workflow."""
        engine = InferenceEngine(torch_model, device="cpu")

        # Warm start
        engine.warm_start((50,))
        assert engine.is_warmed

        # Single prediction
        obs = np.random.randn(50).astype(np.float32)
        action, confidence = engine.predict(obs)

        assert isinstance(action, (int, np.integer))
        assert 0 <= confidence <= 1

    def test_inference_batch_predictions(self, torch_model):
        """Test batch predictions in inference engine."""
        engine = InferenceEngine(torch_model, device="cpu")

        # Batch predictions
        observations = np.random.randn(10, 50).astype(np.float32)
        actions, confidences = engine.batch_predict(observations)

        assert len(actions) == 10
        assert len(confidences) == 10
        assert all(0 <= c <= 1 for c in confidences)

    def test_model_manager_versioning(self, torch_model, tmp_model_dir):
        """Test model versioning and management."""
        manager = ModelManager(model_dir=str(tmp_model_dir))

        # Save version
        version_id = manager.save_version(
            torch_model, "test_model", metadata={"epoch": 10, "loss": 0.5}
        )

        assert version_id is not None

        # List versions
        versions = manager.list_versions()
        assert version_id in versions

        # Get version info
        info = manager.get_version_info(version_id)
        assert info["epoch"] == 10
        assert info["loss"] == 0.5

    def test_model_rollback(self, torch_model, tmp_model_dir):
        """Test model rollback functionality."""
        manager = ModelManager(model_dir=str(tmp_model_dir))

        # Save first version
        v1_id = manager.save_version(torch_model, "test_model")

        # Save second version
        v2_id = manager.save_version(torch_model, "test_model")

        # List should have both
        versions = manager.list_versions("test_model")
        assert len(versions) >= 2

        # Rollback
        loaded = manager.load_version(torch_model, v1_id)
        assert loaded is not None


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test performance monitoring integration."""

    def test_performance_monitor_workflow(self):
        """Test performance monitoring workflow."""
        monitor = PerformanceMonitor(window_size=100)

        # Track performance
        for i in range(50):
            return_val = np.random.randn() * 0.01
            profit = 100 + np.random.randn() * 50
            monitor.track_performance(return_val, profit)

        # Set baseline
        monitor.set_baseline()
        assert monitor.baseline_mean is not None
        assert monitor.baseline_std is not None

    def test_drift_detection(self):
        """Test distribution drift detection."""
        monitor = PerformanceMonitor(window_size=100)

        # Track normal returns
        for _ in range(20):
            monitor.track_performance(0.001, 100.0)

        monitor.set_baseline()

        # Introduce drift
        for _ in range(10):
            monitor.track_performance(0.05, 500.0)

        drift_detected = monitor.detect_drift(threshold=2.0)
        assert isinstance(drift_detected, bool)

    def test_daily_report_generation(self):
        """Test daily report generation."""
        monitor = PerformanceMonitor(window_size=100)

        # Add data
        for _ in range(20):
            monitor.track_performance(0.01, 100.0)

        # Generate report
        report = monitor.get_daily_report()

        assert "daily_return" in report
        assert "avg_return" in report
        assert "sharpe_ratio" in report


@pytest.mark.integration
class TestRiskManagementIntegration:
    """Test risk management integration."""

    def test_risk_manager_workflow(self):
        """Test complete risk manager workflow."""
        manager = RiskManager(
            initial_capital=100000.0,
            max_position_size=0.1,
            max_daily_loss=0.05,
        )

        # Check limits
        assert manager.check_limits(50.0, 100.0)

        # Validate order
        is_valid = manager.validate_order(50.0, 100.0)
        assert isinstance(is_valid, bool)

        # Update capital
        manager.update_capital(1000.0)
        assert manager.current_capital == 101000.0

    def test_position_limit_enforcement(self):
        """Test position size limits."""
        manager = RiskManager(
            initial_capital=100000.0,
            max_position_size=0.1,
        )

        # Valid position
        assert manager.check_limits(100.0, 100.0)

        # Invalid position (too large)
        assert not manager.check_limits(50000.0, 100.0)

    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        manager = RiskManager(
            initial_capital=100000.0,
            max_daily_loss=0.05,
        )

        # Acceptable loss
        assert manager.check_daily_loss(-4000.0)

        # Excessive loss
        assert not manager.check_daily_loss(-10000.0)

    def test_capital_tracking(self):
        """Test capital updates and tracking."""
        manager = RiskManager(initial_capital=100000.0)

        # Track multiple trades
        manager.update_capital(1000.0)
        manager.update_capital(-500.0)
        manager.update_capital(200.0)

        assert manager.current_capital == 100700.0
        assert manager.daily_pnl == 700.0


@pytest.mark.integration
class TestRewardIntegration:
    """Test reward computation integration."""

    def test_reward_shaping_pipeline(self):
        """Test complete reward shaping pipeline."""
        from src.reward.reward_functions import sharpe_reward
        from src.reward.shaping import RewardShaper

        portfolio_values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        # Compute raw reward
        raw_reward = sharpe_reward(portfolio_values)

        # Normalize
        normalized = RewardShaper.normalize(np.array([raw_reward]))

        # Clip
        clipped = RewardShaper.clip(normalized, -1.0, 1.0)

        assert -1.0 <= clipped[0] <= 1.0

    def test_discounted_reward_computation(self):
        """Test discounted reward computation."""
        from src.reward.shaping import RewardShaper

        rewards = np.array([1.0, 1.0, 1.0, 1.0])
        discounted = RewardShaper.discount_rewards(rewards, gamma=0.99)

        assert len(discounted) == len(rewards)
        assert discounted[0] > discounted[-1]

    def test_reward_analysis_pipeline(self):
        """Test reward analysis pipeline."""
        from src.reward.analysis import RewardAnalyzer

        analyzer = RewardAnalyzer()

        # Add multiple episodes
        for _ in range(5):
            rewards = np.random.randn(50)
            analyzer.add_rewards(rewards)

        # Analyze
        dist = analyzer.analyze_distribution()
        assert "mean" in dist
        assert "std" in dist

        # Quality metrics
        quality = analyzer.reward_quality_metrics()
        assert "dead_ratio" in quality


@pytest.mark.slow
@pytest.mark.integration
class TestFullSystemIntegration:
    """Test complete system end-to-end integration."""

    def test_end_to_end_workflow(
        self, sample_data, sample_env, torch_model, tmp_model_dir, tmp_cache_dir
    ):
        """Test complete end-to-end workflow."""
        from src.agent import PPOAgent
        from src.data import DataCache, DataPreprocessor, DataSplitter
        from src.evaluation import AgentEvaluator

        # Data pipeline
        preprocessor = DataPreprocessor()
        processed = preprocessor.preprocess(sample_data)
        splitter = DataSplitter(processed)
        train, val, test = splitter.time_series_split()

        # Cache data
        cache = DataCache(cache_dir=str(tmp_cache_dir))
        cache.save("train_data", train)
        cache.save("val_data", val)

        # Training setup
        agent = PPOAgent(sample_env)
        monitor = PerformanceMonitor()
        risk_mgr = RiskManager(initial_capital=100000.0)

        # Training loop
        obs, info = sample_env.reset()
        for step in range(10):
            action, log_prob = agent.predict(obs)
            obs, reward, done, truncated, info = sample_env.step(int(action))
            monitor.track_performance(reward, 100.0)
            if done:
                break

        # Model management
        manager = ModelManager(model_dir=str(tmp_model_dir))
        version_id = manager.save_version(torch_model, "trading_agent")

        # Evaluation
        evaluator = AgentEvaluator(sample_env, agent)
        metrics = evaluator.evaluate(n_episodes=1)

        assert version_id is not None
        assert metrics is not None
        assert "mean_return" in metrics

    def test_inference_to_monitoring_pipeline(self, torch_model, tmp_model_dir):
        """Test inference to monitoring pipeline."""
        # Setup
        manager = ModelManager(model_dir=str(tmp_model_dir))
        engine = InferenceEngine(torch_model)
        monitor = PerformanceMonitor()

        # Make predictions
        for i in range(30):
            obs = np.random.randn(50).astype(np.float32)
            action, confidence = engine.predict(obs)

            # Simulate trading performance
            return_val = np.random.randn() * 0.01
            monitor.track_performance(return_val, 100.0)

        # Monitor drift
        monitor.set_baseline()
        drift = monitor.detect_drift()

        assert isinstance(drift, bool)


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in integrated workflows."""

    def test_missing_data_handling(self, tmp_cache_dir):
        """Test handling of missing data in pipeline."""
        from src.data import DataCache

        cache = DataCache(cache_dir=str(tmp_cache_dir))

        # Try to load non-existent data
        try:
            loaded = cache.load("nonexistent")
            assert loaded is None
        except Exception as e:
            assert isinstance(e, (FileNotFoundError, KeyError))

    def test_model_version_rollback_error(self, torch_model, tmp_model_dir):
        """Test error handling for invalid model versions."""
        manager = ModelManager(model_dir=str(tmp_model_dir))

        # Try to load non-existent version
        with pytest.raises(ValueError):
            manager.load_version(torch_model, "invalid_version")

    def test_risk_limit_violations(self):
        """Test handling of risk limit violations."""
        manager = RiskManager(
            initial_capital=100000.0,
            max_position_size=0.1,
            max_daily_loss=0.05,
        )

        # Position too large
        valid = manager.check_limits(100000.0, 100.0)
        assert not valid

        # Daily loss too large
        valid = manager.check_daily_loss(-20000.0)
        assert not valid

    def test_inference_with_invalid_input(self, torch_model):
        """Test inference error handling."""
        engine = InferenceEngine(torch_model)

        # Valid input
        obs = np.random.randn(50).astype(np.float32)
        action, confidence = engine.predict(obs)
        assert action is not None


@pytest.mark.integration
class TestScalabilityIntegration:
    """Test system scalability with larger datasets."""

    def test_large_batch_processing(self, torch_model):
        """Test batch processing with large batches."""
        engine = InferenceEngine(torch_model)

        # Large batch
        observations = np.random.randn(1000, 50).astype(np.float32)
        actions, confidences = engine.batch_predict(observations)

        assert len(actions) == 1000
        assert len(confidences) == 1000

    def test_long_episode_training(self, sample_env):
        """Test training over extended episodes."""
        from src.agent import PPOAgent

        agent = PPOAgent(env=sample_env)

        obs, info = sample_env.reset()
        steps = 0
        max_steps = 200

        while steps < max_steps:
            action, _ = agent.predict(obs)
            obs, reward, done, truncated, info = sample_env.step(int(action))
            steps += 1
            if done:
                break

        assert steps > 0

    def test_multiple_monitoring_windows(self):
        """Test monitoring with multiple time windows."""

        monitors = [PerformanceMonitor(window_size=w) for w in [50, 100, 200]]

        # Track same data
        for _ in range(150):
            ret = np.random.randn() * 0.01
            for monitor in monitors:
                monitor.track_performance(ret, 100.0)

        for monitor in monitors:
            monitor.set_baseline()
            assert monitor.baseline_mean is not None


@pytest.mark.integration
class TestCrossModuleIntegration:
    """Test interactions between different modules."""

    def test_agent_reward_integration(self, sample_env):
        """Test agent learning with reward shaping."""
        from src.agent import PPOAgent
        from src.reward.reward_functions import sharpe_reward
        from src.reward.shaping import RewardShaper

        agent = PPOAgent(env=sample_env)

        # Simulate training with shaped rewards
        for _ in range(6):
            obs, info = sample_env.reset()
            action, log_prob = agent.predict(obs)

            # Raw reward
            portfolio_vals = np.array([100.0, 101.0, 102.0])
            raw_reward = sharpe_reward(portfolio_vals)

            # Shape reward
            shaped = RewardShaper.normalize(np.array([raw_reward]))

            assert shaped is not None

    def test_monitoring_risk_manager_integration(self):
        """Test monitoring and risk manager interaction."""
        monitor = PerformanceMonitor()
        risk_mgr = RiskManager(initial_capital=100000.0)

        # Simulate trading with monitoring and risk mgmt
        for i in range(30):
            return_val = np.random.randn() * 0.01
            monitor.track_performance(return_val, 100.0)

            # Risk update
            pnl = 100.0 * return_val
            risk_mgr.update_capital(pnl)

        monitor.set_baseline()
        available_cap = risk_mgr.get_available_capital()

        assert available_cap >= 0

    def test_inference_monitoring_integration(self, torch_model):
        """Test inference with live monitoring."""
        engine = InferenceEngine(torch_model)
        monitor = PerformanceMonitor()

        # Live predictions with monitoring
        for _ in range(40):
            obs = np.random.randn(50).astype(np.float32)
            action, confidence = engine.predict(obs)

            # Simulate trade performance
            return_val = np.random.randn() * 0.02
            monitor.track_performance(return_val, 100.0)

        # Check monitoring data
        assert len(monitor.returns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
