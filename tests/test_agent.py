"""
Agent Testing Module

Comprehensive unit tests for PPO Agent, networks, and buffer components.
Tests cover initialization, prediction, training, persistence, and edge cases.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

# Conditional imports - test will be skipped if dependencies are missing
try:
    from stable_baselines3 import PPO

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from src.agent import (ActorCriticNetwork, ExperienceBuffer, PPOAgent,
                       RolloutBuffer)

logger = logging.getLogger(__name__)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_model_dir():
    """Create temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_observations():
    """Generate sample observations for testing."""
    return np.random.randn(32, 50).astype(np.float32)


@pytest.fixture
def sample_actions():
    """Generate sample actions for testing."""
    return np.random.randint(0, 3, 32)


@pytest.fixture
def sample_rewards():
    """Generate sample rewards for testing."""
    return np.random.randn(32).astype(np.float32)


# ============================================================================
# ActorCriticNetwork Tests
# ============================================================================


class TestActorCriticNetwork:
    """Test suite for ActorCriticNetwork architecture."""

    def test_network_initialization(self):
        """Test network initializes with correct dimensions."""
        state_dim = 50
        action_dim = 3
        hidden_dims = [256, 256]

        network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        )

        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert network.use_layer_norm is True

    def test_forward_pass_output_shapes(self):
        """Test forward pass produces correct output shapes."""
        state_dim = 50
        action_dim = 3
        batch_size = 16

        network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        state = torch.randn(batch_size, state_dim)
        action_probs, value = network(state)

        assert action_probs.shape == (batch_size, action_dim)
        assert value.shape == (batch_size, 1)

        # Verify action probabilities sum to 1
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_get_action_sampling(self):
        """Test action sampling from policy."""
        network = ActorCriticNetwork(state_dim=50, action_dim=3)
        state = torch.randn(1, 50)

        action, log_prob, value = network.get_action(state)

        assert action.shape == torch.Size([1])
        assert log_prob.shape == torch.Size([1])
        assert value.shape == torch.Size([1, 1])
        assert 0 <= action.item() < 3

    def test_evaluate_action(self):
        """Test action evaluation under current policy."""
        network = ActorCriticNetwork(state_dim=50, action_dim=3)
        state = torch.randn(16, 50)
        action = torch.randint(0, 3, (16,))

        log_prob, value, entropy = network.evaluate(state, action)

        assert log_prob.shape == torch.Size([16])
        assert value.shape == torch.Size([16])
        assert entropy.shape == torch.Size([16])
        assert (entropy > 0).all()  # Entropy should be positive

    def test_weight_initialization(self):
        """Test weights are properly initialized."""
        network = ActorCriticNetwork(state_dim=50, action_dim=3)

        for param in network.parameters():
            # Check no NaN or Inf values
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()

            # Check reasonable initialization ranges
            assert param.abs().max() < 10.0

    def test_activation_functions(self):
        """Test different activation functions."""
        for activation in ["relu", "tanh"]:
            network = ActorCriticNetwork(
                state_dim=50,
                action_dim=3,
                activation=activation,
            )
            state = torch.randn(8, 50)
            action_probs, value = network(state)

            assert action_probs.shape == (8, 3)
            assert value.shape == (8, 1)

    def test_layer_normalization_option(self):
        """Test layer normalization can be disabled."""
        network_with_ln = ActorCriticNetwork(
            state_dim=50,
            action_dim=3,
            use_layer_norm=True,
        )

        network_without_ln = ActorCriticNetwork(
            state_dim=50,
            action_dim=3,
            use_layer_norm=False,
        )

        state = torch.randn(8, 50)

        # Both should work but potentially produce different outputs
        probs1, value1 = network_with_ln(state)
        probs2, value2 = network_without_ln(state)

        assert probs1.shape == probs2.shape == (8, 3)
        assert value1.shape == value2.shape == (8, 1)


# ============================================================================
# ExperienceBuffer Tests
# ============================================================================


class TestExperienceBuffer:
    """Test suite for ExperienceBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initializes with correct capacity."""
        max_size = 1000
        buffer = ExperienceBuffer(max_size=max_size)

        assert buffer.max_size == max_size
        assert buffer.size() == 0
        assert not buffer.is_full()

    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = ExperienceBuffer(max_size=100)

        obs = np.random.randn(50).astype(np.float32)
        next_obs = np.random.randn(50).astype(np.float32)

        buffer.add(
            observation=obs,
            action=1,
            reward=1.5,
            next_observation=next_obs,
            done=False,
            log_prob=-0.5,
        )

        assert buffer.size() == 1

    def test_sample_batch(self):
        """Test sampling batch from buffer."""
        buffer = ExperienceBuffer(max_size=1000)

        # Add multiple experiences
        for _ in range(100):
            obs = np.random.randn(50).astype(np.float32)
            next_obs = np.random.randn(50).astype(np.float32)
            buffer.add(
                observation=obs,
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_observation=next_obs,
                done=False,
                log_prob=np.random.randn(),
            )

        batch = buffer.sample(batch_size=32)

        assert "observations" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "next_observations" in batch
        assert "dones" in batch
        assert "log_probs" in batch
        assert batch["observations"].shape == (32, 50)

    def test_get_all_experiences(self):
        """Test retrieving all experiences."""
        buffer = ExperienceBuffer(max_size=1000)

        for i in range(50):
            obs = np.full(50, i, dtype=np.float32)
            next_obs = np.full(50, i + 1, dtype=np.float32)
            buffer.add(
                observation=obs,
                action=i % 3,
                reward=float(i),
                next_observation=next_obs,
                done=(i == 49),
            )

        batch = buffer.get_all()

        assert batch["observations"].shape == (50, 50)
        assert len(batch["actions"]) == 50
        assert np.array_equal(batch["rewards"], np.arange(50))

    def test_buffer_overflow_behavior(self):
        """Test buffer behavior when exceeding max size."""
        max_size = 10
        buffer = ExperienceBuffer(max_size=max_size)

        for i in range(20):  # Add more than max_size
            obs = np.full(50, i, dtype=np.float32)
            next_obs = np.full(50, i + 1, dtype=np.float32)
            buffer.add(
                observation=obs,
                action=i % 3,
                reward=float(i),
                next_observation=next_obs,
                done=False,
            )

        # Buffer should only keep last max_size experiences
        assert buffer.size() == max_size
        assert buffer.is_full()

    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = ExperienceBuffer(max_size=100)

        for _ in range(50):
            obs = np.random.randn(50).astype(np.float32)
            next_obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, next_obs, False)

        assert buffer.size() > 0
        buffer.clear()
        assert buffer.size() == 0

    def test_sample_error_on_insufficient_data(self):
        """Test error when sampling with insufficient data."""
        buffer = ExperienceBuffer(max_size=100)

        # Try to sample without enough data
        with pytest.raises(ValueError):
            buffer.sample(batch_size=32)

    def test_get_all_error_on_empty_buffer(self):
        """Test error when getting all from empty buffer."""
        buffer = ExperienceBuffer(max_size=100)

        with pytest.raises(ValueError):
            buffer.get_all()


# ============================================================================
# RolloutBuffer Tests
# ============================================================================
class TestRolloutBuffer:
    """Test suite for RolloutBuffer (on-policy)."""

    def test_rollout_buffer_initialization(self):
        """Test rollout buffer initialization."""
        size = 2048
        buffer = RolloutBuffer(size=size)

        assert buffer.size == size
        assert buffer.position == 0
        assert not buffer.is_full()

    def test_add_to_rollout_buffer(self):
        """Test adding experiences to rollout buffer."""
        buffer = RolloutBuffer(size=100)

        obs = np.random.randn(50).astype(np.float32)

        buffer.add(
            observation=obs,
            action=1,
            reward=1.5,
            done=False,
            value=0.8,
            log_prob=-0.5,
        )

        assert buffer.position == 1

    def test_compute_advantages_gae(self):
        """Test Generalized Advantage Estimation."""
        buffer = RolloutBuffer(size=100)

        # Add experiences
        for i in range(50):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(
                observation=obs,
                action=i % 3,
                reward=1.0 + np.random.randn() * 0.1,
                done=(i == 49),
                value=0.5,
                log_prob=-0.5,
            )

        buffer.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)

        # Check advantages and returns computed
        assert (buffer.advantages != 0).any()  # Not all zeros
        assert (buffer.returns != 0).any()
        assert buffer.advantages.shape == (100,)
        assert buffer.returns.shape == (100,)

    def test_mini_batch_generation(self):
        """Test mini-batch generation from rollout."""
        buffer = RolloutBuffer(size=100)

        for i in range(50):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, i % 3, 1.0, False, 0.5, -0.5)

        buffer.compute_advantages(last_value=0.0)

        batches = list(buffer.get_mini_batches(batch_size=16))

        assert len(batches) > 0
        for batch in batches:
            assert "observations" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch

    def test_rollout_buffer_overflow(self):
        """Test rollout buffer prevents overflow."""
        buffer = RolloutBuffer(size=10)

        # Fill buffer to capacity
        for i in range(10):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, False, 0.5, -0.5)

        # Try to add one more - should raise error
        with pytest.raises(RuntimeError):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, False, 0.5, -0.5)

    def test_rollout_buffer_clear(self):
        """Test clearing rollout buffer."""
        buffer = RolloutBuffer(size=100)

        for i in range(50):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, i % 3, 1.0, False, 0.5, -0.5)

        assert buffer.position == 50
        buffer.clear()
        assert buffer.position == 0


# ============================================================================
# PPOAgent Tests
# ============================================================================


@pytest.mark.skipif(not HAS_SB3, reason="Stable Baselines3 not installed")
class TestPPOAgent:
    """Test suite for PPO Agent (Stable-Baselines3 wrapper)."""

    def test_agent_initialization_with_env(self, sample_env):
        """Test agent initialization with environment."""
        agent = PPOAgent(
            env=sample_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
        )

        assert agent.env is not None
        assert agent.model is not None
        assert isinstance(agent.model, PPO)

    def test_agent_hyperparameters(self, sample_env):
        """Test hyperparameter storage and retrieval."""
        hyperparams = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        }

        agent = PPOAgent(env=sample_env, **hyperparams)

        retrieved = agent.get_hyperparams()

        for key, value in hyperparams.items():
            assert retrieved[key] == value

    def test_agent_predict(self, sample_env):
        """Test agent prediction."""
        agent = PPOAgent(env=sample_env)

        obs, _ = sample_env.reset()
        action, _ = agent.predict(obs, deterministic=True)

        assert isinstance(action, (int, np.integer))

    def test_agent_learn(self, sample_env):
        """Test agent training."""
        agent = PPOAgent(
            env=sample_env,
            verbose=0,
        )

        initial_timesteps = agent.total_timesteps

        # Train for small number of steps
        agent.learn(total_timesteps=128, log_interval=10)

        # Verify training progressed
        assert agent.total_timesteps > initial_timesteps

    def test_agent_save_load(self, sample_env, tmp_model_dir):
        """Test agent model saving and loading."""
        agent1 = PPOAgent(env=sample_env)

        model_path = str(tmp_model_dir / "test_agent")

        agent1.save(model_path)
        assert Path(model_path + ".zip").exists()

        agent2 = PPOAgent.load(model_path, env=sample_env)
        assert agent2.model is not None

    def test_set_hyperparams_recreates_model(self, sample_env):
        """Test setting hyperparams recreates model."""
        agent = PPOAgent(env=sample_env, learning_rate=1e-4)

        initial_model = agent.model

        agent.set_hyperparams(learning_rate=5e-4)

        # Model should be recreated
        assert agent.model is not initial_model

    def test_agent_deterministic_vs_stochastic(self, sample_env):
        """Test deterministic vs stochastic predictions."""
        agent = PPOAgent(env=sample_env)

        obs, _ = sample_env.reset()

        # Multiple stochastic predictions should vary
        actions_stochastic = [agent.predict(obs, deterministic=False) for _ in range(10)]

        # Deterministic predictions should be same
        actions_deterministic = [agent.predict(obs, deterministic=True) for _ in range(10)]

        assert len(set(actions_deterministic)) == 1  # All same

    def test_agent_model_not_initialized_error(self):
        """Test error handling when model not initialized."""
        agent = PPOAgent.__new__(PPOAgent)
        agent.model = None

        with pytest.raises(RuntimeError):
            obs = np.random.randn(50)
            agent.predict(obs)


# ============================================================================
# Integration Tests
# ============================================================================
class TestAgentIntegration:
    """Integration tests combining multiple components."""

    def test_network_with_buffer(self):
        """Test network forward pass with buffer-provided data."""
        network = ActorCriticNetwork(state_dim=50, action_dim=3)
        buffer = ExperienceBuffer(max_size=1000)

        # Populate buffer
        for _ in range(100):
            obs = np.random.randn(50).astype(np.float32)
            next_obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, next_obs, False)

        batch = buffer.sample(32)
        states = torch.from_numpy(batch["observations"])

        action_probs, values = network(states)

        assert action_probs.shape == (32, 3)
        assert values.shape == (32, 1)

    def test_rollout_collection_workflow(self):
        """Test typical rollout collection workflow."""
        network = ActorCriticNetwork(state_dim=50, action_dim=3)
        rollout_buffer = RolloutBuffer(size=2048)

        # Simulate rollout collection
        for step in range(100):
            obs = np.random.randn(50).astype(np.float32)

            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            action, log_prob, value = network.get_action(state_tensor)

            rollout_buffer.add(
                observation=obs,
                action=action.item(),
                reward=np.random.randn(),
                done=(step == 99),
                value=value.item(),
                log_prob=log_prob.item(),
            )

        rollout_buffer.compute_advantages(last_value=0.0)

        batches = list(rollout_buffer.get_mini_batches(batch_size=32))
        assert len(batches) > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================
class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_network_invalid_dimensions(self):
        """Test network with invalid dimensions and suppress no-op warnings."""
        import warnings

        # Suppress the UserWarning specifically for this test block
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Initializing zero-element tensors is a no-op"
            )

            # Test zero state dimension
            network = ActorCriticNetwork(state_dim=0, action_dim=3)
            assert network.state_dim == 0
            assert network.action_dim == 3

            # Test zero action dimension
            network = ActorCriticNetwork(state_dim=50, action_dim=0)
            assert network.state_dim == 50
            assert network.action_dim == 0

    def test_network_invalid_activation(self):
        """Test network with invalid activation function."""
        with pytest.raises(ValueError):
            ActorCriticNetwork(
                state_dim=50,
                action_dim=3,
                activation="invalid",
            )

    def test_buffer_sample_exceeds_size(self):
        """Test sampling more than buffer contains."""
        buffer = ExperienceBuffer(max_size=100)

        for i in range(10):
            obs = np.random.randn(50).astype(np.float32)
            next_obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, next_obs, False)

        with pytest.raises(ValueError):
            buffer.sample(batch_size=32)  # More than 10

    def test_rollout_buffer_position_bounds(self):
        """Test rollout buffer position stays within bounds."""
        buffer = RolloutBuffer(size=10)

        for i in range(10):
            obs = np.random.randn(50).astype(np.float32)
            buffer.add(obs, 1, 1.0, False, 0.5, -0.5)
            assert buffer.position == i + 1


# ============================================================================
# Utilities and Helpers
# ============================================================================
def test_imports():
    """Test all required imports are available."""
    from src.agent import (ActorCriticNetwork, ExperienceBuffer, PPOAgent,
                           RolloutBuffer)

    assert PPOAgent is not None
    assert ActorCriticNetwork is not None
    assert ExperienceBuffer is not None
    assert RolloutBuffer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
