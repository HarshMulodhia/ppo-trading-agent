"""
PPO Trainer - Training Pipeline and Orchestration

Main training loop for the PPO agent, including:
- Experience collection
- Policy and value function updates
- Learning rate scheduling
- Checkpoint management
- Validation monitoring
- Early stopping
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml

from ..agent import PPOAgent
from ..environment import TradingEnv
from ..reward import RewardShaper
from ..utils import timing

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    Rollout buffer for storing experience trajectories.

    Stores observations, actions, rewards, and other trajectory data
    for computing advantages and policy updates.
    """

    def __init__(self, max_size: int = 2048):
        """
        Initialize rollout buffer.

        Args:
            max_size: Maximum buffer size before reset
        """
        self.max_size = max_size
        self.clear()

    def clear(self) -> None:
        """Clear all stored data."""
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []
        self.size: int = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            observation: State observation
            action: Action taken
            reward: Reward received
            value: Value function estimate
            log_prob: Log probability of action
            done: Episode termination flag
        """
        if self.size >= self.max_size:
            logger.warning("Rollout buffer full, clearing old data")
            self.clear()

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all stored data as numpy arrays.

        Returns:
            Tuple of (observations, actions, rewards, values, log_probs)
        """
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
        )

    def __len__(self) -> int:
        """Get buffer size."""
        return self.size


class PPOTrainer:
    """
    Main trainer for PPO agent.

    Orchestrates the complete training pipeline including:
    - Experience collection through environment interaction
    - Advantage and return computation (GAE)
    - Policy and value function updates
    - Learning rate scheduling
    - Checkpoint management
    - Validation and early stopping
    """

    def __init__(
        self,
        env: TradingEnv,
        agent: PPOAgent,
        config_path: Optional[str] = None,
        model_dir: str = "models/",
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Args:
            env: Trading environment for training
            agent: PPO agent to train
            config_path: Path to training configuration YAML
            model_dir: Directory for saving models
            device: Device for training (cpu or cuda)
        """
        self.env = env
        self.agent = agent
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_path)

        # Training state
        self.global_step = 0
        self.episode = 0
        self.best_validation_reward = -np.inf
        self.consecutive_no_improvement = 0

        # Buffers and storage
        self.rollout_buffer = RolloutBuffer(max_size=self.config.get("rollout_buffer_size", 2048))
        self.reward_shaper = RewardShaper()

        # Training history
        self.training_history: Optional[Mapping[str, List]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "sharpe_ratios": [],
            "validation_rewards": [],
            "learning_rates": [],
        }

        logger.info(f"Trainer initialized. Config: {self.config}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load training configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        default_config = {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "episodes": 1000,
            "steps_per_episode": 500,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "num_epochs": 3,
            "num_minibatches": 4,
            "max_grad_norm": 0.5,
            "rollout_buffer_size": 2048,
            "checkpoint_interval": 100,
            "validation_interval": 50,
            "early_stopping": {"enabled": True, "patience": 50, "min_delta": 0.0001},
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and "training" in config:
                    default_config.update(config["training"])
                    logger.info(f"Loaded config from {config_path}")

        return default_config

    @timing
    def collect_rollout(self, num_steps: Optional[int] = None) -> Tuple[float, int]:
        """
        Collect experience rollout from environment.

        Args:
            num_steps: Number of steps to collect (None = full episode)

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        num_steps = num_steps or self.config["steps_per_episode"]

        observation = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(num_steps):
            # Get action from agent
            action, log_prob = self.agent.get_action(observation)

            # Get value estimate
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                value = self.agent.get_value(obs_tensor).item()

            # Execute action in environment
            next_observation, reward, done, info = self.env.step(action)

            # Shape reward if configured
            shaped_reward = reward  # Reward shaping can be added here

            # Store in buffer
            self.rollout_buffer.add(
                observation=observation,
                action=action,
                reward=shaped_reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            observation = next_observation

            if done:
                break

        return episode_reward, episode_length

    def _compute_advantages(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute generalized advantage estimation (GAE).

        Args:
            rewards: Episode rewards
            values: Value estimates
            dones: Episode termination flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        next_value = 0
        gae = 0

        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]

        # Compute advantages backwards through episode
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    @timing
    def update_policy(self) -> Dict[str, float]:
        """
        Perform policy updates using collected rollout.

        Returns:
            Dictionary of loss metrics
        """
        # Get batch from buffer
        observations, actions, rewards, values, old_log_probs = self.rollout_buffer.get_batch()

        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, values, np.zeros(len(rewards)))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = self.config["batch_size"]
        # num_minibatches = self.config["num_minibatches"]
        num_epochs = self.config["num_epochs"]

        # Convert to tensors
        observations_tensor = torch.FloatTensor(observations).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}

        # Update for multiple epochs
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.random.permutation(len(observations))

            # Minibatch updates
            for i in range(0, len(observations), batch_size):
                batch_indices = indices[i : i + batch_size]

                # Get minibatch
                obs_batch = observations_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                old_log_probs_batch = old_log_probs_tensor[batch_indices]

                # Get new log probs and values
                new_log_probs = self.agent.get_log_probs(obs_batch, actions_batch)
                values = self.agent.get_value(obs_batch)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1 - self.config["clip_ratio"], 1 + self.config["clip_ratio"])
                    * advantages_batch
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (values.squeeze() - returns_batch).pow(2).mean()

                # Entropy bonus
                entropy = self.agent.get_entropy(obs_batch).mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config["value_coeff"] * value_loss
                    - self.config["entropy_coeff"] * entropy
                )

                # Update metrics
                metrics["policy_loss"] = policy_loss.item()
                metrics["value_loss"] = value_loss.item()
                metrics["entropy"] = entropy.item()
                metrics["clip_fraction"] = (
                    (torch.abs(ratio - 1.0) > self.config["clip_ratio"]).float().mean().item()
                )

                # Backward pass
                self.agent.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), self.config["max_grad_norm"]
                )
                self.agent.optimizer.step()

        # Clear buffer
        self.rollout_buffer.clear()

        return metrics

    def _update_learning_rate(self, episode: int) -> None:
        """
        Update learning rate according to schedule.

        Args:
            episode: Current episode number
        """
        schedule_type = self.config.get("learning_rate_schedule", "constant")

        if schedule_type == "constant":
            lr = self.config["learning_rate"]
        elif schedule_type == "linear_decay":
            # Linear decay over episodes
            max_episodes = self.config["episodes"]
            lr = self.config["learning_rate"] * (1 - episode / max_episodes)
        elif schedule_type == "exponential_decay":
            # Exponential decay
            decay_rate = 0.9995
            lr = self.config["learning_rate"] * (decay_rate**episode)
        else:
            lr = self.config["learning_rate"]

        # Ensure minimum learning rate
        lr = max(lr, 1e-6)

        # Update optimizer
        for param_group in self.agent.optimizer.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, episode: int, tag: str = "") -> Path:
        """
        Save training checkpoint.

        Args:
            episode: Episode number
            tag: Optional tag for checkpoint name

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_ep{episode}"
        if tag:
            checkpoint_name += f"_{tag}"
        checkpoint_name += ".pt"

        checkpoint_path = self.model_dir / checkpoint_name

        checkpoint = {
            "episode": episode,
            "global_step": self.global_step,
            "agent_state": self.agent.model.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
            "config": self.config,
            "history": self.training_history,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.agent.model.load_state_dict(checkpoint["agent_state"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.episode = checkpoint["episode"]
        self.global_step = checkpoint["global_step"]
        self.training_history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def validate(self, validation_env: TradingEnv, num_episodes: int = 5) -> float:
        """
        Validate agent performance on validation set.

        Args:
            validation_env: Validation environment
            num_episodes: Number of validation episodes

        Returns:
            Average validation reward
        """
        self.agent.model.eval()
        validation_rewards = []

        with torch.no_grad():
            for _ in range(num_episodes):
                obs = validation_env.reset()
                episode_reward = 0.0
                done = False

                while not done:
                    action = self.agent.select_action(obs, deterministic=True)
                    obs, reward, done, _ = validation_env.step(action)
                    episode_reward += reward

                validation_rewards.append(episode_reward)

        self.agent.model.train()

        avg_validation_reward = np.mean(validation_rewards)
        return avg_validation_reward

    @timing
    def train(
        self,
        num_episodes: Optional[int] = None,
        validation_env: Optional[TradingEnv] = None,
        validation_interval: Optional[int] = None,
        checkpoint_interval: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            num_episodes: Number of episodes to train
            validation_env: Environment for validation
            validation_interval: Validate every N episodes
            checkpoint_interval: Save checkpoint every N episodes
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        num_episodes = num_episodes or self.config["episodes"]
        validation_interval = validation_interval or self.config.get("validation_interval", 50)
        checkpoint_interval = checkpoint_interval or self.config.get("checkpoint_interval", 100)

        logger.info(f"Starting training for {num_episodes} episodes")

        try:
            for episode in range(num_episodes):
                # Update learning rate
                self._update_learning_rate(episode)

                # Collect rollout
                episode_reward, episode_length = self.collect_rollout()

                # Update policy
                metrics = self.update_policy()

                # Store history
                self.training_history["episode_rewards"].append(episode_reward)
                self.training_history["episode_lengths"].append(episode_length)
                self.training_history["policy_losses"].append(metrics["policy_loss"])
                self.training_history["value_losses"].append(metrics["value_loss"])

                # Validation
                if validation_env and (episode + 1) % validation_interval == 0:
                    val_reward = self.validate(validation_env)
                    self.training_history["validation_rewards"].append(val_reward)

                    # Check for improvement
                    if (
                        val_reward
                        > self.best_validation_reward + self.config["early_stopping"]["min_delta"]
                    ):
                        self.best_validation_reward = val_reward
                        self.consecutive_no_improvement = 0

                        # Save best model
                        self.save_checkpoint(episode, tag="best")
                    else:
                        self.consecutive_no_improvement += 1

                        # Early stopping
                        if self.config["early_stopping"]["enabled"]:
                            if (
                                self.consecutive_no_improvement
                                >= self.config["early_stopping"]["patience"]
                            ):
                                logger.info(f"Early stopping at episode {episode}")
                                break

                    if verbose:
                        logger.info(
                            f"Episode {episode + 1}: "
                            f"Reward={episode_reward:.2f}, "
                            f"Val_Reward={val_reward:.2f}, "
                            f"Loss={metrics['policy_loss']:.4f}"
                        )

                # Checkpoint saving
                if (episode + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(episode)

                # Progress logging
                if verbose and (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.training_history["episode_rewards"][-10:])
                    logger.info(
                        f"Episode {episode + 1}: "
                        f"Avg_Reward={avg_reward:.2f}, "
                        f"Policy_Loss={metrics['policy_loss']:.4f}"
                    )

                self.episode = episode + 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(self.episode, tag="interrupted")
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.save_checkpoint(self.episode, tag="error")
            raise

        # Save final model
        self.save_checkpoint(num_episodes - 1, tag="final")
        logger.info(f"Training completed. Total episodes: {self.episode}")

        return self.training_history

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics.

        Returns:
            Dictionary of training metrics
        """
        metrics = {
            "total_episodes": self.episode,
            "global_steps": self.global_step,
            "best_validation_reward": self.best_validation_reward,
            "current_episode_reward": (
                self.training_history["episode_rewards"][-1]
                if self.training_history["episode_rewards"]
                else 0
            ),
        }

        if self.training_history["episode_rewards"]:
            metrics["avg_reward"] = np.mean(self.training_history["episode_rewards"])
            metrics["max_reward"] = np.max(self.training_history["episode_rewards"])
            metrics["min_reward"] = np.min(self.training_history["episode_rewards"])

        return metrics
