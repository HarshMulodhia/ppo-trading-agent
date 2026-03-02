"""
Experience Replay Buffer Module

Efficient experience storage and sampling for training.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience replay buffer for storing and sampling trajectories.

    Features:
    - Fixed-size circular buffer
    - Efficient memory usage
    - Random batch sampling
    - Experience accumulation
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: Optional[float] = None,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            log_prob: Log probability of action (optional)
        """
        experience = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "log_prob": log_prob,
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> dict:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary with batched experiences
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size ({len(self.buffer)}) < batch size ({batch_size})")

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]

        # Stack experiences
        batch = {
            "observations": np.array([e["observation"] for e in experiences]),
            "actions": np.array([e["action"] for e in experiences]),
            "rewards": np.array([e["reward"] for e in experiences]),
            "next_observations": np.array([e["next_observation"] for e in experiences]),
            "dones": np.array([e["done"] for e in experiences]),
        }

        if experiences[0]["log_prob"] is not None:
            batch["log_probs"] = np.array([e["log_prob"] for e in experiences])

        return batch

    def get_all(self) -> dict:
        """
        Get all experiences in buffer.

        Returns:
            Dictionary with all experiences
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        experiences = list(self.buffer)

        batch = {
            "observations": np.array([e["observation"] for e in experiences]),
            "actions": np.array([e["action"] for e in experiences]),
            "rewards": np.array([e["reward"] for e in experiences]),
            "next_observations": np.array([e["next_observation"] for e in experiences]),
            "dones": np.array([e["done"] for e in experiences]),
        }

        if experiences[0]["log_prob"] is not None:
            batch["log_probs"] = np.array([e["log_prob"] for e in experiences])

        return batch

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        logger.info("Experience buffer cleared")

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.max_size


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO).

    Stores experiences from a rollout and provides
    efficient batch sampling with advantage computation.
    """

    def __init__(self, size: int = 2048):
        """
        Initialize rollout buffer.

        Args:
            size: Buffer capacity
        """
        self.size = size
        self.observations = np.zeros((size,), dtype=object)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Add experience to rollout buffer.

        Args:
            observation: Observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            value: Value estimate
            log_prob: Log probability of action
        """
        if self.position >= self.size:
            raise RuntimeError("Rollout buffer is full")

        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob

        self.position += 1

    def compute_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute advantages using GAE.

        Args:
            last_value: Value estimate for last state
            gamma: Discount factor
            gae_lambda: GAE decay parameter
        """
        advantages = np.zeros(self.size, dtype=np.float32)
        last_advantage = 0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_advantage
            last_advantage = advantages[t]

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_mini_batches(self, batch_size: int):
        """
        Generate mini-batches from rollout buffer.

        Args:
            batch_size: Size of mini-batches

        Yields:
            Mini-batch dictionaries
        """
        indices = np.random.permutation(self.position)

        for start_idx in range(0, self.position, batch_size):
            end_idx = min(start_idx + batch_size, self.position)
            batch_indices = indices[start_idx:end_idx]

            yield {
                "observations": self.observations[batch_indices],
                "actions": self.actions[batch_indices],
                "old_log_probs": self.log_probs[batch_indices],
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
            }

    def clear(self) -> None:
        """Clear buffer."""
        self.position = 0
        logger.info("Rollout buffer cleared")

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.position == self.size
