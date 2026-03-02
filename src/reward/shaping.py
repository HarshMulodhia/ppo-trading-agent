"""
Reward Shaping Module

Utilities for shaping and transforming rewards.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RewardShaper:
    """
    Utilities for reward shaping.

    Features:
    - Normalization
    - Standardization
    - Clipping
    - Penalty application
    """

    @staticmethod
    def normalize(
        rewards: np.ndarray,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> np.ndarray:
        """
        Normalize rewards to range [min_val, max_val].

        Args:
            rewards: Reward values
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Normalized rewards
        """
        if len(rewards) == 0:
            return rewards

        reward_min = np.min(rewards)
        reward_max = np.max(rewards)

        if reward_max == reward_min:
            return np.full_like(rewards, (min_val + max_val) / 2, dtype=float)

        # Min-max normalization
        normalized = (rewards - reward_min) / (reward_max - reward_min)
        normalized = normalized * (max_val - min_val) + min_val

        return normalized.astype(float)

    @staticmethod
    def standardize(rewards: np.ndarray) -> np.ndarray:
        """
        Standardize rewards (zero mean, unit std).

        Args:
            rewards: Reward values

        Returns:
            Standardized rewards
        """
        if len(rewards) == 0:
            return rewards

        mean = np.mean(rewards)
        std = np.std(rewards)

        if std == 0:
            return np.zeros_like(rewards, dtype=float)

        standardized = (rewards - mean) / std
        return standardized.astype(float)

    @staticmethod
    def clip(
        rewards: np.ndarray,
        min_val: float,
        max_val: float,
    ) -> np.ndarray:
        """
        Clip rewards to range.

        Args:
            rewards: Reward values
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clipped rewards
        """
        return np.clip(rewards, min_val, max_val)

    @staticmethod
    def add_penalty(
        rewards: np.ndarray,
        penalty: float,
    ) -> np.ndarray:
        """
        Add penalty to rewards.

        Args:
            rewards: Reward values
            penalty: Penalty value

        Returns:
            Penalized rewards
        """
        return rewards - penalty

    @staticmethod
    def exponential_shaping(
        rewards: np.ndarray,
        factor: float = 1.0,
    ) -> np.ndarray:
        """
        Apply exponential shaping.

        Args:
            rewards: Reward values
            factor: Shaping factor

        Returns:
            Shaped rewards
        """
        shaped = np.sign(rewards) * (np.exp(factor * np.abs(rewards)) - 1)
        return shaped.astype(float)

    @staticmethod
    def power_shaping(
        rewards: np.ndarray,
        power: float = 0.5,
    ) -> np.ndarray:
        """
        Apply power shaping.

        Args:
            rewards: Reward values
            power: Power exponent

        Returns:
            Shaped rewards
        """
        shaped = np.sign(rewards) * (np.abs(rewards) ** power)
        return shaped.astype(float)

    @staticmethod
    def discount_rewards(
        rewards: np.ndarray,
        gamma: float = 0.99,
    ) -> np.ndarray:
        """
        Compute discounted rewards (returns).

        Args:
            rewards: Immediate rewards
            gamma: Discount factor

        Returns:
            Discounted cumulative rewards
        """
        discounted = np.zeros_like(rewards, dtype=float)
        cumulative = 0.0

        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + gamma * cumulative
            discounted[t] = cumulative

        return discounted
