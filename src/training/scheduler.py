"""
Learning Rate Scheduler Module

Adaptive learning rate scheduling during training.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LRScheduler:
    """Base learning rate scheduler."""

    def __init__(self, initial_lr: float):
        """
        Initialize scheduler.

        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0

    def step(self) -> float:
        """
        Update learning rate.

        Returns:
            Updated learning rate
        """
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_lr = self.initial_lr
        self.step_count = 0


class ExponentialDecay(LRScheduler):
    """Exponential decay scheduler."""

    def __init__(
        self, initial_lr: float, decay_rate: float = 0.95, steps_per_decay: int = 1000
    ):
        """
        Initialize exponential decay.

        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate per step (< 1)
            steps_per_decay: Steps between decay
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.steps_per_decay = steps_per_decay

    def step(self) -> float:
        """Update learning rate with exponential decay."""
        if self.step_count > 0 and self.step_count % self.steps_per_decay == 0:
            self.current_lr = self.initial_lr * (
                self.decay_rate ** (self.step_count / self.steps_per_decay)
            )

        self.step_count += 1
        return self.current_lr


class StepDecay(LRScheduler):
    """Step-wise decay scheduler."""

    def __init__(self, initial_lr: float, step_size: int = 1000, gamma: float = 0.1):
        """
        Initialize step decay.

        Args:
            initial_lr: Initial learning rate
            step_size: Steps between decay
            gamma: Decay factor
        """
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> float:
        """Update learning rate with step decay."""
        decay_steps = self.step_count // self.step_size
        self.current_lr = self.initial_lr * (self.gamma**decay_steps)
        self.step_count += 1
        return self.current_lr


class CosineAnnealing(LRScheduler):
    """Cosine annealing scheduler."""

    def __init__(self, initial_lr: float, total_steps: int, min_lr: float = 0.0):
        """
        Initialize cosine annealing.

        Args:
            initial_lr: Initial learning rate
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.total_steps = total_steps
        self.min_lr = min_lr

    def step(self) -> float:
        """Update learning rate with cosine annealing."""
        progress = self.step_count / self.total_steps
        self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + np.cos(np.pi * progress)
        )
        self.step_count += 1
        return self.current_lr


class PolynomialDecay(LRScheduler):
    """Polynomial decay scheduler."""

    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
    ):
        """
        Initialize polynomial decay.

        Args:
            initial_lr: Initial learning rate
            total_steps: Total training steps
            power: Polynomial power
            min_lr: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr

    def step(self) -> float:
        """Update learning rate with polynomial decay."""
        if self.step_count >= self.total_steps:
            self.current_lr = self.min_lr
        else:
            progress = self.step_count / self.total_steps
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * (
                (1 - progress) ** self.power
            )

        self.step_count += 1
        return self.current_lr


class WarmupScheduler(LRScheduler):
    """Learning rate warmup followed by decay."""

    def __init__(
        self, initial_lr: float, warmup_steps: int, base_scheduler: LRScheduler
    ):
        """
        Initialize warmup scheduler.

        Args:
            initial_lr: Initial learning rate
            warmup_steps: Number of warmup steps
            base_scheduler: Scheduler to use after warmup
        """
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler

    def step(self) -> float:
        """Update with warmup then base scheduler."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            progress = self.step_count / self.warmup_steps
            self.current_lr = self.initial_lr * progress
        else:
            # Use base scheduler
            self.current_lr = self.base_scheduler.step()

        self.step_count += 1
        return self.current_lr
