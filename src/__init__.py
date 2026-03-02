"""
Source Package for PPO Trading Agent

Modules:
- agent: PPO agent and neural network architectures
- data: Data loading and preprocessing
- deployment: Production deployment utilities
- environment: Custom trading environment with Gymnasium integration
- evaluation: Backtesting and performance metrics
- reward: Reward function implementations
- training: Training pipeline and configuration
- utils: General utilities and helpers
"""

__version__ = "1.0.0"
__author__ = "Harsh Mulodhia"

from . import (agent, data, deployment, environment, evaluation, reward,
               training, utils)

__all__ = [
    "agent",
    "data",
    "deployment",
    "environment",
    "evaluation",
    "reward",
    "training",
    "utils",
]
