"""
Neural Network Architectures for Trading Agent

Provides Actor-Critic network implementations using PyTorch.
"""

from collections.abc import Callable
from typing import List, Tuple

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Neural Network for PPO.

    Architecture:
    - Shared backbone with layer normalization
    - Separate actor head (policy output)
    - Separate critic head (value output)

    Features:
    - Xavier initialization
    - LayerNorm for stability
    - Flexible hidden layer sizes
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        """
        Initialize Actor-Critic Network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh')
            use_layer_norm: Use LayerNorm
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm

        # Select activation function
        self.activation: Callable[[], nn.Module]
        if activation.lower() == "relu":
            self.activation = nn.ReLU
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build shared backbone
        self.backbone = self._build_backbone(state_dim, hidden_dims)

        # Build actor head
        self.actor = self._build_actor_head(hidden_dims[-1], action_dim)

        # Build critic head
        self.critic = self._build_critic_head(hidden_dims[-1])

        # Initialize weights
        self._initialize_weights()

    def _build_backbone(self, input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
        """Build shared backbone network."""
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(self.activation())
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_actor_head(self, input_dim: int, action_dim: int) -> nn.Sequential:
        """Build actor (policy) head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            self.activation(),
            nn.Linear(input_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )

    def _build_critic_head(self, input_dim: int) -> nn.Sequential:
        """Build critic (value) head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            self.activation(),
            nn.Linear(input_dim // 2, 1),
        )

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor

        Returns:
            (action_probs, value)
        """
        # Extract features from backbone
        features = self.backbone(state)

        # Actor head outputs action probabilities
        action_probs = self.actor(features)

        # Critic head outputs value estimate
        value = self.critic(features)

        return action_probs, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and log probability from policy.

        Args:
            state: State tensor

        Returns:
            (action, log_probability, value)
        """
        action_probs, value = self(state)

        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)

        # Sample action
        action = dist.sample()

        # Get log probability
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action under current policy.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            (log_probability, value, entropy)
        """
        action_probs, value = self(state)

        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)

        # Get log probability of specific actions
        log_prob = dist.log_prob(action)

        # Get entropy for exploration
        entropy = dist.entropy()

        return log_prob, value.squeeze(), entropy


def create_network(
    state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256], **kwargs
) -> ActorCriticNetwork:
    """
    Factory function to create Actor-Critic network.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional arguments to ActorCriticNetwork

    Returns:
        ActorCriticNetwork instance
    """
    return ActorCriticNetwork(
        state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims, **kwargs
    )
