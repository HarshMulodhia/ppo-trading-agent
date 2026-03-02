"""
Inference Engine Module

Production inference for trading agents.
"""

import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Production inference engine.

    Features:
    - Single and batch predictions
    - Model warming
    - Confidence scores
    - Latency optimization
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize inference engine.

        Args:
            model: Trained model
            device: Device ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.is_warmed = False

    def warm_start(self, obs_shape: Tuple) -> None:
        """
        Warm start the model.

        Args:
            obs_shape: Observation shape
        """
        with torch.no_grad():
            dummy_obs = torch.randn(1, *obs_shape).to(self.device)
            _ = self.model(dummy_obs)

        self.is_warmed = True
        logger.info("Model warm start complete")

    def predict(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Make single prediction.

        Args:
            observation: Observation array

        Returns:
            Tuple of (action, confidence)
        """
        with torch.no_grad():
            # Convert to tensor
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

            # Forward pass
            output = self.model(obs_tensor)

            # Extract action and confidence
            if isinstance(output, tuple):
                logits, values = output
                action = torch.argmax(logits[0]).item()
                confidence = torch.softmax(logits[0], dim=0).max().item()
            else:
                action = torch.argmax(output[0]).item()
                confidence = torch.softmax(output[0], dim=0).max().item()

        return action, float(confidence)

    def batch_predict(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch predictions.

        Args:
            observations: Batch of observations

        Returns:
            Tuple of (actions, confidences)
        """
        with torch.no_grad():
            # Convert to tensor
            obs_tensor = torch.from_numpy(observations).float().to(self.device)

            # Forward pass
            output = self.model(obs_tensor)

            # Extract actions and confidences
            if isinstance(output, tuple):
                logits, values = output
                actions = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
            else:
                actions = torch.argmax(output, dim=1).cpu().numpy()
                confidences = torch.softmax(output, dim=1).max(dim=1)[0].cpu().numpy()

        return actions, confidences

    def get_confidence(self, observation: np.ndarray) -> float:
        """Get confidence score."""
        _, confidence = self.predict(observation)
        return confidence
