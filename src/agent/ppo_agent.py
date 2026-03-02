"""
PPO Agent Wrapper

Wrapper around Stable-Baselines3 PPO for convenient trading agent management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO Agent wrapper for trading.

    Provides convenient interface around Stable-Baselines3 PPO for:
    - Model creation and training
    - Prediction and evaluation
    - Model persistence
    - Hyperparameter management
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        seed: int = 42,
        verbose: int = 1,
    ):
        """
        Initialize PPO Agent.

        Args:
            env: Gymnasium environment
            learning_rate: Adam learning rate
            n_steps: Steps per rollout
            batch_size: Mini-batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE decay
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function loss coefficient
            seed: Random seed
            verbose: Verbosity level
        """
        self.env = env
        self.hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
        }
        self.seed = seed
        self.verbose = verbose

        # Create PPO model
        self.model: Optional[PPO] = None
        self._create_model()

    def _create_model(self) -> None:
        """Create PPO model with current hyperparameters."""
        try:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.hyperparams["learning_rate"],
                n_steps=self.hyperparams["n_steps"],
                batch_size=self.hyperparams["batch_size"],
                n_epochs=self.hyperparams["n_epochs"],
                gamma=self.hyperparams["gamma"],
                gae_lambda=self.hyperparams["gae_lambda"],
                clip_range=self.hyperparams["clip_range"],
                ent_coef=self.hyperparams["ent_coef"],
                vf_coef=self.hyperparams["vf_coef"],
                seed=self.seed,
                verbose=self.verbose,
                device="cpu",
            )
            logger.info("PPO model created successfully")
        except Exception as e:
            logger.error(f"Failed to create PPO model: {e}")
            raise

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 10,
    ) -> None:
        """
        Train agent.

        Args:
            total_timesteps: Total training steps
            callback: Training callback
            log_interval: Logging interval
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        logger.info(f"Training for {total_timesteps} timesteps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
        )
        logger.info("Training completed")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Predict action for observation.

        Args:
            obs: Observation
            deterministic: Use deterministic policy

        Returns:
            (action, additional_info)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action), None

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Save path (without extension)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, env=None) -> "PPOAgent":
        """
        Load model from disk.

        Args:
            path: Model path (without extension)
            env: Environment (optional, from model if not provided)

        Returns:
            PPOAgent instance
        """
        path = Path(path)
        if not path.exists() and not (path.parent / (path.name + ".zip")).exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = PPO.load(str(path), env=env, device="cpu")

        agent = cls.__new__(cls)
        agent.model = model
        agent.env = env or model.get_env()
        agent.hyperparams = {}  # Load from model parameters
        agent.seed = None
        agent.verbose = 0

        logger.info(f"Model loaded from {path}")
        return agent

    def set_hyperparams(self, **kwargs) -> None:
        """
        Update hyperparameters and recreate model.

        Args:
            **kwargs: Hyperparameters to update
        """
        self.hyperparams.update(kwargs)
        self._create_model()
        logger.info("Hyperparameters updated")

    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Get current hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        return self.hyperparams.copy()

    @property
    def total_timesteps(self) -> int:
        """Get total training timesteps."""
        if self.model is None:
            return 0
        return self.model.num_timesteps
