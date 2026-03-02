"""
Training Configuration Management

Handles loading, validation, and management of training configurations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class TrainingConfig:
    """
    Training configuration manager.

    Handles loading YAML configuration files with validation and
    provides convenient access to hyperparameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML file
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(path, "r") as f:
                self.config = yaml.safe_load(f) or {}

            logger.info(f"Configuration loaded from {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def save(self, config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)

            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots)

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    raise KeyError(f"Configuration key not found: {key}")
            else:
                raise KeyError(f"Cannot access nested key in non-dict: {key}")

        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports nested keys)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_ppo_params(self) -> Dict[str, Any]:
        """Get PPO hyperparameters."""
        return self.get("ppo", {}).copy()

    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.get("training", {}).copy()

    def get_environment_params(self) -> Dict[str, Any]:
        """Get environment parameters."""
        return self.get("environment", {}).copy()

    def get_network_params(self) -> Dict[str, Any]:
        """Get network architecture parameters."""
        return self.get("network", {}).copy()

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"TrainingConfig(path={self.config_path}, keys={list(self.config.keys())})"
