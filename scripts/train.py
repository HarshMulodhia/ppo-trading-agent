"""
Training Script for PPO Trading Agent

This script trains a PPO agent on historical trading data with:
- Configurable hyperparameters
- TensorBoard monitoring
- Model checkpointing
- Periodic evaluation
- Early stopping support
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Load and manage training configuration"""

    def __init__(self, config_path: str):
        """Initialize config from YAML file"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)


class CustomLoggingCallback(BaseCallback):
    """Custom callback for logging training metrics"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            logger.info(f"Step {self.n_calls}: Exploring...")
        return True


def load_data(config: TrainingConfig) -> tuple:
    """
    Load and preprocess trading data.

    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("Loading market data...")

    symbol = config["environment"]["symbol"]
    source = config["environment"]["data_source"]

    if source == "yfinance":
        import yfinance as yf

        start = config["environment"]["train_start"]
        end = config["environment"]["test_end"]
        df = yf.download(symbol, start=start, end=end)
    else:
        raise ValueError(f"Unknown data source: {source}")

    # Compute technical indicators
    logger.info("Computing technical indicators...")
    df = compute_indicators(df, config)

    # Normalize features
    df = normalize_features(df)

    # Split data
    train_end = config["environment"]["train_end"]
    val_start = config["environment"]["val_start"]
    val_end = config["environment"]["val_end"]
    test_start = config["environment"]["test_start"]

    train_df = df[:train_end]
    val_df = df[val_start:val_end]
    test_df = df[test_start:]

    logger.info(f"Train: {len(train_df)} bars, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def compute_indicators(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Compute technical indicators"""
    try:
        import talib
    except ImportError:
        logger.warning("TA-Lib not installed, using simple indicators")
        return compute_simple_indicators(df, config)

    params = config["environment"]["indicator_params"]

    # RSI
    df["rsi"] = talib.RSI(df["Close"], timeperiod=params["rsi_period"])

    # MACD
    df["macd"], df["macd_signal"], _ = talib.MACD(
        df["Close"], fastperiod=params["macd_short"], slowperiod=params["macd_long"]
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["Close"], timeperiod=params["bb_period"], nbdevup=params["bb_std"]
    )

    return df.dropna()


def compute_simple_indicators(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Compute simple indicators without TA-Lib"""
    params = config["environment"]["indicator_params"]

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params["rsi_period"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params["rsi_period"]).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=params["macd_short"], adjust=False).mean()
    exp2 = df["Close"].ewm(span=params["macd_long"], adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma = df["Close"].rolling(window=params["bb_period"]).mean()
    std = df["Close"].rolling(window=params["bb_period"]).std()
    df["bb_upper"] = sma + (std * params["bb_std"])
    df["bb_middle"] = sma
    df["bb_lower"] = sma - (std * params["bb_std"])

    return df.dropna()


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features to zero mean, unit variance"""
    # Normalize price
    df["close_norm"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()

    # Normalize RSI
    df["rsi_norm"] = (df["rsi"] - 50) / 50

    # Normalize MACD
    df["macd_norm"] = df["macd"] / (df["macd"].std() + 1e-8)

    # Bollinger Band position
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (bb_range + 1e-8)

    # Normalize volume
    df["volume_norm"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-8)

    return df


def create_environment(df: pd.DataFrame, config: TrainingConfig):
    """Create trading environment"""
    from src.environment import TradingEnv

    # Convert to numpy array for efficiency
    data_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_norm",
        "rsi_norm",
        "macd_norm",
        "bb_position",
        "volume_norm",
    ]
    df_data = df[data_cols].values

    env = TradingEnv(
        df=df_data,
        initial_capital=config["environment"]["initial_capital"],
        max_position=config["environment"]["max_position"],
        transaction_cost_pct=config["environment"]["transaction_cost_pct"],
        lookback_window=config["environment"]["lookback_window"],
    )

    # Add time limit
    env = TimeLimit(env, max_episode_steps=len(df) - 100)

    return env


def train_agent(
    config: TrainingConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str = "models"
):
    """Train PPO agent"""

    logger.info("Creating training environment...")
    train_env = create_environment(train_df, config)
    train_env = DummyVecEnv([lambda: train_env])

    logger.info("Creating validation environment...")
    val_env = create_environment(val_df, config)
    val_env = DummyVecEnv([lambda: val_env])

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True)

    # Setup callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(Path(output_dir) / "best_model"),
        log_path=str(log_dir),
        eval_freq=config["training"]["eval_freq"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=str(Path(output_dir) / "checkpoints"),
        name_prefix="ppo_trading",
        save_replay_buffer=False,
    )

    callbacks = [eval_callback, checkpoint_callback]

    # Create PPO agent
    logger.info("Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        verbose=config["training"]["verbose"],
        tensorboard_log=str(log_dir),
        seed=config["debugging"]["seed"],
    )

    # Train agent
    logger.info(f"Training for {config['training']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callbacks,
        log_interval=config["training"]["log_interval"],
    )

    # Save final model
    final_model_path = Path(output_dir) / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(str(final_model_path))
    logger.info(f"Model saved to {final_model_path}")

    return model


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train PPO Trading Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--output", type=str, default="models", help="Output directory for models")

    args = parser.parse_args()

    # Load configuration
    config = TrainingConfig(args.config)

    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output}")

    # Load data
    train_df, val_df, test_df = load_data(config)

    # Train agent
    model = train_agent(config, train_df, val_df, args.output)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
