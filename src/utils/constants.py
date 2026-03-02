"""
Constants Module

Global constants and configurations.
"""

# Trading Constants
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 6.5
TRADING_MINUTES_PER_HOUR = 60

# Risk Constants
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_MAX_LEVERAGE = 1.0
DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%

# Model Constants
DEFAULT_LEARNING_RATE = 0.0003
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_ROLLOUT_BUFFER_SIZE = 2048

# Data Constants
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
DATA_CACHE_FORMAT = "pickle"  # or 'json'

# Reward Constants
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_SHARPE_LOOKBACK = 252
DEFAULT_MAX_DRAWDOWN_LOOKBACK = 252

# Action Space
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

# Observation Features
NUM_PRICE_FEATURES = 5  # OHLCV
NUM_TECHNICAL_FEATURES = 10  # RSI, MACD, etc
NUM_TOTAL_FEATURES = NUM_PRICE_FEATURES + NUM_TECHNICAL_FEATURES

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Paths
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_RESULTS_DIR = "results"

# Default Configuration Dictionary
DEFAULT_CONFIG = {
    "initial_capital": DEFAULT_INITIAL_CAPITAL,
    "transaction_cost": DEFAULT_TRANSACTION_COST,
    "slippage": DEFAULT_SLIPPAGE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "epochs": DEFAULT_EPOCHS,
    "trading_days_per_year": TRADING_DAYS_PER_YEAR,
}
