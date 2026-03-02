"""
Helpers Module

General utility functions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file

    Returns:
        True if successful
    """
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary or None
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving average.

    Args:
        data: Input array
        window: Window size

    Returns:
        Moving average array
    """
    return pd.Series(data).rolling(window=window).mean().values


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """
    Compute exponential moving average.

    Args:
        data: Input array
        span: EMA span

    Returns:
        EMA array
    """
    return pd.Series(data).ewm(span=span).mean().values


def resample_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample time series data.

    Args:
        data: DataFrame with datetime index
        frequency: Resampling frequency (e.g., '1H', '1D')

    Returns:
        Resampled DataFrame
    """
    return data.resample(frequency).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )


def format_metrics(metrics: Dict[str, float], decimals: int = 4) -> str:
    """
    Format metrics for display.

    Args:
        metrics: Metrics dictionary
        decimals: Decimal places

    Returns:
        Formatted string
    """
    output = ""
    for key, value in metrics.items():
        if isinstance(value, float):
            output += f"{key}: {value:.{decimals}f}\n"
        else:
            output += f"{key}: {value}\n"
    return output


def create_directories(*dirs: str) -> None:
    """
    Create directories if they don't exist.

    Args:
        *dirs: Directory paths
    """
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    try:
        return Path(filepath).stat().st_size
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return 0


def ensure_positive(value: float, min_val: float = 1e-8) -> float:
    """Ensure value is positive."""
    return max(float(value), min_val)


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division with zero handling."""
    if abs(denominator) < 1e-8:
        return 0.0
    return float(numerator / denominator)
