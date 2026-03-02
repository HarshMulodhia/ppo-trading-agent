"""
Data Splitter Module

Time-series aware data splitting.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Time-series aware data splitting.

    Features:
    - Chronological train/val/test splits
    - Rolling window splits
    - Walk-forward validation
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize splitter.

        Args:
            data: Input DataFrame
        """
        self.data = data.copy()
        self.train = None
        self.val = None
        self.test = None

    def time_series_split(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation

        Returns:
            Train, validation, test DataFrames
        """
        n = len(self.data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        self.train = self.data.iloc[:train_size]
        self.val = self.data.iloc[train_size : train_size + val_size]
        self.test = self.data.iloc[train_size + val_size :]

        logger.info(
            f"Split data: train={len(self.train)}, "
            f"val={len(self.val)}, test={len(self.test)}"
        )

        return self.train, self.val, self.test

    def rolling_split(self, window_size: int, step_size: int):
        """
        Generate rolling windows for walk-forward validation.

        Args:
            window_size: Size of training window
            step_size: Step size between windows

        Yields:
            Tuple of (train, test) DataFrames
        """
        n = len(self.data)

        for start in range(0, n - window_size, step_size):
            train_end = start + window_size
            test_end = min(train_end + step_size, n)

            train = self.data.iloc[start:train_end]
            test = self.data.iloc[train_end:test_end]

            if len(test) > 0:
                yield train, test

    def get_train(self) -> pd.DataFrame:
        """Get training data."""
        if self.train is None:
            raise ValueError("Data not split yet. Call time_series_split() first.")
        return self.train

    def get_val(self) -> pd.DataFrame:
        """Get validation data."""
        if self.val is None:
            raise ValueError("Data not split yet. Call time_series_split() first.")
        return self.val

    def get_test(self) -> pd.DataFrame:
        """Get test data."""
        if self.test is None:
            raise ValueError("Data not split yet. Call time_series_split() first.")
        return self.test

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
